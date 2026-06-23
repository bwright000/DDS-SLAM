#!/usr/bin/env python3
"""GSFlowGate — per-pixel MAPPING depth up-weight for the EndoGSLAM static-map deformation chase (v1).

Flow-AND-depth discriminant: 2D Sampson residual (deform | specular | parallax) AND a 3D surface-change
residual (reproject the ref MoGe depth by the ESTIMATED inter-frame camera motion vs the current depth)
-> keeps ONLY genuine surface motion (deformation moves the surface; specular slides on it -> depth
residual ~0; modelled camera motion cancels). Returns ones (NOP) on warm-up / F-fit / reprojection
failure. RAFT loads ONCE. Lifts flow_track._raft_flow/_sampson byte-for-byte; this class is net-new.
Default-off via cfg['flow_map']['enable'] -> never imported on the base path (parity).

Geometry-path lever (NeRF-agent #1): the returned w in [1,1+lam] multiplies the per-pixel DEPTH residual
in EndoGSLAM's mapping get_loss, so the geometry gradient (means3D) concentrates on the moved surface.
PER-PIXEL, never a region median (NeRF-agent #3). NOT mean-preserving (floor at 1 -> static-pixel gradient
never reduced = the gradient-level global-blur guard).
"""
import numpy as np
from collections import deque

from Addons.motion.flow_track import load_raft, _raft_flow, _sampson  # lifted sensor (byte-for-byte)


class GSFlowGate:
    def __init__(self, cfg, device):
        fm = (cfg or {}).get('flow_map', {})
        self.enable        = bool(fm.get('enable', False))
        self.device        = device
        self.lam           = float(fm.get('lam', 1.0))            # w in [1, 1+lam]
        self.deadband      = float(fm.get('deadband', 3.0))       # flow Sampson deadband (px)
        self.soft_scale    = float(fm.get('soft_scale', 5.0))     # flow ramp width (px)
        self.depth_db      = float(fm.get('depth_deadband', 2.0)) # 3D surface-change deadband (metric)
        self.depth_soft    = float(fm.get('depth_soft', 4.0))     # depth ramp width
        self.require_depth = bool(fm.get('require_depth', True))  # AND the depth gate (False -> Sampson only)
        self.ref_stride    = int(fm.get('ref_stride', 8))         # ref = frame (t - ref_stride)
        self.w_max         = float(fm.get('w_max', 3.0))          # hard clamp guard
        self.ransac_th     = float(fm.get('ransac_thresh', 1.0))
        self.small         = bool(fm.get('raft_small', False))
        self.uniform       = bool(fm.get('uniform_ctrl', False))  # CONTROL arm: constant up-weight (no gating)
        self.K = None
        self._model = self._tf = None
        self._buf = deque(maxlen=max(self.ref_stride, 1))         # causal (color_u8, depth, w2c, t)
        self.last_p99 = 0.0; self.last_frac = 0.0                 # telemetry (eval/runbook read these)
        if self.enable:
            self._model, self._tf = load_raft(device, small=self.small)

    def set_intrinsics(self, K):   # call once before step()
        self.K = np.asarray(K, np.float64).reshape(3, 3)

    def step(self, cur_color_u8, cur_depth, cur_pose_w2c, time_idx):
        """-> [H,W] float32 in [1, 1+lam] (CURRENT grid). ones on warm-up/disabled/any failure."""
        H, W = cur_color_u8.shape[:2]; ones = np.ones((H, W), np.float32)
        if not self.enable:
            return ones
        if len(self._buf) < self.ref_stride:                     # WARM-UP -> NOP, enqueue
            self._push(cur_color_u8, cur_depth, cur_pose_w2c, time_idx); return ones
        ref_c, ref_d, ref_w2c, ref_t = self._buf[0]
        if ref_t >= time_idx:                                    # strict causality
            self._push(cur_color_u8, cur_depth, cur_pose_w2c, time_idx); return ones

        flow_resid, flow = self._flow_sampson(ref_c, cur_color_u8)        # ref grid; zeros on F-fail
        self.last_p99 = float(np.percentile(flow_resid, 99))
        if self.uniform:                                        # CONTROL: constant up-weight, no gating
            self.last_frac = 1.0
            self._push(cur_color_u8, cur_depth, cur_pose_w2c, time_idx)
            return np.full((H, W), 1.0 + self.lam, np.float32)
        w_flow = np.clip((flow_resid - self.deadband) / max(self.soft_scale, 1e-6), 0.0, 1.0)
        if self.require_depth and self.K is not None:
            dres, dvalid = self._depth_reproj_residual(ref_d, cur_depth, ref_w2c, cur_pose_w2c)
            w_depth = np.clip((dres - self.depth_db) / max(self.depth_soft, 1e-6), 0.0, 1.0) * dvalid
        else:
            w_depth = np.ones((H, W), np.float32)                # Sampson-only fallback
        g = w_flow * w_depth                                     # AND (both must fire)
        w_ref = np.clip(1.0 + self.lam * g, 1.0, self.w_max).astype(np.float32)
        w_map = self._forward_warp(w_ref, flow)                  # ref grid -> current grid; holes=1.0
        self.last_frac = float(np.mean(w_map > 1.0 + 1e-3))
        self._push(cur_color_u8, cur_depth, cur_pose_w2c, time_idx)
        return w_map.astype(np.float32)

    # ---- internals ----
    def _push(self, c, d, p, t):
        self._buf.append((c.copy(), np.asarray(d, np.float32).copy(),
                          np.asarray(p, np.float64).reshape(4, 4).copy(), int(t)))

    def _flow_sampson(self, ref_bgr, cur_bgr):
        import cv2
        flow = _raft_flow(self._model, self._tf, ref_bgr, cur_bgr, self.device)
        H, W = flow.shape[:2]
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
        idx = np.linspace(0, len(p1) - 1, min(4000, len(p1))).astype(np.int64)
        F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, self.ransac_th, 0.999)
        if F is None or F.shape != (3, 3):
            return np.zeros((H, W), np.float32), flow
        return _sampson(F.astype(np.float64), p1, p2).reshape(H, W), flow

    def _depth_reproj_residual(self, ref_depth, cur_depth, ref_w2c, cur_w2c):
        """3D surface-change residual on the REF grid: reproject ref depth by the est camera motion,
        compare predicted vs observed current depth. invalid -> resid 0, valid 0 (NOP, no false fire)."""
        H, W = ref_depth.shape
        FX, FY, CX, CY = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        z = ref_depth.astype(np.float64)
        x = (uu - CX) / FX * z; y = (vv - CY) / FY * z
        P = np.stack([x, y, z, np.ones_like(z)], -1).reshape(-1, 4)
        T_rel = cur_w2c @ np.linalg.inv(ref_w2c)                  # inter-frame camera motion (w2c_cur @ inv(w2c_ref))
        Pc = (T_rel @ P.T).T[:, :3]
        Xc, Yc, Zc = Pc[:, 0], Pc[:, 1], Pc[:, 2]; eps = 1e-6
        front = Zc > eps
        up = FX * Xc / np.where(front, Zc, 1.0) + CX
        vp = FY * Yc / np.where(front, Zc, 1.0) + CY
        ui = np.round(up).astype(np.int64); vi = np.round(vp).astype(np.int64)
        inb = front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z.reshape(-1) > 0)
        ui = np.clip(ui, 0, W - 1); vi = np.clip(vi, 0, H - 1)
        Dcur = cur_depth.astype(np.float64)[vi, ui]; inb &= (Dcur > 0)
        resid = np.abs(Zc - Dcur); resid[~inb] = 0.0
        return resid.reshape(H, W).astype(np.float32), inb.reshape(H, W).astype(np.float32)

    def _forward_warp(self, w_ref, flow):
        """ref-grid weight -> current grid via integer-rounded flow (last-writer-wins; holes=1.0=NOP)."""
        H, W = w_ref.shape; out = np.ones((H, W), np.float32)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        tu = np.round(uu + flow[..., 0]).astype(np.int64); tv = np.round(vv + flow[..., 1]).astype(np.int64)
        m = (tu >= 0) & (tu < W) & (tv >= 0) & (tv < H)
        out[tv[m], tu[m]] = w_ref[vv[m], uu[m]]
        return out
