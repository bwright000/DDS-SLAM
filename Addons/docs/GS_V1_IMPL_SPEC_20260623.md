# GS v1 — MAPPING CATCH-UP: build-ready implementation spec (2026-06-23)

Per-pixel DEPTH up-weight in the EndoGSLAM mapping loss, gated by a flow-AND-depth motion sensor, so
the static SH-0 Gaussian map re-fits the moved geometry where the scene deformed. Default-off ⇒
byte-identical to base. All adversarial fixes folded in (cited inline as `[FIX-n]`).

Verified against live code:
- `EndoGSLAM/scripts/main.py` — `get_loss` def L198-200; mapping depth L264-270; mapping RGB L271-279;
  weighting+sum L281-282; frame loop L609; tracking L650-720; mapping driver L724-856; keyframe vs
  current-frame branch L789-798 (mode A, `distance_keyframe_selection=False`) and L808-819 (mode B,
  `distance_keyframe_selection=True` — the CRCD default); call site L822-828; per-frame `device` L412.
- `EndoGSLAM/configs/crcd/crcd_base.py` — mapping `loss_weights=dict(im=1.0, depth=1.0)` L89;
  `distance_keyframe_selection=True` L44.
- `DDS-SLAM/Addons/motion/flow_track.py` — `load_raft`, `_raft_flow`, `_sampson`, `flow_residual`.
- `DDS-SLAM/Addons/gs/attribution_panel.py` — P99/`DEADBAND=3.0` convention L119/L142/L177.
- `DDS-SLAM/Addons/gs/overlay/scripts/gs_eval.py` — `render_metrics` L81-113, `main` L116-185.
- `DDS-SLAM/Addons/gs/overnight_crcd.sh` — idempotent env-knob injector L17-37, `run_arm` L39-55.

---

## (0) Scope

**What v1 is.** EndoGSLAM is a *static* Gaussian map (no deformation field). Tissue deforms frame to
frame. In a static representation the ONLY way deformation renders is the per-frame mapping step
*re-fitting the current deformed depth*. v1 measures, per mapping frame, the scene motion the camera
cannot explain (flow-AND-depth discriminant) and **multiplies up the per-pixel DEPTH (geometry)
residual** there, so the geometry gradient concentrates on the moved surface and the map "catches up".

**Why the DEPTH term, not RGB `[FIX: NeRF-agent #1]`.** Mapping `loss_weights = {im:1.0, depth:1.0}`
(crcd_base L89) — depth is FIRST-CLASS (no NeRF 0.1-vs-1000 SDF imbalance). In `get_loss`(mapping) the
depth render flows gradient to `means3D/scales/opacities` via `transform_to_frame(gaussians_grad=True,
camera_grad=False)` (main.py L216-220) → `transformed_params2depthplussilhouette` → the geometry IS the
thing the depth term moves. The RGB term is `0.8*l1 + 0.2*(1-ssim)` (L279), a small appearance budget;
an RGB up-weight was empirically a NO-OP for deformation. **Depth-only is primary; SSIM stays global
(not per-pixel weightable); colour up-weight is NOT in v1.**

**Why per-pixel, not region-median `[FIX: NeRF-agent #3]`.** A region median logged moving=0 on a frame
where 7.9% of pixels moved >3px. `w_map` stays a full `[1,H,W]` map.

**The static-chase CEILING → v2.** Catching the current frame's depth re-fits *that frame*; it cannot
interpolate deformation between frames or carry a temporally-consistent deforming surface. When v1's
held-out-dynamic gain saturates (or the guard fails), that is the ceiling → escalate to **v2 (a deform
field)**. Both outcomes are clean metric-first results.

---

## (1) Verified hooks (exact file:line)

| Hook | Location | Note |
|---|---|---|
| `get_loss` signature | main.py **L198-200** | add ONE default-`None` kwarg (Edit A) |
| Mapping depth residual | main.py **L269-270** | `torch.abs(curr_data['depth']-depth)[mask].mean()` — the up-weight site (Edit B) |
| `weighted_losses`×`loss_weights` | main.py **L281** | depth weight 1.0 — first-class |
| Per-frame loop top | main.py **L609** | `for time_idx in tqdm(...)` — heartbeat + sensor call site |
| `device` definition | main.py **L412** | local of `rgbd_slam` ONLY — NOT in `get_loss` scope `[FIX-C/device]` |
| Mapping call site | main.py **L822-828** | builds `iter_data`, calls `get_loss(...,mapping=True)` — attach weight here |
| Current-frame branch (mode B, CRCD default) | main.py **L810-814** | `selected_keyframe_ids == len(keyframe_list)` ⇒ current frame |
| Current-frame branch (mode A) | main.py **L789-793** | `selected_rand_keyframe_idx == -1` ⇒ current frame |
| `intrinsics`, `cam`, render res | main.py L515-526 | `intrinsics` is a live 3×3 at **desired** res = render res `[FIX: no native split]` |

---

## (2) The depth-term up-weight diff (literal; NOT mean-preserving + why)

### Edit A — `get_loss` signature (main.py L198-200), append ONE kwarg

```python
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None,
             map_weight=None):
```
`map_weight`: torch tensor `[1,H,W]` in `[1,1+λ]`, on `depth.device`, `depth.dtype`, **already
forward-warped to the CURRENT grid and already at render res** (built at the call site). `None` ⇒ off.
Appended last ⇒ all existing positional calls (tracking L665, mapping L826) are unaffected `[FIX-D parity]`.

### Edit B — mapping depth residual (main.py L264-270)

Replace:
```python
    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
```
with:
```python
    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            depth_resid = torch.abs(curr_data['depth'] - depth)
            if mapping and map_weight is not None:
                # v1 MAPPING CATCH-UP: per-pixel up-weight of the DEPTH (geometry) residual on the
                # pixels the scene moved -> the static map re-fits the moved surface harder.
                # NOT mean-preserving: plain .mean() over N=mask.sum(), w floored at 1 (see below).
                assert map_weight.shape == depth.shape, (map_weight.shape, depth.shape)  # [1,H,W]==[1,H,W]
                depth_resid = depth_resid * map_weight
            losses['depth'] = depth_resid[mask].mean()
```

- Multiply is inside the `else` (mapping) branch ONLY — tracking path byte-untouched.
- `map_weight` is detached (loss *scaling*, no grad); gradient to geometry flows through `depth`. So
  `∂loss/∂(geom)` is scaled per-pixel by `w` ⇒ moved pixels exert larger geometry gradients.

### Why NON-mean-preserving — and why Draft D's normalized form is REJECTED `[FIX: normalization]`

Adversarial finding (confirmed): the normalized form `(depth_resid*w)[mask].sum() / w[mask].sum()`
merely **redistributes a fixed loss budget**. It (a) pins the global magnitude so only relative
re-weighting survives, AND (b) actively *reduces* the static-pixel gradient (their weight 1 ÷ mean(w)>1)
— inducing exactly the global-blur it claimed to prevent. **Use the plain `.mean()` over `N=mask.sum()`
with `w` floored at 1** (`w∈[1,1+λ]`):
- moved-pixel residuals enter at `w×` natural magnitude ⇒ strictly larger total geometry gradient
  concentrated there (the lever bites);
- the floor of 1 guarantees the unmoved-region gradient is NEVER reduced below baseline (the real
  global-blur guard, enforced at the gradient level);
- `w≡1` everywhere ⇒ exactly the base `.mean()` ⇒ byte-identical default-off.

### Adam global-scale caveat (state in the eval; do not mis-attribute) `[FIX: Adam]`

The mapping optimizer is Adam, re-initialized every frame (main.py L772). Adam normalizes per-coordinate
⇒ a weight map that fires **near-uniformly** over the whole valid mask is ~cancelled (a global LR boost,
not a localized re-fit). The lever only bites via the **relative per-pixel/per-Gaussian variation** of
`w`. The `[1,1+λ]` floor-at-1 design already yields non-uniform `w` when only a sub-region moves (the
intended case) ⇒ contrast survives Adam. **Telemetry guard:** the sensor logs the up-weighted fraction
`frac = mean(w>1.0+1e-3)`; if `frac > 0.5` on a frame, log `FM_WARN frac=… (near-global; Adam-cancelled,
gain suspect)`. The uniform-control arm (§8) is the empirical check that a real win is localized, not a
global LR change.

---

## (3) `gs_flow_gate.py` — the MAPPING adapter (skeleton + reprojection math + dtype contract)

**File:** `DDS-SLAM/Addons/motion/gs_flow_gate.py` (net-new; lifts the `flow_track` sensor).

### The flow-AND-depth discriminant (load-bearing math)

A 2D flow residual conflates deformation, specular slide, and camera parallax. With the estimated pose +
MoGe depth we add a **3D surface-change residual**: reproject the reference depth by the inter-frame
camera motion and compare predicted vs observed current depth. Deformation moves the surface
(depth-residual high); specular slides on it (depth-residual ~0); modeled camera motion cancels.

Conventions (verified): `K` 3×3, `FX=K[0,0] FY=K[1,1] CX=K[0,2] CY=K[1,2]`. Poses **w2c** 4×4,
column-vector (`p_cam = w2c @ p_world`). Depth camera-space `+z` (metric MoGe, `sc_factor=1`).

1. Back-project every ref pixel (ref-cam frame): `x=(u-CX)/FX·D_ref`, `y=(v-CY)/FY·D_ref`, `z=D_ref`.
2. Inter-frame camera motion: `T_rel = w2c_cur @ inv(w2c_ref)`; `P_cur = (T_rel @ [P_ref;1])[:3] = [Xc,Yc,Zc]`.
   `Zc` = predicted current depth if the surface is static.
3. Project: `u_p = FX·Xc/Zc + CX`, `v_p = FY·Yc/Zc + CY` (only where `Zc>eps`, in-bounds).
4. `depth_resid[v,u] = |Zc − D_cur(round(v_p),round(u_p))|`; invalid (out-of-bounds, `Zc≤eps`,
   `D_ref≤0`, `D_cur≤0`) ⇒ `resid=0, valid=False` (no evidence of motion ⇒ NOP, never a false up-weight).

Both residuals are computed on the **ref grid** (well-defined AND); the final `w_ref` is **forward-warped
to the current grid** by integer-rounded flow so it aligns with the current-frame mapping residual
`[FIX: ref-vs-cur grid alignment]`. **Pose source = the ESTIMATED w2c** (`cam_unnorm_rots/cam_trans` at
`t`/`ref`), not GT — must match the camera the map renders from, or `T_rel` mismodels parallax.

### Resolution / dtype contract `[FIX: native-split, arange-dtype, BGR, device]`

- **NO native-vs-render split.** EndoGSLAM `intrinsics`/color/depth are all at desired = render res. RAFT
  + reproject run at that single grid; `w_map` returned at render res `[1,H,W]`. **No `cv2.resize`,
  no K-rescale, no `cv2` import in main.py** — the adapter does the whole job and returns a GPU tensor.
- **Color → sensor boundary** (single contract): SLAM color is torch `[3,H,W]` float RGB in `[0,1]`.
  The sensor wants `[H,W,3] uint8 BGR`:
  `img_u8 = (color.permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()[:, :, ::-1]` (RGB→BGR; the sensor
  does BGR2RGB internally → restores true RGB).
- `np.arange(W, dtype=np.float64)` / `np.arange(H, dtype=np.float64)` — **keyword `dtype=`** (positional
  `np.float32` is read as `stop` ⇒ TypeError).
- Returned tensor: `torch.from_numpy(w_map).to(device=depth.device, dtype=depth.dtype).unsqueeze(0)`
  ⇒ `[1,H,W]`, asserted `== depth.shape` at the multiply (Edit B).

### Skeleton

```python
#!/usr/bin/env python3
"""GSFlowGate — per-pixel MAPPING depth up-weight for the EndoGSLAM static-map deformation chase (v1).
Flow-AND-depth discriminant: 2D Sampson (deform|specular|parallax) AND 3D surface-change residual
(reproject ref MoGe depth by the est inter-frame camera motion vs current depth) -> keeps ONLY genuine
surface motion. Returns ones (NOP) on warm-up / F-fit / reprojection failure. RAFT loads ONCE.
Lifts flow_track._raft_flow/_sampson byte-for-byte; this class is net-new. Default-off via cfg."""
import numpy as np
from collections import deque
from Addons.motion.flow_track import load_raft, _raft_flow, _sampson  # lifted sensor


class GSFlowGate:
    def __init__(self, cfg, device):
        fm = (cfg or {}).get('flow_map', {})
        self.enable        = bool(fm.get('enable', False))
        self.device        = device
        self.lam           = float(fm.get('lam', 1.0))            # w in [1, 1+lam]
        self.deadband      = float(fm.get('deadband', 3.0))       # flow Sampson deadband (px)
        self.soft_scale    = float(fm.get('soft_scale', 5.0))     # flow ramp width (px)
        self.depth_db      = float(fm.get('depth_deadband', 2.0)) # 3D surface-change deadband (metric, mm-scale)
        self.depth_soft    = float(fm.get('depth_soft', 4.0))     # depth ramp width
        self.require_depth = bool(fm.get('require_depth', True))  # AND the depth gate (False -> Sampson only)
        self.ref_stride    = int(fm.get('ref_stride', 8))         # ref = frame (t - ref_stride)
        self.w_max         = float(fm.get('w_max', 3.0))          # hard clamp guard
        self.ransac_th     = float(fm.get('ransac_thresh', 1.0))
        self.small         = bool(fm.get('raft_small', False))
        self.uniform       = bool(fm.get('uniform_ctrl', False))  # CONTROL arm: constant up-weight (§8)
        self.K = None
        self._model = self._tf = None
        self._buf = deque(maxlen=max(self.ref_stride, 1))         # causal (color_u8, depth, w2c, t)
        self.last_p99 = 0.0; self.last_frac = 0.0                 # telemetry (eval reads these)
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
        H, W = ref_depth.shape
        FX, FY, CX, CY = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        uu, vv = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        z = ref_depth.astype(np.float64)
        x = (uu - CX) / FX * z; y = (vv - CY) / FY * z
        P = np.stack([x, y, z, np.ones_like(z)], -1).reshape(-1, 4)
        T_rel = cur_w2c @ np.linalg.inv(ref_w2c)
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
        H, W = w_ref.shape; out = np.ones((H, W), np.float32)
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        tu = np.round(uu + flow[..., 0]).astype(np.int64); tv = np.round(vv + flow[..., 1]).astype(np.int64)
        m = (tu >= 0) & (tu < W) & (tv >= 0) & (tv < H)
        out[tv[m], tu[m]] = w_ref[vv[m], uu[m]]
        return out
```

`depth_deadband` is the one knob worth a sweep — set it to the per-snippet MoGe depth-noise floor once
measured, else genuine-but-small deformation gets a noisy gate (and a sub-noise deadband saturates the
gate everywhere — the uniform-control degenerate §8 catches that).

---

## (4) Held-out-dynamic render metric in `gs_eval.py` (+ global-blur guard)

The failure mode of any "fit harder where it moved" lever is **memorization**: in online SLAM every
frame is mapped, so naive `gs_eval` (renders each frame at its own est pose, L128-141) cannot tell a real
win from overfitting. Extend `gs_eval.py` additively (legacy `metrics.txt` block byte-unchanged).

### 4a. Held-out split (the clean definition) `[FIX: holdout path absent → implement it]`

The temporal hold-out MUST exist or the headline is only a weak view-gap proxy. **Implement it**, flag-
gated default-off:
- Config key `eval_holdout_every: 0` (default ⇒ base bit-identical). When `k>0`, in the mapping driver,
  a frame with `time_idx % k == k-1` is **tracked** (gets an est pose) but **excluded from the mapping
  loss + `add_new_gaussians`** — skip the `if time_idx==0 or (time_idx+1)%map_every==0` mapping block for
  it. Write the deterministic indices to `run/holdout_idx.npy`.
- `gs_eval` renders all frames at est pose (existing path), partitions metrics by membership in
  `holdout_idx`. **Headline = HELD-OUT ∩ DYNAMIC.** If `holdout_idx.npy` is absent, `gs_eval` prints a
  loud `WARNING: no mapping-holdout -> view-gap proxy only (weaker)` and refuses to headline a "win".

### 4b. DYNAMIC label from an INDEPENDENT signal `[FIX: shared-sensor circularity]`

v1's weight and the dynamic label must NOT be the same Sampson residual on the same arm, or the metric
scores frames v1 fit hardest. Two decouplings, both required:
- **Freeze the dynamic set from the BASE (OFF) arm** and apply that SAME frozen set to both arms.
- **Label depth-residual-based, not flow-Sampson-based:** per frame, motion score = P99 of the **3D
  surface-change residual** (`_depth_reproj_residual` between consecutive frames, base-arm est poses) —
  the signal v1's *flow* head never directly consumes. `DYNAMIC iff score > depth_deadband`. (Optionally
  also report the flow-P99 label for cross-check, but the frozen depth-residual label is the arbiter.)
- Persist `run/p99_per_frame.npy` + `run/dynamic_idx.npy` (from the base arm) so the on arm reads them.

### 4c. Reporting matrix (per run → `metrics_split.json`)

Refactor `render_metrics` (L94-113) to also **return the per-frame arrays** `(P[],S[],L[],Dl[])` (it
already builds the lists — return them, not only `nanmean`). Then aggregate by boolean masks
(`dynamic_mask`, `holdout_mask`) with `np.nanmean`; no second render pass.

| subset | PSNR | SSIM | LPIPS | L1-Depth(mm) | n |
|---|---|---|---|---|---|
| HELD-OUT ∩ DYNAMIC **(HEADLINE)** | | | | | |
| HELD-OUT ∩ STATIC *(GUARD §4d)* | | | | | |
| HELD-OUT ∩ ALL | | | | | |
| FITTED ∩ DYNAMIC *(memorization-prone)* | | | | | |
| ALL frames *(legacy single number, back-compat)* | | | | | |

Sim3-ATE stays per-run whole-trajectory (pose is global; keep the single `ate` from L143-144).

### 4d. GUARD — global-blur detector `[FIX: sensor-coupled guard]`

The co-adapt failure: up-weighting moved pixels blurs the whole map.
- **Frame-level guard:** `HELD-OUT ∩ STATIC` PSNR/SSIM/LPIPS, ON vs OFF.
- **Region-level guard, on a SENSOR-INDEPENDENT mask:** within dynamic held-out frames, the static region
  is the complement of the **depth-residual** dynamic pixels (NOT the flow mask v1 keys on) — so smear
  near v1's own deadband is not excused. PSNR on that region, ON vs OFF.
- **Pass rule (printed):** PASS iff `STATIC PSNR_on ≥ PSNR_off − τ` AND `region PSNR_on ≥ PSNR_off − τ`,
  `τ = 1× base seed-std` (default `0.20 dB`); SSIM guard `≥ off − 0.005`. FAIL ⇒
  `GUARD FAIL: static render dropped … > τ — global-blur, v1 win NOT free`. A dynamic win with a guard
  FAIL is NOT a win. Emit `guard_pass`, `guard_static_delta_db`, `guard_region_delta_db` to JSON.

### 4e. Keep the 5 metrics + 6-panel video

Legacy `ALL frames` row reproduces today's printed block byte-for-byte. Add to the 6-panel video a
**w_map weight-map panel** + **reproj-depth-residual panel** (writes `run/frame_tags.txt`
`idx score DYN|STA HELD|FIT` if `generate_video.py` can't take a per-frame list — additive, degrade
gracefully). Auto-ship `metrics_split.json`, `frame_tags.txt`, `p99_per_frame.npy`, mp4 to `--drive_root`.

### 4f. attribution_panel cache `[FIX: cache never written]`

After the scan loop (attribution_panel.py ~L223) add:
`np.save(os.path.join(a.out,'p99_per_frame.npy'), np.array(p99_all))` and a parallel
`ref_index.npy` (the `ref` each P99 belongs to). `gs_eval` that reads a cache MUST assert
`stride` and ref-convention equality (or fail loud) — do not silently mix strides.

---

## (5) flow_track lift map

| `flow_track.py` symbol | Use in v1 | Touch? |
|---|---|---|
| `load_raft(device, small)` | RAFT loaded ONCE in `GSFlowGate.__init__` | reuse |
| `_raft_flow(model,tf,a_bgr,b_bgr,device)` | dense flow, ref→cur, in `_flow_sampson` | reuse |
| `_sampson(F,p1,p2)` | per-pixel Sampson, ref grid | reuse |
| `flow_residual(...)` | reference impl (the AND-gate inlines `_raft_flow`+`_sampson` to also keep `flow` for the forward-warp) | pattern |
| `region_route` / `residual_to_weight` / `agreement_gate` | NOT used (region-median = NeRF-agent #3 failure) | ignore |

`flow_track.py` is **reused byte-for-byte** (no fork). `gs_flow_gate.py` is the only net-new module.

---

## (6) Config block (env-driven; injected idempotently)

Injected by the runbook's `inject_flowmap_knobs.py` (overnight_crcd.sh L17-37 pattern). **Every knob
defaults to base ⇒ off == base.** The main.py hook reads via `.get` (next section) so a config WITHOUT
the block is still base `[FIX-D]`.

```python
# ---- v1 flow_map: MAPPING CATCH-UP (per-pixel depth up-weight where the scene moved) ----
_FLOWMAP = bool(int(os.environ.get("FLOW_MAP", 0)))          # MASTER ENABLE (0 = base; no import, no RAFT)
config["flow_map"] = dict(
    enable        = _FLOWMAP,
    lam           = float(os.environ.get("FM_LAMBDA", 1.0)),     # peak extra depth weight (w in [1,1+lam])
    deadband      = float(os.environ.get("FM_DEADBAND", 3.0)),   # Sampson px below = camera/noise
    soft_scale    = float(os.environ.get("FM_SOFT", 5.0)),       # flow ramp width (px)
    depth_deadband= float(os.environ.get("FM_DEPTH_DB", 2.0)),   # 3D surface-change deadband (metric)
    depth_soft    = float(os.environ.get("FM_DEPTH_SOFT", 4.0)), # depth ramp width
    require_depth = bool(int(os.environ.get("FM_REQ_DEPTH", 1))),# AND the depth gate (0 -> Sampson only)
    ref_stride    = int(os.environ.get("FM_REF_STRIDE", 8)),     # ref = max(0, t - ref_stride)
    raft_small    = bool(int(os.environ.get("FM_RAFT_SMALL", 0))),
    w_max         = float(os.environ.get("FM_WMAX", 3.0)),       # hard per-pixel clamp
    uniform_ctrl  = bool(int(os.environ.get("FM_UNIFORM", 0))),  # CONTROL arm: constant up-weight (§8)
    eval_holdout_every = int(os.environ.get("FM_HOLDOUT_EVERY", 0)),  # 0 = no holdout (base); 5 = clean metric
)
```

Per-pixel gate: `w_flow = clip((sampson-deadband)/soft_scale,0,1)`,
`w_depth = clip((surf_change-depth_deadband)/depth_soft,0,1)·valid`, `g = w_flow*w_depth`,
`w = clip(1+lam·g, 1, w_max)`. `g≡0 ⇒ w≡1 ⇒ base`.

---

## (7) Default-off PARITY proof + Inc-0 test

### Why OFF is provably bit-identical (4 independent reasons)
1. **Config:** main.py hook reads `config.get('flow_map', {}).get('enable', False)` — a config without
   the block (pristine `crcd_base.py`, half-injected clone, or the Tier-A test) ⇒ `False` ⇒ base. The OFF
   path does NOT depend on the injector having run `[FIX-D]`.
2. **Import isolation:** `from Addons.motion.gs_flow_gate import GSFlowGate` is inside `if enable:` — never
   imported on the base path. No RAFT load, no torchvision import, no RNG draw.
3. **No undefined names:** the current-frame guard uses values already in scope — there is **no
   `_current_time_idx`** anywhere `[FIX: NameError]`. The weight is attached at the call site ONLY for the
   current-frame branch (below); inside `get_loss` the test is `map_weight is not None`.
4. **No `cv2` in main.py:** the adapter does its own resize/reproject and returns a GPU tensor; main.py's
   import set is byte-unchanged `[FIX: cv2]`. `device` is used only at the call site (L412 scope), never
   inside `get_loss` `[FIX: device]`.

### The call-site hook (current-frame ONLY) `[FIX: keyframe-iter misapplication]`

The weight is computed ONCE per frame (after tracking, before the mapping iter loop), and attached to
`iter_data` ONLY when the selected mapping iter is the **current frame** — never a keyframe-replay iter
(stale/mismatched weight) and never tracking. CRCD default is mode B (`distance_keyframe_selection=True`,
L808-819): current frame ⇔ `selected_keyframe_ids == len(keyframe_list)`. Sketch:

```python
# once per frame, after tracking converges, before the mapping iter loop (main.py ~L776):
fm_w = None
if config.get('flow_map', {}).get('enable', False) and time_idx >= _gate.ref_stride:
    img_u8 = (color.permute(1,2,0).clamp(0,1)*255).byte().cpu().numpy()[:, :, ::-1]   # RGB->BGR
    cur_w2c = _est_w2c(params, time_idx)            # build from cam_unnorm_rots/cam_trans (est pose)
    wm = _gate.step(img_u8, depth[0].cpu().numpy(), cur_w2c.cpu().numpy(), time_idx)   # [H,W]
    fm_w = torch.from_numpy(wm).to(device=depth.device, dtype=depth.dtype).unsqueeze(0)  # [1,H,W]

# inside the mapping iter loop, the CURRENT-frame branch ONLY (mode B L810-814 / mode A L789-793):
#   map_weight_this_iter = fm_w if (current_frame_branch) else None
# pass it through:
loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx,
        config['mapping']['loss_weights'], config['mapping']['use_sil_for_loss'],
        config['mapping']['sil_thres'], config['mapping']['use_l1'],
        config['mapping']['ignore_outlier_depth_loss'], mapping=True,
        map_weight=map_weight_this_iter)
```
`_gate = GSFlowGate(config, device)` built once before the frame loop (only if enabled);
`_gate.set_intrinsics(intrinsics[:3,:3].cpu().numpy())` once after `intrinsics` exists (L515-526).

### Inc-0 test — `Addons/gs/regression/test_flowmap_inc0.py`

GPU GS is NOT byte-reproducible (CUDA rasteriser). Two tiers:

**Tier A — STATIC import-isolation (HARD gate, must pass):**
1. Load `crcd_base.py` with `FLOW_MAP` unset and `FLOW_MAP=0`; assert config deep-equal and
   `flow_map['enable'] is False`.
2. `builtins.__import__` tripwire that raises if `Addons.motion.gs_flow_gate` is imported; run ~3 frames
   of `rgbd_slam` with `FLOW_MAP=0` ⇒ no import ⇒ pass (the sensor is truly dead on base).
3. AST/diff check: every `main.py` edit vs base sits inside `if config.get('flow_map',{}).get('enable')…`
   or the appended-kwarg default — never on an unconditional line.

**Tier B — EXECUTION-TRACE (`FM_LAMBDA=0`, advisory):** run `FLOW_MAP=1 FM_LAMBDA=0` (sensor runs, `w≡1`
since `lam·g=0` and floor=1) vs base for 6 frames, fixed `SEED=0`; capture the per-iter mapping
`losses['depth']` trace; assert equal within `rtol=1e-4`. This RUNS now (no `_current_time_idx` to crash
on `[FIX]`) and proves `lam=0` reduces to base. Rasteriser jitter can exceed 1e-4 on some GPUs ⇒ Tier B
is advisory; **Tier A is the gate.** Print one grep line `>>> INC0 PARITY PASS|FAIL`.

---

## (8) Runbook (readable logging) + A/B plan

**File:** `Addons/gs/flow_map_ab_20260623.sh`. Conventions from overnight_crcd.sh; `TQDM_DISABLE=1` +
banners + a 20s heartbeat (garbled tqdm once hid an hour-long hang).

### Base pinned for BOTH arms `[FIX: base definition mismatch]`

Draft D's `nflr_ref` (`FWD_PROP=0 LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2`, SH-0) is NOT the pristine
`crcd_base.py` (forward_prop=True, full lrs). Export the SAME `nflr_ref` env for BOTH arms so the ONLY
differing variable is `FLOW_MAP`. Add `gaussian_simplification`/SH-0 to the injector only if SH-0 is the
intended base; otherwise pin the base to pristine and say so. The injected `flow_map` block defaulting off
makes `flowmap-off == base` independent of injection state.

```bash
#!/usr/bin/env bash
# v1 flow_map MAPPING-CATCHUP A/B on EndoGSLAM CRCD. base vs flow_map vs uniform-control, n=3,
# on the DYNAMIC snippet (highest depth-residual P99 from attribution survey). Default-off == base.
# Parity-gated (Inc-0) BEFORE any run. READABLE logs (banners + heartbeat + tqdm-suppressed).
set -uo pipefail
ENDO=${ENDO:-/content/EndoGSLAM}; CFG=configs/crcd/crcd_base.py
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/GS_flowmap_ab_20260623}
LOG="$ENDO/_flowmap_logs"; mkdir -p "$LOG"; cd "$ENDO"
export TQDM_DISABLE=1
export OMP_NUM_THREADS=$(( $(nproc) / ${PARALLEL:-1} ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
BASE="FWD_PROP=0 LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2"   # nflr_ref pinned for ALL arms
SNIPPETS=${SNIPPETS:-"C1_001"}                         # set from the attribution survey (max depth-P99)
banner(){ echo; echo "==================== $* ($(date +%H:%M:%S)) ===================="; }

banner "STAGE 0  inject flow_map block (idempotent)"
python Addons/gs/inject_flowmap_knobs.py "$ENDO/$CFG" || { echo "FATAL inject"; exit 1; }

banner "STAGE 1  parity gate (Inc-0)"
python Addons/gs/regression/test_flowmap_inc0.py "$ENDO/$CFG" 2>&1 | tee "$LOG/parity.log"
grep -q '>>> INC0 PARITY PASS' "$LOG/parity.log" || { echo "FATAL: parity FAIL -> ABORT"; exit 1; }

run_one(){   # scene tag "EXTRA_ENV" seed
  local sc="$1" tag="$2" extra="$3" seed="${4:-0}"
  local rn="${sc}_${tag}_s${seed}" out="experiments/CRCD_base/${rn}"
  mkdir -p "$out"; [ -f "$out/.DONE" ] && { echo "[$rn] .DONE skip"; return 0; }
  banner "RUN $rn  env:[$BASE $extra]"
  ( env RUN_TAG="$tag" SEED="$seed" SCENE_NUM=0 FM_HEARTBEAT=1 $BASE $extra \
        python scripts/main.py "$CFG" > "$LOG/${rn}.slam.log" 2>&1 ) & local pid=$!
  local t0=$(date +%s)
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local hb=$(grep -a 'FM_HB' "$LOG/${rn}.slam.log" | tail -1); local el=$(( $(date +%s) - t0 ))
    [ -n "$hb" ] && printf "  [%s] %s · elapsed %dm%02ds\n" "$rn" "${hb#*FM_HB }" $((el/60)) $((el%60))
  done
  wait $pid || { echo "[$rn] !! SLAM FAILED"; tail -8 "$LOG/${rn}.slam.log"; touch "$out/.FAILED"; return 1; }
  banner "EVAL $rn"
  env RUN_TAG="$tag" SEED="$seed" $BASE $extra python scripts/gs_eval.py --config "$CFG" --run "$out" \
        --p99_cache "$out/p99_per_frame.npy" --holdout_every "${HOLDOUT:-5}" \
        > "$LOG/${rn}.eval.log" 2>&1 || { echo "[$rn] !! EVAL FAILED"; tail -8 "$LOG/${rn}.eval.log"; touch "$out/.FAILED"; return 1; }
  [ -d /content/drive/MyDrive ] && { d="$DRIVE/$rn"; mkdir -p "$d"; \
     cp "$out"/metrics*.* "$out"/*_6panel.mp4 "$out"/est_c2w_data.txt "$LOG/${rn}".*.log "$d/" 2>/dev/null; }
  touch "$out/.DONE"
  echo "[$rn] DONE -> $(grep -E 'HELD-OUT|GUARD|dynPSNR' "$LOG/${rn}.eval.log" | tr -s ' \n' ' ')"
}

banner "A/B  base vs flow_map vs uniform-ctrl  n=3  on [$SNIPPETS]"
for sc in $SNIPPETS; do for s in 0 1 2; do
  run_one "$sc" base    ""                                                                          "$s"
  run_one "$sc" flowmap "FLOW_MAP=1 FM_LAMBDA=1.0 FM_DEADBAND=3.0 FM_DEPTH_DB=2.0 FM_HOLDOUT_EVERY=${HOLDOUT:-5}" "$s"
  run_one "$sc" unictrl "FLOW_MAP=1 FM_LAMBDA=1.0 FM_UNIFORM=1 FM_HOLDOUT_EVERY=${HOLDOUT:-5}"      "$s"
done; done
```

Heartbeat in main.py (gated on `FM_HEARTBEAT`, one line, NOT tqdm), top of each frame iter:
`print(f"FM_HB frame {time_idx+1}/{num_frames} · {time.time()-loop_t0:.0f}s · eta {eta:.0f}s", flush=True)`.
With `TQDM_DISABLE=1` the log is clean banners + `FM_HB` + the eval summary.

### A/B plan

| Item | Spec |
|---|---|
| Arms | `base` (nflr_ref). `flowmap` (= base + `FLOW_MAP=1 FM_LAMBDA=1 FM_DEADBAND=3 FM_DEPTH_DB=2`). `unictrl` (= base + constant up-weight, NO gating) — **the specificity control `[FIX]`**. Only differing var = the flag. |
| Snippet | The one the attribution survey flags max **depth-residual P99** (run survey FIRST; pass via `SNIPPETS=`). |
| n | 3 seeds (0,1,2) per arm. Base seed-std = noise floor; win must clear it (non-overlapping). |
| Held-out DYNAMIC | k=5 mapping-holdout (§4a), DYNAMIC by frozen base depth-residual P99 (§4b). Headline = HELD-OUT∩DYNAMIC. |
| PRIMARY | `flowmap dynPSNR > base dynPSNR` AND `flowmap dynPSNR > unictrl dynPSNR` (must beat the uniform control), n=3 non-overlapping. |
| GUARD | HELD-OUT∩STATIC + sensor-independent non-dynamic-region PSNR ≥ base − τ (§4d). Breach kills the result. |
| Secondary | full-frame PSNR/SSIM/LPIPS + Sim3 ATE (mapping-only ⇒ ATE ~flat; a large move = a leak). |
| Two diagnostic sets | NUMERICAL table {dynPSNR, nondynPSNR, PSNR/SSIM/LPIPS, ATE}×n=3 + seed-std; VISUAL 6-panel (RGB \| rendered \| depth \| **w_map** \| flow-resid \| reproj-depth-resid). Auto-ship. |
| Decision | dynPSNR win (n=3 non-overlap) AND beats unictrl AND guard held ⇒ **KEEP v1**. Flat/loss OR guard breach OR no-beat-uniform ⇒ static-chase ceiling ⇒ escalate to **v2 (deform field)**. |

---

## (9) Implementation CHECKLIST (each step independently testable)

1. **`gs_flow_gate.py`** (§3). Test: unit-test on two synthetic depth frames with a known rigid camera
   move (depth_resid≈0 everywhere ⇒ `w≡1`) and a known local depth bump (`w>1` only there); assert
   `arange(dtype=)`, returns ones on warm-up + F-fit failure, `frac` telemetry sane.
2. **Config block** (§6) via `inject_flowmap_knobs.py` (idempotent, anchor-asserted). Test: load twice
   (env unset / `FLOW_MAP=0`) ⇒ deep-equal, `enable is False`.
3. **main.py Edit A+B** (§2). Test: call `get_loss(...,mapping=True, map_weight=None)` ⇒ identical to old
   path; with a `w≡1` tensor ⇒ identical loss value (within fp tol).
4. **main.py call-site hook** (§7), current-frame-only attach. Test: assert `map_weight` is `None` on
   every keyframe-replay iter and non-None only on the current-frame iter.
5. **Inc-0 parity** `test_flowmap_inc0.py` (§7) — Tier A hard gate, Tier B `lam=0` trace. Run BEFORE any
   A/B; gate the runbook on `>>> INC0 PARITY PASS`.
6. **Holdout in main.py** (§4a), `eval_holdout_every` default 0; write `holdout_idx.npy`. Test: with
   key 0 ⇒ no skipped frames (base); with 5 ⇒ `(time_idx%5==4)` frames tracked but never mapped.
7. **`render_metrics` refactor** (§4c) → return per-frame arrays (means unchanged). Test: `nanmean` of
   the arrays == old scalar return.
8. **`gs_eval` subset matrix + guard** (§4b/4c/4d) → `metrics_split.json`, frozen base depth-residual
   dynamic label, sensor-independent guard region, `guard_pass`. Test on a finished base run dir.
9. **attribution_panel cache** (§4f) — `np.save` p99 + ref_index. Test: file appears, length == #pairs.
10. **6-panel video panels** (§4e) — w_map + reproj-resid panels / `frame_tags.txt`. Test: video renders,
    degrades gracefully if `generate_video.py` lacks `--frame_tags`.
11. **Runbook** `flow_map_ab_20260623.sh` (§8) — stage banners, heartbeat, parity-then-A/B, uniform-ctrl
    arm, summary. Test: dry-run `SNIPPETS=C1_001 HOLDOUT=5` 6-frame smoke.
12. **Attribution survey FIRST** to pick the DYNAMIC snippet; then n=3 A/B; then read the decision (§8).

---

## (10) Risks + open decisions

- **R1 — Adam global-scale cancellation `[FIX]`:** a near-global `w` is an LR boost, not a re-fit. Floor-
  at-1 + non-uniform-on-subregion design + the `frac>0.5` warning + the **uniform-control arm** (v1 must
  beat it) are the three guards. *Open:* the right `lam`/`w_max` so contrast survives without an effective
  global LR change — sweep small.
- **R2 — depth_deadband below MoGe noise ⇒ saturation degenerate:** set it to the per-snippet depth-noise
  floor; the uniform control + `frac` telemetry detect saturation. *Open:* measure the floor per snippet.
- **R3 — holdout reduces map coverage:** k=5 drops 20% of mapping frames from BOTH arms equally (fair),
  but a sparse snippet may degrade both. *Open:* k=5 vs k=10 — report coverage.
- **R4 — depth-residual dynamic label needs reasonable est poses:** on a bad-ATE snippet the reproj
  label is noisy. Mitigation: label from the BASE arm (most stable poses), frozen, applied to both.
- **R5 — forward-warp last-writer-wins + nearest sampling:** adequate for smooth weights; bilinear
  `D_cur` sampling + splatting are deferred accuracy upgrades.
- **R6 — `require_depth=False` Sampson-only fallback** loses specular rejection; keep depth ON for the
  headline; the fallback is only for snippets with unusable depth (logged).
- **Decision deferred to the run:** if dynPSNR beats base but NOT the uniform control ⇒ the win is a
  global LR effect, NOT localized catch-up ⇒ NOT a v1 win (tune `lam`/deadbands or escalate to v2).
