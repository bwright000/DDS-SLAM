#!/usr/bin/env python3
"""In-loop CAUSAL flow-residual for the DDS-SLAM tracking down-weight (flow-as-sensor).

Self-contained (decoupled from the offline feature_flow_probe). At tracking frame t it computes a
per-pixel CAMERA-vs-SCENE residual between a PAST reference frame and the current frame — STRICT
causality: the reference index < t. RAFT dense flow -> ONE fundamental-matrix fit (RANSAC) ->
Sampson distance per pixel. Static (incl. camera parallax) -> ~0; deforming/tool tissue -> high.
The caller turns resid -> a per-ray weight that down-weights deforming pixels in the pose solve.
Runs ONCE per frame (not per tracking iteration). Imported ONLY when flow_track.enable=true — the
base path never touches it. Validated offline: synthetic 0.00 vs 7.44; GT camera-timing +0.94.
"""
import numpy as np


def load_raft(device, small=False):
    from torchvision.models.optical_flow import (raft_small, raft_large,
                                                 Raft_Small_Weights, Raft_Large_Weights)
    w = (Raft_Small_Weights if small else Raft_Large_Weights).DEFAULT
    m = (raft_small if small else raft_large)(weights=w, progress=False).to(device).eval()
    return m, w.transforms()


def _raft_flow(model, tf, a_bgr, b_bgr, device):
    """Dense flow a->b at full res. a,b: [H,W,3] uint8 BGR. Returns [H,W,2] (u=dx,v=dy)."""
    import torch, cv2
    import torch.nn.functional as Fn
    def prep(im):
        return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None]
    ta, tb = tf(prep(a_bgr), prep(b_bgr))
    H, W = ta.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8
    ta = Fn.pad(ta, (0, pw, 0, ph), mode='replicate'); tb = Fn.pad(tb, (0, pw, 0, ph), mode='replicate')
    with torch.inference_mode():
        fl = model(ta.to(device), tb.to(device))[-1]
    return fl[0, :, :H, :W].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def _sampson(F, p1, p2):
    N = len(p1)
    x1 = np.hstack([p1, np.ones((N, 1))]); x2 = np.hstack([p2, np.ones((N, 1))])
    Fx1 = x1 @ F.T; Ftx2 = x2 @ F
    num = np.sum(x2 * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return np.sqrt(num / den).astype(np.float32)


def flow_residual(ref_bgr, cur_bgr, model, tf, device, ransac_thresh=1.0, max_fit=4000):
    """Per-pixel Sampson residual [H,W]: camera-consistent (incl. parallax) ~0, scene motion high.
    Returns zeros if the F-fit fails (too few correspondences) -> caller sees a neutral weight."""
    import cv2
    flow = _raft_flow(model, tf, ref_bgr, cur_bgr, device)
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(max_fit, len(p1))).astype(np.int64)   # deterministic subsample
    F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or F.shape != (3, 3):
        return np.zeros((H, W), np.float32)
    return _sampson(F.astype(np.float64), p1, p2).reshape(H, W)


def camera_motion(ref_bgr, cur_bgr, model, tf, device):
    """|median flow vector| = the dominant rigid motion = 'is the camera moving' proxy
    (validated +0.94 vs GT camera). Robust to a deforming minority (median sits on the static
    majority). Used by the ON/OFF gate: small -> camera still, large -> camera moving."""
    flow = _raft_flow(model, tf, ref_bgr, cur_bgr, device)
    gvec = np.median(flow.reshape(-1, 2), axis=0)
    return float(np.linalg.norm(gvec))


def load_dino(device, backbone='dinov2_vits14_reg'):
    """DINOv2 (reg) backbone for the per-region grouping (on-the-fly, torch.hub auto-download)."""
    import torch
    return torch.hub.load('facebookresearch/dinov2', backbone).to(device).eval()


def dino_grid(rgb_bgr, dino_model, device):
    """[gh,gw,C] DINO patch grid from an RGB(BGR) frame (patch-14)."""
    import torch, cv2
    im = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = im.shape[:2]; gh, gw = (H // 14) * 14, (W // 14) * 14
    im = cv2.resize(im, (gw, gh))
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(device)
    with torch.inference_mode():
        tok = dino_model.forward_features(t)['x_norm_patchtokens'][0].cpu().numpy()
    return tok.reshape(gh // 14, gw // 14, -1).astype(np.float32)


def agreement_gate(ref_bgr, cur_bgr, dino_g, raft_model, raft_tf, device,
                   n_groups=12, ransac_thresh=1.0, deadband=3.0, min_px=50, seed=0):
    """The probe's per-region camera/scene test, as a frame-level signal. Pool flow into DINO regions
    and ask: do the per-region motion vectors AGREE with ONE rigid (camera) motion?
    Returns (cam_mag, disagree_frac):
      cam_mag       = |median flow| (consensus motion magnitude -> is anything moving)
      disagree_frac = fraction of DINO regions whose motion DISAGREES with the consensus rigid motion
                      (median Sampson residual > deadband). LOW = features agree (camera/rigid),
                      HIGH = features disagree (scene deforming).
    Gate: TRACK iff cam_mag > cam_thresh AND disagree_frac <= disagree_thresh."""
    import cv2
    from sklearn.cluster import KMeans
    flow = _raft_flow(raft_model, raft_tf, ref_bgr, cur_bgr, device)
    H, W = flow.shape[:2]
    cam_mag = float(np.linalg.norm(np.median(flow.reshape(-1, 2), axis=0)))
    # per-pixel Sampson residual vs the ONE consensus rigid motion (parallax-aware)
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(4000, len(p1))).astype(np.int64)
    F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or F.shape != (3, 3):
        return cam_mag, 0.0                       # no fit -> treat as agreeing (rigid)
    resid = _sampson(F.astype(np.float64), p1, p2).reshape(H, W)
    # pool to DINO regions; count regions that disagree with the consensus
    gh, gw, C = dino_g.shape
    X = dino_g.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    dis, tot = 0, 0
    for k in range(n_groups):
        m = lab == k
        if m.sum() < min_px:
            continue
        tot += 1
        if float(np.median(resid[m])) > deadband:
            dis += 1
    return cam_mag, (dis / max(tot, 1))


def region_route(ref_bgr, cur_bgr, dino_g, raft_model, raft_tf, device,
                 n_groups=12, ransac_thresh=1.0, deadband=3.0, min_px=50, seed=0):
    """E0 MAPPING router: per-pixel MOVING-mask [H,W] in {0,1}. 1 = scene-moving region (route the
    deformation field HERE — let Δx warp it), 0 = static/camera region (field OFF -> stays sharp).
    This is the INVERSE-of-tracking signal: the SAME per-DINO-region camera-vs-scene test the
    tracking gate uses (median Sampson residual vs the ONE consensus rigid motion > deadband), but
    returned DENSE (the per-region decision painted back to pixels) instead of collapsed to a scalar
    fraction. Returns all-zeros if the F-fit fails -> field off everywhere (neutral). This is the
    WHAT-MOVES axis only; the tissue-vs-tool WHAT-KIND split (tool exclusion) is a later rung."""
    import cv2
    from sklearn.cluster import KMeans
    flow = _raft_flow(raft_model, raft_tf, ref_bgr, cur_bgr, device)
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(4000, len(p1))).astype(np.int64)
    F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, ransac_thresh, 0.999)
    route = np.zeros((H, W), np.float32)
    if F is None or F.shape != (3, 3):
        return route                              # no fit -> field off everywhere (neutral)
    resid = _sampson(F.astype(np.float64), p1, p2).reshape(H, W)
    gh, gw, C = dino_g.shape
    X = dino_g.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    for k in range(n_groups):
        m = lab == k
        if m.sum() < min_px:
            continue
        if float(np.median(resid[m])) > deadband:
            route[m] = 1.0                        # moving region -> route the field here
    return route


def residual_to_weight(resid, alpha=0.5, w_min=0.1, w_max=1.0, deadband=0.0):
    """resid[...] -> down-weight. DEADBAND: w=1 for resid<=deadband, so clean/camera frames (low,
    NOISY residual) are a TRUE NOP (uniform weight -> no pose perturbation) and the down-weight
    CONCENTRATES on clear deformation: w = clip(1/(1+alpha*max(0, resid-deadband))).
    deadband=0 reproduces the original broad behaviour."""
    excess = np.maximum(resid - deadband, 0.0)
    w = 1.0 / (1.0 + alpha * excess)
    return np.clip(w, w_min, w_max).astype(np.float32)
