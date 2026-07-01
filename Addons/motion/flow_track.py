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
                 n_groups=12, ransac_thresh=1.0, deadband=3.0, min_px=50, seed=0,
                 mode='region', smooth=5, soft_scale=0.0):
    """E0 MAPPING router: per-pixel MOVING-mask [H,W] in {0,1}. 1 = scene-moving (route the deformation
    field HERE), 0 = static/camera (field OFF -> stays sharp). The WHAT-MOVES axis. Two modes:
      mode='region' (default): paint whole DINO k-means regions whose MEDIAN Sampson (vs a fitted
        FUNDAMENTAL matrix) > deadband. Scene-level + robust BUT (a) COARSE — 14px patches x ~12 regions,
        NEAREST-upsampled -> blobs bleed the field into bg, can't follow partial deformation; and (b) the
        fundamental matrix is DEGENERATE for forward/ZOOM camera motion -> it flags the camera zoom as
        deformation.
      mode='pixel': fit a global HOMOGRAPHY (models camera rotation+ZOOM exactly, where F is degenerate)
        and threshold the per-pixel REPROJECTION residual ||p2 - H@p1|| at FULL resolution -> fixes BOTH the
        low-res bleed AND the zoom false-positive. Denoised (median + morph open/close). dino_g unused (caller
        may pass None). Caveat: H conflates large parallax/non-planarity with deformation (small for smooth
        endoscopic tissue; the 3D depth+pose residual is the upgrade once the pose un-freezes in the combine).
    Returns all-zeros if the fit fails -> field off everywhere (neutral)."""
    import cv2
    from sklearn.cluster import KMeans
    flow = _raft_flow(raft_model, raft_tf, ref_bgr, cur_bgr, device)
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2); p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(4000, len(p1))).astype(np.int64)
    if mode == 'pixel':
        Hmat, _ = cv2.findHomography(p1[idx], p2[idx], cv2.RANSAC, ransac_thresh)
        if Hmat is None:
            return np.zeros((H, W), np.float32)            # no fit -> field off (neutral)
        p1h = np.concatenate([p1, np.ones((len(p1), 1), np.float32)], 1)
        proj = (Hmat.astype(np.float32) @ p1h.T).T
        proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
        rmap = np.linalg.norm(proj - p2, axis=1).reshape(H, W).astype(np.float32)   # per-pixel reproj residual
        _mb = smooth if smooth in (3, 5) else 5                                     # cv2 medianBlur float32: k in {3,5}
        rs = cv2.medianBlur(rmap, _mb) if (smooth and smooth > 1) else rmap
        if soft_scale and soft_scale > 0:
            # SOFT routing (user 06-21): apply the field PROPORTIONALLY to the camera-subtracted residual
            # ("residual -> deformation weight") instead of a hard on/off. w ramps 0->1 over
            # [deadband, deadband+soft_scale]: pixels ~consistent with the camera get ~0 (sharp), clearly
            # deforming pixels get ~1, ambiguous get a graded weight -> NO black-and-white boundaries, and it
            # degrades smoothly when the route is slightly wrong. Light Gaussian for spatial coherence.
            route = np.clip((rs - deadband) / soft_scale, 0.0, 1.0).astype(np.float32)
            return cv2.GaussianBlur(route, (0, 0), 1.5).astype(np.float32)
        route = (rs > deadband).astype(np.float32)                                  # hard binary (default)
        if smooth and smooth > 1:
            _k = np.ones((smooth, smooth), np.uint8)
            route = cv2.morphologyEx(route, cv2.MORPH_OPEN, _k)                     # drop isolated speckle
            route = cv2.morphologyEx(route, cv2.MORPH_CLOSE, _k)                    # fill small holes
        return route.astype(np.float32)
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


def rigid_flow_residual(ref_bgr, cur_bgr, ref_depth, T_rel, fx, fy, cx, cy, model, tf, device):
    """DEPTH-ANCHORED per-pixel rigid-flow residual [H,W] (the depth-supervisor; replaces the 2D
    fundamental-matrix Sampson of flow_residual). For each REF pixel: back-project by its depth ->
    ref-camera 3D point, push through the candidate relative pose T_rel (ref-cam -> cur-cam, 4x4),
    re-project -> the RIGID flow the camera ALONE would produce at that pixel's distance (parallax:
    big near, small far). Compare to the observed RAFT flow; ||observed - rigid|| is ~0 where the
    region moves WITH the camera (any depth), high where it moves INDEPENDENTLY (tool / deforming
    tissue). No F-matrix -> no forward/planar degeneracy, not hijacked by a large moving object;
    depth disambiguates 'near static thing with big parallax' from 'thing moving on its own'.
      ref_depth : [H,W] Z-depth, SAME metric scale as the poses (DDS tracks in the scaled-depth frame).
      T_rel     : [4,4] = inv(c2w_cur) @ c2w_ref in the OPENCV convention (ref-cam -> cur-cam). DDS stores
                  poses + batch['direction'] in OPENGL (z=-1, y-flipped), so the CALLER must convert the
                  c2w's GL->CV (c2w @ diag(1,-1,-1,1)) BEFORE forming T_rel -- this function builds OpenCV
                  ray dirs internally (z=+1) and projects with z>0, so an OpenGL T_rel would push every
                  point behind the camera (Z<0) and the bad-mask would zero the whole residual (a no-op).
    Invalid-depth / behind-camera pixels -> 0 residual (neutral; no spurious down-weight)."""
    f_obs = _raft_flow(model, tf, ref_bgr, cur_bgr, device)               # [H,W,2] observed ref->cur
    H, W = f_obs.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float32)  # OpenCV (z=+1)
    X = ref_depth[..., None].astype(np.float32) * dirs                     # [H,W,3] ref-cam 3D (Z=+depth)
    R = T_rel[:3, :3].astype(np.float32); t = T_rel[:3, 3].astype(np.float32)
    Xc = X @ R.T + t                                                       # [H,W,3] cur-cam 3D
    Z = Xc[..., 2]
    Zc = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    u = fx * Xc[..., 0] / Zc + cx; v = fy * Xc[..., 1] / Zc + cy           # predicted cur pixel (OpenCV)
    f_rig = np.stack([u - uu, v - vv], -1).astype(np.float32)              # predicted RIGID flow
    resid = np.linalg.norm(f_obs - f_rig, axis=-1).astype(np.float32)      # [H,W]
    bad = (ref_depth <= 0) | (Z <= 1e-6) | ~np.isfinite(resid)
    resid[bad] = 0.0
    return resid


def region_soft_weight(resid, dino_g, n_groups=12, mad_c=2.0, w_floor_px=1.0,
                       w_min=0.1, min_px=50, seed=0):
    """Pool the rigid-flow residual into DINO k-means regions and map each region's MEDIAN residual to
    a soft trust weight [H,W] in [w_min,1]: 1 = moves-with-camera (trust for the pose solve), ->0 =
    moves independently (down-weight). NO fixed deadband -- the knee is set ADAPTIVELY from robust
    residual stats: scale = max(mad_c*1.4826*MAD, w_floor_px), and the weight uses the region's EXCESS
    OVER the rigid median, so a region is down-weighted only when it is an OUTLIER vs the rigid
    majority. A fully-rigid / near-still frame -> excess~0 -> w~1 everywhere (true NOP), independent of
    scene / dataset. Returns (w[H,W] float32, med, mad, scale)."""
    import cv2
    from sklearn.cluster import KMeans
    H, W = resid.shape
    r = resid.reshape(-1).astype(np.float32); r = r[np.isfinite(r)]
    med = float(np.median(r)) if r.size else 0.0
    mad = float(np.median(np.abs(r - med))) if r.size else 0.0
    scale = max(mad_c * 1.4826 * mad, float(w_floor_px), 1e-6)   # 1e-6 floor: never divide by 0 if w_floor_px=0 & MAD=0
    gh, gw, C = dino_g.shape
    X = dino_g.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    w = np.ones((H, W), np.float32)
    for k in range(n_groups):
        m = lab == k
        if int(m.sum()) < min_px:
            continue
        r_k = float(np.median(resid[m]))
        excess = max(0.0, r_k - med)
        w[m] = 1.0 / (1.0 + (excess / scale) ** 2)
    return np.clip(w, w_min, 1.0).astype(np.float32), med, mad, scale


def depth_pooled_weight(ref_bgr, cur_bgr, depth, dino_g, model, tf, device,
                        n_groups=12, mad_c=2.0, w_floor_px=1.0, w_min=0.1, min_px=50, seed=0):
    """MODE B (SIMPLE, pose-free) -- pool BOTH flow AND depth into the DINO k-means regions, then bring every
    region to a COMMON PLANE by multiplying its median flow by its median depth: v_k = median_flow_k *
    median_depth_k. Camera TRANSLATION parallax is f*t/Z, so v_k = f*t (distance-INVARIANT) -> all static
    regions collapse onto one camera-consensus vector regardless of distance ('universal motion'); a region
    moving independently of the camera (tool/deformer) lands off the consensus. Weight = soft threshold on
    each region's DEVIATION from the robust (median) consensus, MAD-scaled with a sub-pixel floor -> NO fixed
    deadband. Pose-free: no rigid-flow prediction, so it sidesteps L0's const-velocity-prior noise floor; the
    median consensus is robust to a large tool. CAVEAT: exact only for translation -- camera ROTATION flow is
    depth-independent, so *depth mis-scales it (contaminates on rotation-heavy frames). depth may be up-to-
    scale (relative); the consensus + deviations are all in the same units so the absolute scale cancels.
    Returns (w[H,W] float32, med, mad, scale)."""
    import cv2
    from sklearn.cluster import KMeans
    flow = _raft_flow(model, tf, ref_bgr, cur_bgr, device)                 # [H,W,2]
    H, W = flow.shape[:2]
    gh, gw, C = dino_g.shape
    X = dino_g.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    dvalid = (depth > 0) & np.isfinite(depth)
    dmed = float(np.median(depth[dvalid])) if dvalid.any() else 1.0        # median depth (relative unit)
    vK = np.zeros((n_groups, 2), np.float32); ok = np.zeros(n_groups, bool)
    for k in range(n_groups):
        m = (lab == k) & dvalid
        if int(m.sum()) < min_px:
            continue
        vK[k] = np.median(flow[m], axis=0) * float(np.median(depth[m]))    # depth-normalised region flow
        ok[k] = True
    if int(ok.sum()) < 2:
        return np.ones((H, W), np.float32), 0.0, 0.0, 0.0
    cons = np.median(vK[ok], axis=0)                                       # camera-translation consensus
    dev = np.linalg.norm(vK - cons[None, :], axis=1)                       # per-region deviation [n_groups]
    med = float(np.median(dev[ok])); mad = float(np.median(np.abs(dev[ok] - med)))
    scale = max(mad_c * 1.4826 * mad, float(w_floor_px) * dmed, 1e-6)      # floor = ~1px of flow at median depth (+1e-6 guard)
    w = np.ones((H, W), np.float32)
    for k in range(n_groups):
        if not ok[k]:
            continue
        excess = max(0.0, dev[k] - med)
        w[lab == k] = 1.0 / (1.0 + (excess / scale) ** 2)
    return np.clip(w, w_min, 1.0).astype(np.float32), med, mad, scale


def rigid_solve_pnp(ref_bgr, cur_bgr, ref_depth, fx, fy, cx, cy, model, tf, device,
                    tool_mask=None, flow_advance_px=1.5, reproj_px=2.0, min_inliers=200, max_fit=6000):
    """MODE A (the hardened estimator; replaces the F-matrix gate). Robust 2D-3D PnP camera-motion solve:
    RAFT flow ref->cur gives matches p <-> p'=p+flow; back-project the REF pixels only (X_r = D_r*Kinv*p) and
    solve the cur-camera pose that reprojects X_r onto p' via cv2.solvePnPRansac. REF DEPTH ONLY -- D_cur is
    never used, so no double depth-noise; forward motion is observable from radial pixel looming, not noisy
    z-differencing. Tool pixels (tool_mask True) are HARD-EXCLUDED before solving (a rigidly-moving tool is a
    valid SE3 and would hijack a robust fit). Returns None (caller falls back to const-velocity) if below the
    flow-noise floor / too few valid px / RANSAC fails / too few inliers. Else (T_rel_cv [4,4] OpenCV
    ref-cam->cur-cam, resid_px [H,W] per-pixel reprojection residual, info). Up-to-scale (relative ref depth)
    -> NOT a metric anchor; the caller uses T_rel INIT-ONLY (the SDF tracker can overrule it) and pools
    resid_px -> the per-ray trust weight."""
    import cv2
    flow = _raft_flow(model, tf, ref_bgr, cur_bgr, device)                 # [H,W,2]
    H, W = flow.shape[:2]
    if float(np.median(np.linalg.norm(flow.reshape(-1, 2), axis=1))) < float(flow_advance_px):
        return None                                                        # below the flow-noise floor -> don't solve
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float32)  # OpenCV z=+1
    Xr = ref_depth[..., None].astype(np.float32) * dirs                    # [H,W,3] ref-cam 3D
    p2 = np.stack([uu + flow[..., 0], vv + flow[..., 1]], -1).astype(np.float32)  # [H,W,2] cur pixels
    valid = ((ref_depth > 0) & np.isfinite(ref_depth) &
             (p2[..., 0] >= 0) & (p2[..., 0] < W) & (p2[..., 1] >= 0) & (p2[..., 1] < H))
    if tool_mask is not None:
        valid &= ~tool_mask
    obj = Xr[valid].reshape(-1, 3); img = p2[valid].reshape(-1, 2)
    if len(obj) < min_inliers:
        return None
    idx = np.linspace(0, len(obj) - 1, min(int(max_fit), len(obj))).astype(np.int64)   # deterministic subsample
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    try:
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj[idx].astype(np.float64), img[idx].astype(np.float64), K, None,
            reprojectionError=float(reproj_px), iterationsCount=200, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        return None
    if (not ok) or inl is None or len(inl) < min_inliers:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float32); T[:3, :3] = R.astype(np.float32); T[:3, 3] = tvec.reshape(3).astype(np.float32)
    proj = (R @ Xr.reshape(-1, 3).T.astype(np.float64) + tvec).T           # [HW,3]
    zraw = proj[:, 2]
    z = np.where(zraw <= 1e-6, 1e-6, zraw)
    pe = np.stack([fx * proj[:, 0] / z + cx, fy * proj[:, 1] / z + cy], -1)
    resid = np.linalg.norm(pe - p2.reshape(-1, 2), axis=1).reshape(H, W).astype(np.float32)
    # neutral-out (resid 0) no-depth / out-of-bounds / tool AND BEHIND-CAMERA points: a behind-cam point has z
    # clipped to 1e-6 -> huge reprojection residual that would poison the region median/MAD -> the trust weights.
    bad = (~valid) | (zraw.reshape(H, W) <= 1e-6)
    resid[bad] = 0.0
    info = dict(inlier_frac=float(len(inl) / max(len(idx), 1)), n=int(len(idx)),
                t_norm=float(np.linalg.norm(tvec)),
                rot_deg=float(np.degrees(np.linalg.norm(rvec))),
                reproj_med=float(np.median(resid[valid])) if valid.any() else 0.0)
    return T, resid, info
