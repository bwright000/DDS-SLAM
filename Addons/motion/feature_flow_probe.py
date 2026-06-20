#!/usr/bin/env python3
"""Feature optical-flow probe — measure motion, SPLIT camera vs scene, route to tracking/mapping.

THE STEP (memory project_combine_routing_model_20260619, "flow-as-sensor"): the per-pixel sigma^2
hedge is blind to deformation (appearance-preserving: liver slides -> low residual). We MEASURE motion
with optical flow, then split it into CAMERA motion (-> tracking) and SCENE motion (-> mapping):
  1. RAFT dense per-pixel flow (frame t -> t+1).
  2. fit ONE camera to the flow's majority via the fundamental matrix (RANSAC). How much each pixel
     VIOLATES that one camera's epipolar geometry = the Sampson distance = the SCENE-motion residual.
     Static tissue (incl. camera parallax) satisfies it -> ~0; deforming/tool tissue violates it -> high.
     Depth-FREE: parallax is camera-consistent so it stays ~0 (no MoGe needed for this first cut;
     pose+depth flow-residual is the metric upgrade -- machinery in bake_motion_residual.py).
  3. group pixels by DINO feature (k-means, warm-started across frames so groups don't flash) and
     pool the residual per region -> low=static (trust in TRACKING), high=deforming (route to MAPPING).

Both diagnostic sets (the standing two-set rule):
  VISUAL  - a video: frame | dense flow | confident | DINO groups | per-feature arrows |
            residual (deformation heatmap) | routing (red=deforming) | region-avg flow.
  NUMERIC - region_EPE (per-feature vs dense flow), explained/coherence, AND the camera split:
            mean/p90 Sampson residual, deforming fraction, per-region residual+deform flag.
  Confidence = forward-backward flow consistency. Coherence is gated on real motion (>min_motion).

Standalone, torch>=2 env (torchvision RAFT + optional torch.hub DINO). No SLAM, GPU optional (CPU ok).
  python Addons/motion/feature_flow_probe.py \
    --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png' \
    --out_dir output/flowprobe_c1 --n_groups 12 --stride 8 --max_frames 80
  python Addons/motion/feature_flow_probe.py --selftest      # metric + camera-split math, no RAFT/DINO/data
"""
import os, sys, glob, argparse
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# fixed BGR palette for up to 16 groups
PALETTE = np.array([[60, 60, 60], [0, 200, 0], [0, 0, 230], [230, 160, 0], [200, 0, 200],
                    [0, 200, 200], [120, 80, 200], [80, 160, 80], [0, 120, 255], [255, 120, 0],
                    [120, 200, 0], [200, 120, 120], [0, 80, 160], [160, 0, 80], [80, 200, 160],
                    [160, 160, 0]], np.uint8)


# ----------------------------- metric math (selftest-covered) -----------------------------
def region_vectors(flow, lab, conf, K, min_px=10, min_motion=1.0):
    """Per-region ROBUST (median) flow over CONFIDENT pixels + coherence/centroid/counts.
    Coherence is gated on real motion (|median|>min_motion) -> NaN on near-static regions
    (where 1-spread/|median| is ill-defined and explodes negative)."""
    vecs = np.full((K, 2), np.nan, np.float32); coh = np.full(K, np.nan, np.float32)
    npx = np.zeros(K, int); ncf = np.zeros(K, int); cent = np.zeros((K, 2), np.float32)
    for k in range(K):
        m = (lab == k); npx[k] = int(m.sum())
        if m.any():
            ys, xs = np.where(m); cent[k] = [xs.mean(), ys.mean()]
        mc = m & conf; ncf[k] = int(mc.sum())
        if mc.sum() < min_px:
            continue
        fl = flow[mc]                                   # [n,2]
        med = np.median(fl, axis=0); vecs[k] = med
        if np.linalg.norm(med) > min_motion:            # only score coherence where there IS motion
            d = np.linalg.norm(fl - med, axis=1).mean()
            coh[k] = 1.0 - d / (np.linalg.norm(med) + 1e-3)
    return vecs, coh, npx, ncf, cent


def region_field(flow_shape, lab, vecs):
    """Replace each pixel's flow with its REGION's vector (NaN regions -> 0)."""
    field = np.zeros(flow_shape, np.float32)
    for k in range(len(vecs)):
        if not np.isnan(vecs[k, 0]):
            field[lab == k] = vecs[k]
    return field


def epe(a, b, conf):
    e = np.linalg.norm(a - b, axis=-1)
    return float(e[conf].mean()) if conf.any() else float('nan')


def patch_field(flow, gh, gw):
    import cv2
    H, W = flow.shape[:2]
    small = cv2.resize(flow, (gw, gh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def sampson(F, p1, p2):
    """Sampson distance of correspondences p1<->p2 under F. p1,p2:[N,2]. Returns [N] (pixel-ish).
    Measures how far each correspondence is from satisfying the ONE camera's epipolar geometry."""
    N = len(p1)
    x1 = np.hstack([p1, np.ones((N, 1), np.float64)])
    x2 = np.hstack([p2, np.ones((N, 1), np.float64)])
    Fx1 = x1 @ F.T                                   # [N,3]
    Ftx2 = x2 @ F                                    # [N,3]
    num = np.sum(x2 * Fx1, axis=1) ** 2              # (x2^T F x1)^2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2 + 1e-12
    return np.sqrt(num / den).astype(np.float32)


def camera_split(flow, conf, ransac_thresh=1.0, max_fit=4000, seed=0):
    """Fit ONE camera (fundamental matrix, RANSAC) to the confident flow; Sampson residual = scene motion.
    Returns resid[H,W] (per-pixel, deformation proxy) and ok(bool)."""
    import cv2
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2)
    p2 = p1 + flow.reshape(-1, 2)
    ci = np.where(conf.ravel())[0]
    if len(ci) < 16:
        return np.zeros((H, W), np.float32), False
    rng = np.random.RandomState(seed)
    fit = ci if len(ci) <= max_fit else ci[rng.permutation(len(ci))[:max_fit]]
    F, _ = cv2.findFundamentalMat(p1[fit], p2[fit], cv2.FM_RANSAC, ransac_thresh, 0.999)
    if F is None or F.shape != (3, 3):
        return np.zeros((H, W), np.float32), False
    return sampson(F.astype(np.float64), p1, p2).reshape(H, W), True


def deform_threshold(resid, conf, ransac_thresh):
    """Adaptive 'is this deforming' cut: robust noise floor (median+3*MAD) but >= the RANSAC tolerance."""
    r = resid[conf]
    if r.size == 0:
        return max(ransac_thresh, 1.0)
    med = np.median(r); mad = np.median(np.abs(r - med)) + 1e-6
    return float(max(ransac_thresh, med + 3 * 1.4826 * mad))


def run_selftest():
    """Validate the metric math AND the camera split on synthetic data — no RAFT/DINO/data."""
    import cv2
    rng = np.random.RandomState(0)
    # (1) region flow / EPE / coherence
    H = W = 90; K = 3
    lab = np.zeros((H, W), int); lab[:, 30:60] = 1; lab[:, 60:] = 2
    true_vec = np.array([[5, 0], [0, 4], [-3, -3]], np.float32)
    flow = np.zeros((H, W, 2), np.float32)
    for k in range(K):
        flow[lab == k] = true_vec[k]
    flow += rng.randn(H, W, 2) * 0.3
    conf = np.ones((H, W), bool)
    vecs, coh, npx, ncf, cent = region_vectors(flow, lab, conf, K)
    e_reg = epe(flow, region_field(flow.shape, lab, vecs), conf)
    gvec = np.median(flow.reshape(-1, 2), axis=0)
    e_glob = epe(flow, np.broadcast_to(gvec, flow.shape), conf)
    m1 = (e_reg < 0.6) and (e_glob > 2.0) and np.allclose(vecs, true_vec, atol=0.4) and (coh.min() > 0.7)
    print(f"SELFTEST flow: region_EPE={e_reg:.2f} (<0.6) global_EPE={e_glob:.2f} (>2) coh={np.round(coh,2)} -> {'ok' if m1 else 'FAIL'}")
    # (2) camera split: horizontal-translation camera w/ depth parallax = static (low Sampson);
    #     a vertically-moving patch = scene motion (high Sampson).
    n = 800
    x = rng.uniform(50, 600, n); y = rng.uniform(50, 400, n)
    dx = rng.uniform(3, 30, n)                       # parallax: different horizontal shift by depth
    p1 = np.stack([x, y], 1).astype(np.float32)
    p2 = np.stack([x + dx, y], 1).astype(np.float32)  # static under horizontal cam motion (dy=0)
    mov = np.zeros(n, bool); mov[:60] = True
    p2[mov, 1] += rng.uniform(6, 15, mov.sum())       # 60 points also move vertically = scene motion
    F, _ = cv2.findFundamentalMat(p1[~mov], p2[~mov], cv2.FM_RANSAC, 1.0, 0.999)
    s = sampson(F.astype(np.float64), p1, p2)
    s_static, s_move = float(s[~mov].mean()), float(s[mov].mean())
    m2 = (s_static < 1.0) and (s_move > 3.0) and (s_move > 3 * s_static)
    print(f"SELFTEST camera: Sampson static(+parallax)={s_static:.2f} (<1) moving={s_move:.2f} (>3) -> {'ok' if m2 else 'FAIL'}")
    ok = m1 and m2
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} — features reproduce the flow; camera split separates parallax(static) from scene motion.")
    return 0 if ok else 1


# ----------------------------- RAFT + DINO (need torch) -----------------------------
def load_raft(device, small):
    from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
    w = (Raft_Small_Weights if small else Raft_Large_Weights).DEFAULT
    m = (raft_small if small else raft_large)(weights=w, progress=False).to(device).eval()
    return m, w.transforms()


def raft_flow(model, tf, a_bgr, b_bgr, device):
    """Dense flow a->b at full res. a,b:[H,W,3] uint8 BGR. Returns [H,W,2] (u=dx,v=dy)."""
    import torch, cv2
    def prep(im):
        return torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None]
    ta, tb = tf(prep(a_bgr), prep(b_bgr))
    H, W = ta.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8
    ta = torch.nn.functional.pad(ta, (0, pw, 0, ph), mode='replicate')
    tb = torch.nn.functional.pad(tb, (0, pw, 0, ph), mode='replicate')
    with torch.inference_mode():
        fl = model(ta.to(device), tb.to(device))[-1]
    return fl[0, :, :H, :W].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def fb_confidence(f_fwd, f_bwd, alpha=0.05, beta=1.0):
    import cv2
    H, W = f_fwd.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    bw = cv2.remap(f_bwd, uu + f_fwd[..., 0], vv + f_fwd[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(f_fwd + bw, axis=-1)
    mag = np.linalg.norm(f_fwd, axis=-1) + np.linalg.norm(bw, axis=-1)
    return err < (alpha * mag + beta), err


def dino_grid_for(rgb_path, args, hub_state):
    if args.dino_dir:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        cands = glob.glob(os.path.join(args.dino_dir, stem + '*')) or glob.glob(os.path.join(args.dino_dir, args.dino_glob))
        if cands:
            return np.load(sorted(cands)[0]).astype(np.float32)
    import torch, cv2
    if hub_state.get('model') is None:
        print("  (no baked DINO -> loading torch.hub dinov2_vits14_reg on the fly)")
        hub_state['model'] = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg').to(hub_state['device']).eval()
    m = hub_state['model']; dev = hub_state['device']
    im = cv2.cvtColor(cv2.imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = im.shape[:2]; gh, gw = (H // 14) * 14, (W // 14) * 14
    im = cv2.resize(im, (gw, gh))
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(dev)
    with torch.inference_mode():
        tok = m.forward_features(t)['x_norm_patchtokens'][0].cpu().numpy()
    return tok.reshape(gh // 14, gw // 14, -1).astype(np.float32)


def kmeans_labels(grid, K, seed=0, init=None):
    """k-means on the DINO grid. WARM-START with previous centroids (init) so cluster IDs stay
    consistent across frames -> the groups don't flash colours."""
    from sklearn.cluster import KMeans
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    use_init = init if (init is not None and len(init) == K) else 'k-means++'
    km = KMeans(K, n_init=(1 if isinstance(use_init, np.ndarray) else 4), random_state=seed, init=use_init).fit(X)
    return km.labels_.reshape(gh, gw).astype(np.uint8), km.cluster_centers_.astype(np.float32)


# ----------------------------- viz -----------------------------
def flow_color(flow):
    import cv2
    h, w = flow.shape[:2]; hsv = np.zeros((h, w, 3), np.uint8); hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def label_color(lab, K):
    return PALETTE[np.clip(lab, 0, min(K, len(PALETTE)) - 1)]


def heat(scalar, vmax):
    import cv2
    x = np.clip(scalar / max(vmax, 1e-6), 0, 1)
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def draw_arrows(frame, vecs, cent, scale=4.0):
    import cv2
    out = frame.copy()
    for k in range(len(vecs)):
        if np.isnan(vecs[k, 0]):
            continue
        x, y = cent[k]; dx, dy = vecs[k] * scale
        cv2.arrowedLine(out, (int(x), int(y)), (int(x + dx), int(y + dy)), (255, 255, 255), 2, tipLength=0.3)
        cv2.putText(out, f"{k}:{np.linalg.norm(vecs[k]):.1f}", (int(x) + 3, int(y) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _lbl(img, text):
    import cv2
    img = img.copy(); cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--rgb_dir'); ap.add_argument('--rgb_glob', default='*l.png')
    ap.add_argument('--dino_dir', default=''); ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--out_dir', default='output/flowprobe')
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--max_frames', type=int, default=80)
    ap.add_argument('--raft_small', action='store_true')
    ap.add_argument('--downscale', type=float, default=1.0)
    ap.add_argument('--no_fb', action='store_true')
    ap.add_argument('--ransac_thresh', type=float, default=1.0, help='F-matrix RANSAC px tolerance')
    ap.add_argument('--fps', type=int, default=8)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())

    import cv2, torch
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    assert len(R) >= 2, f'need >=2 frames (got {len(R)} from {args.rgb_dir}/{args.rgb_glob})'
    idx = list(range(0, len(R) - args.stride, args.stride))[:args.max_frames]
    print(f"[flowprobe] {len(R)} frames | {len(idx)} pairs (stride {args.stride}) | device {dev} | "
          f"RAFT-{'small' if args.raft_small else 'large'} | K={args.n_groups} | FB={'off' if args.no_fb else 'on'}")

    model, tf = load_raft(dev, args.raft_small)
    hub = {'model': None, 'device': dev}
    vw = None; prev_cent = None
    agg = {'mag': [], 'conf': [], 'epe_reg': [], 'epe_glob': [], 'epe_patch': [], 'coh': [],
           'resid': [], 'resid_hi': [], 'deform_frac': []}
    table = None; rows = []

    for n, i in enumerate(idx):
        a = cv2.imread(R[i]); b = cv2.imread(R[i + args.stride])
        if args.downscale != 1.0:
            a = cv2.resize(a, None, fx=args.downscale, fy=args.downscale)
            b = cv2.resize(b, None, fx=args.downscale, fy=args.downscale)
        H, W = a.shape[:2]
        flow = raft_flow(model, tf, a, b, dev)
        conf = np.ones((H, W), bool) if args.no_fb else fb_confidence(flow, raft_flow(model, tf, b, a, dev))[0]

        grid = dino_grid_for(R[i], args, hub); gh, gw = grid.shape[:2]
        lab_s, prev_cent = kmeans_labels(grid, args.n_groups, init=prev_cent)
        lab = cv2.resize(lab_s, (W, H), interpolation=cv2.INTER_NEAREST)

        vecs, coh, npx, ncf, cent = region_vectors(flow, lab, conf, args.n_groups)
        e_reg = epe(flow, region_field(flow.shape, lab, vecs), conf)
        gvec = np.median(flow[conf].reshape(-1, 2), axis=0) if conf.any() else np.zeros(2)
        e_glob = epe(flow, np.broadcast_to(gvec, flow.shape), conf)
        e_patch = epe(flow, patch_field(flow, gh, gw), conf)

        resid, ok = camera_split(flow, conf, args.ransac_thresh)            # CAMERA vs SCENE split
        thr = deform_threshold(resid, conf, args.ransac_thresh)
        deform = (resid > thr) & conf
        # per-region residual (median over confident px) + deform flag
        rres = np.full(args.n_groups, np.nan, np.float32)
        for k in range(args.n_groups):
            mc = (lab == k) & conf
            if mc.sum() >= 10:
                rres[k] = float(np.median(resid[mc]))

        mag = float(np.linalg.norm(flow, axis=-1).mean()); cf = float(conf.mean())
        agg['mag'].append(mag); agg['conf'].append(cf); agg['epe_reg'].append(e_reg)
        agg['epe_glob'].append(e_glob); agg['epe_patch'].append(e_patch); agg['coh'].append(float(np.nanmean(coh)))
        agg['resid'].append(float(np.mean(resid[conf])) if conf.any() else np.nan)
        agg['resid_hi'].append(float(np.percentile(resid[conf], 90)) if conf.any() else np.nan)
        agg['deform_frac'].append(float(deform.sum()) / max(conf.sum(), 1))
        # per-pair time series: cam_mag = |median flow| (dominant rigid = camera proxy) vs resid (scene)
        rows.append((i, i + args.stride, round(mag, 3), round(float(np.linalg.norm(gvec)), 3),
                     round(agg['resid'][-1], 3), round(agg['deform_frac'][-1], 4)))
        if n == len(idx) // 2:
            table = (i, vecs.copy(), coh.copy(), rres.copy(), thr, npx.copy())

        # ---- viz: 2x4 ----
        fc = flow_color(flow)
        grp = cv2.addWeighted(a, 0.45, label_color(lab, args.n_groups), 0.55, 0)
        arr = draw_arrows(a, vecs, cent)
        rmap = heat(resid, max(thr * 3, 2.0))
        route = a.copy(); route[deform] = (0, 0, 255)
        route = cv2.addWeighted(a, 0.5, route, 0.5, 0)
        rfield = flow_color(region_field(flow.shape, lab, vecs))
        confv = cv2.cvtColor((conf * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panels = [_lbl(a, f"frame {i}"), _lbl(fc, "dense flow"), _lbl(confv, f"confident {cf:.0%}"),
                  _lbl(grp, f"DINO groups K={args.n_groups}"), _lbl(arr, "per-feature flow"),
                  _lbl(rmap, f"residual (deform)  thr={thr:.1f}"), _lbl(route, f"routing  deform {agg['deform_frac'][-1]:.0%}"),
                  _lbl(rfield, f"region-avg flow  EPE {e_reg:.1f}")]
        canvas = np.vstack([np.hstack(panels[:4]), np.hstack(panels[4:8])])
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(args.out_dir, 'feature_flow.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
        if n % 10 == 0 or n == len(idx) - 1:
            print(f"  pair {n:3d} (f{i:04d}->{i+args.stride:04d}) mag={mag:5.2f} conf={cf:4.0%} "
                  f"EPE reg={e_reg:.2f} glob={e_glob:.2f} | resid mean={agg['resid'][-1]:.2f} p90={agg['resid_hi'][-1]:.2f} "
                  f"deform={agg['deform_frac'][-1]:.0%}")
    if vw is not None:
        vw.release()

    def mean(x): return float(np.nanmean(x)) if x else float('nan')
    eR, eG, eP = mean(agg['epe_reg']), mean(agg['epe_glob']), mean(agg['epe_patch'])
    explained = 1.0 - eR / eG if eG > 1e-6 else float('nan')
    print(f"\n=== AGGREGATE ({len(idx)} pairs) ===")
    print(f"  dense flow magnitude   : {mean(agg['mag']):.2f} px/frame")
    print(f"  confident fraction (FB): {mean(agg['conf']):.0%}")
    print(f"  region_EPE / global / patch : {eR:.2f} / {eG:.2f} / {eP:.2f} px   (explained {explained:+.2f})")
    print(f"  mean per-region coherence   : {mean(agg['coh']):.2f}  (motion-gated; ->1 = each feature moves as a unit)")
    print(f"  --- CAMERA vs SCENE split ---")
    print(f"  Sampson residual mean / p90 : {mean(agg['resid']):.2f} / {mean(agg['resid_hi']):.2f} px  (0 = camera-consistent)")
    print(f"  deforming fraction          : {mean(agg['deform_frac']):.0%}  (scene motion -> mapping; rest -> tracking)")
    if table is not None:
        i, vecs, coh, rres, thr, npx = table
        print(f"\n=== per-region @ frame {i} (deform thr={thr:.1f}) ===\n  reg   npx    |flow|px  coherence  residual  route")
        for k in range(len(vecs)):
            mg = np.linalg.norm(vecs[k]) if not np.isnan(vecs[k, 0]) else np.nan
            rt = 'MAP(deform)' if (not np.isnan(rres[k]) and rres[k] > thr) else 'track(static)'
            print(f"  {k:3d} {npx[k]:7d}   {mg:6.2f}    {coh[k]:+.2f}     {rres[k]:6.2f}   {rt}")
    import json, csv
    json.dump({k: mean(v) for k, v in agg.items()} | {'region_EPE': eR, 'global_EPE': eG, 'patch_EPE': eP, 'explained': explained},
              open(os.path.join(args.out_dir, 'feature_flow_metrics.json'), 'w'), indent=2)
    with open(os.path.join(args.out_dir, 'feature_flow_pairs.csv'), 'w', newline='') as fp:   # per-pair time series for GT-timing
        w = csv.writer(fp); w.writerow(['frame_a', 'frame_b', 'flow_mag', 'cam_mag', 'resid_mean', 'deform_frac']); w.writerows(rows)
    print(f"\nvideo  -> {os.path.join(args.out_dir, 'feature_flow.mp4')}")
    print(f"pairs  -> {os.path.join(args.out_dir, 'feature_flow_pairs.csv')}")
    print(f"metrics-> {os.path.join(args.out_dir, 'feature_flow_metrics.json')}")


if __name__ == '__main__':
    main()
