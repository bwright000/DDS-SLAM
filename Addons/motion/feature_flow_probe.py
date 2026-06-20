#!/usr/bin/env python3
"""Feature optical-flow probe — measure per-DINO-feature motion and CHECK it matches the real flow.

THE STEP (memory project_combine_routing_model_20260619, "flow-as-sensor"): the per-pixel sigma^2
hedge is blind to deformation (appearance-preserving: liver slides, liver pixels -> low residual).
The fix is to MEASURE motion with optical flow and POOL it per DINO feature-region:
  1. RAFT dense per-pixel flow (frame t -> t+1).
  2. group pixels by DINO feature (k-means on the baked grid, label-free) = "the features".
  3. robust-average the flow over CONFIDENT pixels in each group -> one per-feature motion vector.
  (4. [NEXT step, NOT here] subtract camera-predicted flow -> residual = deformation; the pose+depth
      camera model already exists in Addons/motion/bake_motion_residual.py.)

Answers ONE question with BOTH diagnostic sets (the standing two-set rule):
  VISUAL  - a video: frame | dense flow | DINO grouping | per-feature flow ARROWS | region-avg flow.
            Eyeball whether the feature arrows match the motion you see.
  NUMERIC - region_EPE = EPE(region-averaged flow, dense flow) over confident px (does the per-feature
            vector reproduce the real flow?), BRACKETED by global_EPE (1 vector = camera-ish) and
            patch_EPE (finest, per-DINO-patch), an explained-fraction, per-region coherence, and a
            per-region motion table. Confidence = forward-backward flow consistency.

Standalone, torch>=2 env (torchvision RAFT + optional torch.hub DINO). No SLAM, GPU optional (CPU ok, slow).
  python Addons/motion/feature_flow_probe.py \
    --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png' \
    --dino_dir data/CRCD/C1_001/dino_reg --dino_glob '*_dino.npy' \
    --out_dir output/flowprobe_c1 --n_groups 6 --stride 2 --max_frames 60
  python Addons/motion/feature_flow_probe.py --selftest      # metric math, no RAFT/DINO/data
"""
import os, sys, glob, argparse
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# fixed BGR palette for up to 8 groups (distinct, colour-blind-ish)
PALETTE = np.array([[60, 60, 60], [0, 200, 0], [0, 0, 230], [230, 160, 0], [200, 0, 200],
                    [0, 200, 200], [120, 80, 200], [80, 160, 80]], np.uint8)


# ----------------------------- metric math (selftest-covered) -----------------------------
def region_vectors(flow, lab, conf, K, min_px=10):
    """Per-region ROBUST (median) flow over CONFIDENT pixels + coherence/centroid/counts."""
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
        d = np.linalg.norm(fl - med, axis=1).mean()     # spread about the median
        coh[k] = 1.0 - d / (np.linalg.norm(med) + 1e-3)  # 1 = moves as a unit, <=0 = incoherent
    return vecs, coh, npx, ncf, cent


def region_field(flow_shape, lab, vecs):
    """Replace each pixel's flow with its REGION's vector (NaN regions -> 0)."""
    field = np.zeros(flow_shape, np.float32)
    for k in range(len(vecs)):
        if not np.isnan(vecs[k, 0]):
            field[lab == k] = vecs[k]
    return field


def epe(a, b, conf):
    """Mean endpoint error between two flow fields over confident pixels."""
    e = np.linalg.norm(a - b, axis=-1)
    return float(e[conf].mean()) if conf.any() else float('nan')


def patch_field(flow, gh, gw):
    """Per-DINO-patch mean flow (finest feature granularity) broadcast back to pixels."""
    import cv2
    H, W = flow.shape[:2]
    small = cv2.resize(flow, (gw, gh), interpolation=cv2.INTER_AREA)     # block-mean per patch cell
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def run_selftest():
    """Validate the metric math on synthetic flow — no RAFT/DINO/data."""
    rng = np.random.RandomState(0); H = W = 90; K = 3
    lab = np.zeros((H, W), int); lab[:, 30:60] = 1; lab[:, 60:] = 2          # 3 vertical strips
    true_vec = np.array([[5, 0], [0, 4], [-3, -3]], np.float32)
    flow = np.zeros((H, W, 2), np.float32)
    for k in range(K):
        flow[lab == k] = true_vec[k]
    flow += rng.randn(H, W, 2) * 0.3                                          # small noise
    conf = np.ones((H, W), bool)
    vecs, coh, npx, ncf, cent = region_vectors(flow, lab, conf, K)
    e_reg = epe(flow, region_field(flow.shape, lab, vecs), conf)
    gvec = np.median(flow.reshape(-1, 2), axis=0)
    e_glob = epe(flow, np.broadcast_to(gvec, flow.shape), conf)
    # mis-aligned grouping (one strip split wrong) must be WORSE
    badlab = (np.arange(W) % 3)[None, :].repeat(H, 0)
    bv, *_ = region_vectors(flow, badlab, conf, 3)
    e_bad = epe(flow, region_field(flow.shape, badlab, bv), conf)
    print(f"SELFTEST region_EPE={e_reg:.3f} (aligned, ~noise) | global_EPE={e_glob:.3f} (>>region) | "
          f"mis-aligned_EPE={e_bad:.3f} (>>region)")
    print(f"  coherence={np.round(coh,2)} (all ~1) | recovered vecs ok={np.allclose(vecs, true_vec, atol=0.4)}")
    ok = (e_reg < 0.6) and (e_glob > 2.0) and (e_bad > 2.0) and np.allclose(vecs, true_vec, atol=0.4) and (coh.min() > 0.7)
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} — aligned grouping reproduces the flow; global/mis-aligned do not.")
    return 0 if ok else 1


# ----------------------------- RAFT + DINO (need torch) -----------------------------
def load_raft(device, small):
    import torch
    from torchvision.models.optical_flow import raft_small, raft_large, Raft_Small_Weights, Raft_Large_Weights
    w = (Raft_Small_Weights if small else Raft_Large_Weights).DEFAULT
    m = (raft_small if small else raft_large)(weights=w, progress=False).to(device).eval()
    return m, w.transforms()


def raft_flow(model, tf, a_bgr, b_bgr, device):
    """Dense flow a->b at full res. a,b: [H,W,3] uint8 BGR. Returns [H,W,2] (u=dx, v=dy)."""
    import torch, cv2
    def prep(im):
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1)[None]                  # uint8 [1,3,H,W]
    ta, tb = tf(prep(a_bgr), prep(b_bgr))                                     # -> float, normalised
    H, W = ta.shape[-2:]; ph, pw = (-H) % 8, (-W) % 8                         # RAFT needs /8
    ta = torch.nn.functional.pad(ta, (0, pw, 0, ph), mode='replicate')
    tb = torch.nn.functional.pad(tb, (0, pw, 0, ph), mode='replicate')
    with torch.inference_mode():
        fl = model(ta.to(device), tb.to(device))[-1]                         # [1,2,H8,W8] final refine
    return fl[0, :, :H, :W].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def fb_confidence(f_fwd, f_bwd, alpha=0.05, beta=1.0):
    """Forward-backward consistency mask: warp f_bwd along f_fwd; |f_fwd+f_bwd_warp| small => trust."""
    import cv2
    H, W = f_fwd.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    bw = cv2.remap(f_bwd, uu + f_fwd[..., 0], vv + f_fwd[..., 1],
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.linalg.norm(f_fwd + bw, axis=-1)
    mag = np.linalg.norm(f_fwd, axis=-1) + np.linalg.norm(bw, axis=-1)
    return err < (alpha * mag + beta), err


def dino_grid_for(rgb_path, args, hub_state):
    """Return the [gh,gw,C] DINO grid for a frame: baked .npy if present, else torch.hub on the fly."""
    if args.dino_dir:
        stem = os.path.splitext(os.path.basename(rgb_path))[0]
        cands = glob.glob(os.path.join(args.dino_dir, stem + '*')) or \
                glob.glob(os.path.join(args.dino_dir, args.dino_glob))
        if cands:
            return np.load(sorted(cands)[0]).astype(np.float32)
    # fallback: compute on the fly (dinov2_vits14_reg, patch-14)
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
        tok = m.forward_features(t)['x_norm_patchtokens'][0].cpu().numpy()   # [gh/14*gw/14, C]
    return tok.reshape(gh // 14, gw // 14, -1).astype(np.float32)


def kmeans_labels(grid, K, seed=0):
    from sklearn.cluster import KMeans
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C); X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return KMeans(K, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)


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


def draw_arrows(frame, vecs, cent, scale=4.0):
    import cv2
    out = frame.copy()
    for k in range(len(vecs)):
        if np.isnan(vecs[k, 0]):
            continue
        x, y = cent[k]; dx, dy = vecs[k] * scale
        p0 = (int(x), int(y)); p1 = (int(x + dx), int(y + dy))
        cv2.arrowedLine(out, p0, p1, (255, 255, 255), 2, tipLength=0.3)
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
    ap.add_argument('--n_groups', type=int, default=6)
    ap.add_argument('--stride', type=int, default=2, help='use every Nth frame pair')
    ap.add_argument('--max_frames', type=int, default=60)
    ap.add_argument('--raft_small', action='store_true', help='RAFT-small (faster, less accurate)')
    ap.add_argument('--downscale', type=float, default=1.0, help='resize frames by this factor (speed)')
    ap.add_argument('--no_fb', action='store_true', help='skip forward-backward confidence (1 RAFT call/pair)')
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
    H0, W0 = cv2.imread(R[0]).shape[:2]
    vw = None
    agg = {'mag': [], 'conf': [], 'epe_reg': [], 'epe_glob': [], 'epe_patch': [], 'coh': []}
    table = None

    for n, i in enumerate(idx):
        a = cv2.imread(R[i]); b = cv2.imread(R[i + args.stride])
        if args.downscale != 1.0:
            a = cv2.resize(a, None, fx=args.downscale, fy=args.downscale)
            b = cv2.resize(b, None, fx=args.downscale, fy=args.downscale)
        H, W = a.shape[:2]
        flow = raft_flow(model, tf, a, b, dev)
        if args.no_fb:
            conf = np.ones((H, W), bool)
        else:
            flow_b = raft_flow(model, tf, b, a, dev)
            conf, _ = fb_confidence(flow, flow_b)

        grid = dino_grid_for(R[i], args, hub)
        gh, gw = grid.shape[:2]
        lab_s = kmeans_labels(grid, args.n_groups)
        lab = cv2.resize(lab_s, (W, H), interpolation=cv2.INTER_NEAREST)

        vecs, coh, npx, ncf, cent = region_vectors(flow, lab, conf, args.n_groups)
        e_reg = epe(flow, region_field(flow.shape, lab, vecs), conf)
        gvec = np.median(flow[conf].reshape(-1, 2), axis=0) if conf.any() else np.zeros(2)
        e_glob = epe(flow, np.broadcast_to(gvec, flow.shape), conf)
        e_patch = epe(flow, patch_field(flow, gh, gw), conf)
        mag = float(np.linalg.norm(flow, axis=-1).mean()); cf = float(conf.mean())
        agg['mag'].append(mag); agg['conf'].append(cf); agg['epe_reg'].append(e_reg)
        agg['epe_glob'].append(e_glob); agg['epe_patch'].append(e_patch)
        agg['coh'].append(float(np.nanmean(coh)))

        if n == len(idx) // 2:                                              # per-region table @ mid frame
            table = (i, vecs.copy(), coh.copy(), npx.copy(), ncf.copy())

        # ---- viz video ----
        fc = flow_color(flow)
        grp = cv2.addWeighted(a, 0.45, label_color(lab, args.n_groups), 0.55, 0)
        arr = draw_arrows(a, vecs, cent)
        rfield = flow_color(region_field(flow.shape, lab, vecs))
        confv = cv2.cvtColor((conf * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panels = [_lbl(a, f"frame {i}"), _lbl(fc, "dense flow"), _lbl(confv, f"confident {cf:.0%}"),
                  _lbl(grp, f"DINO groups K={args.n_groups}"), _lbl(arr, "per-feature flow"),
                  _lbl(rfield, f"region-avg flow  EPE {e_reg:.2f}px")]
        row1 = np.hstack(panels[:3]); row2 = np.hstack(panels[3:]); canvas = np.vstack([row1, row2])
        if vw is None:
            vw = cv2.VideoWriter(os.path.join(args.out_dir, 'feature_flow.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (canvas.shape[1], canvas.shape[0]))
        vw.write(canvas)
        if n % 10 == 0 or n == len(idx) - 1:
            print(f"  pair {n:3d} (f{i:04d}->{i+args.stride:04d}) mag={mag:5.2f}px conf={cf:4.0%} "
                  f"EPE reg={e_reg:.2f} glob={e_glob:.2f} patch={e_patch:.2f}")
    if vw is not None:
        vw.release()

    # ----------------------------- summary -----------------------------
    def mean(x): return float(np.nanmean(x)) if x else float('nan')
    eR, eG, eP = mean(agg['epe_reg']), mean(agg['epe_glob']), mean(agg['epe_patch'])
    explained = 1.0 - eR / eG if eG > 1e-6 else float('nan')
    print(f"\n=== AGGREGATE ({len(idx)} pairs) ===")
    print(f"  dense flow magnitude   : {mean(agg['mag']):.2f} px/frame")
    print(f"  confident fraction (FB): {mean(agg['conf']):.0%}")
    print(f"  region_EPE  (per-feature flow vs dense): {eR:.2f} px   <- THE 'does feature-flow match real flow' number")
    print(f"  global_EPE  (1 vector, camera-ish)     : {eG:.2f} px")
    print(f"  patch_EPE   (finest, per-DINO-patch)   : {eP:.2f} px")
    print(f"  explained fraction 1-region/global     : {explained:+.2f}  (->1 = features capture the motion structure)")
    print(f"  mean per-region coherence              : {mean(agg['coh']):.2f}  (->1 = each feature moves as a unit)")
    if table is not None:
        i, vecs, coh, npx, ncf = table
        print(f"\n=== per-region motion @ frame {i} ===\n  reg   npx   conf%   |flow|px   dx,dy        coherence")
        for k in range(len(vecs)):
            v = vecs[k]; mg = np.linalg.norm(v)
            vt = f"({v[0]:+5.1f},{v[1]:+5.1f})" if not np.isnan(v[0]) else "   (n/a)   "
            print(f"  {k:3d} {npx[k]:6d} {100*ncf[k]/max(npx[k],1):5.0f}  {mg:7.2f}   {vt}  {coh[k]:+.2f}")
    import json
    json.dump({k: mean(v) for k, v in agg.items()} | {'region_EPE': eR, 'global_EPE': eG,
              'patch_EPE': eP, 'explained': explained},
              open(os.path.join(args.out_dir, 'feature_flow_metrics.json'), 'w'), indent=2)
    print(f"\nvideo  -> {os.path.join(args.out_dir, 'feature_flow.mp4')}")
    print(f"metrics-> {os.path.join(args.out_dir, 'feature_flow_metrics.json')}")


if __name__ == '__main__':
    main()
