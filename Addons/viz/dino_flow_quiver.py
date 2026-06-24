#!/usr/bin/env python3
"""Raw RGB + per-DINO-region optical-flow vectors -- the flow_agree camera/scene test, made visual.

For each frame: segment into DINOv2 k-means regions (exactly as the agreement gate does), and draw ONE
arrow per region = that region's MEDIAN optical-flow vector, over the raw RGB. Arrow colour = the gate's
verdict: GREEN = the region agrees with the single rigid camera motion (median Sampson residual <=
deadband), RED = it disagrees (scene deformation / tool). The title carries the disagree count -- the
exact quantity the gate thresholds. Faint tint = the DINO regions themselves.

Reuses Addons/motion/flow_track (load_raft / load_dino / _raft_flow / dino_grid / _sampson) so it is
faithful to what flow_agree sees. Runs CPU or GPU.

Usage:
  python Addons/viz/dino_flow_quiver.py --frames_dir data/CRCD/C1_001/video_frames \
    --indices 60 180 300 --stride 8 --out figs/dino_flow_quiver.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from Addons.motion.flow_track import load_raft, load_dino, _raft_flow, dino_grid, _sampson  # noqa: E402


def load_frame(frames_dir, i):
    files = sorted(glob.glob(os.path.join(frames_dir, '*.png')) +
                   glob.glob(os.path.join(frames_dir, '*.jpg')))
    files = [f for f in files if 'right' not in os.path.basename(f).lower()]
    if i >= len(files):
        raise IndexError(f"index {i} >= {len(files)} frames in {frames_dir}")
    return cv2.imread(files[i]), os.path.basename(files[i]), len(files)


def region_flow(ref_bgr, cur_bgr, raft, tf, dino, device, n_groups=12, deadband=3.0,
                ransac_thresh=1.0, min_px=50, seed=0):
    """Per DINO region: (centroid_x, centroid_y, median_flow_u, median_flow_v, median_sampson).
    Plus the region-label map for the faint tint."""
    flow = _raft_flow(raft, tf, ref_bgr, cur_bgr, device)
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2)
    p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(4000, len(p1))).astype(np.int64)
    F, _ = cv2.findFundamentalMat(p1[idx], p2[idx], cv2.FM_RANSAC, ransac_thresh, 0.999)
    resid = (_sampson(F.astype(np.float64), p1, p2).reshape(H, W)
             if (F is not None and F.shape == (3, 3)) else np.zeros((H, W), np.float32))
    grid = dino_grid(cur_bgr, dino, device)
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    lab = KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw).astype(np.uint8)
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    regions = []
    for k in range(n_groups):
        m = lab == k
        if m.sum() < min_px:
            continue
        ys, xs = np.where(m)
        fv = np.median(flow[m], axis=0)
        regions.append((float(xs.mean()), float(ys.mean()), float(fv[0]), float(fv[1]),
                        float(np.median(resid[m]))))
    return lab, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--indices', type=int, nargs='+', required=True)
    ap.add_argument('--stride', type=int, default=8, help='causal reference = t - stride')
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--deadband', type=float, default=3.0)
    ap.add_argument('--target_len', type=float, default=70.0, help='median arrow length in px (auto-scaled)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()

    import torch
    device = torch.device('cpu' if (a.cpu or not torch.cuda.is_available()) else 'cuda')
    print('device:', device)
    raft, tf = load_raft(device)
    dino = load_dino(device)

    n = len(a.indices)
    fig, axes = plt.subplots(1, n, figsize=(7.0 * n, 5.2))
    if n == 1:
        axes = [axes]
    for ax, t in zip(axes, a.indices):
        cur_bgr, name, N = load_frame(a.frames_dir, t)
        ref_bgr, _, _ = load_frame(a.frames_dir, max(0, t - a.stride))
        rgb = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2RGB)
        lab, regions = region_flow(ref_bgr, cur_bgr, raft, tf, dino, device,
                                   n_groups=a.n_groups, deadband=a.deadband)
        # auto-scale arrows so the MEDIAN region vector ~= target_len px (relative magnitudes preserved)
        mags = [np.hypot(u, v) for _, _, u, v, _ in regions]
        med = float(np.median(mags)) if mags else 1.0
        sc = a.target_len / (med + 1e-6)

        ax.imshow(rgb)
        ax.imshow(lab, cmap='tab20', alpha=0.18)        # faint DINO regions
        nd = 0
        for cx, cy, u, v, rmed in regions:
            agree = rmed <= a.deadband
            nd += 0 if agree else 1
            ax.arrow(cx, cy, u * sc, v * sc, color=('lime' if agree else 'red'),
                     width=2.2, head_width=14, head_length=12, length_includes_head=True,
                     zorder=5, ec='black', lw=0.4)
        ax.set_title(f"{name} (idx {t}/{N})  --  {nd}/{len(regions)} regions disagree", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0].legend([Line2D([0], [0], color='lime', lw=4), Line2D([0], [0], color='red', lw=4)],
                   ['agrees with camera motion', 'disagrees (scene / tool)'],
                   loc='lower left', fontsize=8, framealpha=0.85)
    fig.suptitle('Raw RGB + DINOv2 regions and their optical-flow vectors  (the flow_agree camera/scene test)',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
