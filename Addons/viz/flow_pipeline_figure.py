#!/usr/bin/env python3
"""Flow-pipeline figure for DDS-SLAM (meeting): DINOv2 features -> optical flow -> residual flow.

Visualises, for a few CRCD frames, the EXACT signals the model computes (reuses
Addons/motion/flow_track: load_raft / load_dino / _raft_flow / dino_grid), so the figure is faithful
to what `flow_agree` (tracking gate) and `best_deformiters` (map-iter route) actually see.

Per frame t (causal reference = t - stride), one row, columns:
  1. RGB              -- the current frame (context)
  2. DINOv2 features  -- vits14-reg patch grid, PCA->RGB (the WHAT-is-it grouping)
  3. Optical flow     -- RAFT ref->cur, Middlebury colour wheel (RAW motion = camera + scene)
  4. Residual flow    -- per-pixel homography reprojection residual = motion AFTER the dominant
                         camera motion is removed (camera/parallax -> ~0, deformation/tool -> high).
                         THIS is what passes into the map (pixel-mode route, as in best_deformiters).
  5. Map route        -- the soft weight clip((residual-deadband)/soft_scale): the actual per-pixel
                         up-weight that scales the current-frame map optimisation.

Runs on GPU (Colab) or CPU (slow but fine for a handful of frames). RAFT = torchvision, DINOv2 =
torch.hub (auto-download).

Usage (Colab box, rect or raw frames):
  python Addons/viz/flow_pipeline_figure.py --frames_dir data/CRCD/C1_001/video_frames \
    --indices 60 180 300 --stride 8 --out figs/flow_pipeline_c1_001.png
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from Addons.motion.flow_track import load_raft, load_dino, _raft_flow, dino_grid  # noqa: E402


def flow_to_rgb(flow):
    """Middlebury colour wheel: hue = direction, value = magnitude (per-frame normalised)."""
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def dino_pca_rgb(grid, out_hw):
    """[gh,gw,C] DINO patch grid -> [H,W,3] PCA-to-RGB (top-3 components, per-channel normalised)."""
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C).astype(np.float32)
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = X @ Vt[:3].T                      # [gh*gw, 3]
    proj = proj.reshape(gh, gw, 3)
    for c in range(3):                       # robust per-channel min-max (2-98 pct)
        lo, hi = np.percentile(proj[..., c], [2, 98])
        proj[..., c] = np.clip((proj[..., c] - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.resize(proj, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_CUBIC)


def homography_residual(flow, ransac_thresh=1.0, deadband=3.0,
                        soft_scale=2.0, smooth=5, max_fit=4000):
    """Pixel-mode route (== flow_track.region_route mode='pixel'): fit a global homography (models
    camera rotation+zoom where the fundamental matrix is degenerate) and take the per-pixel
    reprojection residual. Returns (residual[H,W], route[H,W]) -- the residual that enters the map and
    the soft up-weight derived from it. Takes the precomputed RAFT flow (one RAFT call per frame)."""
    H, W = flow.shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    p1 = np.stack([uu, vv], -1).reshape(-1, 2)
    p2 = p1 + flow.reshape(-1, 2)
    idx = np.linspace(0, len(p1) - 1, min(max_fit, len(p1))).astype(np.int64)
    Hmat, _ = cv2.findHomography(p1[idx], p2[idx], cv2.RANSAC, ransac_thresh)
    if Hmat is None:
        return np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    p1h = np.concatenate([p1, np.ones((len(p1), 1), np.float32)], 1)
    proj = (Hmat.astype(np.float32) @ p1h.T).T
    proj = proj[:, :2] / (proj[:, 2:3] + 1e-9)
    rmap = np.linalg.norm(proj - p2, axis=1).reshape(H, W).astype(np.float32)
    rs = cv2.medianBlur(rmap, smooth if smooth in (3, 5) else 5)
    route = np.clip((rs - deadband) / max(soft_scale, 1e-6), 0.0, 1.0).astype(np.float32)
    route = cv2.GaussianBlur(route, (0, 0), 1.5)
    return rmap, route


def load_frame(frames_dir, i):
    files = sorted(glob.glob(os.path.join(frames_dir, '*.png')) +
                   glob.glob(os.path.join(frames_dir, '*.jpg')))
    files = [f for f in files if 'right' not in os.path.basename(f).lower()]   # left/mono only
    if i >= len(files):
        raise IndexError(f"index {i} >= {len(files)} frames in {frames_dir}")
    return cv2.imread(files[i]), os.path.basename(files[i]), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--indices', type=int, nargs='+', required=True, help='current-frame indices t')
    ap.add_argument('--stride', type=int, default=8, help='causal reference = t - stride (= ref_stride)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--deadband', type=float, default=3.0)
    ap.add_argument('--soft_scale', type=float, default=2.0)
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()

    import torch
    device = torch.device('cpu' if (a.cpu or not torch.cuda.is_available()) else 'cuda')
    print('device:', device)
    raft, tf = load_raft(device)
    dino = load_dino(device)

    cols = ['RGB (frame t)', 'DINOv2 features (PCA)', 'Optical flow (RAFT, t-%d -> t)' % a.stride,
            'Residual flow -> map', 'Map route (up-weight)']
    rows = len(a.indices)
    fig, axes = plt.subplots(rows, 5, figsize=(20, 4.0 * rows))
    if rows == 1:
        axes = axes[None, :]

    for r, t in enumerate(a.indices):
        cur_bgr, cur_name, n = load_frame(a.frames_dir, t)
        ref_bgr, _, _ = load_frame(a.frames_dir, max(0, t - a.stride))
        rgb = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        flow = _raft_flow(raft, tf, ref_bgr, cur_bgr, device)
        grid = dino_grid(cur_bgr, dino, device)
        dino_rgb = dino_pca_rgb(grid, (H, W))
        resid, route = homography_residual(flow, deadband=a.deadband, soft_scale=a.soft_scale)
        flow_rgb = flow_to_rgb(flow)
        movefrac = float((route > 0.05).mean())

        panels = [rgb, dino_rgb, flow_rgb, None, None]
        for c in range(3):
            axes[r, c].imshow(panels[c])
        im3 = axes[r, 3].imshow(resid, cmap='turbo', vmax=np.percentile(resid, 99))
        fig.colorbar(im3, ax=axes[r, 3], fraction=0.046, pad=0.04)
        im4 = axes[r, 4].imshow(route, cmap='magma', vmin=0, vmax=1)
        fig.colorbar(im4, ax=axes[r, 4], fraction=0.046, pad=0.04)

        for c in range(5):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(f"{cur_name}\n(idx {t}/{n}, move-frac {movefrac:.2f})", fontsize=9)

    fig.suptitle('DDS-SLAM motion pipeline: DINOv2 -> optical flow -> residual (camera removed) -> map route',
                 fontsize=14, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=130, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
