"""
Test 2 (visualization 3): tool-mask Δx vs frame.

Reads the dx_hook NPZ dump + per-frame semantic_instance masks.  Computes
mean ||Δx|| INSIDE the tool mask (dilated) vs OUTSIDE.  Plots both lines
across the snippet.

Test 2's central claim: if the deformation field is absorbing tool motion,
||Δx|| inside the tool mask should be LARGE and should track tool kinematic
motion magnitude.  Outside the mask (background + tissue), ||Δx|| should be
much smaller.

LIMITATION: dx_hook samples canonical 3D points along rays, but we need to
know which of those points project to which 2D pixel.  Approximated here:
re-project sample points back to 2D image coords using the frame's c2w +
intrinsics; bucket by tool-mask membership of the projected pixel.

Usage:
  python diagnosis/phase1/test2_tool_mask_dx.py \
    --dx_dir diagnosis/report/dx_dump_C1_001 \
    --config configs/CRCD/c1_001_paperfaith_lrfix.yaml \
    --semantic_dir F:/Datasets/CRCD-Published/C_1/snippet_001/semantic_instance \
    --out_csv diagnosis/report/test2_C1_001.csv \
    --out_fig diagnosis/report/test2_C1_001.png \
    --name C_1/001
"""

import argparse
import os
import sys
import glob
import json
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)


def project_to_pixel(points_world, c2w, fx, fy, cx, cy, H, W):
    """Project Nx3 world points to (u, v) pixel coords via c2w (camera-to-world)."""
    # world -> camera: c2w[:3,:3]^T @ (p - t)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    p_cam = (points_world - t) @ R  # equivalent to R^T @ (p - t)^T transposed
    # perspective project
    z = p_cam[:, 2]
    u = fx * p_cam[:, 0] / z + cx
    v = fy * p_cam[:, 1] / z + cy
    valid = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return u, v, valid


def dilate_mask(mask, kernel_size=15):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_dir', type=str, required=True)
    parser.add_argument('--config', type=str, required=True, help='To get intrinsics + H/W')
    parser.add_argument('--semantic_dir', type=str, required=True,
                        help='Dir of semantic_instance/frame_*.png (uint8 class IDs)')
    parser.add_argument('--semantic_pattern', type=str, default='*.png')
    parser.add_argument('--tool_class', type=int, default=3, help='CRCD tool class ID')
    parser.add_argument('--dilate_kernel', type=int, default=15)
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--out_fig', type=str, required=True)
    parser.add_argument('--name', type=str, default='snippet')
    args = parser.parse_args()

    from config import load_config
    cfg = load_config(args.config)
    fx = cfg['cam']['fx']; fy = cfg['cam']['fy']
    cx = cfg['cam']['cx']; cy = cfg['cam']['cy']
    H = cfg['cam']['H']; W = cfg['cam']['W']

    # Load NPZ dump
    files = sorted(glob.glob(os.path.join(args.dx_dir, 'frame_*.npz')))
    sem_files = sorted(glob.glob(os.path.join(args.semantic_dir, args.semantic_pattern)))
    n = min(len(files), len(sem_files))
    print(f'Test 2 — tool-mask Δx on {args.name}: {n} frames')

    dx_in_mean = np.zeros(n)
    dx_out_mean = np.zeros(n)
    dx_in_count = np.zeros(n, dtype=int)
    dx_out_count = np.zeros(n, dtype=int)
    tool_pixel_frac = np.zeros(n)

    for i in range(n):
        data = np.load(files[i])
        x_can = data['x_canonical']  # (R, S, 3)
        dx = data['delta_x']
        c2w = data['c2w']
        # Re-project all samples to 2D pixel
        pts = x_can.reshape(-1, 3)
        u, v, valid = project_to_pixel(pts, c2w, fx, fy, cx, cy, H, W)
        u_int = np.clip(u.round().astype(int), 0, W - 1)
        v_int = np.clip(v.round().astype(int), 0, H - 1)
        # Load tool mask + dilate
        sem = cv2.imread(sem_files[i], cv2.IMREAD_UNCHANGED)
        if sem is None:
            print(f'  WARN: failed to load {sem_files[i]}'); continue
        if sem.shape != (H, W):
            sem = cv2.resize(sem, (W, H), interpolation=cv2.INTER_NEAREST)
        tool_mask = (sem == args.tool_class)
        tool_mask_d = dilate_mask(tool_mask, args.dilate_kernel)
        tool_pixel_frac[i] = float(tool_mask.sum()) / tool_mask.size

        # Sample membership
        in_tool = tool_mask_d[v_int, u_int] & valid.numpy() if isinstance(valid, torch.Tensor) else tool_mask_d[v_int, u_int] & valid
        out_tool = (~tool_mask_d[v_int, u_int]) & (valid.numpy() if isinstance(valid, torch.Tensor) else valid)

        dx_norms = np.linalg.norm(dx.reshape(-1, 3), axis=-1) * 1000  # mm
        if in_tool.sum() > 0:
            dx_in_mean[i] = float(dx_norms[in_tool].mean())
            dx_in_count[i] = int(in_tool.sum())
        if out_tool.sum() > 0:
            dx_out_mean[i] = float(dx_norms[out_tool].mean())
            dx_out_count[i] = int(out_tool.sum())

    import pandas as pd
    df = pd.DataFrame({
        'frame': np.arange(n),
        'dx_in_mean_mm': dx_in_mean,
        'dx_out_mean_mm': dx_out_mean,
        'dx_in_count': dx_in_count,
        'dx_out_count': dx_out_count,
        'tool_pixel_frac': tool_pixel_frac,
    })
    df.to_csv(args.out_csv, index=False)
    print(f'Saved CSV: {args.out_csv}')

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ax = axes[0]
    ax.plot(dx_in_mean, label='inside tool mask (dilated)', color='red', linewidth=2)
    ax.plot(dx_out_mean, label='outside tool mask', color='steelblue', linewidth=2)
    ax.set_ylabel('mean ||Δx|| (mm)')
    ax.set_title(f'Test 2: deformation field activity inside vs outside tool mask  ({args.name})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(tool_pixel_frac * 100, color='gray', linewidth=1.5, label='tool pixel %')
    ax.set_xlabel('frame')
    ax.set_ylabel('tool pixel fraction (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'TEST 2 — Tool-cancellation: ||Δx|| inside vs outside tool mask', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=120, bbox_inches='tight')
    print(f'Saved figure: {args.out_fig}')

    # Verdict
    ratio = dx_in_mean.mean() / max(dx_out_mean.mean(), 1e-9)
    print(f'\n=== TEST 2 VERDICT ===')
    print(f'  mean ||Δx|| inside  tool mask: {dx_in_mean.mean():.4f} mm')
    print(f'  mean ||Δx|| outside tool mask: {dx_out_mean.mean():.4f} mm')
    print(f'  ratio inside/outside: {ratio:.2f}')
    if ratio > 1.5:
        print(f'  >>> TOOL CANCELLATION SIGNAL <<<')
        print(f'  Δx is larger INSIDE the tool mask — deformation field is doing work there.')
        print(f'  Tool is rigid; ||Δx|| there should physically be ~zero — suggests gauge absorption.')
    elif ratio > 0.8 and ratio < 1.2:
        print(f'  Δx is roughly EQUAL inside and outside — global smearing pattern.')
        print(f'  Consistent with gauge-absorption applied uniformly across the scene.')
    else:
        print(f'  Δx is SMALLER inside tool mask than outside — field is NOT cancelling tool.')
        print(f'  Tool-cancellation hypothesis not supported on this snippet.')


if __name__ == '__main__':
    main()
