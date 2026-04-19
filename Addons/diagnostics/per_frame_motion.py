"""Per-frame motion analyser for DDS-SLAM outputs.

Computes the same statistics that diagnosed the flat-loss-region bug:
- median, mean, 95/99th pct of |trans[i+1] - trans[i]| for estimated poses
- same for GT (if available)
- ratio estimated/GT (should be ~1 after Fix 1)

Works on either:
  (a) DDS-SLAM est_c2w_data.txt (each line = 12 values = first 3 rows of c2w)
  (b) Co-SLAM est_c2w.txt (each line = [frame_id, tx, ty, tz, qw, qx, qy, qz])

Usage:
    python Addons/per_frame_motion.py <est_poses.txt> [--gt <groundtruth.txt>]
"""

import argparse
import os
import sys
import numpy as np


def load_ddsslam_est(path):
    """DDS-SLAM format: 12 floats per line = first 3 rows of c2w flattened."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 12:
        raise ValueError(f"expected (N,12) for DDS-SLAM, got {data.shape}")
    return data[:, [3, 7, 11]]  # translation column of the 3x4


def load_coslam_est(path):
    """Co-SLAM format: [idx, tx, ty, tz, qw, qx, qy, qz] per line."""
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != 8:
        raise ValueError(f"expected (N,8) for Co-SLAM, got {data.shape}")
    return data[:, 1:4]


def load_gt(path):
    """TUM format: timestamp tx ty tz qx qy qz qw (comments with #)."""
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1:4]


def autodetect_and_load(path):
    """Try both formats and return the one that parses cleanly."""
    data = np.loadtxt(path)
    if data.ndim == 2 and data.shape[1] == 12:
        print(f"[detected DDS-SLAM format] {path}")
        return data[:, [3, 7, 11]]
    if data.ndim == 2 and data.shape[1] == 8:
        print(f"[detected Co-SLAM format] {path}")
        return data[:, 1:4]
    raise ValueError(f"unrecognised pose format, shape={data.shape}")


def stats(trans, label):
    deltas_m = np.linalg.norm(trans[1:] - trans[:-1], axis=1)
    deltas = deltas_m * 1000  # in mm
    print(f"\n--- {label} ---")
    print(f"  n frames: {len(trans)}")
    print(f"  trajectory extent (mm): "
          f"x={1000*(trans[:,0].max()-trans[:,0].min()):.2f} "
          f"y={1000*(trans[:,1].max()-trans[:,1].min()):.2f} "
          f"z={1000*(trans[:,2].max()-trans[:,2].min()):.2f}")
    print(f"  per-frame translation delta (mm):")
    print(f"    median = {np.median(deltas):.4f}")
    print(f"    mean   = {deltas.mean():.4f}")
    print(f"    95 pct = {np.percentile(deltas, 95):.4f}")
    print(f"    99 pct = {np.percentile(deltas, 99):.4f}")
    print(f"    max    = {deltas.max():.4f}")

    bins = [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    hist, _ = np.histogram(deltas, bins=bins)
    print(f"  histogram (bin_mm: count, pct):")
    for i in range(len(hist)):
        print(f"    {bins[i]:5.2f} - {bins[i+1]:5.2f}: {hist[i]:5d}  ({100*hist[i]/len(deltas):5.1f}%)")
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("est", help="path to est_c2w_data.txt or est_c2w.txt")
    ap.add_argument("--gt", default=None, help="optional groundtruth.txt for comparison")
    args = ap.parse_args()

    if not os.path.exists(args.est):
        print(f"est file not found: {args.est}")
        sys.exit(1)

    est_trans = autodetect_and_load(args.est)
    est_deltas = stats(est_trans, label=f"estimated poses: {args.est}")

    if args.gt:
        if not os.path.exists(args.gt):
            print(f"gt file not found: {args.gt}")
            sys.exit(1)
        gt_trans = load_gt(args.gt)
        gt_deltas = stats(gt_trans, label=f"ground truth: {args.gt}")
        print(f"\n--- ratio ---")
        print(f"  median: estimated / gt = {np.median(est_deltas) / max(np.median(gt_deltas), 1e-9):.2f}x")
        print(f"  mean  : estimated / gt = {est_deltas.mean() / max(gt_deltas.mean(), 1e-9):.2f}x")


if __name__ == "__main__":
    main()
