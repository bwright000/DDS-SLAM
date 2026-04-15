"""Measure the actual 3D extent of a SemSuP scene from its MoGe depth + intrinsics.

Tells us what bbox values would correctly contain the scene, so we can
test the bbox-shrink hypothesis without guessing bounds that either
miss the scene (too tight) or leave it attenuated (too loose).

Usage:
    python Addons/inspect_scene_extent.py --datadir data/trail3

Reads:
    {datadir}/rgb/*-left_depth.npy   (MoGe depth in meters * 8.0 per Super.yaml scale)
    (intrinsics hardcoded to Super.yaml: fx=fy=768.99, cx=292.89, cy=291.61, HxW=480x640)

Prints:
    - Depth percentiles in meters
    - Back-projected X/Y/Z extent in meters
    - Recommended bbox (p1-p99 + 20% margin)
    - Current configured bbox (for comparison)

No SLAM, no GPU, no training. ~30 seconds for 151 frames.
"""

import argparse
import glob
import os

import cv2
import numpy as np


# Super.yaml hardcoded intrinsics (rectified left camera)
FX = 768.98551924
FY = 768.98551924
CX = 292.8861567
CY = 291.61479526
H = 480
W = 640

# Super.yaml depth scale (for converting raw NPY values to meters)
PNG_DEPTH_SCALE = 8.0


def backproject(depth_m, fx, fy, cx, cy):
    """depth_m (HxW float32, meters) -> Nx3 float32 world points in camera frame."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    valid = (depth_m > 0) & np.isfinite(depth_m)
    z = depth_m[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def pctiles(arr, qs=(1, 5, 25, 50, 75, 95, 99)):
    return {q: float(np.percentile(arr, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True,
                    help="e.g. data/trail3 (expects rgb/*-left_depth.npy)")
    ap.add_argument("--subsample", type=int, default=20,
                    help="analyse every Nth frame (default 20 -> ~7-8 frames on 151-frame trail)")
    ap.add_argument("--margin", type=float, default=0.2,
                    help="extra bbox margin as fraction of extent (default 0.2 = 20%)")
    args = ap.parse_args()

    npy_paths = sorted(glob.glob(os.path.join(args.datadir, "rgb", "*-left_depth.npy")))
    if not npy_paths:
        print(f"No *-left_depth.npy files in {args.datadir}/rgb/")
        return
    print(f"Found {len(npy_paths)} depth files; sampling every {args.subsample}")
    npy_paths = npy_paths[::args.subsample]

    all_pts = []
    depth_vals = []
    for p in npy_paths:
        raw = np.load(p).astype(np.float32)
        if raw.ndim == 3:
            raw = raw[..., 0]
        depth_m = raw / PNG_DEPTH_SCALE  # convert to meters
        depth_vals.append(depth_m[(depth_m > 0) & np.isfinite(depth_m)])
        pts = backproject(depth_m, FX, FY, CX, CY)
        all_pts.append(pts)

    depth_vals = np.concatenate(depth_vals)
    all_pts = np.concatenate(all_pts, axis=0)

    print("\n=== Depth (meters) ===")
    dp = pctiles(depth_vals)
    for q, v in dp.items():
        print(f"  p{q:02d}: {v:.3f} m")
    print(f"  min: {depth_vals.min():.3f}  max: {depth_vals.max():.3f}  N: {len(depth_vals):,}")

    print("\n=== 3D scene extent (camera frame, meters) ===")
    for i, name in enumerate(["X", "Y", "Z"]):
        axis = all_pts[:, i]
        ap_q = pctiles(axis)
        print(f"  {name}: p1={ap_q[1]:+.3f}  p99={ap_q[99]:+.3f}  extent={ap_q[99]-ap_q[1]:.3f} m")

    print("\n=== Recommended bbox (p1..p99 + margin) ===")
    for i, name in enumerate(["X", "Y", "Z"]):
        axis = all_pts[:, i]
        lo, hi = np.percentile(axis, [1, 99])
        extent = hi - lo
        lo -= args.margin * extent
        hi += args.margin * extent
        print(f"  {name}: [{lo:+.3f}, {hi:+.3f}]  (extent {hi-lo:.3f} m)")

    print("\n=== Current configured bbox (trail3.yaml via Super.yaml) ===")
    print("  X: [-0.7, +0.7]  (extent 1.4 m)")
    print("  Y: [-0.7, +0.7]  (extent 1.4 m)")
    print("  Z: [+0.7, +1.2]  (extent 0.5 m)")

    print("\n=== far-plane vs scene z ===")
    z = all_pts[:, 2]
    print(f"  Super.yaml far=5.0; scene z p99 = {np.percentile(z, 99):.3f}")
    print(f"  → recommend far ~= {np.percentile(z, 99)*1.5:.2f} (1.5x p99)")

    print("\n=== Ratios (oversize factor at current bbox) ===")
    for i, name in enumerate(["X", "Y", "Z"]):
        axis = all_pts[:, i]
        lo, hi = np.percentile(axis, [1, 99])
        scene_extent = hi - lo
        configured = 1.4 if name != "Z" else 0.5
        print(f"  {name}: scene {scene_extent:.3f} m / configured {configured} m = 1/{configured/scene_extent:.1f} (gradient attenuation factor)")


if __name__ == "__main__":
    main()
