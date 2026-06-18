#!/usr/bin/env python3
"""Derive a CRCD snippet's mapping.bound / marching_cubes_bound (Arm-4 item A4-0.5).

WHY THIS EXISTS - the SNI-SLAM (NeRF / tri-plane) CRCD configs need a per-snippet
`mapping.bound` + `marching_cubes_bound` (the scene bounding box). 00_COMMON sec4.0 route (b)
PINS the recipe so the result is reproducible and NOT a free-hand guess (which sec1 forbids):

  From the snippet's frame-0 VALID metric depth (metres):
    - Z (depth) axis : [ P1(d) - M , P99(d) + M ]
    - X, Y axes      : back-project the valid pixels with the rectified intrinsics, then
                       [ P1 - M , P99 + M ] per axis
  Constants are FIXED: M = 0.02 m (2 cm margin), P1/P99 = 1st/99th percentile.
  marching_cubes_bound = bound.

Inputs are the **metric-scaled** delegated depth (A4-0.4: stereo-calibrated, sc_factor=1.0),
so depth_png / depth_scale is already metres - no sc_factor here. Back-projection uses the
OpenCV pinhole convention x=(u-cx)d/fx, y=(v-cy)d/fy, z=d, matching how the committed c1_001
bound was shaped (X,Y centred near 0; Z = forward depth, positive).

!! The pinned recipe is FRAME-0 only (frame-0 camera == world origin). For a snippet with
non-trivial camera motion this under-covers the reconstructed volume; pass --frame_indices to
union several frames (camera-frame union - only valid while motion is small, which the CRCD
snippets are: sub-SNR). A loud warning prints if >1 frame is unioned without pose transform.

!!! Re-derive ALL CRCD bounds from the NEW metric depth - the committed c1_001/c2_001 SNI
bounds were derived from the OLD up-to-scale MoGe depth (~8x off in real metres) and are STALE.

Not for SGS-SLAM / SemGauss-SLAM (3DGS - no mapping.bound). RUNS where the depth is staged.

Usage:
  python Addons/preprocess/derive_crcd_bounds.py \
      --depth_dir data/CRCD/E3_005/depth --calib data/CRCD/E3_005/rectified_calib.txt \
      --depth_scale 10000 --name E3_005 --out configs_bounds/E3_005_bound.yaml
"""
import argparse
import glob
import os

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    print(f"[derive_crcd_bounds] need cv2 (colab_setup installs opencv): {e}")
    raise

M_DEFAULT = 0.02   # metres, PINNED
PCT = 1.0          # P1/P99, PINNED


def read_calib(path):
    """rectified_calib.txt = 'key value' per line (fx fy cx cy baseline_m ... width height)."""
    kv = {}
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            try:
                kv[parts[0]] = float(parts[1])
            except ValueError:
                pass
    for k in ('fx', 'fy', 'cx', 'cy'):
        if k not in kv:
            raise RuntimeError(f"{path}: missing intrinsic '{k}' (have {sorted(kv)})")
    return kv


def load_depth_m(path, scale):
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"cannot read depth {path}")
    arr = arr.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr / float(scale)


def backproject(depth_m, fx, fy, cx, cy, min_d, max_d):
    H, W = depth_m.shape
    us, vs = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    valid = (depth_m > min_d) & (depth_m < max_d) & np.isfinite(depth_m)
    d = depth_m[valid]
    u = us[valid]; v = vs[valid]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    return np.stack([x, y, d], axis=1)  # [N,3] camera frame (= world at frame 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--depth_dir', required=True, help='metric depth dir (data/CRCD/<NAME>/depth)')
    ap.add_argument('--calib', required=True, help='rectified_calib.txt (fx fy cx cy ...)')
    ap.add_argument('--depth_scale', type=float, default=10000.0, help='png value / scale = metres')
    ap.add_argument('--name', default='?', help='snippet name (label only)')
    ap.add_argument('--frame_indices', default='0', help='comma list of sorted-depth indices (default frame 0)')
    ap.add_argument('--margin', type=float, default=M_DEFAULT, help='axis margin, metres (PINNED 0.02)')
    ap.add_argument('--pct', type=float, default=PCT, help='percentile (PINNED 1 -> P1/P99)')
    ap.add_argument('--min_depth_m', type=float, default=0.001)
    ap.add_argument('--max_depth_m', type=float, default=10.0)
    ap.add_argument('--out', default=None, help='write a YAML snippet here')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.depth_dir, '*.png')))
    if not files:
        raise RuntimeError(f"no depth PNGs in {a.depth_dir}")
    idxs = [int(s) for s in a.frame_indices.split(',') if s.strip() != '']
    if len(idxs) > 1:
        print(f"[derive_crcd_bounds] WARN unioning {len(idxs)} frames in the CAMERA frame "
              f"(no pose transform) - valid only for small motion. Recipe default is frame-0 only.")
    kv = read_calib(a.calib)
    fx, fy, cx, cy = kv['fx'], kv['fy'], kv['cx'], kv['cy']

    pts = []
    for i in idxs:
        if i >= len(files):
            raise RuntimeError(f"frame index {i} >= {len(files)} depth files")
        depth = load_depth_m(files[i], a.depth_scale)
        # scale intrinsics if calib resolution differs from the depth map
        if 'width' in kv and 'height' in kv and (depth.shape[1], depth.shape[0]) != (int(kv['width']), int(kv['height'])):
            sx = depth.shape[1] / kv['width']; sy = depth.shape[0] / kv['height']
            fx, fy, cx, cy = kv['fx'] * sx, kv['fy'] * sy, kv['cx'] * sx, kv['cy'] * sy
            print(f"[derive_crcd_bounds] scaled intrinsics to depth res {depth.shape[1]}x{depth.shape[0]}")
        pts.append(backproject(depth, fx, fy, cx, cy, a.min_depth_m, a.max_depth_m))
    P = np.concatenate(pts, axis=0)
    if P.shape[0] < 100:
        raise RuntimeError(f"only {P.shape[0]} valid points - depth_scale wrong or depth empty?")

    lo, hi = a.pct, 100.0 - a.pct
    def axis_bound(col):
        return [round(float(np.percentile(col, lo) - a.margin), 5),
                round(float(np.percentile(col, hi) + a.margin), 5)]
    bound = [axis_bound(P[:, 0]), axis_bound(P[:, 1]), axis_bound(P[:, 2])]

    print(f"\n# {a.name}: bound from {P.shape[0]} pts, frames={idxs}, M={a.margin}m, "
          f"P{a.pct:.0f}/P{100 - a.pct:.0f}, metric depth (sc_factor=1)")
    print("mapping:")
    print(f"  bound: {bound}")
    print(f"  marching_cubes_bound: {bound}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as fh:
            fh.write(f"# {a.name} bound (frames={idxs}, M={a.margin}m, P{a.pct:.0f}/P{100 - a.pct:.0f}, "
                     f"metric depth sc_factor=1; recipe 00_COMMON sec4.0(b))\n")
            fh.write("mapping:\n")
            fh.write(f"  bound: {bound}\n")
            fh.write(f"  marching_cubes_bound: {bound}\n")
        print(f"[derive_crcd_bounds] wrote {a.out}")


if __name__ == '__main__':
    main()
