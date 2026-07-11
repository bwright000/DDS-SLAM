#!/usr/bin/env python3
"""Pose-aware SNI-SLAM scene bound.

THE BUG THIS FIXES: the staged bound.yaml is the camera-relative depth RANGE (~0.1 m, z~[0.01,0.11]),
but SNI-SLAM runs in the ABSOLUTE GT-pose frame (frame 0 anchored to the GT c2w, camera z~0.74 on CRCD).
The tissue therefore lands at world-z ~0.64 -- ~6x OUTSIDE the staged bound -- so normalize_3d_coordinate
clamps every depth-pinned sample to the bound edge, the SDF forms no coherent surface (marching_cubes
fails on EVERY snippet), the tracker gets no depth constraint (drifts + freezes t_z), and every render
is empty. Verified 2026-07-11 from the run's [cfg] bounds vs the est frame-0 camera z.

THE FIX: recompute the bound by unprojecting each frame's depth through the SAME poses + axis flip SNI
actually uses, so the box covers the tissue where SNI places it. This replicates SNI exactly:
  - depth = png / png_depth_scale         (datasets.py:91;  scale=1)
  - pose flip: c2w[:3,1]*=-1; c2w[:3,2]*=-1   (datasets.py:196-197, OpenCV->OpenGL)
  - ray dir  = [(u-cx)/fx, -(v-cy)/fy, -1]    (common.py:91)
  - rays_d   = R_flip @ dir                   (common.py:95)
  - point    = t + depth * rays_d             (get_samples: rays_o + z_vals*rays_d, z_vals~gt_depth)

Writes bound.yaml with mapping.bound + mapping.marching_cubes_bound (the keys mk_sni_cfg reads).

Usage:
  python sni_poseaware_bound.py --data_dir <SNI data/CRCD/NAME (traj.txt + depth/)> \
      --calib <rectified_calib.txt> --out <bound.yaml> --png_depth_scale 10000
"""
import argparse
import glob
import os
import re

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"[poseaware-bound] need cv2: {e}")


def read_calib(p):
    kv = {}
    for line in open(p):
        q = line.split()
        if len(q) >= 2 and q[0] in ('fx', 'fy', 'cx', 'cy'):
            kv[q[0]] = float(q[1])
    for k in ('fx', 'fy', 'cx', 'cy'):
        assert k in kv, f"{p}: missing {k}"
    return kv


def _nkey(p):
    m = re.findall(r'\d+', os.path.basename(p))
    return int(m[-1]) if m else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', required=True, help='SNI layout dir: traj.txt + depth/*.png')
    ap.add_argument('--calib', required=True, help='rectified_calib.txt (fx fy cx cy)')
    ap.add_argument('--out', required=True, help='bound.yaml to (over)write')
    ap.add_argument('--png_depth_scale', type=float, default=10000.0)
    ap.add_argument('--frame_stride', type=int, default=0, help='0 = auto (~60 frames sampled)')
    ap.add_argument('--pixel_stride', type=int, default=12)
    ap.add_argument('--margin_frac', type=float, default=0.10, help='fractional extent margin/axis')
    ap.add_argument('--margin_min', type=float, default=0.03, help='absolute floor margin/axis (m)')
    ap.add_argument('--trim', type=float, default=0.5, help='percentile trimmed each tail (depth outliers)')
    ap.add_argument('--depth_max', type=float, default=5.0, help='ignore depths above this (m; MoGe outliers)')
    a = ap.parse_args()

    K = read_calib(a.calib)
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']

    tp = os.path.join(a.data_dir, 'traj.txt')
    traj = [np.array(list(map(float, l.split()))).reshape(4, 4)
            for l in open(tp) if l.strip()]
    deps = sorted(glob.glob(os.path.join(a.data_dir, 'depth', '*.png')), key=_nkey)
    n = min(len(traj), len(deps))
    assert n > 0, f"no frames (traj={len(traj)} depth={len(deps)})"
    fs = a.frame_stride or max(1, n // 60)

    cam_z = [traj[i][2, 3] for i in range(n)]
    pts = []
    used = 0
    for i in range(0, n, fs):
        raw = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        d = raw.astype(np.float32) / a.png_depth_scale       # metres
        H, W = d.shape[:2]
        c2w = traj[i].astype(np.float64).copy()
        c2w[:3, 1] *= -1                                      # SNI flip -> OpenGL
        c2w[:3, 2] *= -1
        R = c2w[:3, :3]; t = c2w[:3, 3]
        vs = np.arange(0, H, a.pixel_stride)
        us = np.arange(0, W, a.pixel_stride)
        uu, vv = np.meshgrid(us, vs)
        dd = d[vv, uu].astype(np.float64)
        m = (dd > 1e-4) & (dd < a.depth_max) & np.isfinite(dd)
        if not m.any():
            continue
        uu, vv, dd = uu[m], vv[m], dd[m]
        dir_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy,
                            -np.ones_like(uu, dtype=np.float64)], -1)   # (M,3)
        rays_d = dir_cam @ R.T                                # R_flip @ dir  (common.py:95)
        p = t[None, :] + dd[:, None] * rays_d                 # world points
        pts.append(p)
        used += 1
    assert pts, "no valid depth points unprojected"
    P = np.concatenate(pts, 0)

    lo = np.percentile(P, a.trim, axis=0)
    hi = np.percentile(P, 100 - a.trim, axis=0)
    ext = hi - lo
    pad = np.maximum(a.margin_frac * ext, a.margin_min)
    lo -= pad; hi += pad
    bound = [[round(float(lo[k]), 4), round(float(hi[k]), 4)] for k in range(3)]

    with open(a.out, 'w', encoding='utf-8') as f:
        f.write("# pose-aware bound (Addons/eval/sni_poseaware_bound.py): depth unprojected through the\n")
        f.write("# GT poses+flip SNI actually uses -> covers the tissue in SNI's absolute-pose frame.\n")
        f.write("mapping:\n")
        f.write(f"  bound: {bound}\n")
        f.write(f"  marching_cubes_bound: {bound}\n")

    print(f"[poseaware-bound] {n} frames ({used} sampled, stride {fs}), {len(P):,} points -> {a.out}")
    print(f"  camera-position z-range (poses) : {min(cam_z):.3f} .. {max(cam_z):.3f}")
    print(f"  NEW bound: x{bound[0]}  y{bound[1]}  z{bound[2]}")


if __name__ == '__main__':
    main()
