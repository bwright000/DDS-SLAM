#!/usr/bin/env python3
"""Per-snippet scene bound for the fresh SNI-CRCD bench, in the frame-0-RELATIVE frame.

Uses the SAME transform as the CRCD dataloader (frame-0-relative c2w + col-1,2 flip) so the
box lands exactly where SNI places the tissue. Unprojects sub-sampled depth through those poses,
percentile-trims, pads, and writes a per-snippet config that inherits crcd.yaml.

  point = t + depth * (R_flip @ dir),  dir = [(u-cx)/fx, -(v-cy)/fy, -1]   (common.py:91,95)

Usage:
  python compute_bound_relative.py --data_dir <CRCD snippet> --out configs/CRCD/<name>.yaml \
      --name <name> --input_folder data/CRCD/<name> --timesteps 360
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def read_calib(p):
    kv = {}
    for ln in open(p):
        q = ln.split(':') if ':' in ln else ln.split()
        if len(q) >= 2 and q[0].strip() in ('fx', 'fy', 'cx', 'cy'):
            kv[q[0].strip()] = float(q[1])
    return kv


def _fid(p):
    m = re.findall(r'\d+', os.path.basename(p))
    return int(m[0]) if m else -1


def rel_poses(gt_path, n):
    rows = [[float(x) for x in ln.split()[1:8]]
            for ln in open(gt_path)
            if ln.strip() and not ln.startswith('#') and len(ln.split()) >= 8]
    absT = []
    for tx, ty, tz, qx, qy, qz, qw in rows[:n]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        absT.append(T)
    inv0 = np.linalg.inv(absT[0])
    out = []
    for T in absT:
        c = inv0 @ T
        c[:3, 1] *= -1
        c[:3, 2] *= -1
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--input_folder', required=True, help='SNI-side data path written into the config')
    ap.add_argument('--timesteps', type=int, default=0)
    ap.add_argument('--calib', default=None)
    ap.add_argument('--png_depth_scale', type=float, default=10000.0)
    ap.add_argument('--pixel_stride', type=int, default=12)
    ap.add_argument('--frame_stride', type=int, default=0)
    ap.add_argument('--pad_frac', type=float, default=0.15)
    ap.add_argument('--pad_min', type=float, default=0.01)
    ap.add_argument('--trim', type=float, default=0.5)
    ap.add_argument('--depth_max', type=float, default=5.0)
    a = ap.parse_args()

    calib = a.calib or os.path.join(a.data_dir, 'rectified_calib.txt')
    K = read_calib(calib)
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']

    deps = {_fid(p): p for p in glob.glob(os.path.join(a.data_dir, 'depth', '*l.png'))}
    ids = sorted(deps)
    if a.timesteps:
        ids = ids[:a.timesteps]
    poses = rel_poses(os.path.join(a.data_dir, 'groundtruth.txt'), len(ids))
    n = min(len(ids), len(poses))
    fs = a.frame_stride or max(1, n // 60)

    pts = []
    for k in range(0, n, fs):
        raw = cv2.imread(deps[ids[k]], cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        d = raw.astype(np.float32) / a.png_depth_scale
        H, W = d.shape[:2]
        R, t = poses[k][:3, :3], poses[k][:3, 3]
        vs = np.arange(0, H, a.pixel_stride)
        us = np.arange(0, W, a.pixel_stride)
        uu, vv = np.meshgrid(us, vs)
        dd = d[vv, uu].astype(np.float64)
        m = (dd > 1e-4) & (dd < a.depth_max) & np.isfinite(dd)
        uu, vv, dd = uu[m], vv[m], dd[m]
        dir_cam = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -np.ones_like(uu, float)], -1)
        pts.append(t[None] + dd[:, None] * (dir_cam @ R.T))
    assert pts, "no depth points unprojected"
    P = np.concatenate(pts, 0)

    lo = np.percentile(P, a.trim, 0)
    hi = np.percentile(P, 100 - a.trim, 0)
    pad = np.maximum(a.pad_frac * (hi - lo), a.pad_min)
    lo -= pad
    hi += pad
    bound = [[round(float(lo[i]), 4), round(float(hi[i]), 4)] for i in range(3)]

    out_name = os.path.splitext(f"output/CRCD/{a.name}")[0]
    with open(a.out, 'w', encoding='utf-8') as f:
        f.write(f"# {a.name}: per-snippet override for the fresh SNI-CRCD bench (auto-generated bound).\n")
        f.write("inherit_from: configs/CRCD/crcd.yaml\n")
        if a.timesteps:
            f.write(f"# n_img derived from data ({n} frames)\n")
        f.write("mapping:\n")
        f.write(f"  bound: {bound}\n")
        f.write(f"  marching_cubes_bound: {bound}\n")
        f.write("data:\n")
        f.write(f"  input_folder: {a.input_folder}\n")
        f.write(f"  output: {out_name}\n")

    print(f"[bound] {a.name}: {n} frames -> {a.out}")
    print(f"  bound x{bound[0]} y{bound[1]} z{bound[2]}  (relative frame, near origin)")


if __name__ == '__main__':
    main()
