#!/usr/bin/env python3
"""Convert an SGS-SLAM params.npz camera trajectory -> est_c2w_data.txt (Arm-4 A4-1.3).

WHY — the DDS-SLAM harness (Addons/eval/sim3_ate.py:load_est, Addons/viz/generate_video.py)
reads the estimated trajectory as a plain text file, one pose per line, row-major, translation
at indices [3,7,11] (12 floats = 3x4 c2w, or 16 = 4x4). SGS-SLAM does NOT write that: it stores
per-frame poses inside params.npz as `cam_unnorm_rots` (1,4,N) [quaternion w,x,y,z, UNNORMALIZED]
and `cam_trans` (1,3,N), which are **w2c relative to the first camera** (world = first cam frame,
SplaTAM convention). This converts: normalize quat -> R -> w2c [R|t] -> invert to c2w -> write.

Verified against the repo: utils/common_utils.py save_params writes cam_unnorm_rots/cam_trans;
utils/slam_external.py build_rotation uses q=[w,x,y,z]; utils/eval_helpers.py decodes the same.

Usage:
  python Addons/eval/sgsslam_npz_to_est.py --npz <OUT>/params.npz --out <RUN>/est_c2w_data.txt
"""
import argparse
import os

import numpy as np


def quat_to_R(q):
    """q = [w, x, y, z] (will be normalized). Returns 3x3 rotation (SplaTAM build_rotation convention)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True, help='SGS-SLAM params.npz (has cam_unnorm_rots + cam_trans)')
    ap.add_argument('--out', required=True, help='output est_c2w_data.txt (12 floats/line, c2w row-major)')
    a = ap.parse_args()

    d = np.load(a.npz)
    if 'cam_unnorm_rots' not in d or 'cam_trans' not in d:
        raise RuntimeError(f"{a.npz}: missing cam_unnorm_rots/cam_trans (keys: {list(d.keys())})")
    rots = np.asarray(d['cam_unnorm_rots'])   # (1,4,N)
    trans = np.asarray(d['cam_trans'])        # (1,3,N)
    n = rots.shape[-1]
    lines = []
    for i in range(n):
        R = quat_to_R(rots[0, :, i])
        t = np.asarray(trans[0, :, i], dtype=np.float64)
        w2c = np.eye(4); w2c[:3, :3] = R; w2c[:3, 3] = t
        c2w = np.linalg.inv(w2c)              # est_c2w_data.txt is camera-to-world
        lines.append(' '.join(f'{v:.7f}' for v in c2w[:3, :4].reshape(-1)))   # 12 floats, row-major
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, 'w').write('\n'.join(lines) + '\n')
    print(f"[npz->est] wrote {n} poses (12-float c2w/line, trans@[3,7,11]) -> {a.out}")


if __name__ == '__main__':
    main()
