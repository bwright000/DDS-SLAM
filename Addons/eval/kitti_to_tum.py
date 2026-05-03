"""Convert DDS-SLAM est_c2w_data.txt (KITTI 12-float c2w rows) and a TUM-format
groundtruth.txt into a paired TUM-format pair suitable for evo_ape / evo_traj.

est_c2w_data.txt: one pose per line, 12 floats = first 3 rows of 4x4 c2w.
groundtruth.txt:  TUM `ts tx ty tz qx qy qz qw`, one row per source frame.

The dataset slice (e.g. last-4000 frames of an 8465-frame sequence) is set via
--gt_offset: index in GT that aligns with est line 0. For StereoMIS P2_1
last-4000, gt_offset = len(gt) - len(est) = 4465.
"""
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out_est', required=True)
    ap.add_argument('--out_gt', required=True)
    ap.add_argument('--gt_offset', type=int, default=None,
                    help='GT row that aligns with est row 0. Default: len(gt) - len(est).')
    args = ap.parse_args()

    est_rows = np.loadtxt(args.est)
    if est_rows.ndim == 1:
        est_rows = est_rows[None, :]
    N = est_rows.shape[0]

    with open(args.gt) as f:
        gt_body = [ln for ln in f.readlines() if ln.strip() and not ln.lstrip().startswith('#')]
    M = len(gt_body)

    offset = args.gt_offset if args.gt_offset is not None else max(0, M - N)
    if offset + N > M:
        raise SystemExit(f'GT too short: have {M}, need {offset + N}')

    print(f'est rows: {N}  gt rows: {M}  offset: {offset}  -> using gt[{offset}:{offset + N}]')

    with open(args.out_est, 'w') as fe, open(args.out_gt, 'w') as fg:
        for i in range(N):
            mat = est_rows[i].reshape(3, 4)
            t = mat[:, 3]
            q = R.from_matrix(mat[:, :3]).as_quat()  # [x, y, z, w]
            ts = float(i)
            fe.write(f'{ts:.6f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} '
                     f'{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n')
            parts = gt_body[offset + i].split()
            fg.write(f'{ts:.6f} ' + ' '.join(parts[1:]) + '\n')

    print(f'wrote {N} poses -> {args.out_est}')
    print(f'wrote {N} poses -> {args.out_gt}')


if __name__ == '__main__':
    main()
