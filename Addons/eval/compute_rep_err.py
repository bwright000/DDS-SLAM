"""Paper-compatible Reprojection Error for Semantic-SuPer.

Methodology matches the Python-SuPer convention (Lin et al., UCSD-CVRA):
  * For each frame k in [1, T-1]:
      1. anchor each GT point's 3D position from frame-0 depth + (x,y)
      2. transform 3D anchor to frame-k camera coords via T_k_0 = inv(c2w_k) @ c2w_0
      3. project to 2D using fx, fy, cx, cy
      4. pixel-L2 vs ground-truth (x,y) at frame k
  * Aggregate: mean +/- std across all valid (frame, point) pairs.

GT format (trial_<N>_l_pts.npy in v2_data02):
  pickled dict with key 'gt' mapping str frame index ('000000', ...) ->
  ndarray of shape (P, 3) where columns are (x_pixel, y_pixel, valid_flag).

Usage:
  python Addons/eval/compute_rep_err.py \\
      --est_c2w results/.../est_c2w_data.txt \\
      --pts F:/Datasets/SemSup/v2_data02/v2_data/trial_3/rgb/trial_3_l_pts.npy \\
      --depth0 data/Super/trail_3/depth/ref/000000.npy \\
      --fx 768.98551924 --fy 768.98551924 --cx 292.8861567 --cy 291.61479526
"""
import argparse
import numpy as np


def load_c2w(path):
    """Load est_c2w_data.txt: each line = 12 floats = 3x4 SE(3) top rows."""
    mats = []
    with open(path) as f:
        for line in f:
            v = [float(x) for x in line.split()]
            if len(v) != 12:
                continue
            M = np.eye(4)
            M[:3, :] = np.array(v).reshape(3, 4)
            mats.append(M)
    return np.stack(mats)  # (T, 4, 4)


def load_pts(path):
    """Load trial_*_l_pts.npy. Returns dict frame_idx (int) -> (P, 3) array."""
    raw = np.load(path, allow_pickle=True).tolist()
    gt = raw['gt']
    return {int(k): np.asarray(v) for k, v in gt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est_c2w', required=True, help='path to est_c2w_data.txt')
    ap.add_argument('--pts',     required=True, help='path to trial_*_l_pts.npy')
    ap.add_argument('--depth0',  required=True,
                    help='path to frame-0 reference depth .npy (used to anchor 3D points)')
    ap.add_argument('--fx', type=float, default=768.98551924)
    ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567)
    ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--depth_scale', type=float, default=1.0,
                    help='if depth file is uint16 PNG-style scaled, divide by this. '
                         '.npy files are typically already in meters -> keep 1.0.')
    ap.add_argument('--output_csv', default=None, help='optional per-frame CSV')
    args = ap.parse_args()

    poses = load_c2w(args.est_c2w)
    gt    = load_pts(args.pts)
    depth0 = np.load(args.depth0).astype(np.float64).squeeze() / args.depth_scale
    H, W = depth0.shape
    T = len(poses)

    print(f"poses: {T}, gt frames: {len(gt)}, depth0 shape: {depth0.shape}")
    print(f"intrinsics: fx={args.fx} fy={args.fy} cx={args.cx} cy={args.cy}")

    # ---- anchor 3D points from frame-0 GT ----
    pts0 = gt[0]                                # (P, 3)
    us0, vs0, val0 = pts0[:, 0].astype(int), pts0[:, 1].astype(int), pts0[:, 2]
    # clamp to valid pixel range
    us0c = np.clip(us0, 0, W - 1)
    vs0c = np.clip(vs0, 0, H - 1)
    Z0 = depth0[vs0c, us0c]
    valid_anchor = (val0 == 1) & (Z0 > 0) & np.isfinite(Z0)
    X0 = (us0c - args.cx) * Z0 / args.fx
    Y0 = (vs0c - args.cy) * Z0 / args.fy
    P3d_frame0 = np.stack([X0, Y0, Z0, np.ones_like(Z0)], axis=0)  # (4, P)
    P = pts0.shape[0]
    n_anchor = int(valid_anchor.sum())
    print(f"\nanchored {n_anchor}/{P} points from frame-0 depth")

    # ---- per-frame reprojection ----
    c2w_0 = poses[0]
    per_frame_mean = np.full(T, np.nan)
    per_frame_n   = np.zeros(T, dtype=int)
    all_errs = []
    for k in range(T):
        if k not in gt:
            continue
        gt_k = gt[k]
        if gt_k.shape[0] != P:
            continue
        c2w_k = poses[k]
        T_k_0 = np.linalg.inv(c2w_k) @ c2w_0          # frame-0 -> frame-k camera
        P3d_k = T_k_0 @ P3d_frame0                    # (4, P)
        Xk, Yk, Zk = P3d_k[0], P3d_k[1], P3d_k[2]
        u_proj = args.fx * Xk / Zk + args.cx
        v_proj = args.fy * Yk / Zk + args.cy
        u_gt, v_gt, val_k = gt_k[:, 0], gt_k[:, 1], gt_k[:, 2]
        valid = valid_anchor & (val_k == 1) & (Zk > 0) & np.isfinite(u_proj) & np.isfinite(v_proj)
        err = np.sqrt((u_proj - u_gt) ** 2 + (v_proj - v_gt) ** 2)
        err = err[valid]
        if err.size > 0:
            per_frame_mean[k] = float(np.mean(err))
            per_frame_n[k]    = int(err.size)
            all_errs.extend(err.tolist())

    all_errs = np.array(all_errs)
    valid_frames = ~np.isnan(per_frame_mean)
    print(f"\n=== Reprojection Error ===")
    print(f"valid frames                : {int(valid_frames.sum())}/{T}")
    print(f"total (frame,point) samples : {len(all_errs)}")
    print(f"per-sample mean             : {np.mean(all_errs):.3f} px")
    print(f"per-sample std              : {np.std(all_errs):.3f} px")
    print(f"per-sample median           : {np.median(all_errs):.3f} px")
    print(f"per-frame mean (mean of means): {np.nanmean(per_frame_mean):.3f} px")
    print(f"per-frame std                 : {np.nanstd(per_frame_mean):.3f} px")
    print(f"per-frame range               : [{np.nanmin(per_frame_mean):.2f}, {np.nanmax(per_frame_mean):.2f}] px")

    # quick window comparison
    f30  = per_frame_mean[:30]
    f150 = per_frame_mean
    print(f"\nframe 1-30   mean : {np.nanmean(f30):.3f} px")
    print(f"frame 1-150  mean : {np.nanmean(f150):.3f} px")

    if args.output_csv:
        import csv
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['frame', 'rep_err_px_mean', 'n_valid'])
            for k in range(T):
                w.writerow([k, f"{per_frame_mean[k]:.4f}" if not np.isnan(per_frame_mean[k]) else '',
                            per_frame_n[k]])
        print(f"\nper-frame CSV: {args.output_csv}")


if __name__ == '__main__':
    main()
