"""
Reanalyse a debug_log.csv post-hoc.

The raw per-frame trans_err_m / rot_err_rad columns in debug_log.csv compare
est_c2w and gt_c2w without any trajectory alignment. That makes them
uninterpretable when:

  - the tracker operates in a y/z-flipped camera convention (dataset.py:219-220)
    while the GT is loaded in native TUM convention;
  - there is a systematic trajectory scale error (the est trajectory is
    N-times too large — we observed 2.6x on StereoMIS P2_1);
  - there is any net rotation between the est and GT frames.

This script reads an existing debug_log.csv, fits a single Umeyama similarity
between the full est and GT trajectories (two variants: R+t, and R+t+s), and
also computes frame-to-frame relative pose error (RPE). The enhanced CSV is
written next to the input with ``_aligned.csv`` appended; a short summary is
printed to stdout.

  python Addons/diagnostics/reanalyse_debug_csv.py \\
      --debug_dir <...>/demo/debug
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def _quat_to_R(qw, qx, qy, qz):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _R_to_quat(R):
    q = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
    return q[3], q[0], q[1], q[2]           # return (w, x, y, z)


def _geodesic_rot_err(R1, R2):
    Rrel = R1 @ R2.T
    ct = (np.trace(Rrel) - 1) / 2.0
    return float(np.arccos(np.clip(ct, -1.0, 1.0)))


def umeyama(src, tgt, with_scale=True):
    """Fit s, R, t such that  s * R @ src + t  ~ tgt  in LS sense."""
    mu_s = src.mean(0)
    mu_t = tgt.mean(0)
    Xs = src - mu_s
    Xt = tgt - mu_t
    H = Xs.T @ Xt / len(src)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    if with_scale:
        var_s = (Xs ** 2).sum() / len(src)
        s = S.sum() / var_s if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_t - s * R @ mu_s
    return R, t, s


def apply_sim(R, t, s, pts):
    return s * (R @ pts.T).T + t


def reanalyse_csv(in_csv, out_csv=None, in_place=False, verbose=True):
    """Append alignment-aware columns to a debug_log.csv.

    Columns added:
        trans_err_aligned_m, rot_err_aligned_rad   (Umeyama R+t fit)
        trans_err_sim_m,     rot_err_sim_rad       (Umeyama R+t+s fit)
        rpe_trans_m,         rpe_rot_rad           (body-frame RPE)

    If the CSV already contains these columns (e.g. live logger), they are
    refreshed. Returns (out_csv_path, summary_dict).

    Params:
        in_csv     : path to debug_log.csv
        out_csv    : optional output path
        in_place   : if True, overwrite in_csv; out_csv ignored
        verbose    : print summary to stdout
    """
    if in_place:
        out_csv = in_csv
    elif out_csv is None:
        out_csv = os.path.splitext(in_csv)[0] + '_aligned.csv'

    df = pd.read_csv(in_csv)
    N = len(df)
    if verbose:
        print(f'loaded {N} rows from {in_csv}')

    if N < 3:
        if verbose:
            print(f'  too few rows for alignment ({N}) — skipping')
        return out_csv, {'n_rows': N, 'skipped': True}

    est_t = df[['est_tx', 'est_ty', 'est_tz']].to_numpy()
    gt_t = df[['gt_tx', 'gt_ty', 'gt_tz']].to_numpy()

    est_R = np.stack([
        _quat_to_R(r.est_qw, r.est_qx, r.est_qy, r.est_qz)
        for r in df.itertuples(index=False)
    ])
    gt_R = np.stack([
        _quat_to_R(r.gt_qw, r.gt_qx, r.gt_qy, r.gt_qz)
        for r in df.itertuples(index=False)
    ])

    # --- Umeyama R+t (paper-style ATE alignment) ---
    R_rt, t_rt, _ = umeyama(est_t, gt_t, with_scale=False)
    est_rt = apply_sim(R_rt, t_rt, 1.0, est_t)
    est_R_rt = est_R @ R_rt.T  # rotate each pose by R_rt^T... actually we want R_rt @ est_R, but for geodesic comparison the order matters only for sign
    # For pose alignment, new_est_R[i] = R_rt @ est_R[i]
    est_R_rt = np.einsum('ij,kjl->kil', R_rt, est_R)

    trans_err_rt_mm = np.linalg.norm(est_rt - gt_t, axis=1) * 1000
    rot_err_rt_rad = np.array([
        _geodesic_rot_err(est_R_rt[i], gt_R[i]) for i in range(N)
    ])

    # --- Umeyama R+t+s (with scale correction) ---
    R_sim, t_sim, s_sim = umeyama(est_t, gt_t, with_scale=True)
    est_sim = apply_sim(R_sim, t_sim, s_sim, est_t)
    est_R_sim = np.einsum('ij,kjl->kil', R_sim, est_R)
    trans_err_sim_mm = np.linalg.norm(est_sim - gt_t, axis=1) * 1000
    rot_err_sim_rad = np.array([
        _geodesic_rot_err(est_R_sim[i], gt_R[i]) for i in range(N)
    ])

    # --- Relative pose error (RPE), body-frame (Sturm et al. TUM RGB-D) ---
    # Standard form: E_i = (Q_i^{-1} Q_{i+1})^{-1} (P_i^{-1} P_{i+1})
    # = (body-frame GT motion)^{-1} (body-frame est motion)
    # Invariant under any global (R, s, t) — truly alignment-free.
    rpe_trans_m = np.full(N, np.nan)
    rpe_rot_rad = np.full(N, np.nan)
    for i in range(1, N):
        # body-frame motion: Q_prev^{-1} @ Q_curr
        dR_gt = gt_R[i - 1].T @ gt_R[i]
        dt_gt = gt_R[i - 1].T @ (gt_t[i] - gt_t[i - 1])
        dR_est = est_R[i - 1].T @ est_R[i]
        dt_est = est_R[i - 1].T @ (est_t[i] - est_t[i - 1])
        # error E = gt_motion^{-1} @ est_motion
        E_R = dR_gt.T @ dR_est
        E_t = dR_gt.T @ (dt_est - dt_gt)
        rpe_trans_m[i] = float(np.linalg.norm(E_t))
        rpe_rot_rad[i] = _geodesic_rot_err(E_R, np.eye(3))

    # --- Write enhanced CSV ---
    df_out = df.copy()
    df_out['trans_err_aligned_m'] = trans_err_rt_mm / 1000.0
    df_out['rot_err_aligned_rad'] = rot_err_rt_rad
    df_out['trans_err_sim_m'] = trans_err_sim_mm / 1000.0
    df_out['rot_err_sim_rad'] = rot_err_sim_rad
    # If logger already wrote live rpe columns, keep its values and also
    # overwrite with the post-hoc version (identical if logger worked correctly).
    df_out['rpe_trans_m'] = rpe_trans_m
    df_out['rpe_rot_rad'] = rpe_rot_rad
    df_out.to_csv(out_csv, index=False)

    # --- Summary ---
    def rmse(x):
        x = x[~np.isnan(x)]
        return float(np.sqrt((x ** 2).mean())) if x.size else np.nan

    raw_trans_rmse = rmse(df['trans_err_m'].to_numpy() * 1000)
    raw_rot_rmse = rmse(df['rot_err_rad'].to_numpy())

    summary = {
        'n_rows': N,
        'raw_trans_rmse_mm': raw_trans_rmse,
        'raw_rot_rmse_rad': raw_rot_rmse,
        'ate_rt_trans_mm': rmse(trans_err_rt_mm),
        'ate_sim_trans_mm': rmse(trans_err_sim_mm),
        'rpe_trans_mm': rmse(rpe_trans_m * 1000),
        'rpe_rot_rad': rmse(rpe_rot_rad),
        'umeyama_scale': float(s_sim),
        'out_csv': out_csv,
    }

    if not verbose:
        return out_csv, summary

    print()
    print('Trajectory alignment summary')
    print(f'  N frames                        : {N}')
    print()
    print(f'  Raw (CSV as-written)            : trans RMSE = {raw_trans_rmse:7.2f} mm | rot RMSE = {raw_rot_rmse:.4f} rad')
    print(f'  Umeyama (R,t)    — paper ATE    : trans RMSE = {rmse(trans_err_rt_mm):7.2f} mm | rot RMSE = {rmse(rot_err_rt_rad):.4f} rad')
    print(f'  Umeyama (R,t,s)  — with scale   : trans RMSE = {rmse(trans_err_sim_mm):7.2f} mm | rot RMSE = {rmse(rot_err_sim_rad):.4f} rad')
    print(f'  RPE (frame-to-frame, invariant) : trans RMSE = {rmse(rpe_trans_m*1000):7.2f} mm | rot RMSE = {rmse(rpe_rot_rad):.4f} rad')
    print()
    ax = Rotation.from_matrix(R_sim).as_rotvec()
    ang = float(np.linalg.norm(ax))
    axis = ax / (ang + 1e-12)
    print('Umeyama fit details (with scale):')
    print(f'  R axis  = [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]')
    print(f'  R angle = {ang:.4f} rad ({np.degrees(ang):.2f} deg)')
    print(f'  scale s = {s_sim:.4f}   (1/s = {1.0 / s_sim:.4f})')
    print(f'  t       = [{t_sim[0] * 1000:+.2f}, {t_sim[1] * 1000:+.2f}, {t_sim[2] * 1000:+.2f}] mm')
    print()
    print(f'wrote {out_csv}')

    return out_csv, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--debug_dir', required=True,
                    help='folder containing debug_log.csv')
    ap.add_argument('--output_csv', default=None,
                    help='output path (default: debug_log_aligned.csv in debug_dir)')
    ap.add_argument('--in_place', action='store_true',
                    help='overwrite debug_log.csv in place')
    args = ap.parse_args()

    in_csv = os.path.join(args.debug_dir, 'debug_log.csv')
    out_csv = args.output_csv or os.path.join(args.debug_dir, 'debug_log_aligned.csv')
    reanalyse_csv(in_csv, out_csv=out_csv, in_place=args.in_place, verbose=True)


if __name__ == '__main__':
    main()
