"""
Per-frame diagnostic logger for DDS-SLAM debug runs.

Writes `debug_log.csv` with one row per frame containing:
- timestamps, frame id, keyframe flag
- estimated pose (translation + quaternion) and GT pose
- translational + rotational error (geodesic)
- initialization-vs-GT error (how good was const-velocity prediction)
- tracking stats: iterations used, best/last loss, individual loss components, PSNR
- input stats: depth valid fraction, rgb mean

Also saves periodic pose snapshots as .npz for retrospective trajectory analysis.

All fields are optional in log_tracking() — passing None fills the cell with ''.
No runtime dependency: uses only numpy + stdlib csv.
"""

import atexit
import csv
import os
import time

import numpy as np
import torch


def _to_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _rotation_matrix_to_quat(R):
    """Convert 3x3 rotation to quaternion (w, x, y, z). No external deps."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(w), float(x), float(y), float(z)


def _pose_components(c2w):
    """Extract (tx, ty, tz, qw, qx, qy, qz) from a 4x4 pose."""
    p = _to_np(c2w)
    t = p[:3, 3]
    qw, qx, qy, qz = _rotation_matrix_to_quat(p[:3, :3])
    return float(t[0]), float(t[1]), float(t[2]), qw, qx, qy, qz


def _pose_error(est, gt):
    """Translational (m) and rotational (rad, geodesic) error between two 4x4 poses."""
    est = _to_np(est)
    gt = _to_np(gt)
    trans_err = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
    R = est[:3, :3] @ gt[:3, :3].T
    ct = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    rot_err = float(np.arccos(ct))
    return trans_err, rot_err


# Canonical CSV columns — fixed order so post-analysis is trivial
COLUMNS = [
    'wall_s', 'frame_id', 'is_keyframe',
    # estimated pose
    'est_tx', 'est_ty', 'est_tz', 'est_qw', 'est_qx', 'est_qy', 'est_qz',
    # GT pose
    'gt_tx', 'gt_ty', 'gt_tz', 'gt_qw', 'gt_qx', 'gt_qy', 'gt_qz',
    # RAW per-frame error — ONLY meaningful if est and GT share a coord frame.
    # On StereoMIS they don't (dataset.py y/z-flips est but not GT). Use
    # rpe_trans_m / rpe_rot_rad below, or reanalyse_debug_csv.py for aligned ATE.
    'trans_err_m', 'rot_err_rad',
    # const-velocity initialization error (same caveat as trans_err_m)
    'init_trans_err_m', 'init_rot_err_rad',
    # Relative pose error (body-frame, Sturm TUM RGB-D convention) — alignment-invariant
    'rpe_trans_m', 'rpe_rot_rad',
    # tracking optimization stats
    'tracking_iters_used', 'tracking_iters_config',
    'best_loss', 'last_loss',
    # loss components at the BEST-iter pose (the one stored in est_c2w_data)
    'loss_rgb', 'loss_depth', 'loss_sdf', 'loss_fs', 'loss_edge_semantic',
    'psnr',
    # SDF/FS mask diagnostics at BEST-iter pose — explains why loss_sdf/loss_fs may be 0
    'n_fs_samples', 'n_sdf_samples', 'n_back_samples', 'n_total_samples',
    'zval_min', 'zval_max',
    'target_d_min', 'target_d_max', 'target_d_n_valid',
    # input stats
    'depth_valid_frac', 'depth_mean', 'rgb_mean',
    # delta from previous frame (est only; for RPE use rpe_* columns)
    'trans_delta_m', 'rot_delta_rad',
]


class DebugLogger:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, 'debug_log.csv')
        self._csv_f = open(self.csv_path, 'w', newline='', buffering=1)
        self._writer = csv.writer(self._csv_f)
        self._writer.writerow(COLUMNS)
        self._poses_dir = os.path.join(out_dir, 'pose_snapshots')
        os.makedirs(self._poses_dir, exist_ok=True)
        self._prev_est = None
        self._prev_gt = None
        self._start = time.time()
        self._closed = False
        # Run reanalyse_csv on process exit so the CSV always gets
        # trans_err_aligned_m / rpe_trans_m / scale-aware ATE columns even
        # if close() was never called explicitly.
        atexit.register(self._finalise)
        print(f'[DebugLogger] writing to {self.csv_path}')

    def log_tracking(
        self,
        frame_id,
        est_c2w,
        gt_c2w,
        is_keyframe=False,
        init_c2w=None,
        tracking_iters_used=None,
        tracking_iters_config=None,
        best_loss=None,
        last_loss=None,
        loss_components=None,
        psnr=None,
        depth_valid_frac=None,
        depth_mean=None,
        rgb_mean=None,
        sdf_stats=None,
    ):
        """Append one row to debug_log.csv with everything known about this frame."""
        est_t = _pose_components(est_c2w)
        gt_t = _pose_components(gt_c2w)
        trans_err, rot_err = _pose_error(est_c2w, gt_c2w)

        init_trans_err, init_rot_err = ('', '')
        if init_c2w is not None:
            a, b = _pose_error(init_c2w, gt_c2w)
            init_trans_err, init_rot_err = a, b

        trans_delta, rot_delta = ('', '')
        if self._prev_est is not None:
            a, b = _pose_error(est_c2w, self._prev_est)
            trans_delta, rot_delta = a, b

        # RPE (body-frame, alignment-invariant): E = (gt_prev^-1 gt_curr)^-1 (est_prev^-1 est_curr)
        rpe_trans, rpe_rot = ('', '')
        if self._prev_est is not None and self._prev_gt is not None:
            ec = _to_np(est_c2w)
            ep = _to_np(self._prev_est)
            gc = _to_np(gt_c2w)
            gp = _to_np(self._prev_gt)
            eR_p, eR_c = ep[:3, :3], ec[:3, :3]
            gR_p, gR_c = gp[:3, :3], gc[:3, :3]
            dR_est = eR_p.T @ eR_c
            dt_est = eR_p.T @ (ec[:3, 3] - ep[:3, 3])
            dR_gt = gR_p.T @ gR_c
            dt_gt = gR_p.T @ (gc[:3, 3] - gp[:3, 3])
            E_R = dR_gt.T @ dR_est
            E_t = dR_gt.T @ (dt_est - dt_gt)
            rpe_trans = float(np.linalg.norm(E_t))
            ct = np.clip((np.trace(E_R) - 1) / 2, -1.0, 1.0)
            rpe_rot = float(np.arccos(ct))

        self._prev_est = _to_np(est_c2w).copy()
        self._prev_gt = _to_np(gt_c2w).copy()

        lc = loss_components or {}
        st = sdf_stats or {}
        row = [
            f'{time.time() - self._start:.3f}',
            int(frame_id),
            1 if is_keyframe else 0,
            *est_t,
            *gt_t,
            trans_err, rot_err,
            init_trans_err, init_rot_err,
            rpe_trans, rpe_rot,
            '' if tracking_iters_used is None else int(tracking_iters_used),
            '' if tracking_iters_config is None else int(tracking_iters_config),
            '' if best_loss is None else float(best_loss),
            '' if last_loss is None else float(last_loss),
            lc.get('rgb', ''),
            lc.get('depth', ''),
            lc.get('sdf', ''),
            lc.get('fs', ''),
            lc.get('edge_semantic', ''),
            '' if psnr is None else float(psnr),
            st.get('n_fs_samples', ''),
            st.get('n_sdf_samples', ''),
            st.get('n_back_samples', ''),
            st.get('n_total_samples', ''),
            st.get('zval_min', ''),
            st.get('zval_max', ''),
            st.get('target_d_min', ''),
            st.get('target_d_max', ''),
            st.get('target_d_n_valid', ''),
            '' if depth_valid_frac is None else float(depth_valid_frac),
            '' if depth_mean is None else float(depth_mean),
            '' if rgb_mean is None else float(rgb_mean),
            trans_delta, rot_delta,
        ]
        self._writer.writerow(row)

    def _finalise(self):
        """Flush CSV and append alignment-aware columns in-place.

        Called automatically via atexit. Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._csv_f.close()
        except Exception:
            pass
        try:
            from reanalyse_debug_csv import reanalyse_csv
        except ImportError:
            # fall back to import by absolute path (debug_logger is placed
            # on sys.path by ddsslam.py but reanalyse_debug_csv may not be)
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                'reanalyse_debug_csv',
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'reanalyse_debug_csv.py'))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            reanalyse_csv = mod.reanalyse_csv
        try:
            _, summary = reanalyse_csv(self.csv_path, in_place=True, verbose=False)
            print(f'[DebugLogger] CSV finalised: '
                  f'ATE(R,t)={summary.get("ate_rt_trans_mm", float("nan")):.2f}mm, '
                  f'RPE_trans={summary.get("rpe_trans_mm", float("nan")):.3f}mm/frame, '
                  f'scale={summary.get("umeyama_scale", float("nan")):.4f}')
        except Exception as e:
            print(f'[DebugLogger] reanalyse failed (CSV still valid): {e}')

    def save_pose_snapshot(self, est_c2w_data, tag):
        """Dump all current pose estimates to npz for retrospective analysis."""
        ids = sorted(est_c2w_data.keys())
        if not ids:
            return
        arr = np.stack([_to_np(est_c2w_data[i]) for i in ids])
        path = os.path.join(self._poses_dir, f'poses_{tag}.npz')
        np.savez(path, ids=np.array(ids), poses=arr)
        print(f'[DebugLogger] saved {len(ids)} poses -> {path}')

    def close(self):
        """Finalise CSV + run alignment. Idempotent — safe to call twice."""
        self._finalise()
