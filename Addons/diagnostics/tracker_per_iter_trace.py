"""
Instrumented re-run of tracking_render that dumps pose + losses at EVERY iteration.

Answers: for a handful of hand-picked frames, does the tracker (a) converge to
the constant-velocity init, (b) converge to a different minimum, or
(c) oscillate without settling?

Takes a trained checkpoint (scene + final pose dict), seeds predict_current_pose
from the saved est_c2w_data, then runs the inner tracking loop for each target
frame while logging pose/loss after every Adam step. No scene update is
performed (frozen decoder + hash grid). No BA.

Output: <out_dir>/tracker_trace.csv with one row per (frame_id, iter_idx).

  python Addons/diagnostics/tracker_per_iter_trace.py \\
    --config configs/StereoMIS/p2_1.yaml \\
    --checkpoint /content/checkpoint3999.pt \\
    --frames 100 500 1000 2000 3000 3500 \\
    --out_dir /content/tracker_trace
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import load_config
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding
from optimization.utils import (at_to_transform_matrix, matrix_to_axis_angle,
                                 matrix_to_quaternion, qt_to_transform_matrix)


def _geod(R1, R2):
    ct = (np.trace(R1 @ R2.T) - 1) / 2
    return float(np.arccos(np.clip(ct, -1.0, 1.0)))


def _pose_errs(est, gt):
    """(trans_m, rot_rad) between two 4x4 c2w matrices (numpy)."""
    t_err = float(np.linalg.norm(est[:3, 3] - gt[:3, 3]))
    r_err = _geod(est[:3, :3], gt[:3, :3])
    return t_err, r_err


def _rpe_body(est_prev, est_curr, gt_prev, gt_curr):
    """Sturm-style RPE, body-frame (invariant to global alignment)."""
    dR_gt = gt_prev[:3, :3].T @ gt_curr[:3, :3]
    dt_gt = gt_prev[:3, :3].T @ (gt_curr[:3, 3] - gt_prev[:3, 3])
    dR_est = est_prev[:3, :3].T @ est_curr[:3, :3]
    dt_est = est_prev[:3, :3].T @ (est_curr[:3, 3] - est_prev[:3, 3])
    E_R = dR_gt.T @ dR_est
    E_t = dR_gt.T @ (dt_est - dt_gt)
    return float(np.linalg.norm(E_t)), _geod(E_R, np.eye(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--frames', type=int, nargs='+',
                    help='which frame IDs to probe (required)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--iters_override', type=int, default=None,
                    help='override config tracking.iter (default = config value)')
    ap.add_argument('--rot_rep', default='axis_angle',
                    choices=['axis_angle', 'quat'])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    print(f'device: {device}')
    print(f'config: {args.config}')

    # --- build scene ---
    dataset = get_dataset(cfg)
    bb = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bb).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    saved_poses = ckpt['pose']
    model.train()  # loss path active; we manually freeze params below
    for p in model.parameters():
        p.requires_grad_(False)
    print(f'checkpoint: {args.checkpoint} ({len(saved_poses)} poses)')

    iters_run = args.iters_override or cfg['tracking']['iter']
    sample_N = cfg['tracking']['sample']
    lr_rot = cfg['tracking']['lr_rot']
    lr_trans = cfg['tracking']['lr_trans']
    iW, iH = cfg['tracking']['ignore_edge_W'], cfg['tracking']['ignore_edge_H']
    trunc = cfg['training']['trunc'] * cfg['data']['sc_factor']

    matrix_from_tensor = at_to_transform_matrix if args.rot_rep == 'axis_angle' else qt_to_transform_matrix
    matrix_to_tensor = matrix_to_axis_angle if args.rot_rep == 'axis_angle' else matrix_to_quaternion

    # --- output csv ---
    out_csv = os.path.join(args.out_dir, 'tracker_trace.csv')
    f = open(out_csv, 'w', newline='')
    w = csv.writer(f)
    w.writerow([
        'frame_id', 'iter_idx',
        'est_tx', 'est_ty', 'est_tz',
        'init_tx', 'init_ty', 'init_tz',
        'gt_tx', 'gt_ty', 'gt_tz',
        'trans_err_raw_m', 'rot_err_raw_rad',
        'rpe_trans_m', 'rpe_rot_rad',
        'dist_from_init_m', 'dist_from_best_so_far_m',
        'rendering_loss',
        'loss_rgb', 'loss_depth', 'loss_sdf', 'loss_fs', 'loss_edge_semantic',
        'n_fs', 'n_sdf', 'n_back', 'n_total',
    ])

    for frame_id in args.frames:
        if frame_id < 2 or frame_id >= len(dataset):
            print(f'  skip frame {frame_id} (out of range / first two frames)')
            continue
        if frame_id not in saved_poses:
            print(f'  skip frame {frame_id} (missing from checkpoint)')
            continue

        # seed const-velocity init from checkpoint's saved poses
        est_prev_prev = saved_poses[frame_id - 2].to(device)
        est_prev = saved_poses[frame_id - 1].to(device)
        delta = est_prev @ est_prev_prev.float().inverse()
        cur_c2w_init = (delta @ est_prev).detach()
        cur_c2w_init_np = cur_c2w_init.cpu().numpy()

        # fresh optimizer per frame (mirrors tracking_render)
        cur_rot = torch.nn.parameter.Parameter(
            matrix_to_tensor(cur_c2w_init[None, :3, :3]))
        cur_trans = torch.nn.parameter.Parameter(cur_c2w_init[None, :3, 3])
        opt = torch.optim.Adam([
            {'params': cur_rot, 'lr': lr_rot},
            {'params': cur_trans, 'lr': lr_trans},
        ])

        # pull input batch for this frame (dataset __getitem__)
        batch = dataset[frame_id]
        rgb_hw3 = batch['rgb'].to(device)
        depth_hw = batch['depth'].to(device)
        edge_hw = batch['edge_semantic'].to(device)
        border_hw = batch['border'].to(device)
        rays_d_cam_full = batch['direction'].to(device)

        # GT for ground-truth comparison (live, raw)
        gt_c2w = dataset.gt_poses[frame_id].to(device) if dataset.gt_poses is not None else saved_poses[frame_id].to(device)
        gt_c2w_np = gt_c2w.cpu().numpy()
        est_prev_np = est_prev.cpu().numpy()
        gt_prev_np = (dataset.gt_poses[frame_id - 1] if dataset.gt_poses is not None else saved_poses[frame_id - 1]).cpu().numpy()

        # fix sampled ray indices ONCE per frame (matches tracking_render)
        import random
        indice = random.sample(range(dataset.H * dataset.W - (iH * 2) * (dataset.W - iW * 2)),
                               sample_N)
        indice = torch.tensor(indice)
        indice_h = indice % (dataset.H - iH * 2)
        indice_w = indice // (dataset.H - iH * 2)

        target_s = rgb_hw3[iH:-iH, iW:-iW, :][indice_h, indice_w, :]
        target_d = depth_hw[iH:-iH, iW:-iW][indice_h, indice_w].unsqueeze(-1)
        target_edge = edge_hw[iH:-iH, iW:-iW][indice_h, indice_w].unsqueeze(-1)
        border_sel = border_hw[iH:-iH, iW:-iW][indice_h, indice_w].unsqueeze(-1)
        rays_d_cam = rays_d_cam_full[iH:-iH, iW:-iW, :][indice_h, indice_w, :]

        print(f'frame {frame_id}: tracking for {iters_run} iters, sample={sample_N}')

        best_loss = None
        best_pose_np = cur_c2w_init_np.copy()

        for it in range(iters_run):
            opt.zero_grad()
            c2w_est = matrix_from_tensor(cur_rot, cur_trans)  # [1,4,4]
            rays_o = c2w_est[..., :3, -1].repeat(sample_N, 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            if cfg.get('dynamic', False):
                ts = float(frame_id) * torch.ones(sample_N, 1, device=device)
                rays_o = torch.cat([rays_o, ts], dim=1)

            ret = model.forward(rays_o, rays_d, target_s, target_d,
                                target_edge_semantic=target_edge,
                                border=border_sel, UseBorder=True)

            # replicate ddsslam.get_loss_from_ret on StereoMIS weights
            loss = (cfg['training']['rgb_weight'] * ret['rgb_loss']
                    + cfg['training']['depth_weight'] * ret['depth_loss']
                    + cfg['training']['sdf_weight'] * ret['sdf_loss']
                    + cfg['training']['fs_weight'] * ret['fs_loss']
                    + cfg['training']['rgb_weight'] * 0.1 * ret['edge_semantic_loss'])

            # snapshot BEFORE the step (captures the pose this loss was evaluated at)
            with torch.no_grad():
                c2w_np = c2w_est.detach().cpu().numpy()[0]
                te, re = _pose_errs(c2w_np, gt_c2w_np)
                rpe_t, rpe_r = _rpe_body(est_prev_np, c2w_np, gt_prev_np, gt_c2w_np)
                d_init = float(np.linalg.norm(c2w_np[:3, 3] - cur_c2w_init_np[:3, 3]))
                if best_loss is None or loss.item() < best_loss:
                    best_loss = loss.item()
                    best_pose_np = c2w_np.copy()
                d_best = float(np.linalg.norm(c2w_np[:3, 3] - best_pose_np[:3, 3]))
                st = ret.get('sdf_stats') or {}
                w.writerow([
                    frame_id, it,
                    float(c2w_np[0, 3]), float(c2w_np[1, 3]), float(c2w_np[2, 3]),
                    float(cur_c2w_init_np[0, 3]), float(cur_c2w_init_np[1, 3]), float(cur_c2w_init_np[2, 3]),
                    float(gt_c2w_np[0, 3]), float(gt_c2w_np[1, 3]), float(gt_c2w_np[2, 3]),
                    te, re, rpe_t, rpe_r, d_init, d_best,
                    loss.item(),
                    float(ret['rgb_loss'].item()) if ret.get('rgb_loss') is not None else '',
                    float(ret['depth_loss'].item()) if ret.get('depth_loss') is not None else '',
                    float(ret['sdf_loss'].item()) if ret.get('sdf_loss') is not None else '',
                    float(ret['fs_loss'].item()) if ret.get('fs_loss') is not None else '',
                    float(ret['edge_semantic_loss'].item()) if ret.get('edge_semantic_loss') is not None else '',
                    st.get('n_fs_samples', ''), st.get('n_sdf_samples', ''),
                    st.get('n_back_samples', ''), st.get('n_total_samples', ''),
                ])

            loss.backward()
            opt.step()

        print(f'  init_to_gt     : {np.linalg.norm(cur_c2w_init_np[:3,3] - gt_c2w_np[:3,3])*1000:.3f} mm')
        print(f'  best_to_gt     : {np.linalg.norm(best_pose_np[:3,3] - gt_c2w_np[:3,3])*1000:.3f} mm')
        print(f'  best_to_init   : {np.linalg.norm(best_pose_np[:3,3] - cur_c2w_init_np[:3,3])*1000:.3f} mm')

    f.close()
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
