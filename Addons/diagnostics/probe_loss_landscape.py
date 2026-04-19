#!/usr/bin/env python
"""
probe_loss_landscape.py — Diagnostic: does the rendering loss actually respond
to pose perturbations, or is the landscape flat?

Picks one frame, loads the trained model + its est_c2w from the final checkpoint,
and evaluates loss at systematic pose perturbations along each of 6 DoF:
- Translation: ±{0.1, 0.5, 1, 5, 10, 20} mm along x, y, z (camera frame)
- Rotation:    ±{0.1, 0.5, 1, 2, 5, 10} deg around x, y, z

Plots loss vs perturbation for each axis. If loss is quadratic-bowl-shaped with
minimum near zero perturbation → optimizer is undertuned (budget issue). If
loss is flat → rendering is insensitive to pose (scene self-similarity or
TimeNet absorbing error).

Usage (on Colab, dds_env active):
    cd /content/DDS-SLAM
    python probe_loss_landscape.py \\
        --config configs/StereoMIS/p2_1.yaml \\
        --checkpoint output/StereoMIS/P2_1/demo/checkpoint3999.pt \\
        --frames 500,1000,2000,3000,3500
"""
import argparse
import os
import sys
import json
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make local imports work — go up two levels: Addons/diagnostics -> Addons -> DDS-SLAM
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import config as ddsconfig
from ddsslam import DDSSLAM


def axis_angle_to_matrix_np(aa):
    """3-vec axis-angle -> 3x3 rotation matrix."""
    theta = np.linalg.norm(aa)
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def compute_loss_at_pose(slam, batch, c2w, frame_id):
    """Render with the given 4x4 pose and return the full loss."""
    device = slam.device
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)

    # Use all pixels minus ignore edges — same sampling convention as tracking_render
    iW = slam.config['tracking']['ignore_edge_W']
    iH = slam.config['tracking']['ignore_edge_H']
    H = slam.dataset.H - iH * 2
    W = slam.dataset.W - iW * 2

    n_samples = slam.config['tracking']['sample']
    # deterministic sample: uniform grid-ish (same indices every call)
    # use a fixed seed for reproducibility across perturbations
    torch.manual_seed(0)
    indice = torch.randperm(H * W)[:n_samples]
    indice_h, indice_w = indice % H, indice // H

    dirs = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :].to(device)
    rays_d_cam = dirs[indice_h, indice_w, :]
    target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(device)
    target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(device).unsqueeze(-1)
    target_es = batch['edge_semantic'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(device).unsqueeze(-1)
    border = batch['border'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(device).unsqueeze(-1)

    rays_o = c2w_t[:3, -1].unsqueeze(0).repeat(n_samples, 1)
    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_t[:3, :3], -1)

    if slam.config['dynamic']:
        timestamps = (float(frame_id) * torch.ones(rays_o.shape[0], device=device)).unsqueeze(-1)
        rays_o = torch.cat([rays_o, timestamps], dim=1)

    with torch.no_grad():
        slam.model.eval()
        # temporarily put in training mode so forward returns loss components
        slam.model.train()
        ret = slam.model.forward(
            rays_o, rays_d, target_s, target_d,
            target_edge_semantic=target_es, border=border, UseBorder=True,
        )
        slam.model.eval()

    # same weighted combination as tracking
    def _get(k):
        v = ret.get(k)
        return v.item() if v is not None else 0.0

    return {
        'total': slam.get_loss_from_ret(ret).item(),
        'rgb': _get('rgb_loss'),
        'depth': _get('depth_loss'),
        'sdf': _get('sdf_loss'),
        'fs': _get('fs_loss'),
        'edge_semantic': _get('edge_semantic_loss'),
        'psnr': _get('psnr'),
    }


def compute_gt_target_pose(slam, frame_id):
    """
    Compute what est[frame_id] SHOULD be if tracking matched GT motion perfectly.
    Uses relative GT motion from frame 0 applied to est_c2w_data[0] — avoids the
    coord-convention offset between est (y/z flipped) and raw GT.
    """
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    gt_0 = _np(slam.pose_gt[0])
    gt_N = _np(slam.pose_gt[frame_id])
    est_0 = _np(slam.est_c2w_data[0])
    rel_gt_motion = gt_N @ np.linalg.inv(gt_0)
    return rel_gt_motion @ est_0


def probe_frame(slam, dataset, frame_id, trans_perts_m, rot_perts_deg):
    """Run the full 6-axis sweep for one frame. Returns dict of results."""
    print(f'\n=== Probing frame {frame_id} ===')
    batch = dataset[frame_id]
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}
    c2w_base = slam.est_c2w_data[frame_id].cpu().numpy().copy()

    baseline = compute_loss_at_pose(slam, batch, c2w_base, frame_id)
    print(f'  baseline total_loss={baseline["total"]:.6f}  rgb={baseline["rgb"]:.6f}  depth={baseline["depth"]:.6f}  psnr={baseline["psnr"]:.2f}')

    # "What est should be" — GT-motion-derived pose in the same coord frame as est
    gt_target = compute_gt_target_pose(slam, frame_id)
    gt_target_loss = compute_loss_at_pose(slam, batch, gt_target, frame_id)
    delta_trans = float(np.linalg.norm(c2w_base[:3, 3] - gt_target[:3, 3]))
    R_delta = c2w_base[:3, :3] @ gt_target[:3, :3].T
    ct = np.clip((np.trace(R_delta) - 1) / 2, -1.0, 1.0)
    delta_rot_deg = float(np.degrees(np.arccos(ct)))
    print(f'  GT-target pose (relative-GT-motion applied to est[0]):')
    print(f'    loss={gt_target_loss["total"]:.6f}  rgb={gt_target_loss["rgb"]:.6f}  psnr={gt_target_loss["psnr"]:.2f}')
    print(f'    est vs GT-target: {delta_trans*1000:.2f} mm, {delta_rot_deg:.2f} deg')
    print(f'    loss(gt_target) - loss(est) = {gt_target_loss["total"] - baseline["total"]:.6f} '
          f'({"GT is BETTER" if gt_target_loss["total"] < baseline["total"] else "est is BETTER"})')

    results = {
        'frame_id': frame_id,
        'baseline': baseline,
        'gt_target': gt_target_loss,
        'gt_target_delta_trans_m': delta_trans,
        'gt_target_delta_rot_deg': delta_rot_deg,
        'trans': {},
        'rot': {},
    }
    axis_names = ['x', 'y', 'z']

    # Translation sweeps (in camera frame — translate world-position directly in world coords for simplicity)
    for ax, aname in enumerate(axis_names):
        results['trans'][aname] = []
        for dt in trans_perts_m:
            c2w_p = c2w_base.copy()
            c2w_p[ax, 3] += dt
            out = compute_loss_at_pose(slam, batch, c2w_p, frame_id)
            results['trans'][aname].append((dt, out))

    # Rotation sweeps
    for ax, aname in enumerate(axis_names):
        results['rot'][aname] = []
        for dtheta_deg in rot_perts_deg:
            aa = np.zeros(3)
            aa[ax] = np.deg2rad(dtheta_deg)
            R_pert = axis_angle_to_matrix_np(aa)
            c2w_p = c2w_base.copy()
            c2w_p[:3, :3] = c2w_base[:3, :3] @ R_pert
            out = compute_loss_at_pose(slam, batch, c2w_p, frame_id)
            results['rot'][aname].append((dtheta_deg, out))

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--frames', default='500,1000,2000,3000,3500',
                    help='Comma-separated list of frame indices to probe')
    ap.add_argument('--outdir', default='probe_output')
    args = ap.parse_args()

    frames = [int(f) for f in args.frames.split(',')]
    os.makedirs(args.outdir, exist_ok=True)

    # Load config (path in DDS-SLAM requires explicit default_path)
    cfg = ddsconfig.load_config(args.config, './configs/StereoMIS/stereomis.yaml')
    slam = DDSSLAM(cfg)
    slam.load_ckpt(args.checkpoint)
    print(f'Loaded checkpoint: {args.checkpoint}')
    print(f'est_c2w_data has {len(slam.est_c2w_data)} poses')

    trans_perts_m = [-0.020, -0.010, -0.005, -0.001, -0.0001, 0.0,
                      0.0001, 0.001, 0.005, 0.010, 0.020]
    rot_perts_deg = [-10, -5, -2, -1, -0.5, -0.1, 0.0,
                      0.1, 0.5, 1, 2, 5, 10]

    all_results = []
    for fid in frames:
        if fid not in slam.est_c2w_data:
            print(f'SKIP: frame {fid} not in est_c2w_data')
            continue
        res = probe_frame(slam, slam.dataset, fid, trans_perts_m, rot_perts_deg)
        all_results.append(res)

    # Plot per-frame: 2x3 grid, rows = trans/rot, cols = x/y/z
    for res in all_results:
        fid = res['frame_id']
        baseline_loss = res['baseline']['total']
        gt_loss = res['gt_target']['total']
        dt = res['gt_target_delta_trans_m'] * 1000
        dr = res['gt_target_delta_rot_deg']
        fig, ax = plt.subplots(2, 3, figsize=(15, 7))
        suptitle = (f'frame {fid}  |  est_loss={baseline_loss:.4f}  |  '
                    f'gt_target_loss={gt_loss:.4f}  |  Δ(est→gt)={dt:.1f}mm, {dr:.2f}°')
        fig.suptitle(suptitle, fontsize=11)
        for i, aname in enumerate(['x', 'y', 'z']):
            dts = [d * 1000 for d, _ in res['trans'][aname]]  # mm
            losses = [o['total'] for _, o in res['trans'][aname]]
            ax[0, i].plot(dts, losses, '-o', lw=1, ms=3, color='C0')
            ax[0, i].axvline(0, color='k', lw=0.5, alpha=0.3)
            ax[0, i].axhline(baseline_loss, color='C0', lw=1, alpha=0.6, ls='--', label=f'est: {baseline_loss:.4f}')
            ax[0, i].axhline(gt_loss, color='C3', lw=1, alpha=0.6, ls='--', label=f'gt_target: {gt_loss:.4f}')
            ax[0, i].set_title(f'trans_{aname}')
            ax[0, i].set_xlabel('mm'); ax[0, i].set_ylabel('total_loss')
            ax[0, i].legend(fontsize=7, loc='upper center')

            dts_r = [d for d, _ in res['rot'][aname]]
            losses_r = [o['total'] for _, o in res['rot'][aname]]
            ax[1, i].plot(dts_r, losses_r, '-o', lw=1, ms=3, color='C0')
            ax[1, i].axvline(0, color='k', lw=0.5, alpha=0.3)
            ax[1, i].axhline(baseline_loss, color='C0', lw=1, alpha=0.6, ls='--', label=f'est: {baseline_loss:.4f}')
            ax[1, i].axhline(gt_loss, color='C3', lw=1, alpha=0.6, ls='--', label=f'gt_target: {gt_loss:.4f}')
            ax[1, i].set_title(f'rot_{aname}')
            ax[1, i].set_xlabel('deg'); ax[1, i].set_ylabel('total_loss')
            ax[1, i].legend(fontsize=7, loc='upper center')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(args.outdir, f'landscape_frame_{fid:05d}.png')
        plt.savefig(out, dpi=100)
        plt.close()
        print(f'Saved {out}')

    # Cross-frame summary table
    print('\n=== SUMMARY ===')
    print(f"{'frame':<7}{'est_loss':<12}{'gt_loss':<12}{'Δloss':<12}{'est→gt':<14}{'verdict':<20}")
    for res in all_results:
        el = res['baseline']['total']
        gl = res['gt_target']['total']
        dtmm = res['gt_target_delta_trans_m'] * 1000
        drdeg = res['gt_target_delta_rot_deg']
        verdict = 'scene biased' if gl > el * 1.1 else ('ambiguous' if abs(gl - el) / el < 0.1 else 'worse local min')
        print(f"{res['frame_id']:<7}{el:<12.6f}{gl:<12.6f}{gl-el:<+12.6f}{f'{dtmm:.1f}mm,{drdeg:.1f}°':<14}{verdict:<20}")

    # Save raw JSON for later analysis
    def _jsonify(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, (np.floating, float)): return float(v)
        if isinstance(v, (np.integer, int)): return int(v)
        if isinstance(v, dict): return {k: _jsonify(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)): return [_jsonify(x) for x in v]
        return v
    with open(os.path.join(args.outdir, 'results.json'), 'w') as f:
        json.dump([_jsonify(r) for r in all_results], f, indent=2)
    print(f'\nAll results written to {args.outdir}/')


if __name__ == '__main__':
    main()
