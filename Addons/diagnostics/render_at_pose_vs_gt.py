#!/usr/bin/env python
"""
render_at_pose_vs_gt.py — Render full image at est pose and at GT-motion target
pose, side-by-side with actual RGB. Lets us see what the scene "thinks" the
world looks like from each vantage.

Usage:
    python render_at_pose_vs_gt.py \\
        --config configs/StereoMIS/p2_1.yaml \\
        --checkpoint output/StereoMIS/P2_1/demo/checkpoint3999.pt \\
        --frames 500,1500,2500,3500 \\
        --outdir /content/render_comparison
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Go up two levels: Addons/diagnostics -> Addons -> DDS-SLAM root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
import config as ddsconfig
from ddsslam import DDSSLAM


def compute_gt_target_pose(slam, frame_id):
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    gt_0 = _np(slam.pose_gt[0])
    gt_N = _np(slam.pose_gt[frame_id])
    est_0 = _np(slam.est_c2w_data[0])
    rel_gt_motion = gt_N @ np.linalg.inv(gt_0)
    return rel_gt_motion @ est_0


def render_full_image(slam, batch, c2w, frame_id, chunk=8192):
    """Render full H x W image given a 4x4 pose. Returns (rgb, depth) arrays."""
    device = slam.device
    c2w_t = torch.as_tensor(c2w, dtype=torch.float32, device=device)
    H, W = slam.dataset.H, slam.dataset.W

    dirs = batch['direction'].squeeze(0).to(device)  # [H, W, 3]
    rays_d_cam = dirs.reshape(-1, 3)  # [H*W, 3]
    N = rays_d_cam.shape[0]

    rays_o = c2w_t[:3, -1].unsqueeze(0).repeat(N, 1)
    rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_t[:3, :3], -1)

    # Depth-guided sampling (same as tracking_render uses) — avoids missing n_samples config
    target_d = batch['depth'].squeeze(0).to(device).reshape(-1, 1)

    if slam.config['dynamic']:
        timestamps = (float(frame_id) * torch.ones(N, device=device)).unsqueeze(-1)
        rays_o = torch.cat([rays_o, timestamps], dim=1)

    rgb_chunks = []
    depth_chunks = []
    slam.model.eval()
    with torch.no_grad():
        for i in range(0, N, chunk):
            ret = slam.model.render_rays(
                rays_o[i:i+chunk], rays_d[i:i+chunk], target_d=target_d[i:i+chunk]
            )
            rgb_chunks.append(ret['rgb'])
            depth_chunks.append(ret['depth'].squeeze(-1))
    rgb = torch.cat(rgb_chunks, 0).reshape(H, W, 3).cpu().numpy()
    depth = torch.cat(depth_chunks, 0).reshape(H, W).cpu().numpy()
    return rgb.clip(0, 1), depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--frames', default='500,1500,2500,3500')
    ap.add_argument('--outdir', default='render_comparison')
    args = ap.parse_args()

    cfg = ddsconfig.load_config(args.config, './configs/StereoMIS/stereomis.yaml')
    slam = DDSSLAM(cfg)
    slam.load_ckpt(args.checkpoint)
    os.makedirs(args.outdir, exist_ok=True)

    frames = [int(f) for f in args.frames.split(',')]
    for fid in frames:
        if fid not in slam.est_c2w_data:
            continue
        print(f'Rendering frame {fid}...')
        batch = slam.dataset[fid]
        batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}
        gt_rgb = batch['rgb'].squeeze(0).cpu().numpy()  # [H, W, 3] in [0,1]
        gt_depth = batch['depth'].squeeze(0).cpu().numpy()  # [H, W]

        c2w_est = slam.est_c2w_data[fid].cpu().numpy()
        c2w_gt = compute_gt_target_pose(slam, fid)

        rgb_est, depth_est = render_full_image(slam, batch, c2w_est, fid)
        rgb_gt, depth_gt = render_full_image(slam, batch, c2w_gt, fid)

        fig, ax = plt.subplots(2, 3, figsize=(15, 7))
        ax[0, 0].imshow(gt_rgb); ax[0, 0].set_title('actual RGB (ground truth image)'); ax[0, 0].axis('off')
        ax[0, 1].imshow(rgb_est); ax[0, 1].set_title(f'render @ EST pose'); ax[0, 1].axis('off')
        ax[0, 2].imshow(rgb_gt); ax[0, 2].set_title(f'render @ GT-target pose'); ax[0, 2].axis('off')

        dmax = float(max(gt_depth.max(), depth_est.max(), depth_gt.max()))
        ax[1, 0].imshow(gt_depth, cmap='turbo', vmin=0, vmax=dmax); ax[1, 0].set_title('actual depth'); ax[1, 0].axis('off')
        ax[1, 1].imshow(depth_est, cmap='turbo', vmin=0, vmax=dmax); ax[1, 1].set_title('render depth @ EST'); ax[1, 1].axis('off')
        ax[1, 2].imshow(depth_gt, cmap='turbo', vmin=0, vmax=dmax); ax[1, 2].set_title('render depth @ GT-target'); ax[1, 2].axis('off')

        plt.suptitle(f'frame {fid}', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = os.path.join(args.outdir, f'render_frame_{fid:05d}.png')
        plt.savefig(out, dpi=110)
        plt.close()
        print(f'  saved {out}')

    print(f'\nAll renders in {args.outdir}/')


if __name__ == '__main__':
    main()
