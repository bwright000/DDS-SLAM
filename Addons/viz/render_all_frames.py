"""
Render all frames from a saved DDS-SLAM checkpoint.

Renders RGB from the final trained scene at each frame's final estimated pose.
With --save_depth, also writes uint16 depth PNGs (scaled by config's png_depth_scale)
to <output_dir>/depth/. With --save_gt, also dumps the dataset's GT RGB (and, if
--save_depth is set, GT depth) next to the rendered versions.

Run from the DDS-SLAM root directory:
  python Addons/viz/render_all_frames.py \
    --config configs/StereoMIS/p2_1.yaml \
    --checkpoint output/StereoMIS/P2_1/demo/checkpoint3999.pt \
    --output_dir output/StereoMIS/P2_1/rendered_all \
    --save_depth --save_gt
"""

import argparse
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Ensure DDS-SLAM root is on path. File lives at Addons/viz/render_all_frames.py,
# so repo root is three levels up.
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if root not in sys.path:
    sys.path.insert(0, root)

from config import load_config
from datasets.dataset import get_dataset
from model.scene_rep import JointEncoding


def main():
    parser = argparse.ArgumentParser(description='Render all frames from checkpoint')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ray_batch_size', type=int, default=240)
    parser.add_argument('--skip', type=int, default=1, help='Render every N-th frame')
    parser.add_argument('--save_gt', action='store_true', help='Also save GT RGB frames (NNNN_gt.png)')
    parser.add_argument('--save_depth', action='store_true',
                        help='Also render + save depth. Rendered depth -> <output_dir>/depth/NNNN.png (uint16, '
                             'scaled by config[data][png_depth_scale]). GT depth saved too if --save_gt is set.')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, 'depth')
    if args.save_depth:
        os.makedirs(depth_dir, exist_ok=True)
    png_depth_scale = float(config['cam'].get('png_depth_scale', 1000.0))

    # Load dataset
    dataset = get_dataset(config)
    data_loader = DataLoader(dataset, num_workers=0)

    # Load model — must match ddsslam.py:40 signature JointEncoding(config, bounding_box)
    bounding_box = torch.from_numpy(np.array(config['mapping']['bound'])).to(device)
    model = JointEncoding(config, bounding_box).to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    est_c2w_data = ckpt['pose']
    model.eval()

    H, W = dataset.H, dataset.W
    dynamic = config.get('dynamic', False)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frames: {len(dataset)}, Resolution: {W}x{H}, Dynamic: {dynamic}")
    print(f"Rendering to: {args.output_dir} (skip={args.skip})")

    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(dataset), desc="Rendering"):
            if i % args.skip != 0:
                continue
            if i not in est_c2w_data:
                continue

            c2w = est_c2w_data[i].to(device).unsqueeze(0)  # [1, 4, 4]

            # Build rays for full image
            rays_d_cam = batch['direction'].squeeze(0).to(device)  # [H, W, 3]
            target_d = batch['depth'].squeeze(0).to(device).unsqueeze(-1).view(-1, 1)
            target_s = batch['rgb'].squeeze(0)
            target_edge_semantic = batch['edge_semantic'].squeeze(0).to(device).unsqueeze(-1)

            rays_o = c2w[:, :3, 3].repeat(H * W, 1)  # [H*W, 3]
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:, :3, :3], -1).view(-1, 3)

            # Render in batches
            rgb_chunks = []
            depth_chunks = [] if args.save_depth else None
            for j in range(0, rays_d.shape[0], args.ray_batch_size):
                torch.cuda.empty_cache()
                rays_o1 = rays_o[j:j + args.ray_batch_size]
                rays_d1 = rays_d[j:j + args.ray_batch_size]
                target_d1 = target_d[j:j + args.ray_batch_size]

                if dynamic:
                    cur_id = i * torch.ones(rays_o1.shape[0])
                    timestamps = cur_id.to(device)
                    rays_o1 = torch.cat([rays_o1, timestamps.unsqueeze(-1)], dim=1)

                ret = model.forward(rays_o1, rays_d1, target_s, target_d1,
                                    target_edge_semantic=target_edge_semantic,
                                    notFirstMap=False, render_only=True)
                rgb_chunks.append(ret['rgb'].cpu())
                if args.save_depth:
                    depth_chunks.append(ret['depth'].cpu())

            color = torch.cat(rgb_chunks, dim=0).reshape(H, W, 3).numpy()
            color = np.clip(color, 0, 1)
            # Normalize if needed
            if color.max() > 0:
                color = (color - color.min()) / (color.max() - color.min() + 1e-8)

            plt.imsave(os.path.join(args.output_dir, f'{i:04d}.png'), color)

            if args.save_depth:
                depth = torch.cat(depth_chunks, dim=0).reshape(H, W).numpy()
                depth_uint16 = np.clip(depth * png_depth_scale, 0, 65535).astype(np.uint16)
                cv2.imwrite(os.path.join(depth_dir, f'{i:04d}.png'), depth_uint16)

            if args.save_gt:
                gt_color = batch['rgb'].squeeze(0).numpy()
                gt_color = np.clip(gt_color, 0, 1)
                plt.imsave(os.path.join(args.output_dir, f'{i:04d}_gt.png'), gt_color)
                if args.save_depth:
                    gt_depth = batch['depth'].squeeze(0).numpy()
                    gt_depth_uint16 = np.clip(gt_depth * png_depth_scale, 0, 65535).astype(np.uint16)
                    cv2.imwrite(os.path.join(depth_dir, f'{i:04d}_gt.png'), gt_depth_uint16)

    print(f"Done. Rendered frames saved to {args.output_dir}")


if __name__ == '__main__':
    main()
