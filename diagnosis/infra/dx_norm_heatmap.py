"""
Δx norm heatmap overlaid on input RGB (visualization 1).

For a few key frames per snippet, renders the input RGB with a transparent
heatmap showing per-pixel max ||Δx|| along the ray.

Spatial signatures:
  - Smeared globally including over static tissue → gauge absorption
  - Localized at tool-tissue contact → honest deformation
  - Concentrated at edges / silhouettes → depth-error artifact

Uses the dx_hook.py output dump (NPZ files of canonical_x + delta_x).
Re-renders rays at FULL image resolution (not just dump's 4096 sparse
samples), then overlays.

Usage:
  python diagnosis/infra/dx_norm_heatmap.py \
    --config configs/CRCD/c1_001_paperfaith_lrfix.yaml \
    --checkpoint output/CRCD/c1_001_paperfaith_lrfix/demo/checkpoint359.pt \
    --rgb_dir data/CRCD/C1_001/video_frames \
    --output_dir diagnosis/report/dx_heatmap_C1_001 \
    --frames 0,90,180,270,359
"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--rgb_dir', type=str, required=True,
                        help='Dir of input RGB frames (e.g. data/CRCD/<NAME>/video_frames)')
    parser.add_argument('--rgb_pattern', type=str, default='*l.png',
                        help='Glob for left RGB (CRCD uses *l.png; Super uses *.png)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--frames', type=str, default='auto',
                        help='Comma-separated frame indices, or "auto" for 5 spread + 2 high-tool-motion')
    parser.add_argument('--ray_batch_size', type=int, default=240)
    parser.add_argument('--samples_per_ray', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Intrinsics from config (dataset may not be staged)
    cam = cfg['cam']
    H, W = int(cam['H']), int(cam['W'])
    fx, fy = float(cam['fx']), float(cam['fy'])
    cx, cy = float(cam['cx']), float(cam['cy'])

    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bounding_box).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Defensive c2w extraction
    est_c2w = ckpt['pose']
    if isinstance(est_c2w, dict):
        keys_sorted = sorted(est_c2w.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else k)
        est_c2w = torch.stack([torch.as_tensor(est_c2w[k]) for k in keys_sorted], dim=0)
    elif isinstance(est_c2w, list):
        est_c2w = torch.stack([torch.as_tensor(p) for p in est_c2w], dim=0)
    elif not torch.is_tensor(est_c2w):
        est_c2w = torch.as_tensor(est_c2w)
    if est_c2w.dim() == 3 and est_c2w.shape[-2:] == (3, 4):
        pad_row = torch.zeros(est_c2w.shape[0], 1, 4)
        pad_row[..., 0, 3] = 1.0
        est_c2w = torch.cat([est_c2w, pad_row], dim=1)
    est_c2w = est_c2w.to(device).float()
    model.eval()

    # Frame selection
    rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_pattern)))
    n_frames = min(len(rgb_files), est_c2w.shape[0])
    if args.frames == 'auto':
        # 5 evenly-spread frames
        frame_ids = [int(x) for x in np.linspace(0, n_frames - 1, 5).astype(int)]
    else:
        frame_ids = [int(x) for x in args.frames.split(',')]

    near = float(cfg['cam'].get('near', 0.0))
    far = float(cfg['cam'].get('far', 5.0))

    for fidx in tqdm(frame_ids, desc='dx_heatmap'):
        if fidx >= n_frames:
            print(f'  frame {fidx} out of range, skipping')
            continue
        # Load RGB
        rgb = cv2.imread(rgb_files[fidx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H))

        c2w = est_c2w[fidx]
        frame_time = float(fidx)

        # Build per-pixel rays
        i_grid, j_grid = torch.meshgrid(
            torch.arange(W, device=device).float(),
            torch.arange(H, device=device).float(),
            indexing='ij',
        )
        i_grid = i_grid.T  # (H, W)
        j_grid = j_grid.T
        x_cam = (i_grid - cx) / fx
        y_cam = (j_grid - cy) / fy
        dirs = torch.stack([x_cam, y_cam, torch.ones_like(x_cam)], dim=-1)  # (H, W, 3)
        rays_d_world = dirs @ c2w[:3, :3].T
        rays_d_world = rays_d_world / rays_d_world.norm(dim=-1, keepdim=True)
        rays_o_world = c2w[:3, 3].expand(rays_d_world.shape)

        # z samples
        z_vals = torch.linspace(near, far, args.samples_per_ray, device=device)
        # x_canonical(H, W, S, 3)
        x_can = rays_o_world.unsqueeze(2) + rays_d_world.unsqueeze(2) * z_vals.view(1, 1, -1, 1)

        dx_norm_max = torch.zeros(H, W, device=device)
        with torch.no_grad():
            flat = x_can.reshape(-1, 3)  # (H*W*S, 3)
            ft = torch.full(flat.shape[:1] + (1,), frame_time, device=device)
            total = flat.shape[0]
            chunks = []
            for s in range(0, total, args.ray_batch_size * args.samples_per_ray):
                e = min(s + args.ray_batch_size * args.samples_per_ray, total)
                pts_chunk = flat[s:e]
                ft_chunk = ft[s:e]
                if cfg.get('dynamic', False):
                    embed_time = model.embed_time(ft_chunk)
                    embed_pos = model.embed_fre_pos(pts_chunk)
                    h = torch.cat([embed_time, embed_pos], dim=-1)
                    vm = model.time_net(h)
                    vm = torch.where(ft_chunk.reshape(-1, ft_chunk.shape[-1]) == 0,
                                     torch.zeros_like(vm), vm)
                else:
                    vm = torch.zeros_like(pts_chunk)
                chunks.append(vm.norm(dim=-1).cpu())
            dx_norm = torch.cat(chunks, dim=0)
            dx_norm = dx_norm.reshape(H, W, args.samples_per_ray)
            dx_norm_max = dx_norm.max(dim=-1).values  # (H, W)

        dx_np = dx_norm_max.numpy()
        # Convert to mm for readability
        dx_mm = dx_np * 1000

        # Make figure: RGB | heatmap | overlay
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(rgb)
        axes[0].set_title(f'Input RGB (frame {fidx})')
        axes[0].axis('off')

        im = axes[1].imshow(dx_mm, cmap='hot', vmin=0)
        axes[1].set_title(f'max ||Δx|| along ray (mm)\n'
                          f'min={dx_mm.min():.2f}, mean={dx_mm.mean():.2f}, max={dx_mm.max():.2f}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(rgb)
        axes[2].imshow(dx_mm, cmap='hot', alpha=0.5, vmin=0)
        axes[2].set_title('Overlay (50% blend)')
        axes[2].axis('off')

        plt.suptitle(f'Δx spatial signature — frame {fidx}', fontsize=14)
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f'dx_heatmap_{fidx:04d}.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
