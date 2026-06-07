"""
Deformation-off render (Test 0 / 2 / 5 infrastructure).

Loads a DDS-SLAM checkpoint, sets the new config['deformation_off']=True flag
(Approach A per workflow wx3zjzfyh), and renders all frames.  Output is
DDS-SLAM-compatible: <output_dir>/NNNN.jpg + <output_dir>/depth/NNNN.png.

Pair with render_all_frames.py (canonical deformation-ON render) for:
  Test 0: depth-error floor on static frames (residual against I_t)
  Test 2: tool-cancellation residual (residual_off - residual_on inside tool mask)

Usage:
  # Deformation OFF render
  python diagnosis/infra/deform_off_render.py \
    --config configs/CRCD/c1_001_paperfaith_lrfix.yaml \
    --checkpoint output/CRCD/c1_001_paperfaith_lrfix/demo/checkpoint359.pt \
    --output_dir diagnosis/report/deform_off_C1_001

  # Then run canonical deformation-ON via Addons/viz/render_all_frames.py
  # (the existing script unchanged; flag stays False by default)
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Deformation-off render (Δx ≡ 0)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ray_batch_size', type=int, default=240)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--save_depth', action='store_true', default=True)
    parser.add_argument('--save_gt', action='store_true')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    # CRITICAL: this is the diagnosis flag (Approach A per workflow wx3zjzfyh).
    # When True, scene_rep.run_network forces vox_motion=0 → no deformation applied.
    cfg['deformation_off'] = True
    print('[deform_off_render] config["deformation_off"] = True (Δx≡0 inference)')

    # CRCD configs use n_samples_d (not n_samples) — scene_rep.py:313 wants the
    # latter literally.  Normal SLAM uses a different sampling path; our direct
    # render_rays() call needs n_samples present.  Populate it from n_samples_d
    # if missing.
    if 'training' not in cfg:
        cfg['training'] = {}
    if 'n_samples' not in cfg['training']:
        cfg['training']['n_samples'] = cfg['training'].get('n_samples_d', 32)
        print(f'[deform_off_render] set training.n_samples = {cfg["training"]["n_samples"]} (from n_samples_d)')

    os.makedirs(args.output_dir, exist_ok=True)
    depth_dir = os.path.join(args.output_dir, 'depth')
    if args.save_depth:
        os.makedirs(depth_dir, exist_ok=True)
    png_depth_scale = float(cfg['cam'].get('png_depth_scale', 10000.0))

    # Intrinsics from config (dataset may not be staged on fresh Colab)
    cam = cfg['cam']
    H, W = int(cam['H']), int(cam['W'])
    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bounding_box).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Defensive c2w extraction (DDS-SLAM pose can be dict/list/tensor)
    est_c2w_data = ckpt['pose']
    if isinstance(est_c2w_data, dict):
        keys_sorted = sorted(est_c2w_data.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else k)
        est_c2w_data = torch.stack([torch.as_tensor(est_c2w_data[k]) for k in keys_sorted], dim=0)
    elif isinstance(est_c2w_data, list):
        est_c2w_data = torch.stack([torch.as_tensor(p) for p in est_c2w_data], dim=0)
    elif not torch.is_tensor(est_c2w_data):
        est_c2w_data = torch.as_tensor(est_c2w_data)
    if est_c2w_data.dim() == 3 and est_c2w_data.shape[-2:] == (3, 4):
        pad_row = torch.zeros(est_c2w_data.shape[0], 1, 4)
        pad_row[..., 0, 3] = 1.0
        est_c2w_data = torch.cat([est_c2w_data, pad_row], dim=1)
    est_c2w_data = est_c2w_data.to(device).float()
    model.eval()

    # Also need fx, fy, cx, cy from config (dataset not loaded)
    fx, fy = float(cam['fx']), float(cam['fy'])
    cx, cy = float(cam['cx']), float(cam['cy'])
    # Override dataset attrs used downstream
    class CamShim: pass
    dataset = CamShim()
    dataset.H, dataset.W = H, W
    dataset.fx, dataset.fy = fx, fy
    dataset.cx, dataset.cy = cx, cy
    n_frames = est_c2w_data.shape[0]
    indices = list(range(0, n_frames, args.skip))
    print(f'Rendering {len(indices)} frames at output_dir={args.output_dir}')

    for idx in tqdm(indices, desc='deform_off_render'):
        c2w = est_c2w_data[idx]
        frame_time = float(idx)

        # Build full-image rays
        i, j = torch.meshgrid(
            torch.arange(W, device=device).float(),
            torch.arange(H, device=device).float(),
            indexing='ij',
        )
        i = i.T; j = j.T
        dirs = torch.stack([(i - dataset.cx) / dataset.fx,
                            (j - dataset.cy) / dataset.fy,
                            torch.ones_like(i)], dim=-1)
        rays_d = (dirs @ c2w[:3, :3].T)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        # Append timestamp as 4th column
        ts = torch.full(rays_o.shape[:-1] + (1,), frame_time, device=device)
        rays_o4 = torch.cat([rays_o, ts], dim=-1)

        flat_rays_o = rays_o4.reshape(-1, 4)
        flat_rays_d = rays_d.reshape(-1, 3)
        total = flat_rays_o.shape[0]

        color_acc = []
        depth_acc = []
        with torch.no_grad():
            for chunk_start in range(0, total, args.ray_batch_size):
                chunk_end = min(chunk_start + args.ray_batch_size, total)
                ret = model.render_rays(
                    flat_rays_o[chunk_start:chunk_end],
                    flat_rays_d[chunk_start:chunk_end],
                )
                color_acc.append(ret['rgb'].cpu())
                depth_acc.append(ret['depth'].cpu())
        color = torch.cat(color_acc, dim=0).reshape(H, W, 3).numpy()
        depth = torch.cat(depth_acc, dim=0).reshape(H, W).numpy()

        # Save RGB
        color_clipped = np.clip(color, 0.0, 1.0)
        plt.imsave(os.path.join(args.output_dir, f'{idx:04d}.jpg'),
                   color_clipped)
        # Save depth (uint16 at png_depth_scale)
        if args.save_depth:
            depth_uint16 = np.clip(depth * png_depth_scale, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(depth_dir, f'{idx:04d}.png'), depth_uint16)

    print(f'Done. {len(indices)} frames rendered.')


if __name__ == '__main__':
    main()
