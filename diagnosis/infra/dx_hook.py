"""
Δx hook + dump (Test 1 smoking gun infrastructure).

Loads a DDS-SLAM checkpoint, walks the dataset, performs a forward pass per
frame, and dumps (canonical_x, delta_x, frame_id) NPZ files per frame.  The
dump is the input to test1_rigid_decomp.py (local Kabsch analysis).

Per workflow wx3zjzfyh + Wyrd's plan:
  - READ-ONLY hook: instruments model.scene_rep.run_network to extract Δx
    without changing forward behaviour.
  - Inference only — no training, no optimizer steps.
  - Per-sample dump: at the rays used for tracking pixel sampling, capture
    canonical_x BEFORE deformation and Δx = vox_motion (what the deformation
    network output).

Usage:
  python diagnosis/infra/dx_hook.py \
    --config configs/CRCD/c1_001_paperfaith_lrfix.yaml \
    --checkpoint output/CRCD/c1_001_paperfaith_lrfix/demo/checkpoint359.pt \
    --output_dir diagnosis/report/dx_dump_C1_001 \
    --max_frames 360 \
    --rays_per_frame 4096

Output layout:
  <output_dir>/
    frame_NNNN.npz     {x_canonical: (N, 3), delta_x: (N, 3), frame_time: float}
    summary.json       {n_frames, rays_per_frame, config_path, ckpt_path, ...}
"""

import argparse
import os
import sys
import json
import time
import numpy as np
import torch
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description='Δx + canonical-x dump (Test 1 enabler)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Cap on frames; default = all in ckpt')
    parser.add_argument('--rays_per_frame', type=int, default=4096,
                        help='How many random (u,v) pixels to sample per frame')
    parser.add_argument('--samples_per_ray', type=int, default=8,
                        help='How many SDF samples along each ray (8 = paper Sec IV n_stratified subset)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Repo root on path
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    # Intrinsics from config — avoid dataset construction (raw data may not be
    # staged on Colab; only the ckpt + config are needed for the Δx dump).
    cam = cfg['cam']
    H, W = int(cam['H']), int(cam['W'])
    fx, fy = float(cam['fx']), float(cam['fy'])
    cx, cy = float(cam['cx']), float(cam['cy'])

    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bounding_box).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])

    # Defensive c2w extraction — DDS-SLAM ckpt['pose'] can be:
    #   torch tensor (N, 4, 4)
    #   list of (4, 4) tensors
    #   dict {frame_id: tensor}  ← actual DDS-SLAM format
    est_c2w_data = ckpt['pose']
    if isinstance(est_c2w_data, dict):
        # Sort by frame index, stack
        keys_sorted = sorted(est_c2w_data.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else k)
        est_c2w_data = torch.stack([torch.as_tensor(est_c2w_data[k]) for k in keys_sorted], dim=0)
    elif isinstance(est_c2w_data, list):
        est_c2w_data = torch.stack([torch.as_tensor(p) for p in est_c2w_data], dim=0)
    elif not torch.is_tensor(est_c2w_data):
        est_c2w_data = torch.as_tensor(est_c2w_data)
    # Ensure (N, 4, 4)
    if est_c2w_data.dim() == 3 and est_c2w_data.shape[-2:] == (3, 4):
        # 3x4 → pad to 4x4
        pad_row = torch.zeros(est_c2w_data.shape[0], 1, 4)
        pad_row[..., 0, 3] = 1.0
        est_c2w_data = torch.cat([est_c2w_data, pad_row], dim=1)
    est_c2w_data = est_c2w_data.to(device).float()
    model.eval()

    if not cfg.get('dynamic', False):
        print('WARN: cfg.dynamic is False — deformation field is not active. Δx will be 0.')

    n_frames = min(args.max_frames or est_c2w_data.shape[0], est_c2w_data.shape[0])
    rays_per_frame = args.rays_per_frame
    samples_per_ray = args.samples_per_ray
    rng = np.random.default_rng(args.seed)

    print(f'Dumping Δx for {n_frames} frames × {rays_per_frame} rays × {samples_per_ray} samples')
    print(f'Output: {args.output_dir}')

    summary = {
        'n_frames': n_frames,
        'rays_per_frame': rays_per_frame,
        'samples_per_ray': samples_per_ray,
        'H': H, 'W': W,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
        'config': args.config,
        'checkpoint': args.checkpoint,
        'dynamic': cfg.get('dynamic', False),
        'png_depth_scale': cfg['cam'].get('png_depth_scale', 1000.0),
        'sc_factor': cfg.get('data', {}).get('sc_factor', 1.0),
        'bound': cfg['mapping']['bound'],
    }

    # near/far for ray sampling
    near = float(cfg['cam'].get('near', 0.0))
    far  = float(cfg['cam'].get('far', 5.0))

    t0 = time.time()
    for idx in tqdm(range(n_frames), desc='dx_hook'):
        c2w = est_c2w_data[idx]  # 4x4 torch
        # Frame timestamp — MUST match the coordinate the model was TRAINED on,
        # else the field is queried out-of-distribution and Δx/Var_t is garbage.
        # T1.2 (ddsslam.py:253...): time_normalize=True => frame_time = idx / N_frames
        # in [0,1]; else raw integer idx (upstream behaviour).
        if cfg.get('training', {}).get('time_normalize', False):
            frame_time = float(idx) / float(est_c2w_data.shape[0])
        else:
            frame_time = float(idx)

        # Sample random pixels (u, v) per frame
        us = rng.integers(0, W, size=rays_per_frame)
        vs = rng.integers(0, H, size=rays_per_frame)
        us_t = torch.from_numpy(us).to(device).long()
        vs_t = torch.from_numpy(vs).to(device).long()

        # Build rays in CAMERA frame: (x_cam, y_cam, 1) using pinhole
        x_cam = (us_t.float() - cx) / fx
        y_cam = (vs_t.float() - cy) / fy
        z_cam = torch.ones_like(x_cam)
        rays_d_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)
        # Transform to WORLD frame via c2w
        R = c2w[:3, :3]
        T = c2w[:3, 3]
        rays_d_world = (R @ rays_d_cam.T).T
        rays_d_world = rays_d_world / rays_d_world.norm(dim=-1, keepdim=True)
        rays_o_world = T.unsqueeze(0).expand_as(rays_d_world)

        # z samples along rays
        z_vals = torch.linspace(near, far, samples_per_ray, device=device)
        z_vals = z_vals.unsqueeze(0).expand(rays_per_frame, -1)
        # x_canonical = ray endpoint at each sample, BEFORE deformation
        x_canonical = rays_o_world.unsqueeze(1) + rays_d_world.unsqueeze(1) * z_vals.unsqueeze(-1)  # (N, S, 3)

        # Append timestamp as 4th dim (run_network expects (..., 4))
        ts = torch.full_like(x_canonical[..., :1], frame_time)
        pts = torch.cat([x_canonical, ts], dim=-1)
        pts_flat = pts.reshape(-1, 4)

        # FORWARD PASS THROUGH DEFORMATION FIELD ONLY (extract vox_motion)
        # We re-implement the relevant part of scene_rep.run_network so we can
        # capture vox_motion before it gets added to inputs_flat.
        with torch.no_grad():
            pts_xyz = pts_flat[:, :3]
            ft = pts_flat[:, 3].unsqueeze(-1)
            if cfg.get('dynamic', False):
                embed_time = model.embed_time(ft)
                embed_pos = model.embed_fre_pos(pts_xyz)
                h = torch.cat([embed_time, embed_pos], dim=-1)
                vox_motion = model.time_net(h)
                # SAME masking as scene_rep.py:171 — frame_time==0 forced to zero
                vox_motion = torch.where(ft.reshape(-1, ft.shape[-1]) == 0,
                                          torch.zeros_like(vox_motion), vox_motion)
            else:
                vox_motion = torch.zeros_like(pts_xyz)

        # Reshape back to (rays_per_frame, samples_per_ray, 3)
        x_can = x_canonical.cpu().numpy().astype(np.float32)
        dx = vox_motion.reshape(rays_per_frame, samples_per_ray, 3).cpu().numpy().astype(np.float32)

        np.savez_compressed(
            os.path.join(args.output_dir, f'frame_{idx:04d}.npz'),
            x_canonical=x_can,
            delta_x=dx,
            frame_time=np.float32(frame_time),
            c2w=c2w.cpu().numpy().astype(np.float32),
        )

    elapsed_min = (time.time() - t0) / 60
    summary['elapsed_min'] = elapsed_min
    summary['device'] = str(device)

    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Done. {n_frames} NPZ files written in {elapsed_min:.1f} min.')


if __name__ == '__main__':
    main()
