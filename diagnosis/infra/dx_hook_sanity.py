"""
Sanity check for the deformation field BEFORE running the full dx_hook dump.

Loads the ckpt, picks 100 random points + 5 frame_time values, calls
model.time_net directly, and prints the output statistics.  Tells us if
the network is producing meaningful output or genuinely zero.

If output is ~zero across all inputs → the trained network is dormant
(meaningful finding; gauge absorption hypothesis cannot be tested
because the field carries nothing).

If output is non-zero → my dx_hook has a bug; iterate.
"""

import argparse
import os
import sys
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)

    bound = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bound).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f'ckpt keys: {list(ckpt.keys())}')
    print(f'cfg.dynamic = {cfg.get("dynamic")}')

    # Inspect what's in model state_dict for time_net
    msd = ckpt['model']
    time_net_keys = [k for k in msd if 'time_net' in k]
    print(f'time_net keys in ckpt[\'model\']: {time_net_keys}')
    for k in time_net_keys:
        v = msd[k]
        print(f'  {k}: shape={tuple(v.shape)}, '
              f'mean={v.float().mean().item():+.6f}, '
              f'abs_mean={v.float().abs().mean().item():.6f}, '
              f'max={v.float().abs().max().item():.6f}')

    model.load_state_dict(msd)
    model.eval()

    if not cfg.get('dynamic', False):
        print('cfg.dynamic = False — deformation field not active by config. STOP.')
        return

    # Pick 100 random 3D points inside the bound
    rng = np.random.default_rng(0)
    bound_np = np.array(cfg['mapping']['bound'])  # (3, 2)
    pts = rng.uniform(bound_np[:, 0], bound_np[:, 1], size=(100, 3))
    pts_t = torch.from_numpy(pts).float().to(device)

    print('\n=== Calling model.time_net for various frame_time values ===')
    for ft_val in [0.0, 1.0, 10.0, 100.0, 359.0, 729.0]:
        ft = torch.full((100, 1), ft_val, device=device).float()
        with torch.no_grad():
            embed_time = model.embed_time(ft)
            embed_pos = model.embed_fre_pos(pts_t)
            print(f'  embed_time({ft_val}): shape={tuple(embed_time.shape)}, '
                  f'mean={embed_time.mean().item():+.4f}, '
                  f'abs_mean={embed_time.abs().mean().item():.4f}')
            h = torch.cat([embed_time, embed_pos], dim=-1)
            vm = model.time_net(h)
            # Apply the same mask as scene_rep.py:171
            vm_masked = torch.where(ft.reshape(-1, ft.shape[-1]) == 0,
                                     torch.zeros_like(vm), vm)
            print(f'  frame_time={ft_val:6.1f}:  '
                  f'vox_motion mean={vm.mean().item():+.6e}  '
                  f'abs_mean={vm.abs().mean().item():.6e}  '
                  f'max={vm.abs().max().item():.6e}  '
                  f'(after mask: mean={vm_masked.mean().item():+.6e})')

    print('\n=== Calling model.run_network directly (same path as SLAM) ===')
    # Mimic SLAM's forward through run_network
    for ft_val in [1.0, 100.0, 359.0]:
        # Build (1, 1, 4) input: pts(3) + frame_time(1)
        pts_4 = torch.cat([pts_t, torch.full((100, 1), ft_val, device=device)], dim=-1).unsqueeze(0)
        with torch.no_grad():
            out, edge = model.run_network(pts_4)
        print(f'  frame_time={ft_val}: out shape={tuple(out.shape)}, '
              f'mean={out.mean().item():+.6f}, abs_mean={out.abs().mean().item():.6f}')


if __name__ == '__main__':
    main()
