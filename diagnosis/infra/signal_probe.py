"""
T0.1 signal-existence probe — "is there an attribution signal to route on?"

Renders the STATIC canonical (deformation_off=True, Δx≡0) for each frame from
the trained checkpoint's ESTIMATED poses, then asks: does the photometric
residual (|render - GT|) concentrate where the segmentation edge-prior
(exp(-d_edge/10), the same field the oracle gate uses) is HIGH?

  - If residual is HIGHER near the seg-prior (ratio >> 1, Pearson > 0): the static
    map fails exactly where deformation is expected -> a localised signal EXISTS
    for the oracle gate to route on. GO.
  - If residual is uniform vs the prior (ratio ~ 1, Pearson ~ 0): no localised
    signal -> a photometric/seg attribution gate is starved by construction.
    NO-GO -> pivot to track supervision.

This is the cheapest gate (cap with --max_frames). It is intentionally coarse:
GT/seg are resized to the (cropped) train dims for shape-safe alignment; the
ratio/Pearson statistic is robust to a few-pixel crop-vs-resize difference.

Usage:
  python diagnosis/infra/signal_probe.py \
    --config configs/Super/hyp2_baseline.yaml \
    --checkpoint output/hyp2_baseline/demo/checkpoint150.pt \
    --json out/signal_probe.json --max_frames 30 --frame_stride 5
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import cv2


def _tensorize_poses(pose):
    if isinstance(pose, dict):
        keys = sorted(pose.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else k)
        t = torch.stack([torch.as_tensor(pose[k]) for k in keys], dim=0)
    elif isinstance(pose, (list, tuple)):
        t = torch.stack([torch.as_tensor(p) for p in pose], dim=0)
    elif torch.is_tensor(pose):
        t = pose
    else:
        t = torch.as_tensor(np.array(pose))
    if t.dim() == 3 and t.shape[-2:] == (3, 4):
        pad = torch.zeros(t.shape[0], 1, 4)
        pad[..., 0, 3] = 1.0
        t = torch.cat([t, pad], dim=1)
    return t.float()


def _render_image(model, c2w, H, W, fx, fy, cx, cy, frame_time, ray_batch_size, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device).float(),
        torch.arange(H, device=device).float(),
        indexing='ij')
    i = i.T; j = j.T
    dirs = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], dim=-1)
    rays_d = dirs @ c2w[:3, :3].T
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    ts = torch.full(rays_o.shape[:-1] + (1,), frame_time, device=device)
    rays_o4 = torch.cat([rays_o, ts], dim=-1)
    flat_o = rays_o4.reshape(-1, 4)
    flat_d = rays_d.reshape(-1, 3)
    acc = []
    with torch.no_grad():
        for s in range(0, flat_o.shape[0], ray_batch_size):
            e = min(s + ray_batch_size, flat_o.shape[0])
            ret = model.render_rays(flat_o[s:e], flat_d[s:e])
            acc.append(ret['rgb'].cpu())
    return torch.cat(acc, dim=0).reshape(H, W, 3).numpy()


def main():
    ap = argparse.ArgumentParser(description='T0.1 attribution-signal existence probe')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max_frames', type=int, default=30)
    ap.add_argument('--frame_stride', type=int, default=5)
    ap.add_argument('--ray_batch_size', type=int, default=2048)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from config import load_config
    from model.scene_rep import JointEncoding
    from datasets.dataset import get_dataset, compute_edge_semantic

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    cfg['deformation_off'] = True  # T0.1: STATIC canonical render (Δx≡0)
    if 'training' not in cfg:
        cfg['training'] = {}
    if 'n_samples' not in cfg['training']:
        cfg['training']['n_samples'] = cfg['training'].get('n_samples_d', 32)

    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=device)
    model = JointEncoding(cfg, bound).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    est_c2w = _tensorize_poses(ckpt['pose']).to(device)

    dataset = get_dataset(cfg)
    H, W = int(dataset.H), int(dataset.W)
    fx, fy = float(dataset.fx), float(dataset.fy)
    cx, cy = float(dataset.cx), float(dataset.cy)
    img_files = dataset.img_files
    seg_files = dataset.semantic_paths
    N = min(len(img_files), len(seg_files), est_c2w.shape[0])

    frames = list(range(0, N, args.frame_stride))[:args.max_frames]
    print(f'[signal_probe] {len(frames)} frames @ {H}x{W}; deform-off render vs seg-prior')

    res_high, res_low, pear, ratios = [], [], [], []
    dummy_depth = np.zeros((H, W), dtype=np.float32)
    for idx in frames:
        render = np.clip(_render_image(model, est_c2w[idx], H, W, fx, fy, cx, cy,
                                       float(idx), args.ray_batch_size, device), 0.0, 1.0)
        gt = cv2.imread(img_files[idx])
        if gt is None:
            continue
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        gt = cv2.resize(gt, (W, H)).astype(np.float32) / 255.0
        seg = cv2.imread(seg_files[idx])
        seg = cv2.resize(seg, (W, H))
        edge = compute_edge_semantic(seg, dummy_depth).astype(np.float32)  # (H,W) in (0,1]

        residual = np.abs(render - gt).mean(axis=2)  # (H,W)
        thr = float(np.median(edge))
        hi = edge > thr
        lo = ~hi
        if hi.sum() > 0 and lo.sum() > 0:
            rh = float(residual[hi].mean())
            rl = float(residual[lo].mean())
            res_high.append(rh)
            res_low.append(rl)
            ratios.append(rh / (rl + 1e-9))
        r = np.corrcoef(residual.reshape(-1), edge.reshape(-1))[0, 1]
        if np.isfinite(r):
            pear.append(float(r))

    def _m(x):
        return float(np.mean(x)) if len(x) else None

    mean_ratio = _m(ratios)
    mean_pear = _m(pear)
    if mean_ratio is None:
        verdict = 'INCONCLUSIVE (no valid frames)'
    elif mean_ratio > 1.5 and (mean_pear or 0) > 0.1:
        verdict = 'GO: localised signal EXISTS near seg-prior'
    elif mean_ratio > 1.15:
        verdict = 'WEAK: mild localisation (borderline)'
    else:
        verdict = 'NO-GO: residual not localised to seg-prior -> gate starved'

    out = {
        'config': args.config, 'checkpoint': args.checkpoint,
        'n_frames': len(frames), 'HxW': f'{H}x{W}',
        'res_high_mean': _m(res_high), 'res_low_mean': _m(res_low),
        'res_high_over_low': mean_ratio, 'pearson_residual_vs_prior': mean_pear,
        'verdict': verdict,
    }
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
