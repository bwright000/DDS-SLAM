"""
Field-liveness probe (T1.1/T1.3 read-out) — rescaling-invariant.

The deformation field (TimeNet) is bias-free ReLU => positively homogeneous in
its weights: W1->a*W1, W2->W2/a leaves the FUNCTION unchanged. So weight-norm
metrics (timenet_l2) LIE about liveness. This probe works in FUNCTION space:
it evaluates Δx(x,t) on a FIXED canonical grid (same points every frame) across
all frames and decomposes the variance of Δx into TEMPORAL vs SPATIAL.

  - A DEAD field           -> mean_norm ~ 0 (denormal/zero).
  - A DIVERGED field       -> mean_norm huge / non-finite.
  - A STATIC-GAUGE field   -> nonzero Δx but ~constant in time (temporal_frac~0):
                              this is gauge garbage (a fixed offset), NOT motion.
  - A LIVE field           -> finite mean_norm AND temporal_frac substantial:
                              Δx genuinely changes with t = the target.

The key scalar is temporal_frac = Var_t / Var_total (ratio => rescaling-invariant).
cov_t (temporal coefficient of variation of ||Δx||) is a second invariant.

Frame-time is computed to MATCH training (T1.2 time_normalize): idx/N if on, else
raw idx — else the field is queried out-of-distribution and the read-out is invalid.

Usage:
  python diagnosis/infra/field_liveness.py \
    --config configs/Super/pb_oracle.yaml \
    --checkpoint output/pb_oracle/demo/checkpoint150.pt \
    --json out/liveness.json
"""

import argparse
import os
import sys
import json
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description='Rescaling-invariant deformation-field liveness probe')
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True, help='output json path')
    ap.add_argument('--n_points', type=int, default=4096, help='fixed canonical grid size')
    ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=device)
    model = JointEncoding(cfg, bound).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Total frame count N (the time-normalisation denominator must match training)
    pose = ckpt.get('pose')
    if isinstance(pose, dict):
        N = len(pose)
    elif isinstance(pose, (list, tuple)):
        N = len(pose)
    elif torch.is_tensor(pose):
        N = pose.shape[0]
    elif pose is not None:
        N = int(np.array(pose).shape[0])
    else:
        N = int(args.max_frames or 1)

    time_norm = cfg.get('training', {}).get('time_normalize', False)
    anchor_off = cfg.get('deformation_anchor_off', False)
    dynamic = cfg.get('dynamic', False)

    out = {
        'config': args.config, 'checkpoint': args.checkpoint, 'n_points': args.n_points,
        'N_frames_total': N, 'time_normalize': time_norm,
        'deformation_anchor_off': anchor_off, 'dynamic': dynamic,
    }

    if not dynamic:
        out['verdict'] = 'N/A (dynamic=False, field inactive, Δx≡0)'
        with open(args.json, 'w') as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return

    # FIXED canonical grid (same points for every frame -> per-point Var_t valid)
    rng = np.random.default_rng(args.seed)
    lo = bound[:, 0].cpu().numpy()
    hi = bound[:, 1].cpu().numpy()
    pts = rng.uniform(lo, hi, size=(args.n_points, 3)).astype(np.float32)
    pts_t = torch.from_numpy(pts).to(device)

    frames = list(range(0, N, args.frame_stride))
    if args.max_frames:
        frames = frames[:args.max_frames]
    out['n_frames_eval'] = len(frames)

    dx_all = np.zeros((len(frames), args.n_points, 3), dtype=np.float32)
    traj = []
    with torch.no_grad():
        for fi, idx in enumerate(frames):
            ft = (float(idx) / float(N)) if time_norm else float(idx)
            ft_t = torch.full((args.n_points, 1), ft, device=device)
            embed_time = model.embed_time(ft_t)
            embed_pos = model.embed_fre_pos(pts_t)
            h = torch.cat([embed_time, embed_pos], dim=-1)
            vox = model.time_net(h)
            if not anchor_off:
                vox = torch.where(ft_t == 0, torch.zeros_like(vox), vox)
            dx = vox.cpu().numpy().astype(np.float32)
            dx_all[fi] = dx
            traj.append(float(np.linalg.norm(dx, axis=1).mean()))

    norms = np.linalg.norm(dx_all, axis=2)  # (F, P)
    mean_norm = float(norms.mean())
    max_norm = float(norms.max())

    # variance decomposition (sum over xyz components)
    total_var = float(dx_all.reshape(-1, 3).var(axis=0).sum())     # over all (F,P)
    temporal_var = float(dx_all.var(axis=0).mean(axis=0).sum())    # var over F, mean over P
    spatial_var = float(dx_all.mean(axis=0).var(axis=0).sum())     # var over P of time-mean
    temporal_frac = float(temporal_var / total_var) if total_var > 0 else 0.0

    per_pt_tstd = norms.std(axis=0)
    per_pt_tmean = norms.mean(axis=0)
    cov_t = float(per_pt_tstd.mean() / (per_pt_tmean.mean() + 1e-30))
    spatial_std_of_meanmag = float(per_pt_tmean.std())  # is Δx spatially localised?

    if (not np.isfinite(mean_norm)) or mean_norm > 1e3:
        verdict = 'DIVERGED'
    elif mean_norm < 1e-6:
        verdict = 'DEAD (denormal/zero)'
    elif temporal_frac < 0.05:
        verdict = 'STATIC-GAUGE (nonzero Δx but ~constant in time)'
    else:
        verdict = 'LIVE (time-varying Δx)'

    out.update({
        'mean_norm': mean_norm, 'max_norm': max_norm,
        'total_var': total_var, 'temporal_var': temporal_var, 'spatial_var': spatial_var,
        'temporal_frac': temporal_frac, 'cov_t': cov_t,
        'spatial_std_of_meanmag': spatial_std_of_meanmag,
        'traj_mean_norm_per_frame': traj,
        'verdict': verdict,
    })
    with open(args.json, 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
