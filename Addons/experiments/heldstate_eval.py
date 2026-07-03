#!/usr/bin/env python3
"""HELD-STATE oracle eval -- "from the FINAL map, does a correct deformation field render better?"

The online oracle A/B (2026-07-03) judged renders under the paper's RECENCY protocol (each frame
rendered right after being fit), where the static map's continuous re-fit absorbs deformation and a
correct field came out NEGATIVE (-0.24 dB paired). This eval removes the recency mask: load each
arm's FINAL checkpoint (all 151 frames trained) and re-render EVERY frame t from that one frozen
state -- the static map must now blur across the whole pin envelope (+/-27px), the oracle warps back
to each frame's true state. This is the only protocol where multi-state reconstruction can show up.

Decision rule (pre-registered): the oracle must beat static on MOVING-region PSNR (motion-localised,
>0.3 dB) AND not lose global PSNR -> a deformation DOF has reconstruction value on this data (build
the per-KF/graph representation). Otherwise the field's value case is tracking/correspondence only.

Arms (faithful to the online run -- both ckpts' time_net is collapsed-dead, verified 5.6e-39/3.4e-25):
  static : deformation_off (dx = 0), the dead-field map queried as-is
  oracle : the baked teacher dx* injected via the oracle_dx seam, exactly as during its training

  python Addons/experiments/heldstate_eval.py --arm static --ckpt <static.pt> --out output/heldstate/static
  python Addons/experiments/heldstate_eval.py --arm oracle --ckpt <oracle.pt> --out output/heldstate/oracle
  python Addons/experiments/heldstate_eval.py --compare output/heldstate/static output/heldstate/oracle
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle_field_trainer import build, frame_cache, make_rays, sample_grid, masked_eval  # noqa: E402


def render_heldstate(a):
    import cv2
    cfg, ds, model, dev = build(a.config, a.arm, a.deform_scale)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    sd = ck.get('model', ck)
    model.load_state_dict(sd)          # strict: arch identical across arms (deform_oracle = training flag only)
    model.eval()                       # forward early-returns rend_dict -> pure render, no loss graph
    print(f"[{a.arm}] loaded {a.ckpt} ({len(sd)} tensors)")

    H, W = ds.H, ds.W
    frames, dirs = frame_cache(ds, a.arm, a.deform_scale, n_limit=(4 if a.smoke else None))
    normalize_time = bool(cfg['training'].get('time_normalize', False))
    num_frames = ds.num_frames if hasattr(ds, 'num_frames') else len(ds)   # FULL count even in smoke
    os.makedirs(a.out, exist_ok=True)

    hh_full, ww_full = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    hh_full, ww_full = hh_full.reshape(-1), ww_full.reshape(-1)
    rows = []
    with torch.no_grad():
        for fr in range(len(frames)):
            out = torch.zeros(H * W, 3)
            dl1_num, dl1_den = 0.0, 0
            for s in range(0, H * W, a.chunk):
                hh, ww = hh_full[s:s + a.chunk], ww_full[s:s + a.chunk]
                rays_o, rays_d, tgt_rgb, tgt_d, tgt_e = make_rays(frames, dirs, fr, hh, ww, num_frames, dev, normalize_time)
                odx = sample_grid(frames[fr]['dx'], hh, ww, H, W).to(dev) if a.arm == 'oracle' else None
                ret = model.forward(rays_o, rays_d, tgt_rgb, tgt_d, target_edge_semantic=tgt_e, oracle_dx=odx)
                out[s:s + a.chunk] = ret['rgb'].detach().cpu().float()
                valid = tgt_d.squeeze(-1) > 0
                if valid.any():
                    dl1_num += float((ret['depth'].reshape(-1)[valid] - tgt_d.squeeze(-1)[valid]).abs().sum())
                    dl1_den += int(valid.sum())
            img = out.reshape(H, W, 3).clamp(0, 1)
            mse = float(((img - frames[fr]['rgb']) ** 2).mean())
            psnr = -10 * np.log10(max(mse, 1e-12))
            dl1 = dl1_num / max(dl1_den, 1)
            rows.append((frames[fr]['fid'], psnr, dl1))
            cv2.imwrite(os.path.join(a.out, f"{fr:04d}.jpg"),
                        cv2.cvtColor((img.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if fr % 25 == 0 or fr == len(frames) - 1:
                print(f"[heldstate:{a.arm}] frame {fr}  PSNR {psnr:.2f}  depthL1 {dl1:.4f}", flush=True)

    rows = np.array(rows)
    np.savez(os.path.join(a.out, 'perframe.npz'), fid=rows[:, 0], psnr=rows[:, 1], depth_l1=rows[:, 2])
    with open(os.path.join(a.out, 'heldstate_summary.txt'), 'w') as f:
        f.write(f"arm {a.arm}  ckpt {a.ckpt}  deform_scale {a.deform_scale}\n"
                f"HELD-STATE PSNR mean {rows[:, 1].mean():.3f} (std {rows[:, 1].std():.3f}, n={len(rows)})\n"
                f"HELD-STATE depth-L1 mean {rows[:, 2].mean():.5f} (model units, valid px)\n")
    print(open(os.path.join(a.out, 'heldstate_summary.txt')).read())


def compare(a):
    s = np.load(os.path.join(a.static_dir, 'perframe.npz'))
    o = np.load(os.path.join(a.oracle_dir, 'perframe.npz'))
    n = min(len(s['psnr']), len(o['psnr']))
    dp = o['psnr'][:n] - s['psnr'][:n]
    dd = o['depth_l1'][:n] - s['depth_l1'][:n]
    print("=" * 70)
    print(f"HELD-STATE PAIRED A/B ({n} frames, same frozen final map per arm)")
    print(f"  PSNR      static {s['psnr'][:n].mean():7.3f}   oracle {o['psnr'][:n].mean():7.3f}   "
          f"delta {dp.mean():+.3f} (median {np.median(dp):+.3f}); oracle better on {(dp > 0).sum()}/{n}")
    print(f"  depth-L1  static {s['depth_l1'][:n].mean():7.5f}   oracle {o['depth_l1'][:n].mean():7.5f}   "
          f"delta {dd.mean():+.5f} (negative = oracle better)")
    worst = np.argsort(dp)
    print(f"  oracle worst frames: {[int(s['fid'][i]) for i in worst[:5]]}   "
          f"best: {[int(s['fid'][i]) for i in worst[::-1][:5]]}")
    print("=" * 70)
    # motion-localised cut: moving vs static pixels by |dx*| (the decisive split)
    masked_eval(a)
    if dp.mean() > 0.1:
        msg = 'field has reconstruction value; check the moving-region split above for localisation'
    else:
        msg = ('even without recency masking the correct field does not pay: '
               'field value case = tracking/correspondence only')
    print(f"[verdict-hint] held-state global delta {dp.mean():+.3f} dB -> {msg}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/Super/trail3_teacher_off.yaml')
    ap.add_argument('--arm', choices=['static', 'oracle'], default=None)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--deform_scale', type=float, default=1.0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--compare', nargs=2, metavar=('STATIC_DIR', 'ORACLE_DIR'), default=None)
    ap.add_argument('--gt_dir', default='data/Super/trail_3/rgb')
    ap.add_argument('--deform_dir', default='data/Super/trail_3/deform')
    ap.add_argument('--move_thresh', type=float, default=0.02)
    a = ap.parse_args()
    if a.compare:
        a.static_dir, a.oracle_dir = a.compare
        compare(a)
    else:
        assert a.arm and a.ckpt and a.out, "--arm, --ckpt and --out required for rendering"
        render_heldstate(a)
