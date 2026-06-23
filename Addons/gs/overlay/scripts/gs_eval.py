#!/usr/bin/env python3
"""Per-run GS eval+viz + (v1) HELD-OUT-DYNAMIC render metric.

From a finished run's params.npz it: (1) renders every frame at the est pose, (2) computes the 5 metrics
PSNR/SSIM/LPIPS/L1-Depth/Sim3-ATE (legacy metrics.txt block, byte-unchanged = the 'ALL frames' row),
(3) ships the canonical 6-panel video.

v1 additions (additive; default-off reproduces the legacy output):
  --holdout_every k : frames with (t>0 and t%k==k-1) were TRACKED but NOT mapped (set in main.py) -> the
                      clean held-out set. Metrics are partitioned by held-out membership.
  --dynamic_idx F   : freeze the DYNAMIC frame set from the BASE arm (F = base run's dynamic_idx.npy). If
                      absent, compute it from THIS run (and save) — but the runbook freezes from base.
  DYNAMIC label is the P99 of the 3D surface-change residual (consecutive frames, est poses, gt depth) —
  the depth-reproj signal v1's FLOW head never directly consumes (decoupled, not circular).
  -> writes metrics_split.json with the subset matrix; HEADLINE = HELD-OUT n DYNAMIC.

  python scripts/gs_eval.py --config configs/crcd/crcd_base.py --run experiments/CRCD_base/C2_001_flowmap_s0 \
         --holdout_every 5 --dynamic_idx experiments/CRCD_base/C2_001_base_s0/dynamic_idx.npy
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
import yaml
from natsort import natsorted
from importlib.machinery import SourceFileLoader

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE)


def _qwxyz_R(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def render_run(cfg, run):
    """Render every frame at the estimated pose -> run/eval/{color,depth,plots} (no SLAM re-run)."""
    from scripts.main import get_dataset
    from datasets.gradslam_datasets import load_dataset_config
    from utils.eval_helpers import eval_save
    dc = cfg['data']
    gcfg = load_dataset_config(dc['gradslam_data_cfg'])
    ds = get_dataset(config_dict=gcfg, basedir=dc['basedir'], sequence=os.path.basename(dc['sequence']),
                     start=dc['start'], end=dc['end'], stride=dc['stride'],
                     desired_height=dc['desired_image_height'], desired_width=dc['desired_image_width'],
                     device='cuda:0', relative_pose=True, train_or_test='all')
    d = np.load(os.path.join(run, 'params.npz'))
    fp = {k: torch.tensor(d[k]).cuda().float() for k in d.files}
    with torch.no_grad():   # torch 2.x: eval_save's plot imshows grad tensors -> render under no_grad
        eval_save(ds, fp, os.path.join(run, 'eval'), cfg['tracking']['sil_thres'],
                  cfg['mapping']['num_iters'], cfg['mapping']['add_new_gaussians'], save_renders=True)


def est_c2w(run):
    d = np.load(os.path.join(run, 'params.npz'))
    rots, trans = d['cam_unnorm_rots'], d['cam_trans']
    out = []
    for t in range(rots.shape[-1]):
        R = _qwxyz_R(rots[0, :, t].astype(np.float64))           # rel w2c rotation
        tr = trans[0, :, t].astype(np.float64)
        c2w = np.eye(4); c2w[:3, :3] = R.T; c2w[:3, 3] = -R.T @ tr   # w2c -> c2w
        out.append(c2w)
    return np.array(out)


def est_w2c(run):
    d = np.load(os.path.join(run, 'params.npz'))
    rots, trans = d['cam_unnorm_rots'], d['cam_trans']
    out = []
    for t in range(rots.shape[-1]):
        w2c = np.eye(4); w2c[:3, :3] = _qwxyz_R(rots[0, :, t].astype(np.float64))
        w2c[:3, 3] = trans[0, :, t].astype(np.float64)
        out.append(w2c)
    return np.array(out)


def gt_c2w(scene):
    rows = [list(map(float, l.split())) for l in open(os.path.join(scene, 'traj.txt'))
            if l.strip() and not l.startswith('#')]
    c2w = np.array(rows).reshape(-1, 4, 4)
    return np.linalg.inv(c2w[0])[None] @ c2w                      # frame-0-relative


def umeyama(src, dst):
    n = len(src); ms, md = src.mean(0), dst.mean(0); S = src - ms; D = dst - md
    U, Dv, Vt = np.linalg.svd((D.T @ S) / n); W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: W[-1, -1] = -1
    R = U @ W @ Vt; s = np.trace(np.diag(Dv) @ W) / ((S ** 2).sum() / n + 1e-12)
    return (s * (R @ src.T)).T + (md - s * R @ ms), s


def render_metrics(scene, run, gt_scale=10000.0, rd_scale=655.35):
    """-> n, per-frame arrays (P,S,L,Dl) [legacy means = nanmean(arr)]. Also writes depth_png for the video."""
    import lpips as _lp
    from pytorch_msssim import ssim as _ssim
    from PIL import Image
    lp = _lp.LPIPS(net='alex').cuda().eval()
    ed = os.path.join(run, 'eval')
    grgb = natsorted(glob.glob(f"{scene}/frames/*.jpg")) or natsorted(glob.glob(f"{scene}/video_frames/*l.png"))
    gdep = natsorted(glob.glob(f"{scene}/depths/*.png")) or natsorted(glob.glob(f"{scene}/depth/*.png"))
    rrgb = natsorted(glob.glob(f"{ed}/color/*.png"))
    rdep = natsorted(glob.glob(f"{ed}/depth/*.tiff"))
    n = min(len(grgb), len(rrgb))
    assert n > 0, f"no renders in {ed}/color (apply the visall=True patch + render first)"
    os.makedirs(f"{ed}/depth_png", exist_ok=True)
    P, S, L, Dl = [], [], [], []
    for i in range(n):
        g = cv2.cvtColor(cv2.imread(grgb[i]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        r = cv2.cvtColor(cv2.imread(rrgb[i]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        if r.shape[:2] != g.shape[:2]:
            r = cv2.resize(r, (g.shape[1], g.shape[0]))
        gd = cv2.imread(gdep[i], -1).astype(np.float32) / gt_scale if i < len(gdep) else None
        rd = np.array(Image.open(rdep[i])).astype(np.float32) / rd_scale if i < len(rdep) else None
        gt = torch.from_numpy(g).permute(2, 0, 1)[None].cuda()
        rt = torch.from_numpy(r).permute(2, 0, 1)[None].cuda()
        P.append(-10 * np.log10(((gt - rt) ** 2).mean().item() + 1e-12))
        S.append(_ssim(rt, gt, data_range=1.0).item())
        with torch.no_grad():
            L.append(lp(rt * 2 - 1, gt * 2 - 1).item())
        if gd is not None and rd is not None:
            if rd.shape[:2] != gd.shape[:2]:
                rd = cv2.resize(rd, (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(f"{ed}/depth_png/{i:06d}.png", np.clip(rd * gt_scale, 0, 65535).astype(np.uint16))
            m = gd > 0
            Dl.append(float(np.abs(rd[m] - gd[m]).mean()) if m.any() else np.nan)
        else:
            Dl.append(np.nan)
    return n, np.array(P), np.array(S), np.array(L), np.array(Dl)


def _K_from_cfg(cfg):
    g = yaml.safe_load(open(cfg['data']['gradslam_data_cfg'], encoding='utf-8'))['camera_params']
    return np.array([[g['fx'], 0, g['cx']], [0, g['fy'], g['cy']], [0, 0, 1]], np.float64)


def dynamic_label(scene, run, depth_deadband, gt_scale=10000.0):
    """Per-frame DYNAMIC bool from the 3D surface-change residual (consecutive frames, est poses, gt depth)
    — the depth-reproj signal v1's FLOW head never directly consumes (decoupled). dynamic[t] iff P99>db.
    Saves run/p99_per_frame.npy + run/dynamic_idx.npy."""
    from Addons.motion.gs_flow_gate import GSFlowGate
    g = GSFlowGate({}, 'cpu')                                     # enable=False -> no RAFT; reproj is pure numpy
    g.set_intrinsics(_K_cur)                                      # native-res K (set in main); reproj is res-agnostic
    w2c = est_w2c(run)
    gdep = natsorted(glob.glob(f"{scene}/depths/*.png")) or natsorted(glob.glob(f"{scene}/depth/*.png"))
    n = min(len(w2c), len(gdep))
    p99 = np.zeros(n, np.float32)
    for t in range(1, n):
        d_ref = cv2.imread(gdep[t - 1], -1).astype(np.float32) / gt_scale
        d_cur = cv2.imread(gdep[t], -1).astype(np.float32) / gt_scale
        resid, valid = g._depth_reproj_residual(d_ref, d_cur, w2c[t - 1], w2c[t])
        m = valid > 0
        p99[t] = float(np.percentile(resid[m], 99)) if m.any() else 0.0
    np.save(os.path.join(run, 'p99_per_frame.npy'), p99)
    dyn = p99 > depth_deadband
    np.save(os.path.join(run, 'dynamic_idx.npy'), np.flatnonzero(dyn))
    return dyn, p99


def _subset(mask, P, S, L, Dl):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return dict(PSNR=float('nan'), SSIM=float('nan'), LPIPS=float('nan'), L1depth_mm=float('nan'), n=0)
    return dict(PSNR=float(np.nanmean(P[idx])), SSIM=float(np.nanmean(S[idx])),
                LPIPS=float(np.nanmean(L[idx])), L1depth_mm=float(np.nanmean(Dl[idx]) * 1000), n=int(len(idx)))


def main():
    global _K_cur
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--genvideo', default='/content/DDS-SLAM/Addons/viz/generate_video.py')
    ap.add_argument('--skip_render', action='store_true')
    ap.add_argument('--drive_root', default='/content/drive/MyDrive/Outputs/GS_phase0')
    ap.add_argument('--holdout_every', type=int, default=0, help='k: t%%k==k-1 frames are held-out (tracked, not mapped)')
    ap.add_argument('--dynamic_idx', default='', help='freeze DYNAMIC set from this base-arm dynamic_idx.npy')
    ap.add_argument('--depth_deadband', type=float, default=2.0, help='3D surface-change deadband (metric, mm-scale)')
    a = ap.parse_args()
    cfg = SourceFileLoader('cfg', a.config).load_module().config
    scene = os.path.join(cfg['data']['basedir'], os.path.basename(cfg['data']['sequence']))
    _K_cur = _K_from_cfg(cfg)

    if not a.skip_render:
        render_run(cfg, a.run)

    estC, gtC = est_c2w(a.run), gt_c2w(scene) if os.path.exists(os.path.join(scene, 'traj.txt')) else None
    with open(os.path.join(a.run, 'est_c2w_data.txt'), 'w') as f:
        for c in estC:
            f.write(' '.join(f'{v:.8f}' for v in c[:3, :4].reshape(-1)) + '\n')
    if gtC is not None:
        with open(os.path.join(a.run, 'gt_xyz.txt'), 'w') as f:
            for c in gtC:
                x, y, z = c[:3, 3]; f.write(f'0 {x:.8f} {y:.8f} {z:.8f} 0 0 0 1\n')

    n, Pa, Sa, La, Dla = render_metrics(scene, a.run)
    # Sim3 ATE (whole trajectory; pose is global)
    ate = scale = float('nan')
    if gtC is not None:
        e, gg = estC[:, :3, 3], gtC[:, :3, 3]; k = min(len(e), len(gg))
        al, scale = umeyama(e[:k], gg[:k]); ate = float(np.linalg.norm(al - gg[:k], axis=1).mean())

    # --- v1: held-out + dynamic masks + the subset matrix ---
    k = a.holdout_every
    holdout = np.array([(k > 0 and t > 0 and t % k == k - 1) for t in range(n)])
    if a.dynamic_idx and os.path.exists(a.dynamic_idx):
        di = np.load(a.dynamic_idx); dyn = np.zeros(n, bool)
        dyn[di[di < n]] = True; dsrc = f"frozen<-{os.path.basename(os.path.dirname(a.dynamic_idx))}"
    else:
        dynf, _ = dynamic_label(scene, a.run, a.depth_deadband); dyn = dynf[:n]; dsrc = "self"
        if len(dyn) < n: dyn = np.concatenate([dyn, np.zeros(n - len(dyn), bool)])
    split = dict(
        all            = _subset(np.ones(n, bool), Pa, Sa, La, Dla),
        heldout_dynamic= _subset(holdout & dyn, Pa, Sa, La, Dla),     # HEADLINE
        heldout_static = _subset(holdout & ~dyn, Pa, Sa, La, Dla),    # GUARD
        heldout_all    = _subset(holdout, Pa, Sa, La, Dla),
        fitted_dynamic = _subset((~holdout) & dyn, Pa, Sa, La, Dla),  # memorization-prone
        ate_mm=ate * 1000 if ate == ate else float('nan'), sim3_scale=scale,
        holdout_every=k, depth_deadband=a.depth_deadband, dynamic_n=int(dyn.sum()),
        dynamic_src=dsrc, frames=n,
    )
    json.dump(split, open(os.path.join(a.run, 'metrics_split.json'), 'w'), indent=2)

    # legacy metrics.txt (the 'all' row, byte-compatible printed block)
    P, S, L, Dl = float(np.nanmean(Pa)), float(np.nanmean(Sa)), float(np.nanmean(La)), float(np.nanmean(Dla))
    print("=" * 64)
    print(f"  RUN {a.run}   ({n} frames)   dynamic={int(dyn.sum())} held-out={int(holdout.sum())} src={dsrc}")
    print("=" * 64)
    print(f"  ALL            PSNR {P:7.3f}  SSIM {S:.4f}  LPIPS {L:.4f}  L1d {Dl*1000:6.2f}mm")
    hd, hs = split['heldout_dynamic'], split['heldout_static']
    print(f"  HELD-OUT∩DYN   PSNR {hd['PSNR']:7.3f}  SSIM {hd['SSIM']:.4f}  LPIPS {hd['LPIPS']:.4f}   (n={hd['n']})  <- HEADLINE")
    print(f"  HELD-OUT∩STA   PSNR {hs['PSNR']:7.3f}  SSIM {hs['SSIM']:.4f}  LPIPS {hs['LPIPS']:.4f}   (n={hs['n']})  <- GUARD")
    print(f"  Sim3 ATE       {split['ate_mm']:8.3f} mm   (scale {scale:.3f})")
    print("=" * 64)
    with open(os.path.join(a.run, 'metrics.txt'), 'w') as f:
        f.write(f"frames {n}\nPSNR {P}\nSSIM {S}\nLPIPS {L}\nL1depth_mm {Dl*1000}\nATE_mm {ate*1000}\n")

    # canonical 6-panel video
    out = os.path.join(a.run, os.path.basename(a.run) + '_6panel.mp4')
    if gtC is not None:
        cmd = ['python', a.genvideo,
               '--rgb_input_dir', f"{scene}/frames", '--rgb_input_pattern', '*.jpg',
               '--rgb_output_dir', f"{a.run}/eval/color", '--rgb_output_pattern', '*.png',
               '--depth_input_dir', f"{scene}/depths", '--depth_output_dir', f"{a.run}/eval/depth_png",
               '--seg_dir', f"{scene}/semantic_ids", '--seg_pattern', '*.png', '--seg_classmap', '--skip_raw_seg',
               '--trajectory_est', f"{a.run}/est_c2w_data.txt", '--trajectory_gt', f"{a.run}/gt_xyz.txt",
               '--depth_norm', 'robust', '--png_depth_scale', '10000',
               '--output', out, '--fps', '15', '--panel_height', '360', '--panel_width', '480']
        print(f"  6-panel video -> {out}")
        subprocess.run(cmd, check=False)

    # ship the two diagnostic sets + the split json
    if os.path.isdir('/content/drive/MyDrive'):
        dst = os.path.join(a.drive_root, os.path.basename(a.run.rstrip('/')))
        os.makedirs(dst, exist_ok=True)
        for fn in ('metrics.txt', 'metrics_split.json', os.path.basename(out), 'est_c2w_data.txt', 'gt_xyz.txt'):
            src = os.path.join(a.run, fn)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"  shipped (metrics + split + 6-panel video) -> {dst}")
    else:
        print("  [drive] /content/drive/MyDrive not found -> NOT shipped")


if __name__ == '__main__':
    main()
