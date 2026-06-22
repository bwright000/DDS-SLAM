#!/usr/bin/env python3
"""Per-run GS eval+viz (the standing two-diagnostic-sets deliverable for every GS CRCD run).

From a finished run's params.npz it: (1) renders every frame at the estimated pose (EndoGSLAM eval_save,
needs the visall=True patch), (2) computes the 5 metrics PSNR / SSIM / LPIPS / L1-Depth / Sim3-ATE,
(3) ships the CANONICAL 6-panel video via DDS-SLAM/Addons/viz/generate_video.py
(Input-RGB | Rendered-RGB | Input-Depth | Output-Depth | Seg-overlay | estTraj).

  cd /content/EndoGSLAM
  python scripts/gs_eval.py --config configs/crcd/crcd_base.py --run experiments/CRCD_base/C1_001_s0
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
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
    import lpips as _lp
    from pytorch_msssim import ssim as _ssim
    from PIL import Image
    lp = _lp.LPIPS(net='alex').cuda().eval()
    ed = os.path.join(run, 'eval')
    grgb = natsorted(glob.glob(f"{scene}/frames/*.jpg"))
    gdep = natsorted(glob.glob(f"{scene}/depths/*.png"))
    rrgb = natsorted(glob.glob(f"{ed}/color/*.png"))
    rdep = natsorted(glob.glob(f"{ed}/depth/*.tiff"))
    n = min(len(grgb), len(rrgb))
    assert n > 0, f"no renders in {ed}/color (apply the visall=True patch + render first)"
    os.makedirs(f"{ed}/depth_png", exist_ok=True)                 # 16-bit PNG for generate_video's *.png glob
    P, S, L, Dl = [], [], [], []
    for i in range(n):
        g = cv2.cvtColor(cv2.imread(grgb[i]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        r = cv2.cvtColor(cv2.imread(rrgb[i]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        if r.shape[:2] != g.shape[:2]:
            r = cv2.resize(r, (g.shape[1], g.shape[0]))
        gd = cv2.imread(gdep[i], -1).astype(np.float32) / gt_scale
        rd = np.array(Image.open(rdep[i])).astype(np.float32) / rd_scale if i < len(rdep) else np.zeros_like(gd)
        if rd.shape[:2] != gd.shape[:2]:
            rd = cv2.resize(rd, (gd.shape[1], gd.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{ed}/depth_png/{i:06d}.png", np.clip(rd * gt_scale, 0, 65535).astype(np.uint16))
        m = gd > 0
        gt = torch.from_numpy(g).permute(2, 0, 1)[None].cuda()
        rt = torch.from_numpy(r).permute(2, 0, 1)[None].cuda()
        P.append(-10 * np.log10(((gt - rt) ** 2).mean().item() + 1e-12))
        S.append(_ssim(rt, gt, data_range=1.0).item())
        with torch.no_grad():
            L.append(lp(rt * 2 - 1, gt * 2 - 1).item())
        Dl.append(float(np.abs(rd[m] - gd[m]).mean()) if m.any() else np.nan)
    return n, float(np.nanmean(P)), float(np.nanmean(S)), float(np.nanmean(L)), float(np.nanmean(Dl))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--genvideo', default='/content/DDS-SLAM/Addons/viz/generate_video.py')
    ap.add_argument('--skip_render', action='store_true', help='renders already on disk -> metrics+video only')
    ap.add_argument('--drive_root', default='/content/drive/MyDrive/Outputs/GS_phase0',
                    help='auto-ship metrics+video here (per standing two-diagnostic-sets rule)')
    a = ap.parse_args()
    cfg = SourceFileLoader('cfg', a.config).load_module().config
    scene = os.path.join(cfg['data']['basedir'], os.path.basename(cfg['data']['sequence']))

    if not a.skip_render:
        render_run(cfg, a.run)

    # trajectory files for the video (est: 12-val 3x4 c2w/line; gt: 8-val xyz/line)
    estC, gtC = est_c2w(a.run), gt_c2w(scene)
    with open(os.path.join(a.run, 'est_c2w_data.txt'), 'w') as f:
        for c in estC:
            f.write(' '.join(f'{v:.8f}' for v in c[:3, :4].reshape(-1)) + '\n')
    with open(os.path.join(a.run, 'gt_xyz.txt'), 'w') as f:
        for c in gtC:
            x, y, z = c[:3, 3]; f.write(f'0 {x:.8f} {y:.8f} {z:.8f} 0 0 0 1\n')

    # metrics
    n, P, S, L, Dl = render_metrics(scene, a.run)
    e, g = estC[:, :3, 3], gtC[:, :3, 3]
    k = min(len(e), len(g)); al, s = umeyama(e[:k], g[:k])
    ate = float(np.linalg.norm(al - g[:k], axis=1).mean())
    print("=" * 60)
    print(f"  RUN {a.run}   ({n} frames)")
    print("=" * 60)
    print(f"  PSNR      : {P:8.3f} dB")
    print(f"  SSIM      : {S:8.4f}")
    print(f"  LPIPS     : {L:8.4f}")
    print(f"  L1-Depth  : {Dl * 1000:8.2f} mm (up-to-scale)")
    print(f"  Sim3 ATE  : {ate * 1000:8.3f} mm   (scale {s:.3f})")
    print("=" * 60)
    print("  ref SGS c1_001: PSNR 22.6 / SSIM 0.81 / LPIPS 0.31 / DepthL1 23.6mm / ATE 3.31mm")
    with open(os.path.join(a.run, 'metrics.txt'), 'w') as f:
        f.write(f"frames {n}\nPSNR {P}\nSSIM {S}\nLPIPS {L}\nL1depth_mm {Dl*1000}\nATE_mm {ate*1000}\n")

    # canonical 6-panel video
    out = os.path.join(a.run, os.path.basename(a.run) + '_6panel.mp4')
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

    # Auto-ship the two diagnostic sets (metrics + 6-panel video) to Drive (standing rule).
    if os.path.isdir('/content/drive/MyDrive'):
        dst = os.path.join(a.drive_root, os.path.basename(a.run.rstrip('/')))
        os.makedirs(dst, exist_ok=True)
        for fn in ('metrics.txt', os.path.basename(out), 'est_c2w_data.txt', 'gt_xyz.txt'):
            src = os.path.join(a.run, fn)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"  shipped (metrics + 6-panel video) -> {dst}")
    else:
        print("  [drive] /content/drive/MyDrive not found -> NOT shipped (mount Drive to auto-ship)")


if __name__ == '__main__':
    main()
