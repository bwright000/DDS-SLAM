#!/usr/bin/env python3
"""ORACLE-FIELD A/B (option A) -- "given a CORRECT deformation field, does the RENDER improve?"

Offline map trainer, NO SLAM: same model (JointEncoding), identity/frozen poses (SemSup trail_3 GT ==
identity, the bake's gauge), all frames trained jointly for --iters ray-batches. Two arms, same seed
(identical ray-sampling sequence):
  static : deformation_off (the dead-field baseline the SLAM effectively runs)
  oracle : vox_motion = the baked teacher dx* (validated +86.2%/cos .935 vs the held-out pins),
           injected per-ray via the oracle_dx seam (run_network bypasses time_net; constant along the
           ray = surface-attached approximation, fine where the render mass sits).
Judge: PSNR/SSIM/LPIPS over all frames (Addons/eval/eval_rendering.py on the dumped renders) + the
pin-patch panel (--compare): side-by-side GT/static/oracle crops at the largest-envelope pins, where
the static map MUST blur (trial_3 envelope: median +/-27px) and a correct field should not.

Known hazard (from the teacher campaign): raw |dx*| reaches ~0.084 =~ 84% of the render band ->
over-warps the SDF gates. --deform_scale (default 1.0) is the knob; |dx*| stats vs trunc printed.

  python Addons/experiments/oracle_field_trainer.py --config configs/Super/trail3_teacher_off.yaml \
      --arm static|oracle --out output/oracle_ab/<arm> [--iters 20000] [--smoke]
  python Addons/experiments/oracle_field_trainer.py --compare output/oracle_ab/static output/oracle_ab/oracle \
      --config ... --panel_out output/oracle_ab/pin_panel.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def sample_grid(grid, hh, ww, H, W):
    """bilinear-sample a compact [gh,gw,C] grid at pixel coords (rows hh, cols ww) -> [N,C].
    Replicates ddsslam.sample_dino_grid exactly (align_corners=True, full-FOV grid)."""
    gh, gw, C = grid.shape
    g = grid.float().permute(2, 0, 1).unsqueeze(0)
    gy = (torch.as_tensor(hh, dtype=torch.float32) / max(H - 1, 1)) * 2 - 1
    gx = (torch.as_tensor(ww, dtype=torch.float32) / max(W - 1, 1)) * 2 - 1
    coords = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)
    out = F.grid_sample(g, coords, mode='bilinear', align_corners=True)
    return out.squeeze(0).squeeze(-1).permute(1, 0).contiguous()


def build(cfg_path, arm, deform_scale):
    import config as cfgmod
    from datasets.dataset import get_dataset
    from model.scene_rep import JointEncoding
    cfg = cfgmod.load_config(cfg_path)
    cfg['deformation_off'] = True                       # BOTH arms bypass time_net (oracle injects dx directly)
    cfg['training']['deformation_sup_weight'] = 1.0     # forces the dataset to attach the baked dx* grids
    cfg['data']['output'] = cfg['data'].get('output', 'output/oracle_ab')
    ds = get_dataset(cfg)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bound = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(dev)
    model = JointEncoding(cfg, bound).to(dev)
    return cfg, ds, model, dev


def frame_cache(ds, arm, deform_scale, n_limit=None):
    """Pull every frame's rgb/depth/edge/c2w (+ dx grid for the oracle arm) into RAM once."""
    n = len(ds) if n_limit is None else min(n_limit, len(ds))
    frames = []
    for i in range(n):
        it = ds[i]
        f = dict(rgb=it['rgb'].float(), depth=it['depth'].float(),
                 edge=it['edge_semantic'].float(), c2w=it['c2w'].float(),
                 fid=int(it['frame_id']))
        if arm == 'oracle':
            assert 'deform_dx' in it, "deform_dx missing -- stage the baked dx* npz (deform/ subdir)"
            f['dx'] = it['deform_dx'].float() * deform_scale     # [gh,gw,3]
        frames.append(f)
    dirs = ds[0]['direction'].float()                            # shared across frames (fixed intrinsics)
    return frames, dirs


def make_rays(frames, dirs, fr, hh, ww, num_frames, dev, normalize_time):
    f = frames[fr]
    c2w = f['c2w'].to(dev)
    rd_cam = dirs[hh, ww].to(dev)
    rays_d = torch.sum(rd_cam[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, 3].expand(rays_d.shape[0], 3)
    t = float(f['fid']) / num_frames if normalize_time else float(f['fid'])
    rays_o = torch.cat([rays_o, torch.full((rays_d.shape[0], 1), t, device=dev)], dim=1)
    tgt_rgb = f['rgb'][hh, ww].to(dev)
    tgt_d = f['depth'][hh, ww].to(dev).unsqueeze(-1)
    tgt_e = f['edge'][hh, ww].to(dev).unsqueeze(-1)
    return rays_o, rays_d, tgt_rgb, tgt_d, tgt_e


def loss_from_ret(cfg, ret):
    tr = cfg['training']
    loss = tr['rgb_weight'] * ret['rgb_loss'] + tr['depth_weight'] * ret['depth_loss'] \
        + tr['sdf_weight'] * ret['sdf_loss'] + tr['fs_weight'] * ret['fs_loss'] \
        + tr['rgb_weight'] * 0.1 * ret['edge_semantic_loss']
    return loss


def train(a):
    cfg, ds, model, dev = build(a.config, a.arm, a.deform_scale)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0); np.random.seed(0)
    H, W = ds.H, ds.W
    frames, dirs = frame_cache(ds, a.arm, a.deform_scale, n_limit=(4 if a.smoke else None))
    nfr = len(frames)
    normalize_time = bool(cfg['training'].get('time_normalize', False))
    num_frames = ds.num_frames if hasattr(ds, 'num_frames') else len(ds)
    if a.arm == 'oracle':
        mags = torch.stack([f['dx'].norm(dim=-1).max() for f in frames])
        print(f"[oracle] |dx*| max per frame: median {mags.median():.4f} max {mags.max():.4f} "
              f"(scale {a.deform_scale}) vs trunc {cfg['training']['trunc']}")

    # optimizer mirrors create_optimizer's BA groups, minus time_net (irrelevant in both arms)
    dec = [p for n, p in model.decoder.named_parameters() if 'time_net' not in n]
    groups = [{'params': dec, 'weight_decay': 1e-6, 'lr': cfg['mapping']['lr_decoder']},
              {'params': model.embed_fn.parameters(), 'eps': 1e-15, 'lr': cfg['mapping']['lr_embed']}]
    if not cfg['grid']['oneGrid']:
        groups.append({'params': model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': cfg['mapping']['lr_embed_color']})
    opt = torch.optim.Adam(groups, betas=(0.9, 0.99))

    iters = 60 if a.smoke else a.iters
    g = torch.Generator().manual_seed(0)
    for it in range(iters):
        fr = int(torch.randint(0, nfr, (1,), generator=g))
        hh = torch.randint(0, H, (a.batch,), generator=g)
        ww = torch.randint(0, W, (a.batch,), generator=g)
        rays_o, rays_d, tgt_rgb, tgt_d, tgt_e = make_rays(frames, dirs, fr, hh, ww, num_frames, dev, normalize_time)
        odx = sample_grid(frames[fr]['dx'], hh, ww, H, W).to(dev) if a.arm == 'oracle' else None
        ret = model.forward(rays_o, rays_d, tgt_rgb, tgt_d, target_edge_semantic=tgt_e, oracle_dx=odx)
        loss = loss_from_ret(cfg, ret)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 500 == 0 or it == iters - 1:
            print(f"[{a.arm}] it {it:6d}/{iters}  loss {float(loss):.4f}  rgb {float(ret['rgb_loss']):.4f} "
                  f"depth {float(ret['depth_loss']):.4f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(a.out, 'model.pt'))
    render_all(a, cfg, model, frames, dirs, dev, num_frames, normalize_time)


def render_all(a, cfg, model, frames, dirs, dev, num_frames, normalize_time):
    import cv2
    H, W = frames[0]['rgb'].shape[:2]
    os.makedirs(a.out, exist_ok=True)
    psnrs = []
    hh_full, ww_full = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    hh_full, ww_full = hh_full.reshape(-1), ww_full.reshape(-1)
    with torch.no_grad():
        for fr in range(len(frames)):
            out = torch.zeros(H * W, 3)
            for s in range(0, H * W, a.chunk):
                hh, ww = hh_full[s:s + a.chunk], ww_full[s:s + a.chunk]
                rays_o, rays_d, tgt_rgb, tgt_d, tgt_e = make_rays(frames, dirs, fr, hh, ww, num_frames, dev, normalize_time)
                odx = sample_grid(frames[fr]['dx'], hh, ww, H, W).to(dev) if a.arm == 'oracle' else None
                ret = model.forward(rays_o, rays_d, tgt_rgb, tgt_d, target_edge_semantic=tgt_e,
                                    render_only=True, oracle_dx=odx)
                out[s:s + a.chunk] = ret['rgb'].detach().cpu().float()
            img = out.reshape(H, W, 3).clamp(0, 1)
            mse = float(((img - frames[fr]['rgb']) ** 2).mean())
            psnrs.append(-10 * np.log10(max(mse, 1e-12)))
            cv2.imwrite(os.path.join(a.out, f"{fr:04d}.jpg"),
                        cv2.cvtColor((img.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            if fr % 25 == 0:
                print(f"[render:{a.arm}] frame {fr}  PSNR {psnrs[-1]:.2f}", flush=True)
    print(f"[{a.arm}] TRAIN-VIEW PSNR mean {np.mean(psnrs):.3f} (n={len(psnrs)})  -> {a.out}")


def masked_eval(a):
    """Decisive cut: PSNR over MOVING pixels (|dx*| >= thresh) vs STATIC pixels, per arm, from the
    existing renders. Global PSNR dilutes the field's effect (moving tissue = a fraction of the frame);
    this is where a correct field must win if job-1 (sharp fusion under motion) is real."""
    import cv2
    gts = sorted(glob.glob(os.path.join(a.gt_dir, '*left.png')))
    dxs = sorted(glob.glob(os.path.join(a.deform_dir, '*_deform.npz')))
    assert len(gts) == len(dxs), f"gt({len(gts)}) != deform({len(dxs)})"
    acc = {arm: {'mov': [], 'sta': []} for arm in ('static', 'oracle')}
    mov_frac = []
    for i, (gp, dp) in enumerate(zip(gts, dxs)):
        gt = cv2.imread(gp).astype(np.float32) / 255.0
        H, W = gt.shape[:2]
        dx = np.load(dp)['dx'].astype(np.float32)                     # [gh,gw,3]
        mag = torch.from_numpy(np.linalg.norm(dx, axis=-1))
        m = F.interpolate(mag[None, None], size=(H, W), mode='bilinear', align_corners=True)[0, 0].numpy()
        mov = m >= a.move_thresh
        mov_frac.append(float(mov.mean()))
        for arm, d in (('static', a.static_dir), ('oracle', a.oracle_dir)):
            r = cv2.imread(os.path.join(d, f"{i:04d}.jpg"))
            if r is None: continue
            r = r.astype(np.float32) / 255.0
            se = ((r - gt) ** 2).mean(axis=-1)
            for key, msk in (('mov', mov), ('sta', ~mov)):
                if msk.sum() > 100:
                    acc[arm][key].append(-10 * np.log10(max(float(se[msk].mean()), 1e-12)))
    print(f"[masked_eval] move_thresh={a.move_thresh} (|dx*| world units)  moving-pixel fraction: "
          f"median {np.median(mov_frac):.3f}")
    for arm in ('static', 'oracle'):
        print(f"  {arm:7s}  MOVING-region PSNR {np.mean(acc[arm]['mov']):.3f}   "
              f"STATIC-region PSNR {np.mean(acc[arm]['sta']):.3f}   (n={len(acc[arm]['mov'])})")
    dm = np.mean(acc['oracle']['mov']) - np.mean(acc['static']['mov'])
    dstat = np.mean(acc['oracle']['sta']) - np.mean(acc['static']['sta'])
    print(f"  ORACLE-vs-STATIC: moving {dm:+.3f} dB | static {dstat:+.3f} dB "
          f"-> {'FIELD PAYS RENT in moving regions' if dm > 3 * abs(dstat) and dm > 0.3 else 'gain NOT localised to motion -- inspect'}")


def compare(a):
    """Pin-patch panel: GT vs static vs oracle crops at the largest-envelope pins."""
    import cv2
    pins = np.load(os.path.join(REPO, 'Addons/eval/gt_pins/trial_3_l_pts.npy'), allow_pickle=True).item()['gt']
    fkeys = sorted(pins.keys()); P = np.stack([np.asarray(pins[k]) for k in fkeys]).astype(float)
    xy, v = P[..., :2], P[..., 2] > 0
    stats = []
    for k in range(xy.shape[1]):
        m = v[:, k]
        if m.sum() < 20: continue
        p = xy[m, k]; c = p.mean(0); r = np.linalg.norm(p - c, axis=1)
        stats.append((r.max(), int(np.argmax(np.where(m, np.linalg.norm(xy[:, k] - c, axis=1), -1))), k, c))
    stats.sort(reverse=True)
    rows = []
    S = 64
    for env, fmax, k, c in stats[:a.n_pins]:
        x, y = xy[fmax, k].astype(int)
        gt = cv2.imread(sorted(glob.glob(os.path.join(a.gt_dir, '*left.png')))[fmax])
        st = cv2.imread(os.path.join(a.static_dir, f"{fmax:04d}.jpg"))
        orc = cv2.imread(os.path.join(a.oracle_dir, f"{fmax:04d}.jpg"))
        crops = []
        for im in (gt, st, orc):
            Hh, Ww = im.shape[:2]
            y0, x0 = np.clip(y - S, 0, Hh - 2 * S), np.clip(x - S, 0, Ww - 2 * S)
            crops.append(im[y0:y0 + 2 * S, x0:x0 + 2 * S])
        row = np.concatenate(crops, axis=1)
        cv2.putText(row, f"pin{k} f{fmax} env{env:.0f}px  GT | static | oracle", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        rows.append(row)
    panel = np.concatenate(rows, axis=0)
    cv2.imwrite(a.panel_out, panel)
    print(f"[compare] wrote {a.panel_out} ({len(rows)} pins)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/Super/trail3_teacher_off.yaml')
    ap.add_argument('--arm', choices=['static', 'oracle'], default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--iters', type=int, default=20000)
    ap.add_argument('--batch', type=int, default=2048)
    ap.add_argument('--chunk', type=int, default=8192)
    ap.add_argument('--deform_scale', type=float, default=1.0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--compare', nargs=2, metavar=('STATIC_DIR', 'ORACLE_DIR'), default=None)
    ap.add_argument('--masked', nargs=2, metavar=('STATIC_DIR', 'ORACLE_DIR'), default=None)
    ap.add_argument('--gt_dir', default='data/Super/trail_3/rgb')
    ap.add_argument('--deform_dir', default='data/Super/trail_3/deform')
    ap.add_argument('--move_thresh', type=float, default=0.02, help='|dx*| (world units) above which a pixel counts as MOVING')
    ap.add_argument('--panel_out', default='output/oracle_ab/pin_panel.png')
    ap.add_argument('--n_pins', type=int, default=6)
    a = ap.parse_args()
    if a.masked:
        a.static_dir, a.oracle_dir = a.masked
        masked_eval(a)
    elif a.compare:
        a.static_dir, a.oracle_dir = a.compare
        compare(a)
    else:
        assert a.arm and a.out, "--arm and --out required for training"
        train(a)
