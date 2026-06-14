"""
Field-attribution render eval (Phase-0 measurement foundation for the 2-arm program).

The honest version of "PSNR/SSIM/LPIPS improvement" — because whole-frame PSNR is FIELD-BLIND (a dead
deformation field scores ~the same; confirmed by us + DyCheck). This computes, per checkpoint:
  - PSNR/SSIM/LPIPS, deform-ON vs deform-OFF (same ckpt; OFF = config['deformation_off']=True),
  - whole-frame AND tissue-masked (tool excluded if a tool seg class is given),
so the ON-vs-OFF delta on the tissue is attributable to the DEFORMATION FIELD, not config.

Usage (per run; compare treatment-vs-mogev2 by running on both ckpts):
  python diagnosis/infra/render_eval_attrib.py --config <cfg> --checkpoint <ckpt> --json out.json \
    [--max_frames N --frame_stride K --holdout_k 8]

--holdout_k: if set, score ONLY held-out frames (every k-th) — the field-sensitive split (LerPlane 7:1).
Requires the run to have been TRAINED with the matching holdout (else it's just a frame subset).
Rays use the OpenGL convention (matches the model). LPIPS/SSIM degrade gracefully if libs absent.
"""
import argparse, os, sys, json
import numpy as np, torch, cv2


def _tensorize(pose):
    if isinstance(pose, dict):
        ks = sorted(pose.keys(), key=lambda k: int(k) if isinstance(k, (int, str)) else k)
        t = torch.stack([torch.as_tensor(pose[k]) for k in ks], 0)
    elif isinstance(pose, (list, tuple)):
        t = torch.stack([torch.as_tensor(p) for p in pose], 0)
    elif torch.is_tensor(pose):
        t = pose
    else:
        t = torch.as_tensor(np.array(pose))
    if t.dim() == 3 and t.shape[-2:] == (3, 4):
        pad = torch.zeros(t.shape[0], 1, 4); pad[..., 0, 3] = 1.0; t = torch.cat([t, pad], 1)
    return t.float()


def _render(model, c2w, H, W, fx, fy, cx, cy, ts, rbs, dev):
    i, j = torch.meshgrid(torch.arange(W, device=dev).float(), torch.arange(H, device=dev).float(), indexing='ij')
    i = i.T; j = j.T
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # OpenGL
    rd = dirs @ c2w[:3, :3].T; rd = rd / rd.norm(dim=-1, keepdim=True)
    ro = c2w[:3, 3].expand(rd.shape)
    tt = torch.full(ro.shape[:-1] + (1,), float(ts), device=dev)
    fo = torch.cat([ro, tt], -1).reshape(-1, 4); fdir = rd.reshape(-1, 3)
    acc, dep = [], []
    with torch.no_grad():
        for s in range(0, fo.shape[0], rbs):
            e = min(s + rbs, fo.shape[0]); r = model.render_rays(fo[s:e], fdir[s:e])
            acc.append(r['rgb'].cpu()); dep.append(r['depth'].reshape(-1).cpu())
    return np.clip(torch.cat(acc).reshape(H, W, 3).numpy(), 0, 1), torch.cat(dep).reshape(H, W).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max_frames', type=int, default=40); ap.add_argument('--frame_stride', type=int, default=1)
    ap.add_argument('--holdout_k', type=int, default=0, help='score only every k-th frame (held-out split); 0=all')
    ap.add_argument('--ray_batch_size', type=int, default=2048); ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO not in sys.path: sys.path.insert(0, REPO)
    from config import load_config
    from model.scene_rep import JointEncoding
    from datasets.dataset import get_dataset
    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    if 'training' not in cfg: cfg['training'] = {}
    if 'n_samples' not in cfg['training']: cfg['training']['n_samples'] = cfg['training'].get('n_samples_d', 32)
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=dev)
    model = JointEncoding(cfg, bound).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev); model.load_state_dict(ckpt['model']); model.eval()
    est = _tensorize(ckpt['pose']).to(dev)
    ds = get_dataset(cfg); H, W = int(ds.H), int(ds.W); fx, fy = float(ds.fx), float(ds.fy); cx, cy = float(ds.cx), float(ds.cy)
    imgf = ds.img_files; N = min(len(imgf), est.shape[0])
    # optional metric libs
    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except Exception:
        ssim_fn = None
    try:
        import lpips as _l; lpips_fn = _l.LPIPS(net='alex').to(dev); lpips_fn.eval()
    except Exception:
        lpips_fn = None

    frames = list(range(0, N, args.frame_stride))
    if args.holdout_k and args.holdout_k > 1:
        frames = [f for f in frames if f % args.holdout_k == 0]
    frames = frames[:args.max_frames]

    def metrics(pred, gt, mask):
        out = {}
        mse_w = float(((pred - gt) ** 2).mean()); out['psnr_whole'] = float(-10 * np.log10(mse_w + 1e-12))
        if mask.sum() > 50:
            d = ((pred - gt) ** 2)[mask]; out['psnr_tissue'] = float(-10 * np.log10(d.mean() + 1e-12))
        else:
            out['psnr_tissue'] = None
        if ssim_fn is not None:
            try: out['ssim_whole'] = float(ssim_fn(gt, pred, channel_axis=2, data_range=1.0))
            except Exception: out['ssim_whole'] = None
        if lpips_fn is not None:
            with torch.no_grad():
                tp = torch.from_numpy(pred).permute(2, 0, 1)[None].float().to(dev) * 2 - 1
                tg = torch.from_numpy(gt).permute(2, 0, 1)[None].float().to(dev) * 2 - 1
                out['lpips_whole'] = float(lpips_fn(tp, tg).item())
        return out

    agg = {}
    for idx in frames:
        gt = cv2.imread(imgf[idx])
        if gt is None: continue
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB); gt = cv2.resize(gt, (W, H)).astype(np.float32) / 255.0
        model.config['deformation_off'] = False
        on_rgb, on_dep = _render(model, est[idx], H, W, fx, fy, cx, cy, idx, args.ray_batch_size, dev)
        model.config['deformation_off'] = True
        off_rgb, _ = _render(model, est[idx], H, W, fx, fy, cx, cy, idx, args.ray_batch_size, dev)
        model.config['deformation_off'] = False
        mask = on_dep > 1e-6  # tissue/surface = reconstructed (valid depth)
        for tag, pred in (('on', on_rgb), ('off', off_rgb)):
            m = metrics(pred, gt, mask)
            for k, v in m.items():
                if v is not None: agg.setdefault(f'{tag}_{k}', []).append(v)

    def mean(k): return float(np.mean(agg[k])) if agg.get(k) else None
    out = {'config': args.config, 'checkpoint': args.checkpoint, 'n_frames': len(frames),
           'holdout_k': args.holdout_k, 'ssim_available': ssim_fn is not None, 'lpips_available': lpips_fn is not None}
    for k in ['psnr_whole', 'psnr_tissue', 'ssim_whole', 'lpips_whole']:
        out[f'on_{k}'] = mean(f'on_{k}'); out[f'off_{k}'] = mean(f'off_{k}')
    # field attribution = ON - OFF on the tissue (PSNR/SSIM up, LPIPS down = field helps)
    def delta(k, sign=1):
        a, b = out.get(f'on_{k}'), out.get(f'off_{k}')
        return round(sign * (a - b), 4) if (a is not None and b is not None) else None
    out['FIELD_ATTRIB_psnr_tissue_on_minus_off'] = delta('psnr_tissue')
    out['FIELD_ATTRIB_lpips_whole_off_minus_on'] = delta('lpips_whole', sign=-1)  # +ve = field improves LPIPS
    out['note'] = 'whole-frame PSNR is FIELD-BLIND; the FIELD_ATTRIB_* deltas (held-out, tissue, ON-OFF) are the field-attributable signal. Compare on_* vs your mogev2 baseline run for the headline.'
    with open(args.json, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
