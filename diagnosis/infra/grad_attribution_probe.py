"""
Gradient-attribution probe — "when deformation happens, WHO gets updated?"

Battery-7 + the architecture read SAY the map absorbs the deformation gradient and starves the
field — but we INFERRED that. This MEASURES it: at a trained checkpoint, on the moving-tissue
pixels of deformation frames, do a grad-ENABLED render -> photometric loss -> backward, and read
the gradient-norm landing on each component:
    HASH (embed_fn)  |  SDF (sdf_net)  |  COLOR (color_net)  |  FIELD (time_net)
Report per-component grad-norm + the FIELD/MAP ratio (raw, and effective = grad x lr), so we can
see who is actually asked to explain the deformation.
  field grad ~0          -> "signal-not-reaching-field" (plumbing/anchor/time-coord, not a race)
  map >> field           -> "map-absorbs-gradient"      -> confirms THROTTLE-MAP
  field gets a fair share-> "race"/"shared"             -> deadness is the optimisation, not the signal

CORRECTNESS (verified vs live code + adversarially checked):
  * render_rays called DIRECTLY (scene_rep.py:325) under torch.enable_grad — NOT model.forward()
    (returns early w/o loss when not training), NOT under no_grad (the sibling _render helpers use
    no_grad — copying that line silently zeros ALL grads).
  * rays + target_d built EXACTLY like training (ddsslam.py:786-809): rays_d = direction @ c2w_R,
    NOT normalized; target_d = batch['depth'] so z-samples concentrate at the SURFACE (range_d band).
    target_d=None / uniform near..far would attenuate the FIELD bucket and FAKE a 'map-absorbs' verdict.
  * timestamp = idx / ds.num_frames (if time_normalize) — matches training (ddsslam.py:572); frame 0
    excluded (the canonical anchor forces Dx=0 there -> trivially zero field grad).
  * components via model.named_parameters() substrings; model.{sdf_net,color_net,time_net} are batchify
    CLOSURES with 0 params, so one pass + substring buckets = clean, non-duplicated (matches _dec_groups).

Usage (Colab; from DDS-SLAM/):
  python diagnosis/infra/grad_attribution_probe.py --config <cfg> --checkpoint <ckpt> --json out.json \
      [--max_frames 12 --frame_stride 3 --n_rays 4096]
"""
import argparse, os, sys, json
import numpy as np, torch


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


def _component(name):
    if name.startswith('embed_fn_color'): return 'hash_color'   # color hash (oneGrid=False) -> MAP
    if name.startswith('embed_fn'): return 'hash'               # main hash grid -> MAP
    if 'time_net' in name: return 'field'                       # TimeNet -> the FIELD
    if 'sdf_net' in name: return 'sdf'                          # MAP
    if 'color_net' in name: return 'color'                      # MAP
    if 'edgenet_semantic' in name: return 'edge'               # reported, out of map/field race
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max_frames', type=int, default=12); ap.add_argument('--frame_stride', type=int, default=3)
    ap.add_argument('--n_rays', type=int, default=4096); ap.add_argument('--ray_batch_size', type=int, default=4096)
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--seed', type=int, default=0)
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
    dynamic = bool(cfg.get('dynamic', False))
    time_norm = bool(cfg['training'].get('time_normalize', False))
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=dev)
    model = JointEncoding(cfg, bound).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev); model.load_state_dict(ckpt['model']); model.eval()
    model.config['deformation_off'] = False
    for p in model.parameters(): p.requires_grad_(True)
    est = _tensorize(ckpt['pose']).to(dev)
    ds = get_dataset(cfg)
    H, W = int(ds.H), int(ds.W)
    nf = float(getattr(ds, 'num_frames', len(ds.img_files)))
    N = min(len(ds.img_files), est.shape[0])
    crop_warn = (cfg['cam'].get('crop_edge', 0) != 0 or cfg['data'].get('downsample', 1) != 1)

    comps = ['hash', 'hash_color', 'sdf', 'color', 'field', 'edge']
    pcounts = {c: 0 for c in comps}
    for n, p in model.named_parameters():
        c = _component(n)
        if c is not None: pcounts[c] += p.numel()

    out = {'config': args.config, 'checkpoint': args.checkpoint, 'dynamic': dynamic,
           'time_normalize': time_norm, 'num_frames': nf, 'n_rays': args.n_rays, 'crop_warn': crop_warn}
    if not dynamic:
        out['VERDICT_tag'] = 'N/A'
        out['VERDICT'] = 'N/A (dynamic=False -> run_network skips the field; time_net never in the graph).'
        with open(args.json, 'w') as f: json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2)); return

    rng = np.random.default_rng(args.seed)

    # rank frames by GT temporal motion (deformation proxy), exclude frame 0 (anchor)
    scan = list(range(0, N, max(1, args.frame_stride)))
    prev = None; motion = []
    for idx in scan:
        g = ds[idx]['rgb'].reshape(-1, 3).numpy()
        if prev is not None: motion.append((idx, float(np.abs(g - prev).mean())))
        prev = g
    motion.sort(key=lambda kv: kv[1], reverse=True)
    def_frames = [i for i, _ in motion if i != 0][:args.max_frames] or [i for i in scan if i != 0][:args.max_frames]
    out['def_frames'] = [int(i) for i in def_frames]

    sq = {c: 0.0 for c in comps}; per_frame = []; used = 0
    for idx in def_frames:
        d = ds[idx]
        rgb_f = d['rgb'].reshape(-1, 3).to(dev)
        depth_f = d['depth'].reshape(-1).to(dev)
        dir_cam = d['direction'].reshape(-1, 3).to(dev)               # model-convention camera rays
        c2w = est[idx]
        ts = (float(idx) / nf) if time_norm else float(idx)
        valid = torch.nonzero(depth_f > 0, as_tuple=False).reshape(-1)  # surface-concentrated sampling needs depth>0
        if valid.numel() < 64: continue
        pick = valid[torch.from_numpy(rng.integers(0, valid.numel(), size=min(args.n_rays, valid.numel()))).to(dev)]
        rd = dir_cam[pick] @ c2w[:3, :3].T                            # UN-normalized -> matches training units
        ro = c2w[:3, 3].expand(rd.shape)
        fo = torch.cat([ro, torch.full(ro.shape[:-1] + (1,), float(ts), device=dev)], -1)  # [R,4] carries timestamp
        td = depth_f[pick].unsqueeze(-1)                              # real surface depth -> depth-guided z-samples
        tgt = rgb_f[pick]
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            racc, tacc = [], []
            for s in range(0, fo.shape[0], args.ray_batch_size):
                e = min(s + args.ray_batch_size, fo.shape[0])
                r = model.render_rays(fo[s:e], rd[s:e], target_d=td[s:e])
                racc.append(r['rgb']); tacc.append(tgt[s:e])
            loss = ((torch.cat(racc, 0) - torch.cat(tacc, 0)) ** 2).mean()
        if not torch.isfinite(loss): continue
        loss.backward()
        fsq = {c: 0.0 for c in comps}
        for n, p in model.named_parameters():
            c = _component(n)
            if c is None or p.grad is None: continue
            g = p.grad.detach().float()
            if not torch.isfinite(g).all(): g = torch.nan_to_num(g)
            v = float((g * g).sum().item()); sq[c] += v; fsq[c] += v
        per_frame.append({'frame': int(idx), 'rgb_loss': float(loss.item()),
                          **{f'gn_{c}': float(fsq[c] ** 0.5) for c in comps}})
        used += 1

    out['n_def_frames_used'] = used
    if used == 0:
        out['VERDICT_tag'] = 'ERROR'; out['VERDICT'] = 'no usable frames (depth/loss invalid)'
        with open(args.json, 'w') as f: json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2)); return

    norm = {c: float(sq[c] ** 0.5) for c in comps}
    field = norm['field']
    map_raw = float((sq['hash'] + sq['hash_color'] + sq['sdf'] + sq['color']) ** 0.5)
    lr_e = float(cfg['mapping'].get('lr_embed', 0.01)); lr_d = float(cfg['mapping'].get('lr_decoder', 0.01))
    lr_m = float(cfg['training'].get('timenet_lr_mult', 1.0))
    field_eff = field * lr_d * lr_m
    map_eff = norm['hash'] * lr_e + (norm['sdf'] + norm['color'] + norm['hash_color']) * lr_d
    ratio_raw = field / (map_raw + 1e-30)
    ratio_eff = field_eff / (map_eff + 1e-30)

    if field < 1e-9 * (map_raw + 1e-30) or ratio_raw < 0.01:
        tag = 'signal-not-reaching-field'
        v = 'FIELD GRAD ~0: photometric error is not reaching TimeNet (plumbing/anchor/time-coord), not a race.'
    elif ratio_eff < 0.05:
        tag = 'map-absorbs-gradient'
        v = 'MAP ABSORBS THE GRADIENT: map moves >>20x the field per step -> wins the race, stores blurry average -> THROTTLE-MAP justified.'
    elif ratio_eff < 0.5:
        tag = 'map-favoured'
        v = 'MAP-FAVOURED: map moves more per step than the field -> throttle-map / slow-map likely helps.'
    else:
        tag = 'race-or-shared'
        v = 'FIELD COMPETITIVE: field gets a fair share of the effective step -> deadness is the optimisation/ill-posedness, not the signal.'

    out.update({
        'gradnorm': {c: round(norm[c], 8) for c in comps}, 'param_counts': pcounts,
        'FIELD_gradnorm': round(field, 8), 'MAP_gradnorm': round(map_raw, 8),
        'RATIO_field_over_map_raw': round(ratio_raw, 6),
        'RATIO_field_over_map_effective': round(ratio_eff, 6),
        'lr_embed': lr_e, 'lr_decoder': lr_d, 'timenet_lr_mult': lr_m,
        'per_frame': per_frame, 'VERDICT_tag': tag, 'VERDICT': v,
        'note': 'surface-concentrated (real target_d) on moving-tissue deformation frames; effective ratio = grad x lr = who moves more per step; <0.05 = field starved.',
    })
    with open(args.json, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
