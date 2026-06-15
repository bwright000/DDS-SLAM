"""
Verify the gradient-attribution probe is CORRECT — i.e. that "field gets ~0 gradient
(signal-not-reaching-field)" is a real property of the model, not a probe artifact.

Four controls on one deformation frame (surface-concentrated sampling, same as grad_attribution_probe):
  1. CONNECTIVITY  — # time_net params with grad is None after backward (None != 0; a disconnect
                     would otherwise masquerade as "0").
  2. EFFECT        — render the SAME rays with the field ON vs OFF (deformation_off). If the rendered
                     pixels barely change, the field genuinely has ~no effect -> ~0 grad is correct.
                     Also reports the TRUE surface-render PSNR (deciding whether ~6 dB was the model
                     or the off-surface sampling of the residual probes).
  3. FINITE-DIFF   — perturb K random time_net params by +/-eps, central-difference the numerical
                     gradient, compare to the analytical one. FD ~= analytical => the probe computes
                     the gradient correctly. FD nonzero but analytical ~0 => probe BUG.
  4. POSITIVE CTRL — the same measurement on a FRESH random-init model (field demonstrably active)
                     must show a LARGE field-grad => the probe CAN detect field-grad when present.

PASS (reading verified) iff: trained field-grad ~= FD (gradient computed correctly) AND fresh-control
field-grad >> trained (instrument works). Whether that grad is ~0 or not is then the real measurement.

Usage (Colab; needs CUDA/tcnn):
  python diagnosis/infra/verify_grad_probe.py --config <cfg> --checkpoint <ckpt> --json verify.json \
      [--n_rays 2048 --fd_params 40 --eps 0.01]
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


def _field_gradnorm(model):
    sq = 0.0; n_none = 0; n_tot = 0
    for n, p in model.named_parameters():
        if 'time_net' not in n:
            continue
        n_tot += 1
        if p.grad is None:
            n_none += 1; continue
        sq += float(p.grad.detach().double().pow(2).sum().item())
    return sq ** 0.5, n_none, n_tot


def _map_gradnorm(model):
    sq = 0.0
    for n, p in model.named_parameters():
        if 'time_net' in n or p.grad is None:
            continue
        sq += float(p.grad.detach().double().pow(2).sum().item())
    return sq ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--n_rays', type=int, default=2048); ap.add_argument('--fd_params', type=int, default=40)
    ap.add_argument('--eps', type=float, default=1e-2); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
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
    dynamic = bool(cfg.get('dynamic', False)); time_norm = bool(cfg['training'].get('time_normalize', False))
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=dev)
    ds = get_dataset(cfg); H, W = int(ds.H), int(ds.W)
    nf = float(getattr(ds, 'num_frames', len(ds.img_files)))
    N = min(len(ds.img_files), _tensorize(torch.load(args.checkpoint, map_location='cpu')['pose']).shape[0])

    def build_model(load):
        torch.manual_seed(args.seed)
        m = JointEncoding(cfg, bound).to(dev)
        if load:
            ck = torch.load(args.checkpoint, map_location=dev); m.load_state_dict(ck['model'])
        m.eval(); m.config['deformation_off'] = False
        for p in m.parameters(): p.requires_grad_(True)
        return m

    est = _tensorize(torch.load(args.checkpoint, map_location=dev)['pose']).to(dev)
    rng = np.random.default_rng(args.seed)

    # pick the highest-motion deformation frame (excl. 0), build surface-concentrated rays + target_d
    scan = list(range(1, N, 3))
    prev = ds[0]['rgb'].reshape(-1, 3).numpy(); best = (scan[0], -1.0)
    for f in scan:
        g = ds[f]['rgb'].reshape(-1, 3).numpy()
        m_ = float(np.abs(g - prev).mean())
        if m_ > best[1]: best = (f, m_)
    idx = best[0]
    d = ds[idx]
    rgb_f = d['rgb'].reshape(-1, 3).to(dev); depth_f = d['depth'].reshape(-1).to(dev)
    dir_cam = d['direction'].reshape(-1, 3).to(dev); c2w = est[idx]
    ts = (float(idx) / nf) if time_norm else float(idx)
    valid = torch.nonzero(depth_f > 0, as_tuple=False).reshape(-1)
    pick = valid[torch.from_numpy(rng.integers(0, valid.numel(), size=min(args.n_rays, valid.numel()))).to(dev)]
    rd = dir_cam[pick] @ c2w[:3, :3].T
    ro = c2w[:3, 3].expand(rd.shape)
    fo = torch.cat([ro, torch.full(ro.shape[:-1] + (1,), float(ts), device=dev)], -1) if dynamic else ro
    td = depth_f[pick].unsqueeze(-1); tgt = rgb_f[pick]

    def loss_of(model):
        r = model.render_rays(fo, rd, target_d=td)
        return ((r['rgb'] - tgt) ** 2).mean(), r['rgb']

    out = {'config': args.config, 'checkpoint': args.checkpoint, 'frame': int(idx), 'n_rays': int(pick.numel()),
           'dynamic': dynamic, 'time_normalize': time_norm}

    model = build_model(load=True)
    # ---- analytical grad + connectivity ----
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        loss, rgb_on = loss_of(model)
        loss.backward()
    field_an, n_none, n_tot = _field_gradnorm(model)
    map_an = _map_gradnorm(model)
    psnr_on = float(-10 * np.log10(loss.item() + 1e-12))
    out.update({'1_connectivity_timenet_params': n_tot, '1_timenet_grad_None': n_none,
                'analytical_field_gradnorm': field_an, 'map_gradnorm': map_an,
                'RATIO_field_over_map': round(field_an / (map_an + 1e-30), 8)})

    # ---- (2) EFFECT: field ON vs OFF on the same rays + true surface PSNR ----
    with torch.no_grad():
        model.config['deformation_off'] = True
        _, rgb_off = loss_of(model)
        model.config['deformation_off'] = False
        on_off_delta = float((rgb_on.detach() - rgb_off).abs().mean().item())
        psnr_off = float(-10 * np.log10(((rgb_off - tgt) ** 2).mean().item() + 1e-12))
    out.update({'2_field_on_off_render_delta': on_off_delta,
                '2_surface_psnr_field_on': round(psnr_on, 3), '2_surface_psnr_field_off': round(psnr_off, 3)})

    # ---- (3) FINITE-DIFFERENCE on K random time_net params ----
    tn = [(n, p) for n, p in model.named_parameters() if 'time_net' in n]
    flat_idx = []
    for n, p in tn:
        for _ in range(max(1, args.fd_params // max(1, len(tn)))):
            flat_idx.append((n, p, tuple(int(rng.integers(0, s)) for s in p.shape)))
    fd_vals, an_vals = [], []
    for n, p, ix in flat_idx[:args.fd_params]:
        g_an = float(p.grad[ix].item()) if p.grad is not None else 0.0
        orig = float(p.data[ix].item())
        with torch.no_grad():
            p.data[ix] = orig + args.eps; lp, _ = loss_of(model)
            p.data[ix] = orig - args.eps; lm, _ = loss_of(model)
            p.data[ix] = orig
        g_fd = float((lp.item() - lm.item()) / (2 * args.eps))
        fd_vals.append(g_fd); an_vals.append(g_an)
    fd = np.array(fd_vals); an = np.array(an_vals)
    cos = float(np.dot(fd, an) / (np.linalg.norm(fd) * np.linalg.norm(an) + 1e-30))
    out.update({'3_fd_n': len(fd), '3_fd_max_abs': float(np.abs(fd).max()),
                '3_analytical_max_abs': float(np.abs(an).max()),
                '3_fd_vs_analytical_cosine': round(cos, 4),
                '3_fd_vs_analytical_normratio': round(float(np.linalg.norm(fd) / (np.linalg.norm(an) + 1e-30)), 4)})

    # ---- (4) POSITIVE CONTROL: fresh random-init model (field active) ----
    fresh = build_model(load=False)
    fresh.zero_grad(set_to_none=True)
    with torch.enable_grad():
        lf, _ = loss_of(fresh); lf.backward()
    fresh_field, _, _ = _field_gradnorm(fresh)
    out['4_fresh_init_field_gradnorm'] = fresh_field
    out['4_fresh_over_trained_field_ratio'] = round(fresh_field / (field_an + 1e-30), 2)

    # ---- VERDICT ----
    fd_ok = (cos > 0.8) or (out['3_fd_max_abs'] < 1e-4 and out['3_analytical_max_abs'] < 1e-4)  # agree, or both ~0
    detects = fresh_field > 100 * (field_an + 1e-30)                                            # probe sees a live field
    field_starved = out['RATIO_field_over_map'] < 0.01
    if fd_ok and detects:
        out['VERDICT'] = ('VERIFIED: probe computes the gradient correctly (FD agrees) AND detects a live '
                          'field (fresh control). So field/map ratio ' + str(out['RATIO_field_over_map']) +
                          (' with field render-effect %.2g IS REAL' % on_off_delta) +
                          (' -> signal-not-reaching-field confirmed.' if field_starved else ' (field NOT starved here).'))
    elif not fd_ok:
        out['VERDICT'] = ('PROBE SUSPECT: finite-difference does NOT match the analytical gradient '
                          '(cos=%.3f) -> the gradient reading cannot be trusted; debug the probe.' % cos)
    else:
        out['VERDICT'] = ('INCONCLUSIVE: fresh-init control did not show a clearly larger field-grad '
                          '(fresh=%.3g vs trained=%.3g) -> cannot confirm the instrument detects field-grad.'
                          % (fresh_field, field_an))
    with open(args.json, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
