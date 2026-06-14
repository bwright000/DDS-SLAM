"""
Map-absorption probe (Battery-7 follow-up).

Question: with the camera POSE FROZEN, does the static map absorb the per-frame tissue motion
(leaving the deformation field nothing to do), OR is there a real photometric residual at the
moving tissue that a learnt-uncertainty head (Inc-1) could fire on and route to the field?

With pose frozen + a static scene network, the deformation-OFF render is (near-)constant across
frames — so it MUST be wrong wherever tissue actually moved. This probe measures, per checkpoint:
  - GT temporal variance per pixel  (= where tissue moves, data-driven; no labels needed)
  - deformation-OFF residual vs GT, per pixel
  - deformation-ON  residual vs GT, per pixel
and reports the decisive numbers:
  (A) signal-exists : is the OFF residual much larger where GT moves than where it's static,
      and does it correlate with GT motion?  -> there IS something for Inc-1 to fire on.
  (B) field-inert   : is ON ~= OFF (the field currently reduces nothing)?  (expected from Battery-7)
  (C) static-render : is the OFF render ~constant across frames at moving pixels?  (confirms the
      static map literally cannot track the motion -> the residual is genuine, not a fit artefact)

Decision: (A) strong  -> GO Inc-1 (the residual signal exists; routing can work).
          (A) weak     -> motion is sub-SNR OR the map already tracks it -> Inc-1 can't help here.

Run on Colab (needs CUDA/tcnn), from DDS-SLAM/, with the Battery-7 checkpoint:
  python diagnosis/infra/map_absorption_probe.py \
    --config configs/Super/pb7_redesign_s2.yaml \
    --checkpoint output/pb7_redesign_s2/demo/checkpoint150.pt \
    --json map_absorb_s2.json --max_frames 40 --frame_stride 3
Rays use the OpenGL convention (matches the model).
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
    acc = []
    with torch.no_grad():
        for s in range(0, fo.shape[0], rbs):
            e = min(s + rbs, fo.shape[0]); r = model.render_rays(fo[s:e], fdir[s:e])
            acc.append(r['rgb'].cpu())
    return np.clip(torch.cat(acc).reshape(H, W, 3).numpy(), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max_frames', type=int, default=40); ap.add_argument('--frame_stride', type=int, default=3)
    ap.add_argument('--ray_batch_size', type=int, default=2048); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--motion_pct', type=float, default=90.0, help='percentile of GT temporal var = "moving" pixels')
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
    frames = list(range(0, N, args.frame_stride))[:args.max_frames]

    GT, OFF, ON = [], [], []
    for idx in frames:
        gt = cv2.imread(imgf[idx])
        if gt is None: continue
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB); gt = cv2.resize(gt, (W, H)).astype(np.float32) / 255.0
        model.config['deformation_off'] = True
        off = _render(model, est[idx], H, W, fx, fy, cx, cy, idx, args.ray_batch_size, dev)
        model.config['deformation_off'] = False
        on = _render(model, est[idx], H, W, fx, fy, cx, cy, idx, args.ray_batch_size, dev)
        GT.append(gt); OFF.append(off); ON.append(on)
    GT = np.stack(GT); OFF = np.stack(OFF); ON = np.stack(ON)               # [F,H,W,3]

    gt_var = GT.var(0).mean(-1)                                             # [H,W] where tissue moves
    resid_off = ((OFF - GT) ** 2).mean(-1)                                  # [F,H,W]
    resid_on = ((ON - GT) ** 2).mean(-1)
    moff, mon = resid_off.mean(0), resid_on.mean(0)                         # [H,W]
    off_tstd = OFF.std(0).mean(-1)                                          # [H,W] static-render temporal wobble

    thr = np.percentile(gt_var, args.motion_pct)
    move = gt_var >= thr; stat = gt_var < np.percentile(gt_var, 50)
    def psnr(r): return float(-10 * np.log10(r.mean() + 1e-12))

    # (A) signal-exists, (B) field-inert, (C) static-render
    resid_off_move = float(moff[move].mean()); resid_off_stat = float(moff[stat].mean())
    ratio = resid_off_move / (resid_off_stat + 1e-12)
    flat_r, flat_v = moff.flatten(), gt_var.flatten()
    pear = float(np.corrcoef(flat_r, flat_v)[0, 1])
    out = {
        'config': args.config, 'checkpoint': args.checkpoint, 'n_frames': len(GT),
        'psnr_off_whole': psnr(resid_off), 'psnr_on_whole': psnr(resid_on),
        'psnr_off_moving': psnr(resid_off[:, move]), 'psnr_on_moving': psnr(resid_on[:, move]),
        # (A) is the OFF residual concentrated where GT moves?
        'A_resid_off_moving': resid_off_move, 'A_resid_off_static': resid_off_stat,
        'A_moving_over_static_ratio': round(ratio, 3),
        'A_pearson_residoff_vs_gtvar': round(pear, 3),
        'A_gt_var_moving_mean': float(gt_var[move].mean()),
        # (B) does the field reduce the residual at all? (ON vs OFF)
        'B_psnr_on_minus_off_whole': round(psnr(resid_on) - psnr(resid_off), 4),
        'B_psnr_on_minus_off_moving': round(psnr(resid_on[:, move]) - psnr(resid_off[:, move]), 4),
        'B_resid_reduction_moving': round(resid_off_move - float(mon[move].mean()), 8),
        # (C) is the OFF render ~constant across frames at moving pixels? (static map can't track)
        'C_off_temporal_std_moving': float(off_tstd[move].mean()),
        'C_off_temporal_std_static': float(off_tstd[stat].mean()),
    }
    # verdict
    signal = (ratio > 2.0) and (pear > 0.2)
    inert = abs(out['B_psnr_on_minus_off_moving']) < 0.3
    out['VERDICT_signal_exists'] = bool(signal)
    out['VERDICT_field_inert'] = bool(inert)
    out['VERDICT'] = (
        'GO Inc-1: residual signal EXISTS at moving tissue (static render fails there) -> routing can work'
        if signal else
        'NO signal: OFF residual not concentrated at GT motion -> motion sub-SNR OR map tracks it -> Inc-1 cannot help here')
    out['note'] = ('A_*=does the still-render fail where tissue moves (the signal Inc-1 needs). '
                   'B_*=does the field currently reduce it (expected ~0 = inert). '
                   'C_*=is the still-render constant at moving pixels (confirms the residual is genuine).')
    with open(args.json, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
