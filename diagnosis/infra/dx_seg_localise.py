"""
T1.3 GATE METRIC — is the field's deformation localised to where the RIGID model FAILS?

Anti-circular: correlate the RAW-field ||Delta_x|| against the deform-OFF photometric RESIDUAL
(|render_off - GT|), which is INDEPENDENT of the gate's seg input. Seg-gate run vs baseline:
if seg-gating raises Pearson(||Delta_x||, residual), the gate concentrates deformation where the
rigid/canonical model genuinely can't explain the image = the gate HELPS (not just "is different").
Also report Pearson(||Delta_x||, seg-prior) [corroborating] + concentration ratio.

Per frame (OpenGL rays = matches the model; NOT the old OpenCV bug):
  1. render deform-OFF -> rgb_off, depth_off ; residual = mean_ch |rgb_off - GT|
  2. surface pt = cam + ray_dir_norm * depth_off ; eval field ||Delta_x|| at (surface, t) [RAW, no gate]
  3. seg edge-prior = compute_edge_semantic(seg img)
  4. valid = depth>0 ; report Pearson(||dx||,residual), Pearson(||dx||,seg),
     concentration = mean||dx|| in top-residual-decile / mean||dx|| in bottom-decile

Compare seg-gate vs baseline on Pearson(||dx||,residual): higher = gate routes deformation to where
the rigid model fails. GO (screen) = seg-gate clearly > baseline.

Usage:
  python diagnosis/infra/dx_seg_localise.py --config <cfg> --checkpoint <ckpt> --json out.json \
    --max_frames 30 --frame_stride 5
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--max_frames', type=int, default=30); ap.add_argument('--frame_stride', type=int, default=5)
    ap.add_argument('--ray_batch_size', type=int, default=2048); ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO not in sys.path: sys.path.insert(0, REPO)
    from config import load_config
    from model.scene_rep import JointEncoding
    from datasets.dataset import get_dataset, compute_edge_semantic

    dev = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    cfg['deformation_off'] = True  # render the CANONICAL (rigid) scene for the residual
    if 'training' not in cfg: cfg['training'] = {}
    if 'n_samples' not in cfg['training']:
        cfg['training']['n_samples'] = cfg['training'].get('n_samples_d', 32)

    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32, device=dev)
    model = JointEncoding(cfg, bound).to(dev)
    ckpt = torch.load(args.checkpoint, map_location=dev); model.load_state_dict(ckpt['model']); model.eval()
    est = _tensorize(ckpt['pose']).to(dev)

    ds = get_dataset(cfg)
    H, W = int(ds.H), int(ds.W); fx, fy = float(ds.fx), float(ds.fy); cx, cy = float(ds.cx), float(ds.cy)
    imgf, segf = ds.img_files, ds.semantic_paths
    N = min(len(imgf), len(segf), est.shape[0])
    time_norm = cfg.get('training', {}).get('time_normalize', False)
    anchor_off = cfg.get('deformation_anchor_off', False)
    hb = cfg.get('deform_hardbound', 0)

    frames = list(range(0, N, args.frame_stride))[:args.max_frames]
    pear_res, pear_seg, conc, mags = [], [], [], []
    dummy = np.zeros((H, W), np.float32)

    for idx in frames:
        c2w = est[idx]; ft = (idx / N) if time_norm else float(idx)
        # OpenGL rays (match get_camera_rays)
        i, j = torch.meshgrid(torch.arange(W, device=dev).float(), torch.arange(H, device=dev).float(), indexing='ij')
        i = i.T; j = j.T
        dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)
        rd = dirs @ c2w[:3, :3].T; rd = rd / rd.norm(dim=-1, keepdim=True)
        ro = c2w[:3, 3].expand(rd.shape)
        ts = torch.full(ro.shape[:-1] + (1,), float(idx), device=dev)  # render uses deformation_off so t unused
        fo = torch.cat([ro, ts], -1).reshape(-1, 4); fdir = rd.reshape(-1, 3)
        rgb_acc, dep_acc = [], []
        with torch.no_grad():
            for s in range(0, fo.shape[0], args.ray_batch_size):
                e = min(s + args.ray_batch_size, fo.shape[0])
                ret = model.render_rays(fo[s:e], fdir[s:e])
                rgb_acc.append(ret['rgb'].cpu()); dep_acc.append(ret['depth'].reshape(-1).cpu())
        rgb_off = np.clip(torch.cat(rgb_acc).reshape(H, W, 3).numpy(), 0, 1)
        depth = torch.cat(dep_acc).reshape(H, W).numpy()

        gt = cv2.imread(imgf[idx])
        if gt is None: continue
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB); gt = cv2.resize(gt, (W, H)).astype(np.float32) / 255.0
        residual = np.abs(rgb_off - gt).mean(2)
        seg = cv2.resize(cv2.imread(segf[idx]), (W, H))
        edge = compute_edge_semantic(seg, dummy).astype(np.float32)

        # surface points = cam + ray_dir * depth ; eval RAW field ||dx|| there (no gate)
        surf = (ro + rd * torch.from_numpy(depth).to(dev).unsqueeze(-1)).reshape(-1, 3)
        with torch.no_grad():
            ft_t = torch.full((surf.shape[0], 1), ft, device=dev)
            h = torch.cat([model.embed_time(ft_t), model.embed_fre_pos(surf)], -1)
            vox = model.time_net(h)
            if hb and hb > 0: vox = hb * torch.tanh(vox / hb)
            if not anchor_off: vox = torch.where(ft_t == 0, torch.zeros_like(vox), vox)
        dxmag = torch.norm(vox, dim=1).reshape(H, W).cpu().numpy()

        valid = (depth > 1e-6)
        if valid.sum() < 100: continue
        r = residual[valid]; dm = dxmag[valid]; eg = edge[valid]
        def pear(a, b):
            if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
            return float(np.corrcoef(a, b)[0, 1])
        pear_res.append(pear(dm, r)); pear_seg.append(pear(dm, eg)); mags.append(float(dm.mean()))
        hi = r >= np.percentile(r, 90); lo = r <= np.percentile(r, 10)
        if dm[lo].mean() > 1e-9: conc.append(float(dm[hi].mean() / (dm[lo].mean() + 1e-12)))

    def m(x): return float(np.mean(x)) if len(x) else None
    out = {'config': args.config, 'checkpoint': args.checkpoint, 'n_frames': len(pear_res),
           'pearson_dx_vs_residual': m(pear_res),   # PRIMARY (anti-circular): deformation where rigid fails
           'pearson_dx_vs_segprior': m(pear_seg),    # corroborating
           'concentration_hiRes_over_loRes': m(conc),  # >1 = dx concentrated where rigid fails
           'mean_surface_dx': m(mags)}
    out['verdict'] = ('GATE-LOCALISES (dx tracks rigid-failure)' if (m(pear_res) or 0) > 0.15
                      else 'WEAK/NO localisation vs rigid-failure')
    with open(args.json, 'w') as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
