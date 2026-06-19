"""POST-HOC geo_feat visualiser — "what the geofuse/georgbd sigma^2 head sees."

Loads a TRAINED geofuse/georgbd checkpoint and re-renders a few frames; the model's
render_rays now returns the per-ray GEOMETRY feature (geo_feat, the jitter-killer the
fusion adds). PCA-to-RGB it so you can SEE the geometry feature spatially, beside the
input rgb, the DINO feature (the other sigma^2 input), and the sigma^2 it drives.

No re-run / no training dump needed — it reconstructs geo_feat from the checkpoint.
Run on the LIVE runtime after the run (the checkpoint lives in output/.../demo, ephemeral),
or ship the checkpoint to Drive first.

  python Addons/viz/geo_feat_viz.py \
    --config configs/CRCD/c1_001_canon_uncert_dino_reg_rgbdepth_geofuse.yaml \
    --checkpoint output/geofuse_crcd_s0/demo/checkpoint*.pt \
    --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png' \
    --dino_dir data/CRCD/C1_001/dino_reg \
    --uncert_dir output/geofuse_crcd_s0/uncert \
    --output output/geofuse_geo_feat_viz --stride 30
"""
import os, sys, glob, argparse, numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import torch, cv2


def pca_rgb(feat, basis=None):
    """feat [H,W,C] -> [H,W,3] uint8 via top-3 PCA (shared basis if given)."""
    H, W, C = feat.shape
    X = feat.reshape(-1, C).astype(np.float32)
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    if basis is None:
        # top-3 right singular vectors
        try:
            _, _, Vt = np.linalg.svd(Xc[:: max(1, len(Xc) // 20000)], full_matrices=False)
        except np.linalg.LinAlgError:
            Vt = np.eye(C, dtype=np.float32)
        basis = (mu, Vt[:3])
    mu0, V = basis
    Y = (X - mu0) @ V.T               # [N,3]
    lo, hi = np.percentile(Y, 2, 0), np.percentile(Y, 98, 0)
    Y = np.clip((Y - lo) / (hi - lo + 1e-8), 0, 1)
    return (Y.reshape(H, W, 3) * 255).astype(np.uint8), basis


def _label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--rgb_dir', default=''); ap.add_argument('--rgb_glob', default='*l.png')
    ap.add_argument('--dino_dir', default=''); ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--uncert_dir', default='')
    ap.add_argument('--stride', type=int, default=30)
    ap.add_argument('--ray_batch_size', type=int, default=4096)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    ckpt_path = sorted(glob.glob(args.checkpoint))[-1] if any(c in args.checkpoint for c in '*?[') else args.checkpoint
    REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    cfg.setdefault('training', {}).setdefault('n_samples', cfg['training'].get('n_samples_d', 32))
    cam = cfg['cam']; H, W = int(cam['H']), int(cam['W'])
    fx, fy, cx, cy = float(cam['fx']), float(cam['fy']), float(cam['cx']), float(cam['cy'])

    bbox = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bbox).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model']); model.eval()
    if not getattr(model.decoder, '_surface_geo', False):
        print("WARN: model is not surfacing geo_feat (config fuse != geo/geo_rgbd?) — geo_feat panel will be blank.")

    pose = ckpt['pose']
    if isinstance(pose, dict):
        pose = torch.stack([torch.as_tensor(pose[k]) for k in sorted(pose, key=lambda k: int(k))], 0)
    elif isinstance(pose, list):
        pose = torch.stack([torch.as_tensor(p) for p in pose], 0)
    pose = torch.as_tensor(pose).float()
    if pose.dim() == 3 and pose.shape[-2:] == (3, 4):
        pad = torch.zeros(pose.shape[0], 1, 4); pad[..., 0, 3] = 1.0
        pose = torch.cat([pose, pad], 1)
    pose = pose.to(device)

    R = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob))) if args.rgb_dir else []
    D = sorted(glob.glob(os.path.join(args.dino_dir, args.dino_glob))) if args.dino_dir else []
    U = sorted(glob.glob(os.path.join(args.uncert_dir, '*.png'))) if args.uncert_dir else []
    os.makedirs(args.output, exist_ok=True)
    nf = pose.shape[0]
    idxs = list(range(0, nf, args.stride))
    print(f"checkpoint {ckpt_path} | {nf} poses | rendering {len(idxs)} frames -> {args.output}")
    geo_basis = None

    i, j = torch.meshgrid(torch.arange(W, device=device).float(), torch.arange(H, device=device).float(), indexing='ij')
    i = i.T; j = j.T
    dirs = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1)  # OpenGL

    for k in idxs:
        c2w = pose[k]
        rays_d = (dirs @ c2w[:3, :3].T); rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        ts = torch.full(rays_o.shape[:-1] + (1,), float(k), device=device)
        ro, rd = torch.cat([rays_o, ts], -1).reshape(-1, 4), rays_d.reshape(-1, 3)
        gf = []
        with torch.no_grad():
            for s in range(0, ro.shape[0], args.ray_batch_size):
                ret = model.render_rays(ro[s:s + args.ray_batch_size], rd[s:s + args.ray_batch_size])
                gf.append(ret['geo_feat'].cpu() if 'geo_feat' in ret else torch.zeros(ro[s:s + args.ray_batch_size].shape[0], 1))
        geo = torch.cat(gf, 0).reshape(H, W, -1).numpy()
        geo_rgb, geo_basis = pca_rgb(geo, geo_basis)        # shared basis across frames -> temporally stable
        panels = [_label(geo_rgb[..., ::-1], f'geo_feat (PCA)  f{k}')]   # ::-1 -> BGR for cv2

        if R and k < len(R):
            rgb = cv2.resize(cv2.imread(R[k]), (W, H)); panels.insert(0, _label(rgb, 'input rgb'))
        if D and k < len(D):
            d = np.load(D[k]).astype(np.float32)
            if d.ndim == 3:
                d_rgb, _ = pca_rgb(d, None); d_rgb = cv2.resize(d_rgb, (W, H))
                panels.append(_label(d_rgb[..., ::-1], 'DINO (PCA)'))
        if U and k < len(U):
            u = cv2.imread(U[k], cv2.IMREAD_UNCHANGED).astype(np.float32)
            u = cv2.applyColorMap(cv2.normalize(u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_INFERNO)
            panels.append(_label(cv2.resize(u, (W, H)), 'sigma^2'))

        out = cv2.hconcat([cv2.resize(p, (W, H)) for p in panels])
        cv2.imwrite(os.path.join(args.output, f'panel_{k:04d}.jpg'), out)
    print(f"done -> {args.output}/panel_*.jpg  (input | geo_feat | DINO | sigma^2)")


if __name__ == '__main__':
    main()
