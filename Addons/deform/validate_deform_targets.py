#!/usr/bin/env python3
"""Validate the BAKED deformation targets reproduce the Stage-0 gate (+82%) — before any model edit.

The Stage-0 gate matched DINO live at the pins. This reads the BAKED Δx* (patch-resolution .npz from
generate_deform_targets.py), bilinear-samples it at the pin pixels, and re-runs the pin-EPE check. If
the bake is faithful, reduction ~= the gate's +82% and cos ~= +0.92. A gap exposes a bake bug
(indexing, patch-resolution loss, npz round-trip, validity over-masking) that would silently poison
training. Pins are READ-ONLY (held-out judge). Pure numpy, no model.

  python Addons/deform/validate_deform_targets.py \
    --pts Addons/eval/gt_pins/trial_3_l_pts.npy --deform_dir data/Super/trail_3/deform \
    --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy' [--est_c2w ...]
"""
import os, glob, argparse, numpy as np


def load_c2w(p):
    P = []
    for ln in open(p):
        v = ln.split()
        if len(v) >= 12 and not v[0].startswith('#'):
            T = np.eye(4); T[:3, :4] = np.array(list(map(float, v[:12]))).reshape(3, 4); P.append(T)
    return P


def load_pts(p):
    d = np.load(p, allow_pickle=True); d = d.item() if hasattr(d, 'item') else d
    gt = d['gt'] if isinstance(d, dict) and 'gt' in d else d
    return {int(k): np.asarray(v, np.float64) for k, v in gt.items()}


def load_depth(p, pds):
    d = np.load(p).astype(np.float64) if p.endswith('.npy') else __import__('cv2').imread(p, -1).astype(np.float64)
    return d.squeeze() / pds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pts', required=True); ap.add_argument('--deform_dir', required=True)
    ap.add_argument('--deform_glob', default='*_deform.npz')
    ap.add_argument('--depth_dir', required=True); ap.add_argument('--depth_glob', default='*left_depth.npy')
    ap.add_argument('--est_c2w', default=''); ap.add_argument('--ref', type=int, default=0)
    ap.add_argument('--fx', type=float, default=768.98551924); ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567); ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--pds', type=float, default=8.0); ap.add_argument('--ray', default='OpenGL', choices=['OpenGL', 'OpenCV'])
    args = ap.parse_args()
    SY = -1.0 if args.ray == 'OpenGL' else 1.0; SZ = -1.0 if args.ray == 'OpenGL' else 1.0
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    pins = load_pts(args.pts)
    deps = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_glob)))
    defs = sorted(glob.glob(os.path.join(args.deform_dir, args.deform_glob)))
    N = min(len(deps), len(defs))
    if args.est_c2w: poses = load_c2w(args.est_c2w); N = min(N, len(poses))
    else: poses = [np.eye(4) for _ in range(N)]; print("WARNING: identity poses (must match the bake's poses).")
    d0 = load_depth(deps[0], args.pds); H, W = d0.shape
    print(f"frames {N} | image {W}x{H} | ref {args.ref} | est_c2w {'yes' if args.est_c2w else 'identity'}")

    def backproj(u, v, d, T):
        z = d[np.clip(np.round(v).astype(int), 0, H-1), np.clip(np.round(u).astype(int), 0, W-1)]
        cam = np.stack([(u-cx)/fx, SY*(v-cy)/fy, SZ*np.ones_like(u)], -1) * z[..., None]
        return cam @ T[:3, :3].T + T[:3, 3], z

    def sample(mp, u, v):                                       # (Hp,Wp,K) -> (...,K) bilinear; valid -> nearest
        Hp, Wp = mp.shape[:2]
        gx = np.clip(u * Wp / W, 0, Wp-1-1e-3); gy = np.clip(v * Hp / H, 0, Hp-1-1e-3)
        x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int); x1 = x0+1; y1 = y0+1
        wx = (gx-x0)[..., None]; wy = (gy-y0)[..., None]
        return (mp[y0, x0]*(1-wx)*(1-wy) + mp[y0, x1]*wx*(1-wy) + mp[y1, x0]*(1-wx)*wy + mp[y1, x1]*wx*wy)

    r = args.ref; dref = load_depth(deps[r], args.pds)
    g0 = pins[r]
    X0, z0 = backproj(g0[:, 0], g0[:, 1], dref, poses[r])
    rig, tea, cosd, covv = [], [], [], []
    for k in sorted(pins):
        if k == r or k >= N: continue
        gk = pins[k]; dk = load_depth(deps[k], args.pds)
        dd = np.load(defs[k]); dxmap = dd['dx']; vmap = dd['valid']; Hp, Wp = vmap.shape
        Xk, zk = backproj(gk[:, 0], gk[:, 1], dk, poses[k])
        m = (gk[:, 2] == 1) & (g0[:, 2] == 1) & (zk > 1e-3) & (z0 > 1e-3)
        if not m.any(): continue
        dxs = sample(dxmap, gk[m, 0], gk[m, 1])                # baked Δx* at the pins (P,3)
        vb = vmap[np.clip(np.round(gk[m, 1]*Hp/H).astype(int), 0, Hp-1), np.clip(np.round(gk[m, 0]*Wp/W).astype(int), 0, Wp-1)]
        rr = np.linalg.norm(Xk[m]-X0[m], axis=1)
        tt = np.linalg.norm(Xk[m]+dxs-X0[m], axis=1)
        gt = X0[m]-Xk[m]; cs = (dxs*gt).sum(1)/(np.linalg.norm(dxs, axis=1)*np.linalg.norm(gt, axis=1)+1e-12)
        rig += list(rr); tea += list(tt); cosd += list(cs); covv += list(vb.astype(float))
    rig, tea, cosd, covv = map(np.array, (rig, tea, cosd, covv))
    vb = covv > 0.5
    print(f"\n=== BAKED-TARGET pin check (n={len(rig)}; expect ~ gate +82%, cos +0.92) ===")
    for nm, sel in [('all pins', np.ones(len(rig), bool)), ('valid-only (training uses these)', vb)]:
        if sel.sum() == 0: print(f"  {nm}: (none)"); continue
        red = 100*(rig[sel].mean()-tea[sel].mean())/max(rig[sel].mean(), 1e-9)
        print(f"  {nm:34s} n={sel.sum():4d}  rigid {rig[sel].mean():.5f} -> teacher {tea[sel].mean():.5f}  reduction {red:+.1f}%  cos {np.nanmean(cosd[sel]):+.3f}")
    print(f"  pin coverage by valid mask: {vb.mean()*100:.0f}%")
    print("\nPASS if valid-only reduction is within ~10pts of the live gate (+82%) and cos>0.85 -> bake faithful -> build the model loss.")


if __name__ == '__main__':
    main()
