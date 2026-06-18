#!/usr/bin/env python3
"""Bake dense self-supervised deformation TARGETS Δx* for the field teacher (Stage 1 data side).

For every frame k, finds where each bit of tissue sits in the CANONICAL frame (ref, default 0) via
DINO-correspondence + depth, and stores the world-space displacement Δx* = X_ref* - X_k at DINO-patch
resolution. This is EXACTLY the target scene_rep's TimeNet is regressed toward (deform_teacher_loss):
the field at (X_k, t_k) should output Δx*. Same engine the Stage-0 gate validated (+82% pin recovery,
cos +0.92) — backproj/project/bilinear/window-DINO-match, OpenGL, identity poses ok (Δx* is pose-frame
consistent as long as the SAME poses bake + train).

Per frame -> <out_dir>/<stem>_deform.npz : dx(Hp,Wp,3 float32), valid(Hp,Wp bool), trust(Hp,Wp float32
=1-ratio clamped). Training bilinear-samples dx/valid/trust at the sampled pixels (indice_h,indice_w).
Pins are NOT used here (held-out judge). Pure numpy.

  python Addons/deform/generate_deform_targets.py \
    --dino_dir data/Super/trail_3/dino --dino_glob '*_dino.npy' \
    --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy' \
    --out_dir data/Super/trail_3/deform [--est_c2w <run>/est_c2w_data.txt] [--seg_dir ...]
"""
import os, glob, argparse, numpy as np


def load_c2w(p):
    P = []
    for ln in open(p):
        v = ln.split()
        if len(v) >= 12 and not v[0].startswith('#'):
            T = np.eye(4); T[:3, :4] = np.array(list(map(float, v[:12]))).reshape(3, 4); P.append(T)
    return P


def load_depth(p, pds):
    if p.endswith('.npy'):
        d = np.load(p).astype(np.float64)
    else:
        import cv2; d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return d.squeeze() / pds


def load_grid(p):
    g = np.load(p).astype(np.float32)
    known = {384, 768, 1024, 1536}
    ax = next((i for i, s in enumerate(g.shape) if s in known), 2)
    return np.moveaxis(g, ax, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dino_dir', required=True); ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--depth_dir', required=True); ap.add_argument('--depth_glob', default='*left_depth.npy')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--est_c2w', default=''); ap.add_argument('--seg_dir', default=''); ap.add_argument('--seg_glob', default='*left*.png')
    ap.add_argument('--fx', type=float, default=768.98551924); ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567); ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--pds', type=float, default=8.0)
    ap.add_argument('--ref', type=int, default=0, help='canonical reference frame (the field anchor)')
    ap.add_argument('--win', type=int, default=32); ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--ratio_max', type=float, default=0.9, help='Lowe cosine-dist ratio gate (lower=stricter)')
    ap.add_argument('--dx_floor', type=float, default=0.0, help='reject |Δx*|<floor as static (0=keep all)')
    ap.add_argument('--ray', default='OpenGL', choices=['OpenGL', 'OpenCV'])
    ap.add_argument('--chunk', type=int, default=64, help='grid points matched per batch; lower if OOM')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    SY = -1.0 if args.ray == 'OpenGL' else 1.0; SZ = -1.0 if args.ray == 'OpenGL' else 1.0
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    deps = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_glob)))
    grids_p = sorted(glob.glob(os.path.join(args.dino_dir, args.dino_glob)))
    segs = sorted(glob.glob(os.path.join(args.seg_dir, args.seg_glob))) if args.seg_dir else []
    if args.est_c2w:
        poses = load_c2w(args.est_c2w); N = min(len(poses), len(deps), len(grids_p))
    else:
        N = min(len(deps), len(grids_p)); poses = [np.eye(4) for _ in range(N)]
        print("WARNING: no --est_c2w -> IDENTITY poses (fine for SemSup; bake AND train must use the same poses).")
    assert N >= 2, f'need >=2 frames (depth {len(deps)} dino {len(grids_p)})'
    d0 = load_depth(deps[0], args.pds); H, W = d0.shape
    g0 = load_grid(grids_p[0]); Hp, Wp = g0.shape[:2]
    print(f"frames {N} | image {W}x{H} | grid {Hp}x{Wp} | ref {args.ref} | win {args.win} | ratio<{args.ratio_max}")

    # patch-centre pixel for every grid cell (P = Hp*Wp)
    gy, gx = np.meshgrid(np.arange(Hp), np.arange(Wp), indexing='ij')
    PX = ((gx.ravel() + 0.5) * W / Wp); PY = ((gy.ravel() + 0.5) * H / Hp); P = PX.size
    _us = np.arange(-args.win, args.win + 1, args.step)
    OFF = np.stack(np.meshgrid(_us, _us), -1).reshape(-1, 2).astype(np.float64)   # (M,2)

    def backproj(u, v, d, T):                                  # u,v (...,) ; d HxW ; T 4x4 -> world (...,3), z
        z = d[np.clip(np.round(v).astype(int), 0, H-1), np.clip(np.round(u).astype(int), 0, W-1)]
        cam = np.stack([(u-cx)/fx, SY*(v-cy)/fy, SZ*np.ones_like(u)], -1) * z[..., None]
        return cam @ T[:3, :3].T + T[:3, 3], z

    def project(X, T):                                         # world (...,3) -> u,v (inverse of backproj)
        cam = (X - T[:3, 3]) @ T[:3, :3]
        z = SZ * cam[..., 2]; zz = np.where(z > 1e-4, z, 1.0)
        return cx + fx * cam[..., 0] / zz, cy + SY * fy * cam[..., 1] / zz, z > 1e-4

    def bilinear(g, u, v):                                     # g(Hp,Wp,C); u,v(...) -> (...,C) float32
        gxx = np.clip(u * Wp / W, 0, Wp-1-1e-3); gyy = np.clip(v * Hp / H, 0, Hp-1-1e-3)
        x0 = np.floor(gxx).astype(int); y0 = np.floor(gyy).astype(int); x1 = x0+1; y1 = y0+1
        wx = (gxx-x0).astype(np.float32)[..., None]; wy = (gyy-y0).astype(np.float32)[..., None]
        return (g[y0, x0]*(1-wx)*(1-wy) + g[y0, x1]*wx*(1-wy) + g[y1, x0]*(1-wx)*wy + g[y1, x1]*wx*wy)

    r = args.ref; dref = load_depth(deps[r], args.pds); gref = load_grid(grids_p[r])
    qref = gref.reshape(-1, gref.shape[2])                     # not used directly; matches sampled below
    tot_valid = 0; tot = 0
    for k in range(N):
        if k == r:
            np.savez_compressed(os.path.join(args.out_dir, os.path.basename(grids_p[k]).replace('.npy', '') + '_deform.npz'),
                                dx=np.zeros((Hp, Wp, 3), np.float32), valid=np.zeros((Hp, Wp), bool), trust=np.zeros((Hp, Wp), np.float32))
            continue
        dk = load_depth(deps[k], args.pds); gk = load_grid(grids_p[k])
        featq = gk.reshape(P, gk.shape[2])                     # (P,C) patch features at frame k
        Xk, zk = backproj(PX, PY, dk, poses[k])                # (P,3)
        cu, cv, okc = project(Xk, poses[r])                    # (P,) rigid search centres in ref
        mu = np.full(P, np.nan); mv = np.full(P, np.nan); ratio = np.ones(P)
        for s in range(0, P, args.chunk):
            e = min(s + args.chunk, P)
            cand = np.stack([cu[s:e, None] + OFF[None, :, 0], cv[s:e, None] + OFF[None, :, 1]], -1)  # (c,M,2)
            inb = (cand[..., 0] >= 0) & (cand[..., 0] < W) & (cand[..., 1] >= 0) & (cand[..., 1] < H)
            f = bilinear(gref, cand[..., 0], cand[..., 1])     # (c,M,C)
            q = featq[s:e]                                     # (c,C)
            sim = (f * q[:, None, :]).sum(-1) / (np.linalg.norm(f, axis=-1) * np.linalg.norm(q, axis=-1)[:, None] + 1e-9)
            sim = np.where(inb, sim, -2.0)
            b = np.argmax(sim, 1)                              # (c,)
            ci = np.arange(e - s)
            mu[s:e] = cand[ci, b, 0]; mv[s:e] = cand[ci, b, 1]
            # Lowe ratio: best vs best spatially-far candidate
            dd = np.linalg.norm(cand - cand[ci, b][:, None, :], axis=-1)
            far = dd > max(2 * args.step, 6)
            sim_far = np.where(far, sim, -2.0)
            second = sim_far.max(1)
            ratio[s:e] = (1 - sim[ci, b]) / (1 - second + 1e-9)
        Xrs, zrs = backproj(mu, mv, dref, poses[r])            # (P,3)
        dx = (Xrs - Xk).astype(np.float32)
        valid = okc & (zk > 1e-3) & (zrs > 1e-3) & (ratio < args.ratio_max) & np.isfinite(mu)
        if args.dx_floor > 0:
            valid &= (np.linalg.norm(dx, axis=1) > args.dx_floor)
        if segs and k < len(segs):
            import cv2
            sm = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)
            tis = sm[np.clip(PY.astype(int), 0, H-1), np.clip(PX.astype(int), 0, W-1)] > 0   # >0 = non-bg tissue/tool; refine per-dataset later
            valid &= tis
        trust = np.clip(1.0 - ratio, 0, 1).astype(np.float32) * valid
        np.savez_compressed(os.path.join(args.out_dir, os.path.basename(grids_p[k]).replace('.npy', '') + '_deform.npz'),
                            dx=dx.reshape(Hp, Wp, 3), valid=valid.reshape(Hp, Wp), trust=trust.reshape(Hp, Wp))
        tot_valid += int(valid.sum()); tot += P
        if k % 25 == 0 or k == N - 1:
            vm = np.linalg.norm(dx[valid], axis=1) if valid.any() else np.array([0.0])
            print(f"  [{k:3d}/{N}] valid {valid.mean()*100:4.0f}%  |Δx*| med {np.median(vm):.5f}  ratio med {np.median(ratio):.3f}")
    print(f"[deform] DONE {N} frames -> {args.out_dir}  | overall valid {100*tot_valid/max(tot,1):.0f}%")


if __name__ == '__main__':
    main()
