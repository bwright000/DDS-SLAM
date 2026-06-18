#!/usr/bin/env python3
"""TEACHER-vs-pin EPE — Stage-0 ZERO-GPU commitment gate for the deformation-field teacher.

Decides, BEFORE any field code or GPU: can the self-supervised teacher (DINO-correspondence + depth)
recover real tissue motion? If the teacher itself can't beat rigid at the pins, a field trained on it
never will -> ABORT, no GPU spent. The teacher target is exactly Δx* = X_r* - X_k where X_r* is the
DINO-matched location lifted by depth (the thing scene_rep's TimeNet would be regressed toward).

TWO REPORTS:
  (A) PIN gate  [GT-backed, but OPTIMISTIC -- green pins are SALIENT markers]: DINO-match each pin
      pixel from frame k into reference frame r (pose-bounded window), lift via depth+pose, measure how
      close the match lands to the pin's TRUE location (X_r_pin) vs the rigid no-match baseline (X_k).
      reduction>0 & cos>0 => the teacher recovers the motion. Binned by motion magnitude: the SMALL
      bin is the CRCD-scale (~few-px) viability probe. --ref_stride>0 uses r=k-stride (per-frame small
      motion) instead of r=0 (cumulative).
  (B) GENERIC-tissue cycle consistency [GT-free, CRCD-RELEVANT -- random NON-pin tissue]: match
      k->r->k, report pixel cycle error + confident-match fraction. This is what must generalise to
      CRCD, which has no salient markers. Pins pass (A) easily; (B) is the honest generalisation signal.

Pins are READ-ONLY GT, never trained on. Pure numpy (DINO .npy grids + depth + poses + pins) — no
model, no torch — runs anywhere the data is staged.

  python Addons/eval/teacher_vs_pin_epe.py --pts Addons/eval/gt_pins/trial_3_l_pts.npy \
    --est_c2w <run>/est_c2w_data.txt --depth_dir data/Super/trail_3/depth/moge2 \
    --dino_dir data/Super/trail_3/dino --dino_glob '*_dino.npy' [--ref_stride 0]
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
    if p.endswith('.npy'):
        d = np.load(p).astype(np.float64)
    else:
        import cv2; d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float64)
    return d.squeeze() / pds


def load_grid(p):
    g = np.load(p).astype(np.float32)
    if g.ndim != 3:
        raise ValueError(f'grid {p} ndim {g.ndim}')
    # normalise to (Hp,Wp,C): the feature axis is the one matching a known DINO width
    known = {384, 768, 1024, 1536}
    ax = next((i for i, s in enumerate(g.shape) if s in known), 2)
    return np.moveaxis(g, ax, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pts', required=True); ap.add_argument('--est_c2w', required=True)
    ap.add_argument('--depth_dir', required=True); ap.add_argument('--depth_glob', default='*left_depth.npy')
    ap.add_argument('--dino_dir', required=True); ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--fx', type=float, default=768.98551924); ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567); ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--pds', type=float, default=8.0)
    ap.add_argument('--ref_stride', type=int, default=0, help='0=ref frame0 (cumulative); N=ref k-N (small per-frame motion)')
    ap.add_argument('--win', type=int, default=32); ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--n_generic', type=int, default=2000, help='random non-pin tissue samples for the cycle probe; 0=skip')
    ap.add_argument('--ray', default='OpenGL', choices=['OpenGL', 'OpenCV'])
    args = ap.parse_args()
    SY = -1.0 if args.ray == 'OpenGL' else 1.0; SZ = -1.0 if args.ray == 'OpenGL' else 1.0
    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    poses = load_c2w(args.est_c2w); pins = load_pts(args.pts)
    deps = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_glob)))
    grids_p = sorted(glob.glob(os.path.join(args.dino_dir, args.dino_glob)))
    N = min(len(poses), len(deps), len(grids_p))
    assert N >= 2, f'need >=2 aligned frames (poses {len(poses)} depth {len(deps)} dino {len(grids_p)})'
    print(f"frames {N} | pin-frames {len(pins)} | ref_stride {args.ref_stride} | win {args.win} step {args.step} | ray {args.ray}")
    _d0 = load_depth(deps[0], args.pds); H, W = _d0.shape
    print(f"image {W}x{H} | dino grid {load_grid(grids_p[0]).shape}")

    dcache, gcache = {}, {}
    def depth(i):
        if i not in dcache: dcache[i] = load_depth(deps[i], args.pds)
        return dcache[i]
    def grid(i):
        if i not in gcache: gcache[i] = load_grid(grids_p[i])
        return gcache[i]

    def backproj(u, v, i):                                  # pixel(s) in frame i -> world (OpenGL)
        d = depth(i); z = d[np.clip(np.round(v).astype(int), 0, H-1), np.clip(np.round(u).astype(int), 0, W-1)]
        cam = np.stack([(u-cx)/fx, SY*(v-cy)/fy, SZ*np.ones_like(u)], -1) * np.asarray(z)[..., None]
        T = poses[i]; return (cam @ T[:3, :3].T) + T[:3, 3], z

    def project(X, i):                                      # world -> pixel in frame i (inverse of backproj)
        T = poses[i]; cam = (X - T[:3, 3]) @ T[:3, :3]      # cam = (X-t) @ R   (R orthonormal)
        z = SZ * cam[..., 2]; ok = z > 1e-4                 # cam_z = SZ*z  -> z = SZ*cam_z
        zz = np.where(ok, z, 1.0)
        u = cx + fx * cam[..., 0] / zz                      # cam_x = (u-cx)/fx * z
        v = cy + SY * fy * cam[..., 1] / zz                 # cam_y = SY*(v-cy)/fy * z
        return u, v, ok

    def bilinear(g, u, v):                                  # g (Hp,Wp,C); u,v image px -> (...,C)
        Hp, Wp = g.shape[:2]
        gx = np.clip(u * Wp / W, 0, Wp-1-1e-3); gy = np.clip(v * Hp / H, 0, Hp-1-1e-3)
        x0 = np.floor(gx).astype(int); y0 = np.floor(gy).astype(int); x1 = x0+1; y1 = y0+1
        wx = (gx-x0)[..., None]; wy = (gy-y0)[..., None]
        return (g[y0, x0]*(1-wx)*(1-wy) + g[y0, x1]*wx*(1-wy) + g[y1, x0]*(1-wx)*wy + g[y1, x1]*wx*wy)

    _us = np.arange(-args.win, args.win+1, args.step)
    OFF = np.stack(np.meshgrid(_us, _us), -1).reshape(-1, 2).astype(np.float64)   # (M,2) window offsets

    def match(feat_q, gr, cu, cv):                          # query feat -> best (u,v) in grid gr near (cu,cv)
        cand = OFF + np.array([cu, cv])
        m = (cand[:, 0] >= 0) & (cand[:, 0] < W) & (cand[:, 1] >= 0) & (cand[:, 1] < H)
        cand = cand[m]
        if len(cand) < 4: return None
        f = bilinear(gr, cand[:, 0], cand[:, 1])
        sim = f @ feat_q / (np.linalg.norm(f, axis=1) * (np.linalg.norm(feat_q)+1e-9) + 1e-9)
        b = int(np.argmax(sim))
        far = np.linalg.norm(cand - cand[b], axis=1) > max(2*args.step, 6)
        second = sim[far].max() if far.any() else sim[b]
        ratio = (1-sim[b]) / (1-second + 1e-9)              # cosine-dist Lowe ratio; <1 = confident
        return cand[b], float(sim[b]), float(ratio)

    # ---------- (A) PIN gate ----------
    rigid, teach, cosd, ratios, mags = [], [], [], [], []
    for k in sorted(pins):
        r = 0 if args.ref_stride == 0 else max(0, k - args.ref_stride)
        if k == r or k >= N or r >= N: continue
        gk, gr = pins[k], pins[r]; grid_k, grid_r = grid(k), grid(r)
        for p in range(gk.shape[0]):
            if gk[p, 2] != 1 or gr[p, 2] != 1: continue
            uk, vk = gk[p, 0], gk[p, 1]
            Xk, zk = backproj(np.array([uk]), np.array([vk]), k)
            if zk[0] <= 1e-3: continue
            Xrp, zrp = backproj(np.array([gr[p, 0]]), np.array([gr[p, 1]]), r)   # true pin loc @ r
            if zrp[0] <= 1e-3: continue
            cu, cv, ok = project(Xk[0], r)                  # rigid search centre
            if not ok: continue
            feat_q = bilinear(grid_k, np.array([uk]), np.array([vk]))[0]
            res = match(feat_q, grid_r, float(cu), float(cv))
            if res is None: continue
            (ur, vr), sim, ratio = res
            Xrs, zrs = backproj(np.array([ur]), np.array([vr]), r)              # matched loc lifted
            if zrs[0] <= 1e-3: continue
            rr = np.linalg.norm(Xk[0]-Xrp[0]); tt = np.linalg.norm(Xrs[0]-Xrp[0])
            dxs = Xrs[0]-Xk[0]; gtv = Xrp[0]-Xk[0]
            rigid.append(rr); teach.append(tt); ratios.append(ratio); mags.append(np.linalg.norm(gtv))
            cosd.append(float((dxs@gtv)/(np.linalg.norm(dxs)*np.linalg.norm(gtv)+1e-12)))
    rigid, teach, cosd, ratios, mags = map(np.array, (rigid, teach, cosd, ratios, mags))
    print(f"\n=== (A) PIN gate  (n={len(rigid)} pin-obs, world units; ref={'frame0' if args.ref_stride==0 else f'k-{args.ref_stride}'}) ===")
    if len(rigid):
        red = 100*(rigid.mean()-teach.mean())/max(rigid.mean(), 1e-9)
        print(f"  rigid (no match)   : {rigid.mean():.5f}")
        print(f"  teacher (DINO+depth): {teach.mean():.5f}   reduction = {red:+.1f}%   (>0 => teacher recovers motion)")
        print(f"  cos(Δx*, gt)       : {np.nanmean(cosd):+.3f}   (>0 => correct direction; the hollow guard)")
        print(f"  Lowe ratio (conf)  : median {np.median(ratios):.3f}  frac<0.9 {np.mean(ratios<0.9):.2f}  (lower=more discriminative)")
        # magnitude bins -- the SMALL bin is the CRCD-scale viability
        qs = np.quantile(mags, [0, .33, .66, 1.0])
        for lo, hi, nm in [(qs[0], qs[1], 'small (CRCD-scale)'), (qs[1], qs[2], 'med'), (qs[2], qs[3]+1e-9, 'large')]:
            m = (mags >= lo) & (mags < hi)
            if m.sum():
                rd = 100*(rigid[m].mean()-teach[m].mean())/max(rigid[m].mean(), 1e-9)
                print(f"    |gt|∈[{lo:.4f},{hi:.4f}) {nm:18s} n={m.sum():4d}  reduction={rd:+6.1f}%  cos={np.nanmean(cosd[m]):+.2f}")
    else:
        print("  (no valid pin observations)")

    # ---------- (B) GENERIC-tissue cycle consistency ----------
    if args.n_generic > 0:
        rng = np.random.RandomState(0); cyc, conf = [], []
        ks = [k for k in range(1, N)]; rng.shuffle(ks)
        per = max(1, args.n_generic // max(1, min(len(ks), 60)))
        done = 0
        for k in ks:
            if done >= args.n_generic: break
            r = 0 if args.ref_stride == 0 else max(0, k - args.ref_stride)
            if k == r: continue
            grid_k, grid_r = grid(k), grid(r); dk = depth(k)
            us = rng.randint(args.win, W-args.win, per); vs = rng.randint(args.win, H-args.win, per)
            for u0, v0 in zip(us, vs):
                if dk[v0, u0] <= 1e-3: continue
                Xk, _ = backproj(np.array([float(u0)]), np.array([float(v0)]), k)
                cu, cv, ok = project(Xk[0], r)
                if not ok: continue
                fq = bilinear(grid_k, np.array([float(u0)]), np.array([float(v0)]))[0]
                m1 = match(fq, grid_r, float(cu), float(cv))
                if m1 is None: continue
                (ur, vr), _, ratio = m1
                f2 = bilinear(grid_r, np.array([ur]), np.array([vr]))[0]
                m2 = match(f2, grid_k, float(u0), float(v0))                     # cycle back
                if m2 is None: continue
                (ub, vb), _, _ = m2
                cyc.append(float(np.hypot(ub-u0, vb-v0))); conf.append(ratio); done += 1
        cyc, conf = np.array(cyc), np.array(conf)
        print(f"\n=== (B) GENERIC-tissue cycle consistency  (n={len(cyc)} random non-pin px; GT-free, CRCD-relevant) ===")
        if len(cyc):
            print(f"  cycle error px     : median {np.median(cyc):.2f}  frac<{args.step*2}px {np.mean(cyc<args.step*2):.2f}  (low=consistent corr)")
            print(f"  Lowe ratio (conf)  : median {np.median(conf):.3f}  frac<0.9 {np.mean(conf<0.9):.2f}")

    print("\nGATE: (A) reduction >> 0 AND cos>0 on the SMALL bin  -> teacher recovers CRCD-scale motion -> BUILD the field loss.")
    print("      (A) fails on small bin, OR (B) cycle error large / ratios~1  -> DINO too coarse for the motion -> SemSup-only / finer corr.")


if __name__ == '__main__':
    main()
