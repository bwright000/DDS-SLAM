#!/usr/bin/env python3
"""Pose-compensated temporal residual = the Motion Teacher's signal (offline, VERIFIABLE).

THE IDEA (see memory project_arm2_deformation_field_diagnosis_20260617): the deformation field has
no dedicated teacher. This bakes one. For consecutive frames k-1,k we KNOW the camera motion (the
SLAM's est pose), so we "undo" it: reproject frame k's surface into k-1's view via pose+depth and
compare appearance. STATIC tissue (only the camera moved) -> matches. DEFORMING tissue (the surface
itself shifted) -> mismatch. The leftover residual = the non-rigid motion = WHERE the field should
deform. It's a MOTION signal (sees deformation even when appearance barely changes) -- exactly the
blind spot of the photometric-residual sigma^2 hedge.

WHY OFFLINE + STANDALONE: this is the correctness-critical step (reprojection direction is precisely
the kind of convention bug this project has retracted before). Bake it, EYEBALL the maps (do they
light up on the deforming tissue and go dark on static regions / the tool?), THEN feed them to the
field. Two-pass: pass-1 = a normal run -> est_c2w; this baker -> motion maps; pass-2 = the teacher run.

INPUTS:
  --est_c2w   est_c2w_data.txt from pass-1 (each line 12 floats = 3x4 c2w, translation cols 3,7,11)
  --rgb_dir --rgb_glob          consecutive RGB frames (sorted == temporal order)
  --depth_dir --depth_glob      per-frame depth .npy (same order); depth_m = npy / --depth_scale
  --seg_dir --seg_glob          OPTIONAL tool/instrument mask -> zero the residual on the tool
  intrinsics (default = SemSup trail3: fx=fy=768.9855 cx=292.886 cy=291.615)
OUTPUT  <out_dir>/<stem>_motion.npy  : float16 [H,W] in [0,1] (frame 0 = zeros; no predecessor)
        + a contact-sheet stat line per frame so a non-zero, non-saturated map is visible in the log.

  python Addons/motion/bake_motion_residual.py \
    --est_c2w output/field_on/demo/est_c2w_data.txt \
    --rgb_dir data/Super/trail_3/rgb --rgb_glob '*left.png' \
    --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy' --depth_scale 8 \
    --out_dir data/Super/trail_3/motion
"""
import cv2, numpy as np, os, glob, argparse


def load_c2w(path):
    P = []
    for ln in open(path):
        v = ln.split()
        if len(v) >= 12 and not v[0].startswith('#'):
            P.append(np.array(list(map(float, v[:12])), float).reshape(3, 4))
    return P  # list of 3x4


def c2w_to_4x4(m):
    T = np.eye(4); T[:3, :4] = m; return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est_c2w', required=True)
    ap.add_argument('--rgb_dir', required=True); ap.add_argument('--rgb_glob', default='*left.png')
    ap.add_argument('--depth_dir', required=True); ap.add_argument('--depth_glob', default='*left_depth.npy')
    ap.add_argument('--depth_scale', type=float, default=8.0, help='depth_m = npy / depth_scale')
    ap.add_argument('--seg_dir', default=None); ap.add_argument('--seg_glob', default='*left.png')
    ap.add_argument('--tool_label', type=int, default=-1, help='seg value to treat as tool (-1 = any nonzero)')
    ap.add_argument('--fx', type=float, default=768.98551924); ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567); ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--norm_pct', type=float, default=95.0, help='per-frame normaliser percentile')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    poses = load_c2w(args.est_c2w)
    rgbs = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    deps = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_glob)))
    segs = sorted(glob.glob(os.path.join(args.seg_dir, args.seg_glob))) if args.seg_dir else []
    N = min(len(poses), len(rgbs), len(deps))
    assert N >= 2, f'need >=2 aligned frames (poses {len(poses)}, rgb {len(rgbs)}, depth {len(deps)})'
    print(f'[motion] {N} frames | est_c2w {len(poses)} rgb {len(rgbs)} depth {len(deps)} seg {len(segs)}')

    def rgb(i):
        im = cv2.imread(rgbs[i], cv2.IMREAD_COLOR)
        return im.astype(np.float32) / 255.0  # [H,W,3] BGR (consistent across frames -> fine for a diff)

    H, W = rgb(0).shape[:2]
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    # frame 0: no predecessor -> zero map
    np.save(os.path.join(args.out_dir, os.path.splitext(os.path.basename(rgbs[0]))[0] + '_motion.npy'),
            np.zeros((H, W), np.float16))

    for k in range(1, N):
        d = np.load(deps[k]).astype(np.float32).squeeze() / args.depth_scale       # metres, frame k
        Ik, Ikm1 = rgb(k), rgb(k - 1)
        # backproject frame-k pixels to 3D (frame-k camera), to world, to frame k-1 camera
        Zk = d
        Xk = (uu - args.cx) * Zk / args.fx
        Yk = (vv - args.cy) * Zk / args.fy
        pts_k = np.stack([Xk, Yk, Zk, np.ones_like(Zk)], -1).reshape(-1, 4).T       # 4 x HW
        T_km1_k = np.linalg.inv(c2w_to_4x4(poses[k - 1])) @ c2w_to_4x4(poses[k])     # k-cam -> k-1-cam
        pts_km1 = T_km1_k @ pts_k                                                    # 4 x HW
        Xp, Yp, Zp = pts_km1[0], pts_km1[1], pts_km1[2]
        valid = (Zk.reshape(-1) > 1e-4) & (Zp > 1e-4)
        up = args.fx * Xp / np.where(Zp == 0, 1e-6, Zp) + args.cx
        vp = args.fy * Yp / np.where(Zp == 0, 1e-6, Zp) + args.cy
        inb = (up >= 0) & (up <= W - 1) & (vp >= 0) & (vp <= H - 1) & valid
        # bilinear-sample frame k-1 at the rigid-reprojected pixels
        upc = np.clip(up, 0, W - 1); vpc = np.clip(vp, 0, H - 1)
        warp = cv2.remap(Ikm1, upc.reshape(H, W), vpc.reshape(H, W),
                         interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # residual AFTER rigid (pose) compensation = non-rigid motion proxy
        resid = np.abs(Ik - warp).mean(-1)                                          # [H,W]
        resid = resid * inb.reshape(H, W)                                           # kill out-of-bounds
        # mask the tool (its motion is independent, not tissue deformation)
        if k < len(segs):
            s = cv2.imread(segs[k], cv2.IMREAD_GRAYSCALE)
            if s is not None and s.shape == resid.shape:
                tool = (s != 0) if args.tool_label < 0 else (s == args.tool_label)
                resid[tool] = 0.0
        # per-frame normalise to [0,1] (robust percentile) -> relative motion, scale-free
        p = np.percentile(resid[resid > 0], args.norm_pct) if (resid > 0).any() else 1.0
        M = np.clip(resid / max(p, 1e-6), 0, 1).astype(np.float16)
        np.save(os.path.join(args.out_dir, os.path.splitext(os.path.basename(rgbs[k]))[0] + '_motion.npy'), M)
        if k % 25 == 1 or k == N - 1:
            mf = float(M.mean()); hi = float((M > 0.5).mean())
            print(f'  [{k:3d}/{N}] mean={mf:.3f} frac>0.5={hi:.3f} inb={inb.mean():.2f}  '
                  f'(EYEBALL: high on deforming tissue, ~0 on static/tool, not all-saturated)')
    print(f'[motion] DONE -> {args.out_dir}  ({len(glob.glob(args.out_dir + "/*_motion.npy"))} maps)')


if __name__ == '__main__':
    main()
