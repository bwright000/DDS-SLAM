#!/usr/bin/env python3
"""
Regauge baked Δx* deformation targets into the SLAM's pose frame.

THE BUG IT FIXES (2026-06-20, decisive): generate_deform_targets.py defaults to IDENTITY poses, but the
SLAM (pose-frozen SemSup) runs in a constant gauge R = diag(1,-1,-1) (an x-axis flip, t=0, identical every
frame -- the dataset's c2w convention). So the baked Δx* (= cam0-camk, identity frame) is in the WRONG frame
relative to where the field is trained/queried (R*camk). The field faithfully learns the identity-frame
target (judge: cos(D,baked)=+0.78) but it points the wrong way in the SLAM frame -> cos(D, X0-Xk)=-0.64,
pins -81%. Proof: the SLAM target is R*(cam0-camk) = R*Δx*, so we just rotate every stored Δx* by R.

This rotates Δx* -> R·Δx* (for R=diag(1,-1,-1): negate the y,z components). It is EXACT whenever the gauge is
a single constant rotation with zero translation (true for pose-frozen / static-camera runs). For a MOVING
camera (per-frame poses differ) a single R is wrong -> re-bake with generate_deform_targets.py --est_c2w
instead; this script refuses if --c2w shows a non-constant rotation.

Idempotent: the applied gauge is stored as the 'gauge' key inside each npz, so re-running is a no-op (it can
NEVER double-flip), and the marker travels with the file when persisted to Drive.

Usage:
  python Addons/deform/regauge_deform_targets.py --deform_dir data/Super/trail_3/deform --gauge xflip
  python Addons/deform/regauge_deform_targets.py --deform_dir <dir> --c2w <run>/est_c2w_data.txt   # derive R from data
"""
import argparse, glob, os
import numpy as np


def load_c2w_R(p):
    """Read the constant rotation from an est_c2w_data.txt (N x 12, rows = flattened 3x4). Verify it is
    constant and translation-free; return the 3x3 R. Errors if the camera moves (regauge would be wrong)."""
    P = np.loadtxt(p)
    if P.ndim == 1:
        P = P[None]
    Rs = P[:, :12].reshape(-1, 3, 4)
    R0 = Rs[0, :3, :3]
    dR = np.abs(Rs[:, :3, :3] - R0).max()
    dt = np.abs(Rs[:, :3, 3]).max()
    if dR > 1e-4:
        raise SystemExit(f"[regauge] ABORT: rotation is NOT constant across frames (max|R-R0|={dR:.2e}). "
                         f"Camera moves -> a single regauge R is wrong; re-bake with --est_c2w instead.")
    if dt > 1e-4:
        print(f"[regauge] WARNING: non-zero translation (max|t|={dt:.2e}); regauge only rotates Δx*, "
              f"which is exact for t=0. Proceeding (rotation dominates for small t).")
    return R0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deform_dir', required=True)
    ap.add_argument('--glob', default='*_deform.npz')
    ap.add_argument('--gauge', default='', choices=['', 'xflip', 'none'],
                    help="xflip = diag(1,-1,-1) (the proven SemSup pose-frozen gauge); none = identity (no-op check).")
    ap.add_argument('--c2w', default='', help='derive R from this est_c2w_data.txt instead of --gauge (robust).')
    args = ap.parse_args()

    if args.c2w:
        R = load_c2w_R(args.c2w)
    elif args.gauge == 'xflip':
        R = np.diag([1.0, -1.0, -1.0])
    elif args.gauge == 'none':
        R = np.eye(3)
    else:
        raise SystemExit("[regauge] give --gauge xflip|none OR --c2w <est_c2w_data.txt>")
    print(f"[regauge] target gauge R =\n{R}")

    files = sorted(glob.glob(os.path.join(args.deform_dir, args.glob)))
    if not files:
        raise SystemExit(f"[regauge] no targets matched {os.path.join(args.deform_dir, args.glob)}")

    done = skip = bad = 0
    for f in files:
        d = dict(np.load(f, allow_pickle=False))
        cur = d['gauge'] if 'gauge' in d else np.eye(3)
        if np.allclose(cur, R, atol=1e-5):
            skip += 1; continue
        if not np.allclose(cur, np.eye(3), atol=1e-5):
            print(f"[regauge] SKIP {os.path.basename(f)}: already gauged to a DIFFERENT R (refusing to compose)."); bad += 1; continue
        dx = d['dx'].astype(np.float64)
        d['dx'] = (dx @ R.T).astype(d['dx'].dtype)     # rotate each displacement vector: (R·v) per pixel
        d['gauge'] = R
        np.savez(f, **d)
        done += 1
    print(f"[regauge] {done} rotated, {skip} already-correct (idempotent skip), {bad} conflicting | dir={args.deform_dir}")
    if bad:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
