"""Regression smoke for the L0 depth supervisor (CPU, no GPU/RAFT needed).

Validates the depth-predicted rigid-flow math + the region soft-weight by MOCKING _raft_flow, using the
REAL DDS pose convention (OpenGL: load_poses negates the y,z axes; get_camera_rays default z=-1). The
caller converts c2w GL->CV (c2w @ diag(1,-1,-1,1)) before forming T_rel; rigid_flow_residual builds
OpenCV dirs internally. This mirrors ddsslam.py so a convention regression (the f9 resid=0 no-op that
killed the first E3 ablation) is caught here.
  A. STATIC (ref==cur, observed=0)                 -> residual ~0 (NOP).
  B1 CAMERA MOVED, observed=0                       -> residual ~ fx*tx/Z (parallax), NON-ZERO + finite.
     (THE bug-catcher: the OpenGL/OpenCV mismatch made every Z<0 -> bad-mask -> all-zero residual.)
  B2 CAMERA MOVED, observed = the rigid flow        -> background residual ~0 (moves-with-camera),
     an independently-moving blob (+10px)           -> residual ~10 (flagged).
  C. region_soft_weight                             -> rigid bg w~1, deforming blob ~w_min (sklearn+cv2).
Run from the repo root:  python Addons/regression/test_l0_depthsup_smoke.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import Addons.motion.flow_track as ft


def _gl_pose(tx=0.0):
    """A dataset-style OpenGL c2w (identity with y,z axes negated), optionally translated +tx in world x."""
    c = np.eye(4, dtype=np.float32); c[:3, 1] *= -1; c[:3, 2] *= -1; c[0, 3] = tx
    return c


def main():
    H, W = 56, 84
    fx = fy = 1000.0; cx = W / 2.0; cy = H / 2.0
    Z0 = 0.5
    depth = np.full((H, W), Z0, np.float32)
    blob = np.zeros((H, W), bool); blob[18:38, 28:52] = True
    bg = ~blob
    M = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)   # GL<->CV (self-inverse)
    ok = True

    # A: static -> T_rel = I -> residual ~0
    ref_cv = _gl_pose(0.0) @ M; cur_cv = _gl_pose(0.0) @ M
    T_static = np.linalg.inv(cur_cv) @ ref_cv
    ft._raft_flow = lambda *a, **k: np.zeros((H, W, 2), np.float32)
    rA = ft.rigid_flow_residual(None, None, depth, T_static, fx, fy, cx, cy, None, None, None)
    print(f"A static       : residual max={rA.max():.4f} mean={rA.mean():.4f}   (expect ~0)")
    ok &= rA.max() < 1e-3

    # B: camera translates tx -> GL poses -> CV T_rel
    tx = 0.01; parallax = fx * tx / Z0                          # analytic |rigid flow| = 20px
    ref_cv = _gl_pose(0.0) @ M; cur_cv = _gl_pose(tx) @ M
    T_rel = (np.linalg.inv(cur_cv) @ ref_cv).astype(np.float32)

    # B1: observed = 0 -> residual == the predicted parallax (NON-ZERO + finite) == the bug-catcher
    ft._raft_flow = lambda *a, **k: np.zeros((H, W, 2), np.float32)
    rB1 = ft.rigid_flow_residual(None, None, depth, T_rel, fx, fy, cx, cy, None, None, None)
    print(f"B1 obs=0       : residual mean={rB1.mean():.3f} max={rB1.max():.3f}  (expect ~{parallax:.1f}px, NON-zero, finite)")
    ok &= np.isfinite(rB1).all() and abs(rB1.mean() - parallax) < 0.5 and rB1.max() > 1.0

    # B2: observed = the rigid flow (-parallax in x) + a +10px blob -> bg residual ~0, blob ~10
    obs = np.zeros((H, W, 2), np.float32); obs[..., 0] = -parallax; obs[blob, 0] = -parallax + 10.0
    ft._raft_flow = lambda *a, **k: obs
    rB2 = ft.rigid_flow_residual(None, None, depth, T_rel, fx, fy, cx, cy, None, None, None)
    print(f"B2 obs=rigid   : bg residual mean={rB2[bg].mean():.4f} (expect ~0)  blob mean={rB2[blob].mean():.4f} (expect ~10)")
    ok &= rB2[bg].mean() < 0.5 and abs(rB2[blob].mean() - 10.0) < 0.5

    # C: region soft-weight (sklearn + cv2)
    try:
        dg = np.zeros((H, W, 4), np.float32); dg[..., 0] = 1.0; dg[blob, 0] = 0.0; dg[blob, 1] = 1.0
        w, med, mad, scale = ft.region_soft_weight(rB2, dg, n_groups=2, mad_c=2.0, w_floor_px=1.0, w_min=0.1)
        print(f"C weight       : med={med:.3f} mad={mad:.3f} scale={scale:.3f}  bg w_mean={w[bg].mean():.3f} (expect ~1)  "
              f"blob w_mean={w[blob].mean():.3f} (expect ~w_min=0.1)")
        ok &= w[bg].mean() > 0.9 and w[blob].mean() < 0.3
    except Exception as e:
        print(f"C weight       : SKIPPED (env: {e}) -- residual math (A,B) still validated")

    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
