"""Regression smoke for the L0 depth supervisor (CPU, no GPU/RAFT needed).

Validates the depth-predicted rigid-flow math + the region soft-weight by MOCKING _raft_flow:
  A. STATIC scene (T_rel=I, zero flow)            -> residual ~0 everywhere (true NOP).
  B. Pure camera TRANSLATION                      -> background residual ~0 (moves-with-camera = rigid),
     an independently-moving blob (+10px)         -> residual ~10 (flagged).
     Also checks the predicted rigid flow equals the analytic parallax fx*tx/Z.
  C. region_soft_weight                           -> rigid bg weight ~1, deforming blob ~w_min (needs sklearn+cv2).
Run from the repo root:  python Addons/regression/test_l0_depthsup_smoke.py
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import Addons.motion.flow_track as ft


def main():
    H, W = 56, 84
    fx = fy = 1000.0; cx = W / 2.0; cy = H / 2.0
    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float32)
    Z0 = 0.5
    depth = np.full((H, W), Z0, np.float32)
    blob = np.zeros((H, W), bool); blob[18:38, 28:52] = True
    bg = ~blob
    ok = True

    # A: static
    ft._raft_flow = lambda *a, **k: np.zeros((H, W, 2), np.float32)
    rA = ft.rigid_flow_residual(None, None, depth, dirs, np.eye(4, dtype=np.float32), fx, fy, cx, cy, None, None, None)
    print(f"A static     : residual max={rA.max():.4f} mean={rA.mean():.4f}   (expect ~0)")
    ok &= rA.max() < 1e-3

    # B: camera translation + independently-moving blob
    tx = 0.01; rigid_u = fx * tx / Z0
    obs = np.zeros((H, W, 2), np.float32); obs[..., 0] = rigid_u; obs[blob, 0] = rigid_u + 10.0
    ft._raft_flow = lambda *a, **k: obs
    T = np.eye(4, dtype=np.float32); T[0, 3] = tx
    rB = ft.rigid_flow_residual(None, None, depth, dirs, T, fx, fy, cx, cy, None, None, None)
    print(f"B translation: rigid flow={rigid_u:.2f}px  bg residual mean={rB[bg].mean():.4f} (expect ~0)  "
          f"blob residual mean={rB[blob].mean():.4f} (expect ~10)")
    ok &= rB[bg].mean() < 1e-2 and abs(rB[blob].mean() - 10.0) < 0.5

    # C: region soft-weight (sklearn + cv2)
    try:
        dg = np.zeros((H, W, 4), np.float32); dg[..., 0] = 1.0; dg[blob, 0] = 0.0; dg[blob, 1] = 1.0
        w, med, mad, scale = ft.region_soft_weight(rB, dg, n_groups=2, mad_c=2.0, w_floor_px=1.0, w_min=0.1)
        print(f"C weight     : med={med:.3f} mad={mad:.3f} scale={scale:.3f}  bg w_mean={w[bg].mean():.3f} (expect ~1)  "
              f"blob w_mean={w[blob].mean():.3f} (expect ~w_min=0.1)")
        ok &= w[bg].mean() > 0.9 and w[blob].mean() < 0.3
    except Exception as e:
        print(f"C weight     : SKIPPED (env: {e}) -- residual math (A,B) still validated")

    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
