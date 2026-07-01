"""CPU smoke for the new flow_track modes (no GPU/RAFT; _raft_flow is mocked with synthetic rigid flow).
  A (rigid_solve_pnp): recover a KNOWN camera SE3 from synthetic rigid flow; and with a tool blob + tool_mask,
    still recover the camera (blob excluded).
  B (depth_pooled_weight): two background regions at DIFFERENT depths under camera translation collapse to ONE
    consensus (distance-invariance) -> w~1; an independently-moving blob -> w down-weighted.
Run from repo root: python Addons/regression/test_newflow_smoke.py
"""
import os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import Addons.motion.flow_track as ft

H, W = 60, 84
fx = fy = 500.0; cx = W / 2.0; cy = H / 2.0
uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
DIRS = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).astype(np.float32)


def rigid_flow(Dr, R, t):
    Xr = Dr[..., None] * DIRS
    Xc = (R @ Xr.reshape(-1, 3).T + t.reshape(3, 1)).T
    z = np.clip(Xc[:, 2], 1e-6, None)
    p2 = np.stack([fx * Xc[:, 0] / z + cx, fy * Xc[:, 1] / z + cy], -1).reshape(H, W, 2)
    return (p2 - np.stack([uu, vv], -1)).astype(np.float32)


def main():
    ok = True
    # ---- A1: recover a known SE3 from uniform-depth rigid flow ----
    Dr = np.full((H, W), 0.6, np.float32)
    rvec_t = np.array([0.010, 0.020, 0.005]); tvec_t = np.array([0.030, -0.012, 0.018])
    R_t, _ = cv2.Rodrigues(rvec_t)
    flow = rigid_flow(Dr, R_t, tvec_t)
    ft._raft_flow = lambda *a, **k: flow
    out = ft.rigid_solve_pnp(None, None, Dr, fx, fy, cx, cy, None, None, None, flow_advance_px=0.1, min_inliers=100)
    assert out is not None, "A1: solve returned None"
    T, resid, info = out
    te = np.linalg.norm(T[:3, 3] - tvec_t); Re = np.degrees(np.linalg.norm(cv2.Rodrigues(T[:3, :3] @ R_t.T)[0]))
    print(f"A1 recover SE3 : |t_err|={te:.4f} (t~{np.linalg.norm(tvec_t):.3f}) R_err={Re:.3f}deg reproj_med={info['reproj_med']:.3f}px  (expect tiny)")
    ok &= te < 3e-3 and Re < 0.5 and info['reproj_med'] < 0.5

    # ---- A2: tool blob with independent motion + tool_mask -> still recover the camera ----
    flow2 = flow.copy(); blob = np.zeros((H, W), bool); blob[15:40, 30:55] = True
    flow2[blob] = np.array([25.0, 18.0], np.float32)  # tool moving its own way
    ft._raft_flow = lambda *a, **k: flow2
    out2 = ft.rigid_solve_pnp(None, None, Dr, fx, fy, cx, cy, None, None, None,
                              tool_mask=blob, flow_advance_px=0.1, min_inliers=100)
    assert out2 is not None, "A2: solve returned None"
    T2 = out2[0]; te2 = np.linalg.norm(T2[:3, 3] - tvec_t)
    print(f"A2 tool-masked : |t_err|={te2:.4f}  (expect tiny -> blob excluded, camera recovered)")
    ok &= te2 < 5e-3

    # ---- B: two depth bands collapse to one consensus; blob flagged ----
    Db = np.where(uu < W / 2, 0.4, 0.8).astype(np.float32)   # left near, right far
    bl = np.zeros((H, W), bool); bl[20:40, 34:50] = True; Db[bl] = 0.6
    flB = rigid_flow(Db, np.eye(3), np.array([0.03, 0.0, 0.0]))  # pure camera x-translation
    flB[bl] = flB[bl] + np.array([20.0, 0.0], np.float32)        # blob moves independently
    ft._raft_flow = lambda *a, **k: flB
    dg = np.zeros((H, W, 4), np.float32)                          # 3 distinct DINO regions: near / far / blob
    dg[uu < W / 2, 0] = 1.0; dg[uu >= W / 2, 1] = 1.0; dg[bl] = 0.0; dg[bl, 2] = 1.0
    w, med, mad, scale = ft.depth_pooled_weight(None, None, Db, dg, None, None, None, n_groups=3, mad_c=2.0)
    near = (uu < W / 2) & ~bl; far = (uu >= W / 2) & ~bl
    print(f"B depth-pool   : near w={w[near].mean():.3f} far w={w[far].mean():.3f} (expect ~1, distance-invariant)  "
          f"blob w={w[bl].mean():.3f} (expect low)")
    ok &= w[near].mean() > 0.9 and w[far].mean() > 0.9 and w[bl].mean() < 0.4

    print("SMOKE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
