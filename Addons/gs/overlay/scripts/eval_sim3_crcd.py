#!/usr/bin/env python3
"""Sim3 ATE for an EndoGSLAM CRCD run (the GS-migration tracking arbiter).

WHY THIS EXISTS: EndoGSLAM ships rigid-Horn ATE (rotation+translation, NO scale). Our depth is MoGe-2
UP-TO-SCALE, so a rigid ATE is dominated by the ~global scale mismatch and *inverts* A/Bs (the standing
DDS-SLAM trap). This computes a SCALE-AWARE Sim3 (Umeyama) alignment and reports scale + est/GT path-ratio
+ dominant-axis |Pearson| alongside ATE — the only honest CRCD tracking metric (CRCD is sub-SNR).

Reads the trajectory from the run's `params.npz` (cam_unnorm_rots[1,4,F], cam_trans[1,3,F]) — always written
when save_checkpoints=True — so it does NOT depend on est_w2c.txt (which EndoGSLAM only writes on a split).

Usage:
    python scripts/eval_sim3_crcd.py --params experiments/CRCD_base/C1_001/params.npz \
                                     --gt data/CRCD/C1_001/groundtruth.txt
"""
import argparse
import numpy as np


def quat_wxyz_to_R(q):
    """[w,x,y,z] (EndoGSLAM convention) -> 3x3 rotation."""
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_xyzw_to_R(q):
    """[x,y,z,w] (TUM convention) -> 3x3 rotation."""
    x, y, z, w = q
    return quat_wxyz_to_R(np.array([w, x, y, z], dtype=np.float64))


def est_centres_from_params(params_path):
    """Estimated camera centres (frame-0-relative) from the EndoGSLAM checkpoint."""
    d = np.load(params_path)
    rots = d["cam_unnorm_rots"]   # [1,4,F] (w,x,y,z, unnormalised)
    trans = d["cam_trans"]        # [1,3,F]
    F = rots.shape[-1]
    centres = np.zeros((F, 3), np.float64)
    for t in range(F):
        R = quat_wxyz_to_R(rots[0, :, t].astype(np.float64))   # rel w2c rotation
        tr = trans[0, :, t].astype(np.float64)                 # rel w2c translation
        # w2c -> c2w: c = -R^T t   (camera centre in the frame-0 world)
        centres[t] = -R.T @ tr
    return centres


def gt_centres_from_tum(gt_path, n_expected=None):
    """CRCD GT camera centres (frame-0-relative) from groundtruth.txt (TUM)."""
    rows = []
    with open(gt_path) as f:
        for ln in f:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            rows.append(list(map(float, ln.split())))
    c2w = []
    for v in rows:
        tx, ty, tz = v[1:4]
        R = quat_xyzw_to_R(v[4:8])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        c2w.append(T)
    c2w = np.array(c2w)
    inv0 = np.linalg.inv(c2w[0])
    rel = inv0[None] @ c2w                    # frame-0-relative, matches the est convention
    centres = rel[:, :3, 3]
    if n_expected is not None and len(centres) != n_expected:
        print(f"[warn] GT frames ({len(centres)}) != est frames ({n_expected}) — pairing the first min().")
    return centres


def umeyama_sim3(src, dst):
    """Least-squares similarity (s,R,t) aligning src->dst (Umeyama 1991). Returns s,R,t,aligned_src."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    Sc, Dc = src - mu_s, dst - mu_d
    Sigma = (Dc.T @ Sc) / n
    U, Dsv, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = (Sc ** 2).sum() / n
    s = np.trace(np.diag(Dsv) @ S) / (var_s + 1e-12)
    t = mu_d - s * R @ mu_s
    aligned = (s * (R @ src.T)).T + t
    return s, R, t, aligned


def path_length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="EndoGSLAM run params.npz")
    ap.add_argument("--gt", required=True, help="CRCD groundtruth.txt (TUM, 360 rows)")
    ap.add_argument("--mm_scale", type=float, default=1000.0,
                    help="multiplier to report ATE in mm if GT translation is in metres (default 1000)")
    args = ap.parse_args()

    est = est_centres_from_params(args.params)
    gt = gt_centres_from_tum(args.gt, n_expected=len(est))
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]

    s, R, t, aligned = umeyama_sim3(est, gt)
    err = np.linalg.norm(aligned - gt, axis=1)
    ate_mean, ate_max, ate_rmse = err.mean(), err.max(), np.sqrt((err ** 2).mean())
    ratio = path_length(est) / (path_length(gt) + 1e-12)
    dom = int(np.argmax(gt.max(0) - gt.min(0)))                 # dominant-motion axis (largest GT range)
    a, b = aligned[:, dom], gt[:, dom]
    pear = abs(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else float("nan")
    M = args.mm_scale

    print("=" * 64)
    print(f"  Sim3 ATE (CRCD)   frames paired: {n}")
    print("=" * 64)
    print(f"  ATE_mean : {ate_mean:.6f}   ({ate_mean * M:.3f} mm if GT in metres)")
    print(f"  ATE_max  : {ate_max:.6f}   ({ate_max * M:.3f} mm)")
    print(f"  ATE_rmse : {ate_rmse:.6f}   ({ate_rmse * M:.3f} mm)")
    print(f"  Sim3 scale s        : {s:.4f}   (est->GT; far from 1 => up-to-scale mismatch, expected)")
    print(f"  est/GT path-ratio   : {ratio:.4f} (≈1 => comparable motion magnitude; <<1 => est barely moves)")
    print(f"  dominant axis       : {'xyz'[dom]} (largest GT range)")
    print(f"  |Pearson|_dom       : {pear:.4f} (does est SHAPE-track the dominant motion?)")
    print("=" * 64)
    print("  NeRF canon for comparison: base ATE_mean 3.15 mm / +uncert 2.43 mm (CRCD c1_001).")
    print("  Sub-SNR: judge on path-ratio + |Pearson| together, never bare ATE.")


if __name__ == "__main__":
    main()
