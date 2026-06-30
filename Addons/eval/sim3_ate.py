#!/usr/bin/env python3
"""Canonical Sim3 (scale-corrected) ATE for DDS-SLAM trajectories.

WHY THIS EXISTS — the bug it fixes:
  tools/eval_ate.py:align() (the pipeline default that writes output.txt) is RIGID Horn with
  NO scale, and pose_evaluation(..., scale=1, ...) passes no scale. On up-to-scale monocular
  depth (MoGe-2: est is ~8x the GT metric scale), that rigid ATE is DOMINATED by the global
  scale mismatch, not tracking quality. On CRCD C1_001 it reports ~38-48mm and even INVERTS an
  A/B (reports +uncertainty as WORSE) while the scale-corrected truth is the opposite.

  This script reports the metrics our methodology actually mandates for up-to-scale / sub-SNR
  surgical sequences (see memory feedback_sim3_ate_misleading_subsnr):
    - Sim3 ATE  (Umeyama WITH scale)         -> tracking quality after removing global scale
    - recovered scale s                       -> how far off the monocular scale is
    - est/GT path-length ratio (after s)      -> over/under-travel
    - dominant-axis |Pearson|  (scale-FREE)   -> shape-tracking, immune to any scale artefact
  It also prints the RIGID ATE so you can SEE the scale confound vs the pipeline number.

Pairing: est line = 12 floats (3x4 row-major c2w) or 16 (4x4); translation = [3,7,11]. GT = TUM
  (timestamp tx ty tz qx qy qz qw, '#' comments). Equal length => 1:1 (est[i]<->GT[i], which is
  how datasets/dataset.py associates by frame index). Unequal => uniform resample with a warning.

Usage:
  python Addons/eval/sim3_ate.py --est run/demo/est_c2w_data.txt \
      --gt data/CRCD/C1_001/groundtruth.txt --name "CRCD C1_001 +uncert" --out sim3_metrics.txt
"""
import argparse
import numpy as np


def load_est(path):
    M = []
    for ln in open(path):
        v = [float(x) for x in ln.split()]
        if len(v) >= 16:        # 4x4 row-major
            M.append([v[3], v[7], v[11]])
        elif len(v) >= 12:      # 3x4 row-major
            M.append([v[3], v[7], v[11]])
    return np.asarray(M, float)


def load_gt_tum(path):
    out = []
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        v = ln.split()
        out.append([float(v[1]), float(v[2]), float(v[3])])
    return np.asarray(out, float)


def umeyama(m, d, with_scale=True):
    """align m -> d (both [N,3]). with_scale=True => Sim3 (recovers s); False => rigid SE3
    (== tools/eval_ate.py:align). Returns (aligned[N,3], scale)."""
    mc, dc = m.mean(0), d.mean(0)
    mm, dd = m - mc, d - dc
    H = mm.T @ dd
    U, S, Vt = np.linalg.svd(H)
    sgn = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, sgn]) @ U.T
    s = (S * np.array([1, 1, sgn])).sum() / (mm * mm).sum() if with_scale else 1.0
    return (s * (R @ m.T)).T + (dc - s * R @ mc), s


def pathlen(x):
    return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())


def evaluate(est, gt):
    ne, ng = len(est), len(gt)
    if ne == ng:
        e, g, paired = est, gt, "1:1"
    else:
        n = min(ne, ng)
        e = est[np.round(np.linspace(0, ne - 1, n)).astype(int)]
        g = gt[np.round(np.linspace(0, ng - 1, n)).astype(int)]
        paired = f"resampled(est={ne},gt={ng}->{n})"
    al_s, s = umeyama(e, g, True)
    al_r, _ = umeyama(e, g, False)
    err_s = np.linalg.norm(al_s - g, axis=1) * 1000.0
    err_r = np.linalg.norm(al_r - g, axis=1) * 1000.0
    dom = int(np.argmax(g.max(0) - g.min(0)))
    # Pearson on the SIM3-ALIGNED est (al_s), NOT the raw est. Sim3 alignment includes a ROTATION, so the
    # raw est axis e[:,dom] is a DIFFERENT physical axis than g[:,dom] -> the raw correlation is rotation-
    # confounded and reads ~0 even when the aligned shape tracks GT well (the E3_005 'Pearson 0.023' artefact;
    # the aligned value was 0.60). al_s is in GT's frame, so per-axis correlation is meaningful. Also report
    # the mean over x,y,z (more robust than betting on one axis).
    pear = float(abs(np.corrcoef(al_s[:, dom], g[:, dom])[0, 1]))
    pear_xyz = float(np.mean([abs(np.corrcoef(al_s[:, k], g[:, k])[0, 1]) for k in range(3)]))
    return {
        "paired": paired, "n": len(e), "scale": float(s),
        "gt_path_mm": pathlen(g) * 1000.0, "path_ratio": pathlen(e) * s / (pathlen(g) + 1e-12),
        "sim3_rmse": float(np.sqrt((err_s ** 2).mean())), "sim3_mean": float(err_s.mean()),
        "sim3_median": float(np.median(err_s)), "sim3_max": float(err_s.max()),
        "pearson_dom": pear, "pearson_xyz": pear_xyz, "rigid_rmse": float(np.sqrt((err_r ** 2).mean())),
        "rigid_mean": float(err_r.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--name', default='')
    ap.add_argument('--out', default=None, help='also append the report to this file')
    a = ap.parse_args()
    r = evaluate(load_est(a.est), load_gt_tum(a.gt))
    lines = [
        f"==== Sim3 ATE: {a.name} ====",
        f"  pairing            : {r['paired']}  (N={r['n']})",
        f"  recovered scale s  : {r['scale']:.4f}   (est ~{1/r['scale']:.1f}x GT metric scale)",
        f"  GT path length     : {r['gt_path_mm']:.1f} mm",
        f"  est/GT path ratio  : {r['path_ratio']:.2f}",
        f"  Sim3 ATE  rmse/mean/median/max : "
        f"{r['sim3_rmse']:.2f} / {r['sim3_mean']:.2f} / {r['sim3_median']:.2f} / {r['sim3_max']:.2f} mm",
        f"  |Pearson| dom axis : {r['pearson_dom']:.3f}   (Sim3-ALIGNED dom axis; scale-free shape tracking)",
        f"  |Pearson| mean xyz : {r['pearson_xyz']:.3f}   (Sim3-aligned, mean over x,y,z)",
        f"  [pipeline RIGID ATE rmse/mean : {r['rigid_rmse']:.1f} / {r['rigid_mean']:.1f} mm  "
        f"<- scale-confounded, do NOT headline]",
    ]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        with open(a.out, 'a') as fh:
            fh.write(txt + "\n")


if __name__ == '__main__':
    main()
