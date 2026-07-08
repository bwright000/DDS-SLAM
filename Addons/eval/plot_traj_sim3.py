#!/usr/bin/env python3
"""Sim3-aligned est-vs-GT trajectory plot (the per-run VISUAL for tracking-only arms).

Loads an estimated trajectory (16-float 4x4 c2w per line, or TUM 8-col) and a GT
trajectory (TUM 8-col, or 16-float), Sim3-aligns est->GT with the SAME Umeyama
(with scale) used by Addons/eval/sim3_ate.py, and writes a 2-panel PNG:
  left  = the dominant GT plane (the two highest-variance axes), aligned est vs GT
  right = per-axis position vs frame (aligned est solid, GT dashed)

Counts must match 1:1 (run after the runbook's GT prefix-truncation guard, which
guarantees it). Standalone on purpose: numpy+matplotlib only.

Usage:
  python plot_traj_sim3.py --est est_c2w_data.txt --gt groundtruth.txt \
      --out traj_plot.png --name "CRCD E3_005 PERSEUS"
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_xyz(path):
    """Translation per row from TUM (8 col: t x y z qx qy qz qw) or 16-float 4x4 (row-major)."""
    xyz = []
    for ln in open(path, encoding='utf-8', errors='ignore'):
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        v = [float(x) for x in ln.split()]
        if len(v) == 16:
            xyz.append([v[3], v[7], v[11]])
        elif len(v) >= 8:
            xyz.append(v[1:4])
        elif len(v) == 12:
            xyz.append([v[3], v[7], v[11]])
        else:
            raise ValueError(f"{path}: unsupported row of {len(v)} values")
    a = np.asarray(xyz, dtype=np.float64)
    if a.ndim != 2 or len(a) < 2:
        raise ValueError(f"{path}: no usable trajectory ({a.shape})")
    return a


def umeyama_sim3(src, dst):
    """Similarity transform (s, R, t) minimising ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s if var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--est', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='trajectory')
    ap.add_argument('--units_mm', action='store_true', default=True,
                    help='annotate axes in mm (positions are metres; plotted x1000)')
    a = ap.parse_args()

    est, gt = load_xyz(a.est), load_xyz(a.gt)
    n = min(len(est), len(gt))
    if len(est) != len(gt):
        print(f"[plot] WARN est={len(est)} gt={len(gt)} -> plotting first {n} of each (prefix)")
    est, gt = est[:n], gt[:n]

    s, R, t = umeyama_sim3(est, gt)
    est_al = (s * (R @ est.T)).T + t
    rms = float(np.sqrt(((est_al - gt) ** 2).sum(1).mean()))

    # dominant GT plane = the two highest-variance axes
    order = np.argsort(gt.var(0))[::-1]
    i, j = int(order[0]), int(order[1])
    lbl = 'xyz'
    K = 1000.0  # metres -> mm

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    ax1.plot(gt[:, i] * K, gt[:, j] * K, 'k--', lw=1.4, label='GT')
    ax1.plot(est_al[:, i] * K, est_al[:, j] * K, 'tab:blue', lw=1.2,
             label='est (Sim3-aligned)')
    ax1.scatter([gt[0, i] * K], [gt[0, j] * K], c='k', s=25, zorder=5)
    ax1.set_xlabel(f'{lbl[i]} (mm)'); ax1.set_ylabel(f'{lbl[j]} (mm)')
    ax1.set_title(f'dominant plane ({lbl[i]}{lbl[j]})'); ax1.axis('equal')
    ax1.legend(); ax1.grid(alpha=0.3)

    f = np.arange(n)
    for k, c in zip(range(3), ('tab:red', 'tab:green', 'tab:blue')):
        ax2.plot(f, gt[:, k] * K, c=c, ls='--', lw=1.0)
        ax2.plot(f, est_al[:, k] * K, c=c, lw=1.1, label=lbl[k])
    ax2.set_xlabel('frame'); ax2.set_ylabel('position (mm)')
    ax2.set_title('per-axis (est solid, GT dashed)')
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f'{a.name}  |  Sim3 scale s={s:.4g}  ATE_rmse={rms * K:.2f} mm  N={n}')
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"[plot] {a.out}  (s={s:.4g}, ATE_rmse={rms * K:.2f} mm, N={n})")


if __name__ == '__main__':
    main()
