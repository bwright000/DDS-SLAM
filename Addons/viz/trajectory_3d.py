#!/usr/bin/env python3
"""3D trajectory comparison for DDS-SLAM CRCD runs (Sim3-aligned to GT) -- meeting figure.

Reuses the CANONICAL Sim3 aligner (Addons/eval/sim3_ate.umeyama) so the plotted paths match the
reported Sim3 ATE exactly (scale-corrected; the only honest comparison on up-to-scale MoGe depth --
the pipeline's rigid output.txt is scale-confounded and must not headline).

Layout (one row): [3D GT+est_0] [3D GT+est_1] ... [dominant-axis position vs frame, all overlaid].
Each 3D panel auto-scales so a tightly-tracking run and an inflated/jittery one are each legible;
the dominant-axis time-series overlays them directly (the scale-free shape / |Pearson| view).
Per-run legend carries Sim3 ATE mean / path-ratio / |Pearson|.

Usage:
  python Addons/viz/trajectory_3d.py --gt <groundtruth.txt> \
    --est best=<est.txt> improved=<est.txt> --out traj.png --title "CRCD C1_001"
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))
from sim3_ate import load_est, load_gt_tum, umeyama, evaluate  # noqa: E402

PALETTE = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728', '#9467bd']  # green, orange, blue, red, purple


def _pair(e, gt):
    """1:1 if equal length, else uniform resample (mirrors sim3_ate.evaluate)."""
    ne, ng = len(e), len(gt)
    if ne == ng:
        return e, gt
    n = min(ne, ng)
    return (e[np.round(np.linspace(0, ne - 1, n)).astype(int)],
            gt[np.round(np.linspace(0, ng - 1, n)).astype(int)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt', required=True)
    ap.add_argument('--est', nargs='+', required=True, help='NAME=PATH pairs (order = panel order)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--title', default='CRCD -- Sim3-aligned trajectories')
    a = ap.parse_args()

    gt = load_gt_tum(a.gt)
    runs = []
    for spec in a.est:
        name, path = spec.split('=', 1)
        e = load_est(path)
        ee, gg = _pair(e, gt)
        al, s = umeyama(ee, gg, True)   # Sim3-align est -> GT frame (scale-corrected)
        r = evaluate(e, gt)
        runs.append((name, al, gg, r))

    dom = int(np.argmax(gt.max(0) - gt.min(0)))   # dominant (largest-extent) axis
    n3d = len(runs)
    fig = plt.figure(figsize=(6.0 * n3d + 6.5, 6.0))

    # one 3D panel per run: GT (black) + that run (colour), each auto-scaled
    for i, (name, al, gg, r) in enumerate(runs):
        ax = fig.add_subplot(1, n3d + 1, i + 1, projection='3d')
        ax.plot(gg[:, 0], gg[:, 1], gg[:, 2], color='k', lw=2.5, label='GT', zorder=10)
        ax.plot(al[:, 0], al[:, 1], al[:, 2], color=PALETTE[i % len(PALETTE)], lw=1.4, alpha=0.9,
                label=name)
        ax.scatter(*gg[0], color='k', s=30, marker='o')       # start
        ax.set_title(f"{name}\nSim3 ATE {r['sim3_mean']:.2f} mm | path-ratio {r['path_ratio']:.2f} | "
                     f"|Pearson| {r['pearson_dom']:.3f}", fontsize=10)
        ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
        ax.legend(loc='upper left', fontsize=9)

    # dominant-axis position vs frame: the scale-free SHAPE-tracking view (all runs overlaid on GT)
    ax2 = fig.add_subplot(1, n3d + 1, n3d + 1)
    gg0 = runs[0][2]
    ax2.plot(np.arange(len(gg0)), gg0[:, dom], color='k', lw=2.8, label='GT', zorder=10)
    for i, (name, al, gg, r) in enumerate(runs):
        f = np.linspace(0, len(al) - 1, len(gg0)).astype(int)
        ax2.plot(np.arange(len(gg0)), al[f, dom], color=PALETTE[i % len(PALETTE)], lw=1.2, alpha=0.9,
                 label=f"{name} (|r|={r['pearson_dom']:.3f})")
    ax2.set_title(f'Dominant axis (axis {dom}) vs frame', fontsize=11)
    ax2.set_xlabel('frame'); ax2.set_ylabel(f'position axis {dom} (m)')
    ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

    fig.suptitle(a.title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(a.out, dpi=140, bbox_inches='tight')
    print('wrote', a.out)
    for name, al, gg, r in runs:
        print(f"  {name:10s} ATE {r['sim3_mean']:.2f} mm  ratio {r['path_ratio']:.2f}  "
              f"|Pearson| {r['pearson_dom']:.3f}")


if __name__ == '__main__':
    main()
