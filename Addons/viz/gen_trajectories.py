#!/usr/bin/env python
"""Trajectory figures, Sim(3)-aligned, dominant plane.
1) figures/C1_001_traj_est_vs_gt.png, C2_001_traj_est_vs_gt.png -- DDS-SLAM base (bench drop)
   vs GT: the over-travel signature (fig:bench_traj).
2) figures/bdds_traj_c2.png -- the headline: GT vs base vs DID-SLAM on C_2/001, the
   largest-travel held-out sequence, over-travel visibly removed. The base is the
   BENCHMARK-staged run (the thesis's canonical base, Tables results_main/results_sota),
   NOT the controlled ablation base. Panel labels are computed from the trajectories, so
   they cannot drift from the tables."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#202124'; MUT = '#5f6368'; GRID = '#e8eaed'
BLUE = '#4a72b0'; WARM = '#c2571a'; GT = '#8a8f96'
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'axes.edgecolor': MUT,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
GTP = {'C1': r"F:\Datasets\CRCD-Published\C_1\snippet_001\groundtruth.txt",
       'C2': r"F:\Datasets\CRCD-Published\C_2\snippet_001\groundtruth.txt"}
BENCH = r"F:\Datasets\Benchmarking-20260702T131352Z-4-003\Benchmarking\DDS-SLAM Bench"
FINAL = r"C:\Users\benli\OneDrive\Desktop\Results\drive-download-20260711T053215Z-2-001\rect_bestbase_bdds_final_20260709"
REBASE = r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_extra_20260707"

def sim3(est_xyz, gt_xyz):
    n = min(len(est_xyz), len(gt_xyz))
    X = est_xyz[:n].T; Y = gt_xyz[:n].T
    mx = X.mean(1, keepdims=True); my = Y.mean(1, keepdims=True)
    Xc = X - mx; Yc = Y - my
    U, S, Vt = np.linalg.svd(Yc @ Xc.T / n)
    d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    s = (S * [1, 1, d]).sum() / (Xc * Xc).sum() * n
    R = U @ D @ Vt; t = my - s * R @ mx
    return (s * R @ X + t).T, Y.T

def load_est(p): return np.loadtxt(p)[:, [3, 7, 11]]
def load_gt(p): return np.loadtxt(p, comments='#')[:, 1:4]

def ate_rmse(aligned, gt):
    """Sim(3) ATE RMSE in mm (aligned/gt are metres, 1:1 paired)."""
    n = min(len(aligned), len(gt))
    e = np.linalg.norm(gt[:n] - aligned[:n], axis=1)
    return float(np.sqrt((e ** 2).mean()) * 1000.0)


def path_ratio(aligned, gt):
    """est/GT path-length ratio after Sim(3) alignment (scale already applied)."""
    L = lambda z: float(np.linalg.norm(np.diff(z, axis=0), axis=1).sum())
    n = min(len(aligned), len(gt))
    return L(aligned[:n]) / L(gt[:n])


def dom_axes(gt):
    v = gt - gt.mean(0)
    order = np.argsort(-np.var(v, axis=0))
    return order[0], order[1]

def panel(ax, gt, curves, title):
    a, b = dom_axes(gt)
    ax.plot(gt[:, a] * 1000, gt[:, b] * 1000, '--', color=GT, lw=1.6, label='ground truth', zorder=2)
    for lab, est, col in curves:
        ax.plot(est[:, a] * 1000, est[:, b] * 1000, '-', color=col, lw=1.3, alpha=0.95, label=lab, zorder=3)
    ax.scatter(gt[0, a] * 1000, gt[0, b] * 1000, marker='o', s=36, color=INK, zorder=4)
    ax.annotate('start', (gt[0, a] * 1000, gt[0, b] * 1000), textcoords='offset points',
                xytext=(6, 6), fontsize=8, color=MUT)
    ax.set_title(title, fontsize=10, color=INK)
    ax.set_xlabel('dominant axis 1 (mm)', fontsize=9); ax.set_ylabel('dominant axis 2 (mm)', fontsize=9)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_aspect('equal')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='best')

# ---- 1) bench figure: base vs GT (C1, C2) ----
for snip, ratio in [('C1', 8.11), ('C2', 8.66)]:
    gt = load_gt(GTP[snip])
    est = load_est(os.path.join(BENCH, f"{snip}_001_base_s0", "est_c2w_data.txt"))
    ea, ga = sim3(est, gt)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
    panel(ax, ga, [('DDS-SLAM (base)', ea, WARM)], f"{snip}_001: base estimate vs ground truth")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{snip}_001_traj_est_vs_gt.png"),
                                    bbox_inches='tight', facecolor='white')
    print(f"wrote {snip}_001_traj_est_vs_gt.png")
    plt.close(fig)

# ---- 2) headline: C2 GT vs benchmark base vs DID-SLAM ----
gt = load_gt(GTP['C2'])
base = load_est(os.path.join(BENCH, "C2_001_base_s0", "est_c2w_data.txt"))
did = load_est(os.path.join(FINAL, "C2_001_calm_v3gate_s0", "est_c2w_data.txt"))
ba, ga = sim3(base, gt)
oa, _ = sim3(did, gt)
lab = lambda name, a: f'{name} (ATE {ate_rmse(a, ga):.2f} mm, path ratio {path_ratio(a, ga):.1f})'
print('C2 base  ->', lab('base', ba))
print('C2 DID   ->', lab('DID-SLAM', oa))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=200, sharex=True, sharey=True)
panel(ax1, ga, [(lab('DDS-SLAM base', ba), ba, WARM)], 'base')
panel(ax2, ga, [(lab('DID-SLAM', oa), oa, BLUE)], 'DID-SLAM (held-out)')
fig.suptitle('C_2/001 - the largest-travel held-out sequence, Sim(3)-aligned', fontsize=11, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "bdds_traj_c2.png"), bbox_inches='tight', facecolor='white')
print("wrote bdds_traj_c2.png")
