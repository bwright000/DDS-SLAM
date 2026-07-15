#!/usr/bin/env python
"""figures/fig_traj3d_compare.png -- 3D Sim(3)-aligned trajectories of every benchmarked
system against ground truth on C_2/001 (the unambiguous-travel sequence).

Each estimate is aligned to GT with the same Umeyama Sim(3) as the evaluation
(gen_trajectories.sim3), then plotted in the GT frame. Two 3D views: all systems
together (left) and DID-SLAM vs base only (right, the headline pair, uncluttered).

Local usage: python Addons/viz/gen_traj3d_compare.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from gen_trajectories import sim3, load_est, load_gt

B = r'F:/Datasets/Benchmarking-20260702T131352Z-4-003/Benchmarking'
GT = r'F:/Datasets/CRCD-Published/C_2/snippet_001/groundtruth.txt'
EST = {
    'DID-SLAM (ours)': (r'C:/Users/benli/OneDrive/Desktop/Results/drive-download-20260711T053215Z-2-001/rect_bestbase_bdds_final_20260709/C2_001_calm_v3gate_s0/est_c2w_data.txt', '#1a73e8', 2.2),
    'DDS-SLAM (base)': (B + r'/DDS-SLAM Bench/C2_001_base_s0/est_c2w_data.txt', '#d93025', 1.4),
    'PERSEUS': (r'F:/Datasets/Resi;ts/PERSEUS_bench_20260708-20260715T072656Z-1-001/PERSEUS_bench_20260708/C2_001_noseg/est_c2w_data.txt', '#188038', 1.4),
    'SNI-SLAM': (r'F:/Datasets/Resi;ts/SNI_fresh_20260714-20260715T072207Z-1-003/SNI_fresh_20260714/C2_001_noconst/est_c2w_data.txt', '#f9ab00', 1.4),
    'SGS-SLAM': (B + r'/SGS-SLAM_CRCD_bench5/C2_001/est_c2w_data.txt', '#9334e6', 1.4),
    'SemGauss-SLAM': (B + r'/SemGauss-SLAM_bench_20260704-20260712T170730Z-2-001/SemGauss-SLAM_bench_20260704/C2_001/est_c2w_data.txt', '#e8710a', 1.4),
    'Semantic-SuPer': (B + r'/SemanticSuPer_crcd_20260703-20260704T145257Z-3-002/SemanticSuPer_crcd_20260703/C2_001/C2_001/est_c2w_data.txt', '#80868b', 1.2),
}


def draw(ax, gt_mm, curves, title):
    c = gt_mm.mean(0)
    r = max(1.2 * np.abs(gt_mm - c).max(), 12)
    ax.plot(*gt_mm.T, color='#202124', lw=2.6, label='ground truth', zorder=10)
    for name, (xyz, colr, lw) in curves.items():
        xyz = xyz.copy()                       # mpl3d does not clip -- NaN points outside the box
        xyz[np.any(np.abs(xyz - c) > r, axis=1)] = np.nan
        ax.plot(*xyz.T, color=colr, lw=lw, alpha=0.9, label=name)
    ax.set_title(title, fontsize=11.5, fontweight='bold', color='#202124', pad=0)
    ax.set_xlabel('x (mm)', fontsize=9, labelpad=-4)
    ax.set_ylabel('y (mm)', fontsize=9, labelpad=-4)
    ax.set_zlabel('z (mm)', fontsize=9, labelpad=-4)
    ax.tick_params(labelsize=7, pad=-2)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def main():
    gt = load_gt(GT)
    gt_mm = (gt - gt.mean(0)) * 1000.0
    curves = {}
    for name, (p, colr, lw) in EST.items():
        raw = np.loadtxt(p, comments='#')
        est = raw[:, [3, 7, 11]] if raw.shape[1] >= 12 else raw[:, 1:4]   # 4x4 rows vs TUM
        n = min(len(est), len(gt))
        aligned, _ = sim3(est[:n], gt[:n])
        curves[name] = ((aligned - gt[:n].mean(0)) * 1000.0, colr, lw)
        print(f'  {name}: {len(est)} poses, {raw.shape[1]} cols, aligned')

    fig = plt.figure(figsize=(15.5, 8.2))
    ATE = {'DID-SLAM (ours)': 3.93, 'DDS-SLAM (base)': 6.78, 'PERSEUS': 4.07,
           'SNI-SLAM': 7.24, 'SGS-SLAM': 7.64, 'SemGauss-SLAM': 8.64, 'Semantic-SuPer': 8.32}
    order = list(EST.keys())
    for i, name in enumerate(order):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')
        draw(ax, gt_mm, {name: curves[name]},
             f"{name}  ({ATE[name]:.2f} mm)")
    # legend cell
    axl = fig.add_subplot(2, 4, 8)
    axl.axis('off')
    import matplotlib.lines as mlines
    axl.legend(handles=[mlines.Line2D([], [], color='#202124', lw=2.6, label='ground truth'),
                        mlines.Line2D([], [], color='#9aa0a6', lw=1.6,
                                      label='estimated trajectory\n(colour per panel)')],
               loc='center', fontsize=12, frameon=False)
    fig.suptitle('C_2/001 -- Sim(3)-aligned trajectories vs ground truth (ATE$_{RMSE}$ per panel)',
                 fontsize=15, fontweight='bold', color='#202124')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), '..', '..', 'figures', 'fig_traj3d_compare.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('wrote', os.path.abspath(out))


if __name__ == '__main__':
    main()
