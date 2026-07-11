#!/usr/bin/env python
"""figures/bench_psnr.png + bench_lpips.png -- per-snippet render comparison,
DDS-SLAM base vs SGS-SLAM (fig:bench_bars). Values verbatim from the thesis
benchmark tables (tab:bench_dds, tab:bench_sgs); DDS C3 pending -> omitted."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#202124'; MUT = '#5f6368'; GRID = '#e8eaed'
BLUE = '#4a72b0'; GREEN = '#3d8a5f'
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'axes.edgecolor': MUT,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
SNIPS = ['C1_001', 'C2_001', 'C3_001', 'E3_005', 'G3_001']
DDS_PSNR = [25.91, 22.57, None, 24.93, 29.68]
SGS_PSNR = [22.45, 17.52, 18.78, 19.15, 18.40]
DDS_LPIPS = [0.448, 0.552, None, 0.483, 0.317]
SGS_LPIPS = [0.315, 0.504, 0.508, 0.397, 0.447]

def bars(vals_a, vals_b, name, ylab, better):
    fig, ax = plt.subplots(figsize=(8.6, 2.9), dpi=200)
    x = np.arange(len(SNIPS)); w = 0.38
    for xo, vals, col, lab in [(-w/2, vals_a, BLUE, 'DDS-SLAM (base)'), (w/2, vals_b, GREEN, 'SGS-SLAM')]:
        for i, v in enumerate(vals):
            if v is None:
                ax.text(x[i] + xo, 0.02 * max(filter(None, vals_a + vals_b)), 'pending',
                        rotation=90, ha='center', va='bottom', fontsize=8, color=MUT)
                continue
            ax.bar(x[i] + xo, v, w * 0.92, color=col, zorder=3)
            ax.text(x[i] + xo, v, f'{v:.2f}'.rstrip('0').rstrip('.') if v < 1 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8, color=INK)
        ax.bar(np.nan, np.nan, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([s.replace('_001', '').replace('_005', '') for s in SNIPS], fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(f'{ylab} per snippet ({better} is better)', fontsize=10, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False, ncols=2, loc='upper right')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches='tight', facecolor='white')
    print("wrote", name)

bars(DDS_PSNR, SGS_PSNR, 'bench_psnr.png', 'PSNR', 'higher')
bars(DDS_LPIPS, SGS_LPIPS, 'bench_lpips.png', 'LPIPS', 'lower')
