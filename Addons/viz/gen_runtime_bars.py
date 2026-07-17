#!/usr/bin/env python
"""figures/runtime_bars.png -- per-frame runtime of every benchmarked system, with
its GPU, against the 30 fps real-time threshold (fig:runtime).

All values recovered from the run logs (see the Runtime paragraph, Section
Discussion): representative s/frame per system. SemGauss-SLAM is shown on BOTH the
Tesla T4 (its C1_001 run) and the A100-80GB (its other runs) to make the ~2.3x
device confound visible. GPU is colour-coded AND labelled on each bar, so identity
never rests on colour alone (blue/orange is colourblind-safe; unknown is neutral grey).

Local: python Addons/viz/gen_runtime_bars.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#202124'; MUT = '#5f6368'; GRID = '#e8eaed'
BLUE = '#4a72b0'; WARM = '#c2571a'; GREY = '#9aa0a6'
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'axes.edgecolor': MUT,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"

# (label, s/frame, gpu)  -- slowest first (drawn top-down)
ROWS = [
    ('DID-SLAM (ours)',      26.4, 'T4'),
    ('DDS-SLAM (base)',      21.5, 'T4'),
    ('SemGauss-SLAM',        14.0, 'T4'),
    ('SemGauss-SLAM',         6.1, 'A100'),
    ('SGS-SLAM',              5.4, 'A100'),
    ('SNI-SLAM',              2.35, 'A100'),
    ('PERSEUS',               0.65, 'T4'),
]
COL = {'T4': WARM, 'A100': BLUE, 'unknown': GREY}
RT = 1.0 / 30.0   # real-time threshold, s/frame

fig, ax = plt.subplots(figsize=(8.2, 3.4), dpi=200)
y = list(range(len(ROWS)))[::-1]   # top row highest
for yi, (lab, v, gpu) in zip(y, ROWS):
    ax.barh(yi, v, height=0.62, color=COL[gpu], edgecolor='white', linewidth=0.8, zorder=3)
    ax.text(v * 1.08, yi, f'{v:.1f}\\,s  ({gpu})'.replace('\\,', ' '),
            va='center', ha='left', fontsize=8.5, color=INK)

ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in ROWS], fontsize=9)
ax.set_xscale('log')
ax.set_xlim(RT * 0.6, 60)
ax.set_xlabel('seconds per frame (log scale)', fontsize=9)

# real-time reference line
ax.axvline(RT, color=INK, lw=1.3, ls='--', zorder=2)
ax.text(RT, len(ROWS) - 0.35, 'real-time\n30 fps', fontsize=8, color=INK,
        ha='center', va='top')

# legend by GPU
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=WARM, label='Tesla T4 (15\\,GB)'.replace('\\,', ' ')),
                   Patch(facecolor=BLUE, label='A100')],
          fontsize=8, frameon=False, loc='lower right')

ax.grid(axis='x', color=GRID, lw=0.7); ax.set_axisbelow(True)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, 'runtime_bars.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('wrote', out)
