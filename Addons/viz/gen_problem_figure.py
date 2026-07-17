#!/usr/bin/env python
"""figures/problem_figure.png -- the motivating "problem" figure for the Introduction
(fig:problem). A single C_2/001 endoscopic frame annotated with the three sources of
image motion a surgical tracker must disentangle: instrument motion, tissue
deformation, and camera motion. Frame-to-model tracking attributes all of it to the
camera pose; separating them is the problem this thesis addresses.

Annotations are placed from the published semantic mask (tool / gallbladder
centroids), so the labels land on the real structures. Conceptual figure: the
motion arrows are illustrative, not computed flow.

Local: python Addons/viz/gen_problem_figure.py
"""
import glob
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = '#202124'
BLUE = '#4a72b0'     # instrument
GREEN = '#1a9850'    # tissue deformation (distinct from the warm tissue + blue/purple)
PURP = '#8452a8'     # camera
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
SNIP = r'F:/Datasets/CRCD-Published/C_2/snippet_001'
IDX = 365

rgbs = sorted(glob.glob(SNIP + '/rgb/*.png')); sems = sorted(glob.glob(SNIP + '/semantic_instance/*.png'))
rgb = cv2.cvtColor(cv2.imread(rgbs[IDX]), cv2.COLOR_BGR2RGB)
sem = cv2.imread(sems[IDX], cv2.IMREAD_UNCHANGED)
if sem.shape[:2] != rgb.shape[:2]:
    sem = cv2.resize(sem, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
H, W = rgb.shape[:2]


def centroid(mask):
    ys, xs = np.where(mask)
    return xs.mean(), ys.mean()


tool_left = (sem == 3) & (np.arange(W)[None, :] < 640)
gb = sem == 2
tcx, tcy = centroid(tool_left)
gcx, gcy = centroid(gb)

fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=200)
ax.imshow(rgb); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

# subtle tints so the structures read
for m, c in [(tool_left | ((sem == 3) & (np.arange(W)[None, :] >= 640)), BLUE), (gb, GREEN)]:
    ov = np.zeros((H, W, 4)); rgba = matplotlib.colors.to_rgba(c)
    ov[m] = (rgba[0], rgba[1], rgba[2], 0.22)
    ax.imshow(ov)


def label(txt, tx, ty, px, py, col):
    ax.annotate(txt, xy=(px, py), xytext=(tx, ty), fontsize=10, fontweight='bold', color='white',
                ha='center', va='center', zorder=10,
                bbox=dict(boxstyle='round,pad=0.35', fc=col, ec='white', lw=1.2),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=2.2,
                                shrinkA=6, shrinkB=4, mutation_scale=16))


# 1 instrument motion — curved motion arrow on the left grasper
ax.add_patch(FancyArrowPatch((tcx - 70, tcy + 60), (tcx + 90, tcy - 40), connectionstyle='arc3,rad=0.3',
                             arrowstyle='-|>', mutation_scale=20, lw=3, color=BLUE, zorder=9))
label('instrument\nmotion', 200, 120, tcx, tcy, BLUE)

# 2 tissue deformation — stretch arrows on the LOWER gallbladder body, clear of the instrument
gdx, gdy = gcx + 45, gcy + 135
for ex, ey in [(gdx - 120, gdy + 100), (gdx + 150, gdy + 20)]:
    ax.add_patch(FancyArrowPatch((gdx, gdy), (ex, ey), arrowstyle='-|>',
                                 mutation_scale=18, lw=3, color=GREEN, zorder=9))
label('tissue\ndeformation', W - 200, H - 90, gdx + 40, gdy + 60, GREEN)

# 3 camera motion — global; dashed inner frame + corner arrows + label
ax.add_patch(FancyBboxPatch((26, 26), W - 52, H - 52, boxstyle='round,pad=0,rounding_size=8',
                            fill=False, ec=PURP, lw=2.2, ls=(0, (6, 5)), zorder=6))
for (ax0, ay0, ax1, ay1) in [(90, 70, 40, 40), (W - 90, 70, W - 40, 40)]:
    ax.add_patch(FancyArrowPatch((ax0, ay0), (ax1, ay1), arrowstyle='-|>', mutation_scale=15, lw=2.4, color=PURP, zorder=7))
ax.text(W / 2, 66, 'camera motion?  (the whole view shifts)', fontsize=10, fontweight='bold',
        color='white', ha='center', va='center', zorder=10,
        bbox=dict(boxstyle='round,pad=0.35', fc=PURP, ec='white', lw=1.2))

# problem statement strip
ax.text(W / 2, H - 26, 'Three sources of image motion. A static-map tracker blames the camera for all of it.',
        fontsize=10.5, color='white', ha='center', va='center', zorder=10,
        bbox=dict(boxstyle='round,pad=0.4', fc=(0, 0, 0, 0.62), ec='none'))

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, 'problem_figure.png')
fig.savefig(out, bbox_inches='tight', facecolor='white', pad_inches=0.02)
print('wrote', out, '| tool', (round(tcx), round(tcy)), 'gb', (round(gcx), round(gcy)))
