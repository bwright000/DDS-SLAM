#!/usr/bin/env python
"""figures/did_architecture.png -- simplified DID-SLAM block diagram for the
Contributions section (fig:did_arch), in the Input -> system -> Output style of the
DDS-SLAM overview, using REAL input/output thumbnails from a C_2/001 run.

Thumbnails: Input {RGB, MoGe-2 depth, SAM 3 semantics, DINO features};
Output {Rendering, Reconstruction (output depth), Camera pose (trajectory)}.
The panel crops come from the C_2/001 base run's panels.mp4 (3x3 grid, 480x360
cells); DINO features are the PCA panel of fig_dino_districts.png. The two
flag-gated contributions are highlighted (blue) against the inherited base (grey).

Local: python Addons/viz/gen_did_architecture.py
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = '#202124'; MUT = '#5f6368'
BASEF = '#eef0f2'; BASEE = '#9aa0a6'
CONF = '#dbe6f3'; CONE = '#4a72b0'
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
PANELS = r'F:/Datasets/Benchmarking-20260702T131352Z-4-003/Benchmarking/DDS-SLAM Bench/C2_001_base_s0/panels.mp4'

# --- pull thumbnails ---
cap = cv2.VideoCapture(PANELS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
ok, fr = cap.read(); cap.release()
assert ok
rgb = fr[35:360, 5:478];      rendered = fr[35:360, 485:958];   depth_in = fr[35:360, 965:1438]
depth_out = fr[395:718, 5:478];  seg = fr[395:718, 965:1438];   traj = fr[752:1076, 8:474]
dino_full = cv2.imread(r'c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures\fig_dino_districts.png')
dino = dino_full[70:560, 585:1095]


def rgbim(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots(figsize=(11.5, 4.7), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')


def thumb(img, x0, x1, y0, y1, label):
    ax.imshow(rgbim(img), extent=[x0, x1, y0, y1], aspect='auto', zorder=3)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=1.2, zorder=4))
    ax.text((x0 + x1) / 2, y0 - 1.4, label, ha='center', va='top', fontsize=7.5, color=MUT, zorder=4)


def box(x0, y0, w, h, face, edge, title, sub):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.2,rounding_size=1.6',
                                fc=face, ec=edge, lw=1.4, zorder=3))
    ax.text(x0 + w / 2, y0 + h - 2.6, title, ha='center', va='center', fontsize=9.5, fontweight='bold', zorder=4)
    ax.text(x0 + w / 2, y0 + 2.3, sub, ha='center', va='center', fontsize=7.6, color=MUT, zorder=4)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=14, lw=1.6, color=MUT, zorder=2))


# ---- Input ----
ax.text(13, 48.5, 'Input', ha='center', fontsize=11, fontweight='bold')
thumb(rgb,       1.5, 12.5, 37, 46.5, 'RGB')
thumb(depth_in, 14,   25,   37, 46.5, 'MoGe-2 depth')
thumb(seg,       1.5, 12.5, 23, 32.5, 'SAM 3 semantics')
thumb(dino,     14,   25,   23, 32.5, 'DINO features')

# ---- DID-SLAM ----
ax.add_patch(FancyBboxPatch((31, 6), 34, 40, boxstyle='round,pad=0.2,rounding_size=1.8',
                            fc='white', ec=INK, lw=1.8, zorder=1))
ax.text(48, 43, 'DID-SLAM', ha='center', fontsize=12, fontweight='bold', zorder=4)
box(33, 30, 30, 9,  BASEF, BASEE, 'Neural-SDF substrate', 'Co-SLAM SDF + deformation warp (DDS-SLAM)')
box(33, 19.5, 30, 9, CONF, CONE, 'Uncertainty head', 'down-weights unexplainable rays')
box(33, 9, 30, 9,   CONF, CONE, 'Motion-attribution gate', 'freezes the pose when still')

# ---- Output ----
ax.text(84, 48.5, 'Output', ha='center', fontsize=11, fontweight='bold')
thumb(rendered,  72, 84.5, 37, 46, 'Rendering')
thumb(depth_out, 72, 84.5, 25, 34, 'Reconstruction')
thumb(traj,      86, 98.5, 30, 46, 'Camera pose')

arrow(25.5, 30.5, 27); arrow(65.5, 70.5, 27)

# contribution key
ax.add_patch(FancyBboxPatch((33, 3.2), 3.4, 2.0, boxstyle='round,pad=0.1,rounding_size=1',
                            fc=CONF, ec=CONE, lw=1.1, zorder=3))
ax.text(37.4, 4.2, 'this thesis (flag-gated); base bit-identical when off', ha='left', va='center', fontsize=7, color=MUT)

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, 'did_architecture.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('wrote', out)
