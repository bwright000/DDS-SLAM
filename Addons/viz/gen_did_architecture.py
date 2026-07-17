#!/usr/bin/env python
"""figures/did_architecture.png -- simplified DID-SLAM block diagram for the
Contributions section (fig:did_arch), in the Input -> system -> Output style of the
DDS-SLAM overview. The two flag-gated contributions are highlighted (blue) against
the inherited DDS-SLAM base (grey); the corrected bundle-adjustment configuration
is a small tag.

Local: python Addons/viz/gen_did_architecture.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = '#202124'; MUT = '#5f6368'
BASEF = '#f1f3f4'; BASEE = '#9aa0a6'          # inherited base (grey)
CONF = '#dbe6f3'; CONE = '#4a72b0'            # contributions (blue)
IOF = '#fdf3e3'; IOE = '#c2571a'              # input/output (warm)
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"

fig, ax = plt.subplots(figsize=(9.2, 3.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis('off')


def box(x, y, w, h, face, edge, title, sub=None, tw='bold', fs=9.5, r=2.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.2,rounding_size={r}',
                                fc=face, ec=edge, lw=1.4, zorder=3))
    if sub is None:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=fs, fontweight=tw, color=INK, zorder=4)
    else:
        ax.text(x + w / 2, y + h - 5, title, ha='center', va='center',
                fontsize=fs, fontweight=tw, color=INK, zorder=4)
        ax.text(x + w / 2, y + h / 2 - 3.5, sub, ha='center', va='center',
                fontsize=8, color=MUT, zorder=4)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=14,
                                 lw=1.6, color=MUT, zorder=2))


# --- Input ---
box(1, 11, 17, 20, IOF, IOE, 'Input',
    'RGB\nMoGe-2 depth\nSAM 3 semantics\nDINO features', tw='bold', fs=10)

# --- DID-SLAM container ---
ax.add_patch(FancyBboxPatch((26, 3), 44, 36, boxstyle='round,pad=0.2,rounding_size=2.4',
                            fc='white', ec=INK, lw=1.8, zorder=1))
ax.text(48, 36, 'DID-SLAM', ha='center', va='center', fontsize=12, fontweight='bold', color=INK, zorder=4)

box(29, 24.5, 38, 8.5, BASEF, BASEE, 'Neural-SDF substrate',
    'Co-SLAM hash-grid SDF + deformation warp (DDS-SLAM)', fs=9.5)
box(29, 14.5, 38, 8.5, CONF, CONE, 'Uncertainty head',
    'down-weights rays the static field cannot explain', fs=9.5)
box(29, 4.5, 38, 8.5, CONF, CONE, 'Motion-attribution gate',
    'freezes the pose when the camera is still', fs=9.5)

# --- Output ---
box(78, 11, 20, 20, IOF, IOE, 'Output',
    'Camera pose\nDense reconstruction\nRendering', tw='bold', fs=10)

arrow(18.5, 21, 25.5, 21)
arrow(70.5, 21, 77.5, 21)

# legend / caption-in-figure
ax.add_patch(FancyBboxPatch((29, 0.2), 4.6, 2.4, boxstyle='round,pad=0.1,rounding_size=1.2',
                            fc=CONF, ec=CONE, lw=1.2, zorder=3))
ax.text(34.4, 1.4, 'this thesis (flag-gated); the base is left bit-identical',
        ha='left', va='center', fontsize=7.5, color=MUT, zorder=4)

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, 'did_architecture.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('wrote', out)
