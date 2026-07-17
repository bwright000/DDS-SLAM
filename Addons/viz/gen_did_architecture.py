#!/usr/bin/env python
"""figures/did_architecture.png -- simplified DID-SLAM block diagram (fig:did_arch)
for Contributions, in the DDS-SLAM overview style with REAL C_2/001 thumbnails.

Input is an angled, horizontally-centred cascade of the four modalities
{RGB, MoGe-2 depth, SAM 3 semantics, DINO features}; Output shows {Rendering,
Reconstruction (output depth), Camera pose (trajectory)}. Panel crops come from the
C_2/001 base run's panels.mp4 (3x3, 480x360 cells); the semantic overlay is the
published semantic_instance raster colourised over RGB; DINO features are the
top-centre PCA panel of fig_dino_districts.png. The two flag-gated contributions
are highlighted (blue) against the inherited base (grey).

Local: python Addons/viz/gen_did_architecture.py
"""
import glob
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Affine2D

INK = '#202124'; MUT = '#5f6368'
BASEF = '#eef0f2'; BASEE = '#9aa0a6'
CONF = '#dbe6f3'; CONE = '#4a72b0'
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK})
FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
PANELS = r'F:/Datasets/Benchmarking-20260702T131352Z-4-003/Benchmarking/DDS-SLAM Bench/C2_001_base_s0/panels.mp4'
SNIP = r'F:/Datasets/CRCD-Published/C_2/snippet_001'


def rgbim(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def crop_ar(img, ar=1.5):
    """centre-crop to aspect ratio ar (w/h)."""
    h, w = img.shape[:2]
    if w / h > ar:
        nw = int(h * ar); x = (w - nw) // 2; return img[:, x:x + nw]
    nh = int(w / ar); y = (h - nh) // 2; return img[y:y + nh, :]


# ---- panels.mp4 crops (mid frame) ----
cap = cv2.VideoCapture(PANELS)
mid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2
cap.set(cv2.CAP_PROP_POS_FRAMES, mid); ok, fr = cap.read(); cap.release(); assert ok
depth_in = fr[35:360, 965:1438]; rendered = fr[35:360, 485:958]
depth_out = fr[395:718, 5:478]
POSE = r'c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures\did_inputs\camera_pose.png'
if not os.path.exists(POSE):
    os.system('python "%s"' % os.path.join(os.path.dirname(__file__), 'gen_camera_pose.py'))
pose = plt.imread(POSE)   # clean trajectory line, transparent background

# ---- clean RGB + colourised SAM 3 semantics for the SAME mid frame ----
rgbs = sorted(glob.glob(SNIP + '/rgb/*.png')); sems = sorted(glob.glob(SNIP + '/semantic_instance/*.png'))
rgb = cv2.imread(rgbs[mid])
sem = cv2.imread(sems[mid], cv2.IMREAD_UNCHANGED)
if sem.shape[:2] != rgb.shape[:2]:
    sem = cv2.resize(sem, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
seg = rgb.copy().astype(np.float32)
for cls, col in {1: (0, 0, 235), 2: (0, 200, 0), 3: (235, 235, 0)}.items():   # liver / gallbladder / tool (BGR)
    m = sem == cls
    seg[m] = 0.45 * rgb[m] + 0.55 * np.array(col, np.float32)
seg = seg.astype(np.uint8)

# ---- DINO features: single clean PCA panel (top-centre of fig_dino_districts) ----
dd = cv2.imread(r'c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures\fig_dino_districts.png')
dino = dd[48:315, 693:1147]

INPUTS = [(crop_ar(rgb), 'RGB'), (crop_ar(depth_in), 'MoGe-2 depth'),
          (crop_ar(seg), 'SAM 3 semantics'), (crop_ar(dino), 'DINO features')]

# ============ figure ============
fig, ax = plt.subplots(figsize=(11.8, 4.7), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis('off')


def box(x0, y0, w, h, face, edge, title, sub):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.2,rounding_size=1.6',
                                fc=face, ec=edge, lw=1.4, zorder=3))
    ax.text(x0 + w / 2, y0 + h - 2.6, title, ha='center', va='center', fontsize=9.5, fontweight='bold', zorder=4)
    ax.text(x0 + w / 2, y0 + 2.3, sub, ha='center', va='center', fontsize=7.6, color=MUT, zorder=4)


def thumb(img, x0, x1, y0, y1, label, lx=None, ly=None):
    ax.imshow(rgbim(img), extent=[x0, x1, y0, y1], aspect='auto', zorder=3)
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='white', lw=1.2, zorder=4))
    ax.text(lx if lx else (x0 + x1) / 2, ly if ly else y0 - 1.4, label, ha='center', va='top', fontsize=7.5, color=MUT, zorder=5)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>', mutation_scale=14, lw=1.6, color=MUT, zorder=6))


# ---- Input: angled cascade, horizontally centred around x=14 ----
ax.text(14, 48.5, 'Input', ha='center', fontsize=11, fontweight='bold')
W, H = 15.5, 10.3            # card size (data units), aspect ~1.5
dx, dy = 3.1, 4.7            # per-card offset (up-right)
cx, cy = 6.0, 14.0          # front (bottom) card origin
SK = -0.30                   # x-shear (radians-ish via skew)
for i, (img, lab) in list(enumerate(INPUTS))[::-1]:   # back-to-front
    tr = (Affine2D().skew(SK, 0).translate(cx + i * dx, cy + i * dy)) + ax.transData
    ax.imshow(rgbim(img), extent=[0, W, 0, H], transform=tr, zorder=3 + i)
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, ec='white', lw=1.4, transform=tr, zorder=3 + i))
    # label at the exposed lower-left of each card, on a subtle white pill
    ax.text(cx + i * dx + SK * (i * dy) - 0.8, cy + i * dy + 0.6, lab,
            ha='right', va='bottom', fontsize=7.4, color=INK, zorder=20,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.72))

# ---- DID-SLAM ----
ax.add_patch(FancyBboxPatch((33, 6), 34, 40, boxstyle='round,pad=0.2,rounding_size=1.8',
                            fc='white', ec=INK, lw=1.8, zorder=1))
ax.text(50, 43, 'DID-SLAM', ha='center', fontsize=12, fontweight='bold', zorder=4)
box(35, 30, 30, 9,  BASEF, BASEE, 'Neural-SDF substrate', 'Co-SLAM SDF + deformation warp (DDS-SLAM)')
box(35, 19.5, 30, 9, CONF, CONE, 'Uncertainty head', 'down-weights unexplainable rays')
box(35, 9, 30, 9,   CONF, CONE, 'Motion-attribution gate', 'freezes the pose when still')

# ---- Output ----
ax.text(84, 48.5, 'Output', ha='center', fontsize=11, fontweight='bold')
thumb(rendered,  73, 85.5, 37, 46, 'Rendering')
thumb(depth_out, 73, 85.5, 25, 34, 'Reconstruction')
ax.imshow(pose, extent=[86.5, 99.5, 28, 46], aspect='auto', zorder=3)   # clean line, no border
ax.text(93, 26.6, 'Camera pose', ha='center', va='top', fontsize=7.5, color=MUT, zorder=5)

arrow(27, 32.5, 27); arrow(67.5, 72.5, 27)

# ---- contribution key ----
ax.add_patch(FancyBboxPatch((35, 3.0), 3.4, 2.0, boxstyle='round,pad=0.1,rounding_size=1',
                            fc=CONF, ec=CONE, lw=1.1, zorder=3))
ax.text(39.4, 4.0, 'DID-SLAM', ha='left', va='center', fontsize=8, color=INK)

fig.tight_layout()
os.makedirs(FIG, exist_ok=True)
out = os.path.join(FIG, 'did_architecture.png')
fig.savefig(out, bbox_inches='tight', facecolor='white')
print('wrote', out)
