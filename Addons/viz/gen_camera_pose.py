#!/usr/bin/env python
"""figures/did_inputs/camera_pose.png -- the DID-SLAM C_2/001 camera trajectory as a
clean line with NO axes, grid, box or background (transparent PNG). For the
'Camera pose' output panel of the DID-SLAM architecture figure.

Sim(3)-aligned estimate, projected on the ground-truth dominant plane, coloured by
normalised time; start marked. No graphic behind the line.

Local: python Addons/viz/gen_camera_pose.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

FINAL = r"C:\Users\benli\OneDrive\Desktop\Results\drive-download-20260711T053215Z-2-001\rect_bestbase_bdds_final_20260709\C2_001_calm_v3gate_s0\est_c2w_data.txt"
GT = r"F:\Datasets\CRCD-Published\C_2\snippet_001\groundtruth.txt"
OUT = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures\did_inputs\camera_pose.png"


def sim3(est, gt):
    n = min(len(est), len(gt)); X = est[:n].T; Y = gt[:n].T
    mx = X.mean(1, keepdims=True); my = Y.mean(1, keepdims=True)
    Xc = X - mx; Yc = Y - my
    U, S, Vt = np.linalg.svd(Yc @ Xc.T / n)
    d = np.sign(np.linalg.det(U @ Vt)); D = np.diag([1, 1, d])
    s = (S * [1, 1, d]).sum() / (Xc * Xc).sum() * n
    R = U @ D @ Vt; t = my - s * R @ mx
    return (s * R @ X + t).T, Y.T


est = np.loadtxt(FINAL)[:, [3, 7, 11]]
gt = np.loadtxt(GT, comments='#')[:, 1:4]
ea, ga = sim3(est, gt)
v = ga - ga.mean(0); order = np.argsort(-np.var(v, axis=0)); a, b = order[0], order[1]
xy = np.column_stack([ea[:, a], ea[:, b]]) * 1000.0   # mm

# colour the line by normalised time
segs = np.stack([xy[:-1], xy[1:]], axis=1)
lc = LineCollection(segs, cmap='viridis', linewidth=3.2, capstyle='round')
lc.set_array(np.linspace(0, 1, len(segs)))

fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=220)
ax.add_collection(lc)
ax.scatter(xy[0, 0], xy[0, 1], s=42, color='#202124', zorder=5)   # start
ax.set_aspect('equal'); ax.autoscale()
m = 0.06 * (xy.max(0) - xy.min(0))
ax.set_xlim(xy[:, 0].min() - m[0], xy[:, 0].max() + m[0])
ax.set_ylim(xy[:, 1].min() - m[1], xy[:, 1].max() + m[1])
ax.axis('off')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, bbox_inches='tight', pad_inches=0.05, transparent=True)
print('wrote', OUT)
