#!/usr/bin/env python3
"""Synthetic gate for fit_rotation_field (constrained-C: 3-dof rotational flow-field fit).
Pure numpy, no GPU. Frame geometry = rectified CRCD (1280x720, fx 1096.7).
  R1 pure rotation + RAFT-level noise      -> omega recovered within 5%
  R2 rotation + 20% tool outliers (+15px)  -> MAD-IRLS still recovers within 8%
  R3 pure lateral translation over sloped depth -> rotation model CANNOT absorb the
     parallax structure (median resid stays up) = de-rotation leaves translation evidence
  R4 zero flow + noise                     -> |omega| ~ 0 (no phantom rotation)
Run:  python Addons/regression/test_rotation_fit.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from Addons.motion.flow_track import fit_rotation_field  # noqa: E402

W, H, F = 1280, 720, 1096.7
rng = np.random.default_rng(0)
ys, xs = np.mgrid[0:H:16, 0:W:16]
PTS = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
C = (W / 2.0, H / 2.0)


def rot_flow(pts, w):
    x = pts[:, 0] - C[0]; y = pts[:, 1] - C[1]
    u = w[0] * x * y / F - w[1] * (F + x * x / F) + w[2] * y
    v = w[0] * (F + y * y / F) - w[1] * x * y / F - w[2] * x
    return np.stack([u, v], 1)


def check(name, cond, detail):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}: {detail}")
    return cond


ok = True
w_true = np.array([0.002, -0.004, 0.001])

# R1 pure rotation + noise
fl = rot_flow(PTS, w_true) + rng.normal(0, 0.2, (len(PTS), 2))
w, _, _, _ = fit_rotation_field(PTS, fl, F, C)
err = np.linalg.norm(w - w_true) / np.linalg.norm(w_true)
ok &= check('R1 pure-rotation recovery', err < 0.05, f'rel err {err * 100:.2f}% (bar 5%)')

# R2 rotation + 20% coherent tool outliers
fl2 = rot_flow(PTS, w_true) + rng.normal(0, 0.2, (len(PTS), 2))
out = rng.random(len(PTS)) < 0.20
fl2[out] += np.array([15.0, -9.0])
w2, _, _, inl2 = fit_rotation_field(PTS, fl2, F, C)
err2 = np.linalg.norm(w2 - w_true) / np.linalg.norm(w_true)
ok &= check('R2 20% tool outliers', err2 < 0.08,
            f'rel err {err2 * 100:.2f}% (bar 8%), outliers kept as inliers: {int((inl2 & out).sum())}')

# R3 lateral translation over sloped depth: u = F*tx/Z, Z 40->80mm across the image.
# The uniform part aliases into pan (narrow-FOV ambiguity, accepted); the DEPTH-STRUCTURED
# deviation must SURVIVE de-rotation -- that residual is the translation/scene evidence.
Z = 40.0 + 40.0 * PTS[:, 0] / W
fl3 = np.stack([F * 0.7 / Z, np.zeros(len(PTS))], 1) + rng.normal(0, 0.2, (len(PTS), 2))  # 9.6..19.2px
w3, _, res3, inl3 = fit_rotation_field(PTS, fl3, F, C)
med3 = float(np.median(res3))
ok &= check('R3 translation parallax survives', med3 > 0.5, f'median resid {med3:.2f}px (bar >0.5)')

# R4 zero flow + noise -> no phantom rotation
fl4 = rng.normal(0, 0.2, (len(PTS), 2))
w4, _, _, _ = fit_rotation_field(PTS, fl4, F, C)
deg4 = float(np.degrees(np.linalg.norm(w4)))
ok &= check('R4 still frame', deg4 < 0.02, f'|omega| {deg4:.4f}deg (bar <0.02)')

print('ALL PASS' if ok else 'FAILURES ABOVE')
sys.exit(0 if ok else 1)
