#!/usr/bin/env python3
"""Synthetic smoke for the GATE v2 vote detector core (_vote_fit) -- no RAFT/DINO/GPU needed.

Simulates the pooled per-region votes (median flow, depth, centroid) for known camera motions on a
depth-diverse scene and checks the detector's DECISION, ATTRIBUTION and TOOL-EXCLUSION:
  V1 STILL      : pure RAFT-floor noise             -> moving=False
  V2 TURN       : uniform slide, depth-blind        -> moving=True, turn dominant, slide/zoom small
  V3 SLIDE      : depth-scaled slide                -> moving=True, detected (attribution split with
                  turn is rank-deficient on flat depth; decision must be right regardless)
  V4 ZOOM       : depth-scaled radial expansion     -> moving=True, zoom dominant ('getting bigger')
  V5 TOOL       : still camera + 3/12 regions moving independently -> moving=False AND the tool
                  regions get low trust, consensus regions keep high trust (exclude-don't-veto)
  V6 TURN+TOOL  : the E3 killer -- camera turning WHILE tool moves -> moving=True (the old gate's
                  conjunction froze here; the vote must not)
Run: python Addons/regression/test_region_vote.py   (exit 0 = all pass)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Addons.motion.flow_track import _vote_fit

W, H = 1280, 720
rng = np.random.RandomState(0)
N = 12
# region layout: centroids on a grid, depths diverse 0.3-0.9m (the depth signature that separates types)
gx, gy = np.meshgrid(np.linspace(200, 1080, 4), np.linspace(120, 600, 3))
pK = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
zK = np.linspace(0.3, 0.9, N).astype(np.float32); rng.shuffle(zK)
ok = np.ones(N, bool)
zmed = float(np.median(zK)); Zn = zmed / zK
ctr = np.array([W / 2, H / 2]); rad = (pK - ctr) / float(np.median(np.linalg.norm(pK - ctr, axis=1)))
noise = lambda s=0.15: rng.randn(N, 2).astype(np.float32) * s   # RAFT pooled-median noise ~0.15px

def run(name, fK, want_moving, checks=(), floor=0.5):
    info, wk = _vote_fit(fK.astype(np.float32), zK, pK, ok, W, H, still_floor_px=floor)
    okk = info['moving'] == want_moving and all(c(info, wk) for c in checks)
    print(f"{name:10s} moving={info['moving']} (want {want_moving}) mag={info['mag']:.2f} "
          f"turn={info['turn']:.2f} slide={info['slide']:.2f} zoom={info['zoom']:.2f} "
          f"inl={info['n_inliers']}/{info['n_valid']} -> {'PASS' if okk else 'FAIL'}")
    return okk

ok_all = True
# V1 STILL: pure noise
ok_all &= run("V1 still", noise(), False)
# V2 TURN: uniform 3px slide (depth-blind) + noise
ok_all &= run("V2 turn", np.tile([3.0, 0.0], (N, 1)) + noise(), True,
              [lambda i, w: i['turn'] + i['slide'] > 2.0])
# V3 SLIDE: depth-scaled 3px slide at median plane + noise
ok_all &= run("V3 slide", np.stack([3.0 * Zn, np.zeros(N)], 1) + noise(), True,
              [lambda i, w: i['mag'] > 2.0])
# V4 ZOOM: radial expansion, depth-scaled ('features getting bigger')
ok_all &= run("V4 zoom", 3.0 * rad * Zn[:, None] + noise(), True,
              [lambda i, w: i['zoom'] > max(1.0, 0.5 * (i['turn'] + i['slide']))])
# V5 TOOL on still camera: 3 regions move 8px on their own -> still + tool excluded
f5 = noise(); tool = [2, 5, 9]; f5[tool] += [8.0, 4.0]
ok_all &= run("V5 tool", f5, False,
              [lambda i, w: max(w[k] for k in tool) < 0.5,
               lambda i, w: np.median([w[k] for k in range(N) if k not in tool]) > 0.9])
# V6 TURN + TOOL (the E3 killer: the old conjunction froze here)
f6 = np.tile([3.0, 0.0], (N, 1)) + noise(); f6[tool] += [-6.0, 5.0]
ok_all &= run("V6 turn+tool", f6, True,
              [lambda i, w: max(w[k] for k in tool) < 0.5])

print(">>> REGION-VOTE " + ("PASS" if ok_all else "FAIL"))
sys.exit(0 if ok_all else 1)
