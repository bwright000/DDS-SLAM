"""
Frame-selection utilities for the DDS-SLAM CRCD diagnostic plan.

Selectors (per Wyrd's plan):
  static          camera moving with real baseline, tissue NOT manipulated
  tool_static     tool present but kinematically not moving
  tool_high_motion  high kinematic tool motion (for Test 2)
  flip            large deformation / flip frames
  class_dominated  frames where a specific class dominates pixel fraction
  rgb_stable      low frame-to-frame photometric change (used with tool_high_motion
                  to find "tool moves but render is static" frames — Test 2's "tell")

Inputs:
  - per-frame motion rate (from motion_rate.py)
  - per-frame semantic masks (uint8 PNG, values 0/1/2/3 for bg/Liver/GB/Tool)
  - per-frame RGB (for rgb_stable diff)
"""

import numpy as np
import cv2
import glob
import os
from pathlib import Path


def class_fractions_per_frame(semantic_dir, pattern='*.png', class_ids=None):
    """For each semantic mask, compute the per-class pixel fraction.
    Returns dict: { 'bg': arr[N], 'liver': arr[N], 'gallbladder': arr[N], 'tool': arr[N] }
    Class IDs: 0=bg, 1=Liver, 2=Gallbladder, 3=Tool (CRCD convention).
    """
    if class_ids is None:
        class_ids = {0: 'bg', 1: 'liver', 2: 'gallbladder', 3: 'tool'}
    files = sorted(glob.glob(os.path.join(semantic_dir, pattern)))
    N = len(files)
    out = {name: np.zeros(N) for name in class_ids.values()}
    for i, f in enumerate(files):
        sem = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if sem is None:
            continue
        total = sem.size
        for cls_id, cls_name in class_ids.items():
            out[cls_name][i] = float((sem == cls_id).sum()) / total
    return out


def select_static_camera(motion_rate_dict, percentile=20):
    """Frames in the lowest-motion percentile of CAMERA motion.
    Camera-side static — for Test 0 (depth-error floor on static stretches).
    """
    rate = motion_rate_dict['combined_mm']
    threshold = np.percentile(rate, percentile)
    mask = rate <= threshold
    return np.where(mask)[0]


def select_camera_moving_real_baseline(motion_rate_dict, min_mm=0.5, max_mm=5.0):
    """Frames where camera moves enough for parallax but not so much it's a flip.
    For Test 0 — need 'real baseline'.
    """
    rate = motion_rate_dict['combined_mm']
    mask = (rate >= min_mm) & (rate <= max_mm)
    return np.where(mask)[0]


def select_tool_high_motion(tool_kinematic_rate, percentile=80):
    """Frames in the top percentile of TOOL motion.
    NOTE: tool_kinematic_rate comes from da Vinci tool kinematics, NOT from
    the camera trajectory.  If tool kinematics aren't separately available,
    fallback: high tool pixel motion (centroid drift in semantic mask).
    """
    threshold = np.percentile(tool_kinematic_rate, percentile)
    return np.where(tool_kinematic_rate >= threshold)[0]


def tool_centroid_motion(semantic_dir, pattern='*.png', tool_class=3):
    """Fallback for tool motion: centroid drift of the tool mask across frames.
    Returns per-frame increment (N-1,) in pixels.
    """
    files = sorted(glob.glob(os.path.join(semantic_dir, pattern)))
    N = len(files)
    centroids = np.zeros((N, 2))
    for i, f in enumerate(files):
        sem = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if sem is None:
            centroids[i] = np.nan
            continue
        tool_mask = (sem == tool_class)
        if tool_mask.sum() < 10:
            centroids[i] = np.nan
            continue
        ys, xs = np.where(tool_mask)
        centroids[i] = [xs.mean(), ys.mean()]
    diffs = np.diff(centroids, axis=0)
    motion = np.linalg.norm(diffs, axis=1)
    motion = np.nan_to_num(motion, nan=0.0)
    return motion


def select_class_dominated(class_fracs, dom_class, threshold=0.30):
    """Frames where `dom_class` exceeds the given pixel-fraction threshold."""
    rate = class_fracs[dom_class]
    return np.where(rate >= threshold)[0]


def select_rgb_stable(rgb_dir, pattern='*.png', stability_percentile=20):
    """Frames with low photometric change vs previous frame.
    For Test 2's "tool moves but render is static" detection.
    Returns indices in lowest-N-percentile of per-frame photometric diff.
    """
    files = sorted(glob.glob(os.path.join(rgb_dir, pattern)))
    N = len(files)
    diff = np.zeros(N - 1)
    prev = None
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        img = img.astype(np.float32)
        if prev is not None:
            diff[i - 1] = float(np.abs(img - prev).mean())
        prev = img
    threshold = np.percentile(diff, stability_percentile)
    return np.where(diff <= threshold)[0]


def select_tool_moves_but_render_static(tool_motion, rgb_diff,
                                         tool_percentile=80, rgb_percentile=20):
    """The Test-2 target frames: tool moves a lot AND render barely changes.
    These are 'the tell' — where the deformation field is most aggressively
    cancelling tool motion.
    Returns intersection of tool_high_motion and rgb_stable frames.
    """
    tool_th = np.percentile(tool_motion, tool_percentile)
    rgb_th = np.percentile(rgb_diff, rgb_percentile)
    n = min(len(tool_motion), len(rgb_diff))
    mask = (tool_motion[:n] >= tool_th) & (rgb_diff[:n] <= rgb_th)
    return np.where(mask)[0]


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--semantic_dir', type=str, required=True)
    ap.add_argument('--rgb_dir', type=str, default=None)
    args = ap.parse_args()
    cf = class_fractions_per_frame(args.semantic_dir)
    print(f'class fractions (per-frame avg):')
    for k, v in cf.items():
        print(f'  {k}: mean={v.mean():.3f}, max={v.max():.3f}')
    print(f'tool-dominated frames (>30%): {len(select_class_dominated(cf, "tool"))}')
    print(f'liver-dominated frames (>30%): {len(select_class_dominated(cf, "liver"))}')
    print(f'gallbladder-dominated frames (>30%): {len(select_class_dominated(cf, "gallbladder"))}')
    tool_motion = tool_centroid_motion(args.semantic_dir)
    print(f'tool centroid motion: mean={tool_motion.mean():.2f}px, max={tool_motion.max():.2f}px')
