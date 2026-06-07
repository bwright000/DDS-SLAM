"""
Per-frame relative-pose magnitude (translation + rotation) for both GT and
estimated trajectories.  Pure CPU.  Built per Wyrd's diagnostic plan
(workflow wx3zjzfyh, 2026-06-05).

Inputs are 4x4 c2w matrices.  Output is per-frame (translation_mm,
rotation_deg, total_mm_per_frame_inc) — the per-frame INCREMENT, not the
absolute pose.  Plan's Cardinal Rule #1: relative not absolute.

Used by:
  - frame_select.py (tool_high_motion, static, flip selectors)
  - test1_rigid_decomp (compare ΔG_t to camera motion rate)
  - test3_per_segment_health (Pearson on per-frame motion rate)
"""

import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


def relative_motion(c2w_a, c2w_b):
    """Return T_{a->b} given two world poses.  c2w_a, c2w_b are 4x4."""
    # T_{a->b} = c2w_b^-1 @ c2w_a  (if c2w means camera-to-world)
    # We use world-to-camera convention here for inter-frame:
    # T_rel = c2w_a @ inv(c2w_b)  expresses b's pose in a's camera frame
    rel = np.linalg.inv(c2w_b) @ c2w_a
    return rel


def pose_magnitude(T):
    """Decompose a 4x4 relative pose into (translation_mm, rotation_deg)."""
    t = T[:3, 3]
    trans_mm = float(np.linalg.norm(t) * 1000.0)
    R = T[:3, :3]
    # Geodesic rotation angle = arccos((trace(R) - 1) / 2)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(trace)))
    return trans_mm, rot_deg


def per_frame_motion_rate(c2w_list):
    """Per-frame increment magnitudes from a list/array of 4x4 c2w poses.

    Returns dict with arrays of length N-1:
      trans_mm        per-frame translation magnitude
      rot_deg         per-frame rotation magnitude (deg)
      combined_mm     translation + (rotation in rad) * approx_arm  (combined scalar)
    """
    c2w_arr = np.asarray(c2w_list)
    N = c2w_arr.shape[0]
    trans = np.zeros(N - 1)
    rot = np.zeros(N - 1)
    for i in range(N - 1):
        rel = relative_motion(c2w_arr[i], c2w_arr[i + 1])
        trans[i], rot[i] = pose_magnitude(rel)
    # Combined scalar: trans + rot_rad * 0.1m  (rotation about a 10cm arm)
    # Useful for ranking frames by total motion regardless of trans/rot split.
    combined = trans + np.radians(rot) * 100.0  # rad * 100mm = mm
    return {'trans_mm': trans, 'rot_deg': rot, 'combined_mm': combined}


def load_groundtruth_tum(path):
    """Load TUM-format groundtruth.txt -> Nx4x4 c2w matrices.

    Format: timestamp tx ty tz qx qy qz qw
    Lines starting with '#' are skipped.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            v = line.split()
            if len(v) < 8:
                continue
            t = np.array(list(map(float, v[1:4])))
            q = np.array(list(map(float, v[4:8])))
            R = Rotation.from_quat(q).as_matrix()  # qx qy qz qw
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t
            rows.append(T)
    return np.stack(rows, axis=0)


def load_est_c2w_data(path):
    """Load DDS-SLAM's est_c2w_data.txt (one row per frame, 12 floats = 3x4 row-major c2w)."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    N = data.shape[0]
    out = np.zeros((N, 4, 4))
    for i in range(N):
        out[i, :3, :4] = data[i, :12].reshape(3, 4)
        out[i, 3, 3] = 1.0
    return out


def load_poses_from_ckpt(ckpt_path):
    """Load 4x4 c2w from DDS-SLAM checkpoint.pt (key 'pose' is a list/tensor)."""
    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    poses = ckpt['pose']
    if isinstance(poses, list):
        # list of per-frame poses
        if hasattr(poses[0], 'cpu'):
            poses = np.stack([p.cpu().numpy() for p in poses], axis=0)
        else:
            poses = np.stack([np.asarray(p) for p in poses], axis=0)
    else:
        poses = poses.cpu().numpy() if hasattr(poses, 'cpu') else np.asarray(poses)
    # Ensure 4x4
    if poses.shape[-2:] == (3, 4):
        pad = np.zeros((*poses.shape[:-2], 1, 4))
        pad[..., 0, 3] = 1.0
        poses = np.concatenate([poses, pad], axis=-2)
    return poses


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', type=str, help='est_c2w_data.txt path')
    ap.add_argument('--gt', type=str, help='groundtruth.txt path (TUM)')
    ap.add_argument('--out_csv', type=str, default='motion_rate.csv')
    args = ap.parse_args()

    if args.est:
        c2w = load_est_c2w_data(args.est)
        motion = per_frame_motion_rate(c2w)
        print(f'est: {len(c2w)} poses, motion stats:')
        for k, v in motion.items():
            print(f'  {k}: mean={v.mean():.3f}, median={np.median(v):.3f}, max={v.max():.3f}')
    if args.gt:
        c2w = load_groundtruth_tum(args.gt)
        motion = per_frame_motion_rate(c2w)
        print(f'gt: {len(c2w)} poses, motion stats:')
        for k, v in motion.items():
            print(f'  {k}: mean={v.mean():.3f}, median={np.median(v):.3f}, max={v.max():.3f}')
