"""
Visualize 3D point clouds from RGB + depth + poses in Rerun.

Backprojects each frame into a colored point cloud using camera intrinsics
and poses, then logs to Rerun for interactive 3D viewing.

Usage:
  python Addons/visualize_pointcloud.py \
    --rgb_dir data/CRCD/C1_001/video_frames \
    --rgb_pattern "*l.png" \
    --depth_dir data/CRCD/C1_001/depth \
    --poses data/CRCD/C1_001/groundtruth.txt \
    --fx 1096.70 --fy 1096.70 --cx 622.81 --cy 383.13 \
    --png_depth_scale 100 \
    --output output/crcd_c1_001_pointclouds.rrd \
    --skip 10

  StereoMIS example:
  python Addons/visualize_pointcloud.py \
    --rgb_dir F:/Datasets/StereoMIS/StereoMIS/P2_1/video_frames \
    --rgb_pattern "*l.png" \
    --depth_dir Output/StereoMIS_depth_maps_RAFT \
    --poses F:/Datasets/StereoMIS/StereoMIS/P2_1/groundtruth.txt \
    --fx 516.95 --fy 516.86 --cx 302.29 --cy 257.49 \
    --png_depth_scale 100 \
    --output output/stereomis_pointclouds.rrd \
    --skip 50 --max_frames 4000
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import rerun as rr
from tqdm import tqdm
from scipy.spatial.transform import Rotation


def load_poses_tum(pose_file):
    """Load TUM-format poses: timestamp tx ty tz qx qy qz qw"""
    poses = []
    with open(pose_file) as f:
        for line in f:
            if line.startswith('#'):
                continue
            vals = list(map(float, line.strip().split()))
            if len(vals) >= 8:
                tx, ty, tz = vals[1], vals[2], vals[3]
                qx, qy, qz, qw = vals[4], vals[5], vals[6], vals[7]
                r = Rotation.from_quat([qx, qy, qz, qw])
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :3] = r.as_matrix()
                c2w[:3, 3] = [tx, ty, tz]
                poses.append(c2w)
    return poses


def load_poses_identity(n, flip_yz=True):
    """Create identity poses (for DDS-SLAM default)."""
    poses = []
    for _ in range(n):
        c2w = np.eye(4, dtype=np.float32)
        if flip_yz:
            c2w[1, 1] = -1
            c2w[2, 2] = -1
        poses.append(c2w)
    return poses


def load_poses_est(est_file):
    """Load estimated poses from est_c2w_data.txt (12 floats per line)."""
    poses = []
    with open(est_file) as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) == 12:
                c2w = np.eye(4, dtype=np.float32)
                c2w[:3, :] = np.array(vals).reshape(3, 4)
                poses.append(c2w)
    return poses


def backproject(rgb, depth_m, c2w, fx, fy, cx, cy, subsample=4):
    """Backproject RGB+depth into colored 3D points."""
    H, W = depth_m.shape
    u, v = np.meshgrid(np.arange(0, W, subsample), np.arange(0, H, subsample))
    u = u.flatten().astype(np.float32)
    v = v.flatten().astype(np.float32)

    d = depth_m[v.astype(int), u.astype(int)]
    colors = rgb[v.astype(int), u.astype(int)]

    valid = d > 0.001
    u, v, d, colors = u[valid], v[valid], d[valid], colors[valid]

    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    z = d

    pts_cam = np.stack([x, y, z], axis=-1)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    pts_world = (R @ pts_cam.T).T + t

    return pts_world, colors


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D point clouds in Rerun')
    parser.add_argument('--rgb_dir', type=str, required=True)
    parser.add_argument('--rgb_pattern', type=str, default='*l.png')
    parser.add_argument('--depth_dir', type=str, required=True)
    parser.add_argument('--depth_pattern', type=str, default='*.png')
    parser.add_argument('--poses', type=str, default=None,
                        help='TUM-format groundtruth.txt or est_c2w_data.txt')
    parser.add_argument('--pose_format', type=str, default='auto',
                        choices=['auto', 'tum', 'est', 'identity'],
                        help='Pose format (auto-detects from file)')
    parser.add_argument('--fx', type=float, required=True)
    parser.add_argument('--fy', type=float, required=True)
    parser.add_argument('--cx', type=float, required=True)
    parser.add_argument('--cy', type=float, required=True)
    parser.add_argument('--png_depth_scale', type=float, default=100)
    parser.add_argument('--output', type=str, default='pointclouds.rrd')
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--subsample', type=int, default=4,
                        help='Spatial subsampling of point cloud')
    parser.add_argument('--frame_slice', type=str, default=None,
                        help='Slice frames, e.g. "-4000:" or ":1500"')
    args = parser.parse_args()

    # Load images and depth
    rgb_files = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_pattern)))
    depth_files = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_pattern)))

    if args.frame_slice:
        rgb_files = eval(f"rgb_files[{args.frame_slice}]")
        depth_files = eval(f"depth_files[{args.frame_slice}]")

    n = min(len(rgb_files), len(depth_files))
    if args.max_frames:
        n = min(n, args.max_frames)
    rgb_files = rgb_files[:n]
    depth_files = depth_files[:n]
    print(f"RGB: {len(rgb_files)}, Depth: {len(depth_files)}")

    # Load poses
    if args.poses and os.path.exists(args.poses):
        fmt = args.pose_format
        if fmt == 'auto':
            with open(args.poses) as f:
                first = f.readline()
                if first.startswith('#'):
                    first = f.readline()
                n_cols = len(first.strip().split())
                fmt = 'tum' if n_cols == 8 else 'est' if n_cols == 12 else 'tum'

        if fmt == 'tum':
            all_poses = load_poses_tum(args.poses)
            if args.frame_slice:
                all_poses = eval(f"all_poses[{args.frame_slice}]")
            poses = all_poses[:n]
        elif fmt == 'est':
            poses = load_poses_est(args.poses)[:n]
        else:
            poses = load_poses_identity(n)
        print(f"Poses: {len(poses)} ({fmt} format)")
    else:
        poses = load_poses_identity(n)
        print(f"Poses: identity ({n} frames)")

    # Rerun
    rr.init('pointcloud_viewer', spawn=False)
    rr.save(args.output)

    for i in tqdm(range(0, n, args.skip), desc="Building point clouds"):
        rgb = cv2.imread(rgb_files[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        depth_raw = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        depth_m = depth_raw.astype(np.float32) / args.png_depth_scale

        c2w = poses[i] if i < len(poses) else np.eye(4)

        pts, colors = backproject(rgb, depth_m, c2w,
                                  args.fx, args.fy, args.cx, args.cy,
                                  subsample=args.subsample)

        rr.set_time('frame', sequence=i)
        rr.log('world/pointcloud', rr.Points3D(pts * 1000, colors=colors, radii=[0.3]))
        rr.log('world/camera', rr.Points3D([c2w[:3, 3] * 1000],
               colors=[(255, 255, 0)], radii=[2.0]))

    print(f"\nSaved to {args.output}")
    print(f"Open: python -m rerun {args.output}")


if __name__ == '__main__':
    main()
