"""
Generate paper-style 3D trajectory plots matching DDS-SLAM Figure 4.

Features:
  - Rainbow-coloured estimated trajectory (time progression)
  - Dotted grey/black GT reference trajectory
  - Axes in millimeters
  - 3D perspective view matching the paper's angle
  - Multiple methods side by side

Usage:
  python Addons/plot_trajectory_paper_style.py \
    --pose_files output/.../est_c2w_data.txt \
    --method_names "DDS-SLAM (Ours)" \
    --gt_poses data/P2_1/groundtruth.txt \
    --gt_format tum \
    --gt_slice -4000 \
    --output trajectory_comparison.png
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm


def load_poses_12(filepath):
    """Load c2w poses from text file (12 floats per line)."""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(mat[:3, 3])
    return np.array(poses)


def load_poses_tum(filepath, frame_slice=None):
    """Load GT poses from TUM format (timestamp tx ty tz qx qy qz qw)."""
    data = np.loadtxt(filepath)
    positions = data[:, 1:4]
    if frame_slice is not None:
        if frame_slice > 0:
            positions = positions[:frame_slice]
        else:
            positions = positions[frame_slice:]
    return positions


def rainbow_line_3d(ax, positions, linewidth=1.5, alpha=0.9):
    """Plot a 3D line with rainbow colouring based on time progression."""
    n = len(positions)
    colors = cm.rainbow(np.linspace(0, 1, n))

    # Create line segments
    points = positions.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, colors=colors[:-1], linewidth=linewidth, alpha=alpha)
    ax.add_collection3d(lc)

    # Add start/end markers
    ax.scatter(*positions[0], color='green', s=50, marker='^', zorder=5, label='Start')
    ax.scatter(*positions[-1], color='red', s=50, marker='v', zorder=5, label='End')


def plot_single_method(ax, est_positions_mm, gt_positions_mm, method_name, ate_mm=None):
    """Plot one method's trajectory in paper style."""
    # GT trajectory — dotted grey
    ax.plot(gt_positions_mm[:, 0], gt_positions_mm[:, 1], gt_positions_mm[:, 2],
            color='grey', linestyle='--', linewidth=1.0, alpha=0.7, label='Reference')

    # Estimated trajectory — rainbow coloured
    rainbow_line_3d(ax, est_positions_mm, linewidth=1.5)

    # Title
    title = method_name
    if ate_mm is not None:
        title += f'\nATE: {ate_mm:.1f}mm'
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

    # Axes labels
    ax.set_xlabel('x (mm)', fontsize=9)
    ax.set_ylabel('y (mm)', fontsize=9)
    ax.set_zlabel('z (mm)', fontsize=9)

    # Make axes equal scale
    all_pts = np.vstack([est_positions_mm, gt_positions_mm])
    mid = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')


def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-style 3D trajectory plots (DDS-SLAM Figure 4)')
    parser.add_argument('--pose_files', type=str, nargs='+', required=True,
                        help='Estimated pose files (est_c2w_data.txt)')
    parser.add_argument('--method_names', type=str, nargs='+', required=True,
                        help='Method names (one per pose file)')
    parser.add_argument('--gt_poses', type=str, required=True,
                        help='GT pose file (TUM format or 12-float format)')
    parser.add_argument('--gt_format', type=str, default='tum',
                        choices=['tum', '12'],
                        help='GT pose format: tum (timestamp tx ty tz qx qy qz qw) or 12 (3x4 matrix)')
    parser.add_argument('--gt_slice', type=int, default=None,
                        help='Slice GT poses: positive=first N, negative=last N')
    parser.add_argument('--ate_values', type=float, nargs='*', default=None,
                        help='ATE RMSE values in mm (one per method)')
    parser.add_argument('--output', type=str, default='trajectory_paper_style.png')
    parser.add_argument('--elev', type=float, default=25,
                        help='Elevation angle for 3D view')
    parser.add_argument('--azim', type=float, default=-60,
                        help='Azimuth angle for 3D view')
    args = parser.parse_args()

    n_methods = len(args.pose_files)
    assert n_methods == len(args.method_names)

    # Load GT poses
    if args.gt_format == 'tum':
        gt_positions = load_poses_tum(args.gt_poses, args.gt_slice)
    else:
        gt_positions = load_poses_12(args.gt_poses)

    # Convert to mm
    gt_mm = gt_positions * 1000.0

    # Create figure
    if n_methods <= 2:
        fig = plt.figure(figsize=(6 * n_methods, 6))
    else:
        fig = plt.figure(figsize=(5 * n_methods, 5))

    for i, (pose_file, name) in enumerate(zip(args.pose_files, args.method_names)):
        ax = fig.add_subplot(1, n_methods, i + 1, projection='3d')

        # Load estimated poses
        est_positions = load_poses_12(pose_file)
        est_mm = est_positions * 1000.0

        # Match length to GT
        n = min(len(est_mm), len(gt_mm))
        est_mm = est_mm[:n]
        gt_mm_matched = gt_mm[:n]

        # ATE value
        ate = args.ate_values[i] if args.ate_values and i < len(args.ate_values) else None

        plot_single_method(ax, est_mm, gt_mm_matched, name, ate)
        ax.view_init(elev=args.elev, azim=args.azim)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Trajectory plot saved: {args.output}")


if __name__ == '__main__':
    main()
