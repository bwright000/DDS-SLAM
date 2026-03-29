"""
Generate static publication-quality figures matching the DDS-SLAM paper.

Figure 3 equivalent: Side-by-side rendering comparison grid
  - Rows = selected frames, Columns = Ground Truth + methods

Figure 4 equivalent: 3D camera trajectory plot
  - Multiple methods overlaid with ground truth

Usage:
  # Rendering comparison (Figure 3)
  python Addons/generate_paper_figures.py rendering \
    --gt_dir data/trial_3/rgb \
    --render_dirs Output/DDS-SLAM-Results/trail3_depth_anything \
                  Output/DDS-SLAM-Results/trail3_monodepth2 \
    --method_names "Depth Anything V2" "Monodepth2" \
    --frames 0 75 150 \
    --output Addons/sample_results/figure3_comparison.png

  # Trajectory plot (Figure 4)
  python Addons/generate_paper_figures.py trajectory \
    --pose_files output/trail3_da/demo/est_c2w_data.txt \
                 output/trail3_mono/demo/est_c2w_data.txt \
    --method_names "Depth Anything V2" "Monodepth2" \
    --gt_poses path/to/gt_poses.txt \
    --output Addons/sample_results/figure4_trajectory.png
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# Utilities
# ============================================================================

def load_poses_from_txt(filepath):
    """Load c2w poses from text file (12 floats per line)."""
    poses = []
    with open(filepath, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) != 12:
                continue
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :] = np.array(vals).reshape(3, 4)
            poses.append(mat)
    return poses


def find_gt_image(gt_dir, frame_idx):
    """Find ground truth left image for a given frame index."""
    # Semantic-SuPer patterns (0-indexed filenames)
    patterns = [
        os.path.join(gt_dir, f'{frame_idx:06d}-left.png'),
        os.path.join(gt_dir, f'{frame_idx:06d}_left.png'),
    ]
    # StereoMIS patterns (1-indexed filenames: frame 0 → 000001l.png)
    patterns += [
        os.path.join(gt_dir, f'{frame_idx + 1:06d}l.png'),
        os.path.join(gt_dir, f'{frame_idx:06d}l.png'),
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    # Fallback: try listing all left images and index
    all_left = sorted(glob.glob(os.path.join(gt_dir, '*-left.png')))
    if not all_left:
        all_left = sorted(glob.glob(os.path.join(gt_dir, '*_left.png')))
    if not all_left:
        all_left = sorted([f for f in glob.glob(os.path.join(gt_dir, '*l.png'))
                          if not f.endswith('r.png')])
    if frame_idx < len(all_left):
        return all_left[frame_idx]
    return None


def find_rendered_image(render_dir, frame_idx):
    """Find rendered image for a given frame index."""
    path = os.path.join(render_dir, f'{frame_idx:04d}.jpg')
    if os.path.exists(path):
        return path
    return None


def load_image_rgb(path):
    """Load image as RGB float32 [0,1]."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


# ============================================================================
# Figure 3: Rendering Comparison Grid
# ============================================================================

def generate_rendering_figure(gt_dir, render_dirs, method_names, frame_indices,
                               output_path, figsize_per_cell=(2.5, 2.0)):
    """Generate a side-by-side rendering comparison grid like paper Figure 3.

    Layout: rows = frames, columns = [Ground Truth, Method1, Method2, ...]
    """
    n_rows = len(frame_indices)
    n_cols = 1 + len(render_dirs)  # GT + methods

    col_labels = ['Ground Truth'] + method_names

    fig_w = figsize_per_cell[0] * n_cols
    fig_h = figsize_per_cell[1] * n_rows + 0.6  # extra for labels

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, frame_idx in enumerate(frame_indices):
        # Ground truth
        gt_path = find_gt_image(gt_dir, frame_idx)
        if gt_path:
            gt_img = load_image_rgb(gt_path)
            axes[row, 0].imshow(gt_img)
        else:
            axes[row, 0].text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=axes[row, 0].transAxes, fontsize=12)

        # Each method
        for col, render_dir in enumerate(render_dirs, start=1):
            render_path = find_rendered_image(render_dir, frame_idx)
            if render_path:
                render_img = load_image_rgb(render_path)
                if render_img is not None:
                    axes[row, col].imshow(render_img)
                else:
                    axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                      transform=axes[row, col].transAxes, fontsize=12)
            else:
                axes[row, col].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                  transform=axes[row, col].transAxes, fontsize=12)

    # Formatting
    for row in range(n_rows):
        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        # Row label (frame number)
        axes[row, 0].set_ylabel(f'Frame {frame_indices[row]}',
                                fontsize=10, fontweight='bold', rotation=0,
                                labelpad=50, va='center')

    # Column labels
    for col in range(n_cols):
        axes[0, col].set_title(col_labels[col], fontsize=11, fontweight='bold', pad=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Rendering comparison saved: {output_path}")


# ============================================================================
# Figure 4: 3D Camera Trajectory Plot
# ============================================================================

def generate_trajectory_figure(pose_files, method_names, output_path,
                                gt_pose_file=None, views=None):
    """Generate 3D camera trajectory plot like paper Figure 4.

    Shows estimated trajectories from multiple methods, optionally with GT.
    Produces a 2x2 grid: 3D view + XY + XZ + YZ projections.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Load all trajectories
    trajectories = {}
    if gt_pose_file and os.path.exists(gt_pose_file):
        gt_poses = load_poses_from_txt(gt_pose_file)
        gt_positions = np.array([p[:3, 3] for p in gt_poses])
        trajectories['Ground Truth'] = gt_positions

    for pf, name in zip(pose_files, method_names):
        if os.path.exists(pf):
            poses = load_poses_from_txt(pf)
            positions = np.array([p[:3, 3] for p in poses])
            trajectories[name] = positions

    if not trajectories:
        print("No valid pose files found")
        return

    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 3D view
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    for i, (name, pos) in enumerate(trajectories.items()):
        color = 'gray' if name == 'Ground Truth' else colors[i % len(colors)]
        lw = 1.5 if name == 'Ground Truth' else 2.0
        ls = '--' if name == 'Ground Truth' else '-'
        ax3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                 color=color, linewidth=lw, linestyle=ls, label=name)
        ax3d.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, s=30, marker='o')
    ax3d.set_xlabel('X (m)', fontsize=9)
    ax3d.set_ylabel('Y (m)', fontsize=9)
    ax3d.set_zlabel('Z (m)', fontsize=9)
    ax3d.set_title('3D View', fontsize=11, fontweight='bold')
    ax3d.legend(fontsize=8, loc='upper left')

    # 2D projections
    proj_configs = [
        (gs[0, 1], 'XY Projection', 0, 1, 'X (m)', 'Y (m)'),
        (gs[1, 0], 'XZ Projection', 0, 2, 'X (m)', 'Z (m)'),
        (gs[1, 1], 'YZ Projection', 1, 2, 'Y (m)', 'Z (m)'),
    ]

    for gs_pos, title, dim1, dim2, xlabel, ylabel in proj_configs:
        ax = fig.add_subplot(gs_pos)
        for i, (name, pos) in enumerate(trajectories.items()):
            color = 'gray' if name == 'Ground Truth' else colors[i % len(colors)]
            lw = 1.5 if name == 'Ground Truth' else 2.0
            ls = '--' if name == 'Ground Truth' else '-'
            ax.plot(pos[:, dim1], pos[:, dim2],
                   color=color, linewidth=lw, linestyle=ls, label=name)
            ax.scatter(pos[0, dim1], pos[0, dim2], color=color, s=30, marker='o')
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Trajectory plot saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate paper-style figures for DDS-SLAM results'
    )
    subparsers = parser.add_subparsers(dest='command', help='Figure type')

    # Rendering comparison (Figure 3)
    render_parser = subparsers.add_parser('rendering',
        help='Generate side-by-side rendering comparison grid (Figure 3)')
    render_parser.add_argument('--gt_dir', type=str, required=True,
                               help='Ground truth RGB directory')
    render_parser.add_argument('--render_dirs', type=str, nargs='+', required=True,
                               help='Rendered output directories (one per method)')
    render_parser.add_argument('--method_names', type=str, nargs='+', required=True,
                               help='Method names (one per render dir)')
    render_parser.add_argument('--frames', type=int, nargs='+', default=[0, 75, 150],
                               help='Frame indices to show (default: 0 75 150)')
    render_parser.add_argument('--output', type=str,
                               default='Addons/sample_results/figure3_comparison.png',
                               help='Output image path')

    # Trajectory plot (Figure 4)
    traj_parser = subparsers.add_parser('trajectory',
        help='Generate 3D camera trajectory plot (Figure 4)')
    traj_parser.add_argument('--pose_files', type=str, nargs='+', required=True,
                             help='Estimated pose files (est_c2w_data.txt)')
    traj_parser.add_argument('--method_names', type=str, nargs='+', required=True,
                             help='Method names (one per pose file)')
    traj_parser.add_argument('--gt_poses', type=str, default=None,
                             help='Ground truth pose file (optional)')
    traj_parser.add_argument('--output', type=str,
                             default='Addons/sample_results/figure4_trajectory.png',
                             help='Output image path')

    args = parser.parse_args()

    if args.command == 'rendering':
        if len(args.render_dirs) != len(args.method_names):
            print("Error: number of --render_dirs must match --method_names")
            sys.exit(1)
        generate_rendering_figure(
            args.gt_dir, args.render_dirs, args.method_names,
            args.frames, args.output
        )
    elif args.command == 'trajectory':
        if len(args.pose_files) != len(args.method_names):
            print("Error: number of --pose_files must match --method_names")
            sys.exit(1)
        generate_trajectory_figure(
            args.pose_files, args.method_names, args.output,
            gt_pose_file=args.gt_poses
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
