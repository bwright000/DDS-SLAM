"""
Generate a multi-panel video from DDS-SLAM inputs and outputs.

Arranges panels in a grid layout and renders frame-by-frame as a video.
Each panel is optional — provide as many or few as needed.

Panels:
  --depth_input_dir    Depth from depth model (colormapped)
  --depth_output_dir   Depth output from SLAM (colormapped)
  --rgb_input_dir      Input RGB frames
  --rgb_output_dir     Rendered RGB from SLAM
  --seg_dir            Segmentation masks
  --trajectory         Trajectory plot (EST vs GT, 3D rotating)

Usage:
  python Addons/generate_video.py \
    --rgb_input_dir output/trail3/rendered_all \
    --rgb_output_dir output/trail3/rendered_all \
    --depth_input_dir data/Super/depth \
    --trajectory_est output/trail3/demo/est_c2w_data.txt \
    --trajectory_gt data/Super/groundtruth.txt \
    --output video_output.mp4 \
    --fps 30 \
    --skip 1

  Minimal example (just RGB input + rendered):
  python Addons/generate_video.py \
    --rgb_input_dir data/frames \
    --rgb_output_dir output/rendered \
    --output comparison.mp4
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm


def load_sorted_images(directory, pattern="*.png"):
    """Load sorted image paths from a directory."""
    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        # Try jpg
        paths = sorted(glob.glob(os.path.join(directory, "*.jpg")))
    return paths


def load_image(path, target_size=None):
    """Load an image, resize if needed, return as RGB uint8."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    # Handle RGBA → convert to RGB (OpenCV loads as BGRA, we want RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 2:
        # Grayscale or depth
        pass
    # Ensure 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if target_size and img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    return img


def colormap_depth(depth_path, target_size=None, png_depth_scale=None):
    """Load a depth map and apply colormap."""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        return None
    depth_f = depth.astype(np.float32)
    if png_depth_scale:
        depth_f = depth_f / png_depth_scale
    # Normalize to [0, 255]
    valid = depth_f[depth_f > 0]
    if len(valid) == 0:
        colored = np.zeros((*depth.shape[:2], 3), dtype=np.uint8)
    else:
        vmin, vmax = valid.min(), valid.max()
        depth_norm = np.clip((depth_f - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)
        depth_norm[depth_f <= 0] = 0
        colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if target_size and colored.shape[:2] != target_size:
        colored = cv2.resize(colored, (target_size[1], target_size[0]))
    return colored


def load_trajectory(est_path, gt_path=None, gt_frames=None):
    """Load estimated and optionally GT trajectories."""
    est_xyz = []
    with open(est_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            vals = list(map(float, line.strip().split()))
            if len(vals) == 8:
                est_xyz.append(vals[1:4])
            elif len(vals) == 12:
                c2w = np.array(vals).reshape(3, 4)
                est_xyz.append(c2w[:3, 3])
    est_xyz = np.array(est_xyz)

    gt_xyz = None
    if gt_path and os.path.exists(gt_path):
        gt_raw = []
        with open(gt_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                vals = list(map(float, line.strip().split()))
                if len(vals) >= 4:
                    gt_raw.append(vals[1:4])
        gt_xyz = np.array(gt_raw)
        if gt_frames:
            gt_xyz = gt_xyz[gt_frames]
        gt_xyz = gt_xyz[:len(est_xyz)]

    return est_xyz, gt_xyz


def horn_align(model, data):
    """Horn's method: find rigid transform from model to data (both 3xN)."""
    model_zc = model - model.mean(1, keepdims=True)
    data_zc = data - data.mean(1, keepdims=True)
    W = model_zc @ data_zc.T
    U, d, Vh = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = data.mean(1, keepdims=True) - rot @ model.mean(1, keepdims=True)
    aligned = rot @ model + trans
    return aligned.T


def overlay_mask_on_rgb(rgb, mask, alpha=0.5):
    """Blend coloured mask over RGB."""
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], axis=-1)
    # Only overlay where mask is not background (not dark grey)
    has_mask = mask.sum(axis=-1) > 100
    blended = rgb.copy()
    blended[has_mask] = (rgb[has_mask] * (1 - alpha) + mask[has_mask] * alpha).astype(np.uint8)
    return blended


def render_trajectory_frame(est_xyz, gt_xyz, current_frame, panel_size, azim_offset=0, align=True):
    """Render a 3D trajectory plot as an image array."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    dpi = 100
    h, w = panel_size
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Horn-align estimated to GT if requested and GT is available
    if align and gt_xyz is not None and len(gt_xyz) == len(est_xyz):
        est_aligned = horn_align(est_xyz.T, gt_xyz.T)
    else:
        est_aligned = est_xyz

    est_mm = est_aligned * 1000
    n = current_frame + 1

    # GT trajectory (full, grey dashed)
    if gt_xyz is not None:
        gt_mm = gt_xyz * 1000
        ax.plot(gt_mm[:, 0], gt_mm[:, 1], gt_mm[:, 2],
                '--', color='#888888', linewidth=1.5, alpha=0.7)

    # Estimated trail up to current frame (jet colormap)
    if n > 1:
        t = np.linspace(0, 1, n)
        segments = np.array([[est_mm[i], est_mm[i + 1]] for i in range(n - 1)])
        colors = plt.cm.jet(t[:-1])
        lc = Line3DCollection(segments, colors=colors, linewidths=1.5)
        ax.add_collection3d(lc)

    # Current position marker
    ax.scatter(*est_mm[current_frame], color='yellow', s=40, zorder=5, edgecolors='black')

    # Axis labels
    ax.set_xlabel('x (mm)', fontsize=7)
    ax.set_ylabel('y (mm)', fontsize=7)
    ax.set_zlabel('z (mm)', fontsize=7)
    ax.tick_params(labelsize=6)

    # Set consistent axis limits from full trajectory
    pad = 5
    for setter, idx in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
        all_pts = est_mm
        if gt_xyz is not None:
            all_pts = np.vstack([est_mm, gt_xyz * 1000])
        setter(all_pts[:, idx].min() - pad, all_pts[:, idx].max() + pad)

    # Rotating view
    ax.view_init(elev=25, azim=-55 + azim_offset)

    # Style
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=0.5)
    fig.canvas.draw()

    # Convert to numpy array
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    if img.shape[:2] != panel_size:
        img = cv2.resize(img, (panel_size[1], panel_size[0]))
    return img


def add_label(img, text, font_scale=0.6, thickness=1):
    """Add a text label at the top of an image."""
    labeled = img.copy()
    # Semi-transparent black bar
    bar_h = int(25 * font_scale / 0.6)
    labeled[:bar_h] = (labeled[:bar_h].astype(np.float32) * 0.4).astype(np.uint8)
    # Text
    cv2.putText(labeled, text, (5, bar_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness,
                cv2.LINE_AA)
    return labeled


def main():
    parser = argparse.ArgumentParser(
        description='Generate multi-panel video from DDS-SLAM inputs/outputs')

    # Panel inputs (all optional)
    parser.add_argument('--rgb_input_dir', type=str, help='Directory of input RGB frames')
    parser.add_argument('--rgb_input_pattern', type=str, default='*_gt.png',
                        help='Glob pattern for input RGB (default: *_gt.png)')
    parser.add_argument('--rgb_output_dir', type=str, help='Directory of rendered RGB frames')
    parser.add_argument('--rgb_output_pattern', type=str, default='[0-9]*.png',
                        help='Glob pattern for rendered RGB (default: [0-9]*.png, excludes _gt)')
    parser.add_argument('--depth_input_dir', type=str, help='Directory of input depth maps')
    parser.add_argument('--depth_output_dir', type=str, help='Directory of output depth maps')
    parser.add_argument('--seg_dir', type=str, help='Directory of segmentation masks')
    parser.add_argument('--trajectory_est', type=str, help='Path to est_c2w_data.txt')
    parser.add_argument('--trajectory_gt', type=str, help='Path to groundtruth.txt')
    parser.add_argument('--trajectory_raw', action='store_true',
                        help='Add a second trajectory panel with NO Horn alignment '
                             '(shows raw est vs GT in their native coord frames)')
    parser.add_argument('--gt_frame_slice', type=str, default=None,
                        help='Slice GT frames, e.g. "-4000:" or ":4000"')
    parser.add_argument('--input_frame_slice', type=str, default=None,
                        help='Slice ALL input-image dirs (rgb/depth/seg) with same Python slice, '
                             'e.g. "-4000:" for back-4000, "::2" for masks at half rate')
    parser.add_argument('--seg_frame_slice', type=str, default=None,
                        help='Override slice just for segmentation (StereoMIS masks run at half rate). '
                             'e.g. "-2000:"')

    # Output
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--skip', type=int, default=1, help='Frame skip (use with rendered_all skip)')
    parser.add_argument('--panel_height', type=int, default=360)
    parser.add_argument('--panel_width', type=int, default=480)
    parser.add_argument('--png_depth_scale', type=float, default=None)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--rotation_speed', type=float, default=0.05,
                        help='Trajectory rotation speed (degrees per frame)')
    args = parser.parse_args()

    panel_size = (args.panel_height, args.panel_width)

    def _slice(paths, expr):
        if not expr:
            return paths
        return eval(f"paths[{expr}]")

    # Discover panels
    panels = []
    panel_data = {}

    if args.rgb_input_dir:
        paths = sorted(glob.glob(os.path.join(args.rgb_input_dir, args.rgb_input_pattern)))
        paths = _slice(paths, args.input_frame_slice)
        if paths:
            panels.append('Input RGB')
            panel_data['Input RGB'] = paths
            print(f"Input RGB: {len(paths)} frames")

    if args.rgb_output_dir:
        paths = sorted([p for p in glob.glob(os.path.join(args.rgb_output_dir, args.rgb_output_pattern))
                        if '_gt' not in os.path.basename(p)])
        if not paths:
            paths = sorted(glob.glob(os.path.join(args.rgb_output_dir, '*.jpg')))
        if paths:
            panels.append('Rendered RGB')
            panel_data['Rendered RGB'] = paths
            print(f"Rendered RGB: {len(paths)} frames")

    if args.depth_input_dir:
        paths = load_sorted_images(args.depth_input_dir)
        paths = _slice(paths, args.input_frame_slice)
        if paths:
            panels.append('Input Depth')
            panel_data['Input Depth'] = paths
            print(f"Input Depth: {len(paths)} frames")

    if args.depth_output_dir:
        paths = sorted(p for p in glob.glob(os.path.join(args.depth_output_dir, '*.png'))
                       if '_gt' not in os.path.basename(p))
        if paths:
            panels.append('Output Depth')
            panel_data['Output Depth'] = paths
            print(f"Output Depth: {len(paths)} frames")

    if args.seg_dir:
        paths = load_sorted_images(args.seg_dir)
        paths = _slice(paths, args.seg_frame_slice or args.input_frame_slice)
        if paths:
            panels.append('Segmentation')
            panel_data['Segmentation'] = paths
            print(f"Segmentation: {len(paths)} frames")
            # Also add overlay panel if we have rendered RGB
            if 'Rendered RGB' in panel_data:
                panels.append('Seg Overlay')
                panel_data['Seg Overlay'] = (panel_data['Rendered RGB'], paths)
                print(f"Seg Overlay: enabled (rendered RGB + seg)")

    if args.trajectory_est:
        panels.append('Trajectory (Horn-aligned)')
        print("Trajectory: enabled (Horn-aligned)")
        if args.trajectory_raw:
            panels.append('Trajectory Raw')
            print("Trajectory Raw: enabled (no alignment)")

    if not panels:
        print("ERROR: No panels specified. Provide at least one input.")
        return

    # Determine master frame count. Use the LONGEST panel so shorter panels
    # (e.g. half-rate StereoMIS masks) are remapped via stride rather than
    # truncating the whole video down to the shortest panel's length.
    panel_lengths = {}
    for k, v in panel_data.items():
        if k == 'Seg Overlay':
            panel_lengths[k] = min(len(v[0]), len(v[1]))
        else:
            panel_lengths[k] = len(v)
    if args.trajectory_est:
        est_xyz, gt_xyz = load_trajectory(args.trajectory_est, args.trajectory_gt)
        if args.gt_frame_slice:
            gt_all = []
            with open(args.trajectory_gt) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    vals = list(map(float, line.strip().split()))
                    if len(vals) >= 4:
                        gt_all.append(vals[1:4])
            gt_all = np.array(gt_all)
            gt_xyz = eval(f"gt_all[{args.gt_frame_slice}]")[:len(est_xyz)]
        panel_lengths['Trajectory (Horn-aligned)'] = len(est_xyz)
        if args.trajectory_raw:
            panel_lengths['Trajectory Raw'] = len(est_xyz)

    n_frames = max(panel_lengths.values()) if panel_lengths else 0
    # Per-panel stride: panels shorter than master advance slower (floor-div mapping).
    panel_stride = {k: max(1, n_frames // max(1, L)) for k, L in panel_lengths.items()}
    for k, s in panel_stride.items():
        if s > 1:
            print(f"  panel '{k}': {panel_lengths[k]} frames -> stride {s} against master {n_frames}")
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)
    print(f"\nTotal frames: {n_frames}, Panels: {len(panels)}")

    # Grid layout
    n_panels = len(panels)
    if n_panels <= 2:
        rows, cols = 1, n_panels
    elif n_panels <= 4:
        rows, cols = 2, 2
    elif n_panels <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    canvas_h = rows * panel_size[0]
    canvas_w = cols * panel_size[1]
    print(f"Layout: {rows}x{cols}, Canvas: {canvas_w}x{canvas_h}")

    # Compute trajectory stride: if rendered panels are sparser than pose count,
    # map video frame_idx -> pose index so trajectory stays in sync with RGB panels.
    trajectory_stride = 1
    if args.trajectory_est and n_frames > 0:
        trajectory_stride = max(1, len(est_xyz) // n_frames)
        if trajectory_stride > 1:
            print(f"Trajectory stride: {trajectory_stride} "
                  f"(video has {n_frames} frames, est has {len(est_xyz)} poses)")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (canvas_w, canvas_h))

    for frame_idx in tqdm(range(n_frames), desc="Rendering video"):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for p_idx, panel_name in enumerate(panels):
            row = p_idx // cols
            col = p_idx % cols
            y0 = row * panel_size[0]
            x0 = col * panel_size[1]

            if panel_name == 'Trajectory (Horn-aligned)':
                pose_idx = min(frame_idx * trajectory_stride, len(est_xyz) - 1)
                azim = frame_idx * args.rotation_speed
                img = render_trajectory_frame(est_xyz, gt_xyz, pose_idx,
                                              panel_size, azim_offset=azim,
                                              align=True)
            elif panel_name == 'Trajectory Raw':
                pose_idx = min(frame_idx * trajectory_stride, len(est_xyz) - 1)
                azim = frame_idx * args.rotation_speed
                img = render_trajectory_frame(est_xyz, gt_xyz, pose_idx,
                                              panel_size, azim_offset=azim,
                                              align=False)
            elif panel_name == 'Seg Overlay':
                rgb_paths, seg_paths = panel_data[panel_name]
                ri = min(frame_idx, len(rgb_paths) - 1)
                si = min(frame_idx // panel_stride.get('Segmentation', 1), len(seg_paths) - 1)
                rgb = load_image(rgb_paths[ri], panel_size)
                seg = load_image(seg_paths[si], panel_size)
                if rgb is not None and seg is not None:
                    img = overlay_mask_on_rgb(rgb, seg, alpha=0.5)
                else:
                    img = rgb if rgb is not None else seg
            elif panel_name in ('Input Depth', 'Output Depth'):
                paths = panel_data[panel_name]
                idx = min(frame_idx // panel_stride[panel_name], len(paths) - 1)
                img = colormap_depth(paths[idx], panel_size, args.png_depth_scale)
            else:
                paths = panel_data[panel_name]
                idx = min(frame_idx // panel_stride[panel_name], len(paths) - 1)
                img = load_image(paths[idx], panel_size)

            if img is not None:
                img = add_label(img, panel_name)
                canvas[y0:y0 + panel_size[0], x0:x0 + panel_size[1]] = img

        # Write frame (convert RGB to BGR for OpenCV)
        writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"\nVideo saved: {args.output} ({n_frames} frames, {n_frames / args.fps:.1f}s)")


if __name__ == '__main__':
    main()
