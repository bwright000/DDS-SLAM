"""
Visualize DDS-SLAM run outputs using Rerun.io.

Post-hoc visualization: loads saved poses, images, depth maps, metrics,
and meshes after a SLAM run completes and displays them in Rerun's viewer.

Shows:
  - 3D camera trajectory (estimated, and GT if provided)
  - Animated camera frustum scrubbing through the sequence
  - GT RGB, rendered RGB, and colormapped depth side-by-side
  - Per-frame metric plots (PSNR, SSIM) from pre-computed CSV
  - Reconstructed mesh in 3D (if provided)
  - Method comparison via --method_name entity prefixing

Usage:
  # Generate metrics CSV first
  python Addons/eval_rendering.py \\
    --gt_dir data/v2_data/trial_3/rgb \\
    --render_dir output/DDS-SLAM-Results/trail3_depth_anything \\
    --output_csv metrics.csv

  # Visualize with metrics
  python Addons/visualize_run.py \\
    --datadir data/v2_data/trial_3 \\
    --posefile output/.../est_c2w_data.txt \\
    --render_dir output/.../trail3_depth_anything \\
    --metrics_csv metrics.csv \\
    --name "trail3 Depth Anything"

  # Method comparison: save two runs into one .rrd
  python Addons/visualize_run.py ... --method_name "Depth Anything" --save compare.rrd
  python Addons/visualize_run.py ... --method_name "Monodepth2" --save compare.rrd --append

Requires: pip install rerun-sdk
"""

import argparse
import csv
import glob
import os
import re

import cv2
import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError:
    raise ImportError(
        "Rerun SDK not found. Install with: pip install rerun-sdk"
    )


# OpenGL camera to OpenCV camera conversion matrix.
GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


# =============================================================================
# Data loading
# =============================================================================

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
    if not poses:
        raise ValueError(f"No poses loaded from {filepath}")
    print(f"Loaded {len(poses)} poses from {filepath}")
    return poses


def get_left_images(datadir):
    """Find all left RGB images sorted by name."""
    files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*-left.png')))
    if not files:
        files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*_left.png')))
    if not files:
        raise FileNotFoundError(f"No left images found in {datadir}/rgb/")
    return files


def get_depth_path(rgb_path):
    """Derive depth .npy path from RGB image path."""
    return rgb_path.replace('-left.png', '-left_depth.npy').replace(
        '_left.png', '_left_depth.npy')


def get_rendered_images(render_dir):
    """Find rendered .jpg images and return dict mapping frame index to path."""
    if not render_dir or not os.path.isdir(render_dir):
        return {}
    files = sorted(glob.glob(os.path.join(render_dir, '*.jpg')))
    rendered = {}
    for f in files:
        basename = os.path.splitext(os.path.basename(f))[0]
        match = re.match(r'^(\d+)$', basename)
        if match:
            rendered[int(match.group(1))] = f
    return rendered


def load_depth(depth_path, depth_scale):
    """Load depth map from .npy and convert to meters."""
    raw = np.load(depth_path).astype(np.float32)
    if raw.ndim > 2:
        raw = raw.reshape(raw.shape[-2:])
    return raw / depth_scale


def load_rgb(rgb_path):
    """Load RGB image as uint8 HWC array."""
    img = cv2.imread(rgb_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_metrics_csv(filepath):
    """Load per-frame metrics from CSV. Returns list of dicts."""
    metrics = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(row)
    print(f"Loaded {len(metrics)} metric rows from {filepath}")
    return metrics


def colormap_depth(depth, max_depth):
    """Apply turbo colormap to depth array with auto-scaling."""
    clamped = np.clip(depth, 0.0, max_depth)
    valid = clamped[clamped > 0]
    if valid.size > 0:
        vmin = float(np.percentile(valid, 2))
        vmax = float(np.percentile(valid, 98))
    else:
        vmin, vmax = 0.0, max_depth
    if vmax - vmin < 1e-6:
        vmax = vmin + 1e-6
    normalized = np.clip((clamped - vmin) / (vmax - vmin), 0.0, 1.0)
    colored = cv2.applyColorMap(
        (normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    colored[depth <= 0] = 0
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


# =============================================================================
# Rerun logging
# =============================================================================

def entity(prefix, path):
    """Build entity path with optional method prefix."""
    if prefix:
        return f"{prefix}/{path}"
    return path


def log_static_data(est_poses, gt_poses=None, prefix=""):
    """Log time-independent data: coordinate frame, trajectory lines."""
    rr.log(entity(prefix, "world"), rr.ViewCoordinates.RUB, static=True)

    est_positions = np.array([p[:3, 3] for p in est_poses])
    extent = np.ptp(est_positions, axis=0).max()
    line_radius = max(extent * 0.003, 1e-5)
    point_radius = max(extent * 0.006, 2e-5)

    rr.log(
        entity(prefix, "world/trajectory/estimated"),
        rr.LineStrips3D(
            [est_positions.tolist()],
            colors=[[30, 120, 255]],
            radii=[line_radius],
        ),
        static=True,
    )
    rr.log(
        entity(prefix, "world/est_positions"),
        rr.Points3D(
            est_positions,
            colors=np.full((len(est_positions), 3), [30, 120, 255], dtype=np.uint8),
            radii=point_radius,
        ),
        static=True,
    )

    if gt_poses is not None:
        gt_positions = np.array([p[:3, 3] for p in gt_poses])
        rr.log(
            entity(prefix, "world/trajectory/gt"),
            rr.LineStrips3D(
                [gt_positions.tolist()],
                colors=[[0, 200, 0]],
                radii=[line_radius],
            ),
            static=True,
        )
        rr.log(
            entity(prefix, "world/gt_positions"),
            rr.Points3D(
                gt_positions,
                colors=np.full((len(gt_positions), 3), [0, 200, 0], dtype=np.uint8),
                radii=point_radius,
            ),
            static=True,
        )


def log_mesh(mesh_path, prefix=""):
    """Load and log a .ply/.obj mesh into the 3D scene."""
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed, skipping mesh (pip install trimesh)")
        return

    print(f"Loading mesh: {mesh_path}")
    mesh = trimesh.load(mesh_path)

    verts = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces, dtype=np.uint32)

    kwargs = dict(vertex_positions=verts, indices=faces)
    if mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
        colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
        kwargs['vertex_colors'] = colors

    rr.log(entity(prefix, "world/mesh"), rr.Mesh3D(**kwargs), static=True)
    print(f"Logged mesh: {len(verts)} vertices, {len(faces)} faces")


def log_mesh_sequence(mesh_dir, prefix=""):
    """Load per-frame meshes from a directory and log them on Rerun's timeline.

    Expects files named frame_XXXX.ply (e.g., frame_0000.ply, frame_0075.ply).
    Each mesh is logged at its corresponding frame index so the Rerun timeline
    scrubber shows the deforming surface over time.
    """
    try:
        import trimesh
    except ImportError:
        print("Warning: trimesh not installed, skipping mesh sequence")
        return

    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, 'frame_*.ply')))
    if not mesh_files:
        print(f"No frame_*.ply files found in {mesh_dir}")
        return

    print(f"Loading {len(mesh_files)} meshes from {mesh_dir}...")
    entity_path = entity(prefix, "world/mesh")

    for mesh_path in mesh_files:
        # Extract frame index from filename (frame_0075.ply -> 75)
        basename = os.path.splitext(os.path.basename(mesh_path))[0]
        frame_idx = int(basename.split('_')[-1])

        mesh = trimesh.load(mesh_path)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.uint32)

        kwargs = dict(vertex_positions=verts, indices=faces)
        if mesh.visual and hasattr(mesh.visual, 'vertex_colors'):
            colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.uint8)
            kwargs['vertex_colors'] = colors

        rr.set_time_sequence("frame", frame_idx)
        rr.log(entity_path, rr.Mesh3D(**kwargs))

    print(f"Logged {len(mesh_files)} meshes on timeline")


def backproject_depth_to_pointcloud(depth, rgb, c2w, fx, fy, cx, cy,
                                     depth_trunc, max_points=50000):
    """Back-project depth map to 3D coloured point cloud in world frame.

    Args:
        depth: (H, W) depth in meters
        rgb: (H, W, 3) uint8 RGB image
        c2w: (4, 4) camera-to-world matrix (OpenGL convention)
        fx, fy, cx, cy: camera intrinsics
        depth_trunc: max valid depth
        max_points: subsample if more valid points than this

    Returns:
        points: (N, 3) world coordinates
        colors: (N, 3) uint8 RGB
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    valid = (depth > 0) & (depth < depth_trunc)
    u_v = u[valid].astype(np.float64)
    v_v = v[valid].astype(np.float64)
    d_v = depth[valid].astype(np.float64)

    if len(u_v) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)

    # Subsample
    if len(u_v) > max_points:
        idx = np.random.choice(len(u_v), max_points, replace=False)
        u_v, v_v, d_v = u_v[idx], v_v[idx], d_v[idx]
    else:
        idx = None

    # OpenGL camera rays (matching DDS-SLAM convention)
    ray_x = (u_v - cx) / fx
    ray_y = -(v_v - cy) / fy
    ray_z = -np.ones_like(u_v)

    pts_cam = np.stack([d_v * ray_x, d_v * ray_y, d_v * ray_z], axis=-1)
    ones = np.ones((len(pts_cam), 1), dtype=np.float64)
    pts_homo = np.concatenate([pts_cam, ones], axis=1)

    pts_world = (c2w @ pts_homo.T).T[:, :3]

    # Get colors
    if idx is not None:
        valid_indices = np.where(valid)
        colors = rgb[valid_indices[0][idx], valid_indices[1][idx]]
    else:
        colors = rgb[valid]

    return pts_world.astype(np.float32), colors.astype(np.uint8)


def log_frame(frame_idx, est_pose, gt_rgb_path, rendered_path,
              depth_path, metrics_row, fx, fy, cx, cy, W, H,
              depth_scale, depth_trunc, prefix="",
              log_pointcloud=True, pc_max_points=50000):
    """Log all per-frame data for a single timestep."""
    rr.set_time("frame", sequence=frame_idx)

    # --- Camera pose + frustum ---
    c2w_cv = est_pose @ GL_TO_CV
    R = c2w_cv[:3, :3]
    t = c2w_cv[:3, 3]

    rr.log(entity(prefix, "world/est_camera"), rr.Transform3D(
        translation=t, mat3x3=R,
    ))
    rr.log(entity(prefix, "world/est_camera/pinhole"), rr.Pinhole(
        focal_length=[fx, fy],
        principal_point=[cx, cy],
        width=W, height=H,
        camera_xyz=rr.ViewCoordinates.RDF,
        image_plane_distance=0.01,
    ))

    # Load depth (used for colormap and point cloud)
    depth = None
    if depth_path and os.path.exists(depth_path):
        depth = load_depth(depth_path, depth_scale)
        depth_colored = colormap_depth(depth, depth_trunc)
        rr.log(entity(prefix, "images/depth_colormap"), rr.Image(depth_colored))

    # --- GT RGB ---
    gt_rgb = None
    if gt_rgb_path and os.path.exists(gt_rgb_path):
        gt_rgb = load_rgb(gt_rgb_path)
        rr.log(entity(prefix, "images/gt_rgb"), rr.Image(gt_rgb))

    # --- Per-frame 3D point cloud ---
    if log_pointcloud and depth is not None and gt_rgb is not None:
        points, colors = backproject_depth_to_pointcloud(
            depth, gt_rgb, est_pose, fx, fy, cx, cy,
            depth_trunc, max_points=pc_max_points
        )
        if len(points) > 0:
            rr.log(entity(prefix, "world/pointcloud"), rr.Points3D(
                points, colors=colors, radii=0.001
            ))

    # --- Rendered RGB ---
    if rendered_path and os.path.exists(rendered_path):
        rendered_rgb = load_rgb(rendered_path)
        rr.log(entity(prefix, "images/rendered_rgb"), rr.Image(rendered_rgb))

    # --- Metrics ---
    if metrics_row:
        if 'psnr' in metrics_row:
            rr.log(entity(prefix, "metrics/psnr"),
                   rr.Scalars(float(metrics_row['psnr'])))
        if 'ssim' in metrics_row:
            rr.log(entity(prefix, "metrics/ssim"),
                   rr.Scalars(float(metrics_row['ssim'])))
        if 'lpips' in metrics_row:
            rr.log(entity(prefix, "metrics/lpips"),
                   rr.Scalars(float(metrics_row['lpips'])))


def build_blueprint(prefix=""):
    """Build a default Rerun blueprint layout."""
    p = prefix + "/" if prefix else ""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin=f"{p}world"),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin=f"{p}images/gt_rgb"),
                    rrb.Spatial2DView(origin=f"{p}images/rendered_rgb"),
                    rrb.Spatial2DView(origin=f"{p}images/depth_colormap"),
                ),
                rrb.TimeSeriesView(origin=f"{p}metrics"),
            ),
            column_shares=[1, 2],
        ),
        auto_views=False,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize DDS-SLAM run outputs using Rerun.io')
    # Data paths
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to dataset (e.g., data/v2_data/trial_3)')
    parser.add_argument('--posefile', type=str, required=True,
                        help='Path to est_c2w_data.txt')
    parser.add_argument('--render_dir', type=str, default='',
                        help='Directory with rendered {frame:04d}.jpg images')
    parser.add_argument('--depth_dir', type=str, default='',
                        help='Directory with depth .npy files (default: {datadir}/rgb/)')
    parser.add_argument('--gt_posefile', type=str, default='',
                        help='Optional GT pose file (same 12-float format)')
    parser.add_argument('--metrics_csv', type=str, default='',
                        help='Per-frame metrics CSV (from eval_rendering.py --output_csv)')
    parser.add_argument('--mesh', type=str, default='',
                        help='Path to static reconstructed mesh (.ply or .obj)')
    parser.add_argument('--mesh_dir', type=str, default='',
                        help='Directory of per-frame meshes (frame_XXXX.ply) for timeline scrubbing')
    # Camera intrinsics
    parser.add_argument('--fx', type=float, default=768.98551924)
    parser.add_argument('--fy', type=float, default=768.98551924)
    parser.add_argument('--cx', type=float, default=292.8861567)
    parser.add_argument('--cy', type=float, default=291.61479526)
    parser.add_argument('--H', type=int, default=480)
    parser.add_argument('--W', type=int, default=640)
    # Depth
    parser.add_argument('--depth_scale', type=float, default=8.0,
                        help='png_depth_scale from config (default: 8.0)')
    parser.add_argument('--depth_trunc', type=float, default=5.0,
                        help='Max depth for colormap normalization (default: 5.0)')
    # Rerun options
    parser.add_argument('--save', type=str, default='',
                        help='Save to .rrd file instead of spawning viewer')
    parser.add_argument('--append', action='store_true',
                        help='Append to existing .rrd file (for method comparison)')
    parser.add_argument('--name', type=str, default='DDS-SLAM',
                        help='Application name for Rerun viewer')
    parser.add_argument('--method_name', type=str, default='',
                        help='Method name prefix for entity paths (for comparison mode)')
    parser.add_argument('--every', type=int, default=1,
                        help='Log every N-th frame (default: 1)')
    parser.add_argument('--no_pointcloud', action='store_true',
                        help='Disable per-frame 3D point cloud logging')
    parser.add_argument('--pc_max_points', type=int, default=50000,
                        help='Max points per frame for point cloud (default: 50000)')
    args = parser.parse_args()

    prefix = args.method_name

    # --- Init Rerun ---
    blueprint = build_blueprint(prefix)
    rr.init(args.name, default_blueprint=blueprint)
    if args.save:
        rr.save(args.save, default_blueprint=blueprint)
    else:
        rr.spawn(default_blueprint=blueprint)

    # --- Load data ---
    est_poses = load_poses_from_txt(args.posefile)
    gt_poses = None
    if args.gt_posefile and os.path.exists(args.gt_posefile):
        gt_poses = load_poses_from_txt(args.gt_posefile)

    rgb_paths = get_left_images(args.datadir)
    rendered = get_rendered_images(args.render_dir)

    # Depth paths
    if args.depth_dir and os.path.isdir(args.depth_dir):
        depth_paths = sorted(glob.glob(os.path.join(args.depth_dir, '*_depth.npy')))
        print(f"Using depth dir: {args.depth_dir} ({len(depth_paths)} files)")
    else:
        depth_paths = [get_depth_path(p) for p in rgb_paths]

    # Metrics
    metrics = []
    if args.metrics_csv and os.path.exists(args.metrics_csv):
        metrics = load_metrics_csv(args.metrics_csv)

    n_frames = min(len(est_poses), len(rgb_paths))
    print(f"Frames: {n_frames} (poses: {len(est_poses)}, images: {len(rgb_paths)})")
    print(f"Rendered images found: {len(rendered)}")
    if gt_poses:
        print(f"GT poses: {len(gt_poses)}")
    if prefix:
        print(f"Method prefix: {prefix}")

    # --- Log static data ---
    log_static_data(est_poses[:n_frames],
                    gt_poses[:n_frames] if gt_poses else None,
                    prefix=prefix)

    # --- Log mesh ---
    if args.mesh and os.path.exists(args.mesh):
        log_mesh(args.mesh, prefix=prefix)
    if args.mesh_dir and os.path.isdir(args.mesh_dir):
        log_mesh_sequence(args.mesh_dir, prefix=prefix)

    # --- Log per-frame data ---
    for i in range(0, n_frames, args.every):
        metrics_row = metrics[i] if i < len(metrics) else None
        log_frame(
            frame_idx=i,
            est_pose=est_poses[i],
            gt_rgb_path=rgb_paths[i] if i < len(rgb_paths) else None,
            rendered_path=rendered.get(i),
            depth_path=depth_paths[i] if i < len(depth_paths) else None,
            metrics_row=metrics_row,
            fx=args.fx, fy=args.fy,
            cx=args.cx, cy=args.cy,
            W=args.W, H=args.H,
            depth_scale=args.depth_scale,
            depth_trunc=args.depth_trunc,
            prefix=prefix,
            log_pointcloud=not args.no_pointcloud,
            pc_max_points=args.pc_max_points,
        )

    print(f"Logged {len(range(0, n_frames, args.every))} frames to Rerun.")
    if args.save:
        print(f"Saved to {args.save}")
    else:
        print("Viewer spawned. Close the Rerun window to exit.")


if __name__ == '__main__':
    main()
