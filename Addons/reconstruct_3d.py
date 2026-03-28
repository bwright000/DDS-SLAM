"""
3D Reconstruction from DDS-SLAM results.

Two reconstruction paths:
  Path A (gt):     Depth-based TSDF fusion using estimated poses + depth maps
  Path B (neural): Neural SDF mesh extraction from trained checkpoint

Usage:
  python Addons/reconstruct_3d.py \
    --config configs/Super/trail3.yaml \
    --checkpoint output/trail3_depth_anything/demo/checkpoint150.pt \
    --datadir data/Super \
    --depth_dir output/DDS-SLAM-Results/depth_maps_depth_anything \
    --posefile output/trail3_depth_anything/demo/est_c2w_data.txt \
    --output_dir output/trail3_depth_anything/demo \
    --mode both
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import trimesh

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


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


def get_depth_files(depth_dir):
    """Find depth .npy files in a directory."""
    files = sorted(glob.glob(os.path.join(depth_dir, '*_depth.npy')))
    if not files:
        files = sorted(glob.glob(os.path.join(depth_dir, '*.npy')))
    if not files:
        raise FileNotFoundError(f"No depth .npy files found in {depth_dir}")
    return files


# ============================================================================
# Path A: Depth-Based TSDF Fusion
# ============================================================================

def reconstruct_depth_open3d(rgb_files, depth_files, poses, intrinsics,
                              depth_scale, depth_trunc, tsdf_voxel,
                              output_path, skip=1):
    """TSDF fusion using Open3D — produces a watertight mesh."""
    import open3d as o3d

    fx, fy, cx, cy = intrinsics
    H, W = cv2.imread(rgb_files[0]).shape[:2]

    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_voxel,
        sdf_trunc=tsdf_voxel * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    n_frames = min(len(rgb_files), len(depth_files), len(poses))
    print(f"Integrating {n_frames // skip} frames into TSDF volume (voxel={tsdf_voxel})...")

    for i in range(0, n_frames, skip):
        color_img = o3d.io.read_image(rgb_files[i])

        depth_np = np.load(depth_files[i]).astype(np.float32)
        depth_meters = depth_np / depth_scale
        # Clamp to valid range
        depth_meters[depth_meters > depth_trunc] = 0.0
        depth_meters[depth_meters < 0] = 0.0
        # Open3D expects depth in mm-scale float or uint16; use meters with create_from_color_and_depth
        depth_o3d = o3d.geometry.Image(depth_meters)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_o3d,
            depth_scale=1.0,  # already in meters
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        # c2w pose from DDS-SLAM (OpenGL convention: Y-up, -Z forward)
        # Open3D expects w2c in OpenCV convention
        c2w = poses[i].copy()
        # OpenGL to OpenCV: flip Y and Z axes
        gl_to_cv = np.diag([1, -1, -1, 1]).astype(np.float64)
        c2w_cv = c2w @ gl_to_cv
        w2c = np.linalg.inv(c2w_cv)

        volume.integrate(rgbd, intrinsic, w2c)

    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(output_path, mesh)
    n_verts = np.asarray(mesh.vertices).shape[0]
    n_faces = np.asarray(mesh.triangles).shape[0]
    print(f"Depth-based mesh saved: {output_path} ({n_verts} vertices, {n_faces} faces)")
    return mesh


def reconstruct_depth_pointcloud(rgb_files, depth_files, poses, intrinsics,
                                  depth_scale, depth_trunc, output_path,
                                  skip=1, max_points_per_frame=50000):
    """Fallback: simple coloured point cloud without Open3D."""
    fx, fy, cx, cy = intrinsics

    all_points = []
    all_colors = []

    n_frames = min(len(rgb_files), len(depth_files), len(poses))
    print(f"Back-projecting {n_frames // skip} frames into point cloud...")

    for i in range(0, n_frames, skip):
        rgb = cv2.imread(rgb_files[i])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        depth_np = np.load(depth_files[i]).astype(np.float32)
        depth_meters = depth_np / depth_scale

        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        valid = (depth_meters > 0) & (depth_meters < depth_trunc)

        u_valid = u[valid].astype(np.float64)
        v_valid = v[valid].astype(np.float64)
        d_valid = depth_meters[valid].astype(np.float64)

        # Subsample if too many points
        if len(u_valid) > max_points_per_frame:
            idx = np.random.choice(len(u_valid), max_points_per_frame, replace=False)
            u_valid, v_valid, d_valid = u_valid[idx], v_valid[idx], d_valid[idx]

        # Back-project to camera coordinates
        x = (u_valid - cx) * d_valid / fx
        y = (v_valid - cy) * d_valid / fy
        z = d_valid

        pts_cam = np.stack([x, y, z, np.ones_like(x)], axis=-1)  # (N, 4)

        # Transform to world coordinates
        c2w = poses[i]
        pts_world = (c2w @ pts_cam.T).T[:, :3]

        # Get colours
        colors = rgb[valid]
        if len(u_valid) < len(colors):
            colors = colors[idx] if 'idx' in dir() else colors[:len(u_valid)]

        all_points.append(pts_world)
        all_colors.append(colors[:len(pts_world)] / 255.0)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)

    cloud = trimesh.PointCloud(points, colors=colors)
    cloud.export(output_path)
    print(f"Point cloud saved: {output_path} ({len(points)} points)")
    return cloud


def reconstruct_from_depth(rgb_files, depth_files, poses, intrinsics,
                           depth_scale, depth_trunc, output_path,
                           tsdf_voxel=0.004, skip=1):
    """Depth-based reconstruction with Open3D fallback."""
    try:
        import open3d
        return reconstruct_depth_open3d(
            rgb_files, depth_files, poses, intrinsics,
            depth_scale, depth_trunc, tsdf_voxel, output_path, skip
        )
    except ImportError:
        print("Open3D not available, falling back to point cloud...")
        pc_path = output_path.replace('.ply', '_pointcloud.ply')
        return reconstruct_depth_pointcloud(
            rgb_files, depth_files, poses, intrinsics,
            depth_scale, depth_trunc, pc_path, skip
        )


# ============================================================================
# Path B: Neural SDF Mesh Extraction
# ============================================================================

def reconstruct_from_neural(config_path, checkpoint_path, output_path,
                            voxel_size=None):
    """Extract mesh from trained DDS-SLAM neural SDF checkpoint."""
    import config as cfg_module
    from model.scene_rep import JointEncoding
    from utils import extract_mesh

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load config
    cfg = cfg_module.load_config(config_path)

    # Build model
    bounding_box = torch.from_numpy(
        np.array(cfg['mapping']['bound'])
    ).float().to(device)

    marching_cube_bound = torch.from_numpy(
        np.array(cfg['mapping']['marching_cubes_bound'])
    ).float().to(device)

    model = JointEncoding(cfg, bounding_box).to(device)

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Determine voxel size
    if voxel_size is None:
        voxel_size = cfg['mesh'].get('voxel_final', 0.02)

    print(f"Extracting neural SDF mesh (voxel_size={voxel_size})...")

    mesh = extract_mesh(
        query_fn=model.query_sdf,
        config=cfg,
        bounding_box=bounding_box,
        marching_cube_bound=marching_cube_bound,
        color_func=model.query_color_sdf if hasattr(model, 'query_color_sdf') else None,
        voxel_size=voxel_size,
        mesh_savepath=output_path
    )

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"Neural SDF mesh saved: {output_path} ({n_verts} vertices, {n_faces} faces)")
    return mesh


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='3D Reconstruction from DDS-SLAM results'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to DDS-SLAM config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file (required for neural mode)')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Path to dataset with rgb/ folder (required for gt mode)')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Path to depth .npy files (if different from datadir/rgb/)')
    parser.add_argument('--posefile', type=str, default=None,
                        help='Path to est_c2w_data.txt (required for gt mode)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output .ply files')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['gt', 'neural', 'both'],
                        help='Reconstruction mode')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Marching cubes voxel size for neural SDF (default from config)')
    parser.add_argument('--tsdf_voxel', type=float, default=0.004,
                        help='TSDF voxel size for depth fusion (default: 0.004)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Use every N-th frame for GT reconstruction')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config for camera params
    sys.path.insert(0, PROJECT_ROOT)
    import config as cfg_module
    cfg = cfg_module.load_config(args.config)

    depth_scale = cfg['cam']['png_depth_scale']
    depth_trunc = cfg['cam']['depth_trunc']
    fx = cfg['cam']['fx']
    fy = cfg['cam']['fy']
    cx = cfg['cam']['cx']
    cy = cfg['cam']['cy']
    intrinsics = (fx, fy, cx, cy)

    print(f"Camera: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    print(f"Depth scale={depth_scale}, trunc={depth_trunc}m")

    # --- Path A: Depth-based reconstruction ---
    if args.mode in ('gt', 'both'):
        if not args.datadir:
            raise ValueError("--datadir required for gt mode")
        if not args.posefile:
            raise ValueError("--posefile required for gt mode")

        rgb_files = get_left_images(args.datadir)
        depth_dir = args.depth_dir or os.path.join(args.datadir, 'rgb')
        depth_files = get_depth_files(depth_dir)
        poses = load_poses_from_txt(args.posefile)

        print(f"\n{'='*60}")
        print("Path A: Depth-based TSDF reconstruction")
        print(f"{'='*60}")
        print(f"RGB frames: {len(rgb_files)}")
        print(f"Depth maps: {len(depth_files)}")
        print(f"Poses: {len(poses)}")

        gt_path = os.path.join(args.output_dir, 'mesh_gt_depth.ply')
        reconstruct_from_depth(
            rgb_files, depth_files, poses, intrinsics,
            depth_scale, depth_trunc, gt_path,
            tsdf_voxel=args.tsdf_voxel, skip=args.skip
        )

    # --- Path B: Neural SDF mesh extraction ---
    if args.mode in ('neural', 'both'):
        if not args.checkpoint:
            raise ValueError("--checkpoint required for neural mode")

        print(f"\n{'='*60}")
        print("Path B: Neural SDF mesh extraction")
        print(f"{'='*60}")

        neural_path = os.path.join(args.output_dir, 'mesh_ddsslam.ply')
        reconstruct_from_neural(
            args.config, args.checkpoint, neural_path,
            voxel_size=args.voxel_size
        )

    print(f"\nDone. Output in {args.output_dir}")


if __name__ == '__main__':
    main()
