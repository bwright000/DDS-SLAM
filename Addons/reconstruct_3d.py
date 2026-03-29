"""
3D Reconstruction from DDS-SLAM results.

Two reconstruction paths:
  Path A (gt):     Depth-based windowed TSDF fusion per frame
  Path B (neural): Time-aware neural SDF mesh extraction per frame

Both paths produce per-frame meshes that can be visualized as a
timeline in Rerun to observe tissue deformation over time.

Usage:
  # All frames, both paths:
  python Addons/reconstruct_3d.py \
    --config configs/Super/trail3.yaml \
    --checkpoint output/.../checkpoint150.pt \
    --datadir data/Super \
    --depth_dir output/.../depth_maps_depth_anything \
    --posefile output/.../est_c2w_data.txt \
    --output_dir output/.../meshes \
    --mode both

  # Single frame, neural only:
  python Addons/reconstruct_3d.py \
    --config configs/Super/trail3.yaml \
    --checkpoint output/.../checkpoint150.pt \
    --output_dir output/.../meshes \
    --mode neural --frames 0,75,150
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import trimesh
from tqdm import tqdm

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


def parse_frame_list(frames_str, n_frames):
    """Parse --frames argument into a list of frame indices."""
    if frames_str is None:
        return list(range(n_frames))
    indices = []
    for part in frames_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return sorted(set(i for i in indices if 0 <= i < n_frames))


def clean_mesh(mesh, min_component_faces=100, smooth_iterations=3,
               smooth_lambda=0.5):
    """Clean a mesh by removing small components and optionally smoothing."""
    if not isinstance(mesh, trimesh.Trimesh):
        try:
            import open3d as o3d
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
                mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                       vertex_colors=colors, process=False)
        except ImportError:
            pass

    if len(mesh.faces) == 0:
        return mesh

    components = trimesh.graph.connected_components(mesh.face_adjacency)
    if len(components) > 1:
        keep_faces = set()
        for component in components:
            if len(component) >= min_component_faces:
                keep_faces.update(component)
        if keep_faces:
            face_mask = np.zeros(len(mesh.faces), dtype=bool)
            face_mask[list(keep_faces)] = True
            mesh.update_faces(face_mask)
            mesh.remove_unreferenced_vertices()

    if smooth_iterations > 0:
        try:
            trimesh.smoothing.filter_laplacian(
                mesh, lamb=smooth_lambda, iterations=smooth_iterations)
        except Exception:
            pass

    return mesh


# ============================================================================
# Path A: Depth-Based Windowed TSDF Fusion
# ============================================================================

def reconstruct_depth_at_frame(rgb_files, depth_files, poses, intrinsics,
                               center_frame, window_size,
                               depth_scale, depth_trunc, tsdf_voxel,
                               sdf_trunc_factor=5.0):
    """TSDF fusion over a window of frames centered on center_frame."""
    import open3d as o3d

    fx, fy, cx, cy = intrinsics
    H, W = cv2.imread(rgb_files[0]).shape[:2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)
    n_frames = min(len(rgb_files), len(depth_files), len(poses))

    half = window_size // 2
    start = max(0, center_frame - half)
    end = min(n_frames, center_frame + half + 1)

    sdf_trunc = tsdf_voxel * sdf_trunc_factor
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=tsdf_voxel,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for i in range(start, end):
        color_img = o3d.io.read_image(rgb_files[i])

        depth_np = np.load(depth_files[i]).astype(np.float32)
        depth_meters = depth_np / depth_scale
        depth_meters[depth_meters > depth_trunc] = 0.0
        depth_meters[depth_meters < 0] = 0.0
        # Filter noisy depth edges
        grad_x = cv2.Sobel(depth_meters, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth_meters, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        depth_meters[grad_mag > 0.5] = 0.0
        depth_o3d = o3d.geometry.Image(depth_meters)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img, depth_o3d,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        c2w = poses[i].copy()
        gl_to_cv = np.diag([1, -1, -1, 1]).astype(np.float64)
        w2c = np.linalg.inv(c2w @ gl_to_cv)

        volume.integrate(rgbd, intrinsic, w2c)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


# ============================================================================
# Path B: Time-Aware Neural SDF Mesh Extraction
# ============================================================================

def load_model(config_path, checkpoint_path):
    """Load DDS-SLAM model from config + checkpoint."""
    import config as cfg_module
    from model.scene_rep import JointEncoding

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = cfg_module.load_config(config_path)

    bounding_box = torch.from_numpy(
        np.array(cfg['mapping']['bound'])
    ).float().to(device)

    marching_cube_bound = torch.from_numpy(
        np.array(cfg['mapping']['marching_cubes_bound'])
    ).float().to(device)

    model = JointEncoding(cfg, bounding_box).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, cfg, bounding_box, marching_cube_bound, device


def reconstruct_neural_at_frame(model, cfg, bounding_box, marching_cube_bound,
                                device, timestamp, voxel_size,
                                isolevel=0.0, mc_truncation=1.0):
    """Extract mesh from neural SDF at a specific timestamp."""
    from utils import extract_mesh
    from functools import partial

    # Create time-aware query functions that bind the timestamp
    sdf_fn = partial(model.query_sdf_at_time, timestamp=timestamp)
    color_fn = partial(model.query_color_at_time, timestamp=timestamp)

    mesh = extract_mesh(
        query_fn=sdf_fn,
        config=cfg,
        bounding_box=bounding_box,
        marching_cube_bound=marching_cube_bound,
        color_func=color_fn,
        voxel_size=voxel_size,
        isolevel=isolevel,
        truncation=mc_truncation,
        mesh_savepath='',  # We'll save manually after cleaning
        skip_normalize=True  # query_sdf_at_time handles normalization
    )

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
                        help='Path to checkpoint .pt (required for neural mode)')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Path to dataset with rgb/ folder (required for gt mode)')
    parser.add_argument('--depth_dir', type=str, default=None,
                        help='Path to depth .npy files')
    parser.add_argument('--posefile', type=str, default=None,
                        help='Path to est_c2w_data.txt (required for gt mode)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save output meshes')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['gt', 'neural', 'both'],
                        help='Reconstruction mode')
    parser.add_argument('--frames', type=str, default=None,
                        help='Frame indices to reconstruct (e.g., "0,75,150" or "0-150"). Default: all')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Marching cubes voxel size for neural SDF (default from config)')
    parser.add_argument('--tsdf_voxel', type=float, default=0.004,
                        help='TSDF voxel size (default: 0.004)')
    parser.add_argument('--tsdf_window', type=int, default=10,
                        help='Number of frames in TSDF window (default: 10)')
    parser.add_argument('--isolevel', type=float, default=0.0,
                        help='Isolevel for marching cubes (default: 0.0)')
    parser.add_argument('--mc_truncation', type=float, default=1.0,
                        help='Marching cubes truncation (default: 1.0)')
    parser.add_argument('--no_clean', action='store_true',
                        help='Skip mesh post-processing')
    parser.add_argument('--min_component_faces', type=int, default=100,
                        help='Remove components with fewer faces (default: 100)')
    parser.add_argument('--smooth_iterations', type=int, default=3,
                        help='Laplacian smoothing iterations (default: 3)')
    parser.add_argument('--sdf_trunc_factor', type=float, default=5.0,
                        help='TSDF sdf_trunc = tsdf_voxel * factor (default: 5.0)')

    args = parser.parse_args()

    sys.path.insert(0, PROJECT_ROOT)
    import config as cfg_module
    cfg = cfg_module.load_config(args.config)

    depth_scale = cfg['cam']['png_depth_scale']
    depth_trunc = cfg['cam']['depth_trunc']
    intrinsics = (cfg['cam']['fx'], cfg['cam']['fy'],
                  cfg['cam']['cx'], cfg['cam']['cy'])
    n_total_frames = cfg.get('timesteps', 151)

    frame_list = parse_frame_list(args.frames, n_total_frames)
    print(f"Reconstructing {len(frame_list)} frames: {frame_list[:5]}{'...' if len(frame_list) > 5 else ''}")

    # --- Path A: Windowed depth-based TSDF per frame ---
    if args.mode in ('gt', 'both'):
        if not args.datadir:
            raise ValueError("--datadir required for gt mode")
        if not args.posefile:
            raise ValueError("--posefile required for gt mode")

        import open3d  # fail fast if missing

        rgb_files = get_left_images(args.datadir)
        depth_dir = args.depth_dir or os.path.join(args.datadir, 'rgb')
        depth_files = get_depth_files(depth_dir)
        poses = load_poses_from_txt(args.posefile)

        gt_dir = os.path.join(args.output_dir, 'gt')
        os.makedirs(gt_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Path A: Windowed TSDF (window={args.tsdf_window} frames)")
        print(f"{'='*60}")

        for frame_id in tqdm(frame_list, desc="TSDF frames"):
            mesh = reconstruct_depth_at_frame(
                rgb_files, depth_files, poses, intrinsics,
                center_frame=frame_id, window_size=args.tsdf_window,
                depth_scale=depth_scale, depth_trunc=depth_trunc,
                tsdf_voxel=args.tsdf_voxel,
                sdf_trunc_factor=args.sdf_trunc_factor
            )
            if not args.no_clean:
                mesh = clean_mesh(mesh,
                                  min_component_faces=args.min_component_faces,
                                  smooth_iterations=args.smooth_iterations)
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(os.path.join(gt_dir, f'frame_{frame_id:04d}.ply'))
            else:
                # Open3D mesh — convert and save
                verts = np.asarray(mesh.vertices)
                faces = np.asarray(mesh.triangles)
                colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
                tri_mesh = trimesh.Trimesh(vertices=verts, faces=faces,
                                           vertex_colors=colors, process=False)
                tri_mesh.export(os.path.join(gt_dir, f'frame_{frame_id:04d}.ply'))

        print(f"Saved {len(frame_list)} TSDF meshes to {gt_dir}/")

    # --- Path B: Time-aware neural SDF per frame ---
    if args.mode in ('neural', 'both'):
        if not args.checkpoint:
            raise ValueError("--checkpoint required for neural mode")

        model, model_cfg, bbox, mc_bound, device = load_model(
            args.config, args.checkpoint
        )

        voxel_size = args.voxel_size
        if voxel_size is None:
            voxel_size = model_cfg['mesh'].get('voxel_final', 0.005)

        neural_dir = os.path.join(args.output_dir, 'neural')
        os.makedirs(neural_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Path B: Time-aware neural SDF (voxel_size={voxel_size})")
        print(f"{'='*60}")

        for frame_id in tqdm(frame_list, desc="Neural frames"):
            mesh = reconstruct_neural_at_frame(
                model, model_cfg, bbox, mc_bound, device,
                timestamp=frame_id, voxel_size=voxel_size,
                isolevel=args.isolevel, mc_truncation=args.mc_truncation
            )
            if not args.no_clean:
                mesh = clean_mesh(mesh,
                                  min_component_faces=args.min_component_faces,
                                  smooth_iterations=args.smooth_iterations)
            mesh.export(os.path.join(neural_dir, f'frame_{frame_id:04d}.ply'))

        print(f"Saved {len(frame_list)} neural meshes to {neural_dir}/")

    print(f"\nDone. Output in {args.output_dir}")


if __name__ == '__main__':
    main()
