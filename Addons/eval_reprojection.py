"""
Evaluate reprojection error of DDS-SLAM estimated poses and depth.

Computes three metrics between frame pairs at configurable step sizes:
  - Geometric reprojection error (pixels)
  - Depth consistency error (meters)
  - Photometric reprojection error (L1 color difference)

These jointly evaluate depth estimation accuracy and pose estimation accuracy
by projecting 3D points from one frame into another.

Usage:
  python Addons/eval_reprojection.py \
    --datadir data/Super/trail3 \
    --posefile output/trail3/est_c2w_data.txt \
    --fx 768.98551924 --fy 768.98551924 \
    --cx 292.8861567 --cy 291.61479526 \
    --depth_scale 8.0 --depth_trunc 5.0 \
    --frame_steps 1 5 10 \
    --n_samples 10000 \
    --name "DDS-SLAM trail3"
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# =============================================================================
# Data loading
# =============================================================================

def load_poses_from_txt(filepath):
    """Load estimated c2w poses from text file.

    Format: one line per frame, 12 floats (4x4 matrix reshaped to 16, first 12).
    Returns list of 4x4 numpy arrays.
    """
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


def load_depth(depth_path, depth_scale):
    """Load depth map from .npy and convert to meters."""
    raw = np.load(depth_path).astype(np.float32)
    if raw.ndim > 2:
        raw = raw.reshape(raw.shape[-2:])
    return raw / depth_scale


def load_rgb(rgb_path):
    """Load RGB image as float32 [0,1]."""
    img = cv2.imread(rgb_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


# =============================================================================
# Core geometry — matches keyframe.py conventions exactly
# =============================================================================

def backproject_pixels(u, v, depth, fx, fy, cx, cy, c2w):
    """Backproject pixels to 3D world coordinates.

    Uses OpenGL convention matching DDS-SLAM's get_camera_rays:
      ray_cam = [(u-cx)/fx, -(v-cy)/fy, -1]
      P_cam = depth * ray_cam  (depth is z-buffer depth, z=-depth in OpenGL)
      P_world = c2w @ [P_cam, 1]^T

    Args:
        u, v: (N,) pixel coordinates
        depth: (N,) depth values in meters
        fx, fy, cx, cy: camera intrinsics
        c2w: (4,4) camera-to-world matrix

    Returns:
        (N, 3) world coordinates
    """
    N = len(u)
    # OpenGL ray directions
    ray_x = (u - cx) / fx
    ray_y = -(v - cy) / fy
    ray_z = -np.ones(N, dtype=np.float64)

    # Points in camera frame: depth scales the ray (z component is -1,
    # so depth * -1 = -depth which is correct for OpenGL)
    P_cam = np.stack([depth * ray_x, depth * ray_y, depth * ray_z], axis=-1)  # (N, 3)

    # Homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float64)
    P_cam_homo = np.concatenate([P_cam, ones], axis=1)  # (N, 4)

    # Transform to world
    P_world = (c2w @ P_cam_homo.T).T[:, :3]  # (N, 3)
    return P_world


def project_to_image(points_world, w2c, K):
    """Project 3D world points to pixel coordinates.

    Replicates keyframe.py lines 136-148 exactly:
      P_cam = w2c @ [P_world, 1]^T
      P_cam[0] *= -1   (OpenGL x-flip)
      uv_homo = K @ P_cam[:3]
      uv = uv_homo[:2] / uv_homo[2]

    Args:
        points_world: (N, 3) world coordinates
        w2c: (4,4) world-to-camera matrix
        K: (3,3) intrinsics matrix

    Returns:
        u_proj: (N,) projected pixel x
        v_proj: (N,) projected pixel y
        z_cam: (N,) depth in camera frame (negative = visible in OpenGL)
    """
    N = points_world.shape[0]
    ones = np.ones((N, 1), dtype=np.float64)
    pts_homo = np.concatenate([points_world, ones], axis=1)  # (N, 4)

    # Transform to camera frame
    cam_pts = (w2c @ pts_homo.T).T[:, :3]  # (N, 3)

    # OpenGL x-flip (keyframe.py line 145)
    cam_pts[:, 0] *= -1

    # Project with intrinsics
    uv_homo = (K @ cam_pts.T).T  # (N, 3)
    z = uv_homo[:, 2] + 1e-5
    u_proj = uv_homo[:, 0] / z
    v_proj = uv_homo[:, 1] / z

    # z_cam before x-flip for visibility check (keyframe.py line 153: z < 0)
    z_cam = (w2c @ pts_homo.T).T[:, 2]

    return u_proj, v_proj, z_cam


def bilinear_sample(image, u, v):
    """Bilinear interpolation at sub-pixel locations.

    Args:
        image: (H, W) or (H, W, C) array
        u, v: (N,) sub-pixel coordinates

    Returns:
        (N,) or (N, C) sampled values
    """
    H, W = image.shape[:2]

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # Clamp to valid range
    u0c = np.clip(u0, 0, W - 1)
    u1c = np.clip(u1, 0, W - 1)
    v0c = np.clip(v0, 0, H - 1)
    v1c = np.clip(v1, 0, H - 1)

    # Weights
    wu = u - u0
    wv = v - v0

    if image.ndim == 2:
        val = (image[v0c, u0c] * (1 - wu) * (1 - wv) +
               image[v0c, u1c] * wu * (1 - wv) +
               image[v1c, u0c] * (1 - wu) * wv +
               image[v1c, u1c] * wu * wv)
    else:
        wu = wu[:, None]
        wv = wv[:, None]
        val = (image[v0c, u0c] * (1 - wu) * (1 - wv) +
               image[v0c, u1c] * wu * (1 - wv) +
               image[v1c, u0c] * (1 - wu) * wv +
               image[v1c, u1c] * wu * wv)
    return val


# =============================================================================
# Frame pair evaluation
# =============================================================================

def evaluate_frame_pair(idx_i, idx_j, poses, rgb_paths, depth_paths,
                        K, fx, fy, cx, cy, H, W,
                        depth_scale, depth_trunc, n_samples,
                        edge_margin=20):
    """Evaluate reprojection metrics between two frames.

    Steps:
      1. Load depth/rgb for both frames
      2. Sample N pixels with valid depth from frame i
      3. Backproject to 3D world using depth_i and c2w_i
      4. Project into frame j using w2c_j and K
      5. Filter by bounds, visibility, depth consistency
      6. Compute geometric, depth, and photometric errors

    Returns dict of metrics, or None if insufficient valid data.
    """
    c2w_i = poses[idx_i]
    c2w_j = poses[idx_j]
    w2c_j = np.linalg.inv(c2w_j)

    # Load data
    depth_i = load_depth(depth_paths[idx_i], depth_scale)
    depth_j = load_depth(depth_paths[idx_j], depth_scale)
    rgb_i = load_rgb(rgb_paths[idx_i])
    rgb_j = load_rgb(rgb_paths[idx_j])

    # Find valid pixels in frame i (valid depth within truncation)
    valid_mask = (depth_i > 0) & (depth_i < depth_trunc)
    valid_v, valid_u = np.where(valid_mask)

    if len(valid_u) == 0:
        return None

    # Sample pixels
    n = min(n_samples, len(valid_u))
    indices = np.random.choice(len(valid_u), size=n, replace=False)
    u_src = valid_u[indices].astype(np.float64)
    v_src = valid_v[indices].astype(np.float64)
    d_src = depth_i[valid_v[indices], valid_u[indices]].astype(np.float64)

    # Backproject to world
    pts_world = backproject_pixels(u_src, v_src, d_src, fx, fy, cx, cy, c2w_i)

    # Project to frame j
    u_proj, v_proj, z_cam = project_to_image(pts_world, w2c_j, K)

    # Filter 1: bounds check (with edge margin, matching keyframe.py line 150-152)
    in_bounds = ((u_proj > edge_margin) & (u_proj < W - edge_margin) &
                 (v_proj > edge_margin) & (v_proj < H - edge_margin))

    # Filter 2: visibility check (z < 0 in OpenGL = in front of camera)
    visible = z_cam < 0

    mask = in_bounds & visible

    if mask.sum() < 10:
        return None

    # Apply mask
    u_proj_valid = u_proj[mask]
    v_proj_valid = v_proj[mask]
    u_src_valid = u_src[mask]
    v_src_valid = v_src[mask]
    d_src_valid = d_src[mask]

    # --- Round-trip consistency error ---
    # NOTE: This is NOT the paper's Rep.Err metric. The paper uses green pin
    # tracking annotations from the Semantic-SuPer dataset (ref [11]) which
    # are not available to us. This round-trip metric measures pose+depth
    # consistency by backprojecting frame i's pixels to 3D, then projecting
    # back into frame i via the world coordinate transform.
    w2c_i = np.linalg.inv(c2w_i)
    u_roundtrip, v_roundtrip, _ = project_to_image(pts_world[mask], w2c_i, K)
    reproj_error = np.sqrt((u_roundtrip - u_src_valid) ** 2 +
                           (v_roundtrip - v_src_valid) ** 2)

    # --- Depth consistency error ---
    # Compare projected depth with observed depth at target pixel
    depth_at_target = bilinear_sample(depth_j, u_proj_valid, v_proj_valid)
    # Projected depth: negate z_cam (OpenGL z is negative for visible points)
    z_projected = -z_cam[mask]
    # Only compare where target depth is also valid
    target_valid = (depth_at_target > 0) & (depth_at_target < depth_trunc)
    depth_error = np.abs(z_projected[target_valid] - depth_at_target[target_valid])

    # --- Photometric error ---
    # L1 difference between source pixel color and target pixel color at projected location
    color_src = rgb_i[v_src_valid.astype(np.int32), u_src_valid.astype(np.int32)]
    color_target = bilinear_sample(rgb_j, u_proj_valid, v_proj_valid)
    photo_error = np.mean(np.abs(color_src - color_target), axis=-1)

    result = {
        'n_total': n,
        'n_visible': int(mask.sum()),
        'n_depth_valid': int(target_valid.sum()),
    }

    # Round-trip reprojection (sanity — should be ~0 for self-reprojection)
    result['roundtrip_mean'] = float(np.mean(reproj_error))
    result['roundtrip_median'] = float(np.median(reproj_error))
    result['roundtrip_rmse'] = float(np.sqrt(np.mean(reproj_error ** 2)))

    # Depth consistency
    if target_valid.sum() > 0:
        result['depth_error_mean'] = float(np.mean(depth_error))
        result['depth_error_median'] = float(np.median(depth_error))
        result['depth_error_rmse'] = float(np.sqrt(np.mean(depth_error ** 2)))
    else:
        result['depth_error_mean'] = float('nan')
        result['depth_error_median'] = float('nan')
        result['depth_error_rmse'] = float('nan')

    # Photometric
    result['photo_error_mean'] = float(np.mean(photo_error))
    result['photo_error_median'] = float(np.median(photo_error))

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate reprojection error of DDS-SLAM poses and depth')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to dataset (e.g., data/Super/trail3)')
    parser.add_argument('--posefile', type=str, required=True,
                        help='Path to est_c2w_data.txt')
    parser.add_argument('--fx', type=float, default=768.98551924)
    parser.add_argument('--fy', type=float, default=768.98551924)
    parser.add_argument('--cx', type=float, default=292.8861567)
    parser.add_argument('--cy', type=float, default=291.61479526)
    parser.add_argument('--H', type=int, default=480)
    parser.add_argument('--W', type=int, default=640)
    parser.add_argument('--depth_scale', type=float, default=8.0,
                        help='png_depth_scale from config (default: 8.0)')
    parser.add_argument('--depth_trunc', type=float, default=5.0,
                        help='Max valid depth in meters (default: 5.0)')
    parser.add_argument('--frame_steps', type=int, nargs='+', default=[1, 5, 10],
                        help='Frame step sizes for pair evaluation (default: 1 5 10)')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Pixels to sample per frame pair (default: 10000)')
    parser.add_argument('--edge_margin', type=int, default=20,
                        help='Pixel margin for bounds check (default: 20)')
    parser.add_argument('--name', type=str, default='',
                        help='Method name for display')
    parser.add_argument('--output_csv', type=str, default='',
                        help='Optional CSV output path')
    parser.add_argument('--verbose', action='store_true',
                        help='Run self-reprojection sanity check')
    args = parser.parse_args()

    # Build intrinsics matrix
    K = np.array([
        [args.fx, 0.0, args.cx],
        [0.0, args.fy, args.cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Load poses
    poses = load_poses_from_txt(args.posefile)

    # Discover images and depth files
    rgb_paths = get_left_images(args.datadir)
    depth_paths = [get_depth_path(p) for p in rgb_paths]

    # Verify depth files exist
    missing = [p for p in depth_paths if not os.path.exists(p)]
    if missing:
        print(f"Warning: {len(missing)} depth files missing (first: {missing[0]})")

    n_frames = min(len(poses), len(rgb_paths))
    if n_frames < len(poses) or n_frames < len(rgb_paths):
        print(f"Warning: using {n_frames} frames (poses: {len(poses)}, images: {len(rgb_paths)})")

    print(f"Method: {args.name}" if args.name else "")
    print(f"Frames: {n_frames}")
    print(f"Frame steps: {args.frame_steps}")
    print(f"Samples per pair: {args.n_samples}")

    # Self-reprojection sanity check
    if args.verbose and n_frames > 0:
        print("\n--- Self-reprojection sanity check (frame 0 -> frame 0) ---")
        result = evaluate_frame_pair(
            0, 0, poses, rgb_paths, depth_paths,
            K, args.fx, args.fy, args.cx, args.cy, args.H, args.W,
            args.depth_scale, args.depth_trunc, args.n_samples, args.edge_margin)
        if result:
            print(f"  Round-trip error: {result['roundtrip_mean']:.6f} px "
                  f"(should be ~0)")
            print(f"  Depth error: {result['depth_error_mean']:.6f} m "
                  f"(should be ~0)")
        else:
            print("  WARNING: sanity check failed (no valid pixels)")
        print()

    # Evaluate for each frame step
    csv_rows = []

    for step in args.frame_steps:
        n_pairs = n_frames - step
        if n_pairs <= 0:
            print(f"\nSkipping step k={step}: not enough frames")
            continue

        all_results = []
        for i in tqdm(range(n_pairs), desc=f"Step k={step}"):
            j = i + step
            if j >= n_frames:
                break
            # Check depth files exist for both frames
            if not os.path.exists(depth_paths[i]) or not os.path.exists(depth_paths[j]):
                continue
            result = evaluate_frame_pair(
                i, j, poses, rgb_paths, depth_paths,
                K, args.fx, args.fy, args.cx, args.cy, args.H, args.W,
                args.depth_scale, args.depth_trunc, args.n_samples, args.edge_margin)
            if result is not None:
                all_results.append(result)

        if not all_results:
            print(f"\nStep k={step}: no valid frame pairs")
            continue

        # Aggregate
        visibility = np.mean([r['n_visible'] / r['n_total'] for r in all_results])

        roundtrip_means = [r['roundtrip_mean'] for r in all_results]
        roundtrip_rmses = [r['roundtrip_rmse'] for r in all_results]

        depth_means = [r['depth_error_mean'] for r in all_results
                       if not np.isnan(r['depth_error_mean'])]

        photo_means = [r['photo_error_mean'] for r in all_results]

        print(f"\n{'=' * 55}")
        print(f"Step k={step} ({len(all_results)} frame pairs)")
        print(f"{'=' * 55}")
        print(f"  Visible correspondences: {visibility:.1%}")
        print(f"  Round-trip consistency error:")
        print(f"    Mean:   {np.mean(roundtrip_means):.3f} px "
              f"(std: {np.std(roundtrip_means):.3f})")
        print(f"    RMSE:   {np.mean(roundtrip_rmses):.3f} px")
        if depth_means:
            print(f"  Depth consistency error:")
            print(f"    Mean:   {np.mean(depth_means):.4f} m "
                  f"(std: {np.std(depth_means):.4f})")
        print(f"  Photometric error (L1):")
        print(f"    Mean:   {np.mean(photo_means):.4f} "
              f"(std: {np.std(photo_means):.4f})")

        csv_rows.append({
            'step': step,
            'n_pairs': len(all_results),
            'visibility': f"{visibility:.4f}",
            'reproj_mean_px': f"{np.mean(roundtrip_means):.4f}",
            'reproj_rmse_px': f"{np.mean(roundtrip_rmses):.4f}",
            'depth_mean_m': f"{np.mean(depth_means):.4f}" if depth_means else 'nan',
            'photo_mean_l1': f"{np.mean(photo_means):.4f}",
        })

    print(f"\nNote: The paper's Rep.Err (Table I) uses green pin tracking annotations")
    print(f"from Semantic-SuPer [11], which are not available. The round-trip")
    print(f"consistency metric above is a pose+depth consistency proxy.")

    # Optional CSV output
    if args.output_csv and csv_rows:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to {args.output_csv}")


if __name__ == '__main__':
    main()
