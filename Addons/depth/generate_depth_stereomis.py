"""
Depth map generation for DDS-SLAM StereoMIS dataset.

Generates 16-bit PNG depth maps in {datadir}/depth/ from left frames
in {datadir}/video_frames/*l.png.

Pixel encoding: depth_meters * png_depth_scale (default 100).
  - pixel value 100 = 1.0 meter
  - pixel value 0   = invalid / no depth

Supports 2 methods:
  - depth_anything : Depth Anything V2 Metric Indoor (monocular, HuggingFace)
  - raft_stereo    : RAFT-Stereo (stereo pairs, requires weights)

Usage:
  python Addons/generate_depth_stereomis.py --datadir data/P2_1 --method depth_anything
  python Addons/generate_depth_stereomis.py --datadir data/P2_1 --method raft_stereo
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_left_images(datadir):
    """Find all left RGB images sorted by name."""
    files = sorted(glob.glob(os.path.join(datadir, 'video_frames', '*l.png')))
    if not files:
        raise FileNotFoundError(f"No left images found in {datadir}/video_frames/")
    print(f"Found {len(files)} left images")
    return files


def save_depth_png(datadir, left_path, depth_meters, png_depth_scale=100):
    """Save metric depth as 16-bit PNG in depth/ directory.

    Uses the same filename as the left frame so sorted glob alignment works.
    """
    depth_dir = os.path.join(datadir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    fname = os.path.basename(left_path)  # e.g., "000001l.png"
    out_path = os.path.join(depth_dir, fname)

    depth_scaled = np.clip(depth_meters * png_depth_scale, 0, 65535)
    depth_uint16 = depth_scaled.astype(np.uint16)

    cv2.imwrite(out_path, depth_uint16)
    return out_path


# =============================================================================
# Method 1: Depth Anything V2 (Metric Indoor)
# =============================================================================
def run_depth_anything(datadir, png_depth_scale, target_h=512, target_w=640):
    """Generate depth using Depth Anything V2 Metric Indoor model."""
    from transformers import pipeline

    print("Loading Depth Anything V2 Metric Indoor model...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
    )

    left_images = get_left_images(datadir)
    for img_path in tqdm(left_images, desc="Depth Anything V2"):
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        result = pipe(image)
        depth = np.array(result["predicted_depth"].squeeze().cpu().numpy(), dtype=np.float32)

        # Resize to target resolution
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        save_depth_png(datadir, img_path, depth, png_depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps in {datadir}/depth/")


# =============================================================================
# Method 2: RAFT-Stereo
# =============================================================================
def setup_raft_stereo():
    """Clone RAFT-Stereo repo and download weights."""
    repo_dir = '/tmp/RAFT-Stereo'
    if not os.path.exists(repo_dir):
        print("Cloning RAFT-Stereo...")
        os.system(f'git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo.git {repo_dir}')

    weights_dir = os.path.join(repo_dir, 'models')
    if not os.path.exists(weights_dir) or not os.listdir(weights_dir):
        print("Downloading RAFT-Stereo weights from Dropbox...")
        os.makedirs(weights_dir, exist_ok=True)
        zip_path = os.path.join(repo_dir, 'models.zip')
        os.system(f'wget -q "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip" -O {zip_path}')
        os.system(f'unzip -q -o {zip_path} -d {repo_dir}')
        os.remove(zip_path)

    return repo_dir


def run_raft_stereo(datadir, png_depth_scale, baseline=0.00416, fx=516.95,
                    target_h=None, target_w=None):
    """Generate depth using RAFT-Stereo with stereo image pairs."""
    repo_dir = setup_raft_stereo()
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, 'core'))

    from raft_stereo import RAFTStereo
    from utils.utils import InputPadder

    class Args:
        restore_ckpt = os.path.join(repo_dir, 'models', 'raftstereo-middlebury.pth')
        shared_backbone = False
        n_downsample = 2
        n_gru_layers = 3
        slow_fast_gru = False
        valid_iters = 32
        hidden_dims = [128] * 3
        corr_implementation = 'reg'
        corr_levels = 4
        corr_radius = 4
        mixed_precision = True
        context_norm = 'batch'

    args = Args()
    print(f"Loading RAFT-Stereo model: {args.restore_ckpt}")
    model = torch.nn.DataParallel(RAFTStereo(args))
    state_dict = torch.load(args.restore_ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.module
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    left_images = get_left_images(datadir)
    with torch.no_grad():
        for left_path in tqdm(left_images, desc="RAFT-Stereo"):
            # Find matching right image: 000001l.png -> 000001r.png
            right_path = left_path.replace('l.png', 'r.png')
            if not os.path.exists(right_path):
                print(f"  Warning: no right image for {os.path.basename(left_path)}, skipping")
                continue

            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float().unsqueeze(0).to(device)

            padder = InputPadder(left_tensor.shape, divis_by=32)
            left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

            _, flow_up = model(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
            disp = -flow_up.squeeze().cpu().numpy()

            disp = padder.unpad(torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)).squeeze().numpy()

            # Convert disparity to depth: Z = fx * baseline / disparity
            min_disp = 0.1
            disp = np.maximum(disp, min_disp)
            depth = (fx * baseline) / disp

            if target_h and target_w and depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            depth = np.clip(depth, 0.0, 10.0)

            save_depth_png(datadir, left_path, depth, png_depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps in {datadir}/depth/")
    sys.path.remove(repo_dir)
    sys.path.remove(os.path.join(repo_dir, 'core'))


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate depth maps for DDS-SLAM StereoMIS dataset')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to StereoMIS sequence (e.g., data/P2_1)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['depth_anything', 'raft_stereo'],
                        help='Depth estimation method')
    parser.add_argument('--png_depth_scale', type=float, default=100.0,
                        help='Depth scale for PNG encoding (default: 100, matches stereomis.yaml)')
    parser.add_argument('--baseline', type=float, default=0.00416,
                        help='Stereo baseline in meters (default: 0.00416m = 4.16mm for P2)')
    parser.add_argument('--fx', type=float, default=516.95,
                        help='Focal length in pixels at 640x512 (default: 516.95 from stereomis.yaml)')
    args = parser.parse_args()

    print(f"Method: {args.method}")
    print(f"Data dir: {args.datadir}")
    print(f"PNG depth scale: {args.png_depth_scale}")

    if args.method == 'depth_anything':
        run_depth_anything(args.datadir, args.png_depth_scale)
    elif args.method == 'raft_stereo':
        print(f"Baseline: {args.baseline}m, fx: {args.fx}px")
        run_raft_stereo(args.datadir, args.png_depth_scale,
                        baseline=args.baseline, fx=args.fx)


if __name__ == '__main__':
    main()
