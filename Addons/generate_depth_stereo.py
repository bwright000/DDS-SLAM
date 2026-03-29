"""
Generate depth maps for StereoMIS using the robust-pose-estimator's pretrained
RAFT-based stereo matching model — the same depth source the DDS-SLAM paper used.

This script:
1. Clones robust-pose-estimator (with RAFT submodule)
2. Loads the pretrained poseNet checkpoint
3. Calls PoseNet.flow2depth() on each stereo pair
4. Saves depth as 16-bit PNG (values = depth_meters * depth_scale)

Usage:
  python Addons/generate_depth_stereo.py \
    --datadir data/P2_1 \
    --depth_scale 100 \
    --checkpoint /tmp/robust-pose-estimator/trained/poseNet_2xf8up4b.pth
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def setup_repo():
    """Clone robust-pose-estimator with submodules if not already present."""
    repo_dir = '/tmp/robust-pose-estimator'
    if not os.path.exists(os.path.join(repo_dir, 'core', 'RAFT', 'core')):
        print("Cloning robust-pose-estimator with submodules...")
        if os.path.exists(repo_dir):
            import shutil
            shutil.rmtree(repo_dir)
        os.system(f'git clone --recurse-submodules https://github.com/aimi-lab/robust-pose-estimator.git {repo_dir}')
    else:
        print(f"Using existing repo at {repo_dir}")
    return repo_dir


def load_model(repo_dir, checkpoint_path, device):
    """Load PoseNet with pretrained weights."""
    # Add repo to path
    sys.path.insert(0, repo_dir)

    # Patch out lietorch imports before loading
    import importlib
    import types

    # Create a comprehensive dummy lietorch module so imports don't fail
    # We only need RAFT's flow2depth — lietorch is for pose estimation which we skip
    lietorch_mock = types.ModuleType('lietorch')

    class DummySE3:
        """Dummy SE3 class to satisfy imports."""
        def __init__(self, *args, **kwargs):
            pass
        @staticmethod
        def Identity(*args, **kwargs):
            return torch.eye(4).unsqueeze(0)
        def vec(self):
            return torch.zeros(1, 7)

    class DummyLieGroupParameter(torch.nn.Parameter):
        """Dummy LieGroupParameter to satisfy imports."""
        def __new__(cls, data=None, *args, **kwargs):
            if data is None:
                data = torch.zeros(1, 7)
            return super().__new__(cls, data)

    lietorch_mock.SE3 = DummySE3
    lietorch_mock.LieGroupParameter = DummyLieGroupParameter
    sys.modules['lietorch'] = lietorch_mock

    from core.pose.pose_net import PoseNet

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Create model from checkpoint config
    model_config = checkp['config']['model']
    model = PoseNet(model_config)

    # Handle DataParallel state dict
    state_dict = checkp['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    model.to(device)
    print("Model loaded successfully")
    return model


def load_calibration(datadir):
    """Load stereo calibration from StereoCalibration.ini or rectified_calib.txt."""
    import configparser

    ini_file = os.path.join(datadir, 'StereoCalibration.ini')
    rect_file = os.path.join(datadir, 'rectified_calib.txt')

    if os.path.exists(rect_file):
        # Parse rectified_calib.txt
        calib = {}
        with open(rect_file) as f:
            for line in f:
                key, val = line.strip().split(': ', 1)
                try:
                    calib[key] = float(val)
                except ValueError:
                    calib[key] = val
        baseline = calib.get('baseline', 0.00416)
        fx = calib.get('fx', 560.02)
        bf = calib.get('bf', baseline * fx)
        print(f"Calibration from rectified_calib.txt: bf={bf:.2f}, baseline={baseline:.6f}m")
        return bf, baseline

    elif os.path.exists(ini_file):
        config = configparser.ConfigParser()
        config.read(ini_file)
        tvec = np.array([float(config['StereoRight'][f'T_{i}']) for i in range(3)])
        baseline_raw = np.linalg.norm(tvec)
        # StereoCalibration.ini uses millimeters for translation
        # da Vinci baseline is ~4-6mm, so if value > 1.0 it's in mm
        if baseline_raw > 1.0:
            baseline = baseline_raw / 1000.0  # convert mm to meters
            print(f"Baseline {baseline_raw:.3f}mm detected (converting to {baseline:.6f}m)")
        else:
            baseline = baseline_raw
        fx = float(config['StereoLeft']['fc_x'])
        bf = baseline * fx
        print(f"Calibration from StereoCalibration.ini: bf={bf:.4f}, baseline={baseline:.6f}m, fx={fx:.2f}")
        return bf, baseline

    else:
        raise FileNotFoundError(f"No calibration file found in {datadir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth using robust-pose-estimator pretrained RAFT stereo matching')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to StereoMIS sequence (e.g., data/P2_1)')
    parser.add_argument('--depth_scale', type=float, default=100.0,
                        help='Depth scale for PNG output (default: 100)')
    parser.add_argument('--checkpoint', type=str,
                        default='/tmp/robust-pose-estimator/trained/poseNet_2xf8up4b.pth',
                        help='Path to pretrained checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {datadir}/depth)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Setup
    repo_dir = setup_repo()

    # Load calibration
    bf, baseline = load_calibration(args.datadir)

    # Load model
    model = load_model(repo_dir, args.checkpoint, device)

    # Find stereo pairs
    left_files = sorted(glob.glob(os.path.join(args.datadir, 'video_frames', '*l.png')))
    if not left_files:
        raise FileNotFoundError(f"No left frames found in {args.datadir}/video_frames/")
    print(f"Found {len(left_files)} stereo pairs")

    # Output directory
    output_dir = args.output_dir or os.path.join(args.datadir, 'depth')
    os.makedirs(output_dir, exist_ok=True)

    # Process each stereo pair
    with torch.no_grad():
        for left_path in tqdm(left_files, desc="RAFT Stereo Depth"):
            right_path = left_path.replace('l.png', 'r.png')
            if not os.path.exists(right_path):
                continue

            # Load images
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            # To tensor [1, 3, H, W] float
            left_t = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            right_t = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

            # Compute depth via RAFT stereo matching
            baseline_t = torch.tensor([baseline]).to(device)
            depth, flow, valid = model.flow2depth(left_t, right_t, baseline_t)

            # depth shape: [1, 1, H, W] or [1, H, W]
            depth_np = depth.squeeze().cpu().numpy().astype(np.float32)

            # Clamp to valid range
            depth_np = np.clip(depth_np, 0.0, 10.0)

            # Save as 16-bit PNG
            depth_png = (depth_np * args.depth_scale).astype(np.uint16)

            # Derive output filename
            frame_name = os.path.splitext(os.path.basename(left_path))[0]
            if frame_name.endswith('l'):
                frame_name = frame_name[:-1]
            out_path = os.path.join(output_dir, f'{frame_name}.png')
            cv2.imwrite(out_path, depth_png)

    print(f"Done. Generated {len(left_files)} depth maps in {output_dir}")

    # Verify a sample
    sample = cv2.imread(os.path.join(output_dir, os.listdir(output_dir)[0]),
                        cv2.IMREAD_UNCHANGED)
    d_m = sample.astype(np.float32) / args.depth_scale
    valid = d_m[d_m > 0]
    print(f"Sample depth: shape={sample.shape}, dtype={sample.dtype}, "
          f"range=[{valid.min():.3f}, {valid.max():.3f}]m, mean={valid.mean():.3f}m")


if __name__ == '__main__':
    main()
