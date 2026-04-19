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
    # Patch in missing inference config keys (see infer_trajectory.py)
    model_config = checkp['config']['model']
    model_config.setdefault('lbgfs_iters', 2)
    model_config.setdefault('use_weights', True)
    model_config.setdefault('img_shape', [640, 512])
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
        baseline_mm = np.linalg.norm(tvec)  # T is in mm

        # We need the rectified bf, not raw. The original robust-pose-estimator
        # computes bf = |Tx_rectified| * fx_rectified via OpenCV stereo rectification.
        # We replicate this by running stereoRectify with the same calibration.
        lkmat = np.eye(3, dtype=np.float64)
        lkmat[0, 0] = float(config['StereoLeft']['fc_x'])
        lkmat[1, 1] = float(config['StereoLeft']['fc_y'])
        lkmat[0, 2] = float(config['StereoLeft']['cc_x'])
        lkmat[1, 2] = float(config['StereoLeft']['cc_y'])

        rkmat = np.eye(3, dtype=np.float64)
        rkmat[0, 0] = float(config['StereoRight']['fc_x'])
        rkmat[1, 1] = float(config['StereoRight']['fc_y'])
        rkmat[0, 2] = float(config['StereoRight']['cc_x'])
        rkmat[1, 2] = float(config['StereoRight']['cc_y'])

        ld = np.array([float(config['StereoLeft'][f'kc_{i}']) for i in range(8)], dtype=np.float64)
        rd = np.array([float(config['StereoRight'][f'kc_{i}']) for i in range(8)], dtype=np.float64)

        rmat = np.array([float(config['StereoRight'][f'R_{i}']) for i in range(9)],
                        dtype=np.float64).reshape(3, 3)

        # Scale intrinsics to target resolution (640x512)
        orig_w = float(config['StereoLeft']['res_x'])
        target_w, target_h = 640, 512
        scale_factor = target_w / orig_w
        h_crop = int(float(config['StereoLeft']['res_y']) * scale_factor - target_h) // 2

        lkmat[:2] *= scale_factor
        rkmat[:2] *= scale_factor
        lkmat[1, 2] -= h_crop
        rkmat[1, 2] -= h_crop

        # Run stereoRectify to get rectified projection matrices
        r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
            cameraMatrix1=lkmat, distCoeffs1=ld,
            cameraMatrix2=rkmat, distCoeffs2=ld,  # Note: original uses ld for both (bug)
            imageSize=(target_w, target_h),
            R=rmat, T=tvec.reshape(3, 1),
            alpha=0
        )

        # Compute bf the same way as robust-pose-estimator's get_rectified_calib()
        # p2[0,3] = Tx * fx (in mm*px), so Tx = p2[0,3]/p2[0,0] (in mm)
        Tx_rect = p2[0, 3] / p2[0, 0]  # mm
        fx_rect = p1[0, 0]  # px
        bf = abs(Tx_rect) * fx_rect  # mm*px (same units as original)
        baseline = abs(Tx_rect) / 1000.0  # meters

        print(f"Rectified calibration: fx={fx_rect:.2f}px, Tx={Tx_rect:.4f}mm")
        print(f"bf={bf:.2f} (mm·px), baseline={baseline:.6f}m")
        return bf, baseline

    else:
        raise FileNotFoundError(f"No calibration file found in {datadir}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate depth using robust-pose-estimator pretrained RAFT stereo matching')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to StereoMIS sequence (e.g., data/P2_1)')
    parser.add_argument('--depth_scale', type=float, default=100.0,
                        help='Depth scale for PNG output (default: 100 for 10mm resolution). Must match png_depth_scale in YAML config.')
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

    # Load instrument masks if available (from StereoMIS masks/ directory)
    mask_dir = os.path.join(args.datadir, 'masks')
    instrument_masks = {}
    if os.path.isdir(mask_dir):
        for mf in sorted(glob.glob(os.path.join(mask_dir, '*.png'))):
            instrument_masks[os.path.basename(mf)] = mf
        print(f"Found {len(instrument_masks)} instrument masks in {mask_dir}")

    # Process each stereo pair
    masked_count = 0
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
            depth_clip_max_mm = 250.0
            scale = 1.0 / depth_clip_max_mm
            baseline_for_model = torch.tensor([bf * scale]).to(device)
            depth, flow, valid = model.flow2depth(left_t, right_t, baseline_for_model)

            # Convert normalized depth [0,1] to meters
            depth_np = depth.squeeze().cpu().numpy().astype(np.float32)
            depth_np = depth_np * depth_clip_max_mm / 1000.0  # to meters

            # Clamp to valid range
            depth_np = np.clip(depth_np, 0.0, 10.0)

            # Apply masking (matching robust-pose-estimator pipeline)
            # 1. Specularity mask: pixels where sum(RGB) >= 3*255*0.96 = 734
            spec_mask = left_img.sum(axis=-1) >= (3 * 255 * 0.96)
            # Erode mask by 11px to expand exclusion zone
            spec_mask_eroded = cv2.dilate(spec_mask.astype(np.uint8),
                                          kernel=np.ones((11, 11))) > 0
            depth_np[spec_mask_eroded] = 0.0

            # 2. Instrument mask from masks/ directory
            frame_name = os.path.splitext(os.path.basename(left_path))[0]
            if frame_name.endswith('l'):
                frame_name = frame_name[:-1]
            mask_key = frame_name + 'l.png'
            if mask_key in instrument_masks:
                inst_mask = cv2.imread(instrument_masks[mask_key], cv2.IMREAD_GRAYSCALE)
                if inst_mask is not None:
                    # In StereoMIS masks: 0 = instrument/invalid, >0 = valid tissue
                    inst_invalid = inst_mask == 0
                    if inst_mask.shape != depth_np.shape:
                        inst_invalid = cv2.resize(inst_invalid.astype(np.uint8),
                                                  (depth_np.shape[1], depth_np.shape[0]),
                                                  interpolation=cv2.INTER_NEAREST) > 0
                    depth_np[inst_invalid] = 0.0

            # 3. flow2depth valid mask (depth <= 0 or > 1.0 normalized)
            valid_np = valid.squeeze().cpu().numpy()
            depth_np[~valid_np] = 0.0

            zero_pct = (depth_np == 0).sum() / depth_np.size * 100
            if zero_pct > 0:
                masked_count += 1

            # Save as 16-bit PNG
            depth_png = (depth_np * args.depth_scale).astype(np.uint16)
            out_path = os.path.join(output_dir, f'{frame_name}.png')
            cv2.imwrite(out_path, depth_png)

    print(f"Done. Generated {len(left_files)} depth maps in {output_dir}")
    print(f"Frames with masked pixels: {masked_count}/{len(left_files)}")

    # Verify a sample
    sample = cv2.imread(os.path.join(output_dir, os.listdir(output_dir)[0]),
                        cv2.IMREAD_UNCHANGED)
    d_m = sample.astype(np.float32) / args.depth_scale
    valid_px = d_m[d_m > 0]
    zero_pct = (d_m == 0).sum() / d_m.size * 100
    print(f"Sample depth: shape={sample.shape}, dtype={sample.dtype}, "
          f"range=[{valid_px.min():.3f}, {valid_px.max():.3f}]m, "
          f"mean={valid_px.mean():.3f}m, zero={zero_pct:.1f}%")


if __name__ == '__main__':
    main()
