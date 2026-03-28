"""
Preprocess StereoMIS dataset for DDS-SLAM.

Extracts stereo frames from video, applies rectification + undistortion
using calibration from StereoCalibration.ini, and resizes to 640x512.

This replaces the robust-pose-estimator preprocessing script which has
dependency issues (lietorch, pinned torch==1.13.0).

Usage:
  python Addons/preprocess_stereomis.py /path/to/StereoMIS/P2_1

  # Or process all sequences:
  python Addons/preprocess_stereomis.py /path/to/StereoMIS --all

Input structure:
  P2_1/
  ├── IFBS_ENDOSCOPE-part0001.mp4
  ├── StereoCalibration.ini
  ├── groundtruth.txt
  └── masks/

Output structure (added to same directory):
  P2_1/
  ├── video_frames/
  │   ├── 000001l.png   (rectified, undistorted, 640x512)
  │   ├── 000001r.png
  │   └── ...
  └── (existing files unchanged)
"""

import argparse
import configparser
import glob
import os

import cv2
import numpy as np
from tqdm import tqdm


def load_calib_ini(fname):
    """Load stereo calibration from .ini file."""
    config = configparser.ConfigParser()
    config.read(fname)

    img_size = (int(float(config['StereoLeft']['res_x'])),
                int(float(config['StereoLeft']['res_y'])))

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

    ld = np.array([float(config['StereoLeft'][f'kc_{i}']) for i in range(8)],
                  dtype=np.float64)
    rd = np.array([float(config['StereoRight'][f'kc_{i}']) for i in range(8)],
                  dtype=np.float64)

    rmat = np.array([float(config['StereoRight'][f'R_{i}']) for i in range(9)],
                    dtype=np.float64).reshape(3, 3)
    tvec = np.array([float(config['StereoRight'][f'T_{i}']) for i in range(3)],
                    dtype=np.float64)

    return {
        'lkmat': lkmat, 'rkmat': rkmat,
        'ld': ld, 'rd': rd,
        'R': rmat, 'T': tvec,
        'img_size': img_size
    }


def build_rectification_maps(cal, target_size=(640, 512)):
    """Compute stereo rectification + undistortion maps.

    Args:
        cal: calibration dict from load_calib_ini
        target_size: (width, height) output size

    Returns:
        maps: dict with lmap1, lmap2, rmap1, rmap2
        new_calib: rectified calibration info
    """
    # Scale intrinsics if resizing
    scale = target_size[0] / cal['img_size'][0]
    h_crop = int((cal['img_size'][1] * scale - target_size[1]) / 2)

    lkmat = cal['lkmat'].copy()
    rkmat = cal['rkmat'].copy()
    lkmat[:2] *= scale
    rkmat[:2] *= scale
    lkmat[1, 2] -= h_crop
    rkmat[1, 2] -= h_crop

    # Stereo rectification
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        cameraMatrix1=lkmat, distCoeffs1=cal['ld'],
        cameraMatrix2=rkmat, distCoeffs2=cal['rd'],
        imageSize=target_size,
        R=cal['R'], T=cal['T'].reshape(3, 1),
        alpha=0
    )

    # Undistortion + rectification maps
    lmap1, lmap2 = cv2.initUndistortRectifyMap(
        lkmat, cal['ld'], r1, p1, target_size, cv2.CV_32FC1)
    rmap1, rmap2 = cv2.initUndistortRectifyMap(
        rkmat, cal['rd'], r2, p2, target_size, cv2.CV_32FC1)

    maps = {'lmap1': lmap1, 'lmap2': lmap2, 'rmap1': rmap1, 'rmap2': rmap2}

    # Rectified intrinsics
    baseline = np.sqrt(np.sum((p2[0, 3] / p2[0, 0]) ** 2))
    new_calib = {
        'fx': p1[0, 0], 'fy': p1[1, 1],
        'cx': p1[0, 2], 'cy': p1[1, 2],
        'baseline': baseline,
        'bf': baseline * p1[0, 0],
        'img_size': target_size,
    }

    return maps, new_calib


def process_sequence(seq_path, target_size=(640, 512)):
    """Process one StereoMIS sequence: extract, rectify, resize frames."""

    # Find calibration and video
    calib_file = os.path.join(seq_path, 'StereoCalibration.ini')
    video_files = glob.glob(os.path.join(seq_path, '*.mp4'))

    if not os.path.exists(calib_file):
        print(f"  Skipping {seq_path}: no StereoCalibration.ini")
        return
    if not video_files:
        print(f"  Skipping {seq_path}: no .mp4 file")
        return

    video_file = video_files[0]
    output_dir = os.path.join(seq_path, 'video_frames')

    # Check if already processed
    if os.path.exists(output_dir):
        existing = glob.glob(os.path.join(output_dir, '*l.png'))
        if existing:
            print(f"  Already processed ({len(existing)} frames). Skipping.")
            return

    print(f"  Video: {os.path.basename(video_file)}")
    print(f"  Calibration: {os.path.basename(calib_file)}")

    # Load calibration and build rectification maps
    cal = load_calib_ini(calib_file)
    print(f"  Raw frame size: {cal['img_size']}")
    maps, new_calib = build_rectification_maps(cal, target_size)
    print(f"  Target size: {target_size}")
    print(f"  Rectified intrinsics: fx={new_calib['fx']:.2f}, fy={new_calib['fy']:.2f}, "
          f"cx={new_calib['cx']:.2f}, cy={new_calib['cy']:.2f}")
    print(f"  Baseline: {new_calib['baseline']:.6f}m, bf={new_calib['bf']:.2f}")

    os.makedirs(output_dir, exist_ok=True)

    # Process video
    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Total frames in video: {total_frames}, FPS: {fps:.1f}")

    frame_idx = 0
    for _ in tqdm(range(total_frames), desc="  Extracting"):
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Split top/bottom stereo
        H = frame.shape[0] // 2
        left_raw = frame[:H, :, :]
        right_raw = frame[H:, :, :]

        # Resize to match calibration's original resolution if needed
        orig_size = cal['img_size']  # (width, height)
        if left_raw.shape[1] != orig_size[0] or left_raw.shape[0] != orig_size[1]:
            left_raw = cv2.resize(left_raw, orig_size)
            right_raw = cv2.resize(right_raw, orig_size)

        # Apply rectification + undistortion
        left_rect = cv2.remap(left_raw, maps['lmap1'], maps['lmap2'],
                              interpolation=cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_raw, maps['rmap1'], maps['rmap2'],
                               interpolation=cv2.INTER_LINEAR)

        # Save
        cv2.imwrite(os.path.join(output_dir, f'{frame_idx:06d}l.png'), left_rect)
        cv2.imwrite(os.path.join(output_dir, f'{frame_idx:06d}r.png'), right_rect)

    cap.release()
    print(f"  Done. Extracted {frame_idx} rectified stereo pairs to {output_dir}")

    # Save rectified calibration for reference
    calib_out = os.path.join(seq_path, 'rectified_calib.txt')
    with open(calib_out, 'w') as f:
        for k, v in new_calib.items():
            f.write(f"{k}: {v}\n")
    print(f"  Rectified calibration saved to {calib_out}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess StereoMIS dataset for DDS-SLAM')
    parser.add_argument('input', type=str,
                        help='Path to sequence (e.g., data/P2_1) or dataset root')
    parser.add_argument('--all', action='store_true',
                        help='Process all sequences in the directory')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=512)
    args = parser.parse_args()

    target_size = (args.width, args.height)

    if args.all:
        # Process all subdirectories
        sequences = sorted(glob.glob(os.path.join(args.input, '*')))
        sequences = [s for s in sequences if os.path.isdir(s)]
        print(f"Found {len(sequences)} sequences")
        for seq in sequences:
            name = os.path.basename(seq)
            print(f"\nProcessing {name}...")
            process_sequence(seq, target_size)
    else:
        name = os.path.basename(args.input)
        print(f"Processing {name}...")
        process_sequence(args.input, target_size)


if __name__ == '__main__':
    main()
