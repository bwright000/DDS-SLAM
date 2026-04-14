"""
Regenerate StereoMIS depth PNGs with mm quantization + specularity/instrument masking.

Takes existing RAFT-Stereo depth PNGs (encoded as meters*100) and produces new PNGs:
- Re-encoded as uint16 = round(depth_meters * 1000)  -> mm resolution
- Zero out specular pixels (RGB sum > 3*255*0.96) with 11x11 erosion
- Zero out instrument/invalid pixels per mask directory

Run from repo root:
  python Addons/regenerate_stereomis_depth.py \
    --depth_in  Output/StereoMIS_depth_maps_RAFT \
    --rgb_dir   F:/Datasets/StereoMIS/StereoMIS/P2_1/video_frames \
    --mask_dir  F:/Datasets/StereoMIS/P2_1/masks \
    --depth_out data/P2_1/depth

Then set png_depth_scale: 1000 in configs/StereoMIS/stereomis.yaml.
"""

import argparse
import glob
import os
import re

import cv2
import numpy as np
from tqdm import tqdm


def parse_frame_num(path):
    base = os.path.splitext(os.path.basename(path))[0]
    digits = re.sub(r'[^0-9]', '', base)
    return int(digits)


def mask_specularities(rgb_bgr, thr=0.96):
    # robust-pose-estimator convention: RGB sum < 3*255*thr is valid
    rgb_sum = rgb_bgr.astype(np.int32).sum(axis=-1)
    valid = rgb_sum < (3 * 255 * thr)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)
    return valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--depth_in', required=True)
    ap.add_argument('--rgb_dir', required=True)
    ap.add_argument('--mask_dir', default='')
    ap.add_argument('--depth_out', required=True)
    ap.add_argument('--in_scale', type=float, default=100.0,
                    help='scale used when encoding the input PNGs (default: 100 = meters*100)')
    ap.add_argument('--out_scale', type=float, default=1000.0,
                    help='scale for output PNGs (default: 1000 = meters*1000 = mm)')
    ap.add_argument('--specularity_thr', type=float, default=0.96)
    ap.add_argument('--limit', type=int, default=0, help='process only first N (0 = all)')
    args = ap.parse_args()

    os.makedirs(args.depth_out, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(args.depth_in, '*.png')))
    if args.limit:
        depth_files = depth_files[:args.limit]
    print(f'Processing {len(depth_files)} depth PNGs')

    # Index masks by frame number (masks only present every other frame)
    mask_files = {}
    if args.mask_dir:
        for p in glob.glob(os.path.join(args.mask_dir, '*.png')):
            mask_files[parse_frame_num(p)] = p
        print(f'Found {len(mask_files)} mask PNGs (keyed by frame number)')

    last_mask = None
    stats = {'unique_in': [], 'unique_out': [], 'valid_pct': []}

    for dpath in tqdm(depth_files):
        frame_num = parse_frame_num(dpath)

        # Load depth (meters*in_scale)
        d = cv2.imread(dpath, cv2.IMREAD_UNCHANGED)
        depth_m = d.astype(np.float32) / args.in_scale
        stats['unique_in'].append(len(np.unique(d)))

        # Build valid mask
        H, W = depth_m.shape
        valid = np.ones((H, W), dtype=bool)

        # Specularity mask from RGB
        rgb_path = os.path.join(args.rgb_dir, f'{frame_num:06d}l.png')
        if os.path.exists(rgb_path):
            rgb = cv2.imread(rgb_path)
            if rgb.shape[:2] != (H, W):
                rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
            valid &= mask_specularities(rgb, thr=args.specularity_thr)

        # Instrument mask (reuse previous when missing)
        if args.mask_dir:
            if frame_num in mask_files:
                mraw = cv2.imread(mask_files[frame_num], cv2.IMREAD_UNCHANGED)
                if mraw.ndim == 3:
                    mraw = mraw[..., 0]
                if mraw.shape[:2] != (H, W):
                    mraw = cv2.resize(mraw, (W, H), interpolation=cv2.INTER_NEAREST)
                last_mask = (mraw > 127)
            if last_mask is not None:
                valid &= last_mask

        # Re-encode depth as uint16 at out_scale, zero where invalid
        depth_scaled = np.clip(depth_m * args.out_scale, 0, 65535)
        depth_scaled[~valid] = 0
        out = depth_scaled.astype(np.uint16)

        cv2.imwrite(os.path.join(args.depth_out, f'{frame_num:06d}.png'), out)
        stats['unique_out'].append(len(np.unique(out)))
        stats['valid_pct'].append(float(valid.mean()) * 100)

    print(f'\nDone. Output: {args.depth_out}')
    print(f'Unique values per frame: in median={np.median(stats["unique_in"]):.0f}, '
          f'out median={np.median(stats["unique_out"]):.0f}')
    print(f'Valid-pixel percentage: median={np.median(stats["valid_pct"]):.1f}%, '
          f'min={min(stats["valid_pct"]):.1f}%, max={max(stats["valid_pct"]):.1f}%')


if __name__ == '__main__':
    main()
