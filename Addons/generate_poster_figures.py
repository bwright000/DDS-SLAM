"""
Export individual frames and metrics for poster compilation.

Outputs to Output/For Poster/:
  - GT and rendered frames at t=0, 75, 150 as individual PNGs
  - metrics.json with all comparison data

Usage:
  python Addons/generate_poster_figures.py
  python Addons/generate_poster_figures.py --output_dir "Output/For Poster"
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_DIR = os.path.join(BASE, 'data', 'trial_3', 'rgb')

METHODS = [
    {
        'name': 'Depth Anything V2',
        'key': 'depth_anything_v2',
        'render_dir': os.path.join(BASE, 'Output', 'DDS-SLAM-Results', 'DDS-SLAM-Results', 'trail3_depth_anything'),
        'psnr': 27.618, 'ssim': 0.755, 'lpips': 0.372,
    },
    {
        'name': 'MoGe (CVPR\'25)',
        'key': 'moge',
        'render_dir': os.path.join(BASE, 'Output', 'DDS-SLAM-Results', 'DDS-SLAM-Results', 'trail3_moge', 'trail3'),
        'psnr': 26.980, 'ssim': 0.746, 'lpips': 0.400,
    },
    {
        'name': 'Monodepth2',
        'key': 'monodepth2',
        'render_dir': os.path.join(BASE, 'Output', 'DDS-SLAM-Results', 'DDS-SLAM-Results', 'trail3_monodepth2'),
        'psnr': 26.894, 'ssim': 0.730, 'lpips': 0.404,
    },
]

PAPER = {'name': 'DDS-SLAM (Paper)', 'psnr': 28.649, 'ssim': 0.797, 'lpips': 0.231}

FRAME_INDICES = [0, 75, 150]


def find_gt_image(frame_idx):
    for pattern in [f'{frame_idx:06d}-left.png', f'{frame_idx:06d}_left.png']:
        p = os.path.join(GT_DIR, pattern)
        if os.path.exists(p):
            return p
    return None


def find_rendered_image(render_dir, frame_idx):
    p = os.path.join(render_dir, f'{frame_idx:04d}.jpg')
    return p if os.path.exists(p) else None


def compute_psnr(gt_path, rend_path):
    gt = cv2.imread(gt_path).astype(np.float32) / 255.0
    rend = cv2.imread(rend_path).astype(np.float32) / 255.0
    rend = cv2.resize(rend, (gt.shape[1], gt.shape[0]))
    mse = np.mean((gt - rend) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(1.0 / np.sqrt(mse)))


def main():
    parser = argparse.ArgumentParser(description='Export poster materials')
    parser.add_argument('--output_dir', type=str, default='Output/For Poster')
    args = parser.parse_args()

    output_dir = os.path.join(BASE, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # --- Export individual frames ---
    for f in FRAME_INDICES:
        # Ground truth
        gt_path = find_gt_image(f)
        if gt_path:
            dst = os.path.join(output_dir, f'gt_frame_{f:03d}.png')
            shutil.copy2(gt_path, dst)
            print(f"  {dst}")

        # Each method
        for m in METHODS:
            rend_path = find_rendered_image(m['render_dir'], f)
            if rend_path:
                # Convert JPG to lossless PNG at GT resolution
                gt_img = cv2.imread(gt_path) if gt_path else None
                rend_img = cv2.imread(rend_path)
                if gt_img is not None and rend_img is not None:
                    rend_img = cv2.resize(rend_img, (gt_img.shape[1], gt_img.shape[0]),
                                          interpolation=cv2.INTER_LANCZOS4)
                dst = os.path.join(output_dir, f'{m["key"]}_frame_{f:03d}.png')
                cv2.imwrite(dst, rend_img)
                print(f"  {dst}")

    # --- Build metrics JSON ---
    per_frame = {}
    for m in METHODS:
        per_frame[m['key']] = {}
        for f in FRAME_INDICES:
            gt_path = find_gt_image(f)
            rend_path = find_rendered_image(m['render_dir'], f)
            if gt_path and rend_path:
                per_frame[m['key']][str(f)] = round(compute_psnr(gt_path, rend_path), 2)

    metrics = {
        'dataset': 'Semantic-SuPer Lab1 (trail3)',
        'num_frames': 151,
        'paper': {
            'name': PAPER['name'],
            'psnr': PAPER['psnr'],
            'ssim': PAPER['ssim'],
            'lpips': PAPER['lpips'],
        },
        'methods': [
            {
                'name': m['name'],
                'key': m['key'],
                'psnr': m['psnr'],
                'ssim': m['ssim'],
                'lpips': m['lpips'],
                'per_frame_psnr': per_frame[m['key']],
            }
            for m in METHODS
        ],
        'frame_indices': FRAME_INDICES,
    }

    json_path = os.path.join(output_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  {json_path}")

    print(f"\nDone. {len(FRAME_INDICES) * (1 + len(METHODS))} frames + metrics.json in {output_dir}")


if __name__ == '__main__':
    main()
