"""
Evaluate rendering quality of DDS-SLAM outputs against ground truth.
Computes PSNR, SSIM, and LPIPS — the metrics used in the DDS-SLAM paper (Table I).

Usage:
  python Addons/eval_rendering.py --gt_dir data/Super/rgb --render_dir output/trail3 --name "Depth Anything"
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np


def compute_psnr(img1, img2):
    """Compute PSNR between two images (float32, 0-1 range)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_ssim(img1, img2):
    """Compute SSIM between two images (float32, 0-1 range)."""
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


def main():
    parser = argparse.ArgumentParser(description='Evaluate rendering quality')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth RGB directory (e.g., data/Super/rgb)')
    parser.add_argument('--render_dir', type=str, required=True,
                        help='Rendered output directory (e.g., output/trail3_depth_anything)')
    parser.add_argument('--name', type=str, default='', help='Method name for display')
    parser.add_argument('--output_csv', type=str, default='',
                        help='Save per-frame metrics to CSV for use with visualize_run.py')
    args = parser.parse_args()

    # Find rendered images
    rendered = sorted(glob.glob(os.path.join(args.render_dir, '*.jpg')))
    if not rendered:
        print(f"No rendered images found in {args.render_dir}")
        return

    # Find ground truth images
    gt_images = sorted(glob.glob(os.path.join(args.gt_dir, '*-left.png')))
    if not gt_images:
        gt_images = sorted(glob.glob(os.path.join(args.gt_dir, '*_left.png')))
    if not gt_images:
        print(f"No ground truth images found in {args.gt_dir}")
        return

    print(f"Method: {args.name}")
    print(f"Rendered: {len(rendered)} images")
    print(f"Ground truth: {len(gt_images)} images")

    # Try to import LPIPS
    try:
        import torch
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')
        if torch.cuda.is_available():
            lpips_fn = lpips_fn.cuda()
        has_lpips = True
        print("LPIPS: available")
    except ImportError:
        has_lpips = False
        print("LPIPS: not available (pip install lpips)")

    psnr_list = []
    ssim_list = []
    lpips_list = []

    n_eval = min(len(rendered), len(gt_images))
    for i in range(n_eval):
        # Load rendered image
        render = cv2.imread(rendered[i])
        render = cv2.cvtColor(render, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Load ground truth
        gt = cv2.imread(gt_images[i])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Resize if needed
        if render.shape != gt.shape:
            render = cv2.resize(render, (gt.shape[1], gt.shape[0]))

        # PSNR
        psnr = compute_psnr(render, gt)
        psnr_list.append(psnr)

        # SSIM
        ssim = compute_ssim(render, gt)
        ssim_list.append(ssim)

        # LPIPS
        if has_lpips:
            import torch
            render_t = torch.from_numpy(render).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0) * 2 - 1
            if torch.cuda.is_available():
                render_t = render_t.cuda()
                gt_t = gt_t.cuda()
            lpips_val = lpips_fn(render_t, gt_t).item()
            lpips_list.append(lpips_val)

    print(f"\n{'='*50}")
    print(f"Results: {args.name} ({n_eval} frames)")
    print(f"{'='*50}")
    print(f"PSNR:  {np.mean(psnr_list):.3f} (std: {np.std(psnr_list):.3f})")
    print(f"SSIM:  {np.mean(ssim_list):.3f} (std: {np.std(ssim_list):.3f})")
    if lpips_list:
        print(f"LPIPS: {np.mean(lpips_list):.3f} (std: {np.std(lpips_list):.3f})")
    print(f"{'='*50}")

    # Save per-frame metrics to CSV
    if args.output_csv:
        with open(args.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['frame', 'psnr', 'ssim']
            if lpips_list:
                header.append('lpips')
            writer.writerow(header)
            for i in range(n_eval):
                row = [i, f"{psnr_list[i]:.4f}", f"{ssim_list[i]:.4f}"]
                if lpips_list:
                    row.append(f"{lpips_list[i]:.4f}")
                writer.writerow(row)
        print(f"\nPer-frame metrics saved to {args.output_csv}")

    # Paper reference (DDS-SLAM Table I, Lab1 = trail3)
    print(f"\nPaper reference (DDS-SLAM, Lab1):")
    print(f"  PSNR:  28.649")
    print(f"  SSIM:  0.797")
    print(f"  LPIPS: 0.231")


if __name__ == '__main__':
    main()
