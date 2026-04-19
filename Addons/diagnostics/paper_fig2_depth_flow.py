#!/usr/bin/env python
"""
paper_fig2_depth_flow.py — Render paper-Fig-2-style panels (RGB | D_t | F_t)
without needing lietorch. Covers the 3 panels that actually validate depth:
per-frame RGB, RAFT-derived depth, and RAFT stereo flow.

Skips ω_2D and ω_3D panels (require full pose-pipeline state).

Usage:
    python paper_fig2_depth_flow.py \\
        /content/p2_1_depthgen \\
        --rpe_root /content/robust-pose-estimator \\
        --outpath /content/rpe_vis_fig2 \\
        --step 800
"""
import sys, os, argparse, yaml
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def flow_to_rgb(flow):
    """HSV-style flow visualization matching RAFT's flow_viz convention."""
    H, W = flow.shape[-2:]
    fx, fy = flow[0], flow[1]
    mag = np.sqrt(fx ** 2 + fy ** 2)
    ang = np.arctan2(fy, fx)
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * (255.0 / max(mag.max(), 1e-6)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Sequence dir (video_frames/ + StereoCalibration.ini)")
    ap.add_argument("--rpe_root", default=os.environ.get("RPE_ROOT", "/content/robust-pose-estimator"))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--outpath", default="/content/rpe_vis_fig2")
    ap.add_argument("--step", type=int, default=800, help="Visualize every Nth frame")
    args = ap.parse_args()

    rpe = os.path.abspath(args.rpe_root)
    if rpe not in sys.path:
        sys.path.insert(0, rpe)

    from dataset.dataset_utils import get_data, StereoVideoDataset
    from core.pose.pose_net import PoseNet

    ckpt = args.checkpoint or os.path.join(rpe, "trained", "poseNet_2xf8up4b.pth")
    cfg_path = args.config or os.path.join(rpe, "configuration", "infer_f2f.yaml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outpath, exist_ok=True)

    dataset, calib = get_data(args.input, config["img_size"], rect_mode=config["rect_mode"])
    print(f"Dataset: {len(dataset)} frames")

    checkp = torch.load(ckpt, map_location="cpu")
    checkp["config"]["model"]["image_shape"] = (config["img_size"][1], config["img_size"][0])
    checkp["config"]["model"]["lbgfs_iters"] = config["slam"]["lbgfs_iters"]
    checkp["config"]["model"]["use_weights"] = config["slam"]["conf_weighing"]
    model = PoseNet(checkp["config"]["model"]).to(device)
    sd = OrderedDict((k.replace("module.", ""), v) for k, v in checkp["state_dict"].items())
    model.load_state_dict(sd)
    model.eval()

    depth_max_mm = config["slam"]["depth_clipping"][1]
    scale = 1.0 / depth_max_mm
    baseline_scaled = torch.tensor(calib["bf"]).unsqueeze(0).float().to(device) * scale

    loader = DataLoader(dataset, num_workers=0, pin_memory=True)
    n_saved = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i % args.step != 0:
                continue
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, mask, _pose, img_number = data
            else:
                limg, rimg, mask, img_number = data

            limg_d = limg.to(device)
            rimg_d = rimg.to(device)

            # Stereo flow (panel F_t)
            stereo_flow = model.flow(limg_d, rimg_d, upsample=True)[0][-1]
            flow_np = stereo_flow.squeeze(0).cpu().numpy()
            flow_rgb = flow_to_rgb(flow_np)

            # Depth (panel D_t)
            depth_norm = (baseline_scaled[:, None, None] / -stereo_flow[:, 0]).clamp(0, 1)
            depth_m = depth_norm.squeeze().cpu().numpy() * (depth_max_mm / 1000.0)

            # RGB left (panel I_t)
            img_np = limg.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            idx = int(img_number.item()) if torch.is_tensor(img_number) else int(img_number[0])

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].imshow(img_np)
            ax[0].set_title(f"I_t (frame {idx:06d})")
            ax[0].axis("off")

            im1 = ax[1].imshow(depth_m, cmap="viridis", vmin=0, vmax=depth_max_mm / 1000.0)
            ax[1].set_title("D_t (m)")
            ax[1].axis("off")
            plt.colorbar(im1, ax=ax[1], fraction=0.046)

            ax[2].imshow(flow_rgb)
            ax[2].set_title("F_t (stereo flow)")
            ax[2].axis("off")

            plt.tight_layout()
            out = os.path.join(args.outpath, f"fig2_{idx:06d}.png")
            plt.savefig(out, dpi=120)
            plt.close()
            n_saved += 1
            print(f"  saved {out}")

    print(f"\nDone. {n_saved} paper-style panels at {args.outpath}")


if __name__ == "__main__":
    main()
