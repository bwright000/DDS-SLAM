#!/usr/bin/env python
"""
generate_depth_stereomis.py — Generate StereoMIS depth maps using
robust-pose-estimator's pretrained PoseNet (poseNet_2xf8up4b.pth).

Mirrors scripts/infer_trajectory.py's data + model setup as faithfully as
possible, then extracts depth via PoseNet.flow2depth() directly. This is
the same depth computation infer_trajectory.py performs internally per
frame — we just export it instead of letting the pose head consume it.

The depth values are bit-identical to what PoseEstimator computes in
frame2frame mode; we skip the pose-estimation work (unnecessary for
depth export) but reuse their config, weights, rectification, and baseline
scaling verbatim.

Usage (on Colab, inside rpe_env):
    cd /content/robust-pose-estimator/scripts
    python /content/DDS-SLAM/generate_depth_stereomis.py \\
        /content/p2_1_local \\
        --rpe_root /content/robust-pose-estimator \\
        --outdir /content/p2_1_local/depth \\
        --png_depth_scale 100

Inputs at INPUT_PATH:
    video_frames/NNNNNNl.png, NNNNNNr.png    (or a video file)
    StereoCalibration.ini

Outputs:
    OUTDIR/NNNNNN.png     (uint16, depth_meters * png_depth_scale,
                           invalid pixels = 0)
"""

import sys
import os
import argparse
import yaml
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_parser():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "input",
        help="StereoMIS sequence dir (needs video_frames/ + StereoCalibration.ini)",
    )
    ap.add_argument(
        "--rpe_root",
        default=os.environ.get("RPE_ROOT", "/content/robust-pose-estimator"),
        help="Path to robust-pose-estimator checkout (default: /content/robust-pose-estimator)",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="PoseNet .pth (default: <rpe_root>/trained/poseNet_2xf8up4b.pth)",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Inference config YAML (default: <rpe_root>/configuration/infer_f2f.yaml)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Depth PNG output dir (default: INPUT/depth)",
    )
    ap.add_argument(
        "--png_depth_scale",
        type=int,
        default=100,
        help="uint16_value = depth_meters * png_depth_scale. DDS-SLAM default: 100.",
    )
    return ap


def main():
    args = build_parser().parse_args()

    rpe_root = os.path.abspath(args.rpe_root)
    if rpe_root not in sys.path:
        sys.path.insert(0, rpe_root)

    from dataset.dataset_utils import get_data, StereoVideoDataset
    from core.pose.pose_net import PoseNet

    ckpt_path = args.checkpoint or os.path.join(rpe_root, "trained", "poseNet_2xf8up4b.pth")
    cfg_path = args.config or os.path.join(rpe_root, "configuration", "infer_f2f.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Config:     {cfg_path}")

    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # --- Data loading: identical to infer_trajectory.py:44 ---
    dataset, calib = get_data(args.input, config["img_size"], rect_mode=config["rect_mode"])
    print(f"Dataset length: {len(dataset)}")
    print(f"Image size (W, H): {tuple(config['img_size'])}")
    print(f"Rectification mode: {config['rect_mode']}")
    print(f"Baseline (bf, px·mm): {calib['bf']}")

    # --- Model setup: identical to pose_estimator.py:26-37 ---
    checkp = torch.load(ckpt_path, map_location="cpu")
    checkp["config"]["model"]["image_shape"] = (config["img_size"][1], config["img_size"][0])
    checkp["config"]["model"]["lbgfs_iters"] = config["slam"]["lbgfs_iters"]
    checkp["config"]["model"]["use_weights"] = config["slam"]["conf_weighing"]
    model = PoseNet(checkp["config"]["model"]).to(device)
    new_state_dict = OrderedDict()
    for k, v in checkp["state_dict"].items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # --- Baseline scaling: identical to pose_estimator.py:41 ---
    depth_max_mm = config["slam"]["depth_clipping"][1]  # 250
    scale = 1.0 / depth_max_mm  # normalizes depth to [0, 1]
    baseline_scaled = torch.tensor(calib["bf"]).unsqueeze(0).float().to(device) * scale
    print(f"depth_clipping: [0, {depth_max_mm}] mm (scale = {scale})")

    # --- Output ---
    outdir = args.outdir or os.path.join(args.input, "depth")
    os.makedirs(outdir, exist_ok=True)
    print(f"Output dir: {outdir}")
    print(f"Encoding: uint16 = round(depth_meters * {args.png_depth_scale})")
    print(f"Max uint16 value at depth_max: ~{int(round((depth_max_mm / 1000.0) * args.png_depth_scale))}")

    loader = DataLoader(dataset, num_workers=1, pin_memory=True)
    n_written = 0
    with torch.no_grad():
        for data in tqdm(loader, total=len(dataset)):
            if isinstance(dataset, StereoVideoDataset):
                limg, rimg, mask, _pose_kin, img_number = data
            else:
                limg, rimg, mask, img_number = data

            # --- Depth compute: identical formula to pose_net.py:127-135 ---
            depth_norm, _flow, valid = model.flow2depth(
                limg.to(device), rimg.to(device), baseline_scaled
            )
            # depth_norm: (1,1,H,W) in [0,1]; invalid pixels clamped to 1.0
            # valid:     (1,1,H,W) bool

            depth_m = depth_norm.squeeze().cpu().numpy() * depth_max_mm / 1000.0
            valid_np = valid.squeeze().cpu().numpy().astype(bool)

            # Encode; invalid → 0 (so DDS-SLAM's `depth > 0` checks skip them)
            depth_uint16 = np.round(depth_m * args.png_depth_scale).astype(np.uint32)
            depth_uint16 = np.clip(depth_uint16, 0, 65535).astype(np.uint16)
            depth_uint16[~valid_np] = 0

            idx = int(img_number.item()) if torch.is_tensor(img_number) else int(img_number[0])
            cv2.imwrite(os.path.join(outdir, f"{idx:06d}.png"), depth_uint16)
            n_written += 1

    print(f"\nDone. Wrote {n_written} depth PNGs to {outdir}")


if __name__ == "__main__":
    main()
