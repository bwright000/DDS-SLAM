"""
preprocess_crcd_published.py — adapter from CRCD-Published format to DDS-SLAM
StereoMIS-style input.

The older preprocess_crcd.py expects the pre-published CRCD format
(frames_left/*.webp + frames_right/*.webp + SAM3 results JSON). The
CRCD-Published format ships PNG frames + intrinsics.yaml + groundtruth.txt +
per-pixel semantic_instance masks already, so most of the work disappears —
we just need to rectify and rename.

Input (CRCD-Published snippet):
  <snippet>/rgb/frame_NNNNNN.png            (pre-rectification left, PNG)
  <snippet>/rgbright/frame_NNNNNN.png       (pre-rectification right)
  <snippet>/semantic_instance/frame_NNNNNN.png  (uint16, pixel = coco_id+1)
  <snippet>/groundtruth.txt                 (TUM: timestamp tx ty tz qx qy qz qw)
  <snippet>/intrinsics.yaml                 (rectified K + stereo + map pickle path)

Output (DDS-SLAM StereoMIS layout consumed by datasets/dataset.py:120):
  <output_dir>/video_frames/NNNNNNl.png     (rectified left)
  <output_dir>/video_frames/NNNNNNr.png     (rectified right)
  <output_dir>/masks/NNNNNN.png             (instrument-class mask, 1=tool, 0=else)
  <output_dir>/groundtruth.txt              (copied verbatim, TUM format)
  <output_dir>/rectified_calib.txt          (fx, fy, cx, cy, baseline_m written for downstream tools)

Frame indices in output start at 000000 and increment sequentially; original
frame numbers (e.g. 011159..012445 for F_1/002) become 000000..001286.

Usage:
  python Addons/preprocess/preprocess_crcd_published.py \\
      --snippet_dir "F:/Datasets/CRCD-Published/F_1/snippet_002" \\
      --calib_pkl   "C:/Users/benli/sam3facebook/cam_cali/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl" \\
      --output_dir  "F:/Datasets/CRCD-Published-DDSStaged/F1_002"

Tool-class mask: pixel_value=3 per info_semantic.json (Tool) → mask=1, else 0.
"""

import argparse
import os
import pickle
import shutil

import cv2
import numpy as np
import yaml
from tqdm import tqdm


def load_stereo_maps(calib_pkl):
    with open(calib_pkl, "rb") as f:
        calib = pickle.load(f)
    return {
        "map_left_x": calib["ecm_map_left_x"],
        "map_left_y": calib["ecm_map_left_y"],
        "map_right_x": calib["ecm_map_right_x"],
        "map_right_y": calib["ecm_map_right_y"],
    }


def write_rectified_calib(intrinsics_yaml_path, output_dir):
    with open(intrinsics_yaml_path) as f:
        intr = yaml.safe_load(f)
    cam = intr["camera"]
    stereo = intr["stereo"]
    out_path = os.path.join(output_dir, "rectified_calib.txt")
    with open(out_path, "w") as f:
        f.write(f"fx {cam['fx']}\n")
        f.write(f"fy {cam['fy']}\n")
        f.write(f"cx {cam['cx']}\n")
        f.write(f"cy {cam['cy']}\n")
        f.write(f"baseline_m {stereo['baseline_m']}\n")
        f.write(f"baseline_mm {stereo['baseline_mm']}\n")
        f.write(f"width {cam['width']}\n")
        f.write(f"height {cam['height']}\n")
    return cam, stereo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snippet_dir", required=True)
    parser.add_argument("--calib_pkl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tool_pixel_value", type=int, default=3,
                        help="pixel value in semantic_instance for the Tool class (default 3 per info_semantic.json)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "video_frames"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    # 1. Load rectification maps + write rectified_calib.txt
    maps = load_stereo_maps(args.calib_pkl)
    cam, stereo = write_rectified_calib(
        os.path.join(args.snippet_dir, "intrinsics.yaml"), args.output_dir)
    print(f"  rectified K: fx={cam['fx']:.3f} fy={cam['fy']:.3f} cx={cam['cx']:.3f} cy={cam['cy']:.3f}")
    print(f"  baseline: {stereo['baseline_m']*1000:.3f} mm")

    # 2. Enumerate frame pairs
    rgb_dir = os.path.join(args.snippet_dir, "rgb")
    rgbright_dir = os.path.join(args.snippet_dir, "rgbright")
    sem_dir = os.path.join(args.snippet_dir, "semantic_instance")

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    print(f"  found {len(rgb_files)} left frames")

    # 3. Rectify + rename
    for i, fname in enumerate(tqdm(rgb_files, desc="rectifying")):
        # Original frame name like "frame_011159.png"
        l_in = os.path.join(rgb_dir, fname)
        r_in = os.path.join(rgbright_dir, fname)
        if not os.path.isfile(r_in):
            print(f"  WARN: missing right frame for {fname}")
            continue

        l_img = cv2.imread(l_in)
        r_img = cv2.imread(r_in)
        l_rect = cv2.remap(l_img, maps["map_left_x"], maps["map_left_y"], cv2.INTER_LINEAR)
        r_rect = cv2.remap(r_img, maps["map_right_x"], maps["map_right_y"], cv2.INTER_LINEAR)

        out_l = os.path.join(args.output_dir, "video_frames", f"{i:06d}l.png")
        out_r = os.path.join(args.output_dir, "video_frames", f"{i:06d}r.png")
        cv2.imwrite(out_l, l_rect)
        cv2.imwrite(out_r, r_rect)

        # Tool mask — pixel == tool_pixel_value → 255 (used as instrument
        # exclusion in tracking, same as masks/ convention in StereoMIS)
        # NOTE: semantic_instance is left-frame, pre-rectification. Rectify
        # with map_left_x/y using NEAREST interpolation to avoid creating
        # fractional class IDs.
        sem_in = os.path.join(sem_dir, fname)
        if os.path.isfile(sem_in):
            sem = cv2.imread(sem_in, cv2.IMREAD_UNCHANGED)
            sem_rect = cv2.remap(sem, maps["map_left_x"], maps["map_left_y"],
                                 cv2.INTER_NEAREST)
            mask = (sem_rect == args.tool_pixel_value).astype(np.uint8) * 255
            out_m = os.path.join(args.output_dir, "masks", f"{i:06d}.png")
            cv2.imwrite(out_m, mask)

    # 4. Copy groundtruth.txt verbatim
    shutil.copy(
        os.path.join(args.snippet_dir, "groundtruth.txt"),
        os.path.join(args.output_dir, "groundtruth.txt"),
    )

    print(f"\nDone. Output at: {args.output_dir}")
    print(f"  video_frames/: {len(rgb_files)*2} files ({len(rgb_files)} pairs)")
    print(f"  masks/: see masks/ dir")
    print(f"  groundtruth.txt + rectified_calib.txt")


if __name__ == "__main__":
    main()
