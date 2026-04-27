"""
Preprocess CRCD data packs for DDS-SLAM.

Converts CRCD motion segments into the directory structure expected by
StereoMISDataset. Supports single snippet or batch processing of all snippets.

Output structure per snippet:
  {output_root}/{EP}_{SNP}/
    video_frames/   000001l.png, 000001r.png, ...
    masks/          000001.png, 000002.png, ...  (half frame rate)
    groundtruth.txt (TUM format poses)
    rectified_calib.txt

Usage (single):
  python Addons/preprocess_crcd.py \
    --snippet_dir "C:/Thesis/CRCD for DDS/Segments/C_1/snippet_001" \
    --calib_file  "C:/Thesis/CRCD for DDS/cam_calib/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl" \
    --results_json "C:/Thesis/CRCD for DDS/C_1 Results/C_1/snippet_001/snippet_001_results.json" \
    --output_dir  data/CRCD/C1_001

Usage (batch — processes all snippets with SAM3 results):
  python Addons/preprocess_crcd.py --batch \
    --segments_root "C:/Thesis/CRCD for DDS/Segments" \
    --results_root  "C:/Thesis/CRCD for DDS/C_1 Results" \
    --calib_file    "C:/Thesis/CRCD for DDS/cam_calib/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl" \
    --output_root   data/CRCD
"""

import argparse
import glob
import json
import os
import pickle
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def load_stereo_calib(calib_path):
    """Load CRCD stereo calibration from pickle file."""
    with open(calib_path, "rb") as f:
        calib = pickle.load(f)

    K = calib["ecm_left_rect_K"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    T = calib["ecm_T"]
    baseline_mm = abs(T[0, 0])
    baseline_m = baseline_mm / 1000.0

    bf = abs(calib["ecm_right_rect_P"][0, 3])

    return {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "baseline_mm": baseline_mm, "baseline_m": baseline_m, "bf": bf,
        "map_left_x": calib["ecm_map_left_x"],
        "map_left_y": calib["ecm_map_left_y"],
        "map_right_x": calib["ecm_map_right_x"],
        "map_right_y": calib["ecm_map_right_y"],
    }


def find_common_frames(snippet_dir):
    """Find frame numbers that exist in both left and right directories."""
    left_dir = os.path.join(snippet_dir, "frames_left")
    right_dir = os.path.join(snippet_dir, "frames_right")

    left_files = {f: os.path.join(left_dir, f) for f in os.listdir(left_dir)
                  if f.endswith(".webp")}
    right_files = {f: os.path.join(right_dir, f) for f in os.listdir(right_dir)
                   if f.endswith(".webp")}

    common = sorted(set(left_files.keys()) & set(right_files.keys()))
    return [(left_files[f], right_files[f]) for f in common]


def load_poses(snippet_dir):
    """Load poses from poses.txt, return list of TUM-format lines."""
    poses_file = os.path.join(snippet_dir, "poses.txt")
    lines = []
    with open(poses_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            line = line.strip()
            if line:
                lines.append(line)
    return lines


COLOR_MAP = {
    "liver": (0, 255, 0),
    "gallbladder": (255, 0, 0),
    "tool": (0, 0, 255),
}


def _load_legacy_masks(data, frame_pairs):
    """Legacy SAM3 results format: data["frames"][i]["masks"][cat]["segmentation"].

    Returns (H, W, per_frame_polys) where per_frame_polys is a list (length =
    len(frame_pairs)) of {cat_lower: [poly1, poly2, ...]} in pixel-space.
    """
    frames = data["frames"]
    H, W = frames[0]["height"], frames[0]["width"]

    per_frame = []
    n = min(len(frame_pairs), len(frames))
    for i in range(n):
        polys = {}
        for cat in COLOR_MAP:
            cat_data = frames[i]["masks"].get(cat)
            if not isinstance(cat_data, list):
                continue
            cat_polys = []
            for inst in cat_data:
                cat_polys.extend(inst.get("segmentation", []))
            if cat_polys:
                polys[cat] = cat_polys
        per_frame.append(polys)
    # pad if frames < frame_pairs (use last frame)
    while len(per_frame) < len(frame_pairs):
        per_frame.append(per_frame[-1] if per_frame else {})
    return H, W, per_frame


def _load_coco_masks(data, frame_pairs):
    """COCO format: data["images"], data["annotations"], data["categories"].

    image_id is the video-global frame index (matches frame_NNNNNN.webp).
    Multiple annotations per image (one per category). segmentation is
    [[x1,y1,...], ...] (multiple polygons = disconnected components).
    """
    H = data["images"][0]["height"]
    W = data["images"][0]["width"]
    cat_id_to_name = {c["id"]: c["name"].lower() for c in data["categories"]}

    # group annotations by image_id, then category-name
    by_image = {}
    for a in data["annotations"]:
        cat_name = cat_id_to_name.get(a["category_id"], "")
        if cat_name not in COLOR_MAP:
            continue
        seg = a["segmentation"]
        if not isinstance(seg, list) or not seg:
            continue
        by_image.setdefault(a["image_id"], {}).setdefault(cat_name, []).extend(seg)

    # Map snippet frame index → image_id by extracting digits from the left filename
    import re
    per_frame = []
    for left_path, _ in frame_pairs:
        base = os.path.splitext(os.path.basename(left_path))[0]
        digits = re.sub(r"[^0-9]", "", base)
        image_id = int(digits) if digits else -1
        per_frame.append(by_image.get(image_id, {}))
    return H, W, per_frame


def rasterize_masks(annotations_json, output_dir, frame_pairs, calib=None, rectify=True):
    """Rasterize semantic polygons to multi-class PNG masks, one per frame.

    Auto-detects legacy SAM3 results format vs COCO format. Polygons are in
    the ORIGINAL (un-rectified) left-camera frame; we apply the same left-
    camera rectification map as RGB/depth so masks stay geometrically aligned.

    Args:
        annotations_json: path to either snippet_NNN_results.json (legacy) or
            snippet_annotations.json (COCO).
        output_dir: where {output_dir}/masks/NNNNNN.png will be written.
        frame_pairs: list of (left_path, right_path) ordered to match the
            output frame numbering. len(masks) == len(frame_pairs) (1:1).
        calib: dict with map_left_x/y (optional).
        rectify: if True and calib provided, apply remap to the canvas.
    """
    with open(annotations_json) as f:
        data = json.load(f)

    if "annotations" in data and "images" in data and "categories" in data:
        fmt = "coco"
        H, W, per_frame = _load_coco_masks(data, frame_pairs)
    elif "frames" in data:
        fmt = "legacy"
        H, W, per_frame = _load_legacy_masks(data, frame_pairs)
    else:
        raise ValueError(f"Unrecognised annotation format in {annotations_json}; "
                         f"keys={list(data.keys())[:5]}")

    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    do_rectify = rectify and calib is not None and "map_left_x" in calib and "map_left_y" in calib

    # 1:1 mask:frame mapping. dataset.py auto-detects ratio at load time.
    n_masks = len(frame_pairs)
    n_with_any = 0
    for mask_idx in range(n_masks):
        polys = per_frame[mask_idx]
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        if polys:
            n_with_any += 1
        for cat, color in COLOR_MAP.items():
            for polygon in polys.get(cat, []):
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(canvas, [pts], color)

        if do_rectify:
            canvas = cv2.remap(canvas, calib["map_left_x"], calib["map_left_y"],
                               interpolation=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        cv2.imwrite(os.path.join(masks_dir, f"{mask_idx + 1:06d}.png"), canvas)

    print(f"  masks: format={fmt}  total={n_masks}  with-content={n_with_any}  empty={n_masks - n_with_any}")
    return n_masks


def process_snippet(snippet_dir, calib, output_dir, results_json=None, rectify=True):
    """Process a single CRCD snippet into DDS-SLAM format."""
    pairs = find_common_frames(snippet_dir)
    if not pairs:
        print(f"  ERROR: No common frames found")
        return 0

    poses = load_poses(snippet_dir)

    n = min(len(pairs), len(poses))
    if len(pairs) != len(poses):
        print(f"  WARNING: {len(pairs)} frame pairs vs {len(poses)} poses, using {n}")
    pairs = pairs[:n]
    poses = poses[:n]

    # Write rectified stereo frames
    frames_dir = os.path.join(output_dir, "video_frames")
    os.makedirs(frames_dir, exist_ok=True)

    for i, (left_path, right_path) in enumerate(tqdm(pairs, desc="  Rectifying", leave=False)):
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if rectify:
            left_img = cv2.remap(left_img, calib["map_left_x"], calib["map_left_y"],
                                 cv2.INTER_LINEAR)
            right_img = cv2.remap(right_img, calib["map_right_x"], calib["map_right_y"],
                                  cv2.INTER_LINEAR)

        idx = i + 1
        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}l.png"), left_img)
        cv2.imwrite(os.path.join(frames_dir, f"{idx:06d}r.png"), right_img)

    # Write poses
    gt_path = os.path.join(output_dir, "groundtruth.txt")
    with open(gt_path, "w") as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        for line in poses:
            f.write(line + "\n")

    # Write calibration
    calib_path = os.path.join(output_dir, "rectified_calib.txt")
    with open(calib_path, "w") as f:
        f.write(f"fx: {calib['fx']}\n")
        f.write(f"fy: {calib['fy']}\n")
        f.write(f"cx: {calib['cx']}\n")
        f.write(f"cy: {calib['cy']}\n")
        f.write(f"baseline: {calib['baseline_m']}\n")
        f.write(f"bf: {calib['bf']}\n")
        f.write(f"img_size: (1280, 720)\n")

    # Rasterize semantic masks. Auto-prefer COCO-format snippet_annotations.json
    # over legacy SAM3 results JSON if both are present in the snippet dir.
    n_masks = 0
    coco_local = os.path.join(snippet_dir, "snippet_annotations.json")
    if os.path.isfile(coco_local) and not (results_json and os.path.exists(results_json)):
        results_json = coco_local
    if results_json and os.path.exists(results_json):
        n_masks = rasterize_masks(results_json, output_dir, pairs,
                                  calib=calib, rectify=rectify)

    return n


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CRCD data packs for DDS-SLAM")

    # Batch mode
    parser.add_argument("--batch", action="store_true",
                        help="Process all snippets found under segments_root")
    parser.add_argument("--segments_root", type=str,
                        help="Root of CRCD segments (contains C_1/, E_3/, F_3/, ...)")
    parser.add_argument("--results_root", type=str, default=None,
                        help="Optional separate root of legacy SAM3 results "
                             "(snippet_NNN_results.json). If snippet_annotations.json "
                             "is in the snippet dir it takes priority.")
    parser.add_argument("--output_root", type=str,
                        help="Output root directory (e.g., data/CRCD)")
    parser.add_argument("--snippets", type=str, default=None,
                        help="Comma-separated list of EP/snippet selectors, e.g. "
                             "'E_3/snippet_001,F_3/snippet_007'. Default: all snippets "
                             "found under segments_root.")

    # Single mode
    parser.add_argument("--snippet_dir", type=str,
                        help="Path to single CRCD snippet")
    parser.add_argument("--output_dir", type=str,
                        help="Output directory for single snippet")
    parser.add_argument("--results_json", type=str, default=None,
                        help="Path to SAM3 results JSON for single snippet")

    # Common
    parser.add_argument("--calib_file", type=str, required=True,
                        help="Path to stereo calibration pickle file")
    parser.add_argument("--no_rectify", action="store_true",
                        help="Skip rectification")
    args = parser.parse_args()

    # Load calibration once
    calib = load_stereo_calib(args.calib_file)
    print(f"Calibration: fx={calib['fx']:.2f}, fy={calib['fy']:.2f}, "
          f"baseline={calib['baseline_mm']:.4f}mm")

    if args.batch:
        if not all([args.segments_root, args.output_root]):
            parser.error("--batch requires --segments_root and --output_root")

        # Build the list of (ep, snp) pairs to process
        targets = []
        if args.snippets:
            for sel in args.snippets.split(","):
                sel = sel.strip().strip("/")
                if "/" not in sel:
                    print(f"  skip {sel} (expected EP/snippet_NNN)")
                    continue
                ep, snp = sel.split("/", 1)
                targets.append((ep, snp))
        else:
            for ep in sorted(os.listdir(args.segments_root)):
                ep_segments = os.path.join(args.segments_root, ep)
                if not os.path.isdir(ep_segments):
                    continue
                for snp in sorted(os.listdir(ep_segments)):
                    if snp.startswith("snippet_") and "tbd" not in snp.lower():
                        targets.append((ep, snp))

        summary = []
        for ep, snp in targets:
            snippet_dir = os.path.join(args.segments_root, ep, snp)
            if not os.path.isdir(snippet_dir):
                print(f"  skip {ep}/{snp} (missing dir)"); continue

            # Annotation source: snippet_annotations.json (COCO) takes priority
            # over legacy {snp}_results.json from --results_root.
            ann_json = None
            coco_local = os.path.join(snippet_dir, "snippet_annotations.json")
            if os.path.isfile(coco_local):
                ann_json = coco_local
            elif args.results_root:
                legacy = os.path.join(args.results_root, ep, snp, f"{snp}_results.json")
                if os.path.isfile(legacy):
                    ann_json = legacy
            # ann_json may still be None — process_snippet handles that gracefully.

            # e.g., E_3/snippet_001 -> E3_001
            ep_short = ep.replace("_", "")
            snp_num = snp.replace("snippet_", "")
            out_name = f"{ep_short}_{snp_num}"
            output_dir = os.path.join(args.output_root, out_name)

            print(f"\n{'='*60}")
            print(f"Processing {ep}/{snp} -> {out_name}")
            print(f"  annotations: {ann_json or '<none>'}")
            n = process_snippet(snippet_dir, calib, output_dir,
                                ann_json, rectify=not args.no_rectify)
            summary.append((out_name, n))
            print(f"  {n} frames processed")

        # Print summary
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len(summary)} snippets processed")
        print(f"{'='*60}")
        total = 0
        for name, n in summary:
            print(f"  {name}: {n} frames")
            total += n
        print(f"  TOTAL: {total} frames")
        print(f"\nOutput: {args.output_root}/")
        print(f"\nNext: generate depth on Colab with:")
        print(f"  python Addons/generate_depth_stereomis.py --datadir data/CRCD/{{snippet}} "
              f"--method raft_stereo --baseline {calib['baseline_m']:.6f} --fx {calib['fx']:.2f}")

    else:
        if not all([args.snippet_dir, args.output_dir]):
            parser.error("Single mode requires --snippet_dir and --output_dir")

        n = process_snippet(args.snippet_dir, calib, args.output_dir,
                            args.results_json, rectify=not args.no_rectify)

        print(f"\nDone! {n} frames in {args.output_dir}")
        print(f"\nNext steps:")
        print(f"  1. Generate depth (stereo): python Addons/generate_depth_stereomis.py "
              f"--datadir {args.output_dir} --method raft_stereo "
              f"--baseline {calib['baseline_m']:.6f} --fx {calib['fx']:.2f}")
        print(f"  2. Generate depth (mono):   python Addons/generate_depth_stereomis.py "
              f"--datadir {args.output_dir} --method depth_anything")
        print(f"  3. Run DDS-SLAM: python ddsslam.py --config configs/CRCD/c1_001.yaml")


if __name__ == "__main__":
    main()
