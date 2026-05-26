"""generate_depth_moge.py — MoGe-2 depth maps with temporal smoothing + global
scale-match for DDS-SLAM consumption.

Why this exists (vs generate_depth.py's --method moge):
  * Adds a temporal median filter across frames to kill the per-frame jitter
    that previously made foundation-model depth (MoGe / DA V2) trigger
    spurious tracker motion (ATE 60-70 mm on legacy SemSup runs).
  * Replaces the per-frame median scale-match (which is itself a jitter
    source) with ONE GLOBAL scale factor across the whole sequence.
  * Output format matches what DDS-SLAM's SuperDataset expects:
        <out>/<frame>-left_depth.npy   (float32, values = depth_m * depth_scale)

Pipeline:
  1. For each *-left.png in --rgb: run MoGe-2-vitl → raw metric depth (m)
  2. Stack 151 depths into [T, H, W]
  3. Apply temporal median filter with window --temporal_window (default 5)
  4. If --ref provided: compute global scale = median(ref) / median(pred) over
     all valid (depth>0, finite) pixels pooled across the sequence
  5. Save each frame as <fid>-left_depth.npy with values = depth_m * scale * depth_scale

Usage:
  python Addons/depth/generate_depth_moge.py \\
      --rgb /content/semsup_data/trial_3/rgb \\
      --ref /content/semsup_data/trial_3/depth/ref \\
      --out /content/depth_out_moge_smoothed \\
      --temporal_window 5 \\
      --depth_scale 8
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _ref_path_for(ref_dir, fid):
    """Mirror _ref_path_for from generate_depth_for_ddsslam.py — accept both
    raw v2_data02 ('<fid>.npy') and DDS-SLAM-format ('<fid>-left_depth.npy')."""
    for name in (f"{fid}.npy", f"{fid}-left_depth.npy"):
        p = os.path.join(ref_dir, name)
        if os.path.isfile(p):
            return p
    return None


def temporal_median_filter(depths, window):
    """Apply per-pixel median filter across a temporal window.

    Args:
        depths: ndarray [T, H, W] float32, with 0/NaN as invalid
        window: odd int, total window size (e.g. 5 = current frame ± 2)

    Returns:
        ndarray [T, H, W] float32, smoothed.
    """
    if window <= 1:
        return depths
    T = depths.shape[0]
    half = window // 2
    out = np.empty_like(depths)
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        # Mask out invalid (zero) values so they don't bias the median
        window_stack = depths[lo:hi]  # [k, H, W]
        # numpy nan-median treats nan as invalid; convert 0s to nan for masking
        nan_stack = np.where(window_stack > 0, window_stack, np.nan)
        with np.errstate(all='ignore'):
            med = np.nanmedian(nan_stack, axis=0)
        # Pixels that are nan everywhere in the window stay 0
        med = np.where(np.isfinite(med), med, 0.0)
        out[t] = med.astype(np.float32)
    return out


def compute_global_scale(pred_stack, ref_dir, rgb_files):
    """Compute a single scalar = median(ref) / median(pred) over all valid
    pixels pooled across the whole sequence.

    Args:
        pred_stack: [T, H, W] float32 predicted depth (after temporal smoothing)
        ref_dir: dir containing REF .npy files
        rgb_files: list of RGB paths (used to derive fid keys)

    Returns:
        scale: float, or 1.0 if no REF data available
    """
    pred_vals = []
    ref_vals = []
    for t, rgb_path in enumerate(rgb_files):
        fid = os.path.basename(rgb_path).split("-")[0]
        ref_path = _ref_path_for(ref_dir, fid)
        if ref_path is None:
            continue
        ref = np.load(ref_path).astype(np.float32).squeeze()
        if ref.ndim != 2 or ref.shape != pred_stack.shape[1:]:
            # try resize ref to pred shape
            if ref.ndim == 2:
                ref = cv2.resize(ref, (pred_stack.shape[2], pred_stack.shape[1]),
                                 interpolation=cv2.INTER_NEAREST)
            else:
                continue
        pred = pred_stack[t]
        # valid pixels: both >0 and finite
        mask = (ref > 1e-3) & (pred > 1e-3) & np.isfinite(ref) & np.isfinite(pred)
        if mask.sum() < 100:
            continue
        pred_vals.append(pred[mask])
        ref_vals.append(ref[mask])

    if not pred_vals:
        print("  WARN: no REF data found for global scale-match, returning scale=1.0")
        return 1.0

    pred_all = np.concatenate(pred_vals)
    ref_all = np.concatenate(ref_vals)
    scale = float(np.median(ref_all) / np.median(pred_all))
    print(f"  pooled medians: pred={np.median(pred_all):.4f}, ref={np.median(ref_all):.4f}")
    print(f"  GLOBAL scale = ref_median / pred_median = {scale:.4f}")
    return scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", required=True, help="dir with *-left.png frames")
    ap.add_argument("--ref", default=None,
                    help="optional REF depth dir for GLOBAL scale-match; if "
                         "omitted, MoGe's native metric depth is used as-is "
                         "(values = depth_m * depth_scale)")
    ap.add_argument("--out", required=True, help="output dir for .npy files")
    ap.add_argument("--temporal_window", type=int, default=5,
                    help="temporal median filter window (odd; 1 = disable)")
    ap.add_argument("--depth_scale", type=float, default=8.0,
                    help="DDS-SLAM's png_depth_scale convention "
                         "(values stored = depth_meters * depth_scale)")
    ap.add_argument("--model_id", default="Ruicheng/moge-2-vitl",
                    help="HuggingFace model ID for MoGe-2")
    ap.add_argument("--resolution_level", type=int, default=9,
                    help="MoGe inference resolution level (higher = sharper, slower)")
    ap.add_argument("--max_depth_m", type=float, default=5.0,
                    help="clamp predicted depth to [0, max_depth_m] meters before scaling")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(args.rgb, "*-left.png")))
    if not rgb_files:
        raise SystemExit(f"ERROR: no *-left.png files in {args.rgb}")
    print(f"Found {len(rgb_files)} RGB frames")
    print(f"Output dir: {args.out}")
    print(f"Temporal window: {args.temporal_window}")
    print(f"Depth scale: {args.depth_scale}")
    print(f"Model: {args.model_id}")

    # -------- Load MoGe-2 --------
    from moge.model.v2 import MoGeModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading MoGe-2 ({args.model_id}) on {device}...")
    model = MoGeModel.from_pretrained(args.model_id).to(device).eval()

    # -------- Stage 1: per-frame inference --------
    print(f"\n[1/3] Running MoGe-2 inference on {len(rgb_files)} frames...")
    H_ref, W_ref = None, None
    depths = []
    with torch.no_grad():
        for rgb_path in tqdm(rgb_files, desc="MoGe-2"):
            img = cv2.imread(rgb_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if H_ref is None:
                H_ref, W_ref = img.shape[:2]
            t = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            out = model.infer(t, resolution_level=args.resolution_level)
            d = out["depth"].cpu().numpy().astype(np.float32)
            d = np.where(np.isfinite(d), d, 0.0)
            d = np.clip(d, 0.0, args.max_depth_m)
            # ensure HxW shape (model may return at a different resolution)
            if d.shape != (H_ref, W_ref):
                d = cv2.resize(d, (W_ref, H_ref), interpolation=cv2.INTER_LINEAR)
            depths.append(d)
    depths = np.stack(depths, axis=0)  # [T, H, W]
    print(f"  Raw depth stack: shape={depths.shape}, range=[{depths[depths>0].min():.4f}, {depths.max():.4f}]m")

    # -------- Stage 2: temporal median smoothing --------
    print(f"\n[2/3] Temporal median filter (window={args.temporal_window})...")
    if args.temporal_window > 1:
        depths_sm = temporal_median_filter(depths, args.temporal_window)
        valid = depths_sm > 0
        if valid.any():
            print(f"  Smoothed range: [{depths_sm[valid].min():.4f}, {depths_sm.max():.4f}]m")
    else:
        depths_sm = depths
        print("  (window=1, skipped)")

    # -------- Stage 3: global scale-match + save --------
    print(f"\n[3/3] Global scale-match + save...")
    scale = 1.0
    if args.ref is not None and os.path.isdir(args.ref):
        scale = compute_global_scale(depths_sm, args.ref, rgb_files)
    else:
        print("  No --ref provided; using MoGe's native metric depth (no scale-match)")

    n_written = 0
    for t, rgb_path in enumerate(tqdm(rgb_files, desc="save")):
        fid = os.path.basename(rgb_path).split("-")[0]
        d = depths_sm[t] * scale * args.depth_scale
        out_path = os.path.join(args.out, f"{fid}-left_depth.npy")
        np.save(out_path, d.astype(np.float32))
        n_written += 1

    print(f"\nWrote {n_written}/{len(rgb_files)} depth maps to {args.out}")
    valid = depths_sm > 0
    if valid.any():
        meters = depths_sm[valid]
        print(f"Final stats (post-scale, pre-storage-multiplier):")
        print(f"  meters range: [{meters.min():.4f}, {meters.max():.4f}] m")
        print(f"  meters median: {np.median(meters):.4f} m")
        print(f"  on-disk (values = m * scale * depth_scale = m * {scale * args.depth_scale:.3f}):")
        print(f"    range: [{meters.min() * scale * args.depth_scale:.3f}, "
              f"{meters.max() * scale * args.depth_scale:.3f}]")


if __name__ == "__main__":
    main()
