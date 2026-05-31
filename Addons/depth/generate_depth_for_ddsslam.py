"""generate_depth_for_ddsslam.py — produce DDS-SLAM-Super-compatible depth maps
from our trained Monodepth2 variants, scale-matched to the v2_data02 REF .npy.

Why scale-match: mono-mode Monodepth2 is scale-ambiguous; even stereo-mode is in
arbitrary "baseline units" until calibrated. REF (the SemSup-distributed depth
.npy) is the only "in the right units for DDS-SLAM" reference we have on disk.
Per-frame median-scale-match converts our prediction into REF's units, which
DDS-SLAM then divides by png_depth_scale=8 to get metric meters.

Output format matches DDS-SLAM's SuperDataset expectation:
    <out_dir>/<frame>-left_depth.npy   (float32, depth*8 in REF's convention)

Usage examples
--------------
Variant A Mono:
    python generate_depth_for_ddsslam.py \\
        --variant variant_a_mono \\
        --weights F:/Datasets/mono2-models/merged/semsup_variant_a_paper-faithful/weights_19 \\
        --rgb     F:/Datasets/SemSup/v2_data02/v2_data/trial_3/rgb \\
        --ref     F:/Datasets/SemSup/v2_data02/v2_data/trial_3/depth \\
        --out     C:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/data/Super/trail_3/depth_variant_a_mono

REF (no inference — just renaming):
    python generate_depth_for_ddsslam.py \\
        --variant ref \\
        --ref     F:/Datasets/SemSup/v2_data02/v2_data/trial_3/depth \\
        --rgb     F:/Datasets/SemSup/v2_data02/v2_data/trial_3/rgb \\
        --out     C:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/data/Super/trail_3/depth_ref
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


MIN_DEPTH = 0.1
MAX_DEPTH = 100.0


def disp_to_depth(disp, min_d, max_d):
    min_disp = 1 / max_d
    max_disp = 1 / min_d
    scaled = min_disp + (max_disp - min_disp) * disp
    return 1 / scaled


def flip_fusion(d_orig, d_flipped):
    _, w = d_orig.shape
    m = 0.5 * (d_orig + d_flipped)
    x = np.linspace(0, 1, w)[None, :]
    l = np.clip(20 * (x - 0.05), 0, 1).repeat(d_orig.shape[0], axis=0)
    r = l[:, ::-1]
    return r * d_orig + l * d_flipped + (1.0 - l - r) * m


def median_scale_match(pred, ref, min_d=0.01, max_d=10.0):
    mask = (ref > min_d) & (ref < max_d) & np.isfinite(pred) & np.isfinite(ref)
    if mask.sum() < 100:
        return pred, 1.0
    return pred * (np.median(ref[mask]) / np.median(pred[mask])), float(
        np.median(ref[mask]) / np.median(pred[mask])
    )


def load_md2(weights_dir, networks_module):
    encoder = networks_module.ResnetEncoder(num_layers=18, pretrained=False)
    depth_decoder = networks_module.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4)
    )
    # torch.load's `weights_only` kwarg landed in torch 1.13. dds_env (paper-faithful
    # build) is on torch 1.10 → passing it unconditionally is a TypeError. Pass it
    # only when supported so the same script works on both stacks.
    load_kwargs = {"map_location": "cpu"}
    try:
        from packaging.version import parse as _parse_ver
        if _parse_ver(torch.__version__.split("+")[0]) >= _parse_ver("1.13"):
            load_kwargs["weights_only"] = False
    except Exception:
        pass
    enc_sd = torch.load(os.path.join(weights_dir, "encoder.pth"), **load_kwargs)
    encoder.load_state_dict(
        {k: v for k, v in enc_sd.items() if k in encoder.state_dict().keys()},
        strict=False,
    )
    depth_decoder.load_state_dict(
        torch.load(os.path.join(weights_dir, "depth.pth"), **load_kwargs),
        strict=False,
    )
    return encoder, depth_decoder


def predict(rgb_path, encoder, depth_decoder, device, h, w):
    img = Image.open(rgb_path).convert("RGB").resize((w, h), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0

    def _fwd(np_img):
        t = torch.from_numpy(np_img.transpose(2, 0, 1))[None].to(device)
        with torch.no_grad():
            disp = depth_decoder(encoder(t))[("disp", 0)]
            return disp_to_depth(disp, MIN_DEPTH, MAX_DEPTH).squeeze().cpu().numpy()

    return flip_fusion(_fwd(arr), _fwd(arr[:, ::-1, :].copy())[:, ::-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    help="logical name (variant_a_mono, variant_a_stereo, "
                         "afsfm, variant_b_afsfm, ref). 'ref' skips inference "
                         "and just renames the v2_data02 .npys into DDS-SLAM format.")
    ap.add_argument("--weights", default=None,
                    help="weights_19 dir (required unless --variant ref)")
    ap.add_argument("--rgb", required=True,
                    help="directory with *-left.png frames")
    ap.add_argument("--ref", required=True,
                    help="directory with REF .npy files (used for scale matching, "
                         "or as the source when --variant ref)")
    ap.add_argument("--out", required=True,
                    help="output directory (will be created)")
    ap.add_argument("--networks", default=None,
                    help="path to a Monodepth2 networks/ module (defaults to a "
                         "sparse-cloned copy if present in cwd)")
    ap.add_argument("--inference_height", type=int, default=192,
                    help="network input height in pixels. Set to match the "
                         "model's training resolution: 192 for variant_a/c, "
                         "256 for variant_b/b_afsfm default, 240 for aspect-correct "
                         "variant_b_h240. Default 192 preserves legacy behaviour.")
    ap.add_argument("--inference_width", type=int, default=320,
                    help="network input width in pixels. Default 320 — typical for "
                         "all current SemSup variants.")
    args = ap.parse_args()
    H, W = args.inference_height, args.inference_width
    print(f"Inference resolution: {H}×{W} (aspect W/H={W/H:.3f}, "
          f"native 640×480 = 1.333; mismatch causes train-time stretch)")

    os.makedirs(args.out, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(args.rgb, "*-left.png")))
    if not rgb_files:
        print(f"ERROR: no *-left.png files in {args.rgb}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(rgb_files)} frames")

    def _ref_path_for(fid):
        """Return existing REF .npy path under args.ref, trying both naming
        conventions in order: '<fid>.npy' (raw v2_data02 distribution) then
        '<fid>-left_depth.npy' (DDS-SLAM-format dir, e.g. depth/ref/)."""
        for name in (f"{fid}.npy", f"{fid}-left_depth.npy"):
            p = os.path.join(args.ref, name)
            if os.path.isfile(p):
                return p
        return None

    # ---------- REF variant: copy/rename, no inference ----------
    if args.variant == "ref":
        n_copied = 0
        for rgb_path in tqdm(rgb_files, desc="copying REF"):
            fid = os.path.basename(rgb_path).split("-")[0]
            src = _ref_path_for(fid)
            if src is None:
                print(f"  skip frame {fid}: no REF .npy in {args.ref}")
                continue
            dst = os.path.join(args.out, f"{fid}-left_depth.npy")
            ref = np.load(src).astype(np.float32).squeeze()
            np.save(dst, ref)
            n_copied += 1
        print(f"Wrote {n_copied}/{len(rgb_files)} REF depth files to {args.out}")
        return

    # ---------- Trained-variant: load model and infer ----------
    if not args.weights:
        print("ERROR: --weights required unless --variant ref", file=sys.stderr)
        sys.exit(1)

    # Resolve networks module path
    if args.networks is None:
        # Try a couple of common locations
        for cand in [
            os.path.join(os.getcwd(), "monodepth2"),
            os.path.join(os.path.dirname(__file__), "monodepth2"),
            r"F:/Datasets/mono2-models/sanity_check/monodepth2",
        ]:
            if os.path.isdir(os.path.join(cand, "networks")):
                args.networks = cand
                break
    if args.networks is None or not os.path.isdir(os.path.join(args.networks, "networks")):
        print("ERROR: could not locate Monodepth2 networks/ module. Pass --networks.",
              file=sys.stderr)
        sys.exit(1)
    sys.path.insert(0, args.networks)
    import networks as md2_networks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading weights from: {args.weights}")
    encoder, depth_decoder = load_md2(args.weights, md2_networks)
    encoder.to(device).eval()
    depth_decoder.to(device).eval()

    scales = []
    n_written = 0
    for rgb_path in tqdm(rgb_files, desc=args.variant):
        fid = os.path.basename(rgb_path).split("-")[0]
        ref_path = _ref_path_for(fid)
        if ref_path is None:
            print(f"  skip frame {fid}: no REF for scale match in {args.ref}")
            continue
        ref = np.load(ref_path).astype(np.float32).squeeze()

        pred = predict(rgb_path, encoder, depth_decoder, device, H, W)
        pred_full = np.array(
            Image.fromarray(pred).resize((ref.shape[1], ref.shape[0]), Image.LANCZOS)
        )
        pred_scaled, scale = median_scale_match(pred_full, ref)
        scales.append(scale)
        np.save(os.path.join(args.out, f"{fid}-left_depth.npy"),
                pred_scaled.astype(np.float32))
        n_written += 1

    print(f"\nWrote {n_written}/{len(rgb_files)} depth maps to {args.out}")
    if scales:
        print(f"Per-frame scale: min={min(scales):.3f}  "
              f"median={np.median(scales):.3f}  max={max(scales):.3f}  "
              f"std={np.std(scales):.3f}")


if __name__ == "__main__":
    main()
