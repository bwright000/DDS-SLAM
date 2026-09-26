#!/usr/bin/env python3
"""FoundationStereo (NVlabs, CVPR'25) metric depth for a rectified CRCD snippet.

Input  (written by Addons/preprocess/preprocess_crcd_published.py):
  <staged>/video_frames/NNNNNNl.png, NNNNNNr.png   rectified L/R, 1280x720
  <staged>/rectified_calib.txt                      fx, baseline_m (rectified K)
Output:
  <staged>/<out_subdir>/NNNNNN.png   uint16 = depth_m * depth_scale, rectified-LEFT frame, native res, 0 = invalid
  <staged>/<out_subdir>/depth_stats.json          per-snippet percentiles, valid %, invalid breakdown, timing, VRAM
  <staged>/<out_subdir>/.DONE                      written last, only if every frame was produced

depth = fx * B / disparity (disparity in px at native res; rectified P_right shares cx, so d = xL - xR).
A pixel is INVALID (written 0) if any of:
  disp <= min_disp | non-finite | x - d < 0 (right-view out of frame / occluded left strip)
  | left pixel sampled outside the raw left image by the rectification map
  | its right correspondence sampled outside the raw right image | depth outside [min_depth_m, max_depth_m].
The same corpus feeds every benchmark method; per-method unit factors are applied downstream, never here.
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


def read_calib(path):
    kv = {}
    for line in open(path):
        m = re.match(r"\s*([A-Za-z_]+)\s*[: ]\s*([-+0-9.eE]+)", line)
        if m:
            kv[m.group(1)] = float(m.group(2))
    fx = kv["fx"]
    b = kv.get("baseline_m", kv.get("baseline"))
    if b is None:
        sys.exit(f"FATAL: no baseline in {path}")
    return fx, b


def source_valid_masks(calib_pkl, h, w):
    """True where the rectification map samples INSIDE the raw image (else cv2.remap wrote border black)."""
    with open(calib_pkl, "rb") as f:
        c = pickle.load(f)
    out = []
    for side in ("left", "right"):
        mx, my = c[f"ecm_map_{side}_x"], c[f"ecm_map_{side}_y"]
        if mx.shape[:2] != (h, w):
            sys.exit(f"FATAL: rectification map {mx.shape} != frame {(h, w)}")
        out.append((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1))
    return out


REASONS = ("disp_small_or_nan", "left_border", "right_out_of_frame", "right_border", "depth_range")


def disp_to_depth(d, lval, rval, fxb, min_disp, min_depth_m, max_depth_m):
    """disparity (px, HxW) -> metric depth (m, 0 = invalid) + first-reason invalid pixel counts."""
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    bad_d = ~np.isfinite(d) | (d <= min_disp)
    dsafe = np.where(bad_d, 1.0, d)
    xr = xx - dsafe
    bad_r = xr < 0
    xr_i = np.clip(np.round(xr).astype(np.int64), 0, w - 1)
    bad_rb = ~rval[yy, xr_i]
    dep = fxb / dsafe
    bad_z = (dep < min_depth_m) | (dep > max_depth_m)
    bad = np.zeros_like(bad_d)
    why = {}
    for k, m in zip(REASONS, (bad_d, ~lval, bad_r, bad_rb, bad_z)):  # first-reason attribution
        why[k] = int((m & ~bad).sum())
        bad |= m
    return np.where(bad, 0.0, dep), why


def load_model(fs_root, ckpt, valid_iters, hiera):
    sys.path.insert(0, fs_root)
    from omegaconf import OmegaConf
    from core.foundation_stereo import FoundationStereo
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "cfg.yaml"))
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = valid_iters
    cfg["hiera"] = hiera
    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)  # torch>=2.6 defaults weights_only=True
    model.load_state_dict(sd["model"])
    return model.cuda().eval(), args


@torch.no_grad()
def infer_disp(model, InputPadder, left, right, iters, hiera):
    h, w = left.shape[:2]
    t0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)
    with torch.cuda.amp.autocast(True):
        if hiera:
            d = model.run_hierachical(t0, t1, iters=iters, test_mode=True, small_ratio=0.5)
        else:
            d = model.forward(t0, t1, iters=iters, test_mode=True)
    return padder.unpad(d.float()).cpu().numpy().reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True, help="snippet dir with video_frames/ + rectified_calib.txt")
    ap.add_argument("--calib_pkl", required=True, help="rectification-map pickle (border validity)")
    ap.add_argument("--fs_root", required=True, help="FoundationStereo repo checkout")
    ap.add_argument("--ckpt", required=True, help=".../23-51-11/model_best_bp2.pth")
    ap.add_argument("--out_subdir", default="depth_fs")
    ap.add_argument("--depth_scale", type=float, default=10000.0)
    ap.add_argument("--valid_iters", type=int, default=32)
    ap.add_argument("--hiera", type=int, default=0)
    ap.add_argument("--min_disp", type=float, default=0.5)
    ap.add_argument("--min_depth_m", type=float, default=0.005)
    ap.add_argument("--max_depth_m", type=float, default=0.5)
    ap.add_argument("--max_frames", type=int, default=0, help="0 = all (smoke: small N)")
    args = ap.parse_args()

    vf = os.path.join(args.staged, "video_frames")
    lefts = sorted(glob.glob(os.path.join(vf, "*l.png")))
    pairs = [(l, l[:-5] + "r.png") for l in lefts]
    missing = [r for _, r in pairs if not os.path.exists(r)]
    if not lefts or missing:
        sys.exit(f"FATAL: {len(lefts)} left frames, {len(missing)} missing right frames (e.g. {missing[:1]})")
    if args.max_frames:
        pairs = pairs[: args.max_frames]
    fx, base = read_calib(os.path.join(args.staged, "rectified_calib.txt"))
    fxb = fx * base

    h, w = cv2.imread(pairs[0][0], cv2.IMREAD_UNCHANGED).shape[:2]
    lval, rval = source_valid_masks(args.calib_pkl, h, w)
    out = os.path.join(args.staged, args.out_subdir)
    os.makedirs(out, exist_ok=True)
    done = os.path.join(out, ".DONE")
    if os.path.exists(done):
        os.remove(done)

    model, _ = load_model(args.fs_root, args.ckpt, args.valid_iters, args.hiera)
    from core.utils.utils import InputPadder
    torch.cuda.reset_peak_memory_stats()

    reasons = {k: 0 for k in REASONS}
    sample, n_px, n_valid, n_done, n_inf, t_inf = [], 0, 0, 0, 0, 0.0
    rng = np.random.default_rng(0)
    print(f"[fs] {os.path.basename(args.staged)}: {len(pairs)} pairs {w}x{h}  fx*B={fxb:.4f} px*m  "
          f"iters={args.valid_iters} hiera={args.hiera}", flush=True)
    for i, (lp, rp) in enumerate(pairs):
        stem = os.path.basename(lp)[:-5]
        op = os.path.join(out, f"{stem}.png")
        if os.path.exists(op):
            dep = cv2.imread(op, cv2.IMREAD_UNCHANGED).astype(np.float64) / args.depth_scale
        else:
            L, R = imageio.imread(lp)[..., :3], imageio.imread(rp)[..., :3]
            if L.shape[:2] != (h, w) or R.shape[:2] != (h, w):
                sys.exit(f"FATAL: {stem} size {L.shape}/{R.shape} != {(h, w)}")
            t = time.time()
            d = infer_disp(model, InputPadder, L, R, args.valid_iters, args.hiera)
            torch.cuda.synchronize()
            t_inf += time.time() - t
            n_inf += 1

            dep, why = disp_to_depth(d, lval, rval, fxb, args.min_disp, args.min_depth_m, args.max_depth_m)
            for k in reasons:
                reasons[k] += why[k]
            u16 = np.clip(np.round(dep * args.depth_scale), 0, 65535).astype(np.uint16)
            cv2.imwrite(op + ".tmp.png", u16)
            os.replace(op + ".tmp.png", op)
            dep = u16.astype(np.float64) / args.depth_scale
        v = dep > 0
        n_px += v.size
        n_valid += int(v.sum())
        if v.any():
            vals = dep[v]
            sample.append(rng.choice(vals, size=min(2000, vals.size), replace=False))
        n_done += 1
        if i % 50 == 0 or i == len(pairs) - 1:
            print(f"[fs]  {i + 1}/{len(pairs)}  valid {100 * v.mean():.1f}%  "
                  f"median {np.median(dep[v]) if v.any() else float('nan'):.4f} m  "
                  f"peakVRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)

    allv = np.concatenate(sample) if sample else np.array([np.nan])
    pct = {f"p{p}": float(np.percentile(allv, p)) for p in (0.1, 1, 5, 25, 50, 75, 95, 99, 99.9)}
    stats = {
        "snippet": os.path.basename(os.path.normpath(args.staged)), "frames": n_done, "width": w, "height": h,
        "fx": fx, "baseline_m": base, "depth_scale": args.depth_scale,
        "valid_pct": 100.0 * n_valid / max(n_px, 1), "depth_m_percentiles": pct,
        "invalid_px_by_reason_new_frames": reasons,
        "frames_inferred_this_call": n_inf, "sec_per_frame": (t_inf / n_inf) if n_inf else None,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "params": {k: getattr(args, k) for k in ("valid_iters", "hiera", "min_disp", "min_depth_m", "max_depth_m")},
        "ckpt": args.ckpt,
    }
    json.dump(stats, open(os.path.join(out, "depth_stats.json"), "w"), indent=2)
    if n_done == len(pairs) and len(glob.glob(os.path.join(out, "*.png"))) >= len(pairs):
        open(done, "w").write(f"{n_done}\n")
    print(f"[fs] DONE {stats['snippet']}: {n_done} frames, valid {stats['valid_pct']:.1f}%, "
          f"p1/p50/p99 = {pct['p1']:.4f}/{pct['p50']:.4f}/{pct['p99']:.4f} m, "
          f"peak {stats['peak_vram_gib']:.2f} GiB", flush=True)


if __name__ == "__main__":
    main()
