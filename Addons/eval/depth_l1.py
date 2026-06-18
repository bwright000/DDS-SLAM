#!/usr/bin/env python3
"""Depth-L1 (input-vs-output, per model) — the Arm-4 benchmark Depth-L1 metric.

WHY THIS EXISTS — 00_COMMON.md §0 Resolved Decision #1 fixes the CRCD Depth-L1 as
**input-vs-output, per model**: CRCD ships no measured GT depth (MoGe-2 is generated),
so Depth-L1 is the mean L1 between each method's RENDERED/PREDICTED depth (output) and
the depth that SAME model was GIVEN as input on that snippet (the MoGe-2 + stereo-scaled
map). It is a per-model self-consistency metric, not a vs-GT accuracy metric — state this
in the paper. No prior implementation existed in the repo (verified 2026-06-17).

Pairing: by numeric frame id parsed from each filename (last run of digits in the stem),
intersecting the ids common to both dirs. Mismatched resolutions are reconciled by resizing
the INPUT depth to the RENDER depth shape with nearest-neighbour (depth must not be blurred).
Invalid pixels (<=0, or outside [min,max] metres in EITHER map) are masked out per frame.

Units: both maps are converted to METRES first (png value / scale; .npy already metres*scale
=> /scale). The input map is multiplied by --sc_factor so it lives in the same scaled frame
the model actually consumed (DDS scales loaded depth by sc_factor at datasets/dataset.py;
for methods with no sc_factor pass --sc_factor 1.0). L1 is reported in MILLIMETRES.

Output: a greppable summary line (parsed by aggregate_crcd_generic.py once extended) plus an
optional per-frame CSV curve.

  Depth-L1 (mm): mean=<M> median=<Md> frames=<N>

Usage:
  python Addons/eval/depth_l1.py \
      --render_depth_dir <OUT>/depth          --render_scale 10000 \
      --input_depth_dir  data/CRCD/<NAME>/depth --input_scale 10000 --sc_factor <SC> \
      --out <OUT>/depth_l1.txt --csv <OUT>/depth_l1_curve.csv
"""
import argparse
import glob
import os
import re

import numpy as np

try:
    import cv2
except ImportError:  # cv2 ships via colab_setup (opencv-contrib); fail loud if absent.
    cv2 = None


def _frame_id(path):
    """Last run of digits in the stem -> int id. Handles 0000.png, gs_0000.png, 123-left_depth.npy."""
    stem = os.path.splitext(os.path.basename(path))[0]
    ids = re.findall(r'\d+', stem)
    return int(ids[-1]) if ids else None


def _index_dir(d, pattern):
    """{frame_id: path} for files matching pattern in dir d (dedup: first wins on id collision)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        fid = _frame_id(p)
        if fid is not None and fid not in out:
            out[fid] = p
    return out


def _load_depth_m(path, scale):
    """Load a depth map and return it in METRES (float32). Supports uint16/any PNG and .npy."""
    if path.endswith('.npy'):
        arr = np.load(path).astype(np.float32)
    else:
        if cv2 is None:
            raise RuntimeError("cv2 not available but a PNG depth was given; install opencv.")
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"failed to read depth image: {path}")
        arr = arr.astype(np.float32)
    if arr.ndim == 3:  # collapse accidental 3-channel depth to its first channel
        arr = arr[..., 0]
    return arr / float(scale)


def main():
    ap = argparse.ArgumentParser(description="Depth-L1 (input-vs-output, per model), mm.")
    ap.add_argument('--render_depth_dir', required=True, help='rendered/predicted depth dir (the OUTPUT)')
    ap.add_argument('--input_depth_dir', required=True, help='the depth GIVEN to the model (the INPUT)')
    ap.add_argument('--render_pattern', default='*.png', help='glob in render dir (default *.png)')
    ap.add_argument('--input_pattern', default='*.png', help='glob in input dir (default *.png)')
    ap.add_argument('--render_scale', type=float, default=10000.0, help='render png value / scale = metres')
    ap.add_argument('--input_scale', type=float, default=10000.0, help='input value / scale = metres')
    ap.add_argument('--sc_factor', type=float, default=1.0, help='multiply input metres by this (model frame)')
    ap.add_argument('--min_depth_m', type=float, default=0.001, help='mask pixels below this (metres)')
    ap.add_argument('--max_depth_m', type=float, default=10.0, help='mask pixels above this (metres)')
    ap.add_argument('--out', default=None, help='write the summary line here')
    ap.add_argument('--csv', default=None, help='write the per-frame L1 curve here')
    a = ap.parse_args()

    ren = _index_dir(a.render_depth_dir, a.render_pattern)
    inp = _index_dir(a.input_depth_dir, a.input_pattern)
    common = sorted(set(ren) & set(inp))

    if not common:
        msg = (f"Depth-L1 (mm): mean=nan median=nan frames=0  "
               f"[NO COMMON FRAMES: render={len(ren)} input={len(inp)}]")
        print(msg)
        if a.out:
            open(a.out, 'w').write(msg + "\n")
        return

    if len(ren) != len(inp):
        print(f"[depth_l1] WARN render has {len(ren)} frames, input has {len(inp)}; "
              f"using {len(common)} common ids.")

    per_frame = []  # (fid, l1_mm, n_valid)
    for fid in common:
        r = _load_depth_m(ren[fid], a.render_scale)
        i = _load_depth_m(inp[fid], a.input_scale) * a.sc_factor
        if i.shape != r.shape:  # reconcile resolution onto the render grid (nearest, no blur)
            i = cv2.resize(i, (r.shape[1], r.shape[0]), interpolation=cv2.INTER_NEAREST)
        valid = (r > a.min_depth_m) & (r < a.max_depth_m) & \
                (i > a.min_depth_m) & (i < a.max_depth_m)
        n = int(valid.sum())
        if n == 0:
            continue
        l1_mm = float(np.abs(r[valid] - i[valid]).mean() * 1000.0)
        per_frame.append((fid, l1_mm, n))

    if not per_frame:
        msg = "Depth-L1 (mm): mean=nan median=nan frames=0  [NO VALID PIXELS]"
        print(msg)
        if a.out:
            open(a.out, 'w').write(msg + "\n")
        return

    vals = np.array([v for _, v, _ in per_frame], dtype=np.float64)
    mean, median = float(vals.mean()), float(np.median(vals))
    msg = f"Depth-L1 (mm): mean={mean:.3f} median={median:.3f} frames={len(per_frame)}"
    print(msg)
    print(f"[depth_l1] sc_factor={a.sc_factor} render_scale={a.render_scale} "
          f"input_scale={a.input_scale} band=({a.min_depth_m},{a.max_depth_m})m")

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as fh:
            fh.write(msg + "\n")
    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, 'w') as fh:
            fh.write("frame,depth_l1_mm,n_valid\n")
            for fid, v, n in per_frame:
                fh.write(f"{fid},{v:.4f},{n}\n")


if __name__ == '__main__':
    main()
