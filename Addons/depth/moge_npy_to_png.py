#!/usr/bin/env python3
"""Convert MoGe-2 depth .npy files to uint16 PNGs the SLAM loaders read.

WHY THIS EXISTS — generate_depth_moge.py writes ONLY '<fid>-left_depth.npy' (float32, values =
depth_m * depth_scale). It has no PNG output and no --out-as-png flag. The npy->PNG conversion
was previously an INLINE python block inside crcd_depth_gen_remainder_20260616.sh (lines ~103-107)
and run_crcd_4snippets.sh, so a fresh agent told to 'reuse the harness' had no standalone command.
This is that block extracted verbatim into a named, callable tool.

fid convention: input file '<fid>-left_depth.npy' -> output '<fid>.png'. The fid is the basename
up to the first '-' (matches generate_depth_moge.py's '*-left.png' input naming and the preprocess
'NNNNNNl.png' -> '_moge_in/NNNNNN-left.png' symlink step).

Range note: at --depth_scale 10000 with --max_depth_m 5.0, max stored value = 50000 < 65535
(uint16 ceiling), so the clip(.,0,65535) is a safety net, not a lossy clamp, for surgical depth.

Usage:
  python Addons/depth/moge_npy_to_png.py --in _moge_npy --out data/CRCD/<NAME>/depth
"""
import argparse
import glob
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='dir of *-left_depth.npy')
    ap.add_argument('--out', required=True, help='dir for <fid>.png')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(a.inp, '*-left_depth.npy'))):
        fid = os.path.basename(p).split('-')[0]
        out = os.path.join(a.out, f'{fid}.png')
        if os.path.exists(out):
            n += 1
            continue
        cv2.imwrite(out, np.clip(np.load(p).astype(np.float32), 0, 65535).astype(np.uint16))
        n += 1
    print(f'npy->png: {n} files in {a.out}')


if __name__ == '__main__':
    main()
