#!/usr/bin/env python3
"""Diagnose the RAW per-pixel Sampson flow residual (PRE-deadband) on a frame sequence -- to answer the
exact question: was the scene motion BELOW the route deadband (suppressed), or genuinely ~0 (nothing to chase),
or REAL-but-washed-out by the region-MEDIAN coarseness? Reuses the production flow_track.flow_residual so it is
faithful to what map_route/flow_track actually saw. Flow-only, no SLAM, no training.

Reading the output (residual is in PIXELS; the route deadband is 3.0px on the per-region MEDIAN):
  * P99 (and max) << deadband on a frame  -> motion genuinely below threshold there: nothing to chase / suppressed.
  * P99 > deadband but map_route logged moving-frac=0 -> the motion IS real and localized, but the per-region
    MEDIAN washed it out (a small tool can't push a 14px-region median over 3px). Then the SIGNAL DESIGN
    (region-median), not the deadband value, is what silenced the up-weight.
  * P99 ~ 0 everywhere -> there is simply no camera-inconsistent motion in this clip (CRCD sub-SNR).

  python Addons/motion/diag_flow_residual.py --frames data/CRCD/C1_001/video_frames --glob '*l.png' \
      --ref_stride 8 --deadband 3.0 --every 10
"""
import sys, os, glob, argparse
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Addons.motion.flow_track import load_raft, flow_residual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True, help='dir of frames (the SAME ones the run used)')
    ap.add_argument('--glob', default='*l.png')
    ap.add_argument('--ref_stride', type=int, default=8, help='causal ref = t - this (matches map_route)')
    ap.add_argument('--deadband', type=float, default=3.0, help='route deadband to compare against (px)')
    ap.add_argument('--every', type=int, default=10, help='diagnose every Nth frame (speed)')
    ap.add_argument('--ransac_thresh', type=float, default=1.0)
    ap.add_argument('--small', action='store_true', help='RAFT-small (faster, e.g. weak GPU/CPU)')
    args = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tf = load_raft(device, small=args.small)
    frames = sorted(glob.glob(os.path.join(args.frames, args.glob)))
    assert len(frames) > args.ref_stride, f"only {len(frames)} frames matched {args.frames}/{args.glob}"
    print(f"# {len(frames)} frames | ref_stride={args.ref_stride} | deadband={args.deadband}px | "
          f"RAFT-{'small' if args.small else 'large'} on {device}")
    print(f"{'frame':>6} {'P50':>7} {'P90':>7} {'P99':>7} {'max':>8} {'%px>db':>7}  verdict")
    p99s, maxs, n_above = [], [], 0
    for i in range(args.ref_stride, len(frames), args.every):
        ref = cv2.imread(frames[i - args.ref_stride]); cur = cv2.imread(frames[i])
        resid = flow_residual(ref, cur, model, tf, device, ransac_thresh=args.ransac_thresh)
        p50, p90, p99 = (float(x) for x in np.percentile(resid, [50, 90, 99]))
        mx = float(resid.max()); frac = float((resid > args.deadband).mean())
        verdict = ('REAL motion > db' if p99 > args.deadband else
                   ('tiny (<db)' if mx > 0.5 else '~none'))
        print(f"{i:>6} {p50:>7.2f} {p90:>7.2f} {p99:>7.2f} {mx:>8.2f} {100*frac:>6.2f}%  {verdict}")
        p99s.append(p99); maxs.append(mx); n_above += (p99 > args.deadband)
    print(f"\n# SUMMARY over {len(p99s)} sampled frames:")
    print(f"#   median P99 = {np.median(p99s):.2f}px | median max = {np.median(maxs):.2f}px | "
          f"peak max = {np.max(maxs):.2f}px")
    print(f"#   frames with P99 > deadband({args.deadband}px) = {n_above}/{len(p99s)}")
    print(f"#   ANSWER: P99<<db everywhere -> motion below threshold (suppressed/none, lower the deadband to test).")
    print(f"#           P99>db on frames map_route logged 0 -> region-MEDIAN coarseness silenced it, not the deadband.")


if __name__ == '__main__':
    main()
