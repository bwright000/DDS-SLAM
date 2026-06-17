#!/usr/bin/env python3
"""Stereo-anchored METRIC depth for CRCD: SGBM every N frames + smooth linear-ramp scale.

WHY (the x9-10 fix): MoGe-2 monocular depth is up-to-scale, and on a close-up surgical FOV its
"metres" run ~9-10x too large. The canon configs train with sc_factor=1, so the SLAM lives in
MoGe's inflated coordinate while GT is metric -> Sim3 recovers ~0.11 (the x9-10). StereoSGBM on the
rectified L/R pair gives TRUE metric depth (fx*baseline/disp). We anchor MoGe to it.

WHAT this does beyond the old frame-0 anchor (crcd_depth_gen_remainder):
  * SGBM at EVERY --interval frames (default 120), not just frame 0 -> captures scale DRIFT.
  * SMOOTH LINEAR RAMP of the scale between anchors (no depth discontinuity that would corrupt the
    SLAM map; a step function would jump the geometry at each window boundary).
  * BAKES the per-frame metric scale into a new corpus depth/moge2_stereo120/<fid>.png = moge_m*sc_f
    (x out_scale, uint16). Train on this with sc_factor=1 -> est comes out METRIC -> Sim3 scale ~1.
  * Reports the METRIC scene extent (depth percentiles) so the matched config's bound/trunc/range_d
    can be rescaled (neural SLAM is NOT scale-invariant -> the bound MUST follow the depth).

INPUTS (a staged+rectified CRCD snippet dir, as produced by preprocess_crcd_published.py):
  <staged>/video_frames/<fid>l.png , <fid>r.png   (rectified stereo pair)
  <staged>/depth/<fid>.png                          (MoGe metric depth, uint16, = moge_m*in_scale)
  <staged>/rectified_calib.txt                      (baseline_m, fx)
OUTPUT:
  <staged>/depth/moge2_stereo120/<fid>.png          (metric depth, uint16 = moge_m*sc_f*out_scale)
  <staged>/moge2_stereo120_sctrack.txt              (per-frame fid sc_f, for inspection)
  prints METRIC EXTENT (p2/p50/p98 metres) + a suggested bound for the matched config.

  python Addons/depth/stereo120_metric_anchor.py --staged data/CRCD/C1_001 --interval 120
"""
import cv2, numpy as np, os, glob, argparse


def sgbm_metric_depth(left_gray, right_gray, baseline_m, fx):
    """StereoSGBM disparity -> metric depth map (same params as the proven frame-0 anchor)."""
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=7, P1=8 * 49, P2=32 * 49,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
    valid = disp > 0.5
    sd = np.zeros_like(disp)
    sd[valid] = baseline_m * fx / disp[valid]
    return sd, valid


def anchor_scale(staged, fid, baseline_m, fx, in_scale):
    """sc = median(stereo_depth / moge_depth) at one anchor frame. None if too few joint px."""
    left = cv2.imread(f'{staged}/video_frames/{fid}l.png', cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(f'{staged}/video_frames/{fid}r.png', cv2.IMREAD_GRAYSCALE)
    moge = cv2.imread(f'{staged}/depth/{fid}.png', cv2.IMREAD_UNCHANGED)
    if left is None or right is None or moge is None:
        return None
    moge_m = moge.astype(np.float32) / in_scale
    sd, vs = sgbm_metric_depth(left, right, baseline_m, fx)
    vj = vs & (sd > 0.05) & (sd < 3.0) & (moge_m > 0.01)
    if vj.sum() < 500:
        return None
    return float(np.median(sd[vj] / moge_m[vj]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', required=True, help='staged+rectified CRCD snippet dir')
    ap.add_argument('--interval', type=int, default=120, help='SGBM re-anchor every N frames')
    ap.add_argument('--in_scale', type=float, default=10000.0, help='moge PNG = moge_m * in_scale')
    ap.add_argument('--out_scale', type=float, default=10000.0, help='output PNG = metric_m * out_scale')
    ap.add_argument('--max_depth_m', type=float, default=5.0)
    ap.add_argument('--out_subdir', default='moge2_stereo120')
    args = ap.parse_args()

    calib = {}
    for line in open(f'{args.staged}/rectified_calib.txt'):
        k, v = line.strip().split(); calib[k] = float(v)
    baseline_m, fx = calib['baseline_m'], calib['fx']

    lefts = sorted(glob.glob(f'{args.staged}/video_frames/*l.png'))
    fids = [os.path.basename(p).replace('l.png', '') for p in lefts]
    N = len(fids)
    assert N > 0, f'no left frames in {args.staged}/video_frames'

    # ---- 1. anchor scales at every `interval` frame (+ first & last so the ramp is bracketed) ----
    anchor_idx = sorted(set(list(range(0, N, args.interval)) + [0, N - 1]))
    a_idx, a_sc = [], []
    for i in anchor_idx:
        sc = anchor_scale(args.staged, fids[i], baseline_m, fx, args.in_scale)
        if sc is not None:
            a_idx.append(i); a_sc.append(sc)
            print(f'  anchor frame {i:4d} ({fids[i]}): sc_f = {sc:.4f}')
        else:
            print(f'  anchor frame {i:4d} ({fids[i]}): SKIP (too few joint stereo px)')
    assert len(a_sc) >= 1, 'no usable stereo anchor -> cannot build metric depth'
    a_idx, a_sc = np.array(a_idx, float), np.array(a_sc, float)
    print(f'  {len(a_sc)} usable anchors; sc range [{a_sc.min():.4f}, {a_sc.max():.4f}] '
          f'(drift {100*(a_sc.max()/a_sc.min()-1):.1f}%)')

    # ---- 2. smooth linear ramp of sc across ALL frames (np.interp clamps at the ends) ----
    sc_per_frame = np.interp(np.arange(N, dtype=float), a_idx, a_sc)

    # ---- 3. bake metric depth + collect extent ----
    out_dir = f'{args.staged}/depth/{args.out_subdir}'
    os.makedirs(out_dir, exist_ok=True)
    all_med = []
    with open(f'{args.staged}/{args.out_subdir}_sctrack.txt', 'w') as tf:
        for i, fid in enumerate(fids):
            moge = cv2.imread(f'{args.staged}/depth/{fid}.png', cv2.IMREAD_UNCHANGED)
            if moge is None:
                print(f'  WARN missing moge depth {fid} -> skip'); continue
            metric_m = np.clip(moge.astype(np.float32) / args.in_scale * sc_per_frame[i], 0, args.max_depth_m)
            cv2.imwrite(f'{out_dir}/{fid}.png',
                        np.clip(metric_m * args.out_scale, 0, 65535).astype(np.uint16))
            tf.write(f'{fid} {sc_per_frame[i]:.6f}\n')
            v = metric_m[metric_m > 0.01]
            if v.size:
                all_med.append(np.median(v))
    print(f'  baked {len(glob.glob(out_dir + "/*.png"))} metric PNGs -> {out_dir}')

    # ---- 4. report metric extent so the matched config bound/trunc/range_d can be rescaled ----
    if all_med:
        # pooled percentiles over per-frame medians = a robust scene-depth band
        p2, p50, p98 = np.percentile(all_med, [2, 50, 98])
        print('\n=== METRIC SCENE EXTENT (metres) ===')
        print(f'  per-frame median depth: p2={p2:.4f}  p50={p50:.4f}  p98={p98:.4f}')
        print(f'  the canon bound z=[0.68,0.9] was MoGe-scale; metric z ~ [{p2*0.85:.3f}, {p98*1.15:.3f}]')
        print(f'  -> matched config: scale bound/trunc/range_d by ~sc_f median '
              f'({np.median(sc_per_frame):.4f}); sc_factor=1; depth_subdir=depth/{args.out_subdir}')


if __name__ == '__main__':
    main()
