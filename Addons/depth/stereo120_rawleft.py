#!/usr/bin/env python3
"""Stereo-anchored METRIC depth on RAW-LEFT frames (no full rectification).

USER INSIGHT (2026-06-18): the stereo match exists ONLY to fix the depth SCALE. So we run MoGe on
the RAW left frames (skip rectifying every frame), and rectify ONLY the every-N anchor pairs to get
the per-snippet metric scale, which we then apply to the raw-left MoGe depth. The rectification maps
are precomputed in the calib pkl (ecm_map_*), so an anchor rectify is one cv2.remap -> cheap.

SCALE VALIDITY: MoGe's scale ambiguity is a single global multiplier, geometry-independent. So a
scale measured in RECTIFIED geometry (median(rect_stereo)/median(rect_moge) at the anchor) is the
SAME multiplier that maps the RAW-left MoGe to metric. We remap the anchor's raw MoGe to rectified
ONLY to compute the ratio in a matched geometry; the bake applies sc_f to the raw-left MoGe in place.

INPUTS (raw snippet, NOT preprocessed):
  --rgb_left_dir   raw left  frames (CRCD rgb/)        --left_glob  '*.png'
  --rgb_right_dir  raw right frames (CRCD rgbright/)   --right_glob '*.png'
  --moge_dir       MoGe depth PNG on RAW-left (uint16 = moge_m*in_scale)   --moge_glob '*.png'
  --calib_pkl      CRCD calib pickle (ecm_map_left_x/y, ecm_map_right_x/y)
  --intrinsics_yaml  snippet intrinsics.yaml (camera.fx, stereo.baseline_m)
OUTPUT  <out_dir>/<moge_stem>.png : metric depth (uint16 = metric_m*out_scale) in RAW-left geometry
        + prints per-anchor sc_f, drift, and the METRIC SCENE EXTENT.

  python Addons/depth/stereo120_rawleft.py --rgb_left_dir <raw>/rgb --rgb_right_dir <raw>/rgbright \
    --moge_dir <staged>/depth_rawleft --calib_pkl <calib>.pkl --intrinsics_yaml <raw>/intrinsics.yaml \
    --interval 120 --out_dir <staged>/depth/moge2_stereo120_rawleft
"""
import cv2, numpy as np, os, glob, argparse, pickle, yaml


def sgbm_metric_depth(lg, rg, baseline_m, fx):
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=7, P1=8 * 49, P2=32 * 49,
        disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(lg, rg).astype(np.float32) / 16.0
    vs = disp > 0.5
    sd = np.zeros_like(disp); sd[vs] = baseline_m * fx / disp[vs]
    return sd, vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_left_dir', required=True); ap.add_argument('--left_glob', default='*.png')
    ap.add_argument('--rgb_right_dir', required=True); ap.add_argument('--right_glob', default='*.png')
    ap.add_argument('--moge_dir', required=True); ap.add_argument('--moge_glob', default='*.png')
    ap.add_argument('--calib_pkl', required=True)
    ap.add_argument('--intrinsics_yaml', required=True)
    ap.add_argument('--interval', type=int, default=120)
    ap.add_argument('--in_scale', type=float, default=10000.0)
    ap.add_argument('--out_scale', type=float, default=10000.0)
    ap.add_argument('--max_depth_m', type=float, default=5.0)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.calib_pkl, 'rb') as f:
        cal = pickle.load(f)
    mlx, mly = cal['ecm_map_left_x'], cal['ecm_map_left_y']
    mrx, mry = cal['ecm_map_right_x'], cal['ecm_map_right_y']
    intr = yaml.safe_load(open(args.intrinsics_yaml))
    fx, baseline_m = float(intr['camera']['fx']), float(intr['stereo']['baseline_m'])

    L = sorted(glob.glob(os.path.join(args.rgb_left_dir, args.left_glob)))
    R = sorted(glob.glob(os.path.join(args.rgb_right_dir, args.right_glob)))
    M = sorted(glob.glob(os.path.join(args.moge_dir, args.moge_glob)))
    N = min(len(L), len(R), len(M))
    assert N >= 1, f'no aligned frames (L{len(L)} R{len(R)} M{len(M)})'
    print(f'[rawleft] {N} frames | fx={fx:.2f} baseline={baseline_m*1000:.2f}mm interval={args.interval}')

    def rect(img, mx, my):
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR)

    # ---- anchor scales (rectify ONLY these frames) ----
    a_idx, a_sc = [], []
    for i in sorted(set(list(range(0, N, args.interval)) + [0, N - 1])):
        lg = rect(cv2.imread(L[i], cv2.IMREAD_GRAYSCALE), mlx, mly)
        rg = rect(cv2.imread(R[i], cv2.IMREAD_GRAYSCALE), mrx, mry)
        moge_m = cv2.imread(M[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / args.in_scale
        moge_rect = rect(moge_m, mlx, mly)                      # match the stereo geometry for the ratio
        sd, vs = sgbm_metric_depth(lg, rg, baseline_m, fx)
        vj = vs & (sd > 0.05) & (sd < 3.0) & (moge_rect > 0.01)
        if vj.sum() < 500:
            print(f'  anchor {i:4d}: SKIP (few joint px)'); continue
        sc = float(np.median(sd[vj] / moge_rect[vj]))
        a_idx.append(i); a_sc.append(sc)
        print(f'  anchor {i:4d}: sc_f = {sc:.4f}')
    assert a_sc, 'no usable stereo anchor'
    a_idx, a_sc = np.array(a_idx, float), np.array(a_sc, float)
    print(f'  {len(a_sc)} anchors; sc [{a_sc.min():.4f},{a_sc.max():.4f}] drift {100*(a_sc.max()/a_sc.min()-1):.1f}%')

    # ---- smooth linear ramp + bake metric on the RAW-left MoGe (in place geometry) ----
    sc_pf = np.interp(np.arange(N, float), a_idx, a_sc)
    med = []
    for i in range(N):
        moge_m = cv2.imread(M[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / args.in_scale
        metric = np.clip(moge_m * sc_pf[i], 0, args.max_depth_m)
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(M[i])),
                    np.clip(metric * args.out_scale, 0, 65535).astype(np.uint16))
        v = metric[metric > 0.01]
        if v.size: med.append(np.median(v))
    print(f'  baked {len(glob.glob(args.out_dir + "/*.png"))} metric PNGs -> {args.out_dir}')
    if med:
        p2, p50, p98 = np.percentile(med, [2, 50, 98])
        print(f'  METRIC EXTENT m: p2={p2:.4f} p50={p50:.4f} p98={p98:.4f}  (median sc_f={np.median(sc_pf):.4f})')


if __name__ == '__main__':
    main()
