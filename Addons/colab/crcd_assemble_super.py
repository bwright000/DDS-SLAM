"""
crcd_assemble_super.py — reformat a rectified CRCD snippet into Python-SuPer's SuPerDataset layout
(Phase-B of the Semantic-SuPer Arm-4 onboarding), with RESIZE + MoGe metric depth.

Run the UPSTREAM tracker on CRCD, evaluated with the SAME harness as SGS (render PSNR/SSIM/LPIPS +
Depth-L1 now; Sim3 ATE on the exported global T_g next). Native green-pin reproj-err is NOT
computable on CRCD (no pin GT) -> skipped.

DECISIONS (user): depth = MoGe (SGS-comparable, loaded as metric meters, NOT Monodepth2 live);
resize CRCD 1280x720 -> a Super-scale res (default 640x360, keeps 16:9) to cut surfel count, with
the intrinsics scaled to match.

INPUT:
  <staged>/video_frames/<i>l.png + <i>r.png   rectified stereo (preprocess_crcd_published)
  <staged>/semantic_class/<i>.png             4-class map (0=bg,1=Liver,2=Gallbladder,3=Tool)
  <staged>/groundtruth.txt                     TUM GT poses (kept for Sim3 ATE)
  --moge_depth <dir>/<i>.png                   MoGe-2 depth, uint16, value/depth_scale = metres
  --calib rectified_calib.txt                  rectified K (fx,fy,cx,cy at 1280x720) to scale

OUTPUT (SuPerDataset, run_semantic_super.py --data crcd):
  <out>/rgb/<i:06d>-left.png + -right.png                 resized RGB
  <out>/seg/<SRC>/<i:06d>-left.npy [+ -right.npy]         resized one-hot (n_cls,H,W) uint8
  <out>/depth/<i:06d>-left.npy                            resized MoGe depth, float32 METRES
  <out>/crcd_K.txt                                        fx fy cx cy at the resized res (get_K patch reads this)
  <out>/groundtruth.txt
  run: --data crcd --load_depth --depth_dir depth --depth_ext .npy --load_seg --seg_dir seg/<SRC>
       --seg_ext .npy --num_classes N   (NO --tracking_gt_file: CRCD has no green-pins)
"""
import argparse
import glob
import os
import re
import shutil

import cv2
import numpy as np


def _nsort(fs):
    return sorted(fs, key=lambda p: int(re.search(r'(\d+)', os.path.basename(p)).group(1)))


def _read_calib(path):
    kv = {}
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) >= 2:
                try:
                    kv[p[0]] = float(p[1])
                except ValueError:
                    pass
    return kv  # expects fx, fy, cx, cy, width, height


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', required=True)
    ap.add_argument('--moge_depth', required=True, help='CRCD MoGe depth dir (uint16 png; value/depth_scale=metres)')
    ap.add_argument('--calib', required=True, help='rectified_calib.txt (fx,fy,cx,cy at native res)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--img_w', type=int, default=640)
    ap.add_argument('--img_h', type=int, default=360)
    ap.add_argument('--depth_scale', type=float, default=10000.0)
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--seg_src', default='GT')
    ap.add_argument('--no_right_seg', action='store_true')
    a = ap.parse_args()

    rgb_out = os.path.join(a.out, 'rgb')
    seg_out = os.path.join(a.out, 'seg', a.seg_src)
    dep_out = os.path.join(a.out, 'depth')
    for d in (rgb_out, seg_out, dep_out):
        os.makedirs(d, exist_ok=True)

    L = _nsort(glob.glob(os.path.join(a.staged, 'video_frames', '*l.png')))
    R = _nsort(glob.glob(os.path.join(a.staged, 'video_frames', '*r.png')))
    S = _nsort(glob.glob(os.path.join(a.staged, 'semantic_class', '*.png')))
    D = _nsort(glob.glob(os.path.join(a.moge_depth, '*.png')))
    n = min(len(L), len(R), len(S), len(D))
    if n < 2:
        raise SystemExit(f"[assemble-super] too few frames: left={len(L)} right={len(R)} seg={len(S)} depth={len(D)}")
    if len({len(L), len(R), len(S), len(D)}) > 1:
        print(f"[assemble-super] WARN counts left={len(L)} right={len(R)} seg={len(S)} depth={len(D)} -> first {n}")

    # native res from the first left frame; scale the calib to the resize res
    h0, w0 = cv2.imread(L[0]).shape[:2]
    sx, sy = a.img_w / w0, a.img_h / h0
    k = _read_calib(a.calib)
    fx, fy, cx, cy = k['fx'] * sx, k['fy'] * sy, k['cx'] * sx, k['cy'] * sy
    with open(os.path.join(a.out, 'crcd_K.txt'), 'w') as f:
        f.write(f"fx {fx}\nfy {fy}\ncx {cx}\ncy {cy}\nwidth {a.img_w}\nheight {a.img_h}\n")
    print(f"[assemble-super] native {w0}x{h0} -> resize {a.img_w}x{a.img_h}  K: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    sz = (a.img_w, a.img_h)
    for i in range(n):
        cv2.imwrite(os.path.join(rgb_out, f'{i:06d}-left.png'),
                    cv2.resize(cv2.imread(L[i]), sz, interpolation=cv2.INTER_AREA))
        cv2.imwrite(os.path.join(rgb_out, f'{i:06d}-right.png'),
                    cv2.resize(cv2.imread(R[i]), sz, interpolation=cv2.INTER_AREA))
        # seg: class-map -> resize NEAREST -> one-hot (n_cls,H,W) uint8
        seg = cv2.imread(S[i], cv2.IMREAD_UNCHANGED)
        if seg.ndim == 3:
            seg = seg[..., 0]
        seg = cv2.resize(seg.astype(np.uint8), sz, interpolation=cv2.INTER_NEAREST).astype(np.int64)
        seg = np.clip(seg, 0, a.n_classes - 1)
        oh = np.zeros((a.n_classes, a.img_h, a.img_w), dtype=np.uint8)
        for c in range(a.n_classes):
            oh[c][seg == c] = 1
        np.save(os.path.join(seg_out, f'{i:06d}-left.npy'), oh)
        if not a.no_right_seg:
            np.save(os.path.join(seg_out, f'{i:06d}-right.npy'), oh)
        # depth: MoGe uint16 -> metres -> resize NEAREST (preserve depth values) -> float32 .npy
        dep = cv2.imread(D[i], cv2.IMREAD_UNCHANGED).astype(np.float32) / a.depth_scale
        dep = cv2.resize(dep, sz, interpolation=cv2.INTER_NEAREST)
        np.save(os.path.join(dep_out, f'{i:06d}-left.npy'), dep.astype(np.float32))

    gt = os.path.join(a.staged, 'groundtruth.txt')
    if os.path.isfile(gt):
        shutil.copy(gt, os.path.join(a.out, 'groundtruth.txt'))
    else:
        print(f"[assemble-super] WARN no groundtruth.txt -> Sim3 ATE will be skipped")

    print(f"[assemble-super] {n} frames -> {a.out} (rgb/-left+-right, seg/{a.seg_src}/-left.npy one-hot, "
          f"depth/-left.npy metres, crcd_K.txt, groundtruth.txt)")


if __name__ == '__main__':
    main()
