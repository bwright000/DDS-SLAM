"""
crcd_assemble_super.py — reformat a rectified CRCD snippet into Python-SuPer's SuPerDataset layout.

Phase-B of the Semantic-SuPer Arm-4 onboarding: run the UPSTREAM tracker on CRCD, evaluated with the
SAME harness as SGS (Sim3 ATE on the exported global T_g + render PSNR/SSIM/LPIPS + Depth-L1). The
upstream's native green-pin reproj-err is NOT computable on CRCD (no pin GT) and is skipped.

INPUT (from preprocess_crcd_published.py — rectified):
  <staged>/video_frames/<i>l.png   rectified LEFT
  <staged>/video_frames/<i>r.png   rectified RIGHT (CRCD ships stereo -> Semantic-SuPer needs both)
  <staged>/semantic_class/<i>.png  4-class map: 0=bg, 1=Liver, 2=Gallbladder, 3=Tool
  <staged>/groundtruth.txt         TUM GT poses (kept for Sim3 ATE)

OUTPUT (SuPerDataset, consumed by run_semantic_super.py with --data crcd):
  <out>/rgb/<i:06d>-left.png  + <i:06d>-right.png
  <out>/seg/<SRC>/<i:06d>-left.npy [+ -right.npy]   soft (n_cls,H,W) uint8 one-hot (loader argmaxes;
       soft path --sf_soft_seg_point_plane reads semantic_conf). SRC=GT here; DINOv3 head later.
  <out>/groundtruth.txt

Pairing is by SORTED INDEX (robust to non-zero-pad). NO green-pins (CRCD has none).

Usage:
  python Addons/colab/crcd_assemble_super.py --staged data/CRCD_staged/C1_001 \
         --out /content/Super_crcd/C1_001 --n_classes 4 --seg_src GT
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


def _onehot_npy(class_png, n_classes, out_npy):
    """4-class index map (0..n-1) -> one-hot (n_classes,H,W) uint8 .npy (the loader does argmax(0)/
    softmax; one-hot is a valid hard 'confidence' for the soft-seg path)."""
    seg = cv2.imread(class_png, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise RuntimeError(f"cannot read seg {class_png}")
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = np.clip(seg.astype(np.int64), 0, n_classes - 1)
    oh = np.zeros((n_classes, seg.shape[0], seg.shape[1]), dtype=np.uint8)
    for k in range(n_classes):
        oh[k][seg == k] = 1
    np.save(out_npy, oh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--staged', required=True, help='rectified CRCD dir (video_frames/, semantic_class/, groundtruth.txt)')
    ap.add_argument('--out', required=True, help='SuPerDataset output dir')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--seg_src', default='GT', help="subdir under seg/ (GT now; DINOv3 head later)")
    ap.add_argument('--no_right_seg', action='store_true', help='emit left seg only (skip -right duplicate)')
    a = ap.parse_args()

    rgb_out = os.path.join(a.out, 'rgb')
    seg_out = os.path.join(a.out, 'seg', a.seg_src)
    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(seg_out, exist_ok=True)

    L = _nsort(glob.glob(os.path.join(a.staged, 'video_frames', '*l.png')))
    R = _nsort(glob.glob(os.path.join(a.staged, 'video_frames', '*r.png')))
    S = _nsort(glob.glob(os.path.join(a.staged, 'semantic_class', '*.png')))
    n = min(len(L), len(R), len(S))
    if n < 2:
        raise SystemExit(f"[assemble-super] too few frames: left={len(L)} right={len(R)} seg={len(S)} "
                         f"(CRCD needs rectified STEREO + semantic_class; check the staging)")
    if len({len(L), len(R), len(S)}) > 1:
        print(f"[assemble-super] WARN count mismatch left={len(L)} right={len(R)} seg={len(S)} -> first {n}")

    for i in range(n):
        shutil.copy(L[i], os.path.join(rgb_out, f'{i:06d}-left.png'))
        shutil.copy(R[i], os.path.join(rgb_out, f'{i:06d}-right.png'))
        _onehot_npy(S[i], a.n_classes, os.path.join(seg_out, f'{i:06d}-left.npy'))
        if not a.no_right_seg:   # the loader pairs left+right seg; duplicate left (CRCD GT is left-frame)
            shutil.copy(os.path.join(seg_out, f'{i:06d}-left.npy'),
                        os.path.join(seg_out, f'{i:06d}-right.npy'))

    gt = os.path.join(a.staged, 'groundtruth.txt')
    if os.path.isfile(gt):
        shutil.copy(gt, os.path.join(a.out, 'groundtruth.txt'))
    else:
        print(f"[assemble-super] WARN no groundtruth.txt in {a.staged} -> Sim3 ATE will be skipped")

    print(f"[assemble-super] {n} frames -> {a.out}")
    print(f"  rgb/<i>-left.png + -right.png ; seg/{a.seg_src}/<i>-left.npy{'' if a.no_right_seg else ' + -right.npy'} "
          f"({a.n_classes}-class one-hot) ; groundtruth.txt")
    print(f"  run: run_semantic_super.py --data crcd --load_seg --seg_dir seg/{a.seg_src} --seg_ext .npy "
          f"--num_classes {a.n_classes}  (NO --tracking_gt_file: CRCD has no green-pins)")


if __name__ == '__main__':
    main()
