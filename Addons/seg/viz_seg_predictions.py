#!/usr/bin/env python3
"""Visualize the CRCD DINOv2 seg-head: RGB | GT-overlay | Pred-overlay per frame (+ per-frame mIoU).

Loads the trained dinov2_crcd.pth into the SAME DINO2SEG used by train_dinov2_crcd.py, runs it on a
few sampled frames per snippet, colorizes GT vs prediction (4 classes: 0=bg 1=Liver 2=Gallbladder
3=Tool), and writes one PNG per snippet. Default snippets = the 5 HELD-OUT benchmark scenes (the
ones with mIoU ~0.34) so you can see qualitatively where it fails; add a train snippet (e.g.
B2_001) via --snippets to contrast the ~0.93 in-domain behaviour.

Runs in the seg-head env (Colab torch2 + a vendored dinov2 main; same env that trained the head),
NOT the SGS conda env.

Usage (on the tunnel):
  cd /content/DDS-SLAM
  python Addons/seg/viz_seg_predictions.py \
      --crcd_root /content/crcd_seg_data \
      --dinov2_main /content/SemGauss-SLAM/segmentation/facebookresearch_dinov2_main \
      --pth seg/dinov2_crcd.pth --out /content/drive/MyDrive/Outputs/seg/viz --n 6
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
    import torch
except ImportError as e:  # pragma: no cover
    print(f"[viz_seg] need cv2+torch (seg-head env): {e}"); raise

# reuse the exact training-time model + normalization (same dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_dinov2_crcd import DINO2SEG, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# class colours (RGB): bg=grey, Liver=red, Gallbladder=green, Tool=blue
LUT = np.array([[60, 60, 60], [220, 40, 40], [40, 200, 40], [40, 90, 235]], np.uint8)
CLASS_NAMES = ['bg', 'Liver', 'Gallbladder', 'Tool']


def colorize(lab):
    return LUT[np.clip(lab, 0, len(LUT) - 1)]


def overlay(rgb, lab, alpha=0.55):
    col = colorize(lab)
    fg = (lab > 0)[..., None]  # keep RGB for bg, blend colour onto fg classes
    return np.where(fg, (alpha * col + (1 - alpha) * rgb).astype(np.uint8), rgb)


def frame_miou(pred, gt, n_classes):
    ious = []
    for c in range(n_classes):
        p, g = (pred == c), (gt == c)
        u = np.logical_or(p, g).sum()
        if u > 0:
            ious.append(np.logical_and(p, g).sum() / u)
    return float(np.mean(ious)) if ious else 0.0


def label_bar(width, text):
    bar = np.zeros((24, width, 3), np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crcd_root', default='/content/crcd_seg_data', help='dir of <snippet>/{rgb,semantic_instance}')
    ap.add_argument('--snippets', nargs='+', default=['C1_001', 'C2_001', 'E3_005', 'C3_001', 'G3_001'])
    ap.add_argument('--pth', default='seg/dinov2_crcd.pth')
    ap.add_argument('--dinov2_main', required=True)
    ap.add_argument('--out', default='/content/drive/MyDrive/Outputs/seg/viz')
    ap.add_argument('--n', type=int, default=6, help='frames per snippet (evenly spaced)')
    ap.add_argument('--rgb_subdir', default='rgb')
    ap.add_argument('--label_subdir', default='semantic_instance')
    ap.add_argument('--img_h', type=int, default=504)
    ap.add_argument('--img_w', type=int, default=896)
    ap.add_argument('--crop_edge', type=int, default=0)
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--dim', type=int, default=16)
    a = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DINO2SEG(a.img_h, a.img_w, a.n_classes, a.dinov2_main, edge=a.crop_edge, dim=a.dim).to(device).eval()
    sd = torch.load(a.pth, map_location=device)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[viz_seg] loaded {a.pth} (missing={len(miss)} unexpected={len(unexp)})")
    os.makedirs(a.out, exist_ok=True)
    H, W = a.img_h, a.img_w

    for snip in a.snippets:
        rgbs = sorted(glob.glob(os.path.join(a.crcd_root, snip, a.rgb_subdir, '*.png')))
        labs = {os.path.basename(p): p for p in glob.glob(os.path.join(a.crcd_root, snip, a.label_subdir, '*.png'))}
        pairs = [(r, labs[os.path.basename(r)]) for r in rgbs if os.path.basename(r) in labs]
        if not pairs:
            print(f"[{snip}] no (rgb, {a.label_subdir}) pairs under {a.crcd_root}/{snip} -> skip"); continue
        idx = np.linspace(0, len(pairs) - 1, min(a.n, len(pairs))).astype(int)
        rows, mious = [], []
        for i in idx:
            rgb_p, lab_p = pairs[i]
            rgb = cv2.resize(cv2.cvtColor(cv2.imread(rgb_p), cv2.COLOR_BGR2RGB), (W, H))
            x = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
            with torch.no_grad():
                logits = model(torch.from_numpy(x).float().to(device))
            pred = cv2.resize(logits.argmax(1)[0].cpu().numpy().astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            gt = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
            if gt.ndim == 3:
                gt = gt[..., 0]
            gt = cv2.resize(gt.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            m = frame_miou(pred, gt, a.n_classes); mious.append(m)
            tag = os.path.splitext(os.path.basename(rgb_p))[0]
            # RGB | GT overlay | Pred overlay | Pred SOLID (every pixel coloured; bg=grey) so a
            # poor-but-structured prediction is distinguishable from genuine noise.
            quad = np.hstack([rgb, overlay(rgb, gt), overlay(rgb, pred), colorize(pred)])
            bars = np.hstack([label_bar(W, f"{snip} {tag} RGB"), label_bar(W, "GT (overlay)"),
                              label_bar(W, f"Pred (overlay) mIoU={m:.2f}"), label_bar(W, "Pred (solid)")])
            rows.append(np.vstack([bars, quad]))
        grid = np.vstack(rows)
        legend = " ".join(f"{i}={CLASS_NAMES[i]}" for i in range(a.n_classes))
        grid = np.vstack([label_bar(grid.shape[1], f"{snip}  mean mIoU(shown {len(idx)}f)={np.mean(mious):.3f}   classes: {legend}"), grid])
        out_p = os.path.join(a.out, f"{snip}_segviz.png")
        cv2.imwrite(out_p, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"[{snip}] mIoU(shown)={np.mean(mious):.3f}  ({len(idx)} frames) -> {out_p}")
    print(f"[viz_seg] done -> {a.out}")


if __name__ == '__main__':
    main()
