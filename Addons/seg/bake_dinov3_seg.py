#!/usr/bin/env python3
"""Bake DINOv3 4-class seg predictions for a CRCD snippet -> SNI semantic_class PNGs (run B).

SNI's env is py3.7/torch1.11 and CANNOT run DINOv3 (HF transformers, torch>=2). So for the
"predicted-DINOv3-seg" run (B) we predict the seg OFFLINE in torch2 with our trained per-snippet
LOSO head and overwrite the SNI staging's semantic_class/ PNGs. SNI consumes them via
use_gt_semantic=True (it reads semantic_class_NNNNNN.png as the per-pixel label) -> NO SNI code change.

DOMAIN FAITHFULNESS (the load-bearing detail): the LOSO head was trained AND held-out-validated on
RAW (unrectified) CRCD left frames. SNI runs on RECTIFIED frames. So we MIRROR preprocess_crcd_for_sni
exactly: predict on the RAW left frame (in-domain for the head), then rectify the PREDICTED LABEL with
the SAME map_left + cv2.INTER_NEAREST that preprocess used for the GT semantic_class. Result: predicted
and GT seg are produced by the identical "raw -> label -> remap" pipeline (apples-to-apples A/B), the
head sees the distribution it was trained on, and the label registers with SNI's rectified rgb/depth.
Output res = the rectify maps' shape (== rgb/depth res), and 0-based enumeration of sorted raw files
matches preprocess's rgb_{i:06d}.png / semantic_class_{i:06d}.png indexing.

Faithfulness of the head: reuses the EXACT training-time model (DINO2SEG + build_backbone + IMAGENET
norm) from train_dinov2_crcd.py, the same base DINOv3 ViT-B/16 HF dir the LOSO used as BACKBONE_WEIGHTS
(/content/drive/MyDrive/Datasets/DiNO), the same 504x896 / dim16 / n_classes4 / patch16 / n_register4
construction (fold logs: hidden=768 patch=16 n_register=4). Our trained .pth (full backbone +
segmentation_conv) loads over the base; if ANY backbone OR segmentation_conv key fails to load
(e.g. a transformers-version drift changing AutoModel's key layout), we ABORT rather than silently run
the base (non-fine-tuned) backbone.

The per-snippet head MUST be the LEAVE-THAT-SNIPPET-OUT fold (dinov2_crcd_C1_001_SWEEPONLY.pth for
c1_001) so the seg the SLAM consumes never saw the test snippet -> an honest deploy result.

Env: torch2 + transformers + cv2 (NOT the sni env). On Colab: the system python.

Usage:
  python Addons/seg/bake_dinov3_seg.py \
    --raw_rgb_dir /content/crcd_raw/C1_001/rgb \
    --out_seg     /content/sni-slam/data/CRCD/C1_001/semantic_class \
    --calib_npz   /content/calib_maps.npz \
    --base_dinov3 /content/drive/MyDrive/Datasets/DiNO \
    --head_pth    /content/drive/MyDrive/Datasets/seg/loso_dinov3_b2/dinov2_crcd_C1_001_SWEEPONLY.pth \
    --img_h 504 --img_w 896 --expect_n 360
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
    import torch
except ImportError as e:
    print(f"[bake-dino] need cv2+torch2 (system python, NOT the sni env): {e}")
    raise

# reuse the exact training-time model + normalisation (same dir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train_dinov2_crcd import DINO2SEG, build_backbone, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw_rgb_dir', required=True, help='RAW (unrectified) CRCD left frames (*.png), as preprocess reads')
    ap.add_argument('--out_seg', required=True, help='SNI semantic_class/ dir (PNGs overwritten, rectified frame)')
    ap.add_argument('--calib_npz', required=True, help='/content/calib_maps.npz (ecm_map_left_x/y) — the SAME maps preprocess used')
    ap.add_argument('--base_dinov3', required=True, help='base DINOv3 ViT-B/16 HF dir (config.json + model.safetensors)')
    ap.add_argument('--head_pth', required=True, help='trained LOSO head .pth (full backbone + segmentation_conv)')
    ap.add_argument('--dinov2_main', default='', help='unused for dinov3 (build_backbone arg pass-through)')
    ap.add_argument('--img_h', type=int, default=504, help='backbone input H (training res)')
    ap.add_argument('--img_w', type=int, default=896, help='backbone input W (training res)')
    ap.add_argument('--expect_n', type=int, default=0, help='expected frame count (== rectified rgb/depth); asserted if >0')
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--dim', type=int, default=16)
    a = ap.parse_args()

    for p in (a.base_dinov3, a.head_pth, a.raw_rgb_dir, a.calib_npz):
        if not os.path.exists(p):
            raise SystemExit(f"[bake-dino] FATAL: missing path {p}")
    os.makedirs(a.out_seg, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- rectify maps (the SAME ones preprocess used; output res = their shape) -------------
    maps = np.load(a.calib_npz)
    if 'ecm_map_left_x' not in maps or 'ecm_map_left_y' not in maps:
        raise SystemExit(f"[bake-dino] FATAL: {a.calib_npz} lacks ecm_map_left_x/y (keys={list(maps.keys())})")
    mx, my = maps['ecm_map_left_x'], maps['ecm_map_left_y']
    out_h, out_w = mx.shape[:2]
    print(f"[bake-dino] rectify maps {out_w}x{out_h} (== SNI rgb/depth res)")

    # --- build the EXACT training model: base DINOv3 (HF) + 2-conv head, load our trained .pth
    bb, patch, nreg, embed = build_backbone('dinov3', a.dinov2_main, a.base_dinov3)
    print(f"[bake-dino] backbone: patch={patch} n_register={nreg} embed={embed} (expect 16/4/768)")
    model = DINO2SEG(a.img_h, a.img_w, a.n_classes, bb, patch_size=patch, n_register=nreg,
                     edge=0, dim=a.dim, embed=embed).to(device).eval()
    sd = torch.load(a.head_pth, map_location=device)
    if isinstance(sd, dict) and 'state_dict' in sd and not any(k.startswith(('backbone', 'segmentation')) for k in sd):
        sd = sd['state_dict']
    miss, unexp = model.load_state_dict(sd, strict=False)
    seg_miss = [k for k in miss if 'segmentation_conv' in k]
    bb_miss = [k for k in miss if k.startswith('backbone')]
    print(f"[bake-dino] loaded {os.path.basename(a.head_pth)}: missing={len(miss)} unexpected={len(unexp)} "
          f"(seg_miss={len(seg_miss)} backbone_miss={len(bb_miss)})")
    # ABORT on ANY unloaded backbone/seg key: a transformers-version drift would change AutoModel's
    # key layout so the .pth backbone keys silently drop (strict=False) -> base, NOT fine-tuned, weights.
    if seg_miss or bb_miss:
        raise SystemExit(f"[bake-dino] FATAL: keys not loaded -> arch/transformers drift. "
                         f"seg_miss={seg_miss} backbone_miss={len(bb_miss)} e.g.{bb_miss[:3]} unexpected={len(unexp)}")

    raws = sorted(glob.glob(os.path.join(a.raw_rgb_dir, '*.png')))   # lexicographic == preprocess's sorted()
    if not raws:
        raise SystemExit(f"[bake-dino] FATAL: no raw rgb PNGs under {a.raw_rgb_dir}")
    if a.expect_n and len(raws) != a.expect_n:
        print(f"[bake-dino] WARN: raw frames {len(raws)} != expect_n {a.expect_n} (using first {a.expect_n})")
        raws = raws[:a.expect_n]
    print(f"[bake-dino] {len(raws)} raw frames: predict {a.img_w}x{a.img_h} -> rectify -> seg {out_w}x{out_h}")

    hist = np.zeros(a.n_classes, np.int64)
    written = 0
    for i, rp in enumerate(raws):
        bgr = cv2.imread(rp)
        if bgr is None:
            raise SystemExit(f"[bake-dino] FATAL: unreadable {rp}")
        h0, w0 = bgr.shape[:2]
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (a.img_w, a.img_h), interpolation=cv2.INTER_LINEAR)
        x = ((rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        with torch.no_grad():
            logits = model(torch.from_numpy(x).float().to(device))
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)                       # [img_h, img_w]
        pred = cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)              # -> raw res
        pred = cv2.remap(pred, mx, my, cv2.INTER_NEAREST)                               # -> rectified (== GT pipeline)
        cv2.imwrite(os.path.join(a.out_seg, f'semantic_class_{i:06d}.png'), pred.astype(np.uint8))
        written += 1
        for c in range(a.n_classes):
            hist[c] += int((pred == c).sum())

    tot = max(1, int(hist.sum()))
    pct = {c: 100 * hist[c] / tot for c in range(a.n_classes)}
    print(f"[bake-dino] wrote {written} semantic_class PNGs -> {a.out_seg}")
    print("[bake-dino] class px%: " + " ".join(f"{c}:{pct[c]:.1f}" for c in range(a.n_classes)))
    # degeneracy guard: a silently-misloaded head / bad ckpt typically collapses to one class.
    present = sum(1 for c in range(a.n_classes) if hist[c] > 0)
    if present < 2 or max(pct.values()) > 99.0:
        print(f"[bake-dino] WARN: degenerate seg (classes_present={present}, max%={max(pct.values()):.1f}) "
              f"-> the predicted seg looks collapsed; inspect the head/backbone load before trusting run B")


if __name__ == '__main__':
    main()
