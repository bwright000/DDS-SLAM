#!/usr/bin/env python
"""figures/fig_render_compare.png -- three C_2/001 frames rendered by every render-capable
system, 3 rows (frames) x 7 columns (GT + 6 systems). PERSEUS is tracking-only and excluded.

Frame selection: at every SGS keyframe index (the sparsest renderer), compute each model's
PSNR against the GT frame; score the frame by the MINIMUM PSNR across models (frames every
system renders reasonably), and pick the best-scoring frame from each third of the sequence
so the rows span early/mid/late. DID/base renders are cropped from the Rendered-RGB panel of
their panels.mp4 (grid cell row 0 col 1; burned-in label strip skipped) and restored to 16:9.

Local usage: python Addons/viz/gen_render_compare.py --out figures/fig_render_compare.png
"""
import argparse
import os
import re

import cv2
import numpy as np

B = r'F:/Datasets/Benchmarking-20260702T131352Z-4-003/Benchmarking'
SRC = {
    'did_video': r'C:/Users/benli/OneDrive/Desktop/Results/drive-download-20260711T053215Z-2-001/rect_bestbase_bdds_final_20260709/C2_001_calm_v3gate_s0/panels.mp4',
    'base_video': B + r'/DDS-SLAM Bench/C2_001_base_s0/panels.mp4',
    'sni': r'F:/Datasets/Resi;ts/SNI_fresh_20260714-20260715T072207Z-1-003/SNI_fresh_20260714/C2_001_noconst/render',
    'sgs': B + r'/SGS-SLAM_CRCD_bench5/C2_001',
    'semgauss': B + r'/SemGauss-SLAM_bench_20260704-20260712T170730Z-2-001/SemGauss-SLAM_bench_20260704/C2_001',
    'semsup': B + r'/SemanticSuPer_crcd_20260703-20260704T145257Z-3-002/SemanticSuPer_crcd_20260703/C2_001/C2_001',
}
TW, TH = 300, 169
CMP = (320, 180)   # PSNR comparison resolution


def video_tile(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    assert ok, (path, idx)
    return fr[30:360, 480:960]


def model_tiles(idx):
    """dict name->BGR image (native sizes) for frame idx."""
    return {
        'DID-SLAM (ours)': video_tile(SRC['did_video'], idx),
        'DDS-SLAM (base)': video_tile(SRC['base_video'], idx),
        'SNI-SLAM': cv2.imread(f"{SRC['sni']}/{idx}.jpg"),
        'SGS-SLAM': cv2.imread(f"{SRC['sgs']}/{idx}.jpg"),
        'SemGauss-SLAM': cv2.imread(f"{SRC['semgauss']}/{idx}.jpg"),
        'Semantic-SuPer': cv2.imread(f"{SRC['semsup']}/{idx}.jpg"),
    }


def psnr(a, b):
    a = cv2.resize(a, CMP).astype(np.float32) / 255.0
    b = cv2.resize(b, CMP).astype(np.float32) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else -10.0 * np.log10(mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_render_compare.png')
    args = ap.parse_args()

    sgs_idx = sorted(int(m.group(1)) for f in os.listdir(SRC['sgs'])
                     if (m := re.match(r'^(\d+)\.jpg$', f)))
    sgs_idx = [i for i in sgs_idx if i >= 60]   # skip the trivial fresh-map opening
    scores = {}
    for i in sgs_idx:
        gtp = f"{SRC['semgauss']}/{i}_gt.png"
        gt = cv2.imread(gtp)
        if gt is None:
            continue
        try:
            tiles = model_tiles(i)
        except AssertionError:
            continue
        if any(v is None for v in tiles.values()):
            continue
        scores[i] = min(psnr(v, gt) for v in tiles.values())
    a, b = min(scores), max(scores) + 1
    third = (b - a) // 3
    picks = []
    for lo, hi in [(a, a + third), (a + third, a + 2 * third), (a + 2 * third, b)]:
        cand = {i: s for i, s in scores.items() if lo <= i < hi}
        picks.append(max(cand, key=cand.get))
    print('picked frames (best min-PSNR per third):', picks,
          '| scores:', [round(scores[i], 2) for i in picks])

    COLS = ['ground truth', 'DID-SLAM (ours)', 'DDS-SLAM (base)', 'SNI-SLAM',
            'SGS-SLAM', 'SemGauss-SLAM', 'Semantic-SuPer']

    from PIL import Image, ImageDraw, ImageFont
    F = lambda sz, b=False: ImageFont.truetype(
        r'C:\Windows\Fonts\arial' + ('bd' if b else '') + '.ttf', sz)
    GAP, PADT, PADL = 6, 30, 86
    W = PADL + 7 * (TW + GAP) + GAP
    H = PADT + 3 * (TH + GAP) + GAP
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    for c, h in enumerate(COLS):
        d.text((PADL + c * (TW + GAP) + TW // 2, PADT // 2 + 2), h,
               font=F(13, True), fill=(32, 33, 36), anchor='mm')
    for r, idx in enumerate(picks):
        y = PADT + r * (TH + GAP)
        d.text((PADL - 10, y + TH // 2), f'frame\n{idx}', font=F(13),
               fill=(60, 60, 60), anchor='rm', align='right')
        row = [cv2.imread(f"{SRC['semgauss']}/{idx}_gt.png")] + list(model_tiles(idx).values())
        for c, tile in enumerate(row):
            tile = cv2.resize(tile, (TW, TH))
            img.paste(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                      (PADL + c * (TW + GAP), y))
    img.save(args.out)
    print('wrote', args.out, img.size)


if __name__ == '__main__':
    main()
