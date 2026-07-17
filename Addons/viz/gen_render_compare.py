#!/usr/bin/env python
"""figures/fig_render_compare.png -- three C_2/001 frames rendered by every render-capable
system, laid out vertically: 7 rows (GT + 6 systems) x 3 columns (frames). PERSEUS is
tracking-only and excluded.

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
    stats = {}   # idx -> (did_psnr, base_psnr, min_all)
    for i in sgs_idx:
        gt = cv2.imread(f"{SRC['semgauss']}/{i}_gt.png")
        if gt is None:
            continue
        try:
            tiles = model_tiles(i)
        except AssertionError:
            continue
        if any(v is None for v in tiles.values()):
            continue
        ps = {k: psnr(v, gt) for k, v in tiles.items()}
        stats[i] = (ps['DID-SLAM (ours)'], ps['DDS-SLAM (base)'], min(ps.values()))

    # prefer frames where DID beats the base on PSNR (viable for everyone else);
    # fall back to the highest joint minimum. Keep picks >= 40 frames apart.
    pref = sorted((i for i, (d, b, m) in stats.items() if d > b + 0.1 and m >= 8.0),
                  key=lambda i: stats[i][0] - stats[i][1], reverse=True)
    fallback = sorted(stats, key=lambda i: stats[i][2], reverse=True)
    picks = []
    for pool in (pref, fallback):
        for i in pool:
            if len(picks) == 3:
                break
            if all(abs(i - p) >= 40 for p in picks):
                picks.append(i)
    picks = sorted(picks)
    print('picked:', picks, '| (did, base, min):', [tuple(round(x, 2) for x in stats[i]) for i in picks])

    # rows = systems (GT first), columns = the three picked frames
    ROWS = ['ground truth', 'DID-SLAM (ours)', 'DDS-SLAM (base)', 'SNI-SLAM',
            'SGS-SLAM', 'SemGauss-SLAM', 'Semantic-SuPer']

    from PIL import Image, ImageDraw, ImageFont
    F = lambda sz, b=False: ImageFont.truetype(
        r'C:\Windows\Fonts\arial' + ('bd' if b else '') + '.ttf', sz)
    GAP, PADT, PADL = 6, 30, 118
    W = PADL + 3 * (TW + GAP) + GAP
    H = PADT + 7 * (TH + GAP) + GAP
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    # one column per frame; fetch each frame's stack once
    stacks = []
    for idx in picks:
        stacks.append([cv2.imread(f"{SRC['semgauss']}/{idx}_gt.png")]
                      + list(model_tiles(idx).values()))
    for c, idx in enumerate(picks):
        d.text((PADL + c * (TW + GAP) + TW // 2, PADT // 2 + 2), f'frame {idx}',
               font=F(13, True), fill=(32, 33, 36), anchor='mm')
    for r, name in enumerate(ROWS):
        y = PADT + r * (TH + GAP)
        d.text((PADL - 10, y + TH // 2), name, font=F(13, r == 1),
               fill=(32, 33, 36) if r == 1 else (60, 60, 60), anchor='rm')
        for c in range(len(picks)):
            tile = cv2.resize(stacks[c][r], (TW, TH))
            img.paste(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)),
                      (PADL + c * (TW + GAP), y))
    img.save(args.out)
    print('wrote', args.out, img.size)


if __name__ == '__main__':
    main()
