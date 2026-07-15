#!/usr/bin/env python
"""figures/fig_render_compare.png -- one C_2/001 frame rendered by every benchmarked system.

Tiles: GT + DID-SLAM + DDS-SLAM(base) + SNI-SLAM + SGS-SLAM + SemGauss-SLAM + Semantic-SuPer
(PERSEUS renders nothing; noted in the empty cell). DID/base renders are not shipped as frames
in their drops, so they are cropped from the Rendered-RGB panel of panels.mp4 (always grid cell
row 0, col 1 at 480x360) and restored to 16:9. Frame index is chosen from SGS-SLAM's keyframe
set (the sparsest renderer) nearest the requested target.

Local usage:
  python Addons/viz/gen_render_compare.py --target 330 --out figures/fig_render_compare.png
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
TW, TH = 480, 270


def video_render_tile(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    assert ok, (path, idx)
    tile = fr[30:360, 480:960]                # Rendered RGB = grid cell (0,1); skip the burned-in label strip
    return cv2.resize(tile, (TW, TH))         # restore 16:9


def img_tile(path):
    im = cv2.imread(path)
    assert im is not None, path
    return cv2.resize(im, (TW, TH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', type=int, default=330)
    ap.add_argument('--out', default='figures/fig_render_compare.png')
    args = ap.parse_args()

    sgs_idx = sorted(int(m.group(1)) for f in os.listdir(SRC['sgs'])
                     if (m := re.match(r'^(\d+)\.jpg$', f)))
    idx = min(sgs_idx, key=lambda i: abs(i - args.target))
    print(f'frame index: {idx} (nearest SGS keyframe to {args.target})')

    tiles = [
        ('ground truth (frame %d)' % idx, img_tile(f"{SRC['semgauss']}/{idx}_gt.png")),
        ('DID-SLAM (ours)', video_render_tile(SRC['did_video'], idx)),
        ('DDS-SLAM (base)', video_render_tile(SRC['base_video'], idx)),
        ('SNI-SLAM', img_tile(f"{SRC['sni']}/{idx}.jpg")),
        ('SGS-SLAM', img_tile(f"{SRC['sgs']}/{idx}.jpg")),
        ('SemGauss-SLAM', img_tile(f"{SRC['semgauss']}/{idx}.jpg")),
        ('Semantic-SuPer', img_tile(f"{SRC['semsup']}/{idx}.jpg")),
        ('PERSEUS', None),                    # tracking-only
    ]

    from PIL import Image, ImageDraw, ImageFont
    F = lambda sz, b=False: ImageFont.truetype(
        r'C:\Windows\Fonts\arial' + ('bd' if b else '') + '.ttf', sz)
    GAP, PADT = 8, 34
    COLS, ROWS = 4, 2
    W = COLS * (TW + GAP) + GAP
    H = ROWS * (TH + PADT + GAP) + GAP
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)
    for i, (label, tile) in enumerate(tiles):
        r, c = divmod(i, COLS)
        x = GAP + c * (TW + GAP)
        y = GAP + r * (TH + PADT + GAP)
        d.text((x + TW // 2, y + PADT // 2), label, font=F(16, True),
               fill=(32, 33, 36), anchor='mm')
        if tile is None:
            d.rectangle([x, y + PADT, x + TW - 1, y + PADT + TH - 1],
                        outline=(200, 200, 200), width=1)
            d.text((x + TW // 2, y + PADT + TH // 2), 'tracking-only\n(no reconstruction)',
                   font=F(15), fill=(150, 150, 150), anchor='mm', align='center')
        else:
            img.paste(Image.fromarray(cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)), (x, y + PADT))
    img.save(args.out)
    print('wrote', args.out, img.size)


if __name__ == '__main__':
    main()
