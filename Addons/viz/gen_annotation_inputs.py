#!/usr/bin/env python
"""figures/fig_annotation_inputs.png -- the five raw left-camera input frames that
Figure~\\ref{fig:PromptingComparison} (fig_annotation_comparison.png) overlays with our
SAM 3-assisted masks and the CRCD-provided masks.

Same five frames, same order and same labels as that figure, shown unannotated so the
reader can see the source imagery the annotations are judged against. Frames are the
published CRCD snippets, named by their original episode frame index.

Local usage: python Addons/viz/gen_annotation_inputs.py
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = r'F:/Datasets/CRCD-Published'
FRAMES = [
    ('E·3/002', 5722, 'E_3/snippet_002'),
    ('E·3/003', 16777, 'E_3/snippet_003'),
    ('E·3/004', 23760, 'E_3/snippet_004'),
    ('F·3/002', 19443, 'F_3/snippet_002'),
    ('F·3/003', 20100, 'F_3/snippet_003'),
]
TW, TH = 384, 216


def font(sz, bold=False):
    for p in (r'C:\Windows\Fonts\arial' + ('bd' if bold else '') + '.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='figures/fig_annotation_inputs.png')
    args = ap.parse_args()

    GAP, PADT = 6, 22
    W = 5 * TW + 4 * GAP
    H = PADT + TH
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    for i, (label, idx, rel) in enumerate(FRAMES):
        p = f'{ROOT}/{rel}/rgb/frame_{idx:06d}.png'
        im = cv2.imread(p)
        assert im is not None, p
        im = cv2.resize(im, (TW, TH), interpolation=cv2.INTER_AREA)
        x = i * (TW + GAP)
        img.paste(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), (x, PADT))
        d.text((x + TW // 2, PADT // 2), f'{label}  frame {idx}', font=font(13, True),
               fill=(32, 33, 36), anchor='mm')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    print('wrote', args.out, img.size)


if __name__ == '__main__':
    main()
