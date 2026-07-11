#!/usr/bin/env python
"""figures/dino_v2_v3_compare.png -- "what each backbone saw": PCA-RGB of the patch tokens and
probe-patch similarity, DINOv2-S/14 vs base DINOv3-B/16 (adopted) vs the fine-tuned DINOv3 final
layer (the neural-collapse contrast), on the working-instrument frame and the camera-moving frame.
Crops from the box-generated ingest sheet (visualize_dino_ingest_20260709.py output).
Usage: python Addons/viz/gen_dino_compare.py <path-to-ingest_sheet.jpg>
Sheet layout: 5 rows x [rgb, (pca,sim) x 3 frames], tiles 480x270,
rows = [v2base, v3RAW@504x896, v3RAW@720x1280, v3ft_mid(block9), v3ft(final)],
frames = [f310 tail-working, f40 clean-still, f115 camera-moving]."""
import sys, os
from PIL import Image, ImageDraw, ImageFont

FIG = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures"
TW, THT = 480, 270
ROWS = {'v2base': 0, 'v3raw504': 1, 'v3raw720': 2, 'v3ft_mid': 3, 'v3ft': 4}
# col index of each tile in a row: rgb=0, then (pca,sim) per frame
COL = {'rgb': 0, 'pca_tail': 1, 'sim_tail': 2, 'pca_still': 3, 'sim_still': 4,
       'pca_move': 5, 'sim_move': 6}

def tile(sheet, row, col):
    t = sheet.crop((col * TW, row * THT, (col + 1) * TW, (row + 1) * THT))
    if col > 0:                       # trim the burned-in dev label strip on feature tiles
        t = t.crop((0, 30, TW, THT))
    return t

def main():
    sheet = Image.open(sys.argv[1])
    PICK_ROWS = [('DINOv2-S/14 (tested)', 'v2base'),
                 ('base DINOv3-B/16 (adopted)', 'v3raw720'),
                 ('fine-tuned DINOv3, final layer', 'v3ft')]
    PICK_COLS = [('input frame', 'rgb'), ('PCA of patch tokens\n(instrument working)', 'pca_tail'),
                 ('similarity to instrument patch', 'sim_tail'),
                 ('PCA (camera moving)', 'pca_move')]
    F = lambda sz, b=False: ImageFont.truetype(r"C:\Windows\Fonts\arial" + ("bd" if b else "") + ".ttf", sz)
    PADL, PADT, GAP = 230, 64, 8
    OW, OH = 400, 225
    W = PADL + len(PICK_COLS) * (OW + GAP) + 12
    H = PADT + len(PICK_ROWS) * (OH + GAP) + 16
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    for c, (h, _) in enumerate(PICK_COLS):
        d.multiline_text((PADL + c * (OW + GAP) + OW // 2, PADT - 34), h, font=F(14, True),
                         fill=(32, 33, 36), anchor="mm", align="center")
    for r, (lab, key) in enumerate(PICK_ROWS):
        y = PADT + r * (OH + GAP)
        for c, (_, ck) in enumerate(PICK_COLS):
            t = tile(sheet, ROWS[key], COL[ck]).resize((OW, OH))
            img.paste(t, (PADL + c * (OW + GAP), y))
        for i, ln in enumerate(lab.split(', ')):
            d.text((14, y + OH // 2 - 12 + i * 22), ln, font=F(14, True), fill=(32, 33, 36))
    out = os.path.join(FIG, "dino_v2_v3_compare.png")
    img.save(out)
    print("wrote", out, img.size)

if __name__ == '__main__':
    main()
