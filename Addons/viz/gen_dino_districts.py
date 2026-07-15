#!/usr/bin/env python
"""figures/fig_dino_districts.png -- "DINOv3 as the model sees it vs its computed districts".

Per frame (rows): [input frame | PCA-RGB of the DINOv3-B/16 patch tokens | the K=12 k-means
district map]. Runs the DEPLOYED gate path verbatim: Addons.motion.flow_track.load_dino +
dino_grid for the tokens, and the identical KMeans(n_groups, n_init=4, random_state=0) on
l2-normalised tokens that region_vote uses, so the district panel is exactly what the vote
gate computes -- not an offline approximation.

Box usage (needs torch2 + transformers + the DINOv3 HF dir):
  python Addons/viz/gen_dino_districts.py \
      --frames data/CRCD/C1_001/video_frames/<f310>l.png data/CRCD/C1_001/video_frames/<f115>l.png \
      --labels "instrument working" "camera moving" \
      --dino_dir /content/drive/MyDrive/Datasets/DiNO \
      --out /content/drive/MyDrive/Outputs/fig_dino_districts.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def pca_rgb(tokens):
    """[gh,gw,C] tokens -> PCA 3-comp mapped to RGB, per-channel robust-normalised."""
    gh, gw, C = tokens.shape
    X = tokens.reshape(-1, C).astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:3].T                                       # [N,3]
    lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
    P = np.clip((P - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
    return (P.reshape(gh, gw, 3) * 255).astype(np.uint8)


# 12 visually-distinct district colours (BGR)
PALETTE = np.array([
    (60, 76, 231), (43, 160, 43), (180, 119, 31), (14, 127, 255),
    (194, 119, 227), (75, 86, 140), (34, 189, 188), (207, 190, 23),
    (141, 160, 44), (232, 199, 174), (138, 223, 152), (150, 152, 255),
], np.uint8)


def district_panel(frame_bgr, lab_full, n_groups, alpha=0.55):
    """Colour each district, alpha-blend on the frame, draw district boundaries."""
    color = PALETTE[lab_full % len(PALETTE)]
    out = cv2.addWeighted(frame_bgr, 1 - alpha, color, alpha, 0)
    edges = np.zeros(lab_full.shape, np.uint8)
    edges[:-1, :] |= (lab_full[:-1, :] != lab_full[1:, :])
    edges[:, :-1] |= (lab_full[:, :-1] != lab_full[:, 1:])
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    out[edges > 0] = (255, 255, 255)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', nargs='+', required=True, help='input frames (rectified left pngs)')
    ap.add_argument('--labels', nargs='*', default=None, help='row label per frame')
    ap.add_argument('--backbone', default='dinov3_hf')
    ap.add_argument('--dino_dir', default=None, help='HF DINOv3 dir (for dinov3_hf)')
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--panel_w', type=int, default=480)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    from sklearn.cluster import KMeans
    from Addons.motion.flow_track import load_dino, dino_grid

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino = load_dino(device, backbone=args.backbone, v3_dir=args.dino_dir)

    rows = []
    for fp in args.frames:
        bgr = cv2.imread(fp)
        assert bgr is not None, fp
        H, W = bgr.shape[:2]
        g = dino_grid(bgr, dino, device)                    # [gh,gw,C] float32
        gh, gw, C = g.shape

        # deployed clustering, verbatim from region_vote
        X = g.reshape(-1, C)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        lab = KMeans(args.n_groups, n_init=4, random_state=args.seed).fit_predict(X)
        lab = lab.reshape(gh, gw).astype(np.uint8)
        lab_full = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)

        pca = cv2.resize(cv2.cvtColor(pca_rgb(g), cv2.COLOR_RGB2BGR), (W, H),
                         interpolation=cv2.INTER_NEAREST)
        panels = [bgr, pca, district_panel(bgr, lab_full, args.n_groups)]
        ph = int(args.panel_w * H / W)
        rows.append([cv2.resize(p, (args.panel_w, ph)) for p in panels])

    # compose with headers + row labels (PIL for text quality)
    from PIL import Image, ImageDraw, ImageFont

    def get_font(size, bold=False):
        cands = ['/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else ''),
                 r'C:\Windows\Fonts\arial%s.ttf' % ('bd' if bold else '')]
        try:  # matplotlib ships DejaVu -- present wherever the eval stack is
            from matplotlib import font_manager
            cands.insert(0, font_manager.findfont('DejaVu Sans' + (':bold' if bold else '')))
        except Exception:
            pass
        for p in cands:
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
        return ImageFont.load_default()

    font_b, font = get_font(18, bold=True), get_font(15)
    HEADS = ['input frame', 'DINOv3 patch tokens (PCA)', f'k-means districts (K = {args.n_groups})']
    GAP, PADT, PADL = 8, 46, (150 if args.labels else 10)
    pw, ph = args.panel_w, rows[0][0].shape[0]
    Wc = PADL + 3 * (pw + GAP) + 4
    Hc = PADT + len(rows) * (ph + GAP) + 4
    img = Image.new('RGB', (Wc, Hc), 'white')
    d = ImageDraw.Draw(img)
    for c, h in enumerate(HEADS):
        d.text((PADL + c * (pw + GAP) + pw // 2, PADT - 22), h, font=font_b,
               fill=(32, 33, 36), anchor='mm')
    for r, row in enumerate(rows):
        y = PADT + r * (ph + GAP)
        if args.labels and r < len(args.labels):
            d.text((PADL - 12, y + ph // 2), args.labels[r], font=font,
                   fill=(60, 60, 60), anchor='rm')
        for c, p in enumerate(row):
            img.paste(Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)),
                      (PADL + c * (pw + GAP), y))
    img.save(args.out)
    print('wrote', args.out, img.size)


if __name__ == '__main__':
    main()
