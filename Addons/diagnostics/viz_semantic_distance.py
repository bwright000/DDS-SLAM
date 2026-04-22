"""
Visualise the semantic distance field pipeline.

Shows the 4 stages that produce L_m's target:

  [1] raw mask (0 = instrument, 255 = background on StereoMIS)
  [2] Canny edges   (class boundaries)
  [3] inverted edges + distance transform (pixel -> nearest boundary distance, mm)
  [4] final target m(p) = exp(-dist / 10)

Picks 2 rows of frames: 3 binary-mask frames (instrument visible) and 3 uniform
(85% of StereoMIS frames — flat = no edges = near-zero target everywhere).

Usage:
  python Addons/diagnostics/viz_semantic_distance.py \\
      --mask_dir F:/Datasets/StereoMIS/StereoMIS/P2_1/masks \\
      --rgb_dir  F:/Datasets/StereoMIS/StereoMIS/P2_1/video_frames \\
      --out     ../Stereo-MIS\ Base/Results/Debug/semantic_distance_viz.png
"""
import argparse
import glob
import os

import cv2
import numpy as np


def compute_pipeline(mask_bgr):
    """Reproduce dataset.py:compute_edge_semantic stage by stage."""
    # stage 1: raw mask (input)
    m = mask_bgr.copy()

    # stage 2: Canny
    edges = cv2.Canny(m, 1, 1)

    # stage 3: invert + distance transform
    inverted = np.where(edges == 255, 0, 1).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 0, dstType=cv2.CV_32F)

    # stage 4: exp(-d/10) target field
    target = np.exp(-dist / 10.0)

    return m, edges, dist, target


def to_uint8_viz(arr, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    arr = arr.astype(np.float32)
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()
    span = max(vmax - vmin, 1e-8)
    norm = np.clip((arr - vmin) / span, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if u8.ndim == 2:
        u8 = cv2.applyColorMap(u8, colormap)
    return u8


def label_panel(img, text, color=(255, 255, 255)):
    out = img.copy()
    bar_h = 28
    out[:bar_h] = (out[:bar_h].astype(np.float32) * 0.3).astype(np.uint8)
    cv2.putText(out, text, (8, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def pick_frames(masks, n_binary=3, n_uniform=3):
    """Pick masks with variety: some binary (instrument visible), some uniform."""
    binaries = []
    uniforms = []
    # Sample across the whole list
    for p in masks[::max(1, len(masks)//60)]:
        m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        n_unique = len(np.unique(m))
        if n_unique >= 2 and len(binaries) < n_binary:
            binaries.append(p)
        elif n_unique == 1 and len(uniforms) < n_uniform:
            uniforms.append(p)
        if len(binaries) >= n_binary and len(uniforms) >= n_uniform:
            break
    return binaries, uniforms


def frame_num(path):
    import re
    n = re.sub(r'[^0-9]', '', os.path.splitext(os.path.basename(path))[0])
    return int(n) if n else -1


def rgb_for_mask(mask_path, rgb_dir):
    """Find the matching RGB frame for a mask (e.g., mask 4467 -> rgb 4467l.png)."""
    n = frame_num(mask_path)
    candidate = os.path.join(rgb_dir, f'{n:06d}l.png')
    if os.path.isfile(candidate):
        return cv2.imread(candidate)
    return None


def render_row(mask_path, rgb_dir, panel_size=(256, 320)):
    m_bgr = cv2.imread(mask_path)
    if m_bgr is None:
        return None
    m, edges, dist, target = compute_pipeline(m_bgr)
    rgb = rgb_for_mask(mask_path, rgb_dir)
    fn = frame_num(mask_path)

    H, W = panel_size
    def resize(img):
        if img is None:
            return np.zeros((H, W, 3), np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, (W, H))

    # panels
    rgb_v  = label_panel(resize(rgb), f'RGB f{fn}' if rgb is not None else 'RGB (missing)')
    mask_v = label_panel(resize(m),   f'mask (unique={len(np.unique(m))})')
    edg_v  = label_panel(resize(edges), 'Canny edges')
    dist_v = label_panel(resize(to_uint8_viz(dist)), f'dist [{dist.min():.1f}, {dist.max():.1f}] px')
    tgt_v  = label_panel(resize(to_uint8_viz(target, vmin=0.0, vmax=1.0)),
                         f'target exp(-d/10) [{target.min():.3f}, {target.max():.3f}]')

    return np.hstack([rgb_v, mask_v, edg_v, dist_v, tgt_v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask_dir', required=True)
    ap.add_argument('--rgb_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_binary', type=int, default=3)
    ap.add_argument('--n_uniform', type=int, default=3)
    ap.add_argument('--panel_height', type=int, default=256)
    ap.add_argument('--panel_width', type=int, default=320)
    args = ap.parse_args()

    masks = sorted(glob.glob(os.path.join(args.mask_dir, '*.png')))[-2000:]
    print(f'found {len(masks)} back-2000 masks in {args.mask_dir}')

    binaries, uniforms = pick_frames(masks, args.n_binary, args.n_uniform)
    print(f'picked {len(binaries)} binary, {len(uniforms)} uniform')

    panel_size = (args.panel_height, args.panel_width)

    rows = []
    # Section header for binary
    rows.append(section_row('--- FRAMES WITH INSTRUMENT (binary masks, 14.6 % of StereoMIS) ---',
                            5 * panel_size[1], panel_size[0] // 3))
    for p in binaries:
        r = render_row(p, args.rgb_dir, panel_size)
        if r is not None:
            rows.append(r)

    rows.append(section_row('--- UNIFORM MASKS (85.4 % of StereoMIS — no instrument in view) ---',
                            5 * panel_size[1], panel_size[0] // 3))
    for p in uniforms:
        r = render_row(p, args.rgb_dir, panel_size)
        if r is not None:
            rows.append(r)

    if not rows:
        print('no rows rendered, aborting'); return

    canvas = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    cv2.imwrite(args.out, canvas)
    print(f'wrote {args.out}   shape={canvas.shape}')


def section_row(text, width, height):
    bar = np.full((height, width, 3), 30, np.uint8)
    cv2.putText(bar, text, (10, int(height * 0.7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1, cv2.LINE_AA)
    return bar


if __name__ == '__main__':
    main()
