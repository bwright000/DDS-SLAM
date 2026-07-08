#!/usr/bin/env python
"""WHAT DOES THE GATE ACTUALLY INGEST? Render the raw DINO patch-feature grids.

For each probe frame (tail/still/moving) and each featurizer:
  PCA-RGB   : the patch-token grid projected onto 3 principal components (fit JOINTLY over all
              probe frames per featurizer, so colors are comparable across frames) -> what
              structure the features encode, at TRUE patch resolution (nearest-upsampled blocks).
  simmap    : cosine similarity of every patch to one PROBE PATCH on the stapler (--probe_y/x,
              fractional coords) -> is the mover feature-distinct from the tissue it touches?
  stats     : mean cos-sim probe->mover-region vs probe->rest, and mover-vs-rest margin.
              Margin ~0 = the featurizer cannot tell worked tissue from resting tissue
              (clustering was never going to isolate it); margin >> 0 = separable.

Featurizers: v2base (deployed gate: DINOv2-S/14 reg, final layer, native res)
             v3ft_mid (LOSO DINOv3-B/16 fine-tune, block --mid_block, trained res)
             v3ft (same, final layer -- the class-collapsed one, for contrast)

Run:
  python Addons/seg/visualize_dino_ingest_20260709.py \
    --rgb_dir /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/rgb \
    --ckpt   /content/drive/MyDrive/Datasets/seg/loso_dinov3_b2/dinov2_crcd_C1_001_SWEEPONLY.pth \
    --dinov3 /content/drive/MyDrive/Datasets/DiNO \
    --out    /content/drive/MyDrive/Outputs/seg/dino_ingest_viz
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def load_frame(rgb_dir, offset, f):
    import cv2
    p = os.path.join(rgb_dir, f"frame_{offset + f:06d}.png")
    im = cv2.imread(p); assert im is not None, f"missing {p}"
    return im


def pca_rgb(grids):
    """grids: list of [gh,gw,C] -> list of [gh,gw,3] uint8, PCA fit jointly (comparable colors)."""
    X = np.concatenate([g.reshape(-1, g.shape[-1]) for g in grids], 0).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    mu = X.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    out = []
    for g in grids:
        Y = g.reshape(-1, g.shape[-1]).astype(np.float64)
        Y = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-8)
        P = (Y - mu) @ Vt[:3].T
        lo, hi = np.percentile(P, 2, axis=0), np.percentile(P, 98, axis=0)
        P = np.clip((P - lo) / (hi - lo + 1e-9), 0, 1)
        out.append((P.reshape(g.shape[0], g.shape[1], 3) * 255).astype(np.uint8))
    return out


def simmap(grid, py, px):
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    probe = X[int(py * gh) * gw + int(px * gw)]
    return (X @ probe).reshape(gh, gw)


def up(img, W=480, H=270):
    import cv2
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)


def main():
    import cv2, torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dinov3', required=True); ap.add_argument('--out', default='./dino_ingest_viz')
    ap.add_argument('--offset', type=int, default=1560)
    ap.add_argument('--frames', type=int, nargs='+', default=[310, 40, 115],
                    help='probe frames (default: tail-working, clean-still, camera-moving)')
    ap.add_argument('--probe_y', type=float, default=0.55, help='stapler patch, fraction of height')
    ap.add_argument('--probe_x', type=float, default=0.25, help='stapler patch, fraction of width')
    ap.add_argument('--mid_block', type=int, default=9)
    ap.add_argument('--v3_h', type=int, default=504); ap.add_argument('--v3_w', type=int, default=896)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from Addons.motion.flow_track import load_dino, dino_grid
    from Addons.seg.train_dinov2_crcd import load_dinov3_hf, DINO2SEG
    base = load_dino(device)
    bb, patch, nreg, embed = load_dinov3_hf(a.dinov3)
    ft = DINO2SEG(a.v3_h, a.v3_w, 4, bb, patch_size=patch, n_register=nreg, edge=0, dim=16,
                  train_blocks=0, embed=embed).to(device)
    sd = torch.load(a.ckpt, map_location='cpu')
    ft.load_state_dict(sd, strict=False); ft.eval()
    VSZ = (a.v3_h // patch * patch, a.v3_w // patch * patch)

    bb_pristine, _, _, _ = load_dinov3_hf(a.dinov3)          # RAW DINOv3, no fine-tune
    bb_pristine = bb_pristine.to(device).eval()

    def v3_grid(bgr, wrap, size, layer=None):
        """Patch grid from any HF-wrapped DINOv3 at an arbitrary /patch-snapped size."""
        im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        sh, sw = (size[0] // patch) * patch, (size[1] // patch) * patch
        im = cv2.resize(im, (sw, sh))
        mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
        t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(device)
        with torch.inference_mode():
            if layer is None:
                tok = wrap.model(pixel_values=t).last_hidden_state[:, 1 + nreg:, :]
            else:
                tok = wrap.model(pixel_values=t, output_hidden_states=True).hidden_states[layer][:, 1 + nreg:, :]
        return tok[0].cpu().numpy().reshape(sh // patch, sw // patch, -1).astype(np.float32)

    im_h, im_w = load_frame(a.rgb_dir, a.offset, a.frames[0]).shape[:2]
    ARMS = {'v2base(gate today)': lambda b: dino_grid(b, base, device),
            'v3RAW@504x896': lambda b: v3_grid(b, bb_pristine, VSZ),
            f'v3RAW@{im_h}x{im_w}': lambda b: v3_grid(b, bb_pristine, (im_h, im_w)),
            f'v3ft_mid(block{a.mid_block})': lambda b: v3_grid(b, ft.backbone, VSZ, a.mid_block),
            'v3ft(final)': lambda b: v3_grid(b, ft.backbone, VSZ)}
    NAMES = {310: 'TAIL-working', 40: 'clean-STILL', 115: 'camera-MOVING'}

    frames = [load_frame(a.rgb_dir, a.offset, f) for f in a.frames]
    grids = {k: [fn(b) for b in frames] for k, fn in ARMS.items()}
    print("grid shapes:", {k: g[0].shape for k, g in grids.items()})

    # mover-region mask at each grid res, from the SMOOTHED tail diff (for the margin stat)
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(load_frame(a.rgb_dir, a.offset, a.frames[0] + 2), cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = cv2.GaussianBlur(np.abs(g1 - g0), (31, 31), 0)

    rows = []
    for k, gl in grids.items():
        prgb = pca_rgb(gl)
        row = []
        for i, f in enumerate(a.frames):
            fr = up(frames[i])
            if i == 0:                                            # mark the probe patch on the rgb
                cv2.circle(fr, (int(a.probe_x * 480), int(a.probe_y * 270)), 7, (0, 255, 255), 2)
            p = up(prgb[i][:, :, ::-1])
            cv2.putText(p, f"{k} PCA {NAMES.get(f, f)}", (6, 20), 0, 0.55, (255, 255, 255), 2)
            s = simmap(gl[i], a.probe_y, a.probe_x)
            sh = cv2.applyColorMap((np.clip((s + 1) / 2, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            sh = up(sh)
            cv2.putText(sh, "cos-sim to probe", (6, 20), 0, 0.55, (255, 255, 255), 2)
            row += ([fr, p, sh] if i == 0 else [p, sh])
            if i == 0:                                            # separability stat on the tail frame
                dg = cv2.resize(diff, (gl[i].shape[1], gl[i].shape[0]), interpolation=cv2.INTER_AREA)
                mover = dg >= np.percentile(dg, 85)
                inm, outm = float(s[mover].mean()), float(s[~mover].mean())
                print(f"{k:<24} tail probe-sim: mover {inm:+.3f} vs rest {outm:+.3f} margin {inm - outm:+.3f}")
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    outp = os.path.join(a.out, 'ingest_sheet.jpg')
    cv2.imwrite(outp, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[sheet] {outp}  (rows = featurizers; cols = rgb+probe, then per-frame [PCA-RGB, sim-to-probe])")


if __name__ == '__main__':
    main()
