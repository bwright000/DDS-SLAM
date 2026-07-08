#!/usr/bin/env python
"""TRACK-2 VALIDATION v2 (fair matrix + VISUAL contact sheets).

v1's comparison under-served the fine-tune in four ways (user challenge, 2026-07-09):
resolution (v3 ran at ~720x1280, was trained at 504x896), layer (final = most class-collapsed),
no v3-base control, and a district-size confound in n80/IoU. v2 fixes all four and draws
what it measures so the districtings can be eyeballed.

ARMS (all through the IDENTICAL L2-norm -> KMeans(12) -> metrics pipeline):
  v2base   : deployed gate featurizer (natural-image DINOv2-S/14 reg, final layer, native /14 res)
  v3ft     : LOSO fine-tuned DINOv3-B/16, FINAL layer, AT ITS TRAINED RES (--v3_h/--v3_w)
  v3ft_mid : same model, MID layer (--mid_block, default 9 -- untouched by the 2-block fine-tune)
  v3base   : plain HF DINOv3 (no fine-tune), final layer, trained res  [the control]
  head16   : the seg head's 16-d compressed feature (the SNI/SemGauss deploy contract)

VISUAL: per segment a contact sheet PNG -- rows = sample frames, cols =
  [ rgb + mover-blob contour | each arm's districts (12 colors, blob contour on top) | seg pred ]
Saved to --out; drop them back to me and we judge together.

Run (box, torch2; ~10-15 min GPU or slower CPU fallback):
  python Addons/seg/validate_dino_gate_tail_v2_20260709.py \
    --rgb_dir /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/rgb \
    --ckpt   /content/drive/MyDrive/Datasets/seg/loso_dinov3_b2/dinov2_crcd_C1_001_SWEEPONLY.pth \
    --dinov3 /content/drive/MyDrive/Datasets/DiNO \
    --out    /content/drive/MyDrive/Outputs/seg/dino_gate_viz
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
                    [0, 128, 128], [170, 110, 40]], np.uint8)          # 12 districts
SEG_COLORS = np.array([[40, 40, 40], [200, 80, 80], [80, 200, 80], [80, 120, 255]], np.uint8)  # bg,Liver,GB,Tool


def load_frame(rgb_dir, offset, f):
    import cv2
    p = os.path.join(rgb_dir, f"frame_{offset + f:06d}.png")
    im = cv2.imread(p); assert im is not None, f"missing {p}"
    return im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def norm_input(bgr, gh_mult, device, size=None):
    """BGR -> ImageNet-normed tensor; size=(H,W) to force the trained res, else snap to multiple."""
    import cv2, torch
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if size is None:
        H, W = im.shape[:2]
        size = ((H // gh_mult) * gh_mult, (W // gh_mult) * gh_mult)
    im = cv2.resize(im, (size[1], size[0]))
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    return torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(device)


def districts(grid, n_groups=12, seed=0):
    from sklearn.cluster import KMeans
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C).astype(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw)


def metrics(diff, lab, n_groups):
    import cv2
    d = cv2.resize(diff, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_AREA)
    blob = d >= np.percentile(d, 90)
    mass = np.array([float((blob & (lab == g)).sum()) for g in range(n_groups)])
    order = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[order]) / max(1.0, mass.sum())
    n80 = int(np.searchsorted(cum, 0.8) + 1)
    top2 = (lab == order[0]) | (lab == order[1])
    iou = float((top2 & blob).sum()) / max(1.0, float((top2 | blob).sum()))
    dmean = np.array([d[lab == g].mean() if (lab == g).any() else np.inf for g in range(n_groups)])
    q10 = float(np.percentile(dmean[np.isfinite(dmean)], 10))
    return n80, iou, q10


def draw_districts(bgr, lab, blob_full):
    """District colors alpha-blended over the frame + blob contour."""
    import cv2
    H, W = bgr.shape[:2]
    lab_up = cv2.resize(lab.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    col = PALETTE[lab_up % len(PALETTE)][:, :, ::-1]                    # BGR
    out = cv2.addWeighted(bgr, 0.45, col, 0.55, 0)
    edges = cv2.Canny((lab_up * 20).astype(np.uint8), 1, 1)
    out[edges > 0] = (255, 255, 255)
    cnts, _ = cv2.findContours(blob_full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (0, 0, 0), 3)
    cv2.drawContours(out, cnts, -1, (0, 255, 255), 1)
    return out


def main():
    import cv2, torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dinov3', required=True); ap.add_argument('--out', default='./dino_gate_viz')
    ap.add_argument('--offset', type=int, default=1560); ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--v3_h', type=int, default=504); ap.add_argument('--v3_w', type=int, default=896)
    ap.add_argument('--mid_block', type=int, default=9)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from Addons.motion.flow_track import load_dino, dino_grid
    from Addons.seg.train_dinov2_crcd import load_dinov3_hf, DINO2SEG
    base = load_dino(device)
    im0, _ = load_frame(a.rgb_dir, a.offset, 0); H, W = im0.shape[:2]
    bb, patch, nreg, embed = load_dinov3_hf(a.dinov3)
    ft = DINO2SEG(a.v3_h, a.v3_w, 4, bb, patch_size=patch, n_register=nreg, edge=0, dim=16,
                  train_blocks=0, embed=embed).to(device)
    sd = torch.load(a.ckpt, map_location='cpu')
    missing, unexpected = ft.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith('backbone.model.')]
    assert not bad, f"head keys missing: {bad[:4]}"
    ft.eval()
    bb2, _, _, _ = load_dinov3_hf(a.dinov3)          # pristine control (no fine-tune)
    bb2 = bb2.to(device).eval()
    print(f"[arms] v2base(/14 native) | v3ft(final@{a.v3_h}x{a.v3_w}) | v3ft_mid(block{a.mid_block}) "
          f"| v3base(final) | head16")

    VSZ = (a.v3_h // patch * patch, a.v3_w // patch * patch)
    gh3, gw3 = VSZ[0] // patch, VSZ[1] // patch

    def v3_tokens(model_bb, t, layer=None):
        with torch.inference_mode():
            if layer is None:
                out = model_bb.model(pixel_values=t.float())
                tok = out.last_hidden_state[:, 1 + nreg:, :]
            else:
                out = model_bb.model(pixel_values=t.float(), output_hidden_states=True)
                tok = out.hidden_states[layer][:, 1 + nreg:, :]
        return tok[0].cpu().numpy().reshape(gh3, gw3, -1).astype(np.float32)

    def head16_grid(t):
        with torch.inference_mode():
            tok = ft._patch_tokens(t)                      # [1,N,768]
            x = tok.reshape(1, embed, gh3, gw3)
            x = ft.segmentation_conv[0](x)                 # Upsample x4
            x = ft.segmentation_conv[1](x)                 # conv 768->16
        g = x[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return g[::2, ::2]                                 # 2x sub-sample for KMeans cost

    def seg_pred(t):
        with torch.inference_mode():
            return ft(t).argmax(1)[0].cpu().numpy().astype(np.uint8)

    SEGS = {'TAIL': range(283, 356, a.stride), 'STILL': range(21, 56, a.stride), 'MOVING': range(101, 136, a.stride)}
    ARMS = ['v2base', 'v3ft', 'v3ft_mid', 'v3base', 'head16']
    table = {}
    for seg_name, frames in SEGS.items():
        acc = {k: [] for k in ARMS}; panels = []
        for i, f in enumerate(frames):
            bgr, g0 = load_frame(a.rgb_dir, a.offset, f)
            _, g1 = load_frame(a.rgb_dir, a.offset, f + 2)
            diff = np.abs(g1 - g0)
            blob_full = diff >= np.percentile(diff, 90)
            t3 = norm_input(bgr, patch, device, size=VSZ)
            labs = {'v2base': districts(dino_grid(bgr, base, device), a.n_groups),
                    'v3ft': districts(v3_tokens(ft.backbone, t3), a.n_groups),
                    'v3ft_mid': districts(v3_tokens(ft.backbone, t3, layer=a.mid_block), a.n_groups),
                    'v3base': districts(v3_tokens(bb2, t3), a.n_groups),
                    'head16': districts(head16_grid(t3), a.n_groups)}
            for k in ARMS:
                acc[k].append(metrics(diff, labs[k], a.n_groups))
            if i % 3 == 0 and len(panels) < 4:             # 4 sample rows per sheet
                row = [cv2.resize(draw_districts(bgr, np.zeros((1, 1), int), blob_full), (480, 270))]
                cv2.putText(row[0], f"f{f} rgb+blob", (8, 24), 0, 0.7, (255, 255, 255), 2)
                for k in ARMS:
                    p = cv2.resize(draw_districts(bgr, labs[k], blob_full), (480, 270))
                    cv2.putText(p, k, (8, 24), 0, 0.7, (255, 255, 255), 2)
                    row.append(p)
                sg = seg_pred(t3)
                sgc = SEG_COLORS[sg][:, :, ::-1]
                sgc = cv2.resize(sgc, (480, 270), interpolation=cv2.INTER_NEAREST)
                sgv = cv2.addWeighted(cv2.resize(bgr, (480, 270)), 0.5, sgc, 0.5, 0)
                cv2.putText(sgv, "seg pred (blue=Tool)", (8, 24), 0, 0.7, (255, 255, 255), 2)
                row.append(sgv)
                panels.append(np.concatenate(row, axis=1))
        sheet = np.concatenate(panels, axis=0)
        cv2.imwrite(os.path.join(a.out, f"sheet_{seg_name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        table[seg_name] = {k: np.array(acc[k]).mean(0) for k in ARMS}
        print(f"[sheet] {os.path.join(a.out, f'sheet_{seg_name}.jpg')}")

    print(f"\n{'arm':<10}" + "".join(f"{s+'-n80':>11}{s+'-IoU':>10}{s+'-q10':>10}" for s in SEGS))
    for k in ARMS:
        print(f"{k:<10}" + "".join(f"{table[s][k][0]:>11.1f}{table[s][k][1]:>10.2f}{table[s][k][2]:>10.2f}" for s in SEGS))
    for k in ARMS:
        sep = table['MOVING'][k][2] / max(table['TAIL'][k][2], 1e-6)
        print(f"separation(moving/tail) {k}: {sep:.2f}")


if __name__ == '__main__':
    main()
