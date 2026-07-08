#!/usr/bin/env python
"""TRACK-2 VALIDATION (2026-07-08): would surgical-LOSO DINOv3 features fix the vote gate's
"working-instrument still" blindness? Decides BEFORE any SLAM integration is built.

The C1 tail failure (frames 280-359, GT bit-still): stapler works tissue -> image change 2x the
camera-moving segment; the gate's quiet-decile saturates (q10 med 6.9px) because the 12 KMeans
districts built from NATURAL-IMAGE DINOv2-S/14 features smear the mover across districts.
Hypothesis: districts from OUR LOSO-fine-tuned DINOv3-B/16 (C1 never trained; C1 seg mIoU 0.8145,
Tool 0.818) isolate the mover, so quiet districts go genuinely quiet.

Test (no RAFT needed -- frame-diff is the motion proxy, identical for both feature models):
  per frame f: diff = |gray(f+2) - gray(f)|; blob = top-decile diff pixels
  district each frame TWO ways (KMeans 12, mirroring the gate) -> per-districting:
    n80      : #districts needed to cover 80% of blob mass  (mover isolation; lower better)
    best2IoU : IoU(union of top-2 blob districts, blob)      (higher better)
    q10proxy : 10th pct of per-district mean diff            (the quiet-decile analog;
               must be LOW on tail/still and HIGH on moving for a usable gate signal)
  plus the LOSO head's predicted seg: Tool(=3) coverage of the blob + Tool frame-fraction.

PASS = v3 tail q10proxy drops toward still-segment levels while moving stays high (separation
reopens), and/or n80 drops (mover isolated); Tool-on-blob high = mask route viable.

Run (torch2 modern stack; needs transformers + scikit-learn):
  python Addons/seg/validate_dino_gate_tail_20260708.py \
    --rgb_dir /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/rgb \
    --ckpt   /content/drive/MyDrive/Datasets/seg/loso_dinov3_b2/dinov2_crcd_C1_001_SWEEPONLY.pth \
    --dinov3 /content/drive/MyDrive/Datasets/DiNO
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


def load_gray(rgb_dir, offset, f):
    import cv2
    p = os.path.join(rgb_dir, f"frame_{offset + f:06d}.png")
    im = cv2.imread(p)
    assert im is not None, f"missing {p}"
    return im, cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def districts_from_grid(grid, n_groups=12, seed=0):
    from sklearn.cluster import KMeans
    gh, gw, C = grid.shape
    X = grid.reshape(-1, C)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return KMeans(n_groups, n_init=4, random_state=seed).fit_predict(X).reshape(gh, gw)


def v3_grid(model, bgr, device, patch=16):
    """[gh,gw,768] patch grid from the FINE-TUNED LOSO backbone (mirrors flow_track.dino_grid
    but patch-16 and ImageNet-normed like the trainer's eval path)."""
    import cv2, torch
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = im.shape[:2]; gh, gw = (H // patch) * patch, (W // patch) * patch
    im = cv2.resize(im, (gw, gh))
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(device)
    with torch.inference_mode():
        tok = model._patch_tokens(t)[0].cpu().numpy()
    return tok.reshape(gh // patch, gw // patch, -1).astype(np.float32)


def seg_predict(model, bgr, device):
    import cv2, torch
    im = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], np.float32); std = np.array([0.229, 0.224, 0.225], np.float32)
    t = torch.from_numpy((im - mean) / std).permute(2, 0, 1)[None].float().to(device)
    with torch.inference_mode():
        logit = model(t)
    return logit.argmax(1)[0].cpu().numpy()          # [h,w] class ids, 3 = Tool


def frame_metrics(diff, lab, n_groups):
    """diff + district labels at the SAME grid resolution -> (n80, best2IoU, q10proxy)."""
    blob = diff >= np.percentile(diff, 90)
    mass = np.array([float((blob & (lab == g)).sum()) for g in range(n_groups)])
    order = np.argsort(mass)[::-1]
    cum = np.cumsum(mass[order]) / max(1.0, mass.sum())
    n80 = int(np.searchsorted(cum, 0.8) + 1)
    top2 = (lab == order[0]) | (lab == order[1])
    iou = float((top2 & blob).sum()) / max(1.0, float((top2 | blob).sum()))
    dmean = np.array([diff[lab == g].mean() if (lab == g).any() else np.inf for g in range(n_groups)])
    q10 = float(np.percentile(dmean[np.isfinite(dmean)], 10))
    return n80, iou, q10


def main():
    import cv2, torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', required=True)
    ap.add_argument('--ckpt', required=True, help='LOSO DINOv3 SWEEPONLY.pth (full state_dict)')
    ap.add_argument('--dinov3', required=True, help='HF DINOv3 dir (config.json + model.safetensors)')
    ap.add_argument('--offset', type=int, default=1560, help='frame filename offset (C1_001=1560)')
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--stride', type=int, default=4)
    a = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEGS = {'TAIL(GT-still,instrument-working)': range(283, 356, a.stride),
            'EARLY-STILL(clean)': range(21, 56, a.stride),
            'MOVING(camera)': range(101, 136, a.stride)}

    # -- base: the gate's current featurizer (natural-image DINOv2-S/14) --
    from Addons.motion.flow_track import load_dino, dino_grid
    base = load_dino(device)

    # -- ours: LOSO DINOv3-B/16 fine-tune (backbone + head from the checkpoint) --
    from Addons.seg.train_dinov2_crcd import load_dinov3_hf, DINO2SEG
    im0, _ = load_gray(a.rgb_dir, a.offset, 0)
    H, W = im0.shape[:2]
    bb, patch, nreg, embed = load_dinov3_hf(a.dinov3)
    model = DINO2SEG(H, W, 4, bb, patch_size=patch, n_register=nreg, edge=0, dim=16,
                     train_blocks=0, embed=embed).to(device)
    sd = torch.load(a.ckpt, map_location='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith('backbone.model.')]
    print(f"[ckpt] loaded {a.ckpt}: missing={len(missing)} (non-backbone: {bad[:4]}) unexpected={len(unexpected)}")
    assert not bad, "head keys missing -- wrong checkpoint/arch"
    model.eval()

    print(f"{'segment':<36}{'model':<6}{'n80':>5}{'best2IoU':>10}{'q10proxy':>10}{'tool@blob':>11}{'tool%':>7}")
    verdict = {}
    for seg_name, frames in SEGS.items():
        acc = {'base': [], 'v3': []}; tool_cov = []; tool_frac = []
        for f in frames:
            bgr, g0 = load_gray(a.rgb_dir, a.offset, f)
            _, g1 = load_gray(a.rgb_dir, a.offset, f + 2)
            diff = np.abs(g1 - g0)
            gb = dino_grid(bgr, base, device)
            lb = districts_from_grid(gb, a.n_groups)
            db = cv2.resize(diff, (lb.shape[1], lb.shape[0]), interpolation=cv2.INTER_AREA)
            acc['base'].append(frame_metrics(db, lb, a.n_groups))
            gv = v3_grid(model, bgr, device, patch)
            lv = districts_from_grid(gv, a.n_groups)
            dv = cv2.resize(diff, (lv.shape[1], lv.shape[0]), interpolation=cv2.INTER_AREA)
            acc['v3'].append(frame_metrics(dv, lv, a.n_groups))
            seg = seg_predict(model, bgr, device)
            segr = cv2.resize(seg.astype(np.uint8), (diff.shape[1], diff.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            blob = diff >= np.percentile(diff, 90)
            tool = segr == 3
            tool_cov.append(float((tool & blob).sum()) / max(1.0, float(blob.sum())))
            tool_frac.append(float(tool.mean()))
        for k in ('base', 'v3'):
            m = np.array(acc[k]).mean(0)
            tc = np.mean(tool_cov) if k == 'v3' else float('nan')
            tf = np.mean(tool_frac) if k == 'v3' else float('nan')
            print(f"{seg_name:<36}{k:<6}{m[0]:>5.1f}{m[1]:>10.2f}{m[2]:>10.2f}{tc:>11.2f}{tf:>7.2f}")
            verdict[(seg_name, k)] = m

    tail = [k for k in SEGS if k.startswith('TAIL')][0]
    mov = [k for k in SEGS if k.startswith('MOVING')][0]
    sep_base = verdict[(mov, 'base')][2] / max(verdict[(tail, 'base')][2], 1e-6)
    sep_v3 = verdict[(mov, 'v3')][2] / max(verdict[(tail, 'v3')][2], 1e-6)
    print(f"\nquiet-decile separation (moving/tail q10proxy; >1 = gate signal usable): "
          f"base {sep_base:.2f} -> v3 {sep_v3:.2f}")
    print("PASS iff v3 separation > base AND v3 tail n80 < base tail n80 "
          "(mover isolated + stillness visible). tool@blob >~0.5 => mask route ALSO viable.")


if __name__ == '__main__':
    main()
