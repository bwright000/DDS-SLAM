#!/usr/bin/env python3
"""Train ONE CRCD 4-class DINOv2 seg-head -> dinov2_crcd.pth (Arm-4 item A4-0.3).

WHY ONE TRAINER SERVES BOTH SNI-SLAM AND SemGauss-SLAM
------------------------------------------------------
Verified 2026-06-17 against the local repos: SNI-SLAM's `DINO2SEG`
(sni-slam/src/networks/dinov2_seg.py) and SemGauss-SLAM's `DINO2SEG`
(SemGauss-SLAM/utils/dinov2_seg.py) are the SAME architecture:
  - backbone = DINOv2 `vit_base` (dinov2_vitb14, embed=768, patch=14), blocks 0-3
    FROZEN, blocks 4+ trainable (identical requires_grad loop);
  - head = segmentation_conv = Sequential(Upsample(x4), Conv2d(768->dim),
    Upsample(H-2e,W-2e), Conv2d(dim->n_cls)); dim=16 in BOTH method configs.
The only differences are parameter-FREE: the `Upsample` target sizes, the mode-enum
names (SNI {mapping, train/result} == SemGauss {get_feature, get_semantic, classification}),
and `crop_edge`. None of those enter the state_dict. The learnable parameter shapes depend
only on {embed=768, dim=16, n_cls}. So with **n_cls=4, dim=16** a single state_dict has
identical keys+shapes for both, and the SAME `dinov2_crcd.pth` loads into:
  - SemGauss `Segmentation.get_dinov2` -> `load_state_dict(strict=True)`  (exact match), and
  - SNI `ModelManager.get_dinov2`      -> `load_state_dict(strict=False)` (nothing dropped,
    because the final conv is now [4,16,3,3] == SNI's CRCD n_classes=4; this also FIXES the
    bug that SNI's fused 16-d `sem_feat` was Replica-trained, see AUDIT §4.2).

REQUIREMENT for the single .pth to load strict into BOTH: the SNI and SemGauss CRCD configs
must BOTH set n_classes=4, c_dim=16, crop_edge=0 (SNI crcd_sni_base.yaml already does; the
SemGauss CRCD config authored in A4-3.5 must match). crop_edge only changes parameter-free
Upsample sizes, so a mismatch would still LOAD, but keep them equal for fidelity.

WHAT THIS TRAINS
----------------
Per the SemGauss per-scene convention (`dinov2_{scene}.pth`) and SNI's one-head-per-dataset
convention (`dinov2_replica.pth`): by default ONE head over all CRCD snippets found (one
surgical domain, shared 4 classes). Pass --snippets to restrict (e.g. per-snippet head).
RGB = RAW left frames `rgb/frame_*.png`; GT = RAW masks `semantic_instance/frame_*.png`
(uint16, coco_id+1 -> {0=bg,1=Liver,2=Gallbladder,3=Tool}). These are the ORIGINAL left
frames the CRCD segmentations were annotated on (preprocess_crcd_published.py:127 rectifies
the mask only for the video overlay) -> training on raw rgb+mask keeps RGB<->label aligned
and skips slow/lossy rectification. Paired by basename. (Deploy note: SNI/SemGauss consume
the DDS loader's RECTIFIED frames at SLAM time -> mild train(raw)/deploy(rectified) gap.)

The backbone is initialised from the OFFICIAL DINOv2 vitb14 pretrained weights (torch.hub,
or --backbone_weights <path>) so blocks 0-3 are frozen at good features and blocks 4+ +
the conv head fine-tune on CRCD. Training a ViT from random init on small CRCD data would be
useless, so a successful backbone init is REQUIRED (the script exits non-zero if it fails).

RUNS ON COLAB (GPU + staged CRCD). This machine can only syntax-check it.

Usage:
  python Addons/seg/train_dinov2_crcd.py \
      --crcd_root data/CRCD --dinov2_main /content/sni-slam/seg/facebookresearch_dinov2_main \
      --out seg/dinov2_crcd.pth --epochs 40 --img_h 720 --img_w 1280 --crop_edge 0 \
      --n_classes 4 --dim 16 --val_frac 0.1
  # quick wiring check (few iters, tiny):
  python Addons/seg/train_dinov2_crcd.py --crcd_root data/CRCD --dinov2_main <path> --out /tmp/x.pth --smoke
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover - environment guard
    print(f"[train_dinov2_crcd] missing dep: {e}. Run inside the torch2 env (colab_setup.sh).")
    raise

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# --- the seg head (param-name-identical to BOTH repos' DINO2SEG; see module docstring) ----
def make_vit_base(dinov2_main):
    if dinov2_main and dinov2_main not in sys.path:
        sys.path.append(dinov2_main)
    try:
        from dinov2.models import vision_transformer as vits
    except ImportError as e:
        raise RuntimeError(
            f"could not import dinov2 from --dinov2_main='{dinov2_main}'. Point it at a vendored "
            f"facebookresearch_dinov2_main (e.g. sni-slam/seg/facebookresearch_dinov2_main). {e}")
    return vits.__dict__["vit_base"](img_size=518, patch_size=14, init_values=1.0,
                                     ffn_layer="mlp", block_chunks=0)


class DINO2SEG(nn.Module):
    """Same param layout as SNI/SemGauss DINO2SEG: backbone.* + segmentation_conv.{1,3}.*"""
    def __init__(self, img_h, img_w, num_cls, dinov2_main, edge=0, dim=16):
        super().__init__()
        self.embedding_size, self.patch_size = 768, 14
        self.num_class, self.img_h, self.img_w = num_cls, img_h, img_w
        self.backbone = make_vit_base(dinov2_main)
        # freeze blocks 0-3, train 4+ (identical to both repos' loop)
        switch = False
        for name, param in self.backbone.named_parameters():
            if 'blocks.4.' in name:
                switch = True
            param.requires_grad = bool(switch)
        self.segmentation_conv = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(self.embedding_size, dim, (3, 3), padding=(1, 1)),
            nn.Upsample((img_h - 2 * edge, img_w - 2 * edge)),
            nn.Conv2d(dim, num_cls, (3, 3), padding=(1, 1)),
        )
        bh = ((img_h - 2 * edge) // 14) * 14
        bw = ((img_w - 2 * edge) // 14) * 14
        self.upsample = nn.Upsample((bh, bw))

    def forward(self, x):  # full head -> per-pixel logits (= 'get_semantic'/'train' mode)
        x = self.upsample(x)
        bs = x.shape[0]
        gh, gw = int(x.shape[2] / self.patch_size), int(x.shape[3] / self.patch_size)
        out = self.backbone.forward_features(x.float())["x_norm_patchtokens"]
        out = out.reshape(bs, self.embedding_size, gh, gw)
        return self.segmentation_conv(out)  # [B, n_cls, H-2e, W-2e]


DINOV2_VITB14_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'


def init_backbone(model, weights):
    """Load official DINOv2 vitb14 PRETRAINED weights into model.backbone (REQUIRED).
    Uses load_state_dict_from_url (downloads a flat .pth, imports NOTHING) instead of
    torch.hub.load(): the latter executes the official dinov2 hubconf which does
    `from dinov2.hub...`, colliding with the VENDORED `dinov2` already on sys.path
    (make_vit_base) -> 'No module named dinov2.hub'. The .pth download has no such clash."""
    if weights and weights not in ('hub', 'url', ''):
        sd = torch.load(weights, map_location='cpu')
    else:
        sd = torch.hub.load_state_dict_from_url(DINOV2_VITB14_URL, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']  # unwrap if a full-training checkpoint is given
    miss, unexp = model.backbone.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(unexp)
    print(f"[backbone init] loaded {loaded}/{len(sd)} tensors (missing={len(miss)} unexpected={len(unexp)})")
    if loaded < 0.5 * len(sd):
        raise RuntimeError("backbone init loaded <50% of DINOv2 weights - key mismatch; aborting "
                           "(fine-tuning a random ViT on CRCD is useless).")


# --- data --------------------------------------------------------------------------------
class CRCDSeg(Dataset):
    def __init__(self, pairs, img_h, img_w, n_classes):
        self.pairs, self.img_h, self.img_w, self.n_classes = pairs, img_h, img_w, n_classes

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rgb_p, lab_p = self.pairs[i]
        rgb = cv2.cvtColor(cv2.imread(rgb_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1))
        lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[..., 0]
        u = np.unique(lab)
        assert u.max() < self.n_classes, (
            f"{lab_p}: label ids {u.tolist()} exceed n_classes={self.n_classes} - "
            f"raw semantic_instance must be coco_id+1 in {{0,1,2,3}} (00_COMMON sec0 Decision 3).")
        lab = cv2.resize(lab.astype(np.uint8), (self.img_w, self.img_h),
                         interpolation=cv2.INTER_NEAREST)
        return rgb, torch.from_numpy(lab.astype(np.int64))


def gather_pairs(crcd_root, snippets, rgb_subdir, label_subdir, rgb_glob, label_glob,
                 required=True, tag='train'):
    """Pair RGB <-> label per snippet. DEFAULT = RAW left frames (rgb/) + RAW masks
    (semantic_instance/), which is where the CRCD segmentations were actually annotated
    (preprocess_crcd_published.py:127). Pairs by BASENAME when they share names (raw rgb
    'frame_NNNNNN.png' == raw mask 'frame_NNNNNN.png'), else falls back to sorted index."""
    names = snippets or sorted(
        d for d in os.listdir(crcd_root)
        if os.path.isdir(os.path.join(crcd_root, d, rgb_subdir))
        and os.path.isdir(os.path.join(crcd_root, d, label_subdir)))
    pairs = []
    for nm in names:
        rgbs = sorted(glob.glob(os.path.join(crcd_root, nm, rgb_subdir, rgb_glob)))
        labs = sorted(glob.glob(os.path.join(crcd_root, nm, label_subdir, label_glob)))
        lab_by_base = {os.path.basename(p): p for p in labs}
        matched = [(r, lab_by_base[os.path.basename(r)]) for r in rgbs
                   if os.path.basename(r) in lab_by_base]
        if matched:
            sp = matched
        else:  # basenames differ (e.g. rectified '000000l.png' vs '000000.png') -> index pair
            n = min(len(rgbs), len(labs))
            if len(rgbs) != len(labs):
                print(f"[data] WARN {nm}: {len(rgbs)} rgb vs {len(labs)} labels; pairing first {n}.")
            sp = list(zip(rgbs[:n], labs[:n]))
        if not sp:
            print(f"[data] WARN {nm}: no ({rgb_subdir},{label_subdir}) pairs found.")
        pairs += sp
    if not pairs:
        msg = (f"no ({rgb_subdir}, {label_subdir}) pairs under {crcd_root} ({tag} snippets={names}).")
        if required:
            raise RuntimeError(msg)
        print(f"[data] WARN ({tag}) {msg} -> skipping {tag} eval.")
        return []
    print(f"[data] {tag}: {len(pairs)} frames over snippets={names}")
    return pairs


def miou(model, loader, n_classes, device):
    model.eval()
    inter = np.zeros(n_classes); union = np.zeros(n_classes)
    with torch.no_grad():
        for rgb, lab in loader:
            pred = model(rgb.to(device)).argmax(1).cpu().numpy()
            gt = lab.numpy()
            for c in range(n_classes):
                p, g = pred == c, gt == c
                inter[c] += np.logical_and(p, g).sum()
                union[c] += np.logical_or(p, g).sum()
    ious = inter / np.maximum(union, 1)
    return float(ious.mean()), ious


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--crcd_root', required=True, help='dir of preprocessed CRCD snippet folders')
    ap.add_argument('--dinov2_main', required=True, help='path to a vendored facebookresearch_dinov2_main')
    ap.add_argument('--out', required=True, help='output dinov2_crcd.pth')
    ap.add_argument('--snippets', nargs='+', default=None, help='TRAIN on these NAME dirs (default: all found)')
    ap.add_argument('--test_snippets', nargs='+', default=None,
                    help='HELD-OUT eval-only NAME dirs (never trained) - e.g. the 5 benchmark snippets; '
                         'mIoU on these is the generalization number')
    # DEFAULT = RAW left frames + RAW masks (the annotation domain). For rectified, pass
    # --rgb_subdir video_frames --rgb_glob "*l.png" --label_subdir semantic_class.
    ap.add_argument('--rgb_subdir', default='rgb', help='RGB subdir per snippet (default raw: rgb)')
    ap.add_argument('--label_subdir', default='semantic_instance',
                    help='label subdir per snippet (default raw: semantic_instance)')
    ap.add_argument('--rgb_glob', default='*.png', help='RGB glob (default *.png)')
    ap.add_argument('--label_glob', default='*.png', help='label glob (default *.png)')
    ap.add_argument('--backbone_weights', default='url',
                    help="'url' (download dinov2_vitb14_pretrain.pth) or a local .pth path")
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--dim', type=int, default=16)
    ap.add_argument('--img_h', type=int, default=720)
    ap.add_argument('--img_w', type=int, default=1280)
    ap.add_argument('--crop_edge', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_every', type=int, default=2,
                    help='eval held-out test mIoU every K epochs (0=only at end) - watch generalization live')
    ap.add_argument('--test_eval_cap', type=int, default=600,
                    help='periodic held-out eval uses a strided subset of this many frames (final uses full)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true', help='2 epochs, <=16 frames, batch 1 — wiring check')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gp = dict(rgb_subdir=a.rgb_subdir, label_subdir=a.label_subdir,
              rgb_glob=a.rgb_glob, label_glob=a.label_glob)
    pairs = gather_pairs(a.crcd_root, a.snippets, required=True, tag='train', **gp)
    if a.smoke:
        pairs = pairs[:16]; a.epochs = 2; a.batch_size = 1
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(pairs))
    nval = max(1, int(len(pairs) * a.val_frac))
    val = [pairs[i] for i in idx[:nval]]; train = [pairs[i] for i in idx[nval:]]

    tr = DataLoader(CRCDSeg(train, a.img_h, a.img_w, a.n_classes), batch_size=a.batch_size,
                    shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(CRCDSeg(val, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2)
    test_pairs = gather_pairs(a.crcd_root, a.test_snippets, required=False, tag='test', **gp) if a.test_snippets else []
    tt = DataLoader(CRCDSeg(test_pairs, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2) if test_pairs else None
    # periodic held-out readout: a strided subset spanning ALL test snippets (so every class shows)
    quick_pairs = test_pairs[::max(1, len(test_pairs) // max(1, a.test_eval_cap))][:a.test_eval_cap] if test_pairs else []
    tt_quick = DataLoader(CRCDSeg(quick_pairs, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2) if quick_pairs else None

    model = DINO2SEG(a.img_h, a.img_w, a.n_classes, a.dinov2_main, edge=a.crop_edge, dim=a.dim)
    init_backbone(model, a.backbone_weights)
    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    ce = nn.CrossEntropyLoss()

    best = -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for rgb, lab in tr:
            opt.zero_grad()
            loss = ce(model(rgb.to(device)), lab.to(device))
            loss.backward(); opt.step(); tot += loss.item()
        m, ious = miou(model, vl, a.n_classes, device)
        print(f"[ep {ep:03d}] train_loss={tot/max(1,len(tr)):.4f}  val_mIoU={m:.4f}  "
              f"perclass={np.round(ious,3).tolist()}")
        if m >= best:
            best = m
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            torch.save(model.state_dict(), a.out)
        if tt_quick is not None and a.test_every > 0 and (ep + 1) % a.test_every == 0:
            qm, qious = miou(model, tt_quick, a.n_classes, device)  # current model, informational
            print(f"[ep {ep:03d}] HELD-OUT(quick {len(quick_pairs)}f) mIoU={qm:.4f}  "
                  f"perclass={np.round(qious, 3).tolist()}")
    if tt is not None:
        model.load_state_dict(torch.load(a.out, map_location=device))  # best-by-val checkpoint
        tm, tious = miou(model, tt, a.n_classes, device)
        print(f"[HELD-OUT TEST] benchmark-snippet mIoU (NEVER trained) = {tm:.4f}  "
              f"perclass={np.round(tious, 3).tolist()}  (0=bg 1=Liver 2=Gallbladder 3=Tool)")
    print(f"[done] best val_mIoU={best:.4f}  saved -> {a.out}")
    print("[load-compat] strict-loadable by SemGauss Segmentation.get_dinov2 and (strict=False, "
          "0 dropped) by SNI ModelManager.get_dinov2 - requires both CRCD configs n_classes=4, c_dim=16.")


if __name__ == '__main__':
    main()
