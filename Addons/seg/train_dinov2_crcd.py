#!/usr/bin/env python3
"""Train a CRCD 4-class seg-head -> dinov2_crcd.pth (Arm-4 A4-0.3, generalization build 2026-06-19).

This is the GENERALIZED trainer behind the seg-head improvement plan
(benchmarking/SEG_HEAD_improvement_plan_20260619.md §0, research wf w2t409q34). Everything is
FLAG-GATED so the old baseline recipe is reproducible and each lever is a clean A/B:

  RECIPE (the held-out win, backbone-agnostic):
    --train_blocks N   how many of the LAST transformer blocks to fine-tune. 0 = FREEZE the whole
                       backbone, train only the head (the cross-domain recipe; Kumar ICLR'22 LP-FT).
                       8 = the baselines' recipe (blocks 4-11 of 12). DEFAULT 8 (= old behaviour).
    --linear_head      use a single-conv linear head (768->n_cls) instead of the 2-conv head.
    --tune_norms       BitFit-lite: also train LayerNorm affines + biases (cheap PEFT).
    --aug              enable train-time domain-randomization augmentation (off by default).
    --gin              add GIN-lite (random-conv intensity randomization) on top of --aug.
    --loss {ce,wce,focal,dice,wce_dice}   class-balancing to rescue Tool/GB (default ce).
    --val_snippets ... episode/snippet-grouped INNER val for model selection (replaces the leaky
                       random-FRAME split; the old --val_frac path is kept only as a fallback).

  BACKBONE (a 2nd-order, metric-gated A/B - run on the SAME recipe):
    --backbone {dinov2,surgenet,dinov3}
       dinov2   = official ImageNet DINOv2 ViT-B/14 (the current init). patch 14.
       surgenet = SurgeNetXL surgical DINOv2 ViT-B/14 (--backbone_weights = the .pth). patch 14,
                  drops into the unmodified SNI/SemGauss DINO2SEG (pos_embed auto-dropped on load).
       dinov3   = DINOv3 ViT-B/16 (--backbone_weights = gated .pth or hub spec). patch 16, 4 register
                  tokens. DEVIATES from the baselines: the produced .pth does NOT load into the
                  unmodified SNI/SemGauss DINO2SEG (they are /14) - that codebase adaptation is a
                  separate, later step, run only if dinov3 wins the held-out A/B. NOT run-validated
                  offline; the grid math is generalized by patch_size + n_register (see DINO2SEG).

FAITHFULNESS / COMPAT: for dinov2/surgenet the state_dict keys are unchanged regardless of
--train_blocks (torch saves all params; requires_grad only gates grads), so dinov2_crcd.pth still
loads strict into SemGauss + strict=False into SNI iff n_classes=4, dim=16, crop_edge=0.

SPLIT: cross-episode generalization is judged held-out (--test_snippets). For our 5 benchmark
snippets, leave-one-SNIPPET-out == leave-one-EPISODE-out for the 4 single-snippet episodes
(C1/C2/C3/G3); E3_005 shares episode E_3 with the train E3_001-004 so under LOSO it is in-domain
- report it SEPARATELY (the LOSO runbook handles this).

RUNS ON COLAB (GPU + staged CRCD). This machine can only syntax-check it.

Usage (baseline, == old behaviour):
  python Addons/seg/train_dinov2_crcd.py --crcd_root data/CRCD --dinov2_main <main> --out seg/x.pth \
      --snippets ... --test_snippets ... --n_classes 4 --dim 16 --img_h 504 --img_w 896
Usage (improved recipe, frozen + aug + balanced loss + grouped val):
  ... --train_blocks 0 --aug --loss wce_dice --val_snippets B2_001 G2_003
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import cv2
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:  # pragma: no cover - environment guard
    print(f"[train_dinov2_crcd] missing dep: {e}. Run inside the torch2 env (colab_setup.sh).")
    raise

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
DINOV2_VITB14_URL = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'


# --- backbones ---------------------------------------------------------------------------
def make_vit_base(dinov2_main):
    """Vendored facebookresearch DINOv2 ViT-B/14."""
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


def load_dinov2_weights(backbone, weights):
    """Load DINOv2 /14 weights (official url, or a SurgeNetXL surgical .pth). REQUIRED.
    pos_embed is DROPPED when its shape differs (e.g. SurgeNetXL size336 -> 577 vs our 1370):
    strict=False does NOT silence shape mismatch, and DINOv2 re-interpolates pos_embed per forward,
    so dropping it is free (see feedback_pytorch_strict_false_shape_mismatch)."""
    if weights and weights not in ('hub', 'url', ''):
        sd = torch.load(weights, map_location='cpu')
    else:
        sd = torch.hub.load_state_dict_from_url(DINOV2_VITB14_URL, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        sd = sd['model']
    if isinstance(sd, dict) and 'teacher' in sd and isinstance(sd['teacher'], dict):
        sd = {k.replace('backbone.', ''): v for k, v in sd['teacher'].items() if 'backbone.' in k}
    sd = {k: v for k, v in sd.items()
          if not (k == 'pos_embed' and tuple(v.shape) != tuple(backbone.pos_embed.shape))}
    miss, unexp = backbone.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(unexp)
    print(f"[backbone init] dinov2/14 loaded {loaded}/{len(sd)} (missing={len(miss)} unexpected={len(unexp)})")
    if loaded < 0.5 * len(sd):
        raise RuntimeError("backbone init loaded <50% of weights - key mismatch; aborting "
                           "(fine-tuning a random ViT on CRCD is useless).")


def load_dinov3(weights):
    """DINOv3 ViT-B/16 (patch 16, 4 register tokens, 768-dim). Best-effort hub load; NOT
    run-validated offline. weights = a local gated .pth or a hub weights spec. The DINOv3 license
    permits commercial use but the weights are GATED - download once with an accepted-terms HF token.
    Returns (backbone, patch_size=16, n_register=4, embed=768)."""
    repo = os.environ.get('DINOV3_HUB', 'facebookresearch/dinov3')
    entry = os.environ.get('DINOV3_ENTRY', 'dinov3_vitb16')
    try:
        bb = torch.hub.load(repo, entry, weights=weights) if weights else torch.hub.load(repo, entry)
    except Exception as e:  # pragma: no cover - needs network + gated weights
        raise RuntimeError(f"DINOv3 load failed (repo={repo} entry={entry} weights={weights}): {e}. "
                           f"Set DINOV3_HUB/DINOV3_ENTRY or pass a local .pth via --backbone_weights.")
    return bb, 16, 4, 768


def build_backbone(name, dinov2_main, weights):
    name = name.lower()
    if name == 'dinov2':
        bb = make_vit_base(dinov2_main); load_dinov2_weights(bb, weights or 'url'); return bb, 14, 0, 768
    if name == 'surgenet':
        if not weights or weights in ('url', 'hub', ''):
            raise RuntimeError("--backbone surgenet needs --backbone_weights <SurgeNetXL DINOv2_ViTb14 .pth>")
        bb = make_vit_base(dinov2_main); load_dinov2_weights(bb, weights); return bb, 14, 0, 768
    if name == 'dinov3':
        return load_dinov3(weights)
    raise ValueError(f"unknown --backbone {name} (choose dinov2|surgenet|dinov3)")


# --- the seg head (param-name-identical to SNI/SemGauss DINO2SEG for /14 backbones) -------
class DINO2SEG(nn.Module):
    def __init__(self, img_h, img_w, num_cls, backbone, patch_size=14, n_register=0,
                 edge=0, dim=16, train_blocks=8, tune_norms=False, linear_head=False, embed=768):
        super().__init__()
        self.embedding_size, self.patch_size, self.n_register = embed, patch_size, n_register
        self.num_class, self.img_h, self.img_w = num_cls, img_h, img_w
        self.backbone = backbone
        self._set_trainable(train_blocks, tune_norms)
        if linear_head:  # minimal capacity: single conv on frozen features
            self.segmentation_conv = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(embed, num_cls, (3, 3), padding=(1, 1)),
                nn.Upsample((img_h - 2 * edge, img_w - 2 * edge)))
        else:            # the baselines' 2-conv head (keys match SNI/SemGauss at /14)
            self.segmentation_conv = nn.Sequential(
                nn.Upsample(scale_factor=4),
                nn.Conv2d(embed, dim, (3, 3), padding=(1, 1)),
                nn.Upsample((img_h - 2 * edge, img_w - 2 * edge)),
                nn.Conv2d(dim, num_cls, (3, 3), padding=(1, 1)))
        bh = ((img_h - 2 * edge) // patch_size) * patch_size
        bw = ((img_w - 2 * edge) // patch_size) * patch_size
        self.upsample = nn.Upsample((bh, bw))

    def _set_trainable(self, train_blocks, tune_norms):
        """Freeze all, then unfreeze the LAST `train_blocks` transformer blocks (+ final norm).
        train_blocks=0 -> fully frozen backbone (train only the head). Robust to dinov2/dinov3
        naming via the 'blocks.<i>.' regex. Note: state_dict KEYS are unchanged either way."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        idxs = set()
        for nme, _ in self.backbone.named_parameters():
            m = re.search(r'blocks\.(\d+)\.', nme)
            if m:
                idxs.add(int(m.group(1)))
        nblk = (max(idxs) + 1) if idxs else 0
        keep = set(range(max(0, nblk - train_blocks), nblk)) if train_blocks > 0 else set()
        for nme, p in self.backbone.named_parameters():
            m = re.search(r'blocks\.(\d+)\.', nme)
            if train_blocks > 0 and ((m and int(m.group(1)) in keep) or nme.startswith('norm')):
                p.requires_grad = True
            if tune_norms and ('norm' in nme or nme.endswith('.bias')):
                p.requires_grad = True

    def _patch_tokens(self, x):
        """Return patch tokens [B, N, C] (cls + register stripped), for dinov2 and dinov3."""
        out = self.backbone.forward_features(x.float())
        if isinstance(out, dict):
            if 'x_norm_patchtokens' in out:           # dinov2 (and dinov3 hub) expose this directly
                return out['x_norm_patchtokens']
            out = out.get('last_hidden_state', out.get('x', None))
        if not torch.is_tensor(out):
            raise RuntimeError("backbone.forward_features returned no usable patch tokens")
        return out[:, 1 + self.n_register:, :]        # strip cls + register tokens

    def forward(self, x):
        x = self.upsample(x)
        bs = x.shape[0]
        gh, gw = int(x.shape[2] / self.patch_size), int(x.shape[3] / self.patch_size)
        tok = self._patch_tokens(x)                   # [B, N, C]
        assert tok.shape[1] == gh * gw, (
            f"patch-token count {tok.shape[1]} != grid {gh}x{gw}={gh * gw} "
            f"(patch_size={self.patch_size}, n_register={self.n_register}) - check backbone/input size")
        # NOTE: raw reshape (the SNI/SemGauss convention - a fixed deterministic permutation the
        # head learns atop); kept for deploy-faithfulness, applied identically train + eval.
        out = tok.reshape(bs, self.embedding_size, gh, gw)
        return self.segmentation_conv(out)


# --- data --------------------------------------------------------------------------------
def aug_rgb_label(rgb, lab, h, w, rng, gin=False):
    """Train-time domain randomization. rgb uint8 HxWx3 (RGB), lab uint8 HxW, already at (h,w).
    Photometric = RGB-only; geometric (flip + random-resized-crop) applied to BOTH (label NEAREST)."""
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1].copy(); lab = lab[:, ::-1].copy()
    area = rng.uniform(0.6, 1.0); ar = rng.uniform(0.85, 1.18)          # random-resized-crop (zoom-in)
    ch = min(h, int(round((h * w * area * ar) ** 0.5)))
    cw = min(w, int(round((h * w * area / ar) ** 0.5)))
    y0 = int(rng.integers(0, h - ch + 1)); x0 = int(rng.integers(0, w - cw + 1))
    rgb = cv2.resize(rgb[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    lab = cv2.resize(lab[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_NEAREST)
    f = rgb.astype(np.float32)
    f *= rng.uniform(0.7, 1.3)                                          # brightness/gain
    m = f.mean((0, 1), keepdims=True); f = np.clip((f - m) * rng.uniform(0.7, 1.3) + m, 0, 255)  # contrast
    f = 255.0 * np.clip(f / 255.0, 1e-6, 1) ** rng.uniform(0.7, 1.4)    # gamma
    hsv = cv2.cvtColor(np.clip(f, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-10, 10)) % 180            # hue (scope white-balance)
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)  # saturation
    f = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
    if rng.random() < 0.3:                                             # CLAHE on luminance
        l = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2LAB)
        l[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(l[..., 0])
        f = cv2.cvtColor(l, cv2.COLOR_LAB2RGB).astype(np.float32)
    if rng.random() < 0.3:                                             # defocus/motion blur
        k = int(rng.choice([3, 5])); f = cv2.GaussianBlur(f, (k, k), 0)
    if gin and rng.random() < 0.5:                                     # GIN-lite: random-conv blend
        a = rng.uniform(0.1, 0.5)
        ker = rng.normal(0, 1, (3, 3)).astype(np.float32); ker /= np.abs(ker).sum() + 1e-6
        g = np.stack([cv2.filter2D(f[..., c], -1, ker) for c in range(3)], -1)
        f = (1 - a) * f + a * np.clip(g, 0, 255)
    f = f + rng.normal(0, rng.uniform(0, 8), f.shape)                   # sensor noise
    return np.clip(f, 0, 255).astype(np.uint8), lab


class CRCDSeg(Dataset):
    def __init__(self, pairs, img_h, img_w, n_classes, train=False, aug=False, gin=False):
        self.pairs, self.img_h, self.img_w, self.n_classes = pairs, img_h, img_w, n_classes
        self.train, self.aug, self.gin = train, aug, gin

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        rgb_p, lab_p = self.pairs[i]
        rgb = cv2.cvtColor(cv2.imread(rgb_p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)
        lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[..., 0]
        u = np.unique(lab)
        assert u.max() < self.n_classes, (
            f"{lab_p}: label ids {u.tolist()} exceed n_classes={self.n_classes} - "
            f"raw semantic_instance must be coco_id+1 in {{0,1,2,3}} (00_COMMON sec0 Decision 3).")
        lab = cv2.resize(lab.astype(np.uint8), (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
        if self.train and self.aug:
            rgb, lab = aug_rgb_label(rgb, lab, self.img_h, self.img_w, np.random.default_rng(), self.gin)
        rgb = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(rgb.transpose(2, 0, 1)), torch.from_numpy(lab.astype(np.int64))


def gather_pairs(crcd_root, snippets, rgb_subdir, label_subdir, rgb_glob, label_glob,
                 required=True, tag='train'):
    """Pair RGB <-> label per snippet by basename (raw rgb/semantic_instance share names), else index."""
    names = snippets or sorted(
        d for d in os.listdir(crcd_root)
        if os.path.isdir(os.path.join(crcd_root, d, rgb_subdir))
        and os.path.isdir(os.path.join(crcd_root, d, label_subdir)))
    pairs = []
    for nm in names:
        rgbs = sorted(glob.glob(os.path.join(crcd_root, nm, rgb_subdir, rgb_glob)))
        labs = sorted(glob.glob(os.path.join(crcd_root, nm, label_subdir, label_glob)))
        lab_by_base = {os.path.basename(p): p for p in labs}
        matched = [(r, lab_by_base[os.path.basename(r)]) for r in rgbs if os.path.basename(r) in lab_by_base]
        if matched:
            sp = matched
        else:
            n = min(len(rgbs), len(labs))
            if len(rgbs) != len(labs):
                print(f"[data] WARN {nm}: {len(rgbs)} rgb vs {len(labs)} labels; pairing first {n}.")
            sp = list(zip(rgbs[:n], labs[:n]))
        if not sp:
            print(f"[data] WARN {nm}: no ({rgb_subdir},{label_subdir}) pairs found.")
        pairs += sp
    if not pairs:
        msg = f"no ({rgb_subdir}, {label_subdir}) pairs under {crcd_root} ({tag} snippets={names})."
        if required:
            raise RuntimeError(msg)
        print(f"[data] WARN ({tag}) {msg} -> skipping {tag} eval.")
        return []
    print(f"[data] {tag}: {len(pairs)} frames over snippets={names}")
    return pairs


# --- losses ------------------------------------------------------------------------------
def class_weights(pairs, n_classes):
    cnt = np.zeros(n_classes, np.float64)
    for _, lp in pairs:
        l = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
        l = l[..., 0] if l.ndim == 3 else l
        cnt += np.bincount(l.ravel(), minlength=n_classes)[:n_classes]
    w = cnt.sum() / (n_classes * np.maximum(cnt, 1))         # inverse frequency
    return torch.tensor((w / w.mean()).astype(np.float32))


def dice_loss(logits, target, n_classes, eps=1.0):
    p = torch.softmax(logits, 1)
    t = F.one_hot(target, n_classes).permute(0, 3, 1, 2).float()
    inter = (p * t).sum((0, 2, 3)); denom = p.sum((0, 2, 3)) + t.sum((0, 2, 3))
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def make_loss(kind, weights, n_classes, device, gamma=2.0):
    w = weights.to(device) if weights is not None else None
    ce_w = nn.CrossEntropyLoss(weight=w)
    ce = nn.CrossEntropyLoss()
    if kind == 'ce':
        return lambda o, t: ce(o, t)
    if kind == 'wce':
        return lambda o, t: ce_w(o, t)
    if kind == 'dice':
        return lambda o, t: dice_loss(o, t, n_classes)
    if kind == 'wce_dice':
        return lambda o, t: ce_w(o, t) + dice_loss(o, t, n_classes)
    if kind == 'focal':
        def focal(o, t):
            logp = F.log_softmax(o, 1); p = logp.exp()
            pt = p.gather(1, t.unsqueeze(1)).squeeze(1).clamp_(1e-6, 1)
            wt = w[t] if w is not None else 1.0
            return (-wt * (1 - pt) ** gamma * pt.log()).mean()
        return focal
    raise ValueError(f"unknown --loss {kind}")


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
    ap.add_argument('--crcd_root', required=True)
    ap.add_argument('--dinov2_main', required=True, help='vendored facebookresearch_dinov2_main (for /14)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--snippets', nargs='+', default=None, help='TRAIN dirs (default: all found)')
    ap.add_argument('--test_snippets', nargs='+', default=None, help='HELD-OUT eval dirs (never trained)')
    ap.add_argument('--val_snippets', nargs='+', default=None,
                    help='INNER val dirs for model selection (held from TRAIN; replaces the leaky '
                         'random-frame split). Strongly preferred over --val_frac.')
    ap.add_argument('--rgb_subdir', default='rgb'); ap.add_argument('--label_subdir', default='semantic_instance')
    ap.add_argument('--rgb_glob', default='*.png'); ap.add_argument('--label_glob', default='*.png')
    # backbone
    ap.add_argument('--backbone', default='dinov2', choices=['dinov2', 'surgenet', 'dinov3'])
    ap.add_argument('--backbone_weights', default='url', help="'url'|local .pth (surgenet/dinov3 need a path/spec)")
    # recipe / freeze granularity
    ap.add_argument('--train_blocks', type=int, default=8, help='unfreeze last N blocks (0=freeze; 8=baseline)')
    ap.add_argument('--tune_norms', action='store_true', help='BitFit-lite: also train norms+biases')
    ap.add_argument('--linear_head', action='store_true', help='single-conv head instead of the 2-conv head')
    ap.add_argument('--aug', action='store_true', help='train-time domain-randomization augmentation')
    ap.add_argument('--gin', action='store_true', help='add GIN-lite random-conv intensity randomization')
    ap.add_argument('--loss', default='ce', choices=['ce', 'wce', 'focal', 'dice', 'wce_dice'])
    ap.add_argument('--n_classes', type=int, default=4); ap.add_argument('--dim', type=int, default=16)
    ap.add_argument('--img_h', type=int, default=504); ap.add_argument('--img_w', type=int, default=896)
    ap.add_argument('--crop_edge', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_every', type=int, default=2); ap.add_argument('--test_eval_cap', type=int, default=600)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gp = dict(rgb_subdir=a.rgb_subdir, label_subdir=a.label_subdir, rgb_glob=a.rgb_glob, label_glob=a.label_glob)

    # ---- split (episode/snippet-grouped inner val preferred over the leaky random-frame split) ----
    train_names = a.snippets or sorted(
        d for d in os.listdir(a.crcd_root)
        if os.path.isdir(os.path.join(a.crcd_root, d, a.rgb_subdir)))
    if a.val_snippets:
        leak = set(a.val_snippets) & set(a.test_snippets or [])
        assert not leak, f"--val_snippets overlaps --test_snippets {leak} (leakage)"
        tr_names = [n for n in train_names if n not in a.val_snippets]
        train = gather_pairs(a.crcd_root, tr_names, required=True, tag='train', **gp)
        val = gather_pairs(a.crcd_root, list(a.val_snippets), required=True, tag='val(grouped)', **gp)
    else:
        print("[split] WARN no --val_snippets -> leaky random-FRAME val (in-domain, invalid for "
              "generalization selection). Use --val_snippets for honest model selection.")
        pairs = gather_pairs(a.crcd_root, train_names, required=True, tag='train', **gp)
        rng = np.random.default_rng(a.seed); idx = rng.permutation(len(pairs))
        nval = max(1, int(len(pairs) * a.val_frac))
        val = [pairs[i] for i in idx[:nval]]; train = [pairs[i] for i in idx[nval:]]
    if a.smoke:
        train = train[:16]; val = val[:8]; a.epochs = 2; a.batch_size = 1

    tr = DataLoader(CRCDSeg(train, a.img_h, a.img_w, a.n_classes, train=True, aug=a.aug, gin=a.gin),
                    batch_size=a.batch_size, shuffle=True, num_workers=2, drop_last=True)
    vl = DataLoader(CRCDSeg(val, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2)
    test_pairs = gather_pairs(a.crcd_root, a.test_snippets, required=False, tag='test', **gp) if a.test_snippets else []
    tt = DataLoader(CRCDSeg(test_pairs, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2) if test_pairs else None
    quick = test_pairs[::max(1, len(test_pairs) // max(1, a.test_eval_cap))][:a.test_eval_cap] if test_pairs else []
    tt_quick = DataLoader(CRCDSeg(quick, a.img_h, a.img_w, a.n_classes), batch_size=1, num_workers=2) if quick else None

    # ---- model ----
    bb, patch, nreg, embed = build_backbone(a.backbone, a.dinov2_main, a.backbone_weights)
    model = DINO2SEG(a.img_h, a.img_w, a.n_classes, bb, patch_size=patch, n_register=nreg,
                     edge=a.crop_edge, dim=a.dim, train_blocks=a.train_blocks,
                     tune_norms=a.tune_norms, linear_head=a.linear_head, embed=embed).to(device)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nbb = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(f"[model] backbone={a.backbone}/{patch} train_blocks={a.train_blocks} tune_norms={a.tune_norms} "
          f"linear_head={a.linear_head} | trainable={ntr/1e6:.2f}M (backbone {nbb/1e6:.2f}M) | "
          f"aug={a.aug} gin={a.gin} loss={a.loss}")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    cw = class_weights(train, a.n_classes) if a.loss in ('wce', 'focal', 'wce_dice') else None
    if cw is not None:
        print(f"[loss] {a.loss} inverse-freq class weights = {np.round(cw.numpy(), 3).tolist()}")
    crit = make_loss(a.loss, cw, a.n_classes, device)

    best = -1.0
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for rgb, lab in tr:
            opt.zero_grad()
            loss = crit(model(rgb.to(device)), lab.to(device))
            loss.backward(); opt.step(); tot += loss.item()
        m, ious = miou(model, vl, a.n_classes, device)
        print(f"[ep {ep:03d}] train_loss={tot/max(1,len(tr)):.4f}  val_mIoU={m:.4f}  perclass={np.round(ious,3).tolist()}")
        if m > best:                                   # '>' (not '>='): keep the EARLIER best epoch
            best = m
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            torch.save(model.state_dict(), a.out)
        if tt_quick is not None and a.test_every > 0 and (ep + 1) % a.test_every == 0:
            qm, qious = miou(model, tt_quick, a.n_classes, device)
            print(f"[ep {ep:03d}] HELD-OUT(quick {len(quick)}f) mIoU={qm:.4f}  perclass={np.round(qious,3).tolist()}  (informational)")
    if tt is not None:
        model.load_state_dict(torch.load(a.out, map_location=device))
        tm, tious = miou(model, tt, a.n_classes, device)
        print(f"[HELD-OUT TEST] {a.test_snippets} mIoU (NEVER trained) = {tm:.4f}  "
              f"perclass={np.round(tious,3).tolist()}  (0=bg 1=Liver 2=Gallbladder 3=Tool)")
    print(f"[done] best inner-val mIoU={best:.4f}  saved -> {a.out}")
    if a.backbone in ('dinov2', 'surgenet'):
        print("[load-compat] /14 + n_classes=4,dim=16 -> loads strict into SemGauss, strict=False into SNI.")
    else:
        print("[load-compat] dinov3/16 -> does NOT load into the unmodified SNI/SemGauss DINO2SEG "
              "(separate codebase adaptation needed; run only if dinov3 wins the held-out A/B).")


if __name__ == '__main__':
    main()
