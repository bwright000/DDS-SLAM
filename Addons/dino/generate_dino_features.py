#!/usr/bin/env python3
"""Offline DINO dense-feature precompute for DDS-SLAM Inc-1 v2 ('dino' mode).

WHY OFFLINE: DINOv2 needs torch>=2.0 and DINOv3 needs torch>=2.7; the DDS-SLAM run env is
py3.7/torch1.10. So features are baked in the torch>=2 env (exactly like MoGe-2 depth, two-env
split) and the run env only LOADS .npy. This mirrors WildGS-SLAM's predict_img_features ->
_save_features pattern (WildGS-SLAM/src/utils/mono_priors/img_feature_extractors.py:85-140,165-178).

Bakes a per-frame PATCH-GRID feature tensor [gh, gw, C] fp16 to <out_dir>/<stem>_dino.npy.
We store the small grid (patch-14/16 => ~200x smaller than full-res); datasets/dataset.py
upsamples grid->(H,W) at load time (bilinear, matching WildGS depth_video.py:461).

Backbones:
  dinov2_vits14  (DEFAULT, WILDGS-FAITHFUL): torch.hub 'facebookresearch/dinov2' AUTO-downloads
                 (no license gate), patch-14, C=384, forward_features()['x_norm_patchtokens'].
                 This IS what WildGS-SLAM uses (img_feature_extractors.py:76-79,128-132).
  dinov2_vitb14  : C=768 (heavier, still auto-download).
  dinov3_vitb16  : OPT-IN UPGRADE (NOT WildGS). patch-16, C=768. Weights are LICENSE-GATED:
                 request access, download the checkpoint, pass --dinov3_weights <path> (+ optionally
                 --dinov3_repo <local clone>). Needs torch>=2.7. Will NOT auto-download.

Usage (frictionless, WildGS-faithful):
  python Addons/dino/generate_dino_features.py \
      --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png' \
      --out_dir data/CRCD/C1_001/dino --backbone dinov2_vits14
Usage (DINOv3 upgrade, after caching the gated weights):
  python Addons/dino/generate_dino_features.py --backbone dinov3_vitb16 \
      --dinov3_weights /content/drive/MyDrive/dino_cache/dinov3_vitb16.pth ...
"""
import os, glob, argparse
import numpy as np
import torch
from PIL import Image

# ImageNet normalization — identical to WildGS img_feature_extractors.py:112-113.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

DINOV2 = {'dinov2_vits14': 14, 'dinov2_vits14_reg': 14, 'dinov2_vitb14': 14, 'dinov2_vitb14_reg': 14}
DINOV3 = {'dinov3_vits16': 16, 'dinov3_vitb16': 16, 'dinov3_vitl16': 16}


def load_backbone(name, device, dinov3_repo=None, dinov3_weights=None):
    if name in DINOV2:
        # WildGS path: torch.hub auto-download, NO gate (img_feature_extractors.py:78).
        m = torch.hub.load('facebookresearch/dinov2', name).to(device).eval()
        return m, 14, 'dinov2'
    if name in DINOV3:
        assert dinov3_weights and os.path.isfile(dinov3_weights), (
            f"DINOv3 is LICENSE-GATED. Request access, download the checkpoint, and pass "
            f"--dinov3_weights <path> (got {dinov3_weights!r}). Or use --backbone dinov2_vits14.")
        repo = dinov3_repo or 'facebookresearch/dinov3'
        src = 'local' if os.path.isdir(repo) else 'github'
        m = torch.hub.load(repo, name, source=src, weights=dinov3_weights).to(device).eval()
        return m, 16, 'dinov3'
    raise ValueError(f"unknown backbone {name!r} (dinov2_* or dinov3_*)")


@torch.inference_mode()
def dense_grid(model, family, patch, img_pil, device, dtype):
    """RGB PIL -> patch grid [gh, gw, C] fp16 (NOT upsampled)."""
    W0, H0 = img_pil.size
    Hf, Wf = (H0 // patch) * patch, (W0 // patch) * patch          # crop to a multiple of patch
    img = img_pil.resize((Wf, Hf), Image.BILINEAR)                 # WildGS process_image :158-162
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype)
    gh, gw = Hf // patch, Wf // patch
    if family == 'dinov2':
        feat = model.forward_features(x)['x_norm_patchtokens']     # [1, gh*gw, C]  WildGS :129-130
    else:  # dinov3 — register-token-aware; dict if available else slice CLS+registers
        out = model.forward_features(x)
        if isinstance(out, dict) and 'x_norm_patchtokens' in out:
            feat = out['x_norm_patchtokens']
        else:
            tok = out if torch.is_tensor(out) else out['x_norm_clstoken'].new_empty(0)
            n_pref = 1 + int(getattr(model, 'num_register_tokens', 4))
            feat = tok[:, n_pref:, :]
    C = feat.shape[-1]
    grid = feat.reshape(gh, gw, C)
    return grid.float().half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', required=True)
    ap.add_argument('--rgb_glob', default='*left.png')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--backbone', default='dinov2_vits14')          # WildGS-faithful default
    ap.add_argument('--dinov3_repo', default=None, help='local clone of facebookresearch/dinov3')
    ap.add_argument('--dinov3_weights', default=None, help='path to the gated DINOv3 checkpoint')
    ap.add_argument('--fp32', action='store_true', help='disable fp16 (T4 NaN safety)')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32 if args.fp32 else torch.float16
    model, patch, family = load_backbone(args.backbone, device, args.dinov3_repo, args.dinov3_weights)
    if dtype == torch.float16:
        model = model.half()   # cast MODEL to fp16 too, else conv sees fp16 input vs fp32 bias ->
                               # "Input type (c10::Half) and bias type (float) should be the same".
                               # On T4 prefer --fp32 (fp16 DINO convs can NaN; vits14 fp32 is cheap).
    files = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    assert files, f'no frames matched {args.rgb_dir}/{args.rgb_glob}'
    print(f'[dino] backbone={args.backbone} family={family} patch={patch} dtype={dtype} '
          f'frames={len(files)}')
    C = None
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]            # e.g. CRCD 000123l, SemSup 000000left
        out = os.path.join(args.out_dir, f'{stem}_dino.npy')
        if os.path.exists(out):
            continue
        g = dense_grid(model, family, patch, Image.open(f).convert('RGB'), device, dtype)
        C = int(g.shape[-1]); np.save(out, g)                      # [gh, gw, C] fp16
        if i % 50 == 0:
            print(f'  [{i + 1}/{len(files)}] {stem} grid={g.shape}')
    # Sidecar manifest so the run-env loader can read C / patch / backbone WITHOUT importing torch.
    with open(os.path.join(args.out_dir, 'dino_manifest.txt'), 'w') as fh:
        fh.write(f'backbone={args.backbone}\nfamily={family}\npatch={patch}\nC={C}\nn_frames={len(files)}\n')
    print(f'[dino] DONE {len(files)} grids -> {args.out_dir}  (backbone={args.backbone}, C={C}). '
          f'Set uncertainty.dino_dim: {C} in the dino config.')


if __name__ == '__main__':
    main()
