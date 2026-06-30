#!/bin/bash
# ============================================================================
# train_dinov2_seg_loso_t4_20260628.sh  -  DINOv2 seg-head LOSO, OVERNIGHT on a T4.
#
# WHY: the DINOv3 seg runs (Drive seg/loso_dinov3_max + loso_dinov3_b2, 2026-06-23) used the
# IMPROVED recipe, but the only DINOv2 LOSO we have (loso_ref) is the OLD baseline recipe
# (train_blocks=8 / ce / no-aug) -> NOT comparable. This runs DINOv2 at the SAME recipe as the
# dinov3 variants so DINOv2-vs-DINOv3 is apples-to-apples:
#     v2_max  = frozen backbone, head-only + aug + wce_dice   (mirror of loso_dinov3_max)
#     v2_b2   = unfreeze the last 2 transformer blocks + aug + wce_dice (mirror of loso_dinov3_b2)
#
# Everything else matches the dinov3 LOSO defaults: 5-fold leave-one-snippet-out over the benchmark
# (C1/C2/C3/G3 = true cross-episode; E3_005 in-domain, reported separately), inner-val G2_003+F3_007,
# frame_stride 2, 25 epochs, patience 6, 504x896, lr 1e-4. ONLY the backbone (dinov2/14) + train_blocks differ.
#
# Reuses the validated LOSO runbook (train_seg_head_crcd_loso_20260619.sh) which stages CRCD-Published
# from Drive, auto-clones facebookresearch/dinov2, trains, and is RESUME-SAFE (skips folds with a ckpt +
# HELD-OUT TEST). Seg training needs only torch+cv2+numpy (Colab-native) -> NO colab_setup / tinycudann.
#
# T4 budget: frozen (v2_max) ~1.5-2 h/fold -> 5 folds ~ one overnight. v2_b2 has backbone grads (~2-3x)
# -> likely a second night; just re-launch, it resumes. Bump FRAME_STRIDE for a faster (less faithful) pass.
#
# Usage (overnight; survives tunnel drops):
#   nohup bash Addons/colab/train_dinov2_seg_loso_t4_20260628.sh &> /content/v2seg.out & disown
#   tail -f /content/v2seg.out
# Variants:
#   RECIPES="v2_max"           bash ...   # only the cheap frozen recipe (guaranteed overnight)
#   FRAME_STRIDE=3             bash ...   # ~1.5x faster, slightly less faithful than dinov3's stride 2
#   FOLDS="C1_001 C2_001"      bash ...   # a subset of folds
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}; cd "$REPO"
LOSO="$REPO/Addons/colab/train_seg_head_crcd_loso_20260619.sh"
[ -f "$LOSO" ] || { echo "FATAL: missing $LOSO (git pull)"; exit 1; }
LOG=/content/drive/MyDrive/Outputs/seg/v2_overnight_$(date +%Y%m%d).log
mkdir -p "$(dirname "$LOG")"; exec > >(tee -a "$LOG") 2>&1

echo "=== DINOv2 seg-head LOSO overnight (T4)  $(date -Iseconds)  HEAD=$(git rev-parse --short HEAD 2>/dev/null) ==="
# seg training is light: torch+cv2+numpy only (all Colab-native) -> skip the heavy SLAM env build
python -c "import torch,cv2,numpy" 2>/dev/null || pip install -q opencv-contrib-python numpy
export SKIP_ENV=1

# shared recipe (identical to the dinov3 LOSO runs; env-overridable)
export BACKBONE=dinov2 BACKBONE_WEIGHTS=url
export LOSS=${LOSS:-wce_dice} AUG=${AUG:-1} GIN=${GIN:-0} LINEAR_HEAD=0 TUNE_NORMS=0
export VAL_SNIPS=${VAL_SNIPS:-"G2_003 F3_007"} FRAME_STRIDE=${FRAME_STRIDE:-2}
export PATIENCE=${PATIENCE:-6} EPOCHS=${EPOCHS:-25} LR=${LR:-1e-4} IMG_H=${IMG_H:-504} IMG_W=${IMG_W:-896}
export FOLDS=${FOLDS:-"C1_001 C2_001 C3_001 G3_001 E3_005"}

RECIPES=${RECIPES:-"v2_max v2_b2"}     # frozen first (finishes overnight), then blocks2
for R in $RECIPES; do
  case $R in
    v2_max) TB=0 ;;    # frozen head-only      (mirror loso_dinov3_max)
    v2_b2)  TB=2 ;;    # unfreeze last 2 blocks (mirror loso_dinov3_b2)
    *) echo "skip unknown recipe $R (use v2_max|v2_b2)"; continue ;;
  esac
  echo ""; echo "########## RECIPE $R  backbone=dinov2 train_blocks=$TB  $(date -Iseconds) ##########"
  TAG=$R TRAIN_BLOCKS=$TB bash "$LOSO" || echo "[$R] LOSO returned nonzero -> resume-safe, re-launch to continue"
done

echo ""; echo "=== DONE $(date -Iseconds) ==="
echo "Results: /content/drive/MyDrive/Outputs/seg/loso_v2_max + loso_v2_b2 (per-fold .pth + fold_*.log + LOSO SUMMARY)."
echo "Compare vs DINOv3: seg/loso_dinov3_max + loso_dinov3_b2.  Headline = mean over the 4 cross-episode folds (E3_005 separate)."
