#!/bin/bash
# ============================================================================
# train_seg_head_crcd_20260618.sh  -  Arm 4 item A4-0.3 (RUN step).
#   Train ONE CRCD 4-class DINOv2 seg-head -> dinov2_crcd.pth, loadable by BOTH
#   SNI-SLAM and SemGauss-SLAM (same DINO2SEG arch; see CONTRACT.md s7).
#
#   RAW FRAMES (user, 2026-06-18): train on the ORIGINAL left frames + native masks
#   (rgb/ + semantic_instance/), NOT the rectified video_frames/semantic_class -
#   that is the domain the CRCD segmentations were annotated on. No rectification
#   step here (fast; perfectly aligned RGB<->label).
#
#   HELD-OUT (user, 2026-06-18): TRAIN on the 15 NON-benchmark snippets; the 5 benchmark
#   snippets (C1_001 C2_001 E3_005 C3_001 G3_001) are NEVER trained -> held-out test.
#
#   RUNS ON COLAB (A100/T4). Needs only torch+cv2+numpy (Colab-native) + the vendored
#   dinov2 (SemGauss clone) + torch.hub internet. Does NOT need depth or calib.
#
# Usage (on the tunnel, in /content/DDS-SLAM):
#   SKIP_ENV=1 bash Addons/colab/train_seg_head_crcd_20260618.sh
#   EPOCHS=2 SNIPPETS="B2_001 F3_001" SKIP_ENV=1 bash ...   # quick wiring check
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}; cd "$REPO"

# 15 NON-benchmark CRCD snippets = TRAIN; 5 benchmark = held-out TEST (never trained).
SNIPPETS=${SNIPPETS:-"B2_001 E1_001 E3_001 E3_002 E3_003 E3_004 F1_002 F3_001 F3_002 F3_003 F3_004 F3_005 F3_006 F3_007 G2_003"}
TEST_SNIPPETS=${TEST_SNIPPETS:-"C1_001 C2_001 E3_005 C3_001 G3_001"}
IMG_H=${IMG_H:-504}; IMG_W=${IMG_W:-896}     # training res (params are res-agnostic; deploy res independent)
EPOCHS=${EPOCHS:-40}; LR=${LR:-1e-4}
OUT_PTH=${OUT_PTH:-$REPO/seg/dinov2_crcd.pth}
OUT_DRIVE=${OUT_DRIVE:-/content/drive/MyDrive/Outputs/seg}
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
SEGDATA=${SEGDATA:-/content/crcd_seg_data}   # local copy of raw rgb+semantic_instance (no FUSE during train)

echo "=== train_seg_head_crcd $(date -Iseconds) ==="
echo "  TRAIN(15)=$SNIPPETS"
echo "  TEST(held-out)=$TEST_SNIPPETS  res=${IMG_H}x${IMG_W} epochs=$EPOCHS"
if [ "${SKIP_ENV:-0}" = 1 ]; then
  echo "[env] SKIP_ENV=1 -> skipping colab_setup; verifying minimal deps"
  python -c "import torch, cv2, numpy" 2>/dev/null || pip install -q opencv-contrib-python numpy
else
  bash Addons/env/colab_setup.sh --skip-data --skip-tunnel >/dev/null 2>&1 || true
fi
export XFORMERS_DISABLED=${XFORMERS_DISABLED:-1}   # dinov2 attn falls back if xformers absent

# --- locate a vendored dinov2 main (prefer SemGauss's in-repo copy: deterministic) ----------
if [ -z "${DINOV2_MAIN:-}" ]; then
  for c in /content/SemGauss-SLAM/segmentation/facebookresearch_dinov2_main \
           /content/sni-slam/seg/facebookresearch_dinov2_main \
           /content/SNI-SLAM/seg/facebookresearch_dinov2_main; do
    [ -f "$c/dinov2/models/vision_transformer.py" ] && DINOV2_MAIN="$c" && break
  done
fi
[ -n "${DINOV2_MAIN:-}" ] && [ -d "$DINOV2_MAIN" ] || {
  echo "FATAL: no vendored dinov2 main found. Clone SemGauss-SLAM or set DINOV2_MAIN=..."; exit 30; }
echo "dinov2 main: $DINOV2_MAIN"

# snippet NAME -> "EP SID": C1_001 -> C_1 001 ; F3_007 -> F_3 007
ep_sid(){ local n=$1
  [[ "$n" =~ ^[A-Za-z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }
  echo "${n:0:1}_${n:1:1} ${n:3}"; }

# --- stage RAW left rgb + raw semantic_instance to local SSD (NO rectification) -------------
stage_raw(){ local NAME=$1 dst="$SEGDATA/$NAME"
  if [ -d "$dst/rgb" ] && [ "$(ls "$dst/semantic_instance"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$NAME] already staged"; return 0; fi
  local EP SID; read -r EP SID <<< "$(ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$NAME] unparseable name -> skip"; return 1; }
  local SRC=""
  if [ -d "$DRIVE_CRCD/$EP/snippet_$SID/rgb" ]; then
    SRC="$DRIVE_CRCD/$EP/snippet_$SID"
  else
    local T="$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar" ex="/content/crcd_raw/${EP}_snippet_${SID}"
    if [ -f "$T" ]; then mkdir -p "$ex"; tar xf "$T" -C "$ex"
      SRC="$(dirname "$(find "$ex" -type d -name rgb 2>/dev/null | head -1)")"; fi
  fi
  [ -n "$SRC" ] && [ -d "$SRC/rgb" ] && [ -d "$SRC/semantic_instance" ] || {
    echo "[$NAME] raw rgb/semantic_instance not on Drive ($EP/snippet_$SID) -> skip"; return 1; }
  mkdir -p "$dst"
  cp -rn "$SRC/rgb" "$dst/rgb" && cp -rn "$SRC/semantic_instance" "$dst/semantic_instance"
  echo "[$NAME] staged $(ls "$dst/rgb"/*.png 2>/dev/null | wc -l) raw frames"
}

for NAME in $SNIPPETS $TEST_SNIPPETS; do stage_raw "$NAME" || true; done

# --- snippets that actually have raw (rgb + semantic_instance) on local disk ----------------
have(){ [ "$(ls "$SEGDATA/$1/semantic_instance"/*.png 2>/dev/null | wc -l)" -gt 0 ]; }
TRAIN_SNIPS=""; for NAME in $SNIPPETS;      do have "$NAME" && TRAIN_SNIPS="$TRAIN_SNIPS $NAME"; done
TEST_SNIPS="";  for NAME in $TEST_SNIPPETS; do have "$NAME" && TEST_SNIPS="$TEST_SNIPS $NAME"; done
[ -n "$TRAIN_SNIPS" ] || { echo "FATAL: no staged TRAIN snippets (put the 15 raw on Drive)"; exit 1; }
echo "TRAIN on:$TRAIN_SNIPS"
echo "TEST (held-out) on:${TEST_SNIPS:- (none staged -> held-out mIoU skipped)}"
TEST_ARG=""; [ -n "$TEST_SNIPS" ] && TEST_ARG="--test_snippets $TEST_SNIPS"

# --- train (default subdirs = raw rgb / semantic_instance, paired by basename) --------------
mkdir -p "$(dirname "$OUT_PTH")" "$OUT_DRIVE"
python Addons/seg/train_dinov2_crcd.py \
  --crcd_root "$SEGDATA" --snippets $TRAIN_SNIPS $TEST_ARG \
  --dinov2_main "$DINOV2_MAIN" --out "$OUT_PTH" \
  --rgb_subdir rgb --label_subdir semantic_instance --rgb_glob '*.png' --label_glob '*.png' \
  --n_classes 4 --dim 16 --img_h "$IMG_H" --img_w "$IMG_W" --crop_edge 0 \
  --epochs "$EPOCHS" --lr "$LR" --val_frac 0.1 \
  2>&1 | tee "$OUT_DRIVE/train_dinov2_crcd.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: training exited $rc"; exit "$rc"; }

cp -f "$OUT_PTH" "$OUT_DRIVE/dinov2_crcd.pth" 2>/dev/null || true
echo ""
echo "DONE -> $OUT_PTH  (shipped to $OUT_DRIVE/dinov2_crcd.pth + train log)"
echo "  val_mIoU = in-train held-out split; HELD-OUT TEST mIoU = benchmark snippets (never trained)."
echo "NEXT: point SNI model.cnn.pretrained_model_path AND SemGauss model.pretrained_model_path"
echo "      at dinov2_crcd.pth; both CRCD configs must use n_classes=4, c_dim=16, crop_edge=0."
