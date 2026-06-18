#!/bin/bash
# ============================================================================
# train_seg_head_crcd_20260618.sh  -  Arm 4 item A4-0.3 (RUN step).
#   Train ONE CRCD 4-class DINOv2 seg-head -> dinov2_crcd.pth, loadable by BOTH
#   SNI-SLAM and SemGauss-SLAM (same DINO2SEG arch; see CONTRACT.md s7).
#
#   HELD-OUT DESIGN (user, 2026-06-18): TRAIN on the 15 NON-benchmark CRCD snippets;
#   the 5 benchmark snippets (C1_001 C2_001 E3_005 C3_001 G3_001) are NEVER trained on
#   -> they stay a true held-out test set for the SLAM benchmark, and we also report the
#   seg-head's held-out mIoU on them (generalization number). No train-on-eval leakage.
#
#   RUNS ON COLAB (A100/T4). Does NOT need the delegated depth (seg head = RGB + GT mask).
#
# Prereqs on Colab:
#   - /content/DDS-SLAM synced; torch2 env (Addons/env/colab_setup.sh).
#   - Raw CRCD on Drive (CRCD-Published/<EP>/snippet_<SID> or *_staging.tar) + calib pickle,
#     for BOTH the 15 train snippets AND (for the held-out mIoU) the 5 benchmark snippets.
#   - A vendored dinov2 main: SemGauss-SLAM cloned (preferred) OR SNI's downloaded seg/.
#
# Usage (one line on the tunnel):
#   bash Addons/colab/train_seg_head_crcd_20260618.sh
#   EPOCHS=20 bash Addons/colab/train_seg_head_crcd_20260618.sh        # shorter
#   SNIPPETS="B2_001 F3_001" TEST_SNIPPETS="C1_001" bash ...           # custom subset
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
CALIB=${CALIB:-$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}

echo "=== train_seg_head_crcd $(date -Iseconds) ==="
echo "  TRAIN(15)=$SNIPPETS"
echo "  TEST(held-out)=$TEST_SNIPPETS  res=${IMG_H}x${IMG_W} epochs=$EPOCHS"
bash Addons/env/colab_setup.sh --skip-data --skip-tunnel >/dev/null 2>&1 || true
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

# snippet NAME -> "EP SID" (general parser): C1_001 -> C_1 001 ; F3_007 -> F_3 007
ep_sid(){ local n=$1
  [[ "$n" =~ ^[A-Za-z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }
  echo "${n:0:1}_${n:1:1} ${n:3}"; }

# --- preprocess (RGB + semantic_class) if absent, for TRAIN + TEST; depth NOT needed --------
for NAME in $SNIPPETS $TEST_SNIPPETS; do
  OUT="data/CRCD/$NAME"
  if [ -d "$OUT/video_frames" ] && [ "$(ls "$OUT/semantic_class"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$NAME] already preprocessed"; continue; fi
  read -r EP SID <<< "$(ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$NAME] unparseable snippet name -> skip"; continue; }
  RAW="/content/crcd_raw/${EP}_snippet_${SID}"
  if [ ! -d "$RAW" ]; then
    mkdir -p "$RAW"; T="$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar"
    if [ -f "$T" ]; then tar xf "$T" -C "$RAW";
    elif [ -d "$DRIVE_CRCD/$EP/snippet_$SID" ]; then cp -r "$DRIVE_CRCD/$EP/snippet_$SID/." "$RAW/";
    else echo "[$NAME] raw not on Drive ($EP/snippet_$SID) -> skip"; continue; fi; fi
  [ -f "$CALIB" ] || { echo "FATAL: calib pickle missing: $CALIB"; exit 1; }
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir "$RAW" --calib_pkl "$CALIB" --output_dir "$OUT" \
    || { echo "[$NAME] preprocess FAIL"; continue; }
done

# --- snippets that actually have (RGB + semantic_class) on disk -----------------------------
have(){ [ "$(ls "data/CRCD/$1/semantic_class"/*.png 2>/dev/null | wc -l)" -gt 0 ]; }
TRAIN_SNIPS=""; for NAME in $SNIPPETS;       do have "$NAME" && TRAIN_SNIPS="$TRAIN_SNIPS $NAME"; done
TEST_SNIPS="";  for NAME in $TEST_SNIPPETS;  do have "$NAME" && TEST_SNIPS="$TEST_SNIPS $NAME"; done
[ -n "$TRAIN_SNIPS" ] || { echo "FATAL: no preprocessed TRAIN snippets (stage the 15 raw on Drive)"; exit 1; }
echo "TRAIN on:$TRAIN_SNIPS"
echo "TEST (held-out) on:${TEST_SNIPS:- (none staged -> held-out mIoU skipped)}"
TEST_ARG=""; [ -n "$TEST_SNIPS" ] && TEST_ARG="--test_snippets $TEST_SNIPS"

# --- train ---------------------------------------------------------------------------------
mkdir -p "$(dirname "$OUT_PTH")" "$OUT_DRIVE"
python Addons/seg/train_dinov2_crcd.py \
  --crcd_root data/CRCD --snippets $TRAIN_SNIPS $TEST_ARG \
  --dinov2_main "$DINOV2_MAIN" --out "$OUT_PTH" \
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
echo "      On first load assert SemGauss strict load succeeds and SNI reports 0 dropped keys."
