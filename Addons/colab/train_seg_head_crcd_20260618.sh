#!/bin/bash
# ============================================================================
# train_seg_head_crcd_20260618.sh  -  Arm 4 item A4-0.3 (RUN step).
#   Train ONE CRCD 4-class DINOv2 seg-head -> dinov2_crcd.pth, loadable by BOTH
#   SNI-SLAM and SemGauss-SLAM (same DINO2SEG arch; see CONTRACT.md s7).
#   RUNS ON COLAB (A100/T4). Does NOT need the delegated depth (seg head = RGB + GT mask).
#
# Prereqs on Colab:
#   - /content/DDS-SLAM synced (this repo); torch2 env (Addons/env/colab_setup.sh).
#   - Raw CRCD on Drive (CRCD-Published/<EP>/snippet_<SID> or *_staging.tar) + the calib pickle.
#   - A vendored dinov2 main: SemGauss-SLAM cloned (preferred, in-repo) OR SNI's downloaded seg/.
#
# Usage (one line on the tunnel):
#   bash Addons/colab/train_seg_head_crcd_20260618.sh
#   SNIPPETS="C1_001 C2_001" bash Addons/colab/train_seg_head_crcd_20260618.sh   # fast first head
#   DINOV2_MAIN=/content/SemGauss-SLAM/segmentation/facebookresearch_dinov2_main bash ...
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}; cd "$REPO"

SNIPPETS=${SNIPPETS:-"C1_001 C2_001 E3_005 C3_001 G3_001"}   # all 5 by default; override for a quick head
IMG_H=${IMG_H:-504}; IMG_W=${IMG_W:-896}     # training res (params are res-agnostic; deploy res independent)
EPOCHS=${EPOCHS:-40}; LR=${LR:-1e-4}
OUT_PTH=${OUT_PTH:-$REPO/seg/dinov2_crcd.pth}
OUT_DRIVE=${OUT_DRIVE:-/content/drive/MyDrive/Outputs/seg}
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
CALIB=${CALIB:-$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}

echo "=== train_seg_head_crcd $(date -Iseconds) | snippets=$SNIPPETS res=${IMG_H}x${IMG_W} ==="
bash Addons/env/colab_setup.sh --skip-data --skip-tunnel >/dev/null 2>&1 || true
# dinov2 layers fall back to non-xformers attention; silence the optional dep if absent.
export XFORMERS_DISABLED=${XFORMERS_DISABLED:-1}

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

ep_sid(){ case "$1" in
  C1_001) echo "C_1 001";; C2_001) echo "C_2 001";; E3_005) echo "E_3 005";;
  C3_001) echo "C_3 001";; G3_001) echo "G_3 001";; *) echo "";; esac; }

# --- preprocess each snippet (RGB + semantic_class) if absent; depth NOT needed here -------
for NAME in $SNIPPETS; do
  OUT="data/CRCD/$NAME"
  if [ -d "$OUT/video_frames" ] && [ "$(ls "$OUT/semantic_class"/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$NAME] already preprocessed"; continue; fi
  read -r EP SID <<< "$(ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$NAME] unknown snippet -> skip"; continue; }
  RAW="/content/crcd_raw/${EP}_snippet_${SID}"
  if [ ! -d "$RAW" ]; then
    mkdir -p "$RAW"; T="$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar"
    if [ -f "$T" ]; then tar xf "$T" -C "$RAW";
    elif [ -d "$DRIVE_CRCD/$EP/snippet_$SID" ]; then cp -r "$DRIVE_CRCD/$EP/snippet_$SID/." "$RAW/";
    else echo "[$NAME] raw not on Drive -> skip (cannot preprocess)"; continue; fi; fi
  [ -f "$CALIB" ] || { echo "FATAL: calib pickle missing: $CALIB"; exit 1; }
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir "$RAW" --calib_pkl "$CALIB" --output_dir "$OUT" \
    || { echo "[$NAME] preprocess FAIL"; continue; }
done

# --- snippets that actually have (RGB + semantic_class) on disk -----------------------------
TRAIN_SNIPS=""
for NAME in $SNIPPETS; do
  [ "$(ls "data/CRCD/$NAME/semantic_class"/*.png 2>/dev/null | wc -l)" -gt 0 ] && TRAIN_SNIPS="$TRAIN_SNIPS $NAME"
done
[ -n "$TRAIN_SNIPS" ] || { echo "FATAL: no preprocessed snippets to train on"; exit 1; }
echo "training on:$TRAIN_SNIPS"

# --- train ---------------------------------------------------------------------------------
mkdir -p "$(dirname "$OUT_PTH")" "$OUT_DRIVE"
python Addons/seg/train_dinov2_crcd.py \
  --crcd_root data/CRCD --snippets $TRAIN_SNIPS \
  --dinov2_main "$DINOV2_MAIN" --out "$OUT_PTH" \
  --n_classes 4 --dim 16 --img_h "$IMG_H" --img_w "$IMG_W" --crop_edge 0 \
  --epochs "$EPOCHS" --lr "$LR" --val_frac 0.1 \
  2>&1 | tee "$OUT_DRIVE/train_dinov2_crcd.log"
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: training exited $rc"; exit "$rc"; }

cp -f "$OUT_PTH" "$OUT_DRIVE/dinov2_crcd.pth" 2>/dev/null || true
echo ""
echo "DONE -> $OUT_PTH  (shipped to $OUT_DRIVE/dinov2_crcd.pth + train log)"
echo "NEXT: point SNI model.cnn.pretrained_model_path AND SemGauss model.pretrained_model_path"
echo "      at dinov2_crcd.pth; both CRCD configs must use n_classes=4, c_dim=16, crop_edge=0."
echo "      On first load assert SemGauss load_state_dict(strict=True) succeeds and SNI reports"
echo "      0 dropped keys (CONTRACT.md s7) -- if not, retrain with that method's own dinov2 main."
