#!/usr/bin/env bash
# LIGHTWEIGHT flow-probe setup (NO SLAM env / NO tinycudann). Colab ships torch/torchvision/cv2/sklearn;
# DINO loads on-the-fly (torch.hub), RAFT from torchvision. GPU recommended (CPU works, slow).
# Stages CRCD rgb (+GT if present) -> runs feature_flow_probe -> ships the video+metrics to Drive.
#   cd /content/DDS-SLAM-probe && bash Addons/colab/flow_probe_setup_20260620.sh
# Overrides:  EP=C_1 SNIP=001 K=12 STRIDE=8 MAXF=80 BAKE=0
#   BAKE=1  -> pre-bake DINOv2-reg (faster for repeated runs; default uses on-the-fly DINO)
set -uo pipefail
REPO=$(pwd)
EP=${EP:-C_1}; SNIP=${SNIP:-001}; K=${K:-12}; STRIDE=${STRIDE:-8}; MAXF=${MAXF:-80}; BAKE=${BAKE:-0}
CRCD=$REPO/data/CRCD/${EP}_${SNIP}
SRC=/content/drive/MyDrive/Datasets/CRCD-Published/$EP/snippet_$SNIP
OUT=$REPO/output/flowprobe_${EP}_${SNIP}_k${K}
DRIVEOUT=/content/drive/MyDrive/Outputs/flowprobe_${EP}_${SNIP}_k${K}
say(){ echo -e "\n>>> $*"; }

[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted (mount it in a notebook cell first)"; exit 1; }
[ -d "$SRC" ] || { say "FATAL: CRCD-Published missing at $SRC — wrong Drive account? Datasets holds:"; ls /content/drive/MyDrive/Datasets 2>/dev/null; exit 1; }

# 1. stage rgb -> video_frames/*l.png  (+ groundtruth.txt if the snippet has one)
say "[1] stage CRCD rgb (+GT)"
RGB=$([ -d "$SRC/rgb" ] && echo "$SRC/rgb" || echo "$SRC/frame")
[ -d "$RGB" ] || { say "FATAL: no rgb/ or frame/ under $SRC — it holds:"; ls "$SRC"; exit 1; }
mkdir -p "$CRCD/video_frames"
i=0; for f in $(ls "$RGB"/*.png 2>/dev/null | sort); do printf -v n '%06dl.png' "$i"; cp "$f" "$CRCD/video_frames/$n"; i=$((i+1)); done
say "  rgb: $(ls "$CRCD/video_frames"/*l.png 2>/dev/null | wc -l) frames"
GT=$(ls "$SRC"/groundtruth.txt "$SRC"/pose*/*.txt "$SRC"/*pose*.txt 2>/dev/null | head -1 || true)
if [ -n "${GT:-}" ]; then cp "$GT" "$CRCD/groundtruth.txt"; say "  GT staged from $GT ($(grep -cvE '^\s*#|^\s*$' "$CRCD/groundtruth.txt") rows)"; else say "  (no GT under $SRC — flow probe runs fine without it)"; fi

# 2. optional DINO bake (default: probe does it on-the-fly)
DINOARG=()
if [ "$BAKE" = 1 ]; then
  say "[2] bake DINOv2 vits14-reg (first run downloads the backbone)"
  python Addons/dino/generate_dino_features.py --rgb_dir "$CRCD/video_frames" --rgb_glob '*l.png' \
    --out_dir "$CRCD/dino_reg" --backbone dinov2_vits14_reg --fp32 2>&1 | tail -3
  DINOARG=(--dino_dir "$CRCD/dino_reg" --dino_glob '*_dino.npy')
else
  say "[2] (skip bake — DINO on-the-fly in the probe)"
fi

# 3. run the flow probe
say "[3] RUN feature_flow_probe (K=$K stride=$STRIDE max=$MAXF)"
python Addons/motion/feature_flow_probe.py \
  --rgb_dir "$CRCD/video_frames" --rgb_glob '*l.png' "${DINOARG[@]}" \
  --out_dir "$OUT" --n_groups "$K" --stride "$STRIDE" --max_frames "$MAXF"

# 3b. GT-timing check (if a GT was staged): does the flow camera estimate track GT camera motion over time?
if [ -f "$CRCD/groundtruth.txt" ]; then
  say "[3b] GT-timing correlation (flow camera proxy vs GT camera motion)"
  python Addons/motion/gt_timing.py --pairs_csv "$OUT/feature_flow_pairs.csv" \
    --gt "$CRCD/groundtruth.txt" --fx "${FX:-1096.7}" --out_fig "$OUT/gt_timing.png" || echo WARN-gt_timing
else
  say "[3b] (no GT staged -> skip timing check)"
fi

# 4. ship to Drive so the video + timing plot are viewable off-instance
say "[4] ship to Drive"
mkdir -p "$DRIVEOUT" && cp -f "$OUT"/feature_flow.mp4 "$OUT"/feature_flow_metrics.json \
  "$OUT"/feature_flow_pairs.csv "$OUT"/gt_timing.png "$DRIVEOUT/" 2>/dev/null
say "DONE -> $DRIVEOUT/ (feature_flow.mp4, gt_timing.png, metrics.json, pairs.csv). Open from Drive."
