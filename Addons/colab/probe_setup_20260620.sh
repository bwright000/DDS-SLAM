#!/usr/bin/env bash
# LIGHTWEIGHT de-risk PROBE setup (NO SLAM env / NO tinycudann — ~5-10 min):
#   stage CRCD rgb + seg  ->  bake DINOv2 vits14-reg (torch2 + torch.hub)  ->  run the DINO-separability probe.
# Answers GO/NO-GO on the learned what-kind head (does frozen DINO separate tissue/tool/bg?).
#   cd /content/DDS-SLAM-probe && bash Addons/colab/probe_setup_20260620.sh
set -uo pipefail
REPO=$(pwd); CRCD=$REPO/data/CRCD/C1_001
SRC=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
say(){ echo -e "\n>>> $*"; }

[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted (mount it in a notebook cell first)"; exit 1; }
[ -d "$SRC" ] || { say "FATAL: CRCD-Published missing at $SRC — wrong Drive account? Datasets dir holds:"; ls /content/drive/MyDrive/Datasets 2>/dev/null; exit 1; }
pip install -q opencv-python-headless scikit-learn 2>/dev/null || true

# 1. stage rgb (-> video_frames/*l.png) + seg (semantic_class preferred = CLASS labels for the what-kind) -> masks/
say "[1] stage CRCD rgb + seg"
RGB=$([ -d "$SRC/rgb" ] && echo "$SRC/rgb" || echo "$SRC/frame")     # 'frame' fallback (CRCD-Published naming)
[ -d "$RGB" ] || { say "FATAL: no rgb/ or frame/ under $SRC — it holds:"; ls "$SRC"; exit 1; }
mkdir -p "$CRCD/video_frames" "$CRCD/masks"
i=0; for f in $(ls "$RGB"/*.png 2>/dev/null | sort); do printf -v n '%06dl.png' "$i"; cp "$f" "$CRCD/video_frames/$n"; i=$((i+1)); done
say "  rgb: $(ls "$CRCD/video_frames"/*l.png 2>/dev/null | wc -l) frames"
SEGSRC=$([ -d "$SRC/semantic_class" ] && echo "$SRC/semantic_class" || echo "$SRC/semantic_instance")
say "  seg source: $SEGSRC  (semantic_class = the {bg,liver,gallbladder,tool} CLASS labels the probe wants)"
j=0; for sf in $(ls "$SEGSRC"/*.png 2>/dev/null | sort); do printf -v sn '%06dl.png' "$j"; cp "$sf" "$CRCD/masks/$sn"; j=$((j+1)); done
say "  seg: $(ls "$CRCD/masks"/*l.png 2>/dev/null | wc -l) masks"

# 2. bake DINOv2 vits14-reg (torch2 + torch.hub download; ~minutes; NO tinycudann/SLAM env needed)
say "[2] bake DINOv2 vits14-reg (~minutes, first run downloads the backbone)"
python Addons/dino/generate_dino_features.py --rgb_dir "$CRCD/video_frames" --rgb_glob '*l.png' \
  --out_dir "$CRCD/dino_reg" --backbone dinov2_vits14_reg --fp32 2>&1 | tail -4
NPY=$(ls "$CRCD/dino_reg"/*_dino.npy 2>/dev/null | wc -l); say "  dino npy: $NPY"
[ "$NPY" -ge 1 ] || { say "FATAL: DINO bake produced no .npy"; exit 1; }

# 3. run the de-risk probe (prints the seg label histogram -> verify --tool_labels matches the real labels)
say "[3] RUN the DINO-separability probe"
python Addons/seg/dino_separability_probe.py --dino_dir "$CRCD/dino_reg" --dino_glob '*_dino.npy' \
  --seg_dir "$CRCD/masks" --seg_glob '*.png' --tool_labels 3 --bg_labels 0
say "DONE. Read the VERDICT line: gap>0.2 = GO (build the learned what-kind), gap~0 = NO-GO (ship sigma^2-stabilisation only)."
