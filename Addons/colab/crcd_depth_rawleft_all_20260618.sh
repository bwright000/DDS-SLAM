#!/bin/bash
# ============================================================================
# CRCD metric depth — ALL snippets, RAW-LEFT (no full rectification).  2026-06-18.
# Per user: skip rectifying every frame; MoGe on the RAW left frames, rectify ONLY the every-120
# anchor pairs to fix the per-snippet metric SCALE (the x9-10 fix), bake raw_moge*sc_f. Discovers
# every CRCD-Published snippet, .DONE-gated (c1_001 already metric-done -> skipped). MoGe throttled
# so it shares the GPU politely. Output cached to Drive for the SLAM (which then runs on raw-left
# rgb/ + raw intrinsics + this metric depth).
#
#   bash Addons/colab/crcd_depth_rawleft_all_20260618.sh
#   knobs: MOGE_MEM_FRAC=0.5 MOGE_RES=7 INTERVAL=120 THREADS=2 LIMIT=99 (cap #snippets this run)
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
export DDS_DIR=/content/DDS-SLAM
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
CACHE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2
INTERVAL=${INTERVAL:-120}; LIMIT=${LIMIT:-99}; THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
LOG=/content/drive/MyDrive/Outputs/crcd_depth_rawleft_${DATE}.log; mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
cd "$DDS_DIR"
say "=== CRCD raw-left metric depth (ALL snippets) start $(date -Iseconds)  interval=$INTERVAL ==="
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python -c "import torch, cv2, yaml; assert torch.cuda.is_available()" || { say "FATAL: env not ready (run t4_setup_shared first)"; exit 1; }
python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || { say "FATAL: MoGe-2 not importable"; exit 1; }
[ -f "$CALIB_PKL" ] || { say "FATAL: calib pkl missing at $CALIB_PKL"; exit 1; }

# ---- discover every snippet on Drive ----
mapfile -t SNIPS < <(find "$DRIVE_CRCD" -mindepth 2 -maxdepth 2 -type d -name 'snippet_*' | sort)
say "discovered ${#SNIPS[@]} snippets"
done_cnt=0
for SDIR in "${SNIPS[@]}"; do
  [ "$done_cnt" -ge "$LIMIT" ] && { say "LIMIT=$LIMIT reached -> stop"; break; }
  SID=$(basename "$SDIR"); SID=${SID#snippet_}
  EP=$(basename "$(dirname "$SDIR")")
  NAME="${EP//_/}_${SID}"
  CACHE=$CACHE_ROOT/$EP/snippet_$SID
  RAW=/content/crcd_raw/${EP}_snippet_${SID}
  MOUT=$DDS_DIR/data/CRCD/_rawdepth/${NAME}            # working: raw_moge + metric
  echo ""; echo "############ $NAME ($EP/snippet_$SID) ############"
  if [ -f "$CACHE/depth_rawleft_metric/.DONE" ]; then say "  $NAME cached metric -> skip"; continue; fi
  [ -d "$SDIR/rgb" ] && [ -d "$SDIR/rgbright" ] && [ -f "$SDIR/intrinsics.yaml" ] \
    || { say "  $NAME missing rgb/rgbright/intrinsics -> skip"; continue; }

  # ---- stage raw (left, right, intrinsics) ----
  if [ ! -f "$RAW/.STAGED" ]; then
    mkdir -p "$RAW"
    cp -rn "$SDIR/rgb" "$RAW/rgb" && cp -rn "$SDIR/rgbright" "$RAW/rgbright" && cp -n "$SDIR/intrinsics.yaml" "$RAW/" \
      || { say "  $NAME stage FAIL -> skip"; continue; }
    touch "$RAW/.STAGED"
  fi
  NL=$(ls "$RAW/rgb"/*.png 2>/dev/null | wc -l); say "  raw left frames: $NL"
  [ "$NL" -ge 2 ] || { say "  $NAME too few frames -> skip"; continue; }

  # ---- MoGe-2 on RAW-LEFT (throttled) -> depth_rawleft PNG (x10000), named by rgb stem ----
  mkdir -p "$MOUT/depth_rawleft"
  if [ "$(ls "$MOUT/depth_rawleft"/*.png 2>/dev/null | wc -l)" -lt "$NL" ]; then
    rm -rf "$MOUT/_in" "$MOUT/_npy"; mkdir -p "$MOUT/_in" "$MOUT/_npy"
    for f in "$RAW/rgb"/*.png; do st=$(basename "$f" .png); ln -sf "$f" "$MOUT/_in/${st}-left.png"; done
    MOGE_MEM_FRAC=${MOGE_MEM_FRAC:-0.5} MOGE_RES=${MOGE_RES:-7} MOUT="$MOUT" python - <<'PY' || { say "  $NAME MoGe FAIL -> skip"; continue; }
import os, sys, runpy, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MOGE_MEM_FRAC','0.5')), 0)
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
mo=os.environ['MOUT']
sys.argv=['g','--rgb',mo+'/_in','--out',mo+'/_npy','--temporal_window','1','--depth_scale','10000',
          '--max_depth_m','5.0','--resolution_level',os.environ.get('MOGE_RES','7')]
runpy.run_path(os.environ['DDS_DIR']+'/Addons/depth/generate_depth_moge.py', run_name='__main__')
PY
    MOUT="$MOUT" python - <<'PY'
import numpy as np, cv2, glob, os
mo=os.environ['MOUT']
for p in sorted(glob.glob(mo+'/_npy/*-left_depth.npy')):
    st=os.path.basename(p).replace('-left_depth.npy','')
    cv2.imwrite(f'{mo}/depth_rawleft/{st}.png', np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('moge png:', len(glob.glob(mo+'/depth_rawleft/*.png')))
PY
    rm -rf "$MOUT/_in" "$MOUT/_npy"
  fi
  MP=$(ls "$MOUT/depth_rawleft"/*.png 2>/dev/null | wc -l)
  [ "$MP" -ge "$NL" ] || { say "  $NAME moge count $MP < $NL -> skip"; continue; }

  # ---- stereo120 raw-left metric anchor (rectify anchors only) ----
  python Addons/depth/stereo120_rawleft.py \
    --rgb_left_dir "$RAW/rgb" --rgb_right_dir "$RAW/rgbright" \
    --moge_dir "$MOUT/depth_rawleft" --calib_pkl "$CALIB_PKL" --intrinsics_yaml "$RAW/intrinsics.yaml" \
    --interval "$INTERVAL" --out_dir "$MOUT/metric" || { say "  $NAME stereo120 FAIL -> skip"; continue; }
  NB=$(ls "$MOUT/metric"/*.png 2>/dev/null | wc -l)
  [ "$NB" -ge "$NL" ] || { say "  $NAME metric count $NB < $NL -> skip"; continue; }

  # ---- cache to Drive ----
  mkdir -p "$CACHE/depth_rawleft_metric"
  cp -n "$MOUT/metric"/*.png "$CACHE/depth_rawleft_metric/" 2>/dev/null
  sync; touch "$CACHE/depth_rawleft_metric/.DONE"; sync
  say "  $NAME DONE: $NB metric PNGs cached -> $CACHE/depth_rawleft_metric"
  rm -rf "$MOUT" "$RAW"      # free local space for the next snippet
  done_cnt=$((done_cnt+1))
done
say "=== CRCD raw-left metric depth DONE $(date -Iseconds): $done_cnt snippets this run ==="
echo "Cached under $CACHE_ROOT/<EP>/snippet_<SID>/depth_rawleft_metric/ (SLAM uses raw rgb/ + raw intrinsics + this)."
