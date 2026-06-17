#!/bin/bash
# ============================================================================
# ARM 3 (Terminal 3) — CRCD c1_001 METRIC depth via stereo-anchor every 120 frames. 2026-06-17.
# Run AFTER t4_setup_shared (verify-only env). Shares the T4 with the 2 SLAM terminals, so MoGe is
# THROTTLED: low resolution_level + a per-process VRAM cap so it can't starve the SLAM processes.
#
# Fixes the x9-10 scale at the source: MoGe is up-to-scale (~9x too large on a surgical FOV); SGBM on
# the rectified L/R pair is TRUE metric. We re-anchor every 120 frames (smooth linear ramp) and BAKE
# metric depth -> depth/moge2_stereo120/. Train on it with sc_factor=1 + a rescaled bound -> Sim3~1.
#
#   bash Addons/colab/crcd_depth_stereo120_20260617.sh
#   throttle knobs: MOGE_MEM_FRAC=0.4 (frac of 16GB) MOGE_RES=7 (lower=less VRAM) INTERVAL=120 THREADS=2
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
export DDS_DIR=/content/DDS-SLAM
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
CACHE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001
EP=C_1; SID=001; NAME=C1_001
RAW=/content/crcd_raw/${EP}_snippet_${SID}
STAGED=$DDS_DIR/data/CRCD/$NAME
INTERVAL=${INTERVAL:-120}
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128       # reduce fragmentation under the shared T4
LOG=/content/drive/MyDrive/Outputs/crcd_depth_stereo120_${DATE}.log; mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
cd "$DDS_DIR"
say "=== ARM-3 stereo120 metric depth ($NAME) start $(date -Iseconds)  interval=$INTERVAL ==="
[ -f /content/.dds_setup_done ] || say "WARN: shared setup marker missing -> run t4_setup_shared first (proceeding, env may be unready)"
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python -c "import torch, cv2; assert torch.cuda.is_available()" || { say "FATAL: env not ready (run t4_setup_shared first)"; exit 1; }
python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || { say "FATAL: MoGe-2 not importable (setup did not install it)"; exit 1; }
[ -f "$CALIB_PKL" ] || { say "FATAL: calib pickle missing at $CALIB_PKL"; exit 1; }

# ---- 1. stage raw C_1/snippet_001 ----
say "[1] stage raw"
if [ ! -f "$RAW/.STAGED" ]; then
  mkdir -p "$RAW"
  DSRC=$DRIVE_CRCD/$EP/snippet_$SID
  [ -d "$DSRC" ] || { say "FATAL: raw not on Drive ($DSRC)"; exit 1; }
  for it in rgb rgbright semantic_instance groundtruth.txt intrinsics.yaml; do
    [ -e "$DSRC/$it" ] && cp -r "$DSRC/$it" "$RAW/$it" || say "  WARN missing $it"
  done
  touch "$RAW/.STAGED"
fi

# ---- 2. rectify ----
say "[2] preprocess (rectify)"
if [ ! -f "$STAGED/.PREPROCESSED" ]; then
  rm -rf "$STAGED"; mkdir -p "$STAGED"
  python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$RAW" --calib_pkl "$CALIB_PKL" --output_dir "$STAGED" \
    || { say "FATAL: preprocess failed"; exit 1; }
  [ -f "$STAGED/groundtruth.txt" ] || cp "$RAW/groundtruth.txt" "$STAGED/groundtruth.txt"
  touch "$STAGED/.PREPROCESSED"
fi
[ -f "$STAGED/rectified_calib.txt" ] || { say "FATAL: rectified_calib.txt missing"; exit 1; }
N_L=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
say "  rectified left frames: $N_L"

# ---- 3. MoGe-2 metric depth (THROTTLED: low res + VRAM cap so the 2 SLAM procs keep their share) ----
say "[3] MoGe-2 depth gen ($N_L frames) — MOGE_MEM_FRAC=${MOGE_MEM_FRAC:-0.4} MOGE_RES=${MOGE_RES:-7}"
if [ ! -f "$STAGED/depth/.DONE" ] || [ "$(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l)" -lt "$N_L" ]; then
  cd "$STAGED"; rm -rf _moge_in _moge_npy depth.tmp; mkdir -p _moge_in _moge_npy depth.tmp
  for f in video_frames/*l.png; do fid=$(basename "$f" l.png); ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"; done
  MOGE_MEM_FRAC=${MOGE_MEM_FRAC:-0.4} MOGE_RES=${MOGE_RES:-7} python - <<'PY' || { say "FATAL: MoGe gen failed"; cd "$DDS_DIR"; exit 1; }
import os, sys, runpy, torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get('MOGE_MEM_FRAC','0.4')), 0)  # cap this proc's VRAM
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
sys.argv=['g','--rgb','_moge_in','--out','_moge_npy','--temporal_window','1','--depth_scale','10000',
          '--max_depth_m','5.0','--resolution_level',os.environ.get('MOGE_RES','7')]
runpy.run_path(os.environ['DDS_DIR']+'/Addons/depth/generate_depth_moge.py', run_name='__main__')
PY
  python - <<'PY'
import numpy as np, cv2, glob, os
for p in sorted(glob.glob('_moge_npy/*-left_depth.npy')):
    fid=os.path.basename(p).split('-')[0]; out=f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    cv2.imwrite(out, np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('npy->png:', len(glob.glob('depth.tmp/*.png')))
PY
  PNG=$(ls depth.tmp/*.png 2>/dev/null | wc -l)
  [ "$PNG" -ge "$N_L" ] || { say "FATAL: depth count $PNG < $N_L"; cd "$DDS_DIR"; exit 1; }
  rm -rf depth && mv depth.tmp depth; sync; touch depth/.DONE
  rm -rf _moge_in _moge_npy; cd "$DDS_DIR"
fi

# ---- 4. stereo-anchor every $INTERVAL + smooth linear ramp -> METRIC depth corpus ----
say "[4] stereo120 metric anchor (SGBM every $INTERVAL + linear ramp)"
python Addons/depth/stereo120_metric_anchor.py --staged "$STAGED" --interval "$INTERVAL" \
  --in_scale 10000 --out_scale 10000 --max_depth_m 5.0 --out_subdir moge2_stereo120 \
  || { say "FATAL: stereo120 anchor failed"; exit 1; }
NB=$(ls "$STAGED/depth/moge2_stereo120"/*.png 2>/dev/null | wc -l)
say "  baked $NB metric PNGs -> $STAGED/depth/moge2_stereo120"

# ---- 5. cache the metric corpus + sc-track to Drive ----
say "[5] cache metric corpus to Drive: $CACHE/depth_stereo120"
mkdir -p "$CACHE/depth_stereo120"
cp -n "$STAGED/depth/moge2_stereo120"/*.png "$CACHE/depth_stereo120/" 2>/dev/null
cp "$STAGED/moge2_stereo120_sctrack.txt" "$CACHE/" 2>/dev/null
sync; touch "$CACHE/depth_stereo120/.DONE"; sync
say "=== ARM-3 stereo120 DONE $(date -Iseconds): $NB metric PNGs cached. ==="
echo "NEXT (next cycle, NOT tonight): build the matched metric config (bound/trunc/range_d x sc_f,"
echo "  sc_factor=1, depth_subdir=depth/moge2_stereo120) using the METRIC EXTENT printed above,"
echo "  then A/B metric-vs-canon on Sim3 scale + rigid-vs-Sim3 agreement + render/ATE."
