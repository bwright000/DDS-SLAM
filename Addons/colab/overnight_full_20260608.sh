#!/usr/bin/env bash
# =====================================================================
# FULL overnight (2026-06-08): SemSup breakthrough-repro (5 variants)
#                            + StereoMIS MoGe-2 back-4000
# =====================================================================
# Combines repro_semsup_breakthrough_20260608.sh + overnight_stereomis_20260608.sh
# into ONE run with the correct two-env ordering:
#   PHASE 1  ALL MoGe-2 depth gen   -> SYSTEM python (torch>=2)   [dds_env NOT active]
#   PHASE 2  ALL DDS-SLAM training  -> dds_env (torch 1.10)
# (Running the two separate runbooks back-to-back would FAIL: the SemSup one
#  leaves dds_env active, so the StereoMIS MoGe step would run on torch 1.10.)
#
# SemSup = the BREAKTHROUGH way (ddsslam.py -> in-training renders, scored ~30),
# with proper output depth (Super.yaml already has output_depth_scale=10000,
# render_freq=1). Each ckpt is ALSO rendered post-hoc (render_all_frames) so we
# can compare in-training vs post-hoc on the SAME ckpt (isolates the 30-vs-22 gap).
# StereoMIS = chunked MoGe (no OOM) + fail-fast depth + inline ATE.
#
# PREREQS (each line separately to avoid Colab paste-mangling):
#   from google.colab import drive; drive.mount('/content/drive')      # (cell)
#   cd /content && (git clone https://github.com/bwright000/DDS-SLAM.git || true) && cd DDS-SLAM && git checkout main && git pull
#   bash Addons/env/colab_exact_env.sh --skip-data
#   nohup bash Addons/colab/overnight_full_20260608.sh >/content/full.out 2>&1 &
#   tail -f /content/full.out
# =====================================================================
set -uo pipefail
DATE=20260608
REPO=/content/DDS-SLAM
SEMSUP_DS=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
STEREO_DS=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1
SEMSUP_CKPT_DIR=$SEMSUP_DS/checkpoints
STEREO_CKPT_DIR=$STEREO_DS/checkpoints
SEMSUP_MOGE_DIR=$SEMSUP_DS/depth/MoGe2_trail3_${DATE}
STEREO_MOGE_DIR=$STEREO_DS/depth/MoGe2_p2-1-back4000_${DATE}
RUN_OUT=/content/drive/MyDrive/Outputs/DDS-SLAM_full-overnight_${DATE}
SEM_OUT=$RUN_OUT/semsup; STE_OUT=$RUN_OUT/stereomis
mkdir -p "$SEMSUP_CKPT_DIR" "$STEREO_CKPT_DIR" "$SEM_OUT" "$STE_OUT"
LOG=$RUN_OUT/runbook.log
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
[ -d /content/drive/MyDrive ] || { echo "FATAL: Drive not mounted"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "FATAL: no GPU"; exit 1; }
say "=== FULL overnight start $(date -Iseconds) === GPU $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) | HEAD $(git rev-parse --short HEAD 2>/dev/null)"

# --- stage data ---
[ -d "$REPO/data/Super/trail_3/rgb" ] || { mkdir -p "$REPO/data/Super"; say "stage SemSup -> data/Super/trail_3"; cp -r "$SEMSUP_DS" "$REPO/data/Super/trail_3"; }
SNIMG=$( (cd "$REPO/data/Super/trail_3/rgb" 2>/dev/null && ls *left.png 2>/dev/null) | wc -l )
if [ ! -d /content/p2_1_local/video_frames ]; then
  mkdir -p /content/p2_1_local; STAR=$STEREO_DS/P2_1_staging.tar
  [ -f "$STAR" ] && { say "stage StereoMIS from tar"; tar xf "$STAR" -C /content/p2_1_local; } \
    || (cd "$STEREO_DS" && tar cf - video_frames groundtruth.txt StereoCalibration.ini masks 2>/dev/null | tar xf - -C /content/p2_1_local)
fi
( cd /content/p2_1_local/video_frames 2>/dev/null && ls *l.png 2>/dev/null | tail -4000 ) > /content/_p2_frames.txt
TNIMG=$(wc -l < /content/_p2_frames.txt)
say "staged: SemSup=$SNIMG frames, StereoMIS=$TNIMG frames"

# =====================================================================
# PHASE 1 — ALL MoGe-2 depth (SYSTEM python torch>=2; dds_env NOT active)
# =====================================================================
PYSYS=$(command -v python3)
"$PYSYS" -c "import torch,sys;sys.exit(0 if torch.__version__>='2' else 1)" 2>/dev/null || say "WARN: system python torch<2 -> MoGe may fail"
"$PYSYS" -c "import moge" 2>/dev/null || { say "installing moge (system python)"; "$PYSYS" -m pip install -q "git+https://github.com/microsoft/MoGe.git" 2>&1 | tail -2 | tee -a "$LOG"; }

# 1a. SemSup moge2 depth (ref-anchored to depth/ref)
if [ -z "$(ls "$SEMSUP_MOGE_DIR"/*left_depth.npy 2>/dev/null | head -1)" ]; then
  say "MoGe 1a: SemSup moge2 depth"; mkdir -p "$SEMSUP_MOGE_DIR"
  "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$REPO/data/Super/trail_3/rgb" \
    --ref "$REPO/data/Super/trail_3/depth/ref" --out "$SEMSUP_MOGE_DIR" \
    --temporal_window 5 --depth_scale 8 2>&1 | tee -a "$LOG" || say "WARN: SemSup MoGe failed"
else say "SemSup MoGe depth present -> skip"; fi
cp -rn "$SEMSUP_MOGE_DIR" "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}" 2>/dev/null || true

# 1b. StereoMIS back-4000 depth (CHUNKED -> no OOM; metric-direct, scale handled in png conv)
if [ "$(ls "$STEREO_MOGE_DIR"/*.png 2>/dev/null | wc -l)" -lt "$TNIMG" ] && [ "$TNIMG" -gt 0 ]; then
  say "MoGe 1b: StereoMIS back-4000 ($TNIMG frames, chunks of 500)"
  NPY=/content/_ste_npy; rm -rf "$NPY"; mkdir -p "$NPY" "$STEREO_MOGE_DIR"
  CH=500; i=0
  while [ "$i" -lt "$TNIMG" ]; do
    ST=/content/_ste_stage; rm -rf "$ST"; mkdir -p "$ST"
    sed -n "$((i+1)),$((i+CH))p" /content/_p2_frames.txt | while read f; do s="${f%l.png}"; ln -sf "/content/p2_1_local/video_frames/$f" "$ST/${s}-left.png"; done
    say "  StereoMIS MoGe chunk $i..$((i+CH)): $(ls "$ST"/*-left.png 2>/dev/null | wc -l) staged"
    "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$ST" --out "$NPY" --temporal_window 5 --depth_scale 1 2>&1 | tee -a "$LOG" || say "  WARN: chunk $i failed"
    i=$((i+CH))
  done
  "$PYSYS" - "$NPY" "$STEREO_MOGE_DIR" <<'PY'
import sys,glob,os,numpy as np,cv2
nd,pd=sys.argv[1],sys.argv[2]; n=0
for f in sorted(glob.glob(os.path.join(nd,'*-left_depth.npy'))):
    d=np.load(f).astype(np.float32); u=np.clip(d*10000,0,65535).astype(np.uint16)
    cv2.imwrite(os.path.join(pd,os.path.basename(f).replace('-left_depth.npy','')+'.png'),u); n+=1
print('StereoMIS PNGs:',n)
PY
else say "StereoMIS MoGe depth present (or 0 frames) -> skip"; fi

# =====================================================================
# PHASE 2 — ALL DDS-SLAM (dds_env / torch 1.10)
# =====================================================================
[ -f /tmp/dds_env/bin/activate ] || { say "FATAL: /tmp/dds_env missing (colab_exact_env.sh --skip-data)"; exit 1; }
source /tmp/dds_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
python -c "import torch,tinycudann,marching_cubes;print('env OK',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { say "FATAL: dds_env import failed (cache/GPU-arch mismatch)"; exit 1; }

# --- 2a. SemSup 5 variants (breakthrough way + post-hoc diagnostic) ---
declare -A CFGS
CFGS[paperfaith]=configs/Super/trail3_paper_faithful.yaml
CFGS[paperfaith-v2]=configs/Super/trail3_paper_faithful_v2.yaml
CFGS[variantA-stereo]=configs/Super/trail3_variant_a_stereo.yaml
CFGS[variantC-stereo]=configs/Super/trail3_variant_c_stereo.yaml
CFGS[moge2]=configs/Super/trail3_moge2.yaml
OVR=configs/Super/_repro_${DATE}; mkdir -p "$OVR"
for KEY in paperfaith paperfaith-v2 variantA-stereo variantC-stereo moge2; do
  BASE=${CFGS[$KEY]}; NAME=DDS-SLAM_semsup-${KEY}_${DATE}; KDIR=$SEM_OUT/$NAME; mkdir -p "$KDIR"
  [ -f "$KDIR/.DONE" ] && { say "$NAME done -> skip"; continue; }
  CFG=$BASE
  if [ "$KEY" = moge2 ]; then
    CFG=$OVR/moge2.yaml; printf 'inherit_from: %s\ndata:\n  depth_subdir: depth/MoGe2_trail3_%s\n' "$BASE" "$DATE" > "$CFG"
    [ -n "$(ls "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}"/*left_depth.npy 2>/dev/null | head -1)" ] || { say "  SKIP moge2 (no MoGe depth)"; continue; }
  fi
  say "SemSup TRAIN (breakthrough way) $NAME"
  python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || { say "  WARN: train failed $KEY"; continue; }
  OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
  EXP=$(python -c "from config import load_config;print(load_config('$CFG')['data']['exp_name'])")
  mkdir -p "$KDIR/renders_intraining"
  cp "$REPO/$OUT"/[0-9]*.jpg "$KDIR/renders_intraining/" 2>/dev/null
  cp -r "$REPO/$OUT/depth" "$KDIR/output_depth" 2>/dev/null
  cp "$REPO/$OUT/$EXP/output.txt" "$KDIR/" 2>/dev/null || true
  cp -r "$REPO/$OUT/$EXP/debug" "$KDIR/debug" 2>/dev/null || true
  CKPT=$(ls -t "$REPO/$OUT/$EXP"/checkpoint*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    cp "$CKPT" "$SEMSUP_CKPT_DIR/$NAME.pt"; say "  ckpt -> $SEMSUP_CKPT_DIR/$NAME.pt"
    python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
      --output_dir "$KDIR/renders_posthoc" --save_depth --save_gt 2>&1 | tee -a "$LOG" || say "  WARN: posthoc render $KEY"
  fi
  say "  $NAME: in-training=$(ls "$KDIR/renders_intraining"/*.jpg 2>/dev/null|wc -l) | posthoc=$(ls "$KDIR/renders_posthoc"/[0-9]*.png 2>/dev/null|wc -l) | output_depth=$(ls "$KDIR/output_depth"/*.png 2>/dev/null|wc -l)"
  touch "$KDIR/.DONE"
done

# --- 2b. StereoMIS MoGe back-4000 (fail-fast depth; inline ATE) ---
SNAME=DDS-SLAM_stereomis-p2-1-moge-back4000_${DATE}; SDIR=$STE_OUT/$SNAME; mkdir -p "$SDIR"
if [ -f "$SDIR/.DONE" ]; then say "StereoMIS already done -> skip"; else
  NDEP=$(ls "$STEREO_MOGE_DIR"/*.png 2>/dev/null | wc -l)
  say "StereoMIS depth check: $NDEP png vs $TNIMG frames"
  if [ "$TNIMG" -gt 0 ] && [ "$NDEP" -ge "$TNIMG" ]; then
    rm -rf /content/p2_1_local/depth; ln -sf "$STEREO_MOGE_DIR" /content/p2_1_local/depth
    CFG=configs/StereoMIS/p2_1_moge_back4000.yaml
    say "StereoMIS SLAM (4000 frames — the long pole, ~hours)"
    python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN: StereoMIS SLAM crashed"
    OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
    CKPT=$(ls -t "$REPO/$OUT/demo"/checkpoint*.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
      cp "$CKPT" "$STEREO_CKPT_DIR/$SNAME.pt"; say "  ckpt -> $STEREO_CKPT_DIR/$SNAME.pt"
      cp "$REPO/$OUT/demo/output.txt" "$SDIR/ate_output.txt" 2>/dev/null || true
      cp "$REPO/$OUT/demo"/pose_*.png "$SDIR/" 2>/dev/null || true
      python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
        --output_dir "$SDIR/rendered_all" --save_depth --save_gt 2>&1 | tee -a "$LOG" || say "  WARN: StereoMIS render"
      touch "$SDIR/.DONE"
    else say "  FATAL: no StereoMIS ckpt produced (no .DONE -> retries)"; fi
  else say "  FATAL: StereoMIS depth incomplete ($NDEP/$TNIMG) -> SKIP SLAM (no .DONE -> retries). Check Phase-1b MoGe."; fi
fi

say "=== FULL overnight done $(date -Iseconds) ==="
say "SemSup    : $SEM_OUT/<NAME>/{renders_intraining,renders_posthoc,output_depth}/  ckpts $SEMSUP_CKPT_DIR/"
say "StereoMIS : $STE_OUT/$SNAME/{ate_output.txt,rendered_all/}  ckpt $STEREO_CKPT_DIR/$SNAME.pt"
say "EVAL (local): SemSup in-training renders -> expect ~30 ; posthoc -> expect ~22 (same ckpt, isolates render gap)."
