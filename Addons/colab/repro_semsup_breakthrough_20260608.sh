#!/usr/bin/env bash
# =====================================================================
# SemSup BREAKTHROUGH-REPRO: 5 variants, run the SAME way as the
# 2026-06-01 depthsweep (the run that scored ~30 PSNR full-frame).
# 2026-06-08
# =====================================================================
# WHY: the 2026-06-08 rerun scored ~22 (full-frame via render_all_frames),
# but the breakthrough scored ~30 (full-frame via the IN-TRAINING renders
# ddsslam.py writes during the run). Same eval + same GT reproduce both
# (verified), so the gap is the RENDER (method/pose/ckpt), not eval/GT.
# This run reproduces the breakthrough way AND renders each ckpt post-hoc,
# so we can compare in-training vs render_all_frames on the SAME ckpt.
#
# The ONLY intended change vs the breakthrough: output depth saved properly.
# (Breakthrough config had cam.output_depth_scale=None -> fell back to
#  png_depth_scale=8 -> collapsed depth PNGs. Current Super.yaml sets
#  output_depth_scale=10000, so depth is now saved correctly. render_freq is
#  already 1 in Super.yaml, matching the breakthrough's 151 in-training renders.)
#
# PREREQS: Drive mounted; repo pulled; env restored:
#   bash Addons/env/colab_exact_env.sh --skip-data
# RUN (each line separately to avoid Colab paste-mangling):
#   cd /content/DDS-SLAM && git pull
#   nohup bash Addons/colab/repro_semsup_breakthrough_20260608.sh >/content/repro.out 2>&1 &
#   tail -f /content/repro.out      # (or the Drive runbook.log)
# EVAL: done locally afterwards (in-training renders vs GT; expect ~30).
# =====================================================================
set -uo pipefail
DATE=20260608
REPO=/content/DDS-SLAM
SEMSUP_DS=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
SEMSUP_CKPT_DIR=$SEMSUP_DS/checkpoints
SEMSUP_MOGE_DIR=$SEMSUP_DS/depth/MoGe2_trail3_${DATE}
RUN_OUT=/content/drive/MyDrive/Outputs/DDS-SLAM_semsup-breakthrough-repro_${DATE}
mkdir -p "$SEMSUP_CKPT_DIR" "$RUN_OUT"
LOG=$RUN_OUT/runbook.log
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
[ -d /content/drive/MyDrive ] || { echo "FATAL: Drive not mounted"; exit 1; }
say "=== SemSup breakthrough-repro start $(date -Iseconds) === HEAD $(git rev-parse --short HEAD 2>/dev/null)"

# stage SemSup data (Drive trial_3 -> repo trail_3 typo per inventory)
if [ ! -d "$REPO/data/Super/trail_3/rgb" ]; then
  mkdir -p "$REPO/data/Super"; say "staging SemSup -> data/Super/trail_3"; cp -r "$SEMSUP_DS" "$REPO/data/Super/trail_3"
fi

# --- Phase 1: MoGe-2 depth for the moge2 variant (SYSTEM python torch>=2; reuse if present) ---
PYSYS=$(command -v python3)
if [ -z "$(ls "$SEMSUP_MOGE_DIR"/*left_depth.npy 2>/dev/null | head -1)" ]; then
  say "regenerating SemSup MoGe-2 depth (ref-anchored to depth/ref, scale 8)"
  "$PYSYS" -c "import torch,sys;sys.exit(0 if torch.__version__>='2' else 1)" 2>/dev/null || say "WARN: system python torch<2 -> MoGe may fail"
  "$PYSYS" -c "import moge" 2>/dev/null || "$PYSYS" -m pip install -q "git+https://github.com/microsoft/MoGe.git" 2>&1 | tail -2 | tee -a "$LOG"
  mkdir -p "$SEMSUP_MOGE_DIR"
  "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$REPO/data/Super/trail_3/rgb" \
    --ref "$REPO/data/Super/trail_3/depth/ref" --out "$SEMSUP_MOGE_DIR" \
    --temporal_window 5 --depth_scale 8 2>&1 | tee -a "$LOG" || say "WARN: MoGe gen failed (moge2 variant will be skipped if depth missing)"
else
  say "MoGe depth already on Drive ($(ls "$SEMSUP_MOGE_DIR"/*left_depth.npy 2>/dev/null|wc -l) npy) -> skip gen"
fi
cp -rn "$SEMSUP_MOGE_DIR" "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}" 2>/dev/null || true

# --- Phase 2: train the 5 variants the breakthrough way (dds_env / torch 1.10) ---
[ -f /tmp/dds_env/bin/activate ] || { say "FATAL: /tmp/dds_env missing (run colab_exact_env.sh --skip-data)"; exit 1; }
source /tmp/dds_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
python -c "import torch,tinycudann,marching_cubes;print('env OK',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { say "FATAL: dds_env import failed (cache/GPU-arch mismatch)"; exit 1; }

declare -A CFGS
CFGS[paperfaith]=configs/Super/trail3_paper_faithful.yaml
CFGS[paperfaith-v2]=configs/Super/trail3_paper_faithful_v2.yaml
CFGS[variantA-stereo]=configs/Super/trail3_variant_a_stereo.yaml
CFGS[variantC-stereo]=configs/Super/trail3_variant_c_stereo.yaml
CFGS[moge2]=configs/Super/trail3_moge2.yaml
OVR=configs/Super/_repro_${DATE}; mkdir -p "$OVR"

for KEY in paperfaith paperfaith-v2 variantA-stereo variantC-stereo moge2; do
  BASE=${CFGS[$KEY]}; NAME=DDS-SLAM_semsup-${KEY}_${DATE}; KDIR=$RUN_OUT/$NAME; mkdir -p "$KDIR"
  [ -f "$KDIR/.DONE" ] && { say "$NAME done -> skip"; continue; }
  CFG=$BASE
  if [ "$KEY" = moge2 ]; then
    CFG=$OVR/moge2.yaml
    cat > "$CFG" <<YAMLEOF
inherit_from: $BASE
data:
  depth_subdir: depth/MoGe2_trail3_${DATE}
YAMLEOF
    [ -n "$(ls "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}"/*left_depth.npy 2>/dev/null|head -1)" ] || { say "  SKIP moge2 (no MoGe depth)"; continue; }
  fi
  say "TRAIN (breakthrough way) $NAME  ($BASE)"
  python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || { say "  WARN: train failed $KEY"; continue; }
  OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
  EXP=$(python -c "from config import load_config;print(load_config('$CFG')['data']['exp_name'])")
  # collect IN-TRAINING renders (the breakthrough deliverable) + proper output depth + debug + ATE
  mkdir -p "$KDIR/renders_intraining"
  cp "$REPO/$OUT"/[0-9]*.jpg "$KDIR/renders_intraining/" 2>/dev/null
  cp -r "$REPO/$OUT/depth" "$KDIR/output_depth" 2>/dev/null
  cp "$REPO/$OUT/$EXP/output.txt" "$KDIR/" 2>/dev/null || true
  cp -r "$REPO/$OUT/$EXP/debug" "$KDIR/debug" 2>/dev/null || true
  CKPT=$(ls -t "$REPO/$OUT/$EXP"/checkpoint*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    cp "$CKPT" "$SEMSUP_CKPT_DIR/$NAME.pt"; say "  ckpt -> $SEMSUP_CKPT_DIR/$NAME.pt"
    # DIAGNOSTIC: post-hoc render the SAME ckpt -> isolates in-training vs render_all_frames
    python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
      --output_dir "$KDIR/renders_posthoc" --save_depth --save_gt 2>&1 | tee -a "$LOG" || say "  WARN: render_all_frames failed"
  fi
  say "  in-training=$(ls "$KDIR/renders_intraining"/*.jpg 2>/dev/null|wc -l) jpg | output_depth=$(ls "$KDIR/output_depth"/*.png 2>/dev/null|wc -l) | posthoc=$(ls "$KDIR/renders_posthoc"/[0-9]*.png 2>/dev/null|wc -l) png"
  touch "$KDIR/.DONE"
done

say "=== done $(date -Iseconds) ==="
say "EVAL LOCALLY (same GT for both): in-training renders -> expect ~30 ; render_all_frames -> expect ~22."
say "  per variant: $RUN_OUT/<NAME>/{renders_intraining/*.jpg, renders_posthoc/*.png + *_gt.png, output_depth/*.png}"
say "  ckpts: $SEMSUP_CKPT_DIR/DDS-SLAM_semsup-*_${DATE}.pt"
