#!/usr/bin/env bash
# =====================================================================
# Rerun: SemSup (5 configs) + StereoMIS MoGe back-4000  — 2026-06-08
# =====================================================================
# Decisions (user 2026-06-08):
#   - SemSup: RETRAIN the 4 real-depth configs + moge2 (REGENERATE its depth,
#     don't skip).  variantB_ep9 still skipped (its depth is not MoGe and is
#     missing on Drive).
#   - StereoMIS: MoGe-2 back-4000.
#   - Checkpoints saved AS AN ELEMENT OF THE DATASET on Drive, new naming
#     convention  Model_Changes_YYYYMMDD  (Addons/docs/NAMING_CONVENTION.md).
#
# Two-env split (CRITICAL — see memory project_moge2_needs_torch2_not_dds_env):
#   PHASE 1  MoGe-2 depth generation  -> SYSTEM python (torch>=2 + moge pkg)
#   PHASE 2  DDS-SLAM training/eval   -> dds_env  (torch 1.10)
#
# Bugs fixed vs overnight_20260605.sh:
#   (1) overnight ran MoGe gen INSIDE dds_env (torch 1.10) -> would crash.
#   (2) generate_depth_moge.py globs '*-left.png'; StereoMIS frames are '*l.png'
#       -> overnight found 0 frames.  Here we symlink them as '*-left.png'.
#   (3) overnight converted npy->png with an extra *10000 even though npy was
#       already depth_m*depth_scale -> double scale.  Here MoGe runs with
#       --depth_scale 1 for StereoMIS (npy in metres), png = metres*10000 once.
#   (4) StereoMIS ATE eval called Addons/eval/eval_ate.py (does not exist).
#       ddsslam.py already computes ATE inline via tools.eval_ate.pose_evaluation
#       (writes output.txt + pose_*.png), so no separate call is needed.
#   (5) SemSup render eval failed on GT naming — fixed in eval_rendering.py
#       (GT-alongside *_gt.png mode).  This runbook relies on that fix.
# =====================================================================
set -uo pipefail

DATE=20260608
REPO=/content/DDS-SLAM
cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }

# ---- Drive locations -------------------------------------------------
SEMSUP_DS=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3      # dataset root
STEREO_DS=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1            # dataset root
SEMSUP_CKPT_DIR=$SEMSUP_DS/checkpoints
STEREO_CKPT_DIR=$STEREO_DS/checkpoints
SEMSUP_MOGE_DIR=$SEMSUP_DS/depth/MoGe2_trail3_${DATE}                 # depth-as-dataset-element
STEREO_MOGE_DIR=$STEREO_DS/depth/MoGe2_p2-1-back4000_${DATE}
RUN_OUT=/content/drive/MyDrive/Outputs/DDS-SLAM_semsup-stereomis-rerun_${DATE}   # bulky run artefacts
mkdir -p "$SEMSUP_CKPT_DIR" "$STEREO_CKPT_DIR" "$RUN_OUT"
LOG=$RUN_OUT/runbook.log
phase() { echo "[PHASE $1] $(date -u +%H:%M:%S) -- $2" | tee -a "$LOG"; }
echo "=== rerun start $(date -Iseconds) ===" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | tee -a "$LOG"

# ---- stage SemSup data (trial_3 -> trail_3 typo per inventory) -------
if [ ! -d "$REPO/data/Super/trail_3/rgb" ]; then
  mkdir -p "$REPO/data/Super"
  echo "staging SemSup $SEMSUP_DS -> data/Super/trail_3" | tee -a "$LOG"
  cp -r "$SEMSUP_DS" "$REPO/data/Super/trail_3"
fi

# =====================================================================
# PHASE 1 — MoGe-2 depth generation  (SYSTEM python, torch>=2 + moge)
# =====================================================================
# Do NOT activate dds_env here.  Use the system interpreter that has torch>=2.
PYSYS=$(command -v python3)
echo "PHASE 1 interpreter: $PYSYS" | tee -a "$LOG"
"$PYSYS" -c "import torch,sys;assert torch.__version__>='2',torch.__version__;print('torch',torch.__version__)" \
  2>&1 | tee -a "$LOG" || { echo "FATAL: system python lacks torch>=2 for MoGe"; exit 1; }
# Ensure moge is importable (best-effort install; flag if it fails)
"$PYSYS" -c "import moge" 2>/dev/null || {
  echo "installing moge into system python..." | tee -a "$LOG"
  "$PYSYS" -m pip install -q "git+https://github.com/microsoft/MoGe.git" 2>&1 | tail -3 | tee -a "$LOG"
}

# --- 1a. SemSup MoGe-2 (ref-anchored to depth/ref so scale matches SemSup) ---
if [ -z "$(ls "$SEMSUP_MOGE_DIR"/*left_depth.npy 2>/dev/null | head -1)" ]; then
  phase 1a "SemSup MoGe-2 depth (ref-anchored, temporal_window=5, depth_scale=8)"
  mkdir -p "$SEMSUP_MOGE_DIR"
  "$PYSYS" Addons/depth/generate_depth_moge.py \
    --rgb "$REPO/data/Super/trail_3/rgb" \
    --ref "$REPO/data/Super/trail_3/depth/ref" \
    --out "$SEMSUP_MOGE_DIR" \
    --temporal_window 5 \
    --depth_scale 8 \
    2>&1 | tee -a "$LOG" || echo "  WARN: SemSup MoGe gen failed" | tee -a "$LOG"
  # also stage locally for training
  cp -rn "$SEMSUP_MOGE_DIR" "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}" 2>/dev/null || true
else
  echo "  SemSup MoGe depth already present on Drive -> skip gen" | tee -a "$LOG"
  cp -rn "$SEMSUP_MOGE_DIR" "$REPO/data/Super/trail_3/depth/MoGe2_trail3_${DATE}" 2>/dev/null || true
fi

# --- 1b. StereoMIS MoGe-2 (last-4000; metric-direct; npy->uint16 png) ---
if [ -z "$(ls "$STEREO_MOGE_DIR"/*.png 2>/dev/null | head -1)" ]; then
  phase 1b "StereoMIS MoGe-2 depth (last-4000)"
  # Stage P2_1 left frames locally (tarball-first)
  if [ ! -d /content/p2_1_local/video_frames ]; then
    mkdir -p /content/p2_1_local
    STAR=$STEREO_DS/P2_1_staging.tar
    [ -f "$STAR" ] && tar xf "$STAR" -C /content/p2_1_local || \
      (cd "$STEREO_DS" && tar cf - video_frames groundtruth.txt StereoCalibration.ini masks | tar xf - -C /content/p2_1_local)
  fi
  # Symlink last-4000 left frames AS *-left.png so generate_depth_moge globs them.
  STAGE=/content/p2_1_moge_stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
  ( cd /content/p2_1_local/video_frames && ls *l.png 2>/dev/null | tail -4000 | while read f; do
      stem="${f%l.png}"; ln -sf "/content/p2_1_local/video_frames/$f" "$STAGE/${stem}-left.png"; done )
  echo "  staged $(ls "$STAGE"/*-left.png 2>/dev/null | wc -l) left frames" | tee -a "$LOG"
  NPY=/content/p2_1_moge_npy; mkdir -p "$NPY"
  # --depth_scale 1 => npy holds metres (avoids the overnight double-scale bug)
  "$PYSYS" Addons/depth/generate_depth_moge.py \
    --rgb "$STAGE" --out "$NPY" --temporal_window 5 --depth_scale 1 \
    2>&1 | tee -a "$LOG" || echo "  WARN: StereoMIS MoGe gen failed" | tee -a "$LOG"
  # npy(metres) -> uint16 png at png_depth_scale=10000, named <stem>.png
  mkdir -p "$STEREO_MOGE_DIR"
  "$PYSYS" - "$NPY" "$STEREO_MOGE_DIR" <<'PYEOF'
import sys, glob, os, numpy as np, cv2
npy_dir, png_dir = sys.argv[1], sys.argv[2]
n=0
for f in sorted(glob.glob(os.path.join(npy_dir,'*-left_depth.npy'))):
    d_m = np.load(f).astype(np.float32)               # metres
    u16 = np.clip(d_m*10000, 0, 65535).astype(np.uint16)
    stem = os.path.basename(f).replace('-left_depth.npy','')
    cv2.imwrite(os.path.join(png_dir, stem+'.png'), u16); n+=1
print(f'converted {n} npy(metres) -> uint16 png @scale10000')
PYEOF
else
  echo "  StereoMIS MoGe depth already present on Drive -> skip gen" | tee -a "$LOG"
fi

# =====================================================================
# PHASE 2 — DDS-SLAM training + eval  (dds_env, torch 1.10)
# =====================================================================
source /tmp/dds_env/bin/activate
echo "PHASE 2 env: $(python -c 'import torch;print("torch",torch.__version__)')" | tee -a "$LOG"

# ---- SemSup: 5 configs (4 stereo + moge2 with regenerated depth) ----
declare -A SEMSUP
SEMSUP[paperfaith]=configs/Super/trail3_paper_faithful.yaml
SEMSUP[paperfaith-v2]=configs/Super/trail3_paper_faithful_v2.yaml
SEMSUP[variantA-stereo]=configs/Super/trail3_variant_a_stereo.yaml
SEMSUP[variantC-stereo]=configs/Super/trail3_variant_c_stereo.yaml
SEMSUP[moge2]=__moge2_override__   # built below with regenerated depth_subdir

# moge2 override config: canonical moge2 + regenerated MoGe depth dir
MOGE2_CFG=configs/Super/_rerun_trail3_moge2_${DATE}.yaml
cat > "$MOGE2_CFG" <<YAMLEOF
inherit_from: configs/Super/trail3_moge2.yaml
data:
  depth_subdir: depth/MoGe2_trail3_${DATE}
YAMLEOF
SEMSUP[moge2]=$MOGE2_CFG

for KEY in paperfaith paperfaith-v2 variantA-stereo variantC-stereo moge2; do
  CFG=${SEMSUP[$KEY]}
  NAME=DDS-SLAM_semsup-${KEY}_${DATE}
  KDIR=$RUN_OUT/$NAME
  mkdir -p "$KDIR"
  [ -f "$KDIR/.DONE" ] && { echo "$NAME done -> skip" | tee -a "$LOG"; continue; }
  phase "2.$KEY" "TRAIN $NAME  ($CFG)"
  OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
  python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || { echo "  WARN: train failed $KEY" | tee -a "$LOG"; continue; }
  CKPT=$(ls -t "$REPO/$OUT/demo"/checkpoint*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "  FATAL: no ckpt for $KEY" | tee -a "$LOG"; continue; fi
  cp "$CKPT" "$SEMSUP_CKPT_DIR/$NAME.pt"
  echo "  saved -> $SEMSUP_CKPT_DIR/$NAME.pt" | tee -a "$LOG"
  # render (paired GT) + eval (GT-alongside mode)
  RDIR=$KDIR/rendered_all
  python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
    --output_dir "$RDIR" --save_depth --save_gt 2>&1 | tee -a "$LOG"
  python Addons/eval/eval_rendering.py --gt_dir "$RDIR" --render_dir "$RDIR" \
    --name "$NAME" --sequence "Lab1 (trail3)" \
    --output_csv "$KDIR/eval_per_frame.csv" \
    --summary_csv "$RUN_OUT/_semsup_summary.csv" 2>&1 | tee "$KDIR/eval_summary.txt"
  touch "$KDIR/.DONE"
done

# ---- StereoMIS MoGe back-4000 (full SLAM; ATE inline) ----
SNAME=DDS-SLAM_stereomis-p2-1-moge-back4000_${DATE}
SDIR=$RUN_OUT/$SNAME
mkdir -p "$SDIR"
if [ ! -f "$SDIR/.DONE" ]; then
  phase "2.stereomis" "StereoMIS MoGe back-4000 SLAM"
  # ensure data staged
  if [ ! -d /content/p2_1_local/video_frames ]; then
    mkdir -p /content/p2_1_local
    STAR=$STEREO_DS/P2_1_staging.tar
    [ -f "$STAR" ] && tar xf "$STAR" -C /content/p2_1_local
  fi
  # StereoMISDataset reads {datadir}/depth/*.png — symlink the MoGe png dir there
  rm -rf /content/p2_1_local/depth
  ln -sf "$STEREO_MOGE_DIR" /content/p2_1_local/depth
  CFG=configs/StereoMIS/p2_1_moge_back4000.yaml   # tracked; output dir = $SNAME
  python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || echo "  WARN: StereoMIS SLAM failed" | tee -a "$LOG"
  OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
  CKPT=$(ls -t "$REPO/$OUT/demo"/checkpoint*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    cp "$CKPT" "$STEREO_CKPT_DIR/$SNAME.pt"
    echo "  saved -> $STEREO_CKPT_DIR/$SNAME.pt" | tee -a "$LOG"
    # collect ATE (ddsslam.py wrote output.txt + pose_*.png inline) + render
    cp "$REPO/$OUT/demo/output.txt" "$SDIR/ate_output.txt" 2>/dev/null || true
    cp "$REPO/$OUT/demo"/pose_*.png "$SDIR/" 2>/dev/null || true
    python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
      --output_dir "$SDIR/rendered_all" --save_depth --save_gt 2>&1 | tee -a "$LOG"
  else
    echo "  FATAL: no StereoMIS ckpt produced" | tee -a "$LOG"
  fi
  touch "$SDIR/.DONE"
fi

echo "=== rerun complete $(date -Iseconds) ===" | tee -a "$LOG"
echo "Checkpoints:" | tee -a "$LOG"
echo "  SemSup    -> $SEMSUP_CKPT_DIR/DDS-SLAM_semsup-*_${DATE}.pt" | tee -a "$LOG"
echo "  StereoMIS -> $STEREO_CKPT_DIR/$SNAME.pt" | tee -a "$LOG"
echo "Depth (new):" | tee -a "$LOG"
echo "  SemSup MoGe    -> $SEMSUP_MOGE_DIR" | tee -a "$LOG"
echo "  StereoMIS MoGe -> $STEREO_MOGE_DIR" | tee -a "$LOG"
echo "Run artefacts (renders/eval/log) -> $RUN_OUT" | tee -a "$LOG"
echo "SemSup render summary -> $RUN_OUT/_semsup_summary.csv" | tee -a "$LOG"
