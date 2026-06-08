#!/usr/bin/env bash
# Overnight DDS-SLAM run — 2026-06-05/06
# =====================================================================
# Three workloads in priority order:
#
#   A) DDS-SLAM CRCD diagnostic (Phase 1 Tests 0, 1, 2, 5)
#      - dx_hook dump for Test 1 (smoking gun)
#      - deform-off render for Tests 0, 2 (pair with normal render)
#      - Local Kabsch analysis runs locally tomorrow (not in this script)
#      - Snippets: C_1/001, C_2/001 (per user 2026-06-05 selection)
#
#   B) SemSup re-runs with FIXED output depth (commit 4accb4d, png_depth_scale=10000)
#      - 5 configs: paperfaith, paperfaithv2, variantB_ep9_hash19, variantA_stereo,
#        variantC_stereo, moge2 (per user 2026-06-05 selection)
#      - Uses post-hoc render via Addons/viz/render_all_frames.py
#        (loads ckpt + renders at saved poses; ~5-10 min per config)
#      - NOT a full SLAM retrain — re-renders depth/RGB from existing ckpts
#
#   C) StereoMIS Back-4000 with MoGe-2 depth
#      - Last 4000 frames of P2_1 (canonical paper slice)
#      - Full SLAM run with MoGe depth (vs paper RAFT-Stereo baseline)
#      - Compare ATE to paper 8.3 mm + prior 39-47 mm RAFT result
#
# Sentinel-gated (resumable on session restart).
# Drive output:
#   MyDrive/Outputs/dds_overnight_20260605_06/
#     crcd_diagnostic/{C1_001,C2_001}/
#     semsup_rerender/{paperfaith,variantB,...}/
#     stereomis_moge_back4000/
# =====================================================================

set -e

DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_overnight_20260605_06
mkdir -p "$DRIVE_ROOT"

REPO_ROOT=/content/DDS-SLAM
LOG=$DRIVE_ROOT/runbook.log
phase() { echo "[PHASE $1] $(date -u +%H:%M:%S) -- $2" | tee -a "$LOG"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { touch "$1/.DONE"; }

echo "=== DDS-SLAM overnight start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ===" | tee -a "$LOG"

# =====================================================================
# PHASE 0 -- pre-flight
# =====================================================================
phase 0 "pre-flight"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "FATAL: nvidia-smi missing"; exit 1
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU" | tee -a "$LOG"

cd "$REPO_ROOT" || { echo "FATAL: $REPO_ROOT missing"; exit 1; }
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull" | tee -a "$LOG"
else
  git fetch && git merge --ff-only origin/main || echo "  WARN: non-FF on remote" | tee -a "$LOG"
fi
echo "  HEAD: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# Activate env (assumed already restored by colab_setup.sh)
if [ -f /tmp/dds_env/bin/activate ]; then
  source /tmp/dds_env/bin/activate
  echo "  env: /tmp/dds_env activated" | tee -a "$LOG"
fi

# =====================================================================
# WORKLOAD A -- CRCD diagnostic (highest priority — smoking gun)
# =====================================================================
echo "" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"
echo "## WORKLOAD A: CRCD diagnostic Phase 1" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"

# Per-snippet: ckpt path, config path, GT path (CRCD-Published source)
declare -A SNIPPETS_CKPT
declare -A SNIPPETS_CFG
SNIPPETS_CKPT[C1_001]="output/CRCD/c1_001_paperfaith_lrfix/demo"
SNIPPETS_CFG[C1_001]="configs/CRCD/c1_001_paperfaith_lrfix.yaml"
SNIPPETS_CKPT[C2_001]="output/CRCD/c2_001_paperfaith_lrfix/demo"
SNIPPETS_CFG[C2_001]="configs/CRCD/c2_001_paperfaith_lrfix.yaml"

CRCD_DRIVE_OUT=$DRIVE_ROOT/crcd_diagnostic
mkdir -p "$CRCD_DRIVE_OUT"

for KEY in C1_001 C2_001; do
  echo "" | tee -a "$LOG"
  echo "## CRCD snippet $KEY" | tee -a "$LOG"
  SNIPPET_DRIVE=$CRCD_DRIVE_OUT/$KEY
  mkdir -p "$SNIPPET_DRIVE"

  CKPT_DIR=${SNIPPETS_CKPT[$KEY]}
  CFG=${SNIPPETS_CFG[$KEY]}

  if done_marker "$SNIPPET_DRIVE"; then
    echo "  $KEY already done -- skip" | tee -a "$LOG"; continue
  fi

  # Stage CRCD data: raw tarball -> preprocess -> data/CRCD/<KEY>/.
  KEY_LOWER=$(echo "$KEY" | tr A-Z a-z)
  DATA_LOCAL=$REPO_ROOT/data/CRCD/$KEY
  if [ ! -d "$DATA_LOCAL/video_frames" ] || [ -z "$(ls $DATA_LOCAL/video_frames/*.png 2>/dev/null | head -1)" ]; then
    mkdir -p "$DATA_LOCAL"
    case "$KEY" in
      C1_001) EP=C_1; SID=001 ;;
      C2_001) EP=C_2; SID=001 ;;
      F1_002) EP=F_1; SID=002 ;;
      F3_007) EP=F_3; SID=007 ;;
    esac
    RAW_TAR=/content/drive/MyDrive/Datasets/CRCD-Published/${EP}_snippet_${SID}_staging.tar
    RAW_DIR=/content/crcd_raw_$KEY
    if [ ! -d "$RAW_DIR/$EP" ] && [ -f "$RAW_TAR" ]; then
      mkdir -p "$RAW_DIR"
      echo "  extracting $RAW_TAR -> $RAW_DIR" | tee -a "$LOG"
      tar xf "$RAW_TAR" -C "$RAW_DIR" 2>/dev/null
    fi
    SNIPPET_DIR=$(find "$RAW_DIR" -maxdepth 3 -type d -name "snippet_$SID" 2>/dev/null | head -1)
    CALIB_PKL=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
    if [ -n "$SNIPPET_DIR" ] && [ -f "$CALIB_PKL" ]; then
      echo "  preprocessing CRCD $SNIPPET_DIR -> $DATA_LOCAL" | tee -a "$LOG"
      python Addons/preprocess/preprocess_crcd_published.py \
        --snippet_dir "$SNIPPET_DIR" \
        --calib_pkl   "$CALIB_PKL" \
        --output_dir  "$DATA_LOCAL" 2>&1 | tee -a "$LOG" || \
        echo "  WARN: CRCD preprocess failed for $KEY" | tee -a "$LOG"
    else
      echo "  WARN: SNIPPET_DIR=$SNIPPET_DIR or CALIB_PKL not found" | tee -a "$LOG"
    fi
  fi

  # Pull the latest checkpoint from Drive into the local CKPT_DIR if local missing.
  # The June-4 batch put payloads at MyDrive/Outputs/dds_crcd_4snippets_20260604/<NAME>/payload.tgz
  if ! ls "$CKPT_DIR"/checkpoint*.pt >/dev/null 2>&1; then
    PAYLOAD=/content/drive/MyDrive/Outputs/dds_crcd_4snippets_20260604/$KEY/payload.tgz
    if [ -f "$PAYLOAD" ]; then
      mkdir -p "$CKPT_DIR"
      tar xzf "$PAYLOAD" -C "$REPO_ROOT/output/CRCD/$(echo $KEY | tr A-Z a-z)_paperfaith_lrfix/" 2>/dev/null || \
        tar xzf "$PAYLOAD" -C /tmp/ 2>/dev/null
      # find the latest checkpoint
      CKPT=$(find /tmp $CKPT_DIR -name 'checkpoint*.pt' 2>/dev/null | sort -V | tail -1)
      if [ -n "$CKPT" ]; then
        cp -n "$CKPT" "$CKPT_DIR/" 2>/dev/null
        echo "  staged ckpt from $PAYLOAD" | tee -a "$LOG"
      fi
    else
      echo "  WARN: no $PAYLOAD -- skipping $KEY" | tee -a "$LOG"; continue
    fi
  fi
  CKPT=$(ls -t "$CKPT_DIR"/checkpoint*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "  FATAL: no ckpt for $KEY"; continue
  fi

  # Sanity-check the deformation network BEFORE the full dx_hook dump.
  # If time_net outputs are zero across test inputs, dx_hook will produce
  # zero Δx and Test 1 will be non-diagnostic (we caught this 2026-06-08).
  # (Sanity must run AFTER CKPT resolution — fixed 2026-06-08.)
  SANITY_LOG=$SNIPPET_DRIVE/dx_hook_sanity.log
  if [ ! -s "$SANITY_LOG" ] || grep -q "FileNotFoundError" "$SANITY_LOG"; then
    phase "A.0.$KEY" "dx_hook sanity check (verify time_net non-zero)"
    python diagnosis/infra/dx_hook_sanity.py \
      --config "$CFG" --checkpoint "$CKPT" 2>&1 | tee "$SANITY_LOG"
  fi
  echo "  using ckpt: $CKPT" | tee -a "$LOG"

  # -----------------------------------------------------------------
  # A.1 -- dx_hook dump (Test 1)
  # -----------------------------------------------------------------
  DX_DIR=$SNIPPET_DRIVE/dx_dump
  if [ ! -d "$DX_DIR" ] || [ -z "$(ls $DX_DIR/frame_*.npz 2>/dev/null | head -1)" ]; then
    phase "A.1.$KEY" "dx_hook dump (Test 1)"
    python diagnosis/infra/dx_hook.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --output_dir "$DX_DIR" \
      --rays_per_frame 4096 \
      --samples_per_ray 8 \
      2>&1 | tee -a "$LOG" || echo "  WARN: dx_hook failed for $KEY"
  else
    echo "  dx_hook already done" | tee -a "$LOG"
  fi

  # -----------------------------------------------------------------
  # A.2 -- deformation-off render (Tests 0, 2)
  #        Slow (~9 s/frame on T4, ~50 min for 360 frames, ~1h45m for 730).
  #        Skip with SKIP_DEFORM_RENDERS=1 if Test 1 dx_hook is the priority.
  # -----------------------------------------------------------------
  if [ "${SKIP_DEFORM_RENDERS:-0}" = "1" ]; then
    echo "  [SKIP_DEFORM_RENDERS=1] skipping A.2 and A.3 for $KEY" | tee -a "$LOG"
    DEFOFF_DIR=$SNIPPET_DRIVE/deform_off_render
  else
  DEFOFF_DIR=$SNIPPET_DRIVE/deform_off_render
  if [ ! -d "$DEFOFF_DIR" ] || [ -z "$(ls $DEFOFF_DIR/*.jpg 2>/dev/null | head -1)" ]; then
    phase "A.2.$KEY" "deformation-off render (Tests 0/2)"
    python diagnosis/infra/deform_off_render.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --output_dir "$DEFOFF_DIR" \
      2>&1 | tee -a "$LOG" || echo "  WARN: deform_off_render failed for $KEY"
  else
    echo "  deform_off_render already done" | tee -a "$LOG"
  fi

  # -----------------------------------------------------------------
  # A.3 -- canonical (deformation-on) render — ensure we have it for paired comparison
  # -----------------------------------------------------------------
  DEFON_DIR=$SNIPPET_DRIVE/deform_on_render
  if [ ! -d "$DEFON_DIR" ] || [ -z "$(ls $DEFON_DIR/*.jpg 2>/dev/null | head -1)" ]; then
    phase "A.3.$KEY" "canonical (deformation-on) render"
    python Addons/viz/render_all_frames.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --output_dir "$DEFON_DIR" \
      --save_depth --save_gt \
      2>&1 | tee -a "$LOG" || echo "  WARN: render_all_frames failed for $KEY"
  else
    echo "  deform_on_render already done" | tee -a "$LOG"
  fi
  fi  # end SKIP_DEFORM_RENDERS guard

  # -----------------------------------------------------------------
  # A.4 -- Δx norm heatmap overlay on input RGB (visualization 1)
  # -----------------------------------------------------------------
  HEATMAP_DIR=$SNIPPET_DRIVE/dx_heatmap
  RGB_DIR=$REPO_ROOT/data/CRCD/${KEY}/video_frames
  if [ ! -d "$HEATMAP_DIR" ] || [ -z "$(ls $HEATMAP_DIR/*.png 2>/dev/null | head -1)" ]; then
    phase "A.4.$KEY" "Δx heatmap overlays on input RGB"
    python diagnosis/infra/dx_norm_heatmap.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --rgb_dir "$RGB_DIR" \
      --rgb_pattern '*l.png' \
      --output_dir "$HEATMAP_DIR" \
      --frames auto \
      2>&1 | tee -a "$LOG" || echo "  WARN: dx_norm_heatmap failed for $KEY"
  else
    echo "  dx_heatmap already done" | tee -a "$LOG"
  fi

  # -----------------------------------------------------------------
  # A.5 -- SDF cross-section (diagnoses marching_cubes failure if any)
  # -----------------------------------------------------------------
  SDF_DIR=$SNIPPET_DRIVE/sdf_cross_section
  if [ ! -f "$SDF_DIR/sdf_cross_section.png" ]; then
    phase "A.5.$KEY" "SDF cross-section visualization"
    python diagnosis/infra/sdf_cross_section.py \
      --config "$CFG" \
      --checkpoint "$CKPT" \
      --output_dir "$SDF_DIR" \
      --axes xy,xz,yz \
      2>&1 | tee -a "$LOG" || echo "  WARN: sdf_cross_section failed for $KEY"
  else
    echo "  sdf_cross_section already done" | tee -a "$LOG"
  fi

  # -----------------------------------------------------------------
  # A.6 -- Tool-mask Δx line plot (visualization 3 = Test 2 headline)
  # -----------------------------------------------------------------
  TEST2_FIG=$SNIPPET_DRIVE/test2_tool_mask_dx.png
  # CRCD-Published source for semantic_instance masks
  case "$KEY" in
    C1_001) SEM_DIR=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/semantic_instance ;;
    C2_001) SEM_DIR=/content/drive/MyDrive/Datasets/CRCD-Published/C_2/snippet_001/semantic_instance ;;
    *)      SEM_DIR="" ;;
  esac
  if [ -n "$SEM_DIR" ] && [ ! -f "$TEST2_FIG" ]; then
    phase "A.6.$KEY" "Tool-mask Δx line plot (Test 2 visualization)"
    python diagnosis/phase1/test2_tool_mask_dx.py \
      --dx_dir "$DX_DIR" \
      --config "$CFG" \
      --semantic_dir "$SEM_DIR" \
      --out_csv "$SNIPPET_DRIVE/test2_tool_mask_dx.csv" \
      --out_fig "$TEST2_FIG" \
      --name "$KEY" \
      2>&1 | tee -a "$LOG" || echo "  WARN: test2 failed for $KEY"
  else
    echo "  test2 viz already done (or no semantic dir)" | tee -a "$LOG"
  fi

  mark_done "$SNIPPET_DRIVE"
  echo "## $KEY CRCD diagnostic complete" | tee -a "$LOG"
done

# =====================================================================
# WORKLOAD B -- SemSup re-renders (5 configs, post-hoc render with FIXED depth scale)
# =====================================================================
echo "" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"
echo "## WORKLOAD B: SemSup re-renders (fixed output depth)" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"

# Config -> human-readable name (per user request 2026-06-05)
declare -A SEMSUP_CFG
declare -A SEMSUP_HUMAN
SEMSUP_CFG[paperfaith]=configs/Super/trail3_paper_faithful.yaml
SEMSUP_HUMAN[paperfaith]="Paper-faithful (hash 19, n_range_d 16, depth=variant_a_stereo)"
SEMSUP_CFG[paperfaith_v2]=configs/Super/trail3_paper_faithful_v2.yaml
SEMSUP_HUMAN[paperfaith_v2]="Paper-faithful v2 (also fixes deformation freq L_x=10, L_t=4)"
SEMSUP_CFG[variantB_ep9_hash19]=configs/Super/trail3_variant_b_ep9_hash19.yaml
SEMSUP_HUMAN[variantB_ep9_hash19]="Variant B + hash 19 (prior best, 288x384 aspect-correct)"
SEMSUP_CFG[variantA_stereo]=configs/Super/trail3_variant_a_stereo.yaml
SEMSUP_HUMAN[variantA_stereo]="Variant A (192x256) with stereo depth"
SEMSUP_CFG[variantC_stereo]=configs/Super/trail3_variant_c_stereo.yaml
SEMSUP_HUMAN[variantC_stereo]="Variant C (240x320) with stereo depth"
SEMSUP_CFG[moge2]=configs/Super/trail3_moge2.yaml
SEMSUP_HUMAN[moge2]="MoGe-2 monocular depth (pre-fix had bad output depth)"

SEMSUP_DRIVE=$DRIVE_ROOT/semsup_rerender
mkdir -p "$SEMSUP_DRIVE"

# README mapping cryptic -> human name for the supervisor's read
{
  echo "SemSup Re-render Naming Key (overnight 2026-06-05)"
  echo "=============================================================="
  echo "Each subdir under semsup_rerender/<KEY>/ contains:"
  echo "  - rendered_all/{NNNN.jpg,NNNN_gt.png,depth/NNNN.png}"
  echo "  - eval_summary.txt   (PSNR/SSIM/LPIPS per the fixed depth)"
  echo "  - 6panel.mp4 if generated"
  echo ""
  for KEY in paperfaith paperfaith_v2 variantB_ep9_hash19 variantA_stereo variantC_stereo moge2; do
    printf "  %-22s = %s\n" "$KEY" "${SEMSUP_HUMAN[$KEY]}"
  done
} > "$SEMSUP_DRIVE/NAMING_KEY.txt"
cat "$SEMSUP_DRIVE/NAMING_KEY.txt" | tee -a "$LOG"

for KEY in paperfaith paperfaith_v2 variantB_ep9_hash19 variantA_stereo variantC_stereo moge2; do
  echo "" | tee -a "$LOG"
  echo "## SemSup re-render: $KEY  (${SEMSUP_HUMAN[$KEY]})" | tee -a "$LOG"
  KEY_DRIVE=$SEMSUP_DRIVE/$KEY
  mkdir -p "$KEY_DRIVE"
  if done_marker "$KEY_DRIVE"; then
    echo "  already done -- skip" | tee -a "$LOG"; continue
  fi

  CFG=${SEMSUP_CFG[$KEY]}
  # Output dir from config (trail3 family)
  OUT_DIR=$(python -c "
from config import load_config
c = load_config('$CFG')
print(c['data']['output'])
" 2>/dev/null)
  # Map runbook KEY -> actual trail3 dir name on Drive
  case "$KEY" in
    paperfaith)         DRIVE_KEY=trail3_paper_faithful ;;
    paperfaith_v2)      DRIVE_KEY=trail3_paper_faithful_v2 ;;
    variantB_ep9_hash19) DRIVE_KEY=trail3_variant_b_ep9_hash19 ;;
    variantA_stereo)    DRIVE_KEY=trail3_variant_a_stereo ;;
    variantC_stereo)    DRIVE_KEY=trail3_variant_c_stereo ;;
    moge2)              DRIVE_KEY=trail3_moge2 ;;
    *)                  DRIVE_KEY=$KEY ;;
  esac
  CKPT=$(ls -t $REPO_ROOT/$OUT_DIR/demo/checkpoint*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    # Drive search — multiple paths (depthsweep, plain outputs, etc)
    CKPT=$(find \
      /content/drive/MyDrive/Outputs/ddsslam_super_depthsweep_*/$DRIVE_KEY \
      /content/drive/MyDrive/Outputs/$DRIVE_KEY \
      /content/drive/MyDrive/Outputs/$KEY \
      -name "checkpoint*.pt" 2>/dev/null | sort -V | tail -1)
  fi
  if [ -z "$CKPT" ]; then
    # Final fallback — broad find
    CKPT=$(find /content/drive/MyDrive/Outputs -name "checkpoint*.pt" \
      -path "*$DRIVE_KEY*" 2>/dev/null | sort -V | tail -1)
  fi
  if [ -z "$CKPT" ]; then
    echo "  FATAL: no ckpt for $KEY (DRIVE_KEY=$DRIVE_KEY)" | tee -a "$LOG"; continue
  fi
  echo "  using ckpt: $CKPT" | tee -a "$LOG"

  # Stage SemSup data if not present locally.  Canonical Drive source per
  # user 2026-06-08: MyDrive/Datasets/SemSup/v2_data/trial_3/  (with
  # trial_3 vs trail_3 typo — config wants data/Super/trail_3 with 'a').
  if [ ! -d "$REPO_ROOT/data/Super/trail_3" ]; then
    mkdir -p "$REPO_ROOT/data/Super"
    SUPER_DRIVE=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
    if [ -d "$SUPER_DRIVE" ]; then
      echo "  staging Super trail_3 from $SUPER_DRIVE" | tee -a "$LOG"
      cp -r "$SUPER_DRIVE" "$REPO_ROOT/data/Super/trail_3"
    else
      echo "  WARN: Super trail_3 data not at $SUPER_DRIVE — skipping $KEY" | tee -a "$LOG"
      continue
    fi
  fi

  # Substitute depth_subdir for configs whose original depth is missing on
  # Drive (per user 2026-06-08 image of trial_3/depth/).  For post-hoc
  # rendering, the loaded depth is NOT used for the rendered output (the
  # model produces both RGB and depth from its weights).  We only need
  # the dataset to load successfully (i.e. depth_subdir to exist).
  # Missing on Drive: variant_b_ep9 (needed by variantB_ep9_hash19), moge2.
  # Available substitutes: variant_b_afsfm (closest to variantB), variant_a_stereo.
  CFG_OVERRIDE_DIR=$REPO_ROOT/configs/Super/_overnight
  mkdir -p "$CFG_OVERRIDE_DIR"
  case "$KEY" in
    variantB_ep9_hash19)
      SUBSTITUTE=variant_b_afsfm
      CFG_USE=$CFG_OVERRIDE_DIR/$KEY.yaml
      cat > "$CFG_USE" <<YAMLEOF
inherit_from: $CFG
data:
  depth_subdir: depth/$SUBSTITUTE
YAMLEOF
      CFG=$CFG_USE
      echo "  SUBSTITUTED depth_subdir -> $SUBSTITUTE (original depth/variant_b_ep9 missing on Drive)" | tee -a "$LOG"
      ;;
    moge2)
      SUBSTITUTE=variant_a_stereo
      CFG_USE=$CFG_OVERRIDE_DIR/$KEY.yaml
      cat > "$CFG_USE" <<YAMLEOF
inherit_from: $CFG
data:
  depth_subdir: depth/$SUBSTITUTE
YAMLEOF
      CFG=$CFG_USE
      echo "  SUBSTITUTED depth_subdir -> $SUBSTITUTE (original depth/moge2 missing on Drive)" | tee -a "$LOG"
      ;;
  esac

  phase "B.$KEY" "post-hoc render with FIXED depth scale (png_depth_scale=10000)"
  python Addons/viz/render_all_frames.py \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output_dir "$KEY_DRIVE/rendered_all" \
    --save_depth --save_gt \
    2>&1 | tee -a "$LOG" || echo "  WARN: render_all_frames failed for $KEY"

  # Eval PSNR/SSIM/LPIPS — use the staged data's RGB as GT (render_all_frames.py
  # saves NNNN_gt.png alongside NNNN.png IF --save_gt was passed; if missing,
  # we fall back to comparing against data/Super/trail_3/rgb/<NNNN>.png).
  GT_DIR=$KEY_DRIVE/rendered_all
  if ! ls "$GT_DIR"/*_gt.png >/dev/null 2>&1; then
    # No _gt.png saved — point eval at the dataset's RGB dir instead
    GT_DIR=$REPO_ROOT/data/Super/trail_3/rgb
    echo "  no _gt.png in renders; using GT from $GT_DIR" | tee -a "$LOG"
  fi
  python Addons/eval/eval_rendering.py \
    --gt_dir "$GT_DIR" \
    --render_dir "$KEY_DRIVE/rendered_all" \
    --name "$KEY" \
    --output_csv "$KEY_DRIVE/eval_per_frame.csv" \
    --summary_csv "$SEMSUP_DRIVE/_summary.csv" \
    --sequence "Lab1 (trail3)" 2>&1 | tee "$KEY_DRIVE/eval_summary.txt" || \
    echo "  WARN: eval_rendering failed for $KEY"

  # Generate 6-panel video on the spot (per user 2026-06-08: locally before
  # shipping to Drive).  SemSup data has no semantic masks, so 4-panel layout.
  VIDEO=$KEY_DRIVE/${KEY}_6panel.mp4
  if [ ! -f "$VIDEO" ]; then
    phase "B.$KEY.vid" "generate video"
    python Addons/viz/generate_video.py \
      --rgb_input_dir "$REPO_ROOT/data/Super/trail_3/rgb" \
      --rgb_input_pattern '*.png' \
      --rgb_output_dir "$KEY_DRIVE/rendered_all" \
      --rgb_output_pattern '[0-9]*.png' \
      --depth_input_dir "$REPO_ROOT/data/Super/trail_3/depth/variant_a_stereo" \
      --depth_output_dir "$KEY_DRIVE/rendered_all/depth" \
      --trajectory_est "$KEY_DRIVE/rendered_all/est_c2w_data.txt" \
      --trajectory_gt  "$REPO_ROOT/data/Super/trail_3/groundtruth.txt" \
      --output "$VIDEO" \
      --fps 15 \
      --png_depth_scale 10000 \
      2>&1 | tee -a "$LOG" || echo "  WARN: video gen failed for $KEY"
  fi

  mark_done "$KEY_DRIVE"
done

# =====================================================================
# WORKLOAD C -- StereoMIS Back-4000 with MoGe-2 depth
# =====================================================================
echo "" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"
echo "## WORKLOAD C: StereoMIS Back-4000 MoGe-2 (full SLAM)" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"

STEREO_DRIVE=$DRIVE_ROOT/stereomis_moge_back4000
mkdir -p "$STEREO_DRIVE"

if done_marker "$STEREO_DRIVE"; then
  echo "## StereoMIS already done -- skip" | tee -a "$LOG"
else
  # Verify MoGe depth availability
  P2_1_DRIVE=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1
  MOGE_DRIVE_OPTIONS=(
    "/content/drive/MyDrive/Datasets/StereoMisPP/P2_1/depth_moge"
    "/content/drive/MyDrive/StereoMIS_depth_maps_RAFT"
    "/content/drive/MyDrive/StereoMIS_depth_MoGe"
  )
  MOGE_DIR=""
  for d in "${MOGE_DRIVE_OPTIONS[@]}"; do
    if [ -d "$d" ] && [ "$(ls $d/*.png 2>/dev/null | wc -l)" -ge 4000 ]; then
      MOGE_DIR="$d"
      echo "  MoGe depth found: $MOGE_DIR ($(ls $d/*.png | wc -l) files)" | tee -a "$LOG"
      break
    fi
  done

  if [ -z "$MOGE_DIR" ]; then
    echo "  WARN: no MoGe depth dir found on Drive with >=4000 PNGs" | tee -a "$LOG"
    echo "  Checked:" | tee -a "$LOG"
    for d in "${MOGE_DRIVE_OPTIONS[@]}"; do echo "    $d ($(ls $d/*.png 2>/dev/null | wc -l))" | tee -a "$LOG"; done
    echo "  Generating fresh MoGe-2 depth for last 4000 frames..." | tee -a "$LOG"
    # Stage data locally first
    mkdir -p /content/p2_1_local
    if [ ! -d /content/p2_1_local/video_frames ]; then
      cd "$P2_1_DRIVE" && tar cf - video_frames | tar xf - -C /content/p2_1_local
    fi
    MOGE_DIR=/content/p2_1_local/depth_moge_back4000
    mkdir -p "$MOGE_DIR"
    cd "$REPO_ROOT"
    # Generate MoGe for last 4000 frames only
    python Addons/depth/generate_depth_stereomis.py \
      --rgb_dir /content/p2_1_local/video_frames \
      --out_dir "$MOGE_DIR" \
      --slice_back 4000 \
      --model MoGe-2 \
      2>&1 | tee -a "$LOG" || echo "  WARN: MoGe gen failed"
  fi

  phase "C" "StereoMIS Back-4000 SLAM with MoGe-2 depth"
  cd "$REPO_ROOT"
  # Stage data locally — use the pre-built tarball on Drive (per user 2026-06-05)
  if [ ! -d /content/p2_1_local/video_frames ]; then
    mkdir -p /content/p2_1_local
    STEREO_TAR=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1_staging.tar
    if [ -f "$STEREO_TAR" ]; then
      echo "  extracting from $STEREO_TAR ($(du -h "$STEREO_TAR" | cut -f1))" | tee -a "$LOG"
      T0=$(date +%s)
      tar xf "$STEREO_TAR" -C /content/p2_1_local
      echo "  extracted in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$LOG"
    else
      # Fallback: per-file tar from Drive (slow)
      echo "  WARN: $STEREO_TAR missing, using per-file copy (slow)" | tee -a "$LOG"
      cd "$P2_1_DRIVE" && tar cf - video_frames groundtruth.txt StereoCalibration.ini masks | tar xf - -C /content/p2_1_local
      cd "$REPO_ROOT"
    fi
  fi

  # Symlink the MoGe dir into the staged data location expected by the config
  ln -sf "$MOGE_DIR" /content/p2_1_local/depth || true

  # Use a MoGe-specific config (we'll write one inline if missing)
  CFG_MOGE=configs/StereoMIS/p2_1_moge_back4000.yaml
  if [ ! -f "$CFG_MOGE" ]; then
    cat > "$CFG_MOGE" <<'YAMLEOF'
inherit_from: configs/StereoMIS/p2_1.yaml
data:
  datadir: /content/p2_1_local
  output: output/StereoMIS/P2_1_moge_back4000
  exp_name: demo
  depth_subdir: depth
cam:
  png_depth_scale: 10000   # MoGe convention
# Back-4000 slice is hard-coded in datasets/dataset.py:120 (StereoMISDataset.__init__)
# We rely on the existing slicing logic — assumes input dir has 8000+ frames.
YAMLEOF
    echo "  wrote $CFG_MOGE" | tee -a "$LOG"
  fi

  python ddsslam.py --config "$CFG_MOGE" 2>&1 | tee -a "$LOG" || \
    echo "  WARN: StereoMIS SLAM crashed" | tee -a "$LOG"

  # Eval ATE
  python Addons/eval/eval_ate.py "$CFG_MOGE" \
    --csv "$STEREO_DRIVE/ate_summary.csv" \
    --name p2_1_moge_back4000 2>&1 | tee "$STEREO_DRIVE/ate_summary.txt" || \
    echo "  WARN: ATE eval failed" | tee -a "$LOG"

  # Render + eval
  CKPT=$(ls -t $REPO_ROOT/output/StereoMIS/P2_1_moge_back4000/demo/checkpoint*.pt 2>/dev/null | head -1)
  if [ -n "$CKPT" ]; then
    python Addons/viz/render_all_frames.py \
      --config "$CFG_MOGE" \
      --checkpoint "$CKPT" \
      --output_dir "$STEREO_DRIVE/rendered_all" \
      --save_depth --save_gt 2>&1 | tee -a "$LOG"
  fi

  # Ship payload
  if [ -d "$REPO_ROOT/output/StereoMIS/P2_1_moge_back4000" ]; then
    tar czf "$STEREO_DRIVE/payload.tgz" -C "$REPO_ROOT/output/StereoMIS/P2_1_moge_back4000" . 2>/dev/null
  fi
  mark_done "$STEREO_DRIVE"
fi

# =====================================================================
# Final summary
# =====================================================================
echo "" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"
echo "## OVERNIGHT COMPLETE $(date -Iseconds)" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"
echo "Outputs:" | tee -a "$LOG"
echo "  CRCD diagnostic   : $DRIVE_ROOT/crcd_diagnostic/{C1_001,C2_001}/" | tee -a "$LOG"
echo "  SemSup re-renders : $DRIVE_ROOT/semsup_rerender/{6 configs}/" | tee -a "$LOG"
echo "  StereoMIS MoGe    : $DRIVE_ROOT/stereomis_moge_back4000/" | tee -a "$LOG"
echo "  Log               : $LOG" | tee -a "$LOG"
echo "  Naming key        : $DRIVE_ROOT/semsup_rerender/NAMING_KEY.txt" | tee -a "$LOG"
