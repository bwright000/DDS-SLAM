#!/bin/bash
# ============================================================
# run_depth_source_sweep.sh -- Fire DDS-SLAM Super runs sequentially for
# multiple depth sources, saving each to Drive on completion.
#
# Every run uses the patched ddsslam.py (commit 0493cbb) which natively
# saves both RGB renders (.jpg) AND rendered depth output (depth/NNNN.png).
# That's everything the 6-panel video needs from the SLAM side.
#
# Idempotent: if Drive already has a complete output for a config (final
# checkpoint150.pt > 1 MB), the run is skipped. Safe to re-launch after
# a session death.
#
# Per-scene runtime: ~40 min on Colab T4. Five scenes -> ~3.5 hr total.
#
# Usage (after staging data + activating dds_env -- see project_colab_env_gotchas):
#   cd /content/DDS-SLAM
#   source /tmp/dds_env/bin/activate
#   bash scripts/run_depth_source_sweep.sh             # default 5 runs
#   CONFIGS="trail3_variant_b_ep9 trail3_moge2" bash scripts/run_depth_source_sweep.sh   # subset
# ============================================================
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Default sweep -- six runs, in order of (a) validate canonical first, (b) the
# four depth-source comparisons, (c) the targeted hash_size experiment on
# variant_b_ep9, (d) the paper-faithful run that corrects two Super.yaml
# discrepancies vs the paper (Implementation Details Sec IV).
CONFIGS_DEFAULT="trail3_variant_b_ep9 trail3_variant_a_stereo trail3_variant_c_stereo trail3_moge2 trail3_variant_b_ep9_hash19 trail3_paper_faithful"
CONFIGS=${CONFIGS:-$CONFIGS_DEFAULT}

DRIVE_OUT=/content/drive/MyDrive/Outputs/ddsslam_super_depthsweep_$(date +%Y%m%d)
MASTER_LOG=$REPO_ROOT/output/run_depth_source_sweep.log

mkdir -p "$DRIVE_OUT"
mkdir -p "$(dirname "$MASTER_LOG")"
touch "$MASTER_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

# A DDS-SLAM Super run is "complete" only when checkpoint150.pt exists and is
# non-trivial (>1 MB rules out empty/aborted placeholder files). This is the
# single most reliable signal -- the file is only written at the end.
run_complete() {
  local d="$1"
  local f="$d/checkpoint150.pt"
  [ -f "$f" ] && [ "$(stat -c%s "$f" 2>/dev/null || echo 0)" -gt 1000000 ]
}

# Per-config preflight: check that the depth dir referenced exists and has 151 NPYs
check_depth_subdir() {
  local cfg_name="$1"
  local cfg_path="configs/Super/${cfg_name}.yaml"
  if [ ! -f "$cfg_path" ]; then
    log "  ERROR: config not found: $cfg_path"
    return 1
  fi
  local depth_subdir
  depth_subdir=$(grep -E "^\s*depth_subdir:" "$cfg_path" | awk '{print $2}')
  if [ -z "$depth_subdir" ]; then
    # Inherits default (rgb/) -- count the depth NPYs there
    depth_subdir="rgb"
  fi
  local datadir
  datadir=$(grep -E "^\s*datadir:" "$cfg_path" | awk '{print $2}')
  local full="$datadir/$depth_subdir"
  local cnt
  cnt=$(ls "$full"/*-left_depth.npy 2>/dev/null | wc -l)
  if [ "$cnt" -ne 151 ]; then
    log "  WARN: $full has $cnt NPYs (expected 151) -- run may fail"
    return 1
  fi
  log "  OK: $full has 151 NPYs"
  return 0
}

# ---------------------------------------------------------------------
# Preflight: env, repo, monodepth2 (for any in-loop regen)
# ---------------------------------------------------------------------
log "=== Depth-source sweep starting ==="
log "Configs requested:  $CONFIGS"
log "Drive output root:  $DRIVE_OUT"
log ""

if [ ! -f /tmp/dds_env/bin/python ]; then
  log "FATAL: /tmp/dds_env/bin/python missing -- activate dds_env first."
  exit 1
fi
if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
  log "FATAL: core imports failed in current python. Activate dds_env: source /tmp/dds_env/bin/activate"
  exit 1
fi
log "Env OK ($(python -c 'import torch; print(torch.__version__)'))"
log ""

# ---------------------------------------------------------------------
# Main SLAM loop
# ---------------------------------------------------------------------
for cfg_name in $CONFIGS; do
  log "----------------------------------------"
  log "Run: $cfg_name"

  cfg_path="configs/Super/${cfg_name}.yaml"
  if [ ! -f "$cfg_path" ]; then
    log "  SKIP: config not found: $cfg_path"
    continue
  fi

  # Parse output dir from config -- structure is `data.output` + `data.exp_name`
  output_root=$(grep -E "^\s*output:" "$cfg_path" | awk '{print $2}')
  exp_name=$(grep -E "^\s*exp_name:" "$cfg_path" | awk '{print $2}')
  local_out="$output_root/$exp_name"
  drive_scene_out="$DRIVE_OUT/$cfg_name"

  log "  local output:  $local_out"
  log "  drive output:  $drive_scene_out"

  # Skip if Drive already has a complete run for this config
  if run_complete "$drive_scene_out"; then
    log "  SKIP: complete checkpoint150.pt already on Drive ($drive_scene_out)"
    # Make sure local output exists for downstream eval
    if [ ! -d "$local_out" ]; then
      mkdir -p "$local_out"
      cp -r "$drive_scene_out/"* "$local_out/" 2>/dev/null || true
    fi
    continue
  fi
  if [ -d "$drive_scene_out" ] && [ -n "$(ls -A "$drive_scene_out" 2>/dev/null)" ]; then
    log "  REDO: $cfg_name (Drive output exists but checkpoint150.pt missing or partial)"
  fi

  # Verify depth NPYs are staged
  if ! check_depth_subdir "$cfg_name"; then
    log "  SKIP: depth dir not properly staged for $cfg_name"
    continue
  fi

  log "  START $cfg_name (~40 min expected on T4)"
  t0=$(date +%s)
  python -W ignore ddsslam.py --config "$cfg_path" >> "$MASTER_LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - t0 ))
  h=$(( elapsed / 3600 )); m=$(( (elapsed % 3600) / 60 ))

  if run_complete "$local_out"; then
    log "  DONE  $cfg_name (rc=$rc, ${h}h${m}m) -- shipping to Drive..."
    mkdir -p "$drive_scene_out"
    cp -r "$local_out/"* "$drive_scene_out/" 2>>"$MASTER_LOG"
    log "  SAVED -> $drive_scene_out"
  else
    log "  FAIL  $cfg_name (rc=$rc, ${h}h${m}m) -- no complete checkpoint150.pt"
    log "        Local partial output left at: $local_out"
    log "        Not shipping partial to Drive. Sweep continues with next config."
  fi
done

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
log ""
log "=== sweep complete ==="
log "Per-config Drive outputs:"
ls -d "$DRIVE_OUT"/*/ 2>/dev/null | tee -a "$MASTER_LOG"
log ""
log "Per-config checkpoint sizes:"
find "$DRIVE_OUT" -name "checkpoint150.pt" -exec ls -la {} \; 2>/dev/null | tee -a "$MASTER_LOG"
log ""
log "Per-config depth output frame counts:"
for d in "$DRIVE_OUT"/*/; do
  cnt=$(ls "$d/depth"/*.png 2>/dev/null | wc -l)
  echo "  $(basename "$d"): $cnt depth output PNGs" | tee -a "$MASTER_LOG"
done
cp "$MASTER_LOG" "$DRIVE_OUT/run_depth_source_sweep.log" 2>/dev/null
log "Master log -> $DRIVE_OUT/run_depth_source_sweep.log"
