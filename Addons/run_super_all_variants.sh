#!/bin/bash
# run_super_all_variants.sh — fire DDS-SLAM on SemSup trial_3 across all 9
# depth-source variants, save each run's output to Drive, continue on failure.
#
# Usage (on Colab, after Addons/env/colab_setup.sh and after data is staged at
# data/Super/trail_3/ with all depth/<variant>/ subdirs present):
#
#   bash Addons/run_super_all_variants.sh
#
# Per-variant skip via env (default 0 = run all):
#   SKIP_REF=1 SKIP_VARIANT_A_MONO=1 ... bash Addons/run_super_all_variants.sh
#
# Idempotent — if output for a variant is already on Drive, that run is skipped.

set -u   # catch unbound vars; do NOT use -e (we want to continue on per-run failures)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRIVE_OUTPUT_ROOT="${DRIVE_OUTPUT_ROOT:-/content/drive/MyDrive/Outputs/ddsslam_super_trial3}"
MASTER_LOG="$REPO_ROOT/output/run_super_all_variants.log"

mkdir -p "$DRIVE_OUTPUT_ROOT"
mkdir -p "$(dirname "$MASTER_LOG")"
touch "$MASTER_LOG"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

# Each entry: <skip_var_name>:<config_basename>:<output_subdir>
# config_basename resolves to configs/Super/<basename>.yaml
# output_subdir is what ddsslam writes to under output/ (set in the yaml's data.output)
VARIANTS=(
  "SKIP_REF:trail3_ref:trail3_ref"
  "SKIP_VARIANT_A_MONO:trail3_variant_a_mono:trail3_variant_a_mono"
  "SKIP_VARIANT_A_STEREO:trail3_variant_a_stereo:trail3_variant_a_stereo"
  "SKIP_VARIANT_A_MS:trail3_variant_a_ms:trail3_variant_a_ms"
  "SKIP_VARIANT_B_AFSFM:trail3_variant_b_afsfm:trail3_variant_b_afsfm"
  "SKIP_VARIANT_C_MONO:trail3_variant_c_mono:trail3_variant_c_mono"
  "SKIP_VARIANT_C_STEREO:trail3_variant_c_stereo:trail3_variant_c_stereo"
  "SKIP_VARIANT_C_MS:trail3_variant_c_ms:trail3_variant_c_ms"
  "SKIP_AFSFM:trail3_afsfm:trail3_afsfm"
)

log "=== DDS-SLAM Super trial_3 sweep starting ==="
log "Repo:           $REPO_ROOT"
log "Drive output:   $DRIVE_OUTPUT_ROOT"
log ""

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r skip_var cfg_name out_subdir <<< "$entry"

  # Per-variant env skip
  skip_val="${!skip_var:-0}"
  if [ "$skip_val" = "1" ]; then
    log "SKIP  $cfg_name (env $skip_var=1)"
    continue
  fi

  # Idempotency: skip if final output already on Drive
  if [ -d "$DRIVE_OUTPUT_ROOT/$out_subdir" ]; then
    log "SKIP  $cfg_name (already on Drive)"
    continue
  fi

  cfg_path="configs/Super/${cfg_name}.yaml"
  if [ ! -f "$cfg_path" ]; then
    log "FAIL  $cfg_name (config not found at $cfg_path)"
    continue
  fi

  log "START $cfg_name"
  t0=$(date +%s)

  # Stream training stdout/stderr into the master log
  python ddsslam.py --config "$cfg_path" >> "$MASTER_LOG" 2>&1
  rc=$?

  elapsed=$(( $(date +%s) - t0 ))
  h=$(( elapsed / 3600 ))
  m=$(( (elapsed % 3600) / 60 ))

  if [ $rc -eq 0 ]; then
    log "DONE  $cfg_name (${h}h${m}m) — copying output to Drive..."
    if [ -d "output/$out_subdir" ]; then
      cp -r "output/$out_subdir" "$DRIVE_OUTPUT_ROOT/$out_subdir" 2>>"$MASTER_LOG"
      if [ $? -eq 0 ]; then
        log "SAVED $cfg_name → $DRIVE_OUTPUT_ROOT/$out_subdir"
      else
        log "WARN  $cfg_name training succeeded but Drive copy failed"
      fi
    else
      log "WARN  $cfg_name training succeeded but output/$out_subdir does not exist"
    fi
  else
    log "FAIL  $cfg_name (rc=$rc, ${h}h${m}m) — continuing"
  fi

  # Sync master log to Drive after every run (survives session death)
  cp "$MASTER_LOG" "$DRIVE_OUTPUT_ROOT/run_super_all_variants.log" 2>/dev/null
done

log ""
log "=== sweep complete ==="
log "Drive output tree:"
ls -la "$DRIVE_OUTPUT_ROOT" | tee -a "$MASTER_LOG"
cp "$MASTER_LOG" "$DRIVE_OUTPUT_ROOT/run_super_all_variants.log" 2>/dev/null
