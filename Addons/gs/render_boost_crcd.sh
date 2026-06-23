#!/usr/bin/env bash
# 1-DAY RENDER-BOOST sweep on the nofwd_lr operating config. Target: beat DDS-SLAM render PSNR 26.5 on
# CRCD c1_001 (full-frame, apples-to-apples with eval_rendering.py). Levers: full SH (view-dependent color
# for specular tissue), densify, mapping budget. Config-only. nofwd_lr = FWD_PROP=0 + both cam LR x0.2.
# Requires the env-aware crcd_base.py (with SIMPLIFY/KEYFRAME_EVERY/gaussian_simplification knobs).
set -uo pipefail
ENDO=${ENDO:-/content/EndoGSLAM}; CFG=configs/crcd/crcd_base.py
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/GS_render}
LOG="$ENDO/_render_logs"; mkdir -p "$LOG"; cd "$ENDO"
grep -q 'SIMPLIFY' "$ENDO/$CFG" || { echo "FATAL: $CFG lacks the SIMPLIFY knob -> re-paste the full env-aware crcd_base.py first"; exit 1; }

BASE="FWD_PROP=0 LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2"   # nofwd_lr operating config for ALL arms

run_arm(){  # tag  "EXTRA ENV"  seed
  local tag="$1" extra="$2" seed="${3:-0}" rn out
  rn="C1_001_${tag}_s${seed}"; out="experiments/CRCD_base/${rn}"; mkdir -p "$out"
  [ -f "$out/.DONE" ] && { echo "[$rn] .DONE skip"; return 0; }
  echo "=== $(date +%H:%M) ARM $rn  [$BASE $extra] ==="
  if ! env RUN_TAG="$tag" SEED="$seed" $BASE $extra python scripts/main.py "$CFG" > "$LOG/${rn}.slam.log" 2>&1; then
      echo "[$rn] !! SLAM FAILED"; tail -6 "$LOG/${rn}.slam.log"; touch "$out/.FAILED"; return 1; fi
  if ! env RUN_TAG="$tag" SEED="$seed" $BASE $extra python scripts/gs_eval.py --config "$CFG" --run "$out" > "$LOG/${rn}.eval.log" 2>&1; then
      echo "[$rn] !! EVAL FAILED"; tail -6 "$LOG/${rn}.eval.log"; touch "$out/.FAILED"; return 1; fi
  [ -d /content/drive/MyDrive ] && { d="$DRIVE/$rn"; mkdir -p "$d"; cp "$out"/metrics.txt "$out"/*_6panel.mp4 "$out"/est_c2w_data.txt "$out"/gt_xyz.txt "$LOG/${rn}".*.log "$d/" 2>/dev/null; }
  touch "$out/.DONE"; echo "[$rn] DONE -> $(grep -E 'PSNR|SSIM|LPIPS|Sim3 ATE' "$LOG/${rn}.eval.log" | tr -s ' \n' ' ')"
}

# operating baseline (nofwd_lr) re-run = clean render reference for this batch
run_arm nflr_ref      ""                                                0
# full SH = THE lever (view-dependent color; where GS should beat NeRF on specular tissue)
run_arm fullsh        "SIMPLIFY=0"                                      0
run_arm fullsh_dens   "SIMPLIFY=0 DENSIFY=1"                            0
run_arm fullsh_all    "SIMPLIFY=0 DENSIFY=1 MAP_ITERS=60 KEYFRAME_EVERY=4"  0
run_arm fullsh_map60  "SIMPLIFY=0 MAP_ITERS=60"                         0
run_arm fullsh_kf4    "SIMPLIFY=0 KEYFRAME_EVERY=4"                     0
# isolate the non-SH levers (do they help without full SH?)
run_arm dens_only     "DENSIFY=1"                                      0
run_arm map60_only    "MAP_ITERS=60"                                   0
# VRAM fallback if full SH+densify OOMs the T4 at native res
run_arm fullsh_ds2    "SIMPLIFY=0 DENSIFY=1 DOWNSAMPLE=2"              0

echo; echo "===== RENDER SWEEP COMPLETE $(date) ====="
printf "%-22s %8s %7s %7s %8s\n" ARM PSNR SSIM LPIPS ATE_mm
for m in experiments/CRCD_base/*/metrics.txt; do rn=$(basename "$(dirname "$m")")
  v=$(awk '/PSNR/{p=$2}/SSIM/{s=$2}/LPIPS/{l=$2}/ATE_mm/{a=$2}END{printf "%.2f %.3f %.3f %.3f",p,s,l,a}' "$m")
  printf "%-22s %s\n" "$rn" "$v"; done
