#!/usr/bin/env bash
# Overnight CONFIG-ONLY sweep — CRCD c1_001 on EndoGSLAM. Each arm = env overrides on crcd_base.py
# (defaults = base => bit-identical baseline). Per arm: SLAM -> gs_eval (5 metrics + 6-panel video) ->
# ship to Drive. NOT `set -e` (one arm failing must not kill the batch). Resumable (.DONE skips, .FAILED logs).
# Highest-EV arms first so a short night still gets the essentials (baseline + the LR lever).
#
#   bash /content/DDS-SLAM/Addons/gs/overnight_crcd.sh            # full sweep
#   ARMS="base lrt02" bash .../overnight_crcd.sh                  # subset by tag
set -uo pipefail

ENDO=${ENDO:-/content/EndoGSLAM}
CFG=configs/crcd/crcd_base.py
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/GS_phase0}
LOG="$ENDO/_overnight_logs"; mkdir -p "$LOG"
cd "$ENDO"

run_arm(){  # tag  "ENV ASSIGNMENTS"  seed
  local tag="$1" envv="$2" seed="${3:-0}"
  local rn="C1_001_${tag}_s${seed}" out
  out="experiments/CRCD_base/${rn}"; mkdir -p "$out"
  [ -f "$out/.DONE" ] && { echo "[$rn] .DONE -> skip"; return 0; }
  echo "=== $(date +%H:%M) ARM $rn   env:[$envv] ==="
  if ! env RUN_TAG="$tag" SEED="$seed" $envv python scripts/main.py "$CFG" > "$LOG/${rn}.slam.log" 2>&1; then
      echo "[$rn] !! SLAM FAILED -> $LOG/${rn}.slam.log (tail:)"; tail -5 "$LOG/${rn}.slam.log"; touch "$out/.FAILED"; return 1; fi
  if ! env RUN_TAG="$tag" SEED="$seed" $envv python scripts/gs_eval.py --config "$CFG" --run "$out" > "$LOG/${rn}.eval.log" 2>&1; then
      echo "[$rn] !! EVAL FAILED -> $LOG/${rn}.eval.log (tail:)"; tail -5 "$LOG/${rn}.eval.log"; touch "$out/.FAILED"; return 1; fi
  if [ -d /content/drive/MyDrive ]; then
      local d="$DRIVE/$rn"; mkdir -p "$d"
      cp "$out"/metrics.txt "$out"/*_6panel.mp4 "$out"/est_c2w_data.txt "$out"/gt_xyz.txt "$LOG/${rn}".*.log "$d/" 2>/dev/null
  fi
  touch "$out/.DONE"
  echo "[$rn] DONE -> $(grep -E 'PSNR|SSIM|LPIPS|L1-Depth|Sim3 ATE' "$LOG/${rn}.eval.log" | tr -s ' \n' ' ')"
}

# ============================ THE SWEEP (order = highest EV first) ============================
# 1) baseline n=3  -> reference + seed-std (the noise floor a win must clear)
run_arm base ""                                       0
run_arm base ""                                       1
run_arm base ""                                       2
# 2) translation-LR lever (#0 proven DDS fix; zero runtime cost; directly targets the path-ratio jitter)
run_arm lrt05  "LR_TRANS_MULT=0.5"                    0
run_arm lrt02  "LR_TRANS_MULT=0.2"                    0
run_arm lrt01  "LR_TRANS_MULT=0.1"                    0
# 3) rotation-LR + combined gentle pose updates
run_arm lrr02   "LR_ROT_MULT=0.2"                     0
run_arm lrboth  "LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2"   0
# 4) static-heavy CRCD: const-velocity OFF, and combined with the gentle-LR winner candidate
run_arm nofwd     "FWD_PROP=0"                                       0
run_arm nofwd_lr  "FWD_PROP=0 LR_TRANS_MULT=0.2 LR_ROT_MULT=0.2"     0
# 5) render / coverage arms (cost runtime; lower EV/cost) — aid the final model's map quality
run_arm trk30   "TRK_ITERS=30"                        0
run_arm map40   "MAP_ITERS=40"                        0
run_arm densify "DENSIFY=1"                           0
run_arm sil95   "SIL_THRES=0.95"                      0

echo; echo "================= SWEEP COMPLETE $(date) ================="
printf "%-26s %8s %7s %7s %9s %9s\n" ARM PSNR SSIM LPIPS L1d_mm ATE_mm
for m in experiments/CRCD_base/*/metrics.txt; do
  rn=$(basename "$(dirname "$m")")
  vals=$(awk '/PSNR/{p=$2}/SSIM/{s=$2}/LPIPS/{l=$2}/L1depth_mm/{d=$2}/ATE_mm/{a=$2}END{printf "%.2f %.3f %.3f %.2f %.3f",p,s,l,d,a}' "$m")
  printf "%-26s %s\n" "$rn" "$vals"
done
