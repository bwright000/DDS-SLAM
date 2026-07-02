#!/bin/bash
# ============================================================================
# E3 prior LAMBDA-SWEEP -- calibrate the LEAN-CORE zero-motion prior's ONE balance (fix-1: it is one
# calibrated-and-frozen constant, NOT zero constants). The prior = lam_t|d_t|^2 + lam_r|d_rot|^2 competes with
# the SDF tracking loss whose scale is dominated by sdf_weight (~1e3), so the absolute lambda MUST be tuned.
# Runs the `prior` arm on E3_005 (which has genuine still segments) across a log-grid of lambda_t, with
# lambda_r tied = lambda_t * RATIO (rotation vs translation floors ~ Z^2). Per lambda -> flow_diag D2
# path-ratio + D1 moving-rho. FREEZE the lambda where D2 path-ratio -> 1 WITH D1 moving-rho UNHARMED, then
# write lam_r/lam_t into configs/CRCD/crcd_abl_prior_rect.yaml (the frozen headline config).
#   Run: cd /content/DDS-SLAM && git pull && bash Addons/colab/e3_prior_lam_sweep_20260702.sh
#   knobs: LAMS="1e4 1e5 1e6 1e7"  RATIO=0.1   -> Outputs/rect_bestbase_prior_lamT<val>/
# WATCH: path-ratio should fall toward 1 as lambda rises; too-high lambda over-pins -> moving-rho collapses
#   (camera frozen through real motion). The sweet spot is the largest lambda that leaves moving-rho intact.
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LAMS="${LAMS:-1e4 1e5 1e6 1e7}"     # lambda_t grid
RATIO="${RATIO:-0.1}"               # lambda_r : lambda_t
# CRITICAL: keep the Colab runtime ALIVE across all lambdas -- rect_bench calls runtime.unassign() at its end
# unless NO_UNASSIGN is set, so without this only lambda #1 would run and the session would die.
export NO_UNASSIGN=1
for LT in $LAMS; do
  LR=$(python -c "print(${LT}*${RATIO})")
  echo "[prior-sweep] ===== lambda_t=$LT  lambda_r=$LR ====="
  # each lambda reuses the SAME local cell dir (only the Drive DST is per-lambda via DATE); clear stale
  # renders so a crashed lambda cannot poison the next lambda's eval.
  rm -rf output/E3_005_prior_s0
  SNIPPETS=E3_005 ARMS=prior SEEDS=0 DATE="prior_lamT${LT}" \
    DDS_MP_LAM_T="$LT" DDS_MP_LAM_R="$LR" bash "$HERE/rect_bench_best_vs_base_20260626.sh"
done
echo "[prior-sweep] DONE -- compare flow_diag path-ratio + moving-rho across Outputs/rect_bestbase_prior_lamT*/"
echo "[prior-sweep] then FREEZE the winning lam_r/lam_t into configs/CRCD/crcd_abl_prior_rect.yaml"
