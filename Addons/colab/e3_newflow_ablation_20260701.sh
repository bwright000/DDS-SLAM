#!/bin/bash
# ============================================================================
# E3_005 NEW-FLOW ablation -- the flow-supervisor arms vs base, on the tool-dominated snippet the old
# flow-gate INVERTED. Thin wrapper over rect_bench_best_vs_base_20260626.sh.
#   abl_base = improved canon + charbonnier            (tracking control)
#   l0       = L0 depth-supervisor soft re-weighter     (reference; the fixed-Pearson prior result)
#   dpool    = MODE B: pose-free depth-pooled region gate (pool flow+depth into DINO regions, *depth to a
#              common plane, threshold deviation from the consensus)  [the user's simple idea]
#   pnp      = MODE A: robust 2D-3D PnP camera-motion solve, ref-depth-only, tool-masked, flow-floor gated,
#              used INIT-ONLY (up-to-scale, no metric anchor -- per the internal review)
# JUDGE (flow_diag.py, auto per run): D1 CAMERA-ACTIVATION TIMING (Spearman rho(est step, GT step) + moving/
#   still ratio; PASS rho>=0.5 & ratio>=2) and D2 OVER-TRAVEL (path-ratio toward 1; PASS [0.7,1.4]). Plus the
#   regression gate: Sim3 ATE + path-ratio + ALIGNED Pearson (sim3_ate) + PSNR/SSIM/LPIPS + Depth-L1.
#   Baselines to beat (E3 v2): base rho 0.24/pr 2.84 ; l0 0.32/1.90 ; l0sig 0.30/1.42. n=1 = SCREEN ->
#   the winner earns n=3 + the full C1/C2/C3/G3 bench before any claim.
# WATCH FIRST (per the review): D2 path-ratio (does over-travel shrink toward 1?) and D1 rho (does the camera
#   activate in time?) -- NOT ATE/PSNR. For pnp also grep run.log '[solve_pnp]' -> is it solving (not all
#   'const-velocity fallback') and is |t|/inl sane.
# Run (fresh Colab, Drive mounted, repo cloned):
#   cd /content/DDS-SLAM && git pull && bash Addons/colab/e3_newflow_ablation_20260701.sh
#   knobs: SEEDS="0 1 2"  ARMS="abl_base pnp"  DATE=...   Outputs -> MyDrive/Outputs/rect_bestbase_${DATE}/
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SNIPPETS="${SNIPPETS:-E3_005}"
export ARMS="${ARMS:-abl_base l0 dpool pnp}"
export SEEDS="${SEEDS:-0}"
export DATE="${DATE:-newflow_20260701}"
echo "[e3-newflow] snippets='$SNIPPETS' arms='$ARMS' seeds='$SEEDS' -> Outputs/rect_bestbase_${DATE}"
exec bash "$HERE/rect_bench_best_vs_base_20260626.sh"
