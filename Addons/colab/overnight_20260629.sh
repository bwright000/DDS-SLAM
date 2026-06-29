#!/bin/bash
# ============================================================================
# OVERNIGHT 2026-06-29 (one hands-off command):
#   1/2  E3_005 DEPTH-SUPERVISOR ablation (5 arms; the snippet the old flow-gate inverted) -- PRIORITY.
#        Runs with NO_UNASSIGN=1 so the Colab runtime is KEPT for step 2.
#   2/2  SPARE TIME: DDS-SLAM BASE on G3_001 -- the snippet missing from the rect_bestbase_20260627
#        benchmark -- slotted into that same Drive dir (DATE=20260627, base arm = crcd_improved_rect, to
#        match the existing C1/C2/E3/C3 base numbers). Unassigns the runtime when fully done.
# Both legs are resume-safe (.DONE per cell on Drive), so a disconnect/re-run continues where it stopped.
# RUN (fresh Colab, after Drive is mounted + the repo is cloned):
#   cd /content/DDS-SLAM && bash Addons/colab/overnight_20260629.sh
# Knobs pass through, e.g.:  SEEDS="0 1 2" bash Addons/colab/overnight_20260629.sh   (E3 ablation seeds)
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

echo "############################################################"
echo "### OVERNIGHT 1/2: E3_005 depth-supervisor ablation (5 arms)"
echo "############################################################"
NO_UNASSIGN=1 bash "$HERE/e3_depthsup_ablation_20260629.sh" || echo "[overnight] E3 ablation exited non-zero -> continuing to G3"

echo "############################################################"
echo "### OVERNIGHT 2/2: DDS-SLAM base on G3_001 (fills rect_bestbase_20260627)"
echo "############################################################"
SNIPPETS=G3_001 ARMS=base SEEDS=0 DATE=20260627 bash "$HERE/rect_bench_best_vs_base_20260626.sh" || echo "[overnight] G3 base exited non-zero"
echo "[overnight] DONE -- E3 ablation -> Outputs/rect_bestbase_e3abl_20260629/ ; G3 base -> Outputs/rect_bestbase_20260627/"
