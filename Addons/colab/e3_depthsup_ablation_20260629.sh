#!/bin/bash
# ============================================================================
# OVERNIGHT E3_005 DEPTH-SUPERVISOR ablation (thin wrapper over rect_bench_best_vs_base_20260626.sh).
# Tests the NEW depth-supervisor tracking (L0) on the snippet the OLD flow-gate INVERTED (E3_005, the
# tool-dominated one). 5 arms x 1 seed = 5 runs (fits one overnight). The old flow-gate champion is
# NOT re-run here -- its E3 number (Sim3 6.09mm, Pearson .24, INVERTED) is the known reference to beat;
# the clean control is abl_base.
#   abl_base  = improved canon + charbonnier (tracking control = the number to beat)
#   l0        = depth supervisor: depth-predicted rigid-flow residual -> per-DINO-region adaptive soft
#               weight -> tracking down-weight. gate=false (ALWAYS track). mad_c 2.0. No F, no labels.
#   l0aggr    = L0 with an aggressive trust knee (mad_c 1.5)
#   l0sig     = L0 + Inc-1/2 sigma^2 (now FIRES -- the old freeze-gate masked it; scene_rep multiplies
#               the L0 weight by 1/sigma^2). geo v1 = the canon head.
#   l0sigaggr = l0sig with mad_c 1.5
#
# JUDGE (the inversion test): Sim3 ATE + est/GT path-ratio + |Pearson|dom (sim3_ate.py) PRIMARY;
#   render PSNR/SSIM/LPIPS + Depth-L1 SECONDARY. Decisive read: does l0 pull E3 BELOW abl_base and
#   flip Pearson back UP (un-invert), and does sigma^2/aggressiveness push further.
# DIAGNOSIS per run: run.log [depth_sup] per-frame stats + trust_log.csv + 6-panel video now WITH the
#   Trust-Weight panel (which regions were down-weighted) + the sigma^2 panel + a FIXED 4-class seg panel.
# HONESTY: n=1 = a SCREEN (seed coin-flip is the noise floor). The winner earns n=3 + the full 5-snippet
#   bench (C1/C2/C3/G3) before any claim -- an E3-only win can cost elsewhere.
# DEFERRED (documented blockers, not in this screen): the TOOL arm (CRCD semantic_paths globs the binary
#   masks/, so tool_mask_track is a no-op until the seg-label dir is decoupled from the edge field) and
#   the EM-refine arm (needs a safe tracking-loop refactor).
#
# RUN (fresh clone on the A100/T4 box):
#   cd /content/DDS-SLAM && git pull && bash Addons/colab/e3_depthsup_ablation_20260629.sh
#   knobs: SEEDS="0 1 2"  ARMS="abl_base l0"  PARALLEL=2 (default 1; >1 also prewarm-safe via torch.hub cache)
# Resume-safe (.DONE per arm on Drive). Outputs -> MyDrive/Outputs/rect_bestbase_${DATE}/ ; SUMMARY.txt at the end.
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
export SNIPPETS="${SNIPPETS:-E3_005}"
export ARMS="${ARMS:-abl_base l0 l0aggr l0sig l0sigaggr}"
export SEEDS="${SEEDS:-0}"
export DATE="${DATE:-e3abl2_20260630}"   # v2: the v1 (e3abl_20260629) run was VOID -- L0 was a no-op (OpenGL/OpenCV bug, fixed @186b307); new dir so the .DONE-resume-safe runbook re-runs all 5 arms with L0 actually firing
echo "[e3-depthsup-ablation] snippets='$SNIPPETS' arms='$ARMS' seeds='$SEEDS' -> Outputs/rect_bestbase_${DATE}"
exec bash "$HERE/rect_bench_best_vs_base_20260626.sh"
