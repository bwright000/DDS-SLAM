#!/bin/bash
# ============================================================================
# GATE v2 VOTE-DETECTOR OFFLINE BENCH -- E3_005 + C1_001, NO SLAM (minutes, not hours).
# Runs Addons/motion/diag_vote_scan.py: the region-VOTE egomotion detector vs the OLD agreement_gate,
# side-by-side, judged against GT still/moving per frame.
# PASS BAR (from the freeze-confusion analysis of the champion runs):
#   E3_005: old gate froze 117 GT-MOVING frames (28% freeze-precision) -> the VOTE's freeze-precision
#           must be HIGH here (target >=80%), i.e. it fixes the inversion.
#   C1_001: old gate 93% freeze-precision / 88% still-recall -> the VOTE must keep recall (>=80%)
#           WITHOUT losing precision, i.e. it preserves the legitimate C1 freezing.
# The detector touches NO tracker until this passes. still_floor_px is THE calibrated constant:
# if the default 0.5 fails, sweep STILL_FLOOR="0.3 0.5 0.8" and freeze the winner in the config.
# Run (fresh Colab ok -- stages data via the rect-bench, builds env on first call):
#   cd /content/DDS-SLAM && git pull && bash Addons/colab/vote_scan_20260702.sh
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
# ALL 5 bench snippets: E3+C1 = rule DESIGN set; C2/C3/G3 = HELD-OUT validation (don't fit the rule
# to the whole benchmark). Each scan also dumps <out>_votes.npz (per-region flow/depth/centroid/
# residual + old-gate Sampson per frame) so candidate rules replay OFFLINE with no GPU.
SNIPPETS="${SNIPPETS:-E3_005 C1_001 C2_001 C3_001 G3_001}"
EVERY="${EVERY:-3}"
STILL_FLOOR="${STILL_FLOOR:-0.5}"
DRIVE=/content/drive/MyDrive/Outputs/vote_scan_$(date +%Y%m%d)
mkdir -p "$DRIVE"

# stage (env build + rectify + depth) via the rect-bench with NO arms -- staging only, runtime kept alive
SNIPPETS="$SNIPPETS" ARMS="" NO_UNASSIGN=1 bash "$HERE/rect_bench_best_vs_base_20260626.sh" || true

for NAME in $SNIPPETS; do
  DD="$REPO/data/CRCD/$NAME"
  [ -d "$DD/video_frames" ] || { echo "[vote-scan] $NAME NOT STAGED -> skip"; continue; }
  for SF in $STILL_FLOOR; do
    OUT="$DRIVE/${NAME}_floor${SF}"
    echo "[vote-scan] ===== $NAME  still_floor_px=$SF ====="
    python Addons/motion/diag_vote_scan.py --frames_dir "$DD/video_frames" --depth_dir "$DD/depth" \
      --gt "$DD/groundtruth.txt" --out "$OUT" --every "$EVERY" --still_floor_px "$SF" ${VIDEO:+--video} \
      2>&1 | tee "$OUT.log" | tail -25
  done
done
echo "[vote-scan] DONE -> $DRIVE  (read the .json freeze_precision/still_recall: E3 target >=80 prec, C1 >=80 recall)"
