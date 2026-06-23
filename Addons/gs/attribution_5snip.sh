#!/usr/bin/env bash
# Attribution panel across the CRCD testing block — is the flow residual above the gate deadband
# (real motion to detect) on snippets OTHER than the near-rigid c1_001?  LIGHT stage: rectify frames +
# 4-class seg ONLY (preprocess_crcd_published; NO MoGe/anchor/bounds — the panel needs none of it).
# Idempotent (skips staging if frames+seg already there). Resume-safe. Ships verdicts + panels to Drive.
#
#   cd /content/DDS-SLAM && git pull && bash Addons/gs/attribution_5snip.sh
#   SNIPPETS="E3_005" bash Addons/gs/attribution_5snip.sh      # one snippet
#   SMALL=1 ...                                                # raft_small (faster)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
SNIPPETS="${SNIPPETS:-C2_001 E3_005 C3_001 G3_001}"   # c1_001 already done; add it back to re-run all 5
SMALL_FLAG=""; [ "${SMALL:-0}" = "1" ] && SMALL_FLAG="--small"
DPUB="${DPUB:-/content/drive/MyDrive/Datasets/CRCD-Published}"
CALIB="${CALIB:-$DPUB/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}"
DRIVE="${DRIVE:-/content/drive/MyDrive/Outputs/GS_attribution}"; mkdir -p "$DRIVE"
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
# NAME -> CRCD-Published episode/snippet (C2_001 -> C_2/snippet_001, E3_005 -> E_3/snippet_005)
snip_src(){ local n=$1; local ep=${n%_*}; local sn=${n#*_}; echo "$DPUB/${ep:0:1}_${ep:1}/snippet_${sn}"; }

[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
[ -f "$CALIB" ] || { say "FATAL: calib pkl missing at $CALIB"; exit 1; }
python -c "import torchvision, sklearn, matplotlib" 2>/dev/null || pip install -q scikit-learn matplotlib 2>&1 | tail -1

say "=== attribution across: $SNIPPETS  (RAFT=${SMALL:+small}${SMALL:-large}) ==="
for NAME in $SNIPPETS; do
  DD="$REPO/data/CRCD/$NAME"; SRC=$(snip_src "$NAME")
  say "----- $NAME -----"
  # --- light stage (rectify frames + 4-class seg only) if not already present in either layout ---
  if { [ -d "$DD/video_frames" ] && ls "$DD/semantic_class"/*.png >/dev/null 2>&1; } \
     || { [ -d "$DD/frames" ] && ls "$DD/semantic_ids"/*.png >/dev/null 2>&1; }; then
    say "  already staged -> skip rectify"
  else
    [ -d "$SRC/rgb" ] || { say "  SKIP: source rgb missing at $SRC"; continue; }
    [ -f "$SRC/intrinsics.yaml" ] || { say "  SKIP: intrinsics.yaml missing at $SRC"; continue; }
    say "  rectify (light, no MoGe) <- $SRC"
    python Addons/preprocess/preprocess_crcd_published.py \
      --snippet_dir "$SRC" --calib_pkl "$CALIB" --output_dir "$DD" \
      || { say "  rectify FAILED -> skip"; continue; }
  fi
  # --- attribution panel (auto-detects the layout) ---
  python Addons/gs/attribution_panel.py --scene "$DD" --out "output/attribution/$NAME" \
      --drive "$DRIVE" $SMALL_FLAG || { say "  attribution FAILED"; continue; }
done

say "===== VERDICT SUMMARY ====="
printf "%-10s %8s %8s %8s %8s %-12s\n" snippet R_tool R_tissue ratio AUC verdict
for NAME in C1_001 $SNIPPETS; do
  r="output/attribution/$NAME/results.txt"; [ -f "$r" ] || r="$DRIVE/$NAME/results.txt"
  [ -f "$r" ] || { printf "%-10s %s\n" "$NAME" "(no results)"; continue; }
  rt=$(awk -F: '/residual  TOOL/{gsub(/ /,"",$2);print $2}' "$r")
  ri=$(awk -F: '/residual  TISSUE/{gsub(/ /,"",$2);print $2}' "$r")
  ra=$(awk -F'(REAL)' '/ratio  tool\/tissue  \(REAL\)/{n=split($2,a,":");gsub(/[^0-9.]/,"",a[2]);print a[2]}' "$r")
  au=$(awk -F'\\(REAL\\):' '/AUC  R: tool-vs-tissue \(REAL\)/{gsub(/[^0-9.]/,"",$2);print $2}' "$r")
  vd=$(awk '/VERDICT:/{print $3}' "$r")
  printf "%-10s %8s %8s %8s %8s %-12s\n" "$NAME" "${rt:-?}" "${ri:-?}" "${ra:-?}" "${au:-?}" "${vd:-?}"
done
say "Panels + results in $DRIVE/<snippet>/  ·  key read: which snippets have residual >> the ~3px gate deadband (real motion to detect)"
