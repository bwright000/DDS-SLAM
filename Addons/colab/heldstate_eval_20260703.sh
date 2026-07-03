#!/bin/bash
# ============================================================================
# HELD-STATE ORACLE EVAL -- "from the FINAL frozen map, does a correct field render better?"
# The decisive protocol after the oracle-ONLINE negative (-0.24 dB paired, recency-masked):
# re-render EVERY frame from each arm's FINAL checkpoint (no training), static vs oracle-warped.
# NO retraining -- consumes the two checkpoints the 2026-07-03 online A/B already shipped.
# Decides: oracle wins moving-region PSNR -> deformation DOF has reconstruction value (build the
# per-KF/graph representation, GS v2 informed). Oracle loses -> field value = tracking/pin-EPE only.
#   cd /content/DDS-SLAM && git pull && bash Addons/colab/heldstate_eval_20260703.sh
#   knobs: CKPT_STATIC / CKPT_ORACLE (Drive paths)  DEFORM_SCALE=1.0  MOVE_THRESH=0.02
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
say(){ echo -e "\n[$(date +%H:%M:%S)] $*"; }
DEFORM_SCALE="${DEFORM_SCALE:-1.0}"
MOVE_THRESH="${MOVE_THRESH:-0.02}"
MC=/content/drive/MyDrive/Outputs/manual_cells
CKPT_STATIC="${CKPT_STATIC:-$MC/oracle_online_static/checkpoint.pt}"
CKPT_ORACLE="${CKPT_ORACLE:-$MC/oracle_online_oracle/checkpoint.pt}"
DATASET=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
DD=$REPO/data/Super/trail_3
DRIVE=/content/drive/MyDrive/Outputs/heldstate_$(date +%Y%m%d)
mkdir -p "$DRIVE"

[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
[ -f "$CKPT_STATIC" ] || { say "FATAL: static ckpt missing: $CKPT_STATIC"; exit 1; }
[ -f "$CKPT_ORACLE" ] || { say "FATAL: oracle ckpt missing: $CKPT_ORACLE"; exit 1; }
python -c "import torch, tinycudann" 2>/dev/null || { say "env build (~15min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; }
python -c "import lpips" 2>/dev/null || pip install -q lpips   # the online box lacked it; LPIPS required this time

# ---- stage trial_3 (rgb + moge2 depth + baked dx*) -- same block as oracle_field_ab ----
mkdir -p "$DD/rgb" "$DD/depth/moge2" "$DD/deform"
[ "$(ls "$DD/rgb"/*left.png 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/rgb"/*left.png "$DD/rgb/"
[ "$(ls "$DD/deform"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/deform"/*_deform.npz "$DD/deform/"
for SUB in seg pose; do
  [ -d "$DATASET/$SUB" ] && [ ! -d "$DD/$SUB" ] && cp -r "$DATASET/$SUB" "$DD/$SUB" || true
done
if [ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -lt 151 ]; then
  for CAND in "$DATASET/depth/moge2" "$DATASET/MoGe2_trail3_20260608" \
              /content/drive/MyDrive/Datasets/MoGe2_trail3_20260608 \
              /content/drive/MyDrive/MoGe2_trail3_20260608; do
    if [ "$(ls "$CAND"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ]; then
      say "depth: restoring from $CAND"; cp "$CAND"/*left_depth.npy "$DD/depth/moge2/"; break
    fi
  done
fi
say "staged: rgb $(ls "$DD/rgb"/*left.png|wc -l)  depth $(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)  deform $(ls "$DD/deform"/*.npz|wc -l)"
[ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ] || { say "FATAL: depth staging failed (run oracle_field_ab_20260703.sh once to backfill Drive)"; exit 1; }

# ---- SMOKE both arms (4 frames each; integration errors cost 2min, not 1h) ----
say "SMOKE static + oracle (4 frames)"
python Addons/experiments/heldstate_eval.py --arm static --ckpt "$CKPT_STATIC" --out output/heldstate/_smoke_s --smoke || { say "SMOKE static FAILED"; exit 1; }
python Addons/experiments/heldstate_eval.py --arm oracle --ckpt "$CKPT_ORACLE" --out output/heldstate/_smoke_o --smoke --deform_scale "$DEFORM_SCALE" || { say "SMOKE oracle FAILED"; exit 1; }
say ">>> SMOKE PASS"

# ---- full held-state renders (151 frames from each FINAL map) ----
for ARM in static oracle; do
  CK="$CKPT_STATIC"; EXTRA=""
  [ "$ARM" = "oracle" ] && CK="$CKPT_ORACLE" && EXTRA="--deform_scale $DEFORM_SCALE"
  say "ARM $ARM"
  python Addons/experiments/heldstate_eval.py --arm "$ARM" --ckpt "$CK" --out "output/heldstate/$ARM" $EXTRA \
    2>&1 | tee "$DRIVE/heldstate_${ARM}.log" | grep -E "^\[" || { say "ARM $ARM FAILED"; exit 1; }
  python Addons/eval/eval_rendering.py --gt_dir "$DD/rgb" --render_dir "output/heldstate/$ARM" \
    --name "heldstate_$ARM" > "$DRIVE/render_eval_${ARM}.txt" 2>&1 || echo WARN-render-eval
  tail -10 "$DRIVE/render_eval_${ARM}.txt"
done

# ---- paired verdict: global + moving/static split + pin panel ----
say "COMPARE (paired per-frame + masked moving/static split)"
python Addons/experiments/heldstate_eval.py --compare output/heldstate/static output/heldstate/oracle \
  --gt_dir "$DD/rgb" --deform_dir "$DD/deform" --move_thresh "$MOVE_THRESH" \
  2>&1 | tee "$DRIVE/heldstate_verdict.txt"
python Addons/experiments/oracle_field_trainer.py --compare output/heldstate/static output/heldstate/oracle \
  --gt_dir "$DD/rgb" --panel_out "$DRIVE/heldstate_pin_panel.png" || echo WARN-panel

cp output/heldstate/static/perframe.npz "$DRIVE/perframe_static.npz" 2>/dev/null
cp output/heldstate/oracle/perframe.npz "$DRIVE/perframe_oracle.npz" 2>/dev/null
cp output/heldstate/static/heldstate_summary.txt "$DRIVE/summary_static.txt" 2>/dev/null
cp output/heldstate/oracle/heldstate_summary.txt "$DRIVE/summary_oracle.txt" 2>/dev/null
say "DONE -> $DRIVE  (heldstate_verdict.txt + render_eval_{static,oracle}.txt + heldstate_pin_panel.png = the verdict)"
