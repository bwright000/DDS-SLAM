#!/bin/bash
# ============================================================================
# ORACLE-FIELD A/B (option A) -- "given a CORRECT deformation field, does the RENDER improve?"
# Offline map trainer (no SLAM, identity poses) on SemSup trail_3: static (dead-field baseline) vs
# oracle (baked teacher dx* injected via the oracle_dx seam). Decides the render VALUE of the whole
# field-revival direction: trial_3 pins wander a +/-27px envelope, so a static map MUST blur there;
# if even a correct field can't beat it on PSNR/SSIM/LPIPS, render is field-blind in principle and
# the field's value case moves to tracking/correspondence (pin-EPE/STIR).
#   cd /content/DDS-SLAM && git pull && bash Addons/colab/oracle_field_ab_20260703.sh
#   knobs: ITERS=20000  DEFORM_SCALE=1.0  (sweep DEFORM_SCALE if the |dx*|-vs-trunc hazard bites)
# ============================================================================
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../.." && pwd); cd "$REPO"
say(){ echo -e "\n[$(date +%H:%M:%S)] $*"; }
ITERS="${ITERS:-20000}"
DEFORM_SCALE="${DEFORM_SCALE:-1.0}"
DATASET=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
DD=$REPO/data/Super/trail_3
DRIVE=/content/drive/MyDrive/Outputs/oracle_field_$(date +%Y%m%d)
mkdir -p "$DRIVE"

[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
python -c "import torch, tinycudann" 2>/dev/null || { say "env build (~15min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; }

# ---- stage trial_3 (rgb + moge2 depth + baked dx*) ----
mkdir -p "$DD/rgb" "$DD/depth/moge2" "$DD/deform"
[ "$(ls "$DD/rgb"/*left.png 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/rgb"/*left.png "$DD/rgb/"
[ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/depth/moge2"/*left_depth.npy "$DD/depth/moge2/"
[ "$(ls "$DD/deform"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/deform"/*_deform.npz "$DD/deform/"
say "staged: rgb $(ls "$DD/rgb"/*left.png|wc -l)  depth $(ls "$DD/depth/moge2"/*.npy|wc -l)  deform $(ls "$DD/deform"/*.npz|wc -l)"
# rgb-derived aux the Super loader expects (seg masks etc.) come from the same Drive tree if present
for SUB in seg pose; do
  [ -d "$DATASET/$SUB" ] && [ ! -d "$DD/$SUB" ] && cp -r "$DATASET/$SUB" "$DD/$SUB" || true
done

# ---- SMOKE both arms first (integration errors cost 2min, not 2h) ----
say "SMOKE static + oracle"
python Addons/experiments/oracle_field_trainer.py --arm static --out output/oracle_ab/_smoke_s --smoke || { say "SMOKE static FAILED"; exit 1; }
python Addons/experiments/oracle_field_trainer.py --arm oracle --out output/oracle_ab/_smoke_o --smoke --deform_scale "$DEFORM_SCALE" || { say "SMOKE oracle FAILED"; exit 1; }
say ">>> SMOKE PASS"

# ---- full arms (same seed => identical ray sequence) ----
for ARM in static oracle; do
  say "ARM $ARM (iters $ITERS)"
  EXTRA=""; [ "$ARM" = "oracle" ] && EXTRA="--deform_scale $DEFORM_SCALE"
  python Addons/experiments/oracle_field_trainer.py --arm "$ARM" --out "output/oracle_ab/$ARM" --iters "$ITERS" $EXTRA \
    2>&1 | tee "$DRIVE/${ARM}.log" | grep -E "^\[" || { say "ARM $ARM FAILED"; exit 1; }
  CUDA_VISIBLE_DEVICES="" python Addons/eval/eval_rendering.py --gt_dir "$DD/rgb" --render_dir "output/oracle_ab/$ARM" \
    --name "oracle_ab_$ARM" > "$DRIVE/render_eval_${ARM}.txt" 2>&1 || echo WARN-render-eval
  tail -8 "$DRIVE/render_eval_${ARM}.txt"
done

# ---- pin-patch panel (GT | static | oracle at the largest-envelope pins) ----
python Addons/experiments/oracle_field_trainer.py --compare output/oracle_ab/static output/oracle_ab/oracle \
  --gt_dir "$DD/rgb" --panel_out "$DRIVE/pin_panel.png" || echo WARN-panel
cp output/oracle_ab/static/0075.jpg "$DRIVE/static_f75.jpg" 2>/dev/null
cp output/oracle_ab/oracle/0075.jpg "$DRIVE/oracle_f75.jpg" 2>/dev/null
say "DONE -> $DRIVE  (render_eval_{static,oracle}.txt + pin_panel.png = the verdict)"
