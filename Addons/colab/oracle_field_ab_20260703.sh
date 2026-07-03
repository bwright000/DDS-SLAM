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
[ "$(ls "$DD/deform"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ] || cp "$DATASET/deform"/*_deform.npz "$DD/deform/"
# rgb-derived aux the Super loader expects (seg masks etc.)
for SUB in seg pose; do
  [ -d "$DATASET/$SUB" ] && [ ! -d "$DD/$SUB" ] && cp -r "$DATASET/$SUB" "$DD/$SUB" || true
done
# DEPTH: try the known Drive layouts; if none match, GENERATE MoGe fresh (151 frames ~2-3min on T4)
# and BACKFILL the canonical Drive location so no future instance regenerates.
if [ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -lt 151 ]; then
  for CAND in "$DATASET/depth/moge2" "$DATASET/MoGe2_trail3_20260608" \
              /content/drive/MyDrive/Datasets/MoGe2_trail3_20260608 \
              /content/drive/MyDrive/MoGe2_trail3_20260608; do
    if [ "$(ls "$CAND"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ]; then
      say "depth: restoring from $CAND"; cp "$CAND"/*left_depth.npy "$DD/depth/moge2/"; break
    fi
  done
fi
if [ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -lt 151 ]; then
  say "depth: no Drive cache matched -> generating MoGe-2 fresh (~2-3min)"
  PSCALE=$(python -c "import config; print(config.load_config('configs/Super/trail3_teacher_off.yaml')['cam']['png_depth_scale'])")
  say "depth: png_depth_scale from config = $PSCALE"
  mkdir -p "$DD/_mi"; for f in "$DD/rgb"/*left.png; do b=$(basename "$f"); ln -sf "$f" "$DD/_mi/${b%left.png}-left.png"; done
  python Addons/depth/generate_depth_moge.py --rgb "$DD/_mi" --out "$DD/depth/moge2" \
    --temporal_window 1 --depth_scale "$PSCALE" --max_depth_m 5.0 --resolution_level 9 \
    || { say "FATAL: MoGe generation failed"; exit 1; }
  rm -rf "$DD/_mi"
fi
# PERSIST: whatever the source (restore or fresh generation), make sure the CANONICAL Drive location
# (the workspace-doc path every runbook checks FIRST) holds the full set -> no instance ever re-stages.
if [ "$(ls "$DATASET/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -lt 151 ] \
   && [ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ]; then
  mkdir -p "$DATASET/depth/moge2" && cp "$DD/depth/moge2"/*left_depth.npy "$DATASET/depth/moge2/" \
    && say "depth: PERSISTED to canonical Drive path ($DATASET/depth/moge2, $(ls "$DATASET/depth/moge2"/*left_depth.npy|wc -l) npy)" \
    || say "WARN: Drive depth persist failed -- next instance will re-stage"
fi
say "staged: rgb $(ls "$DD/rgb"/*left.png|wc -l)  depth $(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)  deform $(ls "$DD/deform"/*.npz|wc -l)  seg $(ls "$DD/seg/png_masks"/*left.png 2>/dev/null|wc -l)"
[ "$(ls "$DD/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ] || { say "FATAL: depth staging failed"; exit 1; }

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
