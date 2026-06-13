#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-6 — THE BUILD GATE (T1.3) (SemSup, T4). 2026-06-13.
#
# Tier-1 done: reg 0.003 + lr_mult 0.1 = stable ~mm field. Now: does seg-ROUTING make the
# field's deformation localise to WHERE THE RIGID MODEL FAILS (anti-circular), more than the
# bare field? GO -> build the learned attribution head. NO-GO -> routing isn't the lever.
#
# Ray convention fixed (OpenGL, commit 0ed7135) so dx_seg_localise / dx_hook are in the model's
# frame (prior OpenCV bug made direction read backwards). field_liveness is convention-free.
#
# Cells (pose-frozen; baseline must RE-RUN — its ckpt only lived in ephemeral /content):
#   1 baseline      pb5_reg0p003_lrslow  (stable field, NO gate)        <- anchor for the gate metric
#   2 seg-gate      pb6_seg              (+ oracle_routing seg)          <- THE gate test
#   3 hardbound     pb6_hardbound        (tanh cap, reg0, full lr, no gate) <- cleaner stabiliser + insurance
#   4 hardbound+seg pb6_hardbound_seg    (hardbound + gate)             <- decouple gate from soft-reg knife-edge
#
# DECISION: dx_seg_localise.pearson_dx_vs_residual — seg-gate (cell2) > baseline (cell1) = gate routes
# deformation to rigid-failure regions = GO (screen). hardbound cells = is the cap a robust substrate +
# does the gate help independent of the soft-reg sweet spot. n=1 = SCREEN; seeds/shuffled = next overnight.
#
# ~2.25h/cell -> ~9h + read-out. Resume-safe (.DONE). set -uo. VS Code tunnel / notebook bash.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_battery6_${DATE}
LWORK=/content/battery6
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-6 BUILD GATE start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }

activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "modern stack missing -- full rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" || { say "env check FAIL"; exit 1; }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ -d "$REPO/data/Super/trail_3/rgb" ]; then say "SemSup already staged"; return 0; fi
  [ -d "$SRC/rgb" ] || { say "FATAL: SemSup source missing at $SRC"; return 1; }
  say "staging SemSup (trial_3 -> trail_3)"; mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"
}

# $1=cfg  $2=out_base  $3=label
run_test(){
  local CFG=$1 OUTB=$2 LBL=$3
  local DST="$DRIVE/$LBL" LW="$LWORK/$LBL" DEMO="$OUTB/demo"
  done_marker "$DST" && { say "  $LBL already shipped -- skip"; return 0; }
  rm -rf "$LW"; mkdir -p "$LW" "$DST"
  say "=== TEST $LBL  (cfg=$CFG) ==="
  cd "$REPO"; local T0; T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN: $LBL run nonzero exit"
  say "  $LBL run elapsed $(( ($(date +%s)-T0)/60 )) min"
  local CKPT; CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  ERROR: no checkpoint in $DEMO"; return 1; }
  say "  ckpt: $CKPT"

  python diagnosis/infra/field_liveness.py --config "$CFG" --checkpoint "$CKPT" --json "$LW/liveness.json" 2>&1 | tee -a "$LOG" || say "  WARN field_liveness"
  python diagnosis/infra/dx_seg_localise.py --config "$CFG" --checkpoint "$CKPT" --json "$LW/seg_localise.json" --max_frames 30 --frame_stride 5 2>&1 | tee -a "$LOG" || say "  WARN dx_seg_localise"
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"

  python - "$CFG" "$LW" "$DEMO" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, numpy as np
from config import load_config
cfg_path, lw, demo = sys.argv[1], sys.argv[2], sys.argv[3]
c = load_config(cfg_path)
def g(*ks):
    x=c
    for k in ks: x = x.get(k) if isinstance(x,dict) else None
    return x
v = {'oracle_routing': c.get('oracle_routing'), 'deform_hardbound': c.get('deform_hardbound'),
     'deformation_reg_weight': g('training','deformation_reg_weight'), 'timenet_lr_mult': g('training','timenet_lr_mult')}
try:
    L=json.load(open(f'{lw}/liveness.json'))
    for k in ['verdict','mean_norm','temporal_frac','max_norm']: v[f'live_{k}']=L.get(k)
except Exception as e: v['live_verdict']=f'ERR {e}'
try:
    S=json.load(open(f'{lw}/seg_localise.json'))
    v['GATE_pearson_dx_vs_residual']=S.get('pearson_dx_vs_residual')
    v['pearson_dx_vs_segprior']=S.get('pearson_dx_vs_segprior')
    v['concentration_hiRes_over_loRes']=S.get('concentration_hiRes_over_loRes')
except Exception as e: v['GATE_pearson_dx_vs_residual']=f'ERR {e}'
json.dump(v, open(f'{lw}/validate.json','w'), indent=2); print('  VALIDATE:', json.dumps(v))
PY

  mkdir -p "$LW/inline_renders"; cp "$OUTB"/*.jpg "$LW/inline_renders/" 2>/dev/null || true
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  $LBL shipped -> $DST/payload.tgz"
}

nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "SemSup staging failed -- abort"; exit 1; }

# core gate test first (baseline -> seg), then the hardbound decoupling pair
run_test configs/Super/pb5_reg0p003_lrslow.yaml output/pb5_reg0p003_lrslow pb6_baseline
run_test configs/Super/pb6_seg.yaml             output/pb6_seg             pb6_seg
run_test configs/Super/pb6_hardbound.yaml       output/pb6_hardbound       pb6_hardbound
run_test configs/Super/pb6_hardbound_seg.yaml   output/pb6_hardbound_seg   pb6_hardbound_seg

say "=== battery-6 DONE $(date -Iseconds) ==="
say "READ-OUT (per cell validate.json):"
say "  GATE_pearson_dx_vs_residual = does Δx localise to where the RIGID model fails (anti-circular)."
say "  GATE DECISION (screen): pb6_seg > pb6_baseline => seg-routing helps => GO build the learned head."
say "                          pb6_seg ~ pb6_baseline => routing isn't the lever => pivot (tracks/map-side)."
say "  hardbound vs baseline = is the tanh cap a cleaner stable substrate; hardbound_seg vs hardbound ="
say "    does the gate help independent of the soft-reg knife-edge. live_verdict must be LIVE every cell."
say "  n=1 = SCREEN -> a positive gap greenlights a 3-seed + shuffled-label CONFIRMATION overnight, not the build."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop runtime manually)"
