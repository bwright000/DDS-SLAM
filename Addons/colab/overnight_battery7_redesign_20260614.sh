#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-7 — REDESIGN reproducibility re-test (SemSup, T4). 2026-06-14.
#
# Battery-6 found the "stable" field is a SEED COIN-FLIP (same config LIVE 0.12 vs DEAD 5.9e-9) because
# seed_everything() was never called. The redesign fixes the substrate:
#   - seed_everything NOW CALLED (ddsslam.py __init__) -> deterministic per seed,
#   - deform_surface_bind:2.0 -> deformation restricted to ~2*trunc of the surface (kills the off-surface
#     explosion that inflated b6 raw 108 -> tanh saturation),
#   - reg 0.003 + lr_mult 0.1 retained (the stabiliser).
# Run the SAME redesign config at 3 SEEDS. If LIVE + bounded + CONSISTENT across seeds -> the redesign
# fixed the reproducibility failure (the field is now robust, not balanced on a knife-edge).
#
# CPU pre-validation (diagnosis/synth/deform_recovery_cpu.py, ran locally): the field RECOVERS a known
# deformation (struct_corr 0.99) and the bound MUST exceed the deformation magnitude or it saturates
# (bound 0.005 < 0.03 -> rel_epe 3.3). bound 0.04 here is safe (>> ~mm).
#
# ~37 min/cell -> ~2h. Resume-safe (.DONE). set -uo.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_battery7_${DATE}
LWORK=/content/battery7
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-7 REDESIGN start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "modern stack missing -- full rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" || { say "env FAIL"; exit 1; }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ -d "$REPO/data/Super/trail_3/rgb" ]; then say "SemSup staged"; return 0; fi
  [ -d "$SRC/rgb" ] || { say "FATAL: SemSup source missing"; return 1; }
  mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"
}
run_test(){
  local CFG=$1 OUTB=$2 LBL=$3
  local DST="$DRIVE/$LBL" LW="$LWORK/$LBL" DEMO="$OUTB/demo"
  done_marker "$DST" && { say "  $LBL shipped -- skip"; return 0; }
  rm -rf "$LW"; mkdir -p "$LW" "$DST"
  say "=== TEST $LBL ($CFG) ==="; cd "$REPO"; local T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN $LBL nonzero exit"
  say "  $LBL elapsed $(( ($(date +%s)-T0)/60 )) min"
  local CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  ERROR no ckpt"; return 1; }
  python diagnosis/infra/field_liveness.py --config "$CFG" --checkpoint "$CKPT" --json "$LW/liveness.json" 2>&1 | tee -a "$LOG" || say "  WARN liveness"
  python diagnosis/infra/dx_seg_localise.py --config "$CFG" --checkpoint "$CKPT" --json "$LW/seg_localise.json" --max_frames 30 --frame_stride 5 2>&1 | tee -a "$LOG" || say "  WARN localise"
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
  python - "$CFG" "$LW" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json
from config import load_config
c=load_config(sys.argv[1]); lw=sys.argv[2]
v={'seed':c.get('seed'),'deform_surface_bind':c.get('deform_surface_bind'),
   'reg':c.get('training',{}).get('deformation_reg_weight'),'lr_mult':c.get('training',{}).get('timenet_lr_mult')}
try:
    L=json.load(open(f'{lw}/liveness.json'))
    for k in ['verdict','mean_norm','temporal_frac','max_norm']: v[f'live_{k}']=L.get(k)
except Exception as e: v['live_verdict']=f'ERR {e}'
try:
    S=json.load(open(f'{lw}/seg_localise.json')); v['pearson_dx_vs_residual']=S.get('pearson_dx_vs_residual'); v['mean_surface_dx']=S.get('mean_surface_dx')
except Exception: pass
json.dump(v,open(f'{lw}/validate.json','w'),indent=2); print('  VALIDATE:',json.dumps(v))
PY
  mkdir -p "$LW/inline_renders"; cp "$OUTB"/*.jpg "$LW/inline_renders/" 2>/dev/null || true
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  $LBL shipped"
}
nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "stage failed -- abort"; exit 1; }
run_test configs/Super/pb7_redesign_s0.yaml output/pb7_redesign_s0 pb7_redesign_s0
run_test configs/Super/pb7_redesign_s1.yaml output/pb7_redesign_s1 pb7_redesign_s1
run_test configs/Super/pb7_redesign_s2.yaml output/pb7_redesign_s2 pb7_redesign_s2
say "=== battery-7 DONE $(date -Iseconds) ==="
say "READ-OUT: the 3 seeds' validate.json -> live_verdict + live_mean_norm."
say "  REDESIGN WORKS if all 3 = LIVE, bounded (mean_norm in ~mm band, max_norm finite), and CONSISTENT"
say "  (no DEAD/LIVE flip across seeds). That = the coin-flip is fixed -> a ROBUST stable substrate."
say "  Then the field is ready for signal/recovery (synthetic-GT supervision or tracks) = the next milestone."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
