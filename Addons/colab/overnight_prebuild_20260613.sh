#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-4 — PRE-BUILD tests (SemSup only, T4). 2026-06-13.
#
# Tests the "route, don't suppress" thesis BEFORE building the learned gate.
# All pose-FROZEN (field can't hide motion in the camera) + time-fix + reg.
#
#   pb_control      : NO gate  — field trains on full residual (the control).
#   pb_oracle       : GATE on  — Δx routed by seg edge-prior (reg 1e-3).   << decisive
#   pb_oracle_reg1e2: GATE on  — stronger reg 1e-2 (divergence hedge).
#
# DECISION (read field_liveness.json per run):
#   GATE revives a USEFUL field iff oracle is LIVE (temporal_frac substantial,
#   finite mean_norm) AND more spatially-localised than control AND render>=baseline.
#   -> GO: build the learned gate.  oracle dead while control same -> starvation
#   binds -> tracks mandatory.  Both diverge -> reg too weak.
#
# T0.1 signal_probe runs on pb_control's ckpt (deform-off render vs seg-prior):
#   the cheap "does an attribution signal even EXIST" gate.
#
# Read-out per run: field_liveness (rescaling-invariant Var_t) + timenet_audit +
# dx_hook + dx_norm_heatmap + inline renders + pose_path (should be ~0, frozen).
# set -uo (not -e); resume-safe (.DONE markers). Colab VS Code tunnel.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_prebuild_${DATE}
LWORK=/content/prebuild
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-4 PRE-BUILD start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
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

# $1=cfg  $2=out_base (=data.output)  $3=label  $4=run_signal_probe(1/0)
run_test(){
  local CFG=$1 OUTB=$2 LBL=$3 SIG=${4:-0}
  local DST="$DRIVE/$LBL" LW="$LWORK/$LBL" DEMO="$OUTB/demo"
  done_marker "$DST" && { say "  $LBL already shipped -- skip"; return 0; }
  rm -rf "$LW"; mkdir -p "$LW" "$DST"
  say "=== TEST $LBL  (cfg=$CFG) ==="
  cd "$REPO"; local T0; T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN: $LBL run nonzero exit"
  say "  $LBL run elapsed $(( ($(date +%s)-T0)/60 )) min"
  local CKPT; CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  ERROR: no checkpoint in $DEMO -- skip diagnostics"; return 1; }
  say "  ckpt: $CKPT"

  # --- READ-OUT (rescaling-invariant) ---
  python diagnosis/infra/field_liveness.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/liveness.json" --n_points 4096 2>&1 | tee -a "$LOG" || say "  WARN field_liveness"
  python diagnosis/infra/timenet_weight_audit.py --ckpt "$CKPT" --json "$LW/timenet_audit.json" 2>&1 | tee -a "$LOG" || say "  WARN audit"
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
  python diagnosis/infra/dx_norm_heatmap.py --config "$CFG" --checkpoint "$CKPT" \
    --rgb_dir "$REPO/data/Super/trail_3/rgb" --rgb_pattern '*left.png' --output_dir "$LW/dx_heatmap" 2>&1 | tee -a "$LOG" || say "  WARN heatmap"
  if [ "$SIG" = "1" ]; then
    say "  T0.1 signal_probe (deform-off residual vs seg-prior)"
    python diagnosis/infra/signal_probe.py --config "$CFG" --checkpoint "$CKPT" \
      --json "$LW/signal_probe.json" --max_frames 30 --frame_stride 5 2>&1 | tee -a "$LOG" || say "  WARN signal_probe"
  fi

  # --- validate.json (one-glance verdict) ---
  python - "$CFG" "$LW" "$DEMO" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, numpy as np
from config import load_config
cfg_path, lw, demo = sys.argv[1], sys.argv[2], sys.argv[3]
c = load_config(cfg_path)
def g(*ks):
    x=c
    for k in ks: x = x.get(k) if isinstance(x,dict) else None
    return x
v = {'oracle_routing': c.get('oracle_routing'),
     'time_normalize': g('training','time_normalize'),
     'timenet_weight_decay': g('training','timenet_weight_decay'),
     'deformation_reg_weight': g('training','deformation_reg_weight')}
try:
    M=np.loadtxt(f'{demo}/est_c2w_data.txt').reshape(-1,3,4); t=M[:,:3,3]
    v['pose_path_mm']=round(float(np.linalg.norm(np.diff(t,axis=0),axis=1).sum()*1000),4)
except Exception as e: v['pose_path_mm']=f'ERR {e}'
try:
    L=json.load(open(f'{lw}/liveness.json'))
    for k in ['verdict','mean_norm','temporal_frac','cov_t','spatial_std_of_meanmag','max_norm']:
        v[f'live_{k}']=L.get(k)
except Exception as e: v['live_verdict']=f'ERR {e}'
try:
    a=json.load(open(f'{lw}/timenet_audit.json')); v['timenet_l2']=round(a['timenet'][0]['l2'],4)
except Exception as e: v['timenet_l2']=f'ERR {e}'
try:
    s=json.load(open(f'{lw}/signal_probe.json'))
    v['T0_1_verdict']=s.get('verdict'); v['T0_1_ratio']=s.get('res_high_over_low'); v['T0_1_pearson']=s.get('pearson_residual_vs_prior')
except Exception: pass
json.dump(v, open(f'{lw}/validate.json','w'), indent=2); print('  VALIDATE:', json.dumps(v))
PY

  mkdir -p "$LW/inline_renders"; cp "$OUTB"/*.jpg "$LW/inline_renders/" 2>/dev/null || true
  [ -d "$OUTB/depth" ] && cp -r "$OUTB/depth" "$LW/inline_depth" 2>/dev/null || true
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  $LBL shipped -> $DST/payload.tgz"
}

nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "SemSup staging failed -- abort"; exit 1; }

# order: control first (also carries T0.1 signal probe) -> decisive oracle -> reg hedge
run_test configs/Super/pb_control.yaml       output/pb_control        pb_control        1
run_test configs/Super/pb_oracle.yaml        output/pb_oracle         pb_oracle         0
run_test configs/Super/pb_oracle_reg1e2.yaml output/pb_oracle_reg1e2  pb_oracle_reg1e2  0

say "=== battery-4 DONE $(date -Iseconds) ==="
say "READ-OUT: per test validate.json:"
say "  live_verdict  : DEAD / DIVERGED / STATIC-GAUGE / LIVE   (the call)"
say "  live_temporal_frac : fraction of Δx variance that is TEMPORAL (>0.05 = real motion, not gauge)"
say "  T0_1_*  (control only): does an attribution signal EXIST near the seg-prior"
say "DECISION: oracle LIVE + more localised than control + render>=deform-off => BUILD the learned gate."
say "          oracle dead while control same => starvation binds => tracks mandatory."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop runtime manually)"
