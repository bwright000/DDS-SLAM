#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-3 — deformation-field REVIVAL sweep (SemSup only, T4)
#
# Battery-2 showed the field is NOT gradient-dead: wd=1e-6 over-damps it to denormal;
# wd=0 lets it DIVERGE (l2 1e5, dx->Inf). The missing piece is a stabilizing deformation
# regularizer (||dx||^2) — which the authors named (time_smoothness_weight etc.) but never
# wired, and which we just wired (deformation_reg_weight). This battery sweeps that weight,
# pose-frozen + wd=0 + NORMAL lr (no lr x10 confound), to find the stable point between
# denormal-collapse and divergence:
#   wd0_noreg  (reg=0)   — control: does wd=0 alone diverge at normal lr, or grow controlled?
#   reg_0p1 / reg_1 / reg_10 / reg_100  — the regularizer sweep.
# Verdict per run from validate.json: timenet_l2 (dead 0 / stable finite / diverged) +
# dx_t0075 (0 / finite useful / Inf). A run with finite nonzero l2 + finite structured dx +
# improved render = REVIVAL (the redesign element #1 stable point).
#
# Inline render only (render_freq=10). set -uo (not -e); resume-safe. Colab VS Code tunnel.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_hypconfirm3_${DATE}
LWORK=/content/hypconfirm3
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-3 start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
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

# $1=cfg  $2=out_base (=data.output)  $3=label
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
  [ -n "$CKPT" ] || { say "  ERROR: no checkpoint in $DEMO -- skip diagnostics"; return 1; }
  say "  ckpt: $CKPT"
  python diagnosis/infra/timenet_weight_audit.py --ckpt "$CKPT" --json "$LW/timenet_audit.json" 2>&1 | tee -a "$LOG" || say "  WARN audit"
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
  python diagnosis/infra/dx_norm_heatmap.py --config "$CFG" --checkpoint "$CKPT" \
    --rgb_dir "$REPO/data/Super/trail_3/rgb" --rgb_pattern '*-left.png' --output_dir "$LW/dx_heatmap" 2>&1 | tee -a "$LOG" || say "  WARN heatmap"
  python - "$CFG" "$LW" "$DEMO" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, numpy as np
from config import load_config
cfg_path, lw, demo = sys.argv[1], sys.argv[2], sys.argv[3]
c = load_config(cfg_path)
def g(*ks):
    x=c
    for k in ks: x = x.get(k) if isinstance(x,dict) else None
    return x
v = {'timenet_weight_decay': g('training','timenet_weight_decay'),
     'deformation_reg_weight': g('training','deformation_reg_weight'),
     'timenet_lr_mult': g('training','timenet_lr_mult'),
     'track_lr_trans': g('tracking','lr_trans'), 'const_speed': g('tracking','const_speed')}
try:
    M=np.loadtxt(f'{demo}/est_c2w_data.txt').reshape(-1,3,4); t=M[:,:3,3]
    v['pose_path_mm']=float(np.linalg.norm(np.diff(t,axis=0),axis=1).sum()*1000)
except Exception as e: v['pose_path_mm']=f'ERR {e}'
try:
    a=json.load(open(f'{lw}/timenet_audit.json')); v['timenet_l2']=round(a['timenet'][0]['l2'],4); v['timenet_dead']=a['timenet'][0]['dead']
except Exception as e: v['timenet_l2']=f'ERR {e}'
for fr in ['0000','0075']:
    try:
        z=np.load(f'{lw}/dx/frame_{fr}.npz'); n=np.linalg.norm(z['delta_x'].reshape(-1,3),axis=1)
        v[f'dx_t{fr}_mean']=float(n.mean()); v[f'dx_t{fr}_max']=float(n.max())
    except Exception as e: v[f'dx_t{fr}_mean']=f'ERR {e}'
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

run_test configs/Super/hyp3_wd0_noreg.yaml  output/hyp3_wd0_noreg  wd0_noreg
run_test configs/Super/hyp3_reg_0p1.yaml    output/hyp3_reg_0p1    reg_0p1
run_test configs/Super/hyp3_reg_1.yaml      output/hyp3_reg_1      reg_1
run_test configs/Super/hyp3_reg_10.yaml     output/hyp3_reg_10     reg_10
run_test configs/Super/hyp3_reg_100.yaml    output/hyp3_reg_100    reg_100

say "=== battery-3 DONE $(date -Iseconds) ==="
say "READ-OUT: per test validate.json -> timenet_l2 (0=dead / finite=alive / huge=diverged) + dx_t0075 (0 / finite useful / Inf)."
say "REVIVAL = a reg weight with finite-nonzero timenet_l2 + finite structured dx + render >= baseline. That weight = redesign element #1."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop runtime manually)"
