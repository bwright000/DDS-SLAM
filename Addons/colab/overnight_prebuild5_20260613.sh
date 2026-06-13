#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-5 — STABILISATION sweep (SemSup only, T4). 2026-06-13.
#
# Battery-4 bracketed the bistable field (time-fix + wd0, pose-frozen):
#   reg 1e-3 -> DIVERGED (on-surface median |dx| ~4e4); reg 1e-2 -> DEAD (denormal 0).
# The finite-nonzero LIVE equilibrium is INSIDE (1e-3, 1e-2). Battery-3 showed slow
# TimeNet lr (lr_mult 0.1) tames without killing. This sweep finds the stable point:
#   reg {2e-3, 3e-3, 5e-3, 7e-3} x lr_mult {1.0 full, 0.1 slow}, gate OFF.
#
# DECISION (field_liveness.json per run): a run with verdict LIVE (finite mean_norm,
# NOT 0 and NOT >1e3, temporal_frac > 0.05) = the stable reg. That reg then feeds a
# re-run of the oracle A/B (pb_control vs pb_oracle at the stable reg) -> answers T1.3.
#
# Order = most-likely-stable first (mid-band + slow lr), so the informative cells land
# before the ~24h Colab limit. RESUME-SAFE (.DONE markers -> relaunch resumes). No
# signal_probe (T0.1 already GO). render_freq=10 inline. set -uo; VS Code tunnel.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_prebuild5_${DATE}
LWORK=/content/prebuild5
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-5 STABILISATION start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
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

  python diagnosis/infra/field_liveness.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/liveness.json" --n_points 4096 2>&1 | tee -a "$LOG" || say "  WARN field_liveness"
  python diagnosis/infra/timenet_weight_audit.py --ckpt "$CKPT" --json "$LW/timenet_audit.json" 2>&1 | tee -a "$LOG" || say "  WARN audit"
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
v = {'deformation_reg_weight': g('training','deformation_reg_weight'),
     'timenet_lr_mult': g('training','timenet_lr_mult'),
     'time_normalize': g('training','time_normalize'),
     'timenet_weight_decay': g('training','timenet_weight_decay')}
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

# order: most-likely-stable first (mid-band + slow lr), bracket outward
run_test configs/Super/pb5_reg0p003_lrslow.yaml  output/pb5_reg0p003_lrslow  pb5_reg0p003_lrslow
run_test configs/Super/pb5_reg0p005_lrslow.yaml  output/pb5_reg0p005_lrslow  pb5_reg0p005_lrslow
run_test configs/Super/pb5_reg0p003_lrfull.yaml  output/pb5_reg0p003_lrfull  pb5_reg0p003_lrfull
run_test configs/Super/pb5_reg0p005_lrfull.yaml  output/pb5_reg0p005_lrfull  pb5_reg0p005_lrfull
run_test configs/Super/pb5_reg0p002_lrslow.yaml  output/pb5_reg0p002_lrslow  pb5_reg0p002_lrslow
run_test configs/Super/pb5_reg0p007_lrslow.yaml  output/pb5_reg0p007_lrslow  pb5_reg0p007_lrslow
run_test configs/Super/pb5_reg0p002_lrfull.yaml  output/pb5_reg0p002_lrfull  pb5_reg0p002_lrfull
run_test configs/Super/pb5_reg0p007_lrfull.yaml  output/pb5_reg0p007_lrfull  pb5_reg0p007_lrfull

say "=== battery-5 DONE $(date -Iseconds) ==="
say "READ-OUT: per test validate.json -> live_verdict + live_mean_norm (model units)."
say "  PHYSICAL TARGET: tissue moves ~a couple mm ~= 1-4% of working depth (z~1 unit) ~= mean_norm 0.01-0.05."
say "    ~0           = DEAD"
say "    0.01 - 0.05  = RIGHT (physically plausible deformation) <-- pick this reg"
say "    0.1 - ~1     = alive but TOO BIG (cm-dm deformation, non-physical)"
say "    >> 1         = DIVERGED (battery-4 reg1e-3 hit 62000)"
say "  Also check on-ray dx_hook median (near-surface) -- field_liveness mean_norm includes unconstrained"
say "  empty space and can OVERSTATE divergence; trust the band only if on-ray agrees."
say "  Pick the reg whose mean_norm lands in 0.01-0.05 -> re-run oracle A/B at it -> answers T1.3."
say "  If NONE lands there (all DEAD or DIVERGED, no middle): soft leash can't hold the physical target"
say "  -> encode the prior as a HARD bound (Dx = couple_mm * tanh(field)), or try lr_mult 0.01."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop runtime manually)"
