#!/bin/bash
# ============================================================================
# DDS-SLAM BATTERY-2 — field-revival diagnosis (SemSup only, T4)
#
# Part B (is the field doing ANYTHING?):  baseline / baseline_rep / deformoff
#   -> compare backbone weights + pose + render across the 3; deformoff within the
#      baseline<->baseline_rep band == field fully inert.
# Part C (can we FIX it? pose frozen so the field is isolated):
#   posefrozen (ref) / pf_gridslow (map?) / pf_timenet_boost (wd+lr?) /
#   pf_noanchor (t=0 anchor?) / pf_revive (all levers = the fix attempt).
#
# Each run: timenet_weight_audit + dx_hook + dx_norm_heatmap + SELF-VALIDATION
# (echoes resolved knobs + checks the intended mechanism left its fingerprint).
# Inline render only (render_freq=10) -- NO posthoc render_all_frames (saves time).
# set -uo (not -e); resume-safe via .DONE. Paste into a Colab VS Code tunnel term.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_hypconfirm2_${DATE}
LWORK=/content/hypconfirm2
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== battery-2 start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
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

# --- per-test driver: run + diagnostics + SELF-VALIDATE + ship -----------------
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
  # SELF-VALIDATION: echo resolved knobs + check the intended fingerprint
  python - "$CFG" "$LW" "$DEMO" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, glob, os, numpy as np
from config import load_config
cfg_path, lw, demo = sys.argv[1], sys.argv[2], sys.argv[3]
c = load_config(cfg_path)
def g(*ks):
    x=c
    for k in ks: x = x.get(k) if isinstance(x,dict) else None
    return x
v = {'deformation_off': c.get('deformation_off', False),
     'deformation_anchor_off': c.get('deformation_anchor_off', False),
     'lr_embed': g('mapping','lr_embed'),
     'timenet_weight_decay': g('training','timenet_weight_decay'),
     'timenet_lr_mult': g('training','timenet_lr_mult'),
     'track_lr_trans': g('tracking','lr_trans'), 'const_speed': g('tracking','const_speed')}
# pose path (freeze check)
try:
    M=np.loadtxt(f'{demo}/est_c2w_data.txt').reshape(-1,3,4); t=M[:,:3,3]
    v['pose_path_mm']=float(np.linalg.norm(np.diff(t,axis=0),axis=1).sum()*1000)
except Exception as e: v['pose_path_mm']=f'ERR {e}'
# timenet l2 (dead?) from audit
try:
    a=json.load(open(f'{lw}/timenet_audit.json')); v['timenet_l2']=[round(x['l2'],4) for x in a['timenet'][:1]]; v['timenet_dead']=a['timenet'][0]['dead']
except Exception as e: v['timenet_l2']=f'ERR {e}'
# dx at t=0 and t=75 (anchor-off check + revival)
for fr in ['0000','0075']:
    try:
        z=np.load(f'{lw}/dx/frame_{fr}.npz'); n=np.linalg.norm(z['delta_x'].reshape(-1,3),axis=1)
        v[f'dx_t{fr}_mean']=float(n.mean()); v[f'dx_t{fr}_max']=float(n.max())
    except Exception as e: v[f'dx_t{fr}_mean']=f'ERR {e}'
json.dump(v, open(f'{lw}/validate.json','w'), indent=2)
print('  VALIDATE:', json.dumps(v))
PY
  # inline renders (rgb .jpg + depth) -- NO posthoc render
  mkdir -p "$LW/inline_renders"; cp "$OUTB"/*.jpg "$LW/inline_renders/" 2>/dev/null || true
  [ -d "$OUTB/depth" ] && cp -r "$OUTB/depth" "$LW/inline_depth" 2>/dev/null || true
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/groundtruth.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  $LBL shipped -> $DST/payload.tgz"
}

nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "SemSup staging failed -- abort"; exit 1; }

# Part B: is the field doing anything?
run_test configs/Super/hyp2_baseline.yaml        output/hyp2_baseline        baseline
run_test configs/Super/hyp2_baseline_rep.yaml    output/hyp2_baseline_rep    baseline_rep
run_test configs/Super/hyp2_deformoff.yaml       output/hyp2_deformoff       deformoff
# Part C: can we fix it? (pose frozen, structural levers one at a time, then combined)
run_test configs/Super/hyp2_posefrozen.yaml      output/hyp2_posefrozen      posefrozen
run_test configs/Super/hyp2_pf_gridslow.yaml     output/hyp2_pf_gridslow     pf_gridslow
run_test configs/Super/hyp2_pf_timenet_boost.yaml output/hyp2_pf_timenet_boost pf_timenet_boost
run_test configs/Super/hyp2_pf_noanchor.yaml     output/hyp2_pf_noanchor     pf_noanchor
run_test configs/Super/hyp2_pf_revive.yaml       output/hyp2_pf_revive       pf_revive

# Encoding probe (H4): can time_net represent t beyond parity? Run on deformoff ckpt
# (init weights -> actually responds; dead ckpts would read ~0). Best-effort.
say "=== encoding probe (dx_hook_sanity on deformoff ckpt) ==="
DCKPT=$(ls -t output/hyp2_deformoff/demo/checkpoint*.pt 2>/dev/null | head -1)
if [ -n "$DCKPT" ]; then
  python diagnosis/infra/dx_hook_sanity.py --config configs/Super/hyp2_deformoff.yaml --checkpoint "$DCKPT" 2>&1 | tee "$DRIVE/encoding_probe.txt" || say "  WARN dx_hook_sanity (check its args)"
fi

say "=== battery-2 DONE $(date -Iseconds) ==="
say "READ-OUT per test: $DRIVE/<label>/payload.tgz -> validate.json (knobs+fingerprint), timenet_audit.json (dead?), dx/+dx_heatmap/ (Δx revive?)"
say "Part B verdict: deformoff timenet_audit backbone l2 vs baseline/baseline_rep band + render diff."
say "Part C verdict: any pf_* with timenet NOT dead + dx>0 spatially = a revival lever; pf_revive is the combined fix."
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop runtime manually)"
