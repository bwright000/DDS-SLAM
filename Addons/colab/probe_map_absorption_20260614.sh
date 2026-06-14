#!/bin/bash
# ============================================================================
# DDS-SLAM MAP-ABSORPTION PROBE — the pre-Inc-1 gate (SemSup, pose-frozen, T4). 2026-06-14.
#
# Battery-7 showed the field is surface-DEAD across 3 seeds EVEN WITH POSE FROZEN. Architecture read
# (project_map_averaging_mechanism): the map has NO time input -> it can only fit a blurry TIME-AVERAGE
# and WINS the gradient race; the field is starved. KEY QUESTION before building Inc-1:
#   Is there a real photometric RESIDUAL at the moving tissue (static render wrong where tissue moves)
#   for a learnt-uncertainty head to fire on?  -> GO Inc-1.   No residual -> Inc-1 can't help.
#
# Battery-7 did NOT persist the checkpoint (ephemeral). So we RE-TRAIN one seed (s2, the most-alive,
# best-case for showing a signal), PERSIST the checkpoint this time, then run:
#   - map_absorption_probe.py  (A signal-exists / B field-inert / C static-render)
#   - render_eval_attrib.py    (deform ON-vs-OFF tissue-masked PSNR/SSIM/LPIPS)
#   - Inc-0 regression golden + check (proves Arm-2 plumbing is bit-identical to base)
#
# ~37 min train + ~5 min probes. Resume-safe (.DONE). set -uo.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_mapprobe_${DATE}
LWORK=/content/mapprobe
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }

say "=== map-absorption probe start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
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

nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "stage failed -- abort"; exit 1; }

CFG=configs/Super/pb7_redesign_s2.yaml
OUTB=output/pb7_redesign_s2
DEMO="$OUTB/demo"
LW="$LWORK/s2"; mkdir -p "$LW"

# 1. RE-TRAIN s2 (checkpoint was not persisted by battery-7) -------------------
if [ -f "$DRIVE/.TRAINED" ] && [ -f "$DEMO"/checkpoint*.pt ]; then
  say "s2 already trained -- skip"
else
  say "=== RE-TRAIN s2 ($CFG) ==="; T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN nonzero exit"
  say "  train elapsed $(( ($(date +%s)-T0)/60 )) min"; touch "$DRIVE/.TRAINED"
fi
CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
[ -n "$CKPT" ] || { say "  FATAL no ckpt"; exit 1; }
say "  ckpt = $CKPT"

# 2. MAP-ABSORPTION PROBE (the gate) ------------------------------------------
say "=== map_absorption_probe ==="
python diagnosis/infra/map_absorption_probe.py --config "$CFG" --checkpoint "$CKPT" \
  --json "$LW/map_absorb_s2.json" --max_frames 40 --frame_stride 3 2>&1 | tee -a "$LOG" || say "  WARN probe"

# 3. FIELD-ATTRIBUTION render eval (deform ON vs OFF) -------------------------
say "=== render_eval_attrib (ON vs OFF) ==="
python diagnosis/infra/render_eval_attrib.py --config "$CFG" --checkpoint "$CKPT" \
  --json "$LW/field_attrib_s2.json" --max_frames 40 --frame_stride 3 2>&1 | tee -a "$LOG" || say "  WARN attrib"

# 4. INC-0 REGRESSION (golden + check; proves Arm-2 plumbing == base) ---------
say "=== Inc-0 regression (bit-identical gate) ==="
python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml \
  --write-golden 2>&1 | tee -a "$LOG" || say "  WARN golden"
cp Addons/regression/golden_inc0.json "$LW/" 2>/dev/null || true
python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml \
  2>&1 | tee -a "$LOG" || say "  WARN inc0 check"

# 5. SHIP (persist the checkpoint this time + all jsons) ----------------------
say "=== ship ==="
cp "$CKPT" "$LW/checkpoint150.pt" 2>/dev/null || say "  WARN ckpt copy"
cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
tar czf "$DRIVE/payload_s2.tgz.partial" -C "$LW" . && mv "$DRIVE/payload_s2.tgz.partial" "$DRIVE/payload_s2.tgz"
sync; touch "$DRIVE/.DONE"

# 6. READ-OUT ----------------------------------------------------------------
python - "$LW" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, os
lw = sys.argv[1]
try:
    p = json.load(open(f'{lw}/map_absorb_s2.json'))
    print("\n================ MAP-ABSORPTION VERDICT ================")
    print(f"  VERDICT: {p.get('VERDICT')}")
    print(f"  signal_exists={p.get('VERDICT_signal_exists')}  field_inert={p.get('VERDICT_field_inert')}")
    print(f"  (A) resid_off moving/static ratio = {p.get('A_moving_over_static_ratio')}  (want >2)")
    print(f"  (A) pearson(resid_off, gt_var)    = {p.get('A_pearson_residoff_vs_gtvar')}  (want >0.2)")
    print(f"  (B) psnr ON-OFF moving            = {p.get('B_psnr_on_minus_off_moving')}  (~0 = field inert, expected)")
    print(f"  (C) off temporal std @ moving     = {p.get('C_off_temporal_std_moving')}  (~0 = static map can't track)")
    print("  GO Inc-1 if signal_exists=True; else motion sub-SNR or map tracks it.")
except Exception as e:
    print("  ERR reading map_absorb_s2.json:", e)
PY
say "=== map-absorption probe DONE $(date -Iseconds) ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
