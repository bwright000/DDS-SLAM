#!/bin/bash
# ============================================================================
# VERIFY the gradient-attribution probe on the persisted overnight checkpoints. 2026-06-15.
# Confirms "field gets ~0 gradient (signal-not-reaching-field)" is REAL, not a probe artifact,
# via connectivity + effect + finite-difference + positive-control (see verify_grad_probe.py).
# Eval-only, ~3 min/ckpt after env. Reads the persisted ckpt from the overnight Drive payload.
# ============================================================================
set -uo pipefail
REPO=/content/DDS-SLAM
OVN=/content/drive/MyDrive/Outputs/dds_overnight_mapprobe_20260614     # where the overnight shipped
DST=/content/drive/MyDrive/Outputs/dds_verify_grad_20260615; mkdir -p "$DST"
LOG="$DST/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
cd "$REPO"
if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
  say "env rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel
fi
python -c "import torch, tinycudann; assert torch.cuda.is_available()" || { say "env FAIL"; exit 1; }
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
# SemSup must be staged for get_dataset (rgb/depth/direction)
SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
[ -d "$REPO/data/Super/trail_3/rgb" ] || { mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"; }

for s in s0 s1 s2; do
  say "=== verify $s ==="
  PAY="$OVN/$s/payload.tgz"
  [ -f "$PAY" ] || { say "  no payload for $s -- skip"; continue; }
  WK=/content/verify_$s; rm -rf "$WK"; mkdir -p "$WK"
  tar xzf "$PAY" -C "$WK" ./checkpoint150.pt 2>/dev/null || { say "  no ckpt in $s payload -- skip"; continue; }
  python diagnosis/infra/verify_grad_probe.py \
    --config "configs/Super/pb7_redesign_$s.yaml" \
    --checkpoint "$WK/checkpoint150.pt" \
    --json "$DST/verify_$s.json" 2>&1 | tee -a "$LOG" || say "  WARN verify $s"
done
say "=== verify DONE; results in $DST/verify_*.json ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
