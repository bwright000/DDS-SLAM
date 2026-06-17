#!/bin/bash
# ============================================================================
# SHARED SETUP — run ONCE, FIRST, before any of tonight's 3 terminals.  2026-06-17.
# Builds the full modern stack + tcnn for the LIVE GPU (T4=sm_75) + MoGe-2, stages SemSup,
# runs the base-parity gate, and drops a marker (/content/.dds_setup_done). The three run
# terminals (Field Diagnosis / Motion Teacher / Depth Gen) then VERIFY-ONLY against this marker
# and never rebuild/re-stage -> no env races, no triple build, no triple stage.
#
# WHY ONE SETUP: the 3 terminals share ONE Colab T4. If each built the env they'd thrash; if each
# staged SemSup they'd race the same dir. So the heavy, write-once work lives here.
#
#   bash Addons/colab/t4_setup_shared_20260617.sh      # wait for ">>> SETUP COMPLETE" before launching runs
# ============================================================================
set -uo pipefail
REPO=/content/DDS-SLAM
PRISTINE=/content/DDS-SLAM-Base-pristine
PRISTINE_SHA=009977b
MARK=/content/.dds_setup_done
DRIVE=/content/drive/MyDrive/Outputs/t4_setup_$(date +%Y%m%d); mkdir -p "$DRIVE"
LOG="$DRIVE/setup.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
cd "$REPO"
say "=== SHARED SETUP start $(date -Iseconds)  HEAD=$(git rev-parse --short HEAD) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
nvidia-smi -L || true

# --- 1. modern stack + tcnn for the LIVE GPU (probes capability -> T4 rebuilds sm_75 itself) ---
if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
  say "env rebuild (~15 min): colab_setup.sh"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; fi
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python -c "import torch; assert torch.cuda.is_available()" || { say "FATAL: no CUDA after build"; exit 1; }
CC=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo 75)
say "GPU compute capability = sm_$CC"
if ! python - <<'PY' 2>/dev/null
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,
  "log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda')); print("tcnn ok on device")
PY
then
  say "  tcnn probe FAILED on sm_$CC -> rebuild for this arch (~10 min)"
  TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps \
    "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" || { say "FATAL: tcnn rebuild"; exit 1; }
fi
python -c "import lpips" 2>/dev/null || pip install -q lpips || true

# --- 2. MoGe-2 (the Depth-Gen terminal needs it; same torch2 stack) ---
if ! python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null; then
  say "installing MoGe-2..."; pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -3
  python -c 'from moge.model.v2 import MoGeModel' || { say "FATAL: MoGe-2 not importable"; exit 1; }
fi
say "  MoGe-2 importable OK"

# --- 3. stage SemSup (Field-Diag + Motion-Teacher both read it; stage ONCE here) ---
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ ! -d "$REPO/data/Super/trail_3/rgb" ]; then
    [ -d "$SRC/rgb" ] || { say "  WARN SemSup source missing ($SRC) -> SemSup runs will fail"; return 0; }
    mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"; fi
  local DST="$REPO/data/Super/trail_3/depth/moge2"
  [ -d "$DST" ] && [ "$(ls "$DST"/*left_depth.npy 2>/dev/null | wc -l)" -ge 151 ] && { say "  SemSup moge2 depth present"; return 0; }
  local MS=""
  for c in /content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3/depth/MoGe2_trail3_20260608 \
           /content/drive/MyDrive/Datasets/SemSup/MoGe2_trail3_20260608 \
           /content/drive/MyDrive/MoGe2_trail3_20260608 ; do [ -d "$c" ] && MS="$c" && break; done
  [ -n "$MS" ] && { mkdir -p "$DST"; cp "$MS"/*left_depth.npy "$DST"/ 2>/dev/null; say "  staged moge2 depth: $(ls "$DST"/*left_depth.npy 2>/dev/null|wc -l) npy"; } \
               || say "  WARN MoGe2 SemSup depth not found -> SemSup runs will fail"
}
stage_semsup

# --- 4. base-parity gate (the SLAM terminals run pristine-base cells; confirm fork@flags-off == 009977b) ---
parity_gate(){
  local G="$DRIVE/parity"; mkdir -p "$G"
  [ -d "$PRISTINE/.git" ] || git clone -q https://github.com/IRMVLab/DDS-SLAM "$PRISTINE" || { say "  clone FAIL -> skip parity"; return 0; }
  ( cd "$PRISTINE" && git checkout -q "$PRISTINE_SHA" 2>/dev/null )
  mkdir -p "$PRISTINE/Addons/regression"
  cp "$REPO/Addons/regression/test_inc0_bitidentical.py" "$PRISTINE/Addons/regression/" 2>/dev/null
  cp "$REPO/configs/Super/trail3_paper_faithful.yaml" "$PRISTINE/configs/Super/_parity_shared.yaml" 2>/dev/null
  ( cd "$PRISTINE" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/_parity_shared.yaml --golden /content/golden_pristine.json --write-golden ) 2>&1 | tee "$G/pristine_build.txt" >/dev/null
  ( cd "$REPO" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/trail3_paper_faithful.yaml --golden /content/golden_pristine.json ) 2>&1 | tee "$G/fork_check.txt"
  grep -qi "PASS" "$G/fork_check.txt" && say "  >>> PARITY PASS (fork@flags-off == pristine $PRISTINE_SHA)" \
    || say "  >>> PARITY MISMATCH (see $G/fork_check.txt) — reconcile before trusting base cells"
}
parity_gate

sync; date -Iseconds > "$MARK"
say ">>> SETUP COMPLETE -> $MARK written. Launch the 3 terminals now (each verify-only):"
say "    T1: bash Addons/colab/a100_arm2_field_diag_20260617.sh"
say "    T2: bash Addons/colab/t4_motion_teacher_20260617.sh"
say "    T3: bash Addons/colab/crcd_depth_stereo120_20260617.sh"
say "=== SHARED SETUP DONE $(date -Iseconds) ==="
