#!/bin/bash
# ============================================================================
# ONE-SHOT OVERNIGHT ORCHESTRATOR — fire-and-forget.  2026-06-17.
# The Colab cell mounts Drive + clones the repo, then runs THIS. It does everything else:
#   1. shared setup  : full modern stack + tcnn(live GPU) + MoGe + SemSup stage + parity
#   2. run, throttled: Field-Diagnosis (T1, PARALLEL=1) + Depth-Gen (T3, MoGe VRAM-capped) CONCURRENT
#   3. bake          : Motion-Teacher signal maps from field_on poses (for the morning eyeball)
# Resumable: re-running the cell re-enters (env verify is fast; .DONE-gated cells skip).
# All logs + payloads land under MyDrive/Outputs/. NO tunnel, NO manual terminals.
#
# T4 budget: Field-Diag = 1 SLAM proc (~5GB) + Depth-Gen MoGe capped at 0.4*16=6.4GB -> ~11GB, fits.
# NOTE: keep the Colab browser tab OPEN (free Colab disconnects idle sessions); Pro = background exec.
# ============================================================================
set -uo pipefail
REPO=/content/DDS-SLAM; cd "$REPO" 2>/dev/null || { echo "FATAL: $REPO missing (clone step did not run)"; exit 1; }
DATE=$(date +%Y%m%d)
OUT=/content/drive/MyDrive/Outputs/overnight_${DATE}; mkdir -p "$OUT" 2>/dev/null
MLOG="$OUT/MASTER.log"; exec > >(tee -a "$MLOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] [MASTER] $*"; }
say "=== OVERNIGHT START $(date -Iseconds)  HEAD=$(git rev-parse --short HEAD 2>/dev/null) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted — run the mount cell first"; exit 1; }
nvidia-smi -L 2>/dev/null || say "WARN: no GPU visible"

# ---- 1. shared setup (env build ~15 min on a fresh Colab, then stage + parity) ----
say "STEP 1/4: shared setup (t4_setup_shared)"
bash Addons/colab/t4_setup_shared_20260617.sh || { say "FATAL: setup failed — see $OUT/.. setup log"; exit 1; }

# ---- 2. Field-Diag (T1) + Depth-Gen (T3) CONCURRENT, throttled for the shared T4 ----
say "STEP 2/4: launch Field-Diag + Depth-Gen (background, throttled)"
PARALLEL=1 bash Addons/colab/a100_arm2_field_diag_20260617.sh > "$OUT/fielddiag.log" 2>&1 &
FPID=$!
MOGE_MEM_FRAC=0.4 MOGE_RES=7 bash Addons/colab/crcd_depth_rawleft_all_20260618.sh > "$OUT/depthgen.log" 2>&1 &
DPID=$!
say "  Field-Diag pid=$FPID -> $OUT/fielddiag.log   Depth-Gen pid=$DPID -> $OUT/depthgen.log"

# ---- 3. when Field-Diag is done, bake the Motion-Teacher signal (morning eyeball; NO masking yet) ----
wait "$FPID"; say "STEP 3/4: Field-Diag finished -> bake Motion-Teacher maps from field_on poses"
EST=""
for s in 0 1 2; do c="output/_arm2/trail3_field_on_s${s}/demo/est_c2w_data.txt"; [ -f "$c" ] && EST="$c" && break; done
if [ -n "$EST" ] && [ -d data/Super/trail_3/depth/moge2 ]; then
  python Addons/motion/bake_motion_residual.py --est_c2w "$EST" \
    --rgb_dir data/Super/trail_3/rgb --rgb_glob '*left.png' \
    --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy' --depth_scale 8 \
    --out_dir data/Super/trail_3/motion > "$OUT/motion_bake.log" 2>&1 \
    && { mkdir -p "$OUT/motion"; cp data/Super/trail_3/motion/*_motion.npy "$OUT/motion/" 2>/dev/null; \
         say "  motion maps baked -> $OUT/motion (EYEBALL: high on deforming tissue, ~0 on static?)"; } \
    || say "  WARN motion bake failed -> see $OUT/motion_bake.log"
else
  say "  WARN no field_on est_c2w (or no moge2 depth) -> motion bake skipped"
fi

# ---- 4. wait for Depth-Gen (Arm-3 metric depth, the x9-10 fix) ----
say "STEP 4/4: waiting for Depth-Gen"
wait "$DPID"; say "Depth-Gen finished."
say "=== OVERNIGHT DONE $(date -Iseconds) ==="
say "Results: Field-Diag -> MyDrive/Outputs/a100_arm2_field_*/   Depth-Gen -> Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth_stereo120/"
say "         Motion maps -> $OUT/motion (verify before we wire the teacher).  Master log -> $MLOG"
