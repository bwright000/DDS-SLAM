#!/bin/bash
# ============================================================================
# A100 FIX-ARM — global_BA time-convention experiment.  2026-06-16.
# RUN IN A SECOND TERMINAL, SAME Colab session, ALONGSIDE the IMPROVE run.
#
# Question: does fixing the mixed time axis (global_BA fed RAW int time while everything else
# normalised) REVIVE the field -> better SemSup render?  Single variable = the time convention.
#   timeA_raw   (time_normalize:false)            = pristine-upstream all-raw, consistent
#   timeB_mixed (true + global_ba_time_fix:false) = THE BUG (what pb5/6/7 ran)
#   timeC_fixed (true + global_ba_time_fix:true)  = normalised everywhere, consistent
# Judged by render PSNR/SSIM/LPIPS (NOT field-liveness, per metric-first methodology).
#
# Shares the SESSION with the IMPROVE run:
#   * SKIPS env rebuild  — the IMPROVE run already built the modern stack + tcnn sm_80; verify only.
#   * SKIPS parity gate  — this tests OUR fork's time flags, not base-parity.
#   * SKIPS data staging — reuses the SemSup+moge2 the IMPROVE run already staged (just checks it).
#   * Separate Drive/output/LWORK dirs + low PARALLEL so it's polite to the IMPROVE run's GPU/CPU.
# Launch AFTER the IMPROVE run has passed staging (i.e. its log shows "RUN N jobs ...").
# Resume-safe (.DONE per cell).  Override the cap:  PARALLEL=3 bash Addons/colab/a100_fix_time_ab_20260616.sh
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
SEEDS="0 1 2"                                  # field is bistable -> n=3 to not be fooled by a seed
PARALLEL=${PARALLEL:-2}                        # CONSERVATIVE: sharing GPU/CPU with the IMPROVE run
# Cap CPU threads/job — PyTorch/OpenMP default to ALL cores, so parallel jobs oversubscribe ~Nx
# (the GPU is idle; the CPU thrashes). This is the SECONDARY run sharing the box with the IMPROVE
# run, so default to a small, polite value. The two runbooks DON'T coordinate threads: keep
# (improve_jobs*improve_threads + fix_jobs*THREADS) <= nproc. Running ALONE? set THREADS=$((nproc/PARALLEL)).
NPROC=$(nproc 2>/dev/null || echo 8)
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
# (threads/co-scheduling note is in the start banner below; say() is not defined yet at this line)
DDIR=data/Super/trail_3
DRIVE=/content/drive/MyDrive/Outputs/a100_fix_time_${DATE}
LWORK=/content/fixtime; mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }
cd "$REPO"
say "=== FIX-ARM time A/B/C start $(date -Iseconds)  HEAD=$(git rev-parse --short HEAD)  PARALLEL=$PARALLEL  THREADS=$THREADS/job (nproc=$NPROC; co-scheduled? keep improve+fix jobs*threads <= $NPROC) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }

# ---- env: SHARED session — verify only, do NOT rebuild ----
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python -c "import torch, tinycudann; assert torch.cuda.is_available()" \
  || { say "FATAL: env not ready. Start the IMPROVE run first (it builds the modern stack + tcnn sm_80)."; exit 1; }
python -c "import lpips" 2>/dev/null || pip install -q lpips || true

# ---- data: reuse what the IMPROVE run staged — verify, do NOT re-stage ----
[ -d "$REPO/$DDIR/depth/moge2" ] \
  || { say "FATAL: SemSup moge2 depth not staged at $DDIR/depth/moge2. Let the IMPROVE run stage it first (or stage moge2 manually)."; exit 1; }
say "  SemSup moge2 present: $(ls "$REPO/$DDIR/depth/moge2"/*left_depth.npy 2>/dev/null | wc -l) npy"

# ---- 6-panel video (SemSup; no seg-classmap, no uncert since the field configs run uncertainty OFF) ----
make_video(){ local RUN=$1 OUT=$2
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$DDIR/rgb" --rgb_input_pattern '*left.png' \
    --rgb_output_dir "$RUN" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DDIR/depth/moge2" \
    --depth_output_dir "$RUN/depth" --depth_norm robust \
    --seg_dir "$DDIR/seg/png_masks" --seg_pattern '*left.png' --skip_raw_seg \
    --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DDIR/groundtruth.txt" --trajectory_raw --skip_horn_traj \
    --output "$OUT" --fps 15 2>&1 || echo "WARN video failed ($OUT)"
}

# ---- run_one CONFIG SEED -> one (time-variant, seed): train + video + render-eval + package ----
run_one(){ local CFG=$1 S=$2
  local CELL="${CFG}_s${S}" DST="$DRIVE/${CFG}_s${S}" LW="$LWORK/${CFG}_s${S}"
  done_marker "$DST" && { say "  $CELL already done -> skip"; return 0; }
  mkdir -p "$LW" "$DST"
  local OUT="output/_fixtime/${CELL}" OVR="/content/_fixcfg_${CELL}.yaml" RUN="output/_fixtime/${CELL}/demo" PYRC=1 NPOSE=0
  cat > "$OVR" <<YML
inherit_from: configs/Super/${CFG}.yaml
seed: ${S}
data:
  output: ${OUT}
  exp_name: demo
YML
  {
    echo "=== $CELL START $(date -Iseconds) ==="
    python -W ignore - "$OVR" <<'PY'
import os, sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))   # avoid CPU oversubscription across parallel jobs
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
    PYRC=$?; [ "$PYRC" -ne 0 ] && echo "!!! TRAIN CRASHED rc=$PYRC -- this cell will NOT be marked .DONE"
    make_video "$RUN" "$LW/${CELL}_6panel.mp4"
    python Addons/eval/eval_rendering.py --gt_dir "$DDIR/rgb" --render_dir "$RUN" --name "$CELL" --sequence "Lab1 (trail3)" > "$LW/render_metrics.txt" 2>&1 || echo "WARN render-eval"
    CK=$(ls -t "$RUN"/checkpoint*.pt 2>/dev/null | head -1); [ -n "$CK" ] && cp "$CK" "$LW/checkpoint.pt"
    cp "$RUN"/est_c2w_data.txt "$RUN"/output.txt "$LW/" 2>/dev/null || true
    mkdir -p "$LW/frames_sample"; for f in $(ls "$RUN"/[0-9]*.jpg 2>/dev/null | sort | awk 'NR%30==1'); do cp "$f" "$LW/frames_sample/" 2>/dev/null; done
    echo "=== $CELL DONE $(date -Iseconds) ==="
  } > "$LW/run.log" 2>&1
  cp "$LW/run.log" "$DST/run.log" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  NPOSE=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' "$RUN/est_c2w_data.txt" 2>/dev/null); NPOSE=${NPOSE:-0}
  if [ "${PYRC:-1}" -eq 0 ] && [ "${NPOSE:-0}" -ge 1 ]; then
    sync; touch "$DST/.DONE"; rm -f "$DST/.FAILED"; say "  $CELL shipped OK (${NPOSE} poses)"
  else
    rm -f "$DST/.DONE"; echo "rc=${PYRC:-?} npose=${NPOSE:-0} $(date -Iseconds)" > "$DST/.FAILED"
    say "  $CELL FAILED (rc=${PYRC:-?}, ${NPOSE:-0} poses) -> will RE-RUN next launch. see $LW/run.log"
  fi
}

# ---- fan out timeA/B/C x seeds ----
JOBS=()
for CFG in trail3_moge2_timeA_raw trail3_moge2_timeB_mixed trail3_moge2_timeC_fixed; do
  for s in $SEEDS; do JOBS+=("$CFG|$s"); done
done
say "########## FIX time A/B/C: ${#JOBS[@]} jobs (3 variants x ${SEEDS// /,} seeds), ${PARALLEL} parallel ##########"
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r c s <<< "$spec"
  run_one "$c" "$s" &
  running=$((running+1))
  if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running-1)); fi
  sleep 3
done
wait
say "########## all ${#JOBS[@]} FIX jobs finished ##########"

# ---- SUMMARY: render A vs B vs C (does fixing the time axis beat the bug?) ----
say "########## FIX SUMMARY (SemSup render; n=3 mean+/-std) ##########"
python3 - "$LWORK" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, glob, re, numpy as np
LW=sys.argv[1]
print("SemSup MoGe2 render. time convention:  A=raw-consistent  B=MIXED(the bug)  C=fixed-consistent")
rows={}
for cfg,label in [('trail3_moge2_timeA_raw','A raw-consistent'),
                  ('trail3_moge2_timeB_mixed','B MIXED (the bug)'),
                  ('trail3_moge2_timeC_fixed','C fixed-consistent')]:
    P=[]
    for d in sorted(glob.glob(f'{LW}/{cfg}_s*')):
        t=f'{d}/render_metrics.txt'
        if not os.path.isfile(t): continue
        s=open(t).read()
        def g(k):
            m=re.search(k+r'[^0-9-]*([0-9.]+)', s); return float(m.group(1)) if m else None
        ps,ss,lp=g('PSNR'),g('SSIM'),g('LPIPS')
        if ps: P.append((ps,ss or 0,lp or 0))
    if not P: print(f"  {label:<20} (no metrics)"); continue
    A=np.array(P); rows[label]=A[:,0].mean()
    print(f"  {label:<20} n={len(P)}  PSNR={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  SSIM={A[:,1].mean():.3f}  LPIPS={A[:,2].mean():.3f}")
print("\nFIX WORKS if C (and A) BEAT B on PSNR/SSIM/LPIPS past the seed std -> the mixed time axis")
print("  was hurting render -> the time-fix earns a place in the FIX branch.")
print("If A~B~C within noise -> the time bug is not the render problem -> field issue is elsewhere.")
PY
say "=== FIX-ARM DONE $(date -Iseconds) ==="
