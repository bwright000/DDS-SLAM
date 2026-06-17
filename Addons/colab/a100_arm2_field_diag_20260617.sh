#!/bin/bash
# ============================================================================
# A100 ARM-2 — DEFORMATION-FIELD DIAGNOSIS on SemSup (trail3).  2026-06-17.
# RUN IN A SECOND TERMINAL, SAME Colab session, ALONGSIDE the ARM-1 run
# (a100_dino_smoke_ab_20260617.sh), which builds the env + stages SemSup.
#
# QUESTION (the FIX branch): is DDS-SLAM's deformation field doing anything, and if it's inert, is
# the cause weight-decay collapse?  THREE single-variable cells, n=3:
#   field_off     dynamic:False                 = static Co-SLAM (NO warp)   <- the null
#   field_on      dynamic:True, tn_wd 1e-6       = pristine deformable        <- the reference
#   field_on_wd0  dynamic:True, tn_wd 0.0        = field ON, wd-collapse OFF  <- cause-discrimination
#
# 🚨 METRIC HONESTY (this is the whole point — see memory project_arm2_deformation_field_diagnosis):
#   render PSNR/SSIM/LPIPS and rigid pin-reprojection are BOTH structurally FIELD-BLIND
#   (time-averaged map + per-frame pose absorb tissue motion; rigid rep-err warps pins by the CAMERA
#   only, never by the field). So they are recorded as BASELINES + a sanity check, NOT the verdict.
#   This runbook's real product is the SAVED CHECKPOINTS + ests, from which the DECISIVE,
#   field-SENSITIVE metrics are computed POST-HOC (field-warped green-pin EPE + TimeNet ||W||),
#   on the laptop where the warp DIRECTION can be verified carefully (every prior in-line field
#   probe was retracted for a convention bug — we do NOT deploy an unverified field metric overnight).
#
# Shares the session with the ARM-1 run: SKIPS env rebuild / parity / data staging (verify only),
# low PARALLEL, separate Drive/output/work dirs. Launch AFTER ARM-1 logs "RUN N jobs ...".
# Resume-safe (.DONE per cell).  Override:  PARALLEL=2 PTS=/path/to/trial_3_l_pts.npy bash <this>
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
SEEDS="0 1 2"                                  # field is bistable / seed-sensitive -> n=3 (never trust n=1)
PARALLEL=${PARALLEL:-2}                        # polite: sharing GPU/CPU with the ARM-1 run
NPROC=$(nproc 2>/dev/null || echo 8)
THREADS=${THREADS:-2}                          # co-scheduled: keep (arm1_jobs*arm1_thr + this_jobs*THREADS) <= nproc
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
DDIR=data/Super/trail_3
DEPTH0=${DEPTH0:-$REPO/$DDIR/depth/moge2/000000-left_depth.npy}   # anchor depth (model's scale: npy/8 m)
DRIVE=/content/drive/MyDrive/Outputs/a100_arm2_field_${DATE}
LWORK=/content/arm2field; mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }
cd "$REPO"
say "=== ARM-2 FIELD DIAGNOSIS start $(date -Iseconds)  HEAD=$(git rev-parse --short HEAD)  PARALLEL=$PARALLEL  THREADS=$THREADS/job (nproc=$NPROC) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }

# ---- env: SHARED session — verify only, do NOT rebuild ----
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
python -c "import torch, tinycudann; assert torch.cuda.is_available()" \
  || { say "FATAL: env not ready. Start the ARM-1 run first (it builds the modern stack + tcnn sm_80)."; exit 1; }
python -c "import lpips" 2>/dev/null || pip install -q lpips || true

# ---- data: reuse what ARM-1 staged — verify, do NOT re-stage ----
[ -d "$REPO/$DDIR/depth/moge2" ] \
  || { say "FATAL: SemSup moge2 depth not staged at $DDIR/depth/moge2. Let the ARM-1 run stage it first."; exit 1; }
say "  SemSup moge2 present: $(ls "$REPO/$DDIR/depth/moge2"/*left_depth.npy 2>/dev/null | wc -l) npy"
[ -f "$DEPTH0" ] || say "  WARN anchor depth0 missing ($DEPTH0) -> rigid pin rep-err will skip"

# ---- resolve the green-pin GT file (deformation GT; rigid rep-err here, field-warped post-hoc) ----
PTS="${PTS:-}"
if [ -z "$PTS" ]; then
  for c in /content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3/rgb/trial_3_l_pts.npy \
           /content/drive/MyDrive/Datasets/SemSup/v2_data02/v2_data/trial_3/rgb/trial_3_l_pts.npy \
           "$REPO/$DDIR/rgb/trial_3_l_pts.npy" ; do
    [ -f "$c" ] && PTS="$c" && break
  done
fi
[ -n "$PTS" ] && [ -f "$PTS" ] && say "  green-pin GT: $PTS" \
  || say "  WARN no trial_3_l_pts.npy found -> rigid pin rep-err SKIPPED (stage it to Drive or pass PTS=...). Renders + checkpoints still produced for the post-hoc field-warped metric."

# ---- 6-panel video (SemSup; uncertainty OFF in all field cells -> no uncert panel) ----
make_video(){ local RUN=$1 OUT=$2
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$DDIR/rgb" --rgb_input_pattern '*left.png' \
    --rgb_output_dir "$RUN" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DDIR/depth/moge2" \
    --depth_output_dir "$RUN/depth" --depth_norm robust \
    --seg_dir "$DDIR/seg/png_masks" --seg_pattern '*left.png' --skip_raw_seg \
    --trajectory_est "$RUN/demo/est_c2w_data.txt" --trajectory_gt "$DDIR/groundtruth.txt" --trajectory_raw --skip_horn_traj \
    --output "$OUT" --fps 15 2>&1 || echo "WARN video failed ($OUT)"
}

# ---- run_one CFG SEED -> train + video + render-eval + rigid pin rep-err + SAVE CKPT + package ----
run_one(){ local CFG=$1 S=$2
  local CELL="${CFG}_s${S}" DST="$DRIVE/${CFG}_s${S}" LW="$LWORK/${CFG}_s${S}"
  done_marker "$DST" && { say "  $CELL already done -> skip"; return 0; }
  mkdir -p "$LW" "$DST"
  local OUT="output/_arm2/${CELL}" OVR="/content/_arm2cfg_${CELL}.yaml" RUN="output/_arm2/${CELL}/demo" PYRC=1 NPOSE=0
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
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
    PYRC=$?; [ "$PYRC" -ne 0 ] && echo "!!! TRAIN CRASHED rc=$PYRC -- this cell will NOT be marked .DONE"
    make_video "$OUT" "$LW/${CELL}_6panel.mp4"
    python Addons/eval/eval_rendering.py --gt_dir "$DDIR/rgb" --render_dir "$OUT" --name "$CELL" --sequence "Lab1 (trail3)" > "$LW/render_metrics.txt" 2>&1 || echo "WARN render-eval"
    # rigid pin reprojection error (FIELD-BLIND baseline; intrinsics default == SemSup; depth_scale 8 == loader)
    if [ -n "$PTS" ] && [ -f "$PTS" ] && [ -f "$DEPTH0" ]; then
      python Addons/eval/compute_rep_err.py --est_c2w "$RUN/est_c2w_data.txt" --pts "$PTS" \
        --depth0 "$DEPTH0" --depth_scale 8.0 > "$LW/rep_err_rigid.txt" 2>&1 || echo "WARN rep-err"
    fi
    # SAVE the checkpoint (the real product: field-warped EPE + TimeNet ||W|| are computed post-hoc from it)
    CK=$(ls -t "$RUN"/checkpoint*.pt 2>/dev/null | head -1); [ -n "$CK" ] && cp "$CK" "$LW/checkpoint.pt"
    cp "$RUN"/est_c2w_data.txt "$RUN"/output.txt "$LW/" 2>/dev/null || true
    mkdir -p "$LW/frames_sample"; for f in $(ls "$OUT"/[0-9]*.jpg 2>/dev/null | sort | awk 'NR%30==1'); do cp "$f" "$LW/frames_sample/" 2>/dev/null; done
    echo "=== $CELL DONE $(date -Iseconds) ==="
  } > "$LW/run.log" 2>&1
  cp "$LW/run.log" "$DST/run.log" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  NPOSE=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' "$RUN/est_c2w_data.txt" 2>/dev/null); NPOSE=${NPOSE:-0}
  if [ "${PYRC:-1}" -eq 0 ] && [ "${NPOSE:-0}" -ge 1 ]; then
    sync; touch "$DST/.DONE"; rm -f "$DST/.FAILED"; say "  $CELL shipped OK (${NPOSE} poses)"
  else
    rm -f "$DST/.DONE"; echo "rc=${PYRC:-?} npose=${NPOSE:-0} $(date -Iseconds)" > "$DST/.FAILED"
    say "  $CELL FAILED (rc=${PYRC:-?}, ${NPOSE:-0} poses) -> RE-RUN next launch. see $LW/run.log"
  fi
}

# ---- fan out field_off / field_on / field_on_wd0 x seeds (CELLS= to subset) ----
CELLS="${CELLS:-field_off field_on field_on_wd0}"
JOBS=()
for CFG in trail3_field_off trail3_field_on trail3_field_on_wd0; do
  key=${CFG#trail3_}
  case " $CELLS " in *" $key "*) for s in $SEEDS; do JOBS+=("$CFG|$s"); done;; esac
done
say "########## ARM-2 field diag: ${#JOBS[@]} jobs (cells: $CELLS x ${SEEDS// /,} seeds), $PARALLEL parallel ##########"
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r c s <<< "$spec"
  run_one "$c" "$s" &
  running=$((running+1))
  if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running-1)); fi
  sleep 3
done
wait
say "########## all ${#JOBS[@]} ARM-2 jobs finished ##########"

# ---- SUMMARY: field-BLIND baselines (render + rigid pin rep-err) + TimeNet ||W|| from ckpt ----
say "########## ARM-2 SUMMARY (SemSup; n=3 mean+/-std). REMINDER: these are field-BLIND. ##########"
python3 - "$LWORK" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, glob, re, numpy as np
LW=sys.argv[1]
def grab(s,k):
    m=re.search(k+r'[^0-9-]*([0-9][0-9.]*)', s); return float(m.group(1)) if m else None
cells=[('trail3_field_off','field_off (dynamic:False)'),
       ('trail3_field_on','field_on  (wd 1e-6)'),
       ('trail3_field_on_wd0','field_on_wd0 (wd 0)')]
print("\n--- render PSNR/SSIM/LPIPS (field-BLIND: render absorbs deformation via map+pose) ---")
for cfg,label in cells:
    P=[]
    for d in sorted(glob.glob(f'{LW}/{cfg}_s*')):
        t=f'{d}/render_metrics.txt'
        if not os.path.isfile(t): continue
        s=open(t).read(); ps,ss,lp=grab(s,'PSNR'),grab(s,'SSIM'),grab(s,'LPIPS')
        if ps: P.append((ps,ss or 0,lp or 0))
    if not P: print(f"  {label:<26} (no metrics)"); continue
    A=np.array(P); print(f"  {label:<26} n={len(P)}  PSNR={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  SSIM={A[:,1].mean():.3f}  LPIPS={A[:,2].mean():.3f}")
print("\n--- rigid green-pin reprojection error px (field-BLIND: pins warped by CAMERA only) ---")
for cfg,label in cells:
    E=[]
    for d in sorted(glob.glob(f'{LW}/{cfg}_s*')):
        t=f'{d}/rep_err_rigid.txt'
        if not os.path.isfile(t): continue
        m=re.search(r'per-frame mean \(mean of means\):\s*([0-9.]+)', open(t).read())
        if m: E.append(float(m.group(1)))
    if not E: print(f"  {label:<26} (no rep-err — pin GT not staged?)"); continue
    E=np.array(E); print(f"  {label:<26} n={len(E)}  rep_err={E.mean():6.2f}+/-{E.std():.2f} px")
print("\n--- TimeNet output-layer ||W|| from saved checkpoint (the wd-collapse tell) ---")
import torch
for cfg,label in cells:
    Ws=[]
    for d in sorted(glob.glob(f'{LW}/{cfg}_s*')):
        ck=f'{d}/checkpoint.pt'
        if not os.path.isfile(ck): continue
        try:
            sd=torch.load(ck, map_location='cpu', weights_only=False)
            sd=sd.get('model', sd) if isinstance(sd, dict) else sd
            w=[float(v.norm()) for k,v in sd.items() if 'time' in k.lower() and k.lower().endswith('weight')]
            if w: Ws.append(sum(w)/len(w))
        except Exception as e:
            print(f"    ({label} ckpt read failed: {e})")
    if Ws: print(f"  {label:<26} n={len(Ws)}  mean TimeNet-ish ||W||={np.mean(Ws):.4f}")
    else:  print(f"  {label:<26} (no TimeNet weights matched — inspect key names in ckpt)")
print("""
READ (field-BLIND baselines only — do NOT conclude the field is alive/dead from these):
  * field_on ~= field_off on render+rigid-rep-err  -> CONSISTENT with an inert field (expected even
    if the field works, because both metrics are blind). NOT proof. The field-warped pin EPE decides.
  * field_on_wd0 >> field_on on TimeNet ||W||       -> wd-collapse is real -> cheap fix candidate.
    field_on_wd0 ~= field_on                        -> wd exonerated -> starvation/no-teacher cause.
NEXT (POST-HOC, decisive): build the field-WARPED green-pin EPE (anchor + D(x,t) warp + project,
  warp direction verified on ONE downloaded checkpoint) + a SHUFFLED-time negative control. That is
  the only field-SENSITIVE number here and it is intentionally NOT run unattended.""")
PY
say "=== ARM-2 FIELD DIAGNOSIS DONE $(date -Iseconds) ==="
