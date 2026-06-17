#!/bin/bash
# ============================================================================
# A100 — Inc-1 v2 (DINO) A/B/C with a SMOKE GATE.  2026-06-17.
#
# GOAL: does the per-PIXEL DINO uncertainty head (v2, WildGS-faithful) beat the per-POINT geo
#   head (v1) and the base, on DDS-SLAM's own metrics? base vs geo(v1) vs dino(v2), n=3.
#
# WHY A SMOKE GATE: stage-2 (the DINO data-plumbing, commit e896be8) is a 15-site cross-file change
#   that could NOT be tested locally (no GPU/tcnn). So BEFORE the full n=3 we run a short (~5 min)
#   dino run and GATE on it: constructs + trains + advances keyframes + no traceback/NaN. If it
#   fails we ABORT -- base/geo are never touched (the morning-of human-watched verify).
#
# Two-env split (like MoGe-2 depth): DINOv2 needs torch>=2 (torch.hub) but ddsslam runs in the
#   torch1.10 dds_env. So DINO features are BAKED first with a torch>=2 python (ensure_dino ->
#   data/.../dino/*_dino.npy), then the SLAM env only np.load()s them.
#
# Matrix (CRCD c1_001 + SemSup trail3) x {base, geo(v1), dino(v2)} x 3 seeds.
#   CRCD  -> per-frame Sim3 ATE (scale-corrected; tools/eval_ate.py RIGID is the wrong metric).
#   SemSup-> render PSNR/SSIM/LPIPS (MoGe2 depth; ATE fictional).
# Resume-safe (.DONE per cell). NO repo model edits (TF32+seed via launcher/config).
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
PRISTINE=/content/DDS-SLAM-Base-pristine
PRISTINE_SHA=009977b
SEEDS="0 1 2"
DRIVE=/content/drive/MyDrive/Outputs/a100_dino_ab_${DATE}
LWORK=/content/a100work_dino; mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }
say "=== A100 DINO A/B/C start $(date -Iseconds)  HEAD=$(cd $REPO && git rev-parse --short HEAD) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
nvidia-smi -L || true

# --- detect a torch>=2 python for the DINO extractor BEFORE any venv activation (Colab system
#     python is 3.x/torch2). Saved now because activate_dds_env will switch `python` to torch1.10.
DINO_PY=""
for p in python3 /usr/bin/python3 python; do
  if "$p" -c "import torch,sys; sys.exit(0 if int(torch.__version__.split('.')[0])>=2 else 1)" 2>/dev/null; then DINO_PY="$p"; break; fi
done
[ -n "$DINO_PY" ] && say "DINO extractor python = $DINO_PY ($($DINO_PY -c 'import torch;print(torch.__version__)'))" \
                   || say "WARN: no torch>=2 python found -> ensure_dino will try 'python3' and may fail"
DINO_PY=${DINO_PY:-python3}

# ---------------------------------------------------------------------------
activate_dds_env(){
  cd "$REPO"
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "env rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; fi
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
  python -c "import torch; assert torch.cuda.is_available()" || { say "env FAIL (no CUDA)"; exit 1; }
  CC=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo 75)
  say "GPU compute capability = sm_$CC"
  if ! python - <<'PY' 2>/dev/null
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,
  "log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda')); print("tcnn ok on device")
PY
  then
    say "  tinycudann probe FAILED on sm_$CC -> rebuild for this arch (~10 min)"
    TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps \
      "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" || { say "TCNN rebuild FAIL"; exit 1; }
  fi
  python -c "import lpips" 2>/dev/null || pip install -q lpips || true
}

# ---------------------------------------------------------------------------
# GATE 1: base-parity (off-path unchanged by stage-2). fork@all-flags-off == pristine model build.
# stage-2's new branches are all gated (mode=='dino' / 'dino' in batch) so the OFF build draws zero
# new RNG -> this MUST still pass; if it doesn't, stage-2 perturbed the base path.
# ---------------------------------------------------------------------------
parity_gate(){
  local G="$DRIVE/parity"; mkdir -p "$G"; done_marker "$G" && { say "parity done"; return 0; }
  say "########## GATE 1: base-parity vs pristine IRMVLab $PRISTINE_SHA (stage-2 off-path) ##########"
  [ -d "$PRISTINE/.git" ] || git clone -q https://github.com/IRMVLab/DDS-SLAM "$PRISTINE" || { say "  clone FAIL"; touch "$G/.DONE"; return 0; }
  ( cd "$PRISTINE" && git checkout -q "$PRISTINE_SHA" 2>/dev/null )
  mkdir -p "$PRISTINE/Addons/regression"
  cp "$REPO/Addons/regression/test_inc0_bitidentical.py" "$PRISTINE/Addons/regression/"
  cp "$REPO/configs/Super/trail3_paper_faithful.yaml" "$PRISTINE/configs/Super/_parity_shared.yaml"
  ( cd "$PRISTINE" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/_parity_shared.yaml --golden /content/golden_pristine.json --write-golden ) 2>&1 | tee "$G/pristine_build.txt"
  ( cd "$REPO" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/trail3_paper_faithful.yaml --golden /content/golden_pristine.json ) 2>&1 | tee "$G/fork_check.txt"
  if grep -qi "PASS" "$G/fork_check.txt"; then say "  >>> PARITY PASS: stage-2 left the base path bit-identical"
  else say "  >>> PARITY MISMATCH (see $G/fork_check.txt) — stage-2 perturbed base; STOP and reconcile"; fi
  sync; touch "$G/.DONE"
}

# ---------------------------------------------------------------------------
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ ! -d "$REPO/data/Super/trail_3/rgb" ]; then
    [ -d "$SRC/rgb" ] || { say "  WARN SemSup source missing -> SemSup skips"; return 0; }
    mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"; fi
  local DST="$REPO/data/Super/trail_3/depth/moge2"
  [ -d "$DST" ] && [ "$(ls "$DST"/*left_depth.npy 2>/dev/null | wc -l)" -ge 151 ] && { say "  moge2 depth staged"; return 0; }
  local MS=""
  for c in /content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3/depth/MoGe2_trail3_20260608 \
           /content/drive/MyDrive/Datasets/SemSup/MoGe2_trail3_20260608 \
           /content/drive/MyDrive/MoGe2_trail3_20260608 ; do [ -d "$c" ] && MS="$c" && break; done
  [ -n "$MS" ] && { mkdir -p "$DST"; cp "$MS"/*left_depth.npy "$DST"/ 2>/dev/null; say "  staged moge2 depth: $(ls "$DST"/*left_depth.npy 2>/dev/null|wc -l) npy"; } \
               || say "  WARN MoGe2 depth not found -> SemSup A/B SKIPPED"
}
crcd_stage(){ local STAGED="$REPO/data/CRCD/C1_001"
  [ -d "$STAGED/video_frames" ] && return 0
  local SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
  local CALIB=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
  local MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth
  [ -d "$SNIP" ] && [ -f "$CALIB" ] && [ -d "$MOGE" ] || { say "  CRCD prereqs MISSING -> CRCD skips"; return 1; }
  $DINO_PY Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SNIP" --calib_pkl "$CALIB" --output_dir "${STAGED}.tmp" 2>&1 | tee -a "$LOG" && mv "${STAGED}.tmp" "$STAGED" || return 1
  mkdir -p "$STAGED/depth.tmp"; $DINO_PY - "$MOGE" "$STAGED/depth.tmp" <<'PY'
import os,sys,shutil
src,dst=sys.argv[1],sys.argv[2]
for i,f in enumerate(sorted(x for x in os.listdir(src) if x.endswith('.png'))): shutil.copy2(os.path.join(src,f),os.path.join(dst,f'{i:06d}.png'))
print('copied',len(os.listdir(dst)),'MoGe depth')
PY
  rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"; }

# ---------------------------------------------------------------------------
# ensure_dino DATADIR RGBSUB GLOB N  -> bake DINOv2 patch grids to $DATADIR/dino/*_dino.npy.
# Uses the torch>=2 python (DINOv2 via torch.hub). Idempotent: skips if >=N grids already present.
# ---------------------------------------------------------------------------
ensure_dino(){ local DD=$1 RGBSUB=$2 GLOB=$3 N=$4
  local OUT="$DD/dino"
  local have=$(ls "$OUT"/*_dino.npy 2>/dev/null | wc -l)
  [ "$have" -ge "$N" ] && { say "  DINO features present ($have) -> $OUT"; return 0; }
  [ -d "$DD/$RGBSUB" ] || { say "  WARN $DD/$RGBSUB missing -> cannot bake DINO (cell will skip)"; return 1; }
  say "  baking DINOv2 (dinov2_vits14, C=384) from $DD/$RGBSUB (have $have/$N)"
  $DINO_PY "$REPO/Addons/dino/generate_dino_features.py" \
    --rgb_dir "$DD/$RGBSUB" --rgb_glob "$GLOB" --out_dir "$OUT" --backbone dinov2_vits14 2>&1 | tail -8
  local now=$(ls "$OUT"/*_dino.npy 2>/dev/null | wc -l)
  [ "$now" -ge "$N" ] && { say "  DINO baked: $now grids"; return 0; } || { say "  WARN DINO bake incomplete ($now/$N)"; return 1; }
}

# ---------------------------------------------------------------------------
# GATE 2: SMOKE the dino path (the untestable-locally half). Short timeout; PASS iff it constructs,
# trains, advances >=2 keyframes, with NO traceback and NO NaN loss. FAIL -> ABORT (base/geo safe).
# ---------------------------------------------------------------------------
smoke_dino(){
  [ -d "$REPO/data/CRCD/C1_001/dino" ] || { say "  SMOKE skip: no CRCD dino features"; return 1; }
  say "########## GATE 2: SMOKE dino path (~5 min, CRCD c1_001 dino seed 0) ##########"
  local SM=output/_a100/_dino_smoke OVR=/content/_dino_smoke.yaml SLOG=/content/_dino_smoke.log
  cat > "$OVR" <<YML
inherit_from: configs/CRCD/c1_001_canon_uncert_dino.yaml
seed: 0
data:
  output: ${SM}
  exp_name: demo
YML
  timeout 360 python -W ignore - "$OVR" > "$SLOG" 2>&1 <<'PY'
import sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
  # grep -c always prints a count (0 if none); NO '|| echo 0' (that double-prints on no-match).
  # NaN gate uses -w 'nan' (whole word) so it does NOT match "info"/"finance" etc.
  local KF=$(grep -c "add keyframe" "$SLOG" 2>/dev/null)
  local TB=$(grep -c "Traceback" "$SLOG" 2>/dev/null)
  local NAN=$(grep -ciw "nan" "$SLOG" 2>/dev/null)
  KF=${KF:-0}; TB=${TB:-0}; NAN=${NAN:-0}
  say "  smoke signals: keyframes=$KF  tracebacks=$TB  nan/inf_lines=$NAN"
  if [ "$TB" -eq 0 ] && [ "$KF" -ge 2 ] && [ "$NAN" -eq 0 ]; then
    say "  >>> SMOKE PASS: dino head constructs + trains + advances cleanly. Proceeding to full n=3."
    rm -rf "$SM"; return 0
  fi
  say "  >>> SMOKE FAIL -> ABORT. base/geo untouched. Last 40 lines of $SLOG:"; tail -40 "$SLOG"
  return 1
}

# ---------------------------------------------------------------------------
make_video(){ local RUN=$1 DDIR=$2 VT=$3 OUT=$4
  local RGBIN RGBP DI SEG SEGP GT SEGEXTRA=""
  if [ "$VT" = "crcd" ]; then RGBIN="$DDIR/video_frames"; RGBP='*l.png'; DI="$DDIR/depth"; SEG="$DDIR/semantic_class"; SEGP='*.png'; GT="$DDIR/groundtruth.txt"; SEGEXTRA="--seg_classmap"
  else RGBIN="$DDIR/rgb"; RGBP='*left.png'; DI="$DDIR/depth/moge2"; SEG="$DDIR/seg/png_masks"; SEGP='*left.png'; GT="$DDIR/groundtruth.txt"; fi
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$RGBIN" --rgb_input_pattern "$RGBP" \
    --rgb_output_dir "$RUN" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DI" --depth_output_dir "$RUN/depth" --depth_norm robust \
    --seg_dir "$SEG" --seg_pattern "$SEGP" --skip_raw_seg $SEGEXTRA \
    --uncert_dir "$RUN/uncert" \
    --trajectory_est "$RUN/demo/est_c2w_data.txt" --trajectory_gt "$GT" --trajectory_raw \
    --output "$OUT" --fps 15 2>&1 || echo "WARN video failed ($OUT)"
}

# ---------------------------------------------------------------------------
run_one(){ local NAME=$1 GRP=$2 DDIR=$3 VT=$4 S=$5
  local CELL="${NAME}_s${S}" DST="$DRIVE/${NAME}_s${S}" LW="$LWORK/${NAME}_s${S}"
  done_marker "$DST" && { say "  $CELL already done -> skip"; return 0; }
  mkdir -p "$LW" "$DST"
  local OUT="output/_a100/${CELL}" OVR="/content/_cfg_${CELL}.yaml" RUN="output/_a100/${CELL}/demo" PYRC=1 NPOSE=0
  cat > "$OVR" <<YML
inherit_from: configs/${GRP}/${NAME}.yaml
seed: ${S}
data:
  output: ${OUT}
  exp_name: demo
YML
  {
    echo "=== $CELL START $(date -Iseconds) (seed $S, $GRP/$NAME) ==="
    python -W ignore - "$OVR" <<'PY'
import os, sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
    PYRC=$?; [ "$PYRC" -ne 0 ] && echo "!!! TRAIN CRASHED rc=$PYRC -- NOT marking .DONE"
    make_video "$OUT" "$DDIR" "$VT" "$LW/${CELL}_6panel.mp4"
    [ "$VT" = "super" ] && { python Addons/eval/eval_rendering.py --gt_dir "$DDIR/rgb" --render_dir "$OUT" --name "$CELL" --sequence "Lab1 (trail3)" > "$LW/render_metrics.txt" 2>&1 || echo "WARN render-eval"; }
    [ "$VT" != "super" ] && { python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DDIR/groundtruth.txt" --name "$CELL" --out "$LW/sim3_metrics.txt" || echo "WARN sim3-ate"; \
        python Addons/eval/eval_rendering.py --gt_dir "$DDIR/video_frames" --render_dir "$OUT" --name "$CELL" --sequence "CRCD (C1_001)" > "$LW/render_metrics.txt" 2>&1 || echo "WARN crcd render-eval"; }
    CK=$(ls -t "$RUN"/checkpoint*.pt 2>/dev/null | head -1); [ -n "$CK" ] && cp "$CK" "$LW/checkpoint.pt"
    cp "$RUN"/est_c2w_data.txt "$RUN"/output.txt "$LW/" 2>/dev/null || true
    mkdir -p "$LW/frames_sample" "$LW/uncert_sample"
    for f in $(ls "$OUT"/[0-9]*.jpg 2>/dev/null | sort | awk 'NR%30==1'); do cp "$f" "$LW/frames_sample/" 2>/dev/null; done
    for f in $(ls "$OUT"/uncert/[0-9]*.png 2>/dev/null | sort | awk 'NR%30==1'); do cp "$f" "$LW/uncert_sample/" 2>/dev/null; done
    echo "=== $CELL DONE $(date -Iseconds) ==="
  } > "$LW/run.log" 2>&1
  cp "$LW/run.log" "$DST/run.log" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  NPOSE=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' "$RUN/est_c2w_data.txt" 2>/dev/null); NPOSE=${NPOSE:-0}
  if [ "${PYRC:-1}" -eq 0 ] && [ "${NPOSE:-0}" -ge 1 ]; then
    sync; touch "$DST/.DONE"; rm -f "$DST/.FAILED"; say "  $CELL shipped OK (${NPOSE} poses)"
  else
    rm -f "$DST/.DONE"; echo "rc=${PYRC:-?} npose=${NPOSE:-0}" > "$DST/.FAILED"
    say "  $CELL FAILED (rc=${PYRC:-?}, ${NPOSE:-0} poses) -> re-run next launch. see $LW/run.log"
  fi
}

# ---------------------------------------------------------------------------
cd "$REPO"
stage_semsup; crcd_stage || true
# bake DINO features (torch>=2) BEFORE switching to the torch1.10 SLAM env
HAVE_CRCD_DINO=0; HAVE_SEMSUP_DINO=0
[ -d "$REPO/data/CRCD/C1_001/video_frames" ] && ensure_dino "$REPO/data/CRCD/C1_001" video_frames '*l.png' 360 && HAVE_CRCD_DINO=1
[ -d "$REPO/data/Super/trail_3/rgb" ]        && ensure_dino "$REPO/data/Super/trail_3" rgb '*left.png' 151 && HAVE_SEMSUP_DINO=1

activate_dds_env
parity_gate
smoke_dino || { say "########## ABORTED at SMOKE GATE. Fix the dino path, re-launch. ##########"; exit 1; }

# CPU thread cap per parallel job (avoid oversubscription)
PARALLEL=${PARALLEL:-4}
NPROC=$(nproc 2>/dev/null || echo 8); THREADS=${THREADS:-$(( NPROC/PARALLEL > 0 ? NPROC/PARALLEL : 1 ))}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
say "  CPU threads/job=$THREADS (nproc=$NPROC/PARALLEL=$PARALLEL)"

JOBS=()
add_cell(){ for s in $SEEDS; do JOBS+=("$1|$2|$3|$4|$s"); done; }
# base + geo(v1) + dino(v2): geo re-run here = a regression check (should reproduce ~2.43mm) AND the
# clean same-env A/B/C. CELLS env can subset, e.g. CELLS="dino" to only run the new arm.
CELLS="${CELLS:-base geo dino}"
if [ -d "$REPO/data/CRCD/C1_001/video_frames" ]; then
  case " $CELLS " in *" base "*) add_cell c1_001_canon_base CRCD data/CRCD/C1_001 crcd;; esac
  case " $CELLS " in *" geo "*)  add_cell c1_001_canon_uncert CRCD data/CRCD/C1_001 crcd;; esac
  [ "$HAVE_CRCD_DINO" = 1 ] && case " $CELLS " in *" dino "*) add_cell c1_001_canon_uncert_dino CRCD data/CRCD/C1_001 crcd;; esac
fi
if [ -d "$REPO/data/Super/trail_3/depth/moge2" ]; then
  case " $CELLS " in *" base "*) add_cell trail3_moge2_uncert_base Super data/Super/trail_3 super;; esac
  case " $CELLS " in *" geo "*)  add_cell trail3_moge2_uncert Super data/Super/trail_3 super;; esac
  [ "$HAVE_SEMSUP_DINO" = 1 ] && case " $CELLS " in *" dino "*) add_cell trail3_moge2_uncert_dino Super data/Super/trail_3 super;; esac
fi
say "########## RUN ${#JOBS[@]} jobs (cells: $CELLS x ${SEEDS// /,} seeds), $PARALLEL parallel ##########"
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r n g d v s <<< "$spec"
  run_one "$n" "$g" "$d" "$v" "$s" &
  running=$((running+1))
  [ "$running" -ge "$PARALLEL" ] && { wait -n; running=$((running-1)); }
  sleep 3
done
wait
say "########## all ${#JOBS[@]} jobs finished ##########"

# ---------------------------------------------------------------------------
say "########## SUMMARY (n=3 mean +/- std): base vs geo(v1) vs dino(v2) ##########"
python3 - "$LWORK" "$REPO/data/CRCD/C1_001/groundtruth.txt" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, glob, re, numpy as np
LW, GT = sys.argv[1], sys.argv[2]
def est(p):
    P=[];
    if not os.path.isfile(p): return np.zeros((0,3))
    for l in open(p):
        v=l.split()
        if len(v)>=12 and not v[0].startswith('#'): P.append(np.array(list(map(float,v[:12]))).reshape(3,4)[:3,3])
    return np.array(P)
def tum(p):
    P=[]
    if not os.path.isfile(p): return np.zeros((0,3))
    for l in open(p):
        v=l.split()
        if len(v)>=8 and not v[0].startswith('#'): P.append([float(v[1]),float(v[2]),float(v[3])])
    return np.array(P)
def sim3(m,d):
    mc,dc=m.mean(0),d.mean(0); mm,dd=m-mc,d-dc; H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    s=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,s])@U.T
    sc=(S*np.array([1,1,s])).sum()/(mm*mm).sum(); al=(sc*(R@m.T)).T+(dc-sc*R@mc); return al,sc
g=tum(GT)
print("\n--- CRCD c1_001 tracking: Sim3 ATE mm (mean/max) + |Pearson|dom (scale-free) ---")
for cond in ['c1_001_canon_base','c1_001_canon_uncert','c1_001_canon_uncert_dino']:
    rows=[]
    for d in sorted(glob.glob(f'{LW}/{cond}_s*')):
        e=est(f'{d}/est_c2w_data.txt')
        if len(e)<10 or len(g)<10: continue
        n=min(len(e),len(g)); a,sc=sim3(e[:n],g[:n]); pf=np.linalg.norm(a-g[:n],axis=1)*1000
        dom=int(np.argmax(g[:n].max(0)-g[:n].min(0))); pe=abs(np.corrcoef(e[:n][:,dom],g[:n][:,dom])[0,1])
        rows.append((pf.mean(),pf.max(),pe))
    tag={'c1_001_canon_base':'base','c1_001_canon_uncert':'geo(v1)','c1_001_canon_uncert_dino':'dino(v2)'}[cond]
    if not rows: print(f"  {tag:<9} (no seeds)"); continue
    A=np.array(rows)
    print(f"  {tag:<9} n={len(rows)}  ATE_mean={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  "
          f"ATE_max={A[:,1].mean():6.2f}+/-{A[:,1].std():.2f}  |Pear|={A[:,2].mean():.3f}+/-{A[:,2].std():.3f}")
print("\n--- CRCD c1_001 render PSNR/SSIM/LPIPS (frames shown -> spot a mispair) ---")
for cond in ['c1_001_canon_base','c1_001_canon_uncert','c1_001_canon_uncert_dino']:
    P=[]
    for d in sorted(glob.glob(f'{LW}/{cond}_s*')):
        t=f'{d}/render_metrics.txt'
        if not os.path.isfile(t): continue
        s=open(t).read()
        def grab(k):
            m=re.search(k+r'[^0-9-]*([0-9.]+)', s); return float(m.group(1)) if m else None
        nf,ps,ss,lp=grab('Rendered'),grab('PSNR'),grab('SSIM'),grab('LPIPS')
        if ps: P.append((ps,ss or 0,lp or 0,nf or 0))
    tag={'c1_001_canon_base':'base','c1_001_canon_uncert':'geo(v1)','c1_001_canon_uncert_dino':'dino(v2)'}[cond]
    if not P: print(f"  {tag:<9} (no render metrics)"); continue
    A=np.array(P)
    print(f"  {tag:<9} n={len(P)}  PSNR={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  SSIM={A[:,1].mean():.3f}  LPIPS={A[:,2].mean():.3f}  (frames~{int(A[:,3].mean())})")
print("\n--- SemSup trail3_moge2 render PSNR/SSIM/LPIPS ---")
for cond in ['trail3_moge2_uncert_base','trail3_moge2_uncert','trail3_moge2_uncert_dino']:
    P=[]
    for d in sorted(glob.glob(f'{LW}/{cond}_s*')):
        t=f'{d}/render_metrics.txt'
        if not os.path.isfile(t): continue
        s=open(t).read()
        def grab(k):
            m=re.search(k+r'[^0-9-]*([0-9.]+)', s); return float(m.group(1)) if m else None
        ps,ss,lp=grab('PSNR'),grab('SSIM'),grab('LPIPS')
        if ps: P.append((ps,ss or 0,lp or 0))
    tag={'trail3_moge2_uncert_base':'base','trail3_moge2_uncert':'geo(v1)','trail3_moge2_uncert_dino':'dino(v2)'}[cond]
    if not P: print(f"  {tag:<9} (no render metrics)"); continue
    A=np.array(P)
    print(f"  {tag:<9} n={len(P)}  PSNR={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  SSIM={A[:,1].mean():.3f}  LPIPS={A[:,2].mean():.3f}")
print("\nVideos: each cell payload has <cell>_6panel.mp4 (dino cells include the sigma^2 panel).")
PY
say "=== A100 DINO A/B/C DONE $(date -Iseconds) ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
