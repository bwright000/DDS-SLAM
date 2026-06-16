#!/bin/bash
# ============================================================================
# A100 — IMPROVE branch canonical A/B: Inc-1/2 uncertainty vs PRISTINE-LOGIC base.  2026-06-16.
#
# GOAL (user): see the improvement that JUST our Inc-1/2 uncertainty makes, measured against
#   DDS-SLAM's OWN metrics, on a base that is the authors' model EXACTLY.
#
# Locked decisions:
#   * Base = crcd.yaml EXACTLY + authors' surgical lr 1e-4 (configs/CRCD/crcd_basecanon.yaml).
#     SemSup base = authors' Super.yaml (trail3_moge2) with OUR MoGe2 depth.
#   * Model LOGIC stays pristine -> a GATE proves fork@all-flags-off == IRMVLab 009977b model build.
#   * METRIC = PSNR/SSIM/LPIPS (render) + per-frame Sim3 ATE (CRCD tracking). n=3 SEEDS (noise floor).
#   * A100-safe: tinycudann arch sm_80 (cache is sm_75=T4); TF32 OFF (precision-sensitive SDF/pose).
#   * Each run: 6-panel INLINE video (median-scaled output depth) + the inline frames ARE packaged.
#
# Matrix (4 cells x 3 seeds = 12 runs):
#   CRCD  : c1_001_canon_base   vs c1_001_canon_uncert     -> per-frame Sim3 ATE (spike-at-deformation)
#   SemSup: trail3_moge2_uncert_base vs trail3_moge2_uncert -> render PSNR/SSIM/LPIPS (MoGe2 depth)
# Resume-safe (.DONE per cell). set -uo. NO repo model files edited (TF32+seed via launcher/config).
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
PRISTINE=/content/DDS-SLAM-Base-pristine          # fresh clone of IRMVLab/DDS-SLAM @ 009977b
PRISTINE_SHA=009977b
SEEDS="0 1 2"
DRIVE=/content/drive/MyDrive/Outputs/a100_improve_ab_${DATE}
LWORK=/content/a100work; mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }
say "=== A100 IMPROVE A/B start $(date -Iseconds)  HEAD=$(cd $REPO && git rev-parse --short HEAD) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
nvidia-smi -L || true

# ---------------------------------------------------------------------------
# ENV (A100-aware): restore dds env, then ensure tinycudann matches the LIVE GPU arch.
# The cached env was built TCNN_CUDA_ARCHITECTURES=75 (T4). A100=sm_80 -> rebuild tcnn if it
# can't run a probe on the device.
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
    say "  tinycudann probe FAILED on sm_$CC -> force rebuild for this arch (~10 min)"
    TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps \
      "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" \
      || { say "TCNN rebuild FAIL"; exit 1; }
    python - <<'PY' || { say "TCNN still broken"; exit 1; }
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,
  "log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda')); print("tcnn ok after rebuild")
PY
  fi
  python -c "import lpips" 2>/dev/null || pip install -q lpips || true
}

# train(): TF32 OFF (Ampere defaults it ON; the sdf_weight=1000 SDF + pose optimisation is
# precision-sensitive and we want numerics close to the paper RTX3090 / our prior T4). Seed is
# config-driven (ddsslam.py:43 reads config['seed']). runpy keeps __file__ correct. NO repo edit.
train(){ local CFG=$1; say "  train $CFG"; local T0=$(date +%s)
  python -W ignore - "$CFG" <<'PY' 2>&1 | tee -a "$LOG" || say "  WARN train nonzero exit"
import sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
  say "  train $(( ($(date +%s)-T0)/60 )) min"; }

# ---------------------------------------------------------------------------
# GATE: base-parity. Prove fork@all-flags-off == pristine IRMVLab model construction.
# Uses test_inc0_bitidentical (RNG-state-after-build + param count + state_dict keys; GPU-safe -
# immune to tinycudann atomic-add nondeterminism). Same config file fed to BOTH repos -> any diff
# is CODE, not config. Scope: model CONSTRUCTION parity (all behaviour-changes are .get(...,False)
# gated default-off, so default runtime == pristine too).
# ---------------------------------------------------------------------------
parity_gate(){
  local G="$DRIVE/parity"; mkdir -p "$G"; done_marker "$G" && { say "parity done"; return 0; }
  say "########## GATE: base-parity vs pristine IRMVLab $PRISTINE_SHA ##########"
  [ -d "$PRISTINE/.git" ] || git clone -q https://github.com/IRMVLab/DDS-SLAM "$PRISTINE" || { say "  clone FAIL"; touch "$G/.DONE"; return 0; }
  ( cd "$PRISTINE" && git checkout -q "$PRISTINE_SHA" 2>/dev/null )
  local SHARED="$REPO/configs/Super/trail3_paper_faithful.yaml"   # has mapping.bound; copied to both
  mkdir -p "$PRISTINE/Addons/regression"
  cp "$REPO/Addons/regression/test_inc0_bitidentical.py" "$PRISTINE/Addons/regression/"
  cp "$SHARED" "$PRISTINE/configs/Super/_parity_shared.yaml"
  say "  golden <- PRISTINE model build"
  ( cd "$PRISTINE" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/_parity_shared.yaml --golden /content/golden_pristine.json --write-golden ) 2>&1 | tee "$G/pristine_build.txt"
  say "  check FORK @ all-flags-off vs pristine golden"
  ( cd "$REPO" && python Addons/regression/test_inc0_bitidentical.py \
      --config configs/Super/trail3_paper_faithful.yaml --golden /content/golden_pristine.json ) 2>&1 | tee "$G/fork_check.txt"
  cp /content/golden_pristine.json "$G/" 2>/dev/null
  if grep -qi "PASS" "$G/fork_check.txt"; then say "  >>> PARITY PASS: fork@default == pristine IRMVLab model build"
  else say "  >>> PARITY MISMATCH (see $G/fork_check.txt) — RECONCILE before trusting base numbers"; fi
  sync; touch "$G/.DONE"
}

# ---------------------------------------------------------------------------
# STAGE DATA
# ---------------------------------------------------------------------------
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ ! -d "$REPO/data/Super/trail_3/rgb" ]; then
    [ -d "$SRC/rgb" ] || { say "  WARN SemSup source missing ($SRC) -> SemSup will skip"; return 0; }
    mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"; fi
  # MoGe2 depth = the previously-missing subdir. Files are NNNNNN-left_depth.npy (loader globs
  # *left_depth.npy, png_depth_scale=8, scale-consistent with variant_a_stereo — verified 2026-06-16).
  local DST="$REPO/data/Super/trail_3/depth/moge2"
  if [ -d "$DST" ] && [ "$(ls "$DST"/*left_depth.npy 2>/dev/null | wc -l)" -ge 151 ]; then
    say "  moge2 depth already staged ($(ls "$DST"/*left_depth.npy | wc -l) npy)"; return 0; fi
  local MS=""
  for c in \
    /content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3/depth/MoGe2_trail3_20260608 \
    /content/drive/MyDrive/Datasets/SemSup/MoGe2_trail3_20260608 \
    /content/drive/MyDrive/MoGe2_trail3_20260608 \
    /content/drive/MyDrive/Outputs/MoGe2_trail3_20260608 ; do
    [ -d "$c" ] && MS="$c" && break; done
  if [ -n "$MS" ]; then mkdir -p "$DST"; cp "$MS"/*left_depth.npy "$DST"/ 2>/dev/null || cp "$MS"/* "$DST"/
    say "  staged moge2 depth: $(ls "$DST"/*left_depth.npy 2>/dev/null | wc -l) npy from $MS"
  else say "  WARN: MoGe2_trail3_20260608 not found on Drive (searched standard paths) -> SemSup A/B SKIPPED. Set the correct Drive path."; fi
}
crcd_stage(){ local STAGED="$REPO/data/CRCD/C1_001"
  [ -d "$STAGED/video_frames" ] && return 0
  local SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
  local CALIB=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
  local MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth
  [ -d "$SNIP" ] && [ -f "$CALIB" ] && [ -d "$MOGE" ] || { say "  CRCD c1 prereqs MISSING -> CRCD A/B SKIPPED"; return 1; }
  python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SNIP" --calib_pkl "$CALIB" --output_dir "${STAGED}.tmp" 2>&1 | tee -a "$LOG" && mv "${STAGED}.tmp" "$STAGED" || return 1
  mkdir -p "$STAGED/depth.tmp"; python3 - "$MOGE" "$STAGED/depth.tmp" <<'PY'
import os,sys,shutil
src,dst=sys.argv[1],sys.argv[2]
for i,f in enumerate(sorted(x for x in os.listdir(src) if x.endswith('.png'))): shutil.copy2(os.path.join(src,f),os.path.join(dst,f'{i:06d}.png'))
print('copied',len(os.listdir(dst)),'MoGe depth')
PY
  rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"; }

# ---------------------------------------------------------------------------
# 6-panel INLINE video (median-scaled output depth). Auto-discovers panels.
# ---------------------------------------------------------------------------
make_video(){ local RUN=$1 DDIR=$2 VT=$3 OUT=$4
  local RGBIN RGBP DI SEG SEGP GT SEGEXTRA=""
  if [ "$VT" = "crcd" ]; then RGBIN="$DDIR/video_frames"; RGBP='*l.png'; DI="$DDIR/depth"; SEG="$DDIR/semantic_class"; SEGP='*.png'; GT="$DDIR/groundtruth.txt"; SEGEXTRA="--seg_classmap"
  else RGBIN="$DDIR/rgb"; RGBP='*left.png'; DI="$DDIR/depth/moge2"; SEG="$DDIR/seg/png_masks"; SEGP='*left.png'; GT="$DDIR/groundtruth.txt"; fi
  # --uncert_dir is harmless on base cells (no uncert/ -> panel omitted); on +uncertainty cells
  # it adds the model's volume-rendered sigma^2 panel (inferno, robust = the uncertainty render).
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$RGBIN" --rgb_input_pattern "$RGBP" \
    --rgb_output_dir "$RUN" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DI" \
    --depth_output_dir "$RUN/depth" --depth_norm robust \
    --seg_dir "$SEG" --seg_pattern "$SEGP" --skip_raw_seg $SEGEXTRA \
    --uncert_dir "$RUN/uncert" \
    --trajectory_est "$RUN/demo/est_c2w_data.txt" --trajectory_gt "$GT" --trajectory_raw \
    --output "$OUT" --fps 15 2>&1 || echo "WARN video failed ($OUT)"
}

# ---------------------------------------------------------------------------
# run_one NAME GROUP DATADIR VT SEED -> ONE (cell,seed): train + video + eval + package.
# Self-contained + resume-safe (.DONE). ALL heavy output -> its OWN $DST/run.log so parallel
# jobs never interleave (the shared $LOG only gets the one-line ship message). TF32 OFF + seed
# via config (no repo edit). Designed to be backgrounded (run_one ... &) with a concurrency cap.
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
  {   # ---- everything for THIS job into its own log (parallel-safe) ----
    echo "=== $CELL START $(date -Iseconds) (seed $S, $GRP/$NAME) ==="
    python -W ignore - "$OVR" <<'PY'
import os, sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))   # avoid CPU oversubscription across parallel jobs
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]
runpy.run_path('ddsslam.py', run_name='__main__')
PY
    PYRC=$?; [ "$PYRC" -ne 0 ] && echo "!!! TRAIN CRASHED rc=$PYRC -- this cell will NOT be marked .DONE"
    # NOTE: ddsslam writes renders to $OUT/*.jpg, depth to $OUT/depth, uncert to $OUT/uncert (the
    # config output root); est_c2w + checkpoint go to $OUT/demo ($RUN). Eval/video read renders from
    # $OUT (NOT $RUN) -- pointing them at $RUN/demo gave the bogus "3 images" PSNR.
    make_video "$OUT" "$DDIR" "$VT" "$LW/${CELL}_6panel.mp4"
    [ "$VT" = "super" ] && { python Addons/eval/eval_rendering.py --gt_dir "$DDIR/rgb" --render_dir "$OUT" --name "$CELL" --sequence "Lab1 (trail3)" > "$LW/render_metrics.txt" 2>&1 || echo "WARN render-eval"; }
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
    rm -f "$DST/.DONE"; echo "rc=${PYRC:-?} npose=${NPOSE:-0} $(date -Iseconds)" > "$DST/.FAILED"
    say "  $CELL FAILED (rc=${PYRC:-?}, ${NPOSE:-0} poses) -> will RE-RUN next launch. see $LW/run.log"
  fi
}

# ---------------------------------------------------------------------------
cd "$REPO"; activate_dds_env
parity_gate
stage_semsup; crcd_stage || true

# ---- RUN: fan out ALL (cell x seed) jobs, PARALLEL at a time. A single DDS-SLAM run uses
# ~1-3GB of the A100's 40GB and is CPU-bursty (GPU mostly idle), so concurrency fills the gaps.
# env + parity + staging above are sequential/once; ONLY the runs parallelize. Override the cap:
#   PARALLEL=8 bash Addons/colab/a100_improve_ab_20260616.sh   (watch `nvidia-smi`, bump if GPU/CPU spare)
PARALLEL=${PARALLEL:-4}
# Cap CPU threads PER job: PyTorch/OpenMP/MKL each default to ALL cores for intra-op work, so N
# parallel processes oversubscribe ~Nx -> ~Nx slower per iter (the GPU is idle; the CPU thrashes).
# threads ~= cores/PARALLEL keeps total threads ~= cores.
NPROC=$(nproc 2>/dev/null || echo 8)
THREADS=${THREADS:-$(( NPROC/PARALLEL > 0 ? NPROC/PARALLEL : 1 ))}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
say "  CPU threads/job capped at $THREADS (nproc=$NPROC / PARALLEL=$PARALLEL) to avoid oversubscription"
JOBS=()
add_cell(){ for s in $SEEDS; do JOBS+=("$1|$2|$3|$4|$s"); done; }
[ -d "$REPO/data/CRCD/C1_001/video_frames" ]      && { add_cell c1_001_canon_base CRCD data/CRCD/C1_001 crcd; add_cell c1_001_canon_uncert CRCD data/CRCD/C1_001 crcd; }
[ -d "$REPO/data/Super/trail_3/depth/moge2" ]     && { add_cell trail3_moge2_uncert_base Super data/Super/trail_3 super; add_cell trail3_moge2_uncert Super data/Super/trail_3 super; }
say "########## RUN ${#JOBS[@]} jobs (4 cells x ${SEEDS// /,} seeds), ${PARALLEL} in parallel ##########"
running=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r n g d v s <<< "$spec"
  run_one "$n" "$g" "$d" "$v" "$s" &
  running=$((running+1))
  if [ "$running" -ge "$PARALLEL" ]; then wait -n; running=$((running-1)); fi   # free a slot before launching the next
  sleep 3   # stagger CUDA-context creation so launches don't thunder-herd the driver
done
wait
say "########## all ${#JOBS[@]} jobs finished ##########"

# ---------------------------------------------------------------------------
# SUMMARY: n=3 mean+/-std A/B.  CRCD = Sim3 per-frame ATE; SemSup = render metrics.
# ---------------------------------------------------------------------------
say "########## SUMMARY (n=3 mean +/- std) ##########"
python3 - "$LWORK" "$REPO/data/CRCD/C1_001/groundtruth.txt" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, glob, re, numpy as np
LW, GT = sys.argv[1], sys.argv[2]
def est(p):
    P=[]
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
def horn(m,d):
    mc,dc=m.mean(0),d.mean(0); mm,dd=m-mc,d-dc; H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    s=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,s])@U.T
    sc=(S*np.array([1,1,s])).sum()/(mm*mm).sum(); return (sc*(R@m.T)).T+(dc-sc*R@mc)
g=tum(GT)
print("\n--- CRCD c1_001 tracking (Sim3 per-frame ATE mm; watch ATE_max/p90 = deformation spikes) ---")
for cond in ['c1_001_canon_base','c1_001_canon_uncert']:
    rows=[]
    for d in sorted(glob.glob(f'{LW}/{cond}_s*')):
        e=est(f'{d}/est_c2w_data.txt')
        if len(e)<10 or len(g)<10: continue
        n=min(len(e),len(g)); a=horn(e[:n],g[:n]); pf=np.linalg.norm(a-g[:n],axis=1)*1000
        rows.append((pf.mean(),np.percentile(pf,90),pf.max()))
    if not rows: print(f"  {cond:<24} (no seeds)"); continue
    A=np.array(rows)
    print(f"  {cond:<24} n={len(rows)}  mean={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  "
          f"p90={A[:,1].mean():5.2f}+/-{A[:,1].std():.2f}  max={A[:,2].mean():6.2f}+/-{A[:,2].std():.2f}")
print("  (base seed-std = the kernel-noise floor; uncert helps if its max/p90 mean < base by > that std)")
print("\n--- SemSup trail3_moge2 render (PSNR/SSIM/LPIPS) ---")
for cond in ['trail3_moge2_uncert_base','trail3_moge2_uncert']:
    P=[]
    for d in sorted(glob.glob(f'{LW}/{cond}_s*')):
        t=f'{d}/render_metrics.txt'
        if not os.path.isfile(t): continue
        s=open(t).read()
        def grab(k):
            m=re.search(k+r'[^0-9-]*([0-9.]+)', s)
            return float(m.group(1)) if m else None
        ps,ss,lp=grab('PSNR'),grab('SSIM'),grab('LPIPS')
        if ps: P.append((ps,ss or 0,lp or 0))
    if not P: print(f"  {cond:<26} (no render metrics)"); continue
    A=np.array(P)
    print(f"  {cond:<26} n={len(P)}  PSNR={A[:,0].mean():5.2f}+/-{A[:,0].std():.2f}  "
          f"SSIM={A[:,1].mean():.3f}  LPIPS={A[:,2].mean():.3f}")
print("\nVideos: each cell payload has <cell>_6panel.mp4 + frames_sample/. Parity: $DRIVE/parity/fork_check.txt")
PY
say "=== A100 IMPROVE A/B DONE $(date -Iseconds) ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
