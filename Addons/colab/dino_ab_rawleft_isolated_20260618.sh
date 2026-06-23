#!/bin/bash
# ============================================================================
# SIMPLE · ISOLATED · RAW-LEFT  dino-registers A/B.  2026-06-18.
# Self-contained: OWN data sync into THIS clone's data/ (REPO = the script's own location, NOT a
# hardcoded /content/DDS-SLAM) -> never touches the other agent's tree. CRCD uses the RAW LEFT frames
# (no rectification). Minimal: base/geo/dino/dino_reg, n=1 (SEEDS="0 1 2" for n=3). Quick.
#
# RUN FROM A SEPARATE CLONE (full isolation from the seg-head):
#   git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM /content/DDS-SLAM-dinoab
#   cd /content/DDS-SLAM-dinoab && bash Addons/colab/dino_ab_rawleft_isolated_20260618.sh
#   knobs: SEEDS="0 1 2"  CELLS="dino dino_reg"  PARALLEL=3  DATASETS="crcd super"
# ============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"; export DDS_DIR=$REPO   # <- isolation
DATE=$(date +%Y%m%d)
SEEDS="${SEEDS:-0}"; CELLS="${CELLS:-base geo dino dino_reg}"; DATASETS="${DATASETS:-crcd super}"
PARALLEL="${PARALLEL:-3}"; NPROC=$(nproc 2>/dev/null||echo 8); THREADS=$(( NPROC/PARALLEL>0 ? NPROC/PARALLEL : 1 ))
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
DPUB=/content/drive/MyDrive/Datasets/CRCD-Published
DRIVE=/content/drive/MyDrive/Outputs/dino_ab_rawleft_${DATE}; mkdir -p "$DRIVE"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_m(){ [ -f "$1/.DONE" ]; }
say "=== ISOLATED raw-left dino A/B  REPO=$REPO  HEAD=$(git rev-parse --short HEAD)  cells=$CELLS seeds=${SEEDS// /,} ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
DINO_PY=python3
python -c "import torch, tinycudann, marching_cubes" 2>/dev/null || { say "env build (~15min)"; bash Addons/env/colab_setup.sh --skip-data --skip-tunnel; }
python -c "import torch;assert torch.cuda.is_available()" || { say "FATAL: no CUDA"; exit 1; }
# tinycudann arch: colab_setup builds sm_75 (T4). On A100 (sm_80) the cached kernels won't run -> probe the
# LIVE GPU and force-rebuild tcnn for its arch (~10min). No-op on T4. (Ported from the a100 runbooks.)
CC=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo 75)
if ! python - <<'PY' 2>/dev/null
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,"log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda')); print("tcnn ok")
PY
then
  say "tinycudann probe FAILED on sm_$CC -> rebuild for this arch (~10min)"
  TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" || { say "FATAL: TCNN rebuild failed"; exit 1; }
  python -c "import torch,tinycudann as tcnn; tcnn.Encoding(3,{'otype':'HashGrid','n_levels':2,'n_features_per_level':2,'log2_hashmap_size':15,'base_resolution':16,'per_level_scale':1.5})(torch.rand(8,3,device='cuda'))" || { say "FATAL: TCNN still broken on sm_$CC"; exit 1; }
fi
# MoGe-2 is a SEPARATE pip install (colab_setup.sh does not include it) — needed for the raw-left depth.
[[ " $DATASETS " == *crcd* ]] && { python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || { say "installing MoGe-2 (~2min)"; pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -3; python -c 'from moge.model.v2 import MoGeModel' || { say "FATAL: MoGe-2 still not importable"; exit 1; }; }; }

# ---------- stage CRCD RAW-LEFT (own data/, isolated) ----------
CRCD=$REPO/data/CRCD/C1_001
if [[ " $DATASETS " == *crcd* ]] && [ ! -f "$CRCD/.RL_DONE" ]; then
  say "stage CRCD raw-left (rgb -> video_frames/*l.png + MoGe on raw-left)"
  SRC=$DPUB/C_1/snippet_001
  [ -d "$SRC/rgb" ] || { say "FATAL: CRCD-Published rgb missing at $SRC"; exit 1; }
  NSRC=$(ls "$SRC/rgb"/*.png|wc -l); mkdir -p "$CRCD/video_frames" "$CRCD/_mi"
  if [ "$(ls "$CRCD/video_frames"/*l.png 2>/dev/null|wc -l)" -lt "$NSRC" ]; then     # resumable: skip re-copy if frames already staged
    rm -rf "$CRCD/video_frames" "$CRCD/depth"; mkdir -p "$CRCD/video_frames"
    i=0; for f in $(ls "$SRC/rgb"/*.png | sort); do printf -v n '%06dl.png' "$i"; cp "$f" "$CRCD/video_frames/$n"; i=$((i+1)); done; fi
  for f in "$CRCD/video_frames"/*l.png; do n=$(basename "$f"); ln -sf "$f" "$CRCD/_mi/${n%l.png}-left.png"; done
  cp "$SRC/groundtruth.txt" "$CRCD/groundtruth.txt" 2>/dev/null || true
  say "  raw-left frames: $(ls "$CRCD/video_frames"/*l.png|wc -l)"
  $DINO_PY Addons/depth/generate_depth_moge.py --rgb "$CRCD/_mi" --out "$CRCD/_mo" --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 --resolution_level 9 || { say "FATAL MoGe"; exit 1; }
  mkdir -p "$CRCD/depth"; $DINO_PY - "$CRCD" <<'PY'
import numpy as np, cv2, glob, os, sys; c=sys.argv[1]
for p in sorted(glob.glob(c+'/_mo/*-left_depth.npy')):
    st=os.path.basename(p).split('-')[0]; cv2.imwrite(f'{c}/depth/{st}.png', np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('depth pngs:', len(glob.glob(c+'/depth/*.png')))
PY
  # seg (CRCD loader collapses it to a Canny-edge channel) — raw semantic_instance -> masks/*l.png
  # (StereoMISDataset globs {basedir}/masks/*.png at dataset.py:138 — NOT semantic_class/)
  if [ -d "$SRC/semantic_instance" ]; then mkdir -p "$CRCD/masks"; j=0
    for sf in $(ls "$SRC/semantic_instance"/*.png|sort); do printf -v sn '%06dl.png' "$j"; cp "$sf" "$CRCD/masks/$sn"; j=$((j+1)); done
    say "  seg masks: $(ls "$CRCD/masks"/*l.png 2>/dev/null|wc -l)"; fi
  rm -rf "$CRCD/_mi" "$CRCD/_mo"; touch "$CRCD/.RL_DONE"
fi

# ---------- stage SemSup (own data/, already left-frames) ----------
SUP=$REPO/data/Super/trail_3
if [[ " $DATASETS " == *super* ]] && [ ! -d "$SUP/depth/moge2" ]; then
  say "stage SemSup"
  SS=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  [ -d "$SS/rgb" ] && { mkdir -p "$REPO/data/Super"; cp -r "$SS" "$SUP"; }
  mkdir -p "$SUP/depth/moge2"
  for c in "$SS/depth/MoGe2_trail3_20260608" /content/drive/MyDrive/Datasets/SemSup/MoGe2_trail3_20260608; do
    [ -d "$c" ] && cp "$c"/*left_depth.npy "$SUP/depth/moge2/" 2>/dev/null && break; done
  say "  SemSup moge2: $(ls "$SUP/depth/moge2"/*left_depth.npy 2>/dev/null|wc -l) npy"
fi

# ---------- bake DINO vits14 + reg ----------
bake(){ local RGB=$1 OUT=$2 BK=$3 GLOB=$4 N=$5
  [ "$(ls "$OUT"/*_dino.npy 2>/dev/null|wc -l)" -ge "$N" ] && return 0
  $DINO_PY Addons/dino/generate_dino_features.py --rgb_dir "$RGB" --rgb_glob "$GLOB" --out_dir "$OUT" --backbone "$BK" --fp32 2>&1 | tail -3; }
# DINO .npy bake = ONLY for the uncertainty dino cells. flow_track / map_route load DINO LIVE (load_dino,
# torch.hub) so the combine cells (base/flow_*/route*) need NO bake -> guard on a 'dino' cell being requested.
if [[ " $CELLS " == *dino* ]] && [[ " $DATASETS " == *crcd* ]]; then NC=$(ls "$CRCD/video_frames"/*l.png|wc -l)
  bake "$CRCD/video_frames" "$CRCD/dino" dinov2_vits14 '*l.png' "$NC"; bake "$CRCD/video_frames" "$CRCD/dino_reg" dinov2_vits14_reg '*l.png' "$NC"; fi
if [[ " $CELLS " == *dino* ]] && [[ " $DATASETS " == *super* ]]; then NS=$(ls "$SUP/rgb"/*left.png|wc -l)
  bake "$SUP/rgb" "$SUP/dino" dinov2_vits14 '*left.png' "$NS"; bake "$SUP/rgb" "$SUP/dino_reg" dinov2_vits14_reg '*left.png' "$NS"; fi

# ---------- run_one: train (TF32 off + seed) -> render eval (+Sim3 for CRCD) -> ship ----------
run_one(){ local CFG=$1 NAME=$2 S=$3 DD=$4 VT=$5
  local DST="$DRIVE/$NAME" OUT="output/$NAME" RUN="output/$NAME/demo" OVR="/content/_iso_$NAME.yaml"
  done_m "$DST" && { say "  $NAME done -> skip"; return 0; }
  mkdir -p "$DST"
  cat > "$OVR" <<YML
inherit_from: $CFG
seed: $S
data:
  output: $OUT
  exp_name: demo
YML
  { echo "=== $NAME $(date -Iseconds) ==="
    python -W ignore - "$OVR" <<'PY'
import os,sys,runpy,torch
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
sys.argv=['ddsslam.py','--config',sys.argv[1]]; runpy.run_path('ddsslam.py',run_name='__main__')
PY
    RC=$?
    if [ "$VT" = crcd ]; then
      python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DD/groundtruth.txt" --name "$NAME" --out "$DST/sim3.txt" || echo WARN-sim3
      CUDA_VISIBLE_DEVICES="" python -u Addons/eval/eval_rendering.py --gt_dir "$DD/video_frames" --render_dir "$OUT" --name "$NAME" --sequence "CRCD (C1_001)" > "$DST/render.txt" 2>&1 || echo WARN-render
    else
      CUDA_VISIBLE_DEVICES="" python -u Addons/eval/eval_rendering.py --gt_dir "$DD/rgb" --render_dir "$OUT" --name "$NAME" --sequence "Lab1 (trail3)" > "$DST/render.txt" 2>&1 || echo WARN-render
    fi
    # 6-panel video — standing rule: every result ships metrics + the canonical video
    VA=(--rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' --depth_output_dir "$OUT/depth" --uncert_dir "$OUT/uncert" --whatkind_dir "$OUT/whatkind" --output "$DST/panels.mp4" --fps 15)
    if [ "$VT" = crcd ]; then VA+=(--rgb_input_dir "$DD/video_frames" --rgb_input_pattern '*l.png' --depth_input_dir "$DD/depth" --seg_dir "$DD/masks" --seg_classmap --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DD/groundtruth.txt")
    else VA+=(--rgb_input_dir "$DD/rgb" --rgb_input_pattern '*left.png' --depth_input_dir "$DD/depth/moge2"); fi
    python Addons/viz/generate_video.py "${VA[@]}" || echo WARN-video
    # Arm-1 σ² judges — run on the EPHEMERAL uncert/ before it's lost; ship JSON+PNG (uncert cells only).
    # sigma2_quality = AUSE calibration + σ²-vs-motion (the DIRECT metric); sigma2_diagnostics = contrast + per-seg.
    if [ -d "$OUT/uncert" ]; then
      if [ "$VT" = crcd ]; then SQ=(--rgb_dir "$DD/video_frames" --rgb_glob '*l.png'); SD=(--seg_dir "$DD/masks" --seg_glob '*.png' --tool_labels 3 --bg_labels 0)
      else SQ=(--rgb_dir "$DD/rgb" --rgb_glob '*left.png'); SD=(--seg_dir "$DD/seg/png_masks" --seg_glob '*left.png'); fi
      python Addons/eval/sigma2_quality.py --name "$NAME" --uncert_dir "$OUT/uncert" --render_dir "$OUT" --render_glob '[0-9]*.jpg' "${SQ[@]}" --out_json "$DST/${NAME}_sigma2_quality.json" --out_fig "$DST/${NAME}_sigma2_quality.png" || echo WARN-sigq
      python Addons/eval/sigma2_diagnostics.py --name "$NAME" --uncert_dir "$OUT/uncert" "${SQ[@]}" "${SD[@]}" --out_fig "$DST/${NAME}_sigma2_diag.png" || echo WARN-sigd
    fi
    cp "$RUN"/est_c2w_data.txt "$DST/" 2>/dev/null
  } > "$DST/run.log" 2>&1
  N=$(grep -cvE '^\s*#|^\s*$' "$RUN/est_c2w_data.txt" 2>/dev/null); [ "${N:-0}" -ge 1 ] && touch "$DST/.DONE" || echo "FAILED" > "$DST/.FAILED"
  say "  $NAME -> $(grep -h PSNR "$DST/render.txt" 2>/dev/null|head -1) $(grep -h 'mean=' "$DST/sim3.txt" 2>/dev/null|head -1)"
}

declare -A CR=( [base]=c1_001_canon_base [geo]=c1_001_canon_uncert [dino]=c1_001_canon_uncert_dino [dino_reg]=c1_001_canon_uncert_dino_reg [geo_rd]=c1_001_canon_uncert_rgbdepth [dino_reg_rd]=c1_001_canon_uncert_dino_reg_rgbdepth [dino_reg_f]=c1_001_canon_uncert_dino_reg_fused [geofuse]=c1_001_canon_uncert_dino_reg_rgbdepth_geofuse [georgbd]=c1_001_canon_uncert_dino_reg_rgbdepth_georgbd [slot]=c1_001_canon_uncert_dino_reg_slot [slot_v1a]=c1_001_canon_uncert_dino_reg_slot_v1a [flow_track]=c1_001_canon_flow_track [geo_flow]=c1_001_canon_geo_flow [flow_gate]=c1_001_canon_flow_gate [flow_agree]=c1_001_canon_flow_agree [flow_agree_baf]=c1_001_canon_flow_agree_baf [geo_flow_agree_baf]=c1_001_canon_geo_flow_agree_baf [curmap100]=c1_001_canon_curmap100 [stack]=c1_001_canon_flow_agree_baf_curmap100 [decoder64]=c1_001_canon_decoder64 [improved]=c1_001_canon_curmap100_decoder64 [flow_improved]=c1_001_canon_flow_agree_baf_curmap100_decoder64 [toolmask]=c1_001_canon_curmap100_decoder64_toolmask [flow_route]=c1_001_canon_flow_agree_baf_route [flow_route_protect]=c1_001_canon_flow_agree_baf_route_protect [flow_route_attend]=c1_001_canon_flow_agree_baf_route_attend )
declare -A SU=( [base]=trail3_moge2_uncert_base [geo]=trail3_moge2_uncert [dino]=trail3_moge2_uncert_dino [dino_reg]=trail3_moge2_uncert_dino_reg [geo_rd]=trail3_moge2_uncert_rgbdepth [dino_reg_rd]=trail3_moge2_uncert_dino_reg_rgbdepth [dino_reg_f]=trail3_moge2_uncert_dino_reg_fused [geofuse]=trail3_moge2_uncert_dino_reg_rgbdepth_geofuse [slot]=trail3_moge2_uncert_dino_reg_slot )
JOBS=(); for s in $SEEDS; do for c in $CELLS; do
  [[ " $DATASETS " == *crcd* ]] && JOBS+=("configs/CRCD/${CR[$c]}.yaml|${c}_crcd_s$s|$s|$REPO/data/CRCD/C1_001|crcd")
  [[ " $DATASETS " == *super* ]] && JOBS+=("configs/Super/${SU[$c]}.yaml|${c}_super_s$s|$s|$REPO/data/Super/trail_3|super")
done; done
say "########## RUN ${#JOBS[@]} cells, PARALLEL=$PARALLEL ##########"
r=0; for j in "${JOBS[@]}"; do IFS='|' read -r cfg nm s dd vt <<< "$j"; run_one "$cfg" "$nm" "$s" "$dd" "$vt" &
  r=$((r+1)); [ "$r" -ge "$PARALLEL" ] && { wait -n; r=$((r-1)); }; sleep 2; done; wait
say "########## SUMMARY ##########"
for j in "${JOBS[@]}"; do IFS='|' read -r _ nm _ _ _ <<< "$j"
  printf '  %-22s %s | %s\n' "$nm" "$(grep -h PSNR "$DRIVE/$nm/render.txt" 2>/dev/null|head -1)" "$(grep -hE 'mean=|ATE' "$DRIVE/$nm/sim3.txt" 2>/dev/null|head -1)"; done
say "=== DONE.  base/geo/dino/dino_reg on raw-left CRCD + SemSup. Compare dino vs dino_reg. ==="
