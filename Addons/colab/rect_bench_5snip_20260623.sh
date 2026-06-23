#!/bin/bash
# ============================================================================
# RECTIFIED 5-snippet CRCD benchmark — Base DDS-SLAM @ the IMPROVED config.
# Snippets: C1_001 C2_001 E3_005 C3_001 G3_001 (the Arm-4 benchmark set).
# RECTIFIED pipeline (consistent): per snippet ->
#   preprocess_crcd_published (RECTIFY left+right, emit video_frames/*l.png + masks/ + rectified_calib.txt + GT)
#   -> MoGe-2 depth ON THE RECTIFIED left frames (regenerated, NOT the raw-left corpus)
#   -> derive_crcd_bounds (bound from this snippet's rectified depth + rectified calib)
#   -> inject per-snippet config (seed/datadir/timesteps/bound/rectified-intrinsics) over crcd_improved_rect.yaml
#   -> train -> Sim3 ATE (sim3_ate.py) + render PSNR/SSIM/LPIPS (eval_rendering) + Depth-L1 + 6-panel video
#   -> aggregate across the snippets.
# RUN FROM A FRESH CLONE (T4 ok): git clone -b diagnosis-live ... /content/DDS-SLAM-rect && cd it && bash this.
#   knobs: SNIPPETS="C1_001"  SEEDS="0 1 2"  PARALLEL=1
# Resume-safe (.DONE per snippet x seed). Headline tracking only on C2_001 (others sub-SNR -> render-only).
# !! FIRST RUN VALIDATES the rectified pipeline (rectify+MoGe+bound+intrinsics) on snippet 1 before all 5.
# ============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
DATE=$(date +%Y%m%d)
SNIPPETS="${SNIPPETS:-C1_001 C2_001 E3_005 C3_001 G3_001}"; SEEDS="${SEEDS:-0}"
PARALLEL="${PARALLEL:-1}"; NPROC=$(nproc 2>/dev/null||echo 8); THREADS=$(( NPROC/PARALLEL>0 ? NPROC/PARALLEL : 1 ))
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
DPUB=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB=$DPUB/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
DEPTH_SCALE=10000
DRIVE=/content/drive/MyDrive/Outputs/rect_bench_${DATE}; mkdir -p "$DRIVE"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_m(){ [ -f "$1/.DONE" ]; }
say "=== RECTIFIED 5-snippet bench  REPO=$REPO HEAD=$(git rev-parse --short HEAD)  snippets=$SNIPPETS seeds=${SEEDS// /,} ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
[ -f "$CALIB" ] || { say "FATAL: calib pkl missing at $CALIB"; exit 1; }

# ---------- env: colab_setup + tcnn for the live GPU arch + MoGe-2 ----------
DINO_PY=python3
python -c "import torch, tinycudann, marching_cubes" 2>/dev/null || { say "env build (~15min)"; bash Addons/env/colab_setup.sh --skip-data --skip-tunnel; }
python -c "import torch;assert torch.cuda.is_available()" || { say "FATAL: no CUDA"; exit 1; }
CC=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo 75)
if ! python - <<'PY' 2>/dev/null
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,"log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda'))
PY
then say "tcnn rebuild for sm_$CC (~10min)"; TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" || { say "FATAL TCNN"; exit 1; }; fi
python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || { say "installing MoGe-2"; pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1|tail -2; python -c 'from moge.model.v2 import MoGeModel' || { say "FATAL MoGe-2"; exit 1; }; }
python -c "import lpips" 2>/dev/null || pip install -q lpips || true

# ---------- NAME -> CRCD-Published episode/snippet path (C1_001 -> C_1/snippet_001) ----------
snip_src(){ local n=$1; local ep=${n%_*}; local sn=${n#*_}; echo "$DPUB/${ep:0:1}_${ep:1}/snippet_${sn}"; }

# ---------- stage ONE snippet rectified (idempotent: .STAGED marker) ----------
stage_rect(){ local NAME=$1 SRC; local DD="$REPO/data/CRCD/$NAME"; SRC=$(snip_src "$NAME")
  [ -f "$DD/.STAGED" ] && { say "  $NAME already staged"; return 0; }
  [ -d "$SRC/rgb" ] || { say "  FATAL: snippet rgb missing at $SRC"; return 1; }
  say "  rectify $NAME  ($SRC)"
  python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SRC" --calib_pkl "$CALIB" --output_dir "$DD" || { say "  FATAL rectify"; return 1; }
  mkdir -p "$DD/_mi"; for f in "$DD/video_frames"/*l.png; do ln -sf "$f" "$DD/_mi/$(basename "${f%l.png}")-left.png"; done
  $DINO_PY Addons/depth/generate_depth_moge.py --rgb "$DD/_mi" --out "$DD/_mo" --temporal_window 1 --depth_scale $DEPTH_SCALE --max_depth_m 5.0 --resolution_level 9 || { say "  FATAL MoGe"; return 1; }
  mkdir -p "$DD/depth"; $DINO_PY - "$DD" <<'PY'
import numpy as np, cv2, glob, os, sys; c=sys.argv[1]
for p in sorted(glob.glob(c+'/_mo/*-left_depth.npy')):
    st=os.path.basename(p).split('-')[0]; cv2.imwrite(f'{c}/depth/{st}.png', np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('depth pngs:', len(glob.glob(c+'/depth/*.png')))
PY
  # STEREO ANCHOR every 120 frames + smooth linear ramp -> METRIC depth (the prior CRCD scale-match we ran).
  # SGBM on the rectified L/R pair (TRUE metric) bakes depth/moge2_stereo120/*.png = moge_m*sc_f*scale; we then
  # OVERWRITE depth/*.png (the CRCD StereoMISDataset loader globs depth/*.png) so training reads METRIC depth at
  # sc_factor=1 -> DDS-SLAM's hardcoded trunc/range_d/near/far (tuned for metric stereo) apply correctly. (MoGe
  # scale is ~one global per-snippet factor, so the 120-frame ramp mainly smooths residual.) Asserts >=1 anchor.
  $DINO_PY Addons/depth/stereo120_metric_anchor.py --staged "$DD" --interval 120 --in_scale $DEPTH_SCALE --out_scale $DEPTH_SCALE --max_depth_m 5.0 || { say "  FATAL stereo120 anchor"; return 1; }
  cp -f "$DD/depth/moge2_stereo120"/*.png "$DD/depth/" || { say "  FATAL bake metric depth -> depth/"; return 1; }
  say "  $NAME stereo-anchor quality: $(cat "$DD/.anchor_quality" 2>/dev/null || echo '?')"   # PASS/WARN/FAIL gate (FAIL already aborted above)
  $DINO_PY Addons/preprocess/derive_crcd_bounds.py --depth_dir "$DD/depth" --calib "$DD/rectified_calib.txt" --depth_scale $DEPTH_SCALE --name "$NAME" --out "$DD/bound.yaml" || { say "  FATAL bound"; return 1; }
  rm -rf "$DD/_mi" "$DD/_mo"; touch "$DD/.STAGED"
  say "  $NAME staged: $(ls "$DD/video_frames"/*l.png|wc -l) frames, $(ls "$DD/depth"/*.png|wc -l) depth, masks $(ls "$DD/masks"/*.png 2>/dev/null|wc -l)"
}

# ---------- assemble per-snippet config: template + seed/datadir/timesteps/bound/rectified-intrinsics ----------
mk_cfg(){ local NAME=$1 OUT=$2 OVR=$3 S=$4; local DD="$REPO/data/CRCD/$NAME"
  $DINO_PY - "$NAME" "$DD" "$OUT" "$OVR" "$DEPTH_SCALE" "$S" <<'PY'
import sys, yaml, glob, cv2
NAME, DD, OUT, OVR, DS, S = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
b = yaml.full_load(open(f"{DD}/bound.yaml"))                                  # mapping.bound + marching_cubes_bound
ts = len(glob.glob(f"{DD}/video_frames/*l.png"))
img = cv2.imread(sorted(glob.glob(f"{DD}/video_frames/*l.png"))[0]); H, W = img.shape[:2]
intr = {}
for ln in open(f"{DD}/rectified_calib.txt"):
    p = ln.split()
    if len(p) >= 2 and p[0] in ("fx","fy","cx","cy"):
        try: intr[p[0]] = float(p[1])
        except: pass
assert all(k in intr for k in ("fx","fy","cx","cy")), f"rectified_calib.txt missing intrinsics: {intr}"
cfg = {"inherit_from":"configs/CRCD/crcd_improved_rect.yaml", "seed":S, "timesteps":ts,
       "mapping":{"bound":b["mapping"]["bound"], "marching_cubes_bound":b["mapping"]["marching_cubes_bound"]},
       "data":{"datadir":f"data/CRCD/{NAME}", "output":OUT, "exp_name":"demo"},
       "cam":{"fx":intr["fx"],"fy":intr["fy"],"cx":intr["cx"],"cy":intr["cy"],"H":H,"W":W}}  # png/output_depth_scale live in cam.* (base=10000); cam merges deeply (keeps near/far/depth_trunc)
yaml.safe_dump(cfg, open(OVR,"w"), sort_keys=False)
print(f"[cfg] {NAME}: seed={S} ts={ts} HxW={H}x{W} fx={intr['fx']:.1f} cx={intr['cx']:.1f} bound={cfg['mapping']['bound']}")
PY
}

# ---------- run ONE (snippet x seed): train -> Sim3 ATE + render + Depth-L1 + 6-panel video -> ship ----------
run_one(){ local NAME=$1 S=$2; local DD="$REPO/data/CRCD/$NAME" CELL="${NAME}_s${S}"
  local DST="$DRIVE/$CELL" OUT="output/$CELL" RUN="output/$CELL/demo" OVR="/content/_rect_${CELL}.yaml"
  done_m "$DST" && { say "  $CELL done -> skip"; return 0; }
  mkdir -p "$DST"; mk_cfg "$NAME" "$OUT" "$OVR" "$S" || { echo "FAILED cfg" >"$DST/.FAILED"; return 1; }
  { echo "=== $CELL $(date -Iseconds) ==="
    python -W ignore - "$OVR" <<'PY'
import os,sys,runpy,torch
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
sys.argv=['ddsslam.py','--config',sys.argv[1]]; runpy.run_path('ddsslam.py',run_name='__main__')
PY
    python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DD/groundtruth.txt" --name "$CELL" --out "$DST/sim3_metrics.txt" || echo WARN-sim3
    CUDA_VISIBLE_DEVICES="" python -u Addons/eval/eval_rendering.py --gt_dir "$DD/video_frames" --render_dir "$OUT" --name "$CELL" --sequence "CRCD ($NAME)" > "$DST/render_eval.txt" 2>&1 || echo WARN-render
    python Addons/eval/depth_l1.py --render_depth_dir "$OUT/depth" --input_depth_dir "$DD/depth" --render_scale $DEPTH_SCALE --input_scale $DEPTH_SCALE --sc_factor 1.0 --out "$DST/depth_l1.txt" || echo WARN-depthl1
    python Addons/viz/generate_video.py --rgb_input_dir "$DD/video_frames" --rgb_input_pattern '*l.png' \
      --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' --depth_input_dir "$DD/depth" --depth_output_dir "$OUT/depth" \
      --seg_dir "$DD/masks" --seg_classmap --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DD/groundtruth.txt" \
      --output "$DST/panels.mp4" --fps 15 || echo WARN-video
    cp "$RUN/est_c2w_data.txt" "$DD/.anchor_quality" "$DST/" 2>/dev/null
  } > "$DST/run.log" 2>&1
  local N; N=$(grep -cvE '^\s*#|^\s*$' "$RUN/est_c2w_data.txt" 2>/dev/null); [ "${N:-0}" -ge 1 ] && touch "$DST/.DONE" || echo "FAILED" >"$DST/.FAILED"
  say "  $CELL -> $(grep -h PSNR "$DST/render_eval.txt" 2>/dev/null|head -1) | $(grep -hE 'rmse/mean' "$DST/sim3_metrics.txt" 2>/dev/null|head -1)"
}

# ---------- drive ----------
for NAME in $SNIPPETS; do stage_rect "$NAME" || say "  $NAME stage FAILED -> skipped"; done
JOBS=(); for NAME in $SNIPPETS; do [ -f "$REPO/data/CRCD/$NAME/.STAGED" ] && for S in $SEEDS; do JOBS+=("$NAME|$S"); done; done
say "########## RUN ${#JOBS[@]} (snippet x seed), PARALLEL=$PARALLEL ##########"
r=0; for j in "${JOBS[@]}"; do IFS='|' read -r n s <<< "$j"; run_one "$n" "$s" &
  r=$((r+1)); [ "$r" -ge "$PARALLEL" ] && { wait -n; r=$((r-1)); }; sleep 2; done; wait

say "########## AGGREGATE ##########"
NAMES=""; for n in $SNIPPETS; do for s in $SEEDS; do NAMES="$NAMES ${n}_s${s}"; done; done
python Addons/eval/aggregate_crcd_generic.py --root "$DRIVE" --names $NAMES --out "$DRIVE/SUMMARY.txt" 2>&1 | tail -40 || say "(aggregator best-effort: per-cell metrics are in $DRIVE/<snippet>_s<seed>/{sim3_metrics,render_eval,depth_l1}.txt)"
say "=== RECTIFIED 5-snippet bench DONE. Headline tracking = C2_001 only; render+Depth-L1 = all snippets. ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || true
