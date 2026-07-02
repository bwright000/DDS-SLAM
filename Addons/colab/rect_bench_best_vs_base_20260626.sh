#!/bin/bash
# ============================================================================
# RECTIFIED 5-snippet CRCD benchmark — TWO ARMS per snippet:
#   base = DDS-SLAM @ improved config (crcd_improved_rect)
#   best = CHAMPION (crcd_best_rect = best_deformiters + Charbonnier loss)
# Each snippet is STAGED ONCE (rectify+MoGe+anchor); BOTH arms train on the SAME rectified metric depth.
# Snippets: C1_001 C2_001 E3_005 C3_001 G3_001 (the Arm-4 benchmark set).
# RECTIFIED pipeline (consistent): per snippet ->
#   preprocess_crcd_published (RECTIFY left+right, emit video_frames/*l.png + masks/ + rectified_calib.txt + GT)
#   -> MoGe-2 depth ON THE RECTIFIED left frames (regenerated, NOT the raw-left corpus)
#   -> derive_crcd_bounds (bound from this snippet's rectified depth + rectified calib)
#   -> per ARM: inject per-snippet config (seed/datadir/timesteps/bound/rectified-intrinsics) over the arm template
#   -> train -> Sim3 ATE (sim3_ate.py) + render PSNR/SSIM/LPIPS (eval_rendering) + Depth-L1 + 6-panel video
#   -> aggregate across the snippets.
# RUN FROM A FRESH CLONE (T4 ok): git clone -b diagnosis-live ... /content/DDS-SLAM-rect && cd it && bash this.
#   knobs: SNIPPETS="C1_001"  SEEDS="0 1 2"  ARMS="best"
# ORDER: ONE SNIPPET AT A TIME, SHORTEST-FIRST (by source frame count) -> stage once, run BOTH arms, next snippet.
#   => the short snippets land first (fast feedback) and resume granularity is a WHOLE snippet.
# Resume-safe (.DONE per snippet x arm x seed on Drive). Headline tracking only on C2_001 (others sub-SNR).
# ============================================================================
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
DATE="${DATE:-$(date +%Y%m%d)}"   # override (e.g. DATE=20260627) to resume into an existing output dir on a later day
SNIPPETS="${SNIPPETS:-C1_001 C2_001 E3_005 C3_001 G3_001}"; SEEDS="${SEEDS:-0}"
ARMS="${ARMS:-base best}"   # base=crcd_improved_rect (DDS-SLAM) ; best=crcd_best_rect (champion=best_deformiters+charbonnier)
declare -A ARM_TMPL=( [base]=configs/CRCD/crcd_improved_rect.yaml [best]=${BEST_CFG:-configs/CRCD/crcd_best_rect.yaml} \
  [abl_base]=configs/CRCD/crcd_abl_base_rect.yaml [l0]=configs/CRCD/crcd_abl_l0_rect.yaml [l0aggr]=configs/CRCD/crcd_abl_l0aggr_rect.yaml \
  [l0sig]=configs/CRCD/crcd_abl_l0sig_rect.yaml [l0sigaggr]=configs/CRCD/crcd_abl_l0sigaggr_rect.yaml \
  [dpool]=configs/CRCD/crcd_abl_dpool_rect.yaml [pnp]=configs/CRCD/crcd_abl_pnp_rect.yaml \
  [prior]=configs/CRCD/crcd_abl_prior_rect.yaml \
  [cons]=configs/CRCD/crcd_abl_cons_rect.yaml )   # BEST_CFG= override (T4); abl_*/l0*/dpool/pnp/prior/cons = ablation arms; prior=zero-motion prior (lambda sweep via DDS_MP_LAM_T/R env); cons=epoch-consistent poses (BA-jump fix)
PARALLEL="${PARALLEL:-1}"; NPROC=$(nproc 2>/dev/null||echo 8); THREADS=$(( NPROC/PARALLEL>0 ? NPROC/PARALLEL : 1 ))
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
DPUB=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB=$DPUB/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
DEPTH_SCALE=10000
DRIVE=/content/drive/MyDrive/Outputs/rect_bestbase_${DATE}; mkdir -p "$DRIVE"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_m(){ [ -f "$1/.DONE" ]; }
say "=== RECTIFIED bench base-vs-best  REPO=$REPO HEAD=$(git rev-parse --short HEAD)  snippets=$SNIPPETS arms=$ARMS seeds=${SEEDS// /,} ==="
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
# best arm needs the champion config + RAFT/DINOv2 (flow_agree gate). Validate the config + pre-warm the weights
# ONCE here so parallel best jobs don't each re-download.
if echo " $ARMS " | grep -q " best "; then
  [ -f configs/CRCD/crcd_best_rect.yaml ] || { say "FATAL: configs/CRCD/crcd_best_rect.yaml missing (best arm)"; exit 1; }
  python - <<'PY' 2>&1 | tail -1 || say "WARN: RAFT/DINO prewarm failed (best arm may redownload per job)"
import torch; from Addons.motion.flow_track import load_raft, load_dino
d=torch.device('cuda'); load_raft(d); load_dino(d); print("RAFT+DINOv2 prewarmed for the best arm")
PY
fi

# ---------- NAME -> CRCD-Published episode/snippet path (C1_001 -> C_1/snippet_001) ----------
snip_src(){ local n=$1; local ep=${n%_*}; local sn=${n#*_}; echo "$DPUB/${ep:0:1}_${ep:1}/snippet_${sn}"; }

# ---------- stage ONE snippet rectified (idempotent: .STAGED marker) ----------
stage_rect(){ local NAME=$1 SRC; local DD="$REPO/data/CRCD/$NAME"; SRC=$(snip_src "$NAME")
  [ -f "$DD/.STAGED" ] && { say "  $NAME already staged"; return 0; }
  [ -d "$SRC/rgb" ] || { say "  FATAL: snippet rgb missing at $SRC"; return 1; }
  # RESUME: skip rectify / MoGe if COMPLETELY on disk (e.g. a re-run after an anchor-gate fix) -> straight to
  # anchor. Counts must match the source so a run stopped mid-rectify re-does it (no partial reuse).
  _nsrc=$(ls "$SRC/rgb"/*.png 2>/dev/null | wc -l); _nvf=$(ls "$DD/video_frames"/*l.png 2>/dev/null | wc -l)
  if [ "$_nvf" -gt 0 ] && [ "$_nvf" -eq "$_nsrc" ]; then
    say "  rectify $NAME CACHED ($_nvf == $_nsrc frames) -> skip"
  else
    say "  rectify $NAME  ($SRC)"
    python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SRC" --calib_pkl "$CALIB" --output_dir "$DD" || { say "  FATAL rectify"; return 1; }
  fi
  _nf=$(ls "$DD/video_frames"/*l.png 2>/dev/null | wc -l); _nd=$(ls "$DD/depth"/*.png 2>/dev/null | wc -l)
  if [ "$_nd" -gt 0 ] && [ "$_nd" -eq "$_nf" ]; then
    say "  MoGe depth $NAME CACHED ($_nd == $_nf) -> skip"
  else
  mkdir -p "$DD/_mi"; for f in "$DD/video_frames"/*l.png; do ln -sf "$f" "$DD/_mi/$(basename "${f%l.png}")-left.png"; done
  $DINO_PY Addons/depth/generate_depth_moge.py --rgb "$DD/_mi" --out "$DD/_mo" --temporal_window 1 --depth_scale $DEPTH_SCALE --max_depth_m 5.0 --resolution_level 9 || { say "  FATAL MoGe"; return 1; }
  mkdir -p "$DD/depth"; $DINO_PY - "$DD" <<'PY'
import numpy as np, cv2, glob, os, sys; c=sys.argv[1]
for p in sorted(glob.glob(c+'/_mo/*-left_depth.npy')):
    st=os.path.basename(p).split('-')[0]; cv2.imwrite(f'{c}/depth/{st}.png', np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('depth pngs:', len(glob.glob(c+'/depth/*.png')))
PY
  fi
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
mk_cfg(){ local NAME=$1 OUT=$2 OVR=$3 S=$4 TMPL=$5; local DD="$REPO/data/CRCD/$NAME"
  $DINO_PY - "$NAME" "$DD" "$OUT" "$OVR" "$DEPTH_SCALE" "$S" "$TMPL" <<'PY'
import sys, yaml, glob, cv2
NAME, DD, OUT, OVR, DS, S, TMPL = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
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
cfg = {"inherit_from":TMPL, "seed":S, "timesteps":ts,
       "mapping":{"bound":b["mapping"]["bound"], "marching_cubes_bound":b["mapping"]["marching_cubes_bound"]},
       "data":{"datadir":f"data/CRCD/{NAME}", "output":OUT, "exp_name":"demo"},
       "cam":{"fx":intr["fx"],"fy":intr["fy"],"cx":intr["cx"],"cy":intr["cy"],"H":H,"W":W}}  # png/output_depth_scale live in cam.* (base=10000); cam merges deeply (keeps near/far/depth_trunc)
yaml.safe_dump(cfg, open(OVR,"w"), sort_keys=False)
print(f"[cfg] {NAME}: seed={S} ts={ts} HxW={H}x{W} fx={intr['fx']:.1f} cx={intr['cx']:.1f} bound={cfg['mapping']['bound']}")
PY
}

# ---------- run ONE (snippet x seed): train -> Sim3 ATE + render + Depth-L1 + 6-panel video -> ship ----------
run_one(){ local NAME=$1 ARM=$2 S=$3; local DD="$REPO/data/CRCD/$NAME" CELL="${NAME}_${ARM}_s${S}" TMPL="${ARM_TMPL[$ARM]}"
  local DST="$DRIVE/$CELL" OUT="output/$CELL" RUN="output/$CELL/demo" OVR="/content/_rect_${CELL}.yaml"
  done_m "$DST" && { say "  $CELL done -> skip"; return 0; }
  mkdir -p "$DST"; mk_cfg "$NAME" "$OUT" "$OVR" "$S" "$TMPL" || { echo "FAILED cfg" >"$DST/.FAILED"; return 1; }
  { echo "=== $CELL $(date -Iseconds) ==="
    python -W ignore - "$OVR" <<'PY'
import os,sys,runpy,torch
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','2')))
sys.argv=['ddsslam.py','--config',sys.argv[1]]; runpy.run_path('ddsslam.py',run_name='__main__')
PY
    python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DD/groundtruth.txt" --name "$CELL" --out "$DST/sim3_metrics.txt" || echo WARN-sim3
    python Addons/eval/flow_diag.py --est "$RUN/est_c2w_data.txt" --gt "$DD/groundtruth.txt" --trust "$OUT/trust_log.csv" --name "$CELL" --out "$DST/flow_diag.json" --plot "$DST/flow_diag.png" || echo WARN-flowdiag
    CUDA_VISIBLE_DEVICES="" python -u Addons/eval/eval_rendering.py --gt_dir "$DD/video_frames" --render_dir "$OUT" --name "$CELL" --sequence "CRCD ($NAME)" > "$DST/render_eval.txt" 2>&1 || echo WARN-render
    python Addons/eval/depth_l1.py --render_depth_dir "$OUT/depth" --input_depth_dir "$DD/depth" --render_scale $DEPTH_SCALE --input_scale $DEPTH_SCALE --sc_factor 1.0 --out "$DST/depth_l1.txt" || echo WARN-depthl1
    # seg panel: 4-class semantic_class colorizes; the binary masks/ -> all-black (issue-1 fix). uncert/route/
    # trust panels are auto-added ONLY when the arm wrote them (base -> just the seg fix; depth-sup -> trust).
    _SEGDIR="$DD/masks"; [ -d "$DD/semantic_class" ] && _SEGDIR="$DD/semantic_class"
    _UNC=""; [ -d "$OUT/uncert" ] && _UNC="--uncert_dir $OUT/uncert"
    _RT="";  [ -d "$OUT/route" ]  && _RT="--route_dir $OUT/route"
    _TR="";  [ -d "$OUT/trust" ]  && _TR="--trust_dir $OUT/trust"
    python Addons/viz/generate_video.py --rgb_input_dir "$DD/video_frames" --rgb_input_pattern '*l.png' \
      --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' --depth_input_dir "$DD/depth" --depth_output_dir "$OUT/depth" \
      --seg_dir "$_SEGDIR" --seg_classmap $_UNC $_RT $_TR --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DD/groundtruth.txt" \
      --output "$DST/panels.mp4" --fps 15 || echo WARN-video
    cp "$RUN/est_c2w_data.txt" "$RUN/est_c2w_data_raw.txt" "$DD/.anchor_quality" "$OUT/trust_log.csv" "$DST/" 2>/dev/null
  } > "$DST/run.log" 2>&1
  local N; N=$(grep -cvE '^\s*#|^\s*$' "$RUN/est_c2w_data.txt" 2>/dev/null); [ "${N:-0}" -ge 1 ] && touch "$DST/.DONE" || echo "FAILED" >"$DST/.FAILED"
  say "  $CELL -> $(grep -h PSNR "$DST/render_eval.txt" 2>/dev/null|head -1) | $(grep -hE 'rmse/mean' "$DST/sim3_metrics.txt" 2>/dev/null|head -1)"
}

# ---------- drive: ONE SNIPPET AT A TIME, SHORTEST-FIRST ----------
# Order snippets by source frame count (shortest first) so the small ones finish first (fast feedback) and
# resume granularity is a WHOLE snippet. Per snippet: stage ONCE -> run BOTH arms sequentially -> next snippet.
ORDERED=$(for NAME in $SNIPPETS; do
  printf '%s %s\n' "$(ls "$(snip_src "$NAME")/rgb"/*.png 2>/dev/null | wc -l)" "$NAME"
done | sort -n | awk '{print $2}')
say "########## RUN one-at-a-time (shortest-first): $(echo $ORDERED | tr '\n' ' ')  arms=$ARMS seeds=${SEEDS// /,} ##########"
for NAME in $ORDERED; do
  say ">>> SNIPPET $NAME ($(ls "$(snip_src "$NAME")/rgb"/*.png 2>/dev/null | wc -l) src frames)"
  stage_rect "$NAME" || { say "  $NAME stage FAILED -> skipped"; continue; }
  for ARM in $ARMS; do for S in $SEEDS; do run_one "$NAME" "$ARM" "$S"; done; done
done

say "########## AGGREGATE ##########"
NAMES=""; for n in $SNIPPETS; do for arm in $ARMS; do for s in $SEEDS; do NAMES="$NAMES ${n}_${arm}_s${s}"; done; done; done
python Addons/eval/aggregate_crcd_generic.py --root "$DRIVE" --names $NAMES --out "$DRIVE/SUMMARY.txt" 2>&1 | tail -40 || say "(aggregator best-effort: per-cell metrics are in $DRIVE/<snippet>_s<seed>/{sim3_metrics,render_eval,depth_l1}.txt)"
say "=== RECTIFIED 5-snippet bench DONE. Headline tracking = C2_001 only; render+Depth-L1 = all snippets. ==="
if [ -n "${NO_UNASSIGN:-}" ]; then say "NO_UNASSIGN set -> keeping runtime alive (chained run)"; else python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || true; fi
