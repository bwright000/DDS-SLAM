#!/usr/bin/env bash
# =====================================================================
# SURGICAL-SCALE overnight (2026-06-09): StereoMIS + CRCD, MoGe normalised
# to a common 0.12 m surgical median.
# =====================================================================
# WHY: MoGe-2 depth is up-to-scale (~0.6-0.9 m median, ~5x too deep for
# endoscopy) -> raw ATE ~67x paper + constants mismatched to the scene.  Fix:
# rescale every MoGe set to median 0.12 m (global scalar) and use the
# scale-equivalent *_surgical_20260609 configs (all scale-bearing constants
# scaled by the same k).  Sim3 ATE is scale-invariant; this gives physical
# units + meaningful raw scale + constants matched to the scene.
# ALSO fixes the 2026-06-08 Drive-save bug: StereoMIS est_c2w_data.txt was
# never copied to Drive (only ckpt+renders), so Sim3 was impossible.
#
# TWO-ENV ORDERING (mandatory — MoGe needs torch>=2, DDS-SLAM needs torch 1.10):
#   PHASE 1  ALL MoGe depth + CRCD stage/preprocess  -> SYSTEM python (torch>=2)
#   PHASE 2  ALL DDS-SLAM SLAM + eval                -> dds_env (torch 1.10)
#
# StereoMIS depth: chunked metric gen (OOM-safe) THEN one GLOBAL-median scale in
# the consolidation step (per-chunk --target_median would give inconsistent
# scalars).  CRCD depth: single-call generate_depth_moge.py --target_median 0.12.
#
# PREREQS (run each line separately to avoid Colab paste-mangling):
#   from google.colab import drive; drive.mount('/content/drive')      # (cell)
#   cd /content && (git clone https://github.com/bwright000/DDS-SLAM.git || true) && cd DDS-SLAM && git checkout moge-surgical-scale && git pull
#   bash Addons/env/colab_exact_env.sh --skip-data
#   nohup bash Addons/colab/overnight_surgical_20260609.sh >/content/surg.out 2>&1 &
#   tail -f /content/surg.out
# =====================================================================
set -uo pipefail
DATE=20260609
TARGET_MEDIAN=0.12
REPO=/content/DDS-SLAM
STEREO_DS=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1
STEREO_CKPT_DIR=$STEREO_DS/checkpoints
STEREO_SURG_DIR=$STEREO_DS/depth/MoGe2_surgical_p2-1-back4000_${DATE}
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
RUN_OUT=/content/drive/MyDrive/Outputs/DDS-SLAM_surgical_${DATE}
STE_OUT=$RUN_OUT/stereomis; CRCD_OUT=$RUN_OUT/crcd
mkdir -p "$STEREO_CKPT_DIR" "$STE_OUT" "$CRCD_OUT"
LOG=$RUN_OUT/runbook.log
say(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

# CRCD snippet table: NAME EP SID SURGICAL_CONFIG FRAMES   (smallest-first)
CRCD_SNIPPETS=(
  "F3_007  F_3  007  f3_007_surgical_20260609  300"
  "C1_001  C_1  001  c1_001_surgical_20260609  360"
  "C2_001  C_2  001  c2_001_surgical_20260609  730"
  "F1_002  F_1  002  f1_002_surgical_20260609  1287"
)

cd "$REPO" || { echo "FATAL: $REPO missing"; exit 1; }
[ -d /content/drive/MyDrive ] || { echo "FATAL: Drive not mounted"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "FATAL: no GPU"; exit 1; }
[ -f "$CALIB_PKL" ] || say "WARN: CRCD calib pickle missing ($CALIB_PKL) — CRCD preprocess will fail"
say "=== SURGICAL overnight start $(date -Iseconds) === GPU $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) | HEAD $(git rev-parse --short HEAD 2>/dev/null) | target_median=${TARGET_MEDIAN}m"

# --- stage StereoMIS (full sequence; back-4000 slice enforced in dataset.py) ---
if [ ! -d /content/p2_1_local/video_frames ]; then
  mkdir -p /content/p2_1_local; STAR=$STEREO_DS/P2_1_staging.tar
  [ -f "$STAR" ] && { say "stage StereoMIS from tar"; tar xf "$STAR" -C /content/p2_1_local; } \
    || (cd "$STEREO_DS" && tar cf - video_frames groundtruth.txt StereoCalibration.ini masks 2>/dev/null | tar xf - -C /content/p2_1_local)
fi
( cd /content/p2_1_local/video_frames 2>/dev/null && ls *l.png 2>/dev/null | tail -4000 ) > /content/_p2_frames.txt
TNIMG=$(wc -l < /content/_p2_frames.txt)
say "staged StereoMIS=$TNIMG frames"

# =====================================================================
# PHASE 1 — ALL MoGe depth + CRCD stage/preprocess (SYSTEM python torch>=2)
# =====================================================================
PYSYS=$(command -v python3)
"$PYSYS" -c "import torch,sys;sys.exit(0 if torch.__version__>='2' else 1)" 2>/dev/null || say "WARN: system python torch<2 -> MoGe may fail"
"$PYSYS" -c "import moge" 2>/dev/null || { say "installing moge (system python)"; "$PYSYS" -m pip install -q "git+https://github.com/microsoft/MoGe.git" 2>&1 | tail -2 | tee -a "$LOG"; }

# 1a. StereoMIS back-4000 depth: chunked metric gen, then ONE global-median surgical scale
if [ "$(ls "$STEREO_SURG_DIR"/*.png 2>/dev/null | wc -l)" -lt "$TNIMG" ] && [ "$TNIMG" -gt 0 ]; then
  say "MoGe 1a: StereoMIS back-4000 ($TNIMG frames, chunks of 500, metric)"
  NPY=/content/_ste_npy; rm -rf "$NPY"; mkdir -p "$NPY" "$STEREO_SURG_DIR"
  CH=500; i=0
  while [ "$i" -lt "$TNIMG" ]; do
    ST=/content/_ste_stage; rm -rf "$ST"; mkdir -p "$ST"
    sed -n "$((i+1)),$((i+CH))p" /content/_p2_frames.txt | while read f; do s="${f%l.png}"; ln -sf "/content/p2_1_local/video_frames/$f" "$ST/${s}-left.png"; done
    say "  chunk $i..$((i+CH)): $(ls "$ST"/*-left.png 2>/dev/null | wc -l) staged"
    "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$ST" --out "$NPY" --temporal_window 5 --depth_scale 1 2>&1 | tee -a "$LOG" || say "  WARN: chunk $i failed"
    i=$((i+CH))
  done
  # consolidation: global median across ALL frames -> single surgical scalar -> uint16 PNG
  "$PYSYS" - "$NPY" "$STEREO_SURG_DIR" "$TARGET_MEDIAN" <<'PY'
import sys, glob, os, numpy as np, cv2
nd, pd, target = sys.argv[1], sys.argv[2], float(sys.argv[3])
files = sorted(glob.glob(os.path.join(nd, '*-left_depth.npy')))
# global median = median of per-frame medians (NPYs hold raw metres, depth_scale=1)
meds = []
for f in files:
    d = np.load(f).astype(np.float32); v = d[d > 1e-6]
    if v.size >= 100: meds.append(float(np.median(v)))
gmed = float(np.median(meds)) if meds else 0.0
assert gmed > 0, "global median <= 0"
s = target / gmed
print(f'StereoMIS global median={gmed:.4f} m over {len(meds)} frames; surgical scale s={s:.4f}')
n = 0
for f in files:
    d = np.load(f).astype(np.float32)            # metres
    u = np.clip(d * s * 10000.0, 0, 65535).astype(np.uint16)  # surgical_m * png_depth_scale(10000)
    cv2.imwrite(os.path.join(pd, os.path.basename(f).replace('-left_depth.npy', '') + '.png'), u); n += 1
print('StereoMIS surgical PNGs:', n)
PY
else say "StereoMIS surgical depth present (or 0 frames) -> skip"; fi

# 1b. CRCD per snippet: stage raw -> preprocess -> MoGe --target_median 0.12
for ROW in "${CRCD_SNIPPETS[@]}"; do
  read -r NAME EP SID CFG FRAMES <<< "$ROW"
  RAW=/content/crcd_raw/${EP}_snippet_${SID}; STAGED=$REPO/data/CRCD/${NAME}
  if [ -f "$STAGED/depth/.SURGDONE" ]; then say "CRCD $NAME depth done -> skip"; continue; fi
  # stage raw (tarball-first)
  if [ ! -f "$RAW/.STAGED" ]; then
    DRIVE_TAR=$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar
    mkdir -p "$RAW"
    if [ -f "$DRIVE_TAR" ]; then say "CRCD $NAME: extract tarball"; tar xf "$DRIVE_TAR" -C "$RAW" || { say "  FATAL tar $NAME"; continue; }; touch "$RAW/.STAGED"
    else say "  WARN: no tarball $DRIVE_TAR; per-item cp"; for d in rgb rgbright semantic_instance groundtruth.txt intrinsics.yaml; do cp -rn "$DRIVE_CRCD/$EP/snippet_$SID/$d" "$RAW/" 2>/dev/null || true; done; touch "$RAW/.STAGED"; fi
  fi
  # preprocess (rectify) — cv2/system python
  if [ ! -f "$STAGED/.PREPROCESSED" ]; then
    rm -rf "$STAGED"; mkdir -p "$STAGED"
    "$PYSYS" Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$RAW" --calib_pkl "$CALIB_PKL" --output_dir "$STAGED" 2>&1 | tee -a "$LOG" || { say "  FATAL preprocess $NAME"; continue; }
    [ -f "$STAGED/groundtruth.txt" ] || cp "$RAW/groundtruth.txt" "$STAGED/groundtruth.txt"
    touch "$STAGED/.PREPROCESSED"
  fi
  N_L=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' 2>/dev/null | wc -l)
  # MoGe surgical depth (single call; global median over the whole snippet)
  mkdir -p "$STAGED/_moge_in" "$STAGED/_moge_npy" "$STAGED/depth.tmp"
  for f in "$STAGED"/video_frames/*l.png; do fid=$(basename "$f" l.png); ln -sf "$f" "$STAGED/_moge_in/${fid}-left.png"; done
  say "CRCD 1b $NAME: MoGe --target_median $TARGET_MEDIAN ($N_L frames)"
  "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$STAGED/_moge_in" --out "$STAGED/_moge_npy" \
    --temporal_window 1 --depth_scale 10000 --target_median "$TARGET_MEDIAN" --max_depth_m 5.0 2>&1 | tee -a "$LOG" || { say "  FATAL MoGe $NAME"; continue; }
  "$PYSYS" - "$STAGED" <<'PY'
import sys, glob, os, numpy as np, cv2
st = sys.argv[1]; files = sorted(glob.glob(f'{st}/_moge_npy/*-left_depth.npy')); n = 0
for p in files:
    fid = os.path.basename(p).split('-')[0]
    d = np.load(p).astype(np.float32)  # already surgical_m * 10000
    cv2.imwrite(f'{st}/depth.tmp/{fid}.png', np.clip(d, 0, 65535).astype(np.uint16)); n += 1
print(f'CRCD npy->png: {n}')
PY
  rm -rf "$STAGED/depth"; mv "$STAGED/depth.tmp" "$STAGED/depth"; touch "$STAGED/depth/.SURGDONE"
  rm -rf "$STAGED/_moge_in" "$STAGED/_moge_npy"
  say "  $NAME depth: $(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l) PNGs"
done

# =====================================================================
# PHASE 2 — ALL DDS-SLAM (dds_env / torch 1.10)
# =====================================================================
[ -f /tmp/dds_env/bin/activate ] || { say "FATAL: /tmp/dds_env missing (colab_exact_env.sh --skip-data)"; exit 1; }
source /tmp/dds_env/bin/activate
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
python -c "import torch,tinycudann,marching_cubes;print('env OK',torch.__version__,'cuda',torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { say "FATAL: dds_env import failed"; exit 1; }
python -c "import lpips" 2>/dev/null || pip install -q lpips 2>&1 | tail -2 | tee -a "$LOG" || true

# --- 2a. StereoMIS surgical SLAM (+ Drive save incl. est_c2w_data.txt) ---
SNAME=DDS-SLAM_stereomis-p2-1-moge-surgical_${DATE}; SDIR=$STE_OUT/$SNAME; mkdir -p "$SDIR"
if [ -f "$SDIR/.DONE" ]; then say "StereoMIS surgical done -> skip"; else
  NDEP=$(ls "$STEREO_SURG_DIR"/*.png 2>/dev/null | wc -l)
  if [ "$TNIMG" -gt 0 ] && [ "$NDEP" -ge "$TNIMG" ]; then
    rm -rf /content/p2_1_local/depth; ln -sf "$STEREO_SURG_DIR" /content/p2_1_local/depth
    CFG=configs/StereoMIS/p2_1_moge_surgical_${DATE}.yaml
    say "StereoMIS surgical SLAM (4000 frames — long pole)"
    python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN: StereoMIS SLAM crashed"
    OUT=$(python -c "from config import load_config;print(load_config('$CFG')['data']['output'])")
    CKPT=$(ls -t "$REPO/$OUT/demo"/checkpoint*.pt 2>/dev/null | head -1)
    if [ -n "$CKPT" ]; then
      cp "$CKPT" "$STEREO_CKPT_DIR/$SNAME.pt"; say "  ckpt -> $STEREO_CKPT_DIR/$SNAME.pt"
      # *** Drive-save FIX: est_c2w_data.txt + output_relative.txt (the 2026-06-08 miss) ***
      cp "$REPO/$OUT/demo/est_c2w_data.txt"   "$SDIR/" 2>/dev/null || say "  WARN: no est_c2w_data.txt"
      cp "$REPO/$OUT/demo/output_relative.txt" "$SDIR/" 2>/dev/null || true
      cp "$REPO/$OUT/demo/output.txt" "$SDIR/ate_output.txt" 2>/dev/null || true
      cp "$REPO/$OUT/demo"/pose_*.png "$SDIR/" 2>/dev/null || true
      cp /content/p2_1_local/groundtruth.txt "$SDIR/groundtruth.txt" 2>/dev/null || true
      python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
        --output_dir "$SDIR/rendered_all" --save_depth --save_gt 2>&1 | tee -a "$LOG" || say "  WARN: StereoMIS render"
      touch "$SDIR/.DONE"
    else say "  FATAL: no StereoMIS ckpt (no .DONE -> retries)"; fi
  else say "  FATAL: StereoMIS surgical depth incomplete ($NDEP/$TNIMG) -> SKIP"; fi
fi

# --- 2b. CRCD surgical SLAM + eval + Sim3 + ship (incl. est_c2w via demo/) ---
for ROW in "${CRCD_SNIPPETS[@]}"; do
  read -r NAME EP SID CFG FRAMES <<< "$ROW"
  STAGED=$REPO/data/CRCD/${NAME}; OUTPUT=$REPO/output/CRCD/${NAME}_surgical_${DATE}; DST=$CRCD_OUT/$NAME; mkdir -p "$DST"
  [ -f "$DST/.DONE" ] && { say "CRCD $NAME shipped -> skip"; continue; }
  [ -f "$STAGED/depth/.SURGDONE" ] || { say "CRCD $NAME: no surgical depth -> skip"; continue; }
  say "CRCD SLAM $NAME (config $CFG)"
  python ddsslam.py --config "configs/CRCD/${CFG}.yaml" 2>&1 | tee -a "$LOG" || { say "  WARN: $NAME SLAM crashed"; continue; }
  # render eval
  python Addons/eval/eval_rendering.py --gt_dir "$STAGED/video_frames" --render_dir "$OUTPUT" \
    --name "$NAME" --output_csv "$DST/render_eval.csv" --summary_csv "$CRCD_OUT/_render_summary.csv" \
    --sequence "CRCD (${NAME})" 2>&1 | tee "$DST/render_eval.txt" || say "  WARN: render eval $NAME"
  # trajectory metrics (raw / SE3 / Sim3 / path-ratio / Pearson)
  python - "$STAGED/groundtruth.txt" "$OUTPUT/demo/est_c2w_data.txt" "$NAME" > "$DST/summary.txt" 2>&1 <<'PY'
import os, sys, numpy as np
GT, EST, NAME = sys.argv[1], sys.argv[2], sys.argv[3]
def parse_est(p):
    P=[]
    if os.path.isfile(p):
        for l in open(p):
            v=l.split()
            if v and not v[0].startswith('#') and len(v)>=12: P.append(np.array(list(map(float,v[:12]))).reshape(3,4)[:3,3])
    return np.array(P)
def parse_tum(p):
    P=[]
    for l in open(p):
        v=l.split()
        if v and not v[0].startswith('#') and len(v)>=8: P.append([float(v[1]),float(v[2]),float(v[3])])
    return np.array(P)
def horn(m,d,ws=False):
    mc=m.mean(0);dc=d.mean(0);mm=m-mc;dd=d-dc;H=mm.T@dd;U,S,Vt=np.linalg.svd(H)
    ds=np.sign(np.linalg.det(Vt.T@U.T));D=np.diag([1,1,ds]);R=Vt.T@D@U.T
    s=(S*np.array([1,1,ds])).sum()/(mm*mm).sum() if ws else 1.0;t=dc-s*R@mc
    return (s*(R@m.T)).T+t,s
g=parse_tum(GT);e=parse_est(EST);print(f'=== {NAME} ===  GT {len(g)} est {len(e)}')
if len(e)>=10:
    n=min(len(e),len(g));ee,gg=e[:n],g[:n]
    raw=np.linalg.norm(ee-gg,axis=1)*1000
    a3,_=horn(ee,gg,False);e3=np.linalg.norm(a3-gg,axis=1)*1000
    a3s,s=horn(ee,gg,True);e3s=np.linalg.norm(a3s-gg,axis=1)*1000
    ep=np.linalg.norm(np.diff(ee,axis=0),axis=1).sum();gp=np.linalg.norm(np.diff(gg,axis=0),axis=1).sum()
    pe=[np.corrcoef(a3s[:,k],gg[:,k])[0,1] for k in range(3)]
    print(f'  raw RMSE {np.sqrt((raw**2).mean()):.2f} mm | SE3 {e3.mean():.3f} | Sim3 {e3s.mean():.3f} (s={s:.4f})')
    print(f'  est/GT path {ep*1000:.2f}/{gp*1000:.2f} -> {ep/max(gp,1e-12):.2f} | Pearson {tuple(round(p,3) for p in pe)}')
PY
  cat "$DST/summary.txt" | tee -a "$LOG"
  # ship payload (demo/ holds est_c2w_data.txt; renders at OUTPUT root)
  SHIP=(demo); [ -d "$OUTPUT/depth" ] && SHIP+=(depth)
  if ls "$OUTPUT"/*.jpg >/dev/null 2>&1; then mkdir -p "$OUTPUT/renders_rgb"; mv "$OUTPUT"/*.jpg "$OUTPUT/renders_rgb/" 2>/dev/null||true; SHIP+=(renders_rgb); fi
  tar czf "$DST/payload.tgz.partial" -C "$OUTPUT" "${SHIP[@]}" 2>/dev/null && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  cp "$STAGED/groundtruth.txt" "$DST/groundtruth.txt" 2>/dev/null || true
  touch "$DST/.DONE"; say "  $NAME shipped"
done

say "=== SURGICAL overnight done $(date -Iseconds) ==="
say "StereoMIS: $SDIR/{est_c2w_data.txt,ate_output.txt,rendered_all/}  ckpt $STEREO_CKPT_DIR/$SNAME.pt"
say "CRCD     : $CRCD_OUT/<NAME>/{payload.tgz,summary.txt,render_eval.csv}  (Sim3 in summary.txt)"
