#!/bin/bash
# ============================================================================
# DDS-SLAM CRCD 4-snippet preliminary batch — paperfaith_lrfix configs.
#
# Snippets (smallest-first order for resilience):
#   F_3/007 (300 frames)  -> C_1/001 (360 frames)  -> C_2/001 (730 frames)
#     -> F_1/002 (1287 frames)
#
# Per snippet:
#   Phase 1   stage raw from Drive (rgb/, rgbright/, semantic_instance/, GT, intrinsics)
#   Phase 1b  preprocess (rectify via ECM_STEREO_L2R pickle) -> data/CRCD/<S>/
#   Phase 1.5 MoGe-2 depth gen -> data/CRCD/<S>/depth/
#   Phase 1.6 stereo anchor calibration (SGBM on frame 0)
#   Phase 2   GT motion profile sanity (sentinel-flagged, non-blocking)
#   Phase 3   SLAM run -> output/CRCD/<S>_paperfaith_lrfix/
#   Phase 5   PSNR/SSIM/LPIPS render eval
#   Phase 6   extended trajectory metrics + ship to Drive
#
# After all 4: Phase 7 combined summary table for DDS-SLAM vs SNI-SLAM comparison.
#
# Sentinel-gated per snippet per phase; resumable if Colab disconnects.
# Wall on A100: ~3-4 hr total.  On T4: ~16-20 hr.
# ============================================================================
set -euo pipefail
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_crcd_4snippets_${DATE}
mkdir -p "$DRIVE_ROOT"
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== runbook start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }

DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl

# Snippet table: variant_name episode snippet_id slam_config frames
# Order: smallest-first for resilience.
SNIPPETS=(
  "F3_007  F_3  007  f3_007_paperfaith_lrfix  300"
  "C1_001  C_1  001  c1_001_paperfaith_lrfix  360"
  "C2_001  C_2  001  c2_001_paperfaith_lrfix  730"
  "F1_002  F_1  002  f1_002_paperfaith_lrfix  1287"
)

# ----------------------------------------------------------------------------
# Per-item Drive copy with cp/rsync fallback (from StereoMIS runbook lesson)
# ----------------------------------------------------------------------------
copy_item() {
  local SRC=$1 DST=$2 LABEL=$3
  if [ ! -e "$SRC" ]; then
    echo "FATAL: missing on Drive: $SRC"
    return 1
  fi
  if [ -e "$DST" ]; then
    echo "  $LABEL already at destination -- skip"
    return 0
  fi
  local SIZE
  SIZE=$(du -sh "$SRC" 2>/dev/null | cut -f1)
  echo "  copying $LABEL ($SIZE) ..."
  if cp -r "$SRC" "$DST"; then
    return 0
  fi
  echo "  cp -r failed for $LABEL, retrying via rsync --partial..."
  rsync -a --partial "$SRC" "$DST" \
    || { echo "FATAL: rsync also failed for $LABEL"; return 1; }
}

# ----------------------------------------------------------------------------
# Env activation
# ----------------------------------------------------------------------------
activate_dds_env() {
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    echo "modern stack missing -- rebuild"
    bash /content/DDS-SLAM/Addons/env/colab_setup.sh --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { echo "env check FAIL"; exit 1; }
  if ! python -c "import lpips" 2>/dev/null; then
    pip install -q lpips 2>&1 | tail -3 || echo "  WARN: lpips install failed"
  fi
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

ensure_moge() {
  if ! python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null; then
    echo "  installing MoGe-2..."
    pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -5
    python -c 'from moge.model.v2 import MoGeModel' \
      || { echo "FATAL: MoGe-2 not importable"; return 1; }
  fi
}

# ============================================================================
# PHASE 0 -- env + repo sync + GPU report
# ============================================================================
phase 0 "env check + repo sync"
[ -d /content/drive/MyDrive ] || { echo "Drive not mounted -- abort"; exit 1; }
[ -f "$CALIB_PKL" ] || { echo "FATAL: calib pickle missing at $CALIB_PKL"; exit 3; }
cd /content/DDS-SLAM
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull"
else
  git fetch && git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD) \
    || { echo "non-FF on remote -- abort"; exit 1; }
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU"
if [[ ! "$GPU" =~ A100 ]]; then
  echo "  WARN: non-A100 GPU. T4 wall ~16-20 hr.  Continuing anyway."
fi
activate_dds_env

# ============================================================================
# PER-SNIPPET LOOP
# ============================================================================
for ROW in "${SNIPPETS[@]}"; do
  read -r NAME EP SID CONFIG FRAMES <<< "$ROW"
  echo ""
  echo "############################################################"
  echo "## SNIPPET $NAME  ($EP/snippet_$SID, $FRAMES frames)"
  echo "############################################################"

  RAW=/content/crcd_raw/${EP}_snippet_${SID}
  STAGED=/content/DDS-SLAM/data/CRCD/${NAME}
  OUTPUT=/content/DDS-SLAM/output/CRCD/${NAME}_paperfaith_lrfix
  DRIVE_DST=$DRIVE_ROOT/$NAME

  mkdir -p "$DRIVE_DST"

  if done_marker "$DRIVE_DST"; then
    echo "  $NAME already shipped to Drive -- skip whole snippet"
    continue
  fi

  # --------------------------------------------------------------------------
  # PHASE 1 -- stage raw CRCD-Published from Drive
  # Tarball-first (per StereoMIS lesson: per-item cp via Drive FUSE takes
  # ~78 min for 8 GB; single sequential tar read is ~3-5 min).
  # --------------------------------------------------------------------------
  phase "1.$NAME" "stage raw CRCD-Published $EP/snippet_$SID from Drive"
  DRIVE_TAR=$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar
  if [ -f "$RAW/.STAGED" ]; then
    echo "  raw already staged at $RAW"
  elif [ -f "$DRIVE_TAR" ]; then
    echo "  using tarball: $DRIVE_TAR ($(du -h "$DRIVE_TAR" | cut -f1))"
    mkdir -p "$RAW"
    T0=$(date +%s)
    if ! tar xf "$DRIVE_TAR" -C "$RAW"; then
      echo "FATAL: tarball extraction failed for $NAME"
      rm -rf "$RAW"
      exit 3
    fi
    echo "  tarball extracted in $(( ($(date +%s) - T0) / 60 )) min"
    touch "$RAW/.STAGED"
  else
    DRIVE_SRC=$DRIVE_CRCD/$EP/snippet_$SID
    [ -d "$DRIVE_SRC" ] || { echo "FATAL: missing $DRIVE_SRC"; exit 3; }
    echo "  WARN: no tarball at $DRIVE_TAR; using slow per-item cp"
    echo "  TIP: build one-time tarball with:"
    echo "       tar cf $DRIVE_TAR -C $DRIVE_SRC rgb rgbright semantic_instance groundtruth.txt intrinsics.yaml"
    mkdir -p "$RAW"
    copy_item "$DRIVE_SRC/rgb"               "$RAW/rgb"               "rgb"               || exit 3
    copy_item "$DRIVE_SRC/rgbright"          "$RAW/rgbright"          "rgbright"          || exit 3
    copy_item "$DRIVE_SRC/semantic_instance" "$RAW/semantic_instance" "semantic_instance" || exit 3
    copy_item "$DRIVE_SRC/groundtruth.txt"   "$RAW/groundtruth.txt"   "groundtruth.txt"   || exit 3
    copy_item "$DRIVE_SRC/intrinsics.yaml"   "$RAW/intrinsics.yaml"   "intrinsics.yaml"   || exit 3
    touch "$RAW/.STAGED"
  fi
  N_RGB=$(ls "$RAW/rgb" 2>/dev/null | wc -l)
  echo "  raw rgb count: $N_RGB"
  [ "$N_RGB" -ge "$FRAMES" ] || { echo "FATAL: raw frame count $N_RGB < expected $FRAMES"; exit 3; }

  # --------------------------------------------------------------------------
  # PHASE 1b -- preprocess: rectify + rename + copy GT
  # --------------------------------------------------------------------------
  phase "1b.$NAME" "preprocess (rectify with ECM_STEREO_L2R pickle)"
  if [ -f "$STAGED/.PREPROCESSED" ]; then
    echo "  preprocess already complete"
  else
    rm -rf "$STAGED"
    mkdir -p "$STAGED"
    cd /content/DDS-SLAM
    python Addons/preprocess/preprocess_crcd_published.py \
      --snippet_dir "$RAW" \
      --calib_pkl   "$CALIB_PKL" \
      --output_dir  "$STAGED" \
      || { echo "FATAL: preprocess failed for $NAME"; exit 4; }
    # Copy GT (preprocess script may or may not — verify)
    if [ ! -f "$STAGED/groundtruth.txt" ]; then
      cp "$RAW/groundtruth.txt" "$STAGED/groundtruth.txt"
    fi
    touch "$STAGED/.PREPROCESSED"
  fi
  N_L=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' 2>/dev/null | wc -l)
  N_R=$(find "$STAGED/video_frames" -maxdepth 1 -name '*r.png' 2>/dev/null | wc -l)
  N_M=$(find "$STAGED/masks" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
  N_G=$(grep -cv '^#' "$STAGED/groundtruth.txt" 2>/dev/null || echo 0)
  echo "  rectified: left=$N_L  right=$N_R  masks=$N_M  GT=$N_G"
  [ -f "$STAGED/rectified_calib.txt" ] || { echo "FATAL: rectified_calib.txt missing"; exit 4; }

  # --------------------------------------------------------------------------
  # PHASE 1.5 -- MoGe-2 depth generation
  # --------------------------------------------------------------------------
  phase "1.5.$NAME" "MoGe-2 depth gen (~${FRAMES} frames at 1280x720)"
  EXPECTED=$N_L
  ACTUAL_DEPTH=0
  [ -d "$STAGED/depth" ] && ACTUAL_DEPTH=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
  if [ -f "$STAGED/depth/.DONE" ] && [ "$ACTUAL_DEPTH" -ge "$EXPECTED" ]; then
    echo "  depth/ complete ($ACTUAL_DEPTH/$EXPECTED) -- skip"
  else
    ensure_moge || exit 5
    cd "$STAGED"
    mkdir -p _moge_in _moge_npy depth.tmp
    for f in video_frames/*l.png; do
      fid=$(basename "$f" l.png)
      [ -L "_moge_in/${fid}-left.png" ] || ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"
    done
    echo "  symlinks: $(ls _moge_in/ | wc -l)"
    python /content/DDS-SLAM/Addons/depth/generate_depth_moge.py \
      --rgb _moge_in --out _moge_npy \
      --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 \
      || { echo "FATAL: MoGe gen failed"; exit 5; }
    python - <<'PYEOF'
import numpy as np, cv2, glob, os
n_in = sorted(glob.glob('_moge_npy/*-left_depth.npy'))
written = 0
for p in n_in:
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    d = np.load(p).astype(np.float32)
    tmp = f'depth.tmp/.{fid}.tmp.png'
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, out)
    written += 1
print(f'npy->png wrote {written} (of {len(n_in)} npy files)')
PYEOF
    PNG=$(ls depth.tmp/*.png 2>/dev/null | wc -l)
    NPY=$(ls _moge_npy/*-left_depth.npy 2>/dev/null | wc -l)
    if [ "$PNG" -ne "$NPY" ] || [ "$PNG" -lt "$EXPECTED" ]; then
      echo "FATAL: depth count mismatch png=$PNG npy=$NPY expected>=$EXPECTED"
      cd /content/DDS-SLAM
      exit 5
    fi
    rm -rf depth && mv depth.tmp depth
    sync; touch depth/.DONE; sync
    rm -rf _moge_in _moge_npy
    cd /content/DDS-SLAM
  fi
  N_D=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
  echo "  depth ready: $N_D files"

  # --------------------------------------------------------------------------
  # PHASE 1.6 -- frame-0 stereo anchor calibration
  # --------------------------------------------------------------------------
  phase "1.6.$NAME" "stereo anchor calibration"
  if [ -f "$STAGED/.sc_factor" ]; then
    SC_FACTOR=$(cat "$STAGED/.sc_factor")
    echo "  cached: sc_factor = $SC_FACTOR"
  else
    python - <<PYEOF
import cv2, numpy as np, os, sys, glob
STAGED = '$STAGED'

# Read rectified calibration written by preprocess
calib = {}
with open(f'{STAGED}/rectified_calib.txt') as f:
    for line in f:
        k, v = line.strip().split()
        calib[k] = float(v)
baseline_m   = calib['baseline_m']
fx_rectified = calib['fx']
print(f'  baseline    : {baseline_m*1000:.4f} mm')
print(f'  fx rectified: {fx_rectified:.4f} px')

# First frame from preprocess output (= first frame DDS-SLAM consumes)
all_left = sorted(glob.glob(f'{STAGED}/video_frames/*l.png'))
if not all_left:
    print('FATAL: no rectified frames', file=sys.stderr); sys.exit(1)
left_path = all_left[0]
fid = os.path.basename(left_path).replace('l.png', '')
right_path = f'{STAGED}/video_frames/{fid}r.png'
moge_path  = f'{STAGED}/depth/{fid}.png'
print(f'  frame 0     : {fid}')

left  = cv2.imread(left_path,  cv2.IMREAD_GRAYSCALE)
right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
moge_png = cv2.imread(moge_path, cv2.IMREAD_UNCHANGED)
if left is None or right is None or moge_png is None:
    print(f'FATAL: failed to read frame 0 left/right/moge', file=sys.stderr); sys.exit(1)
print(f'  shape       : {left.shape}')
moge_depth_m = moge_png.astype(np.float32) / 10000.0
mvalid = moge_depth_m > 0.01
print(f'  MoGe depth  : median={np.median(moge_depth_m[mvalid]):.3f} m, '
      f'p10={np.percentile(moge_depth_m[mvalid], 10):.3f}, '
      f'p90={np.percentile(moge_depth_m[mvalid], 90):.3f}')

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=7,
    P1=8*1*7**2, P2=32*1*7**2,
    disp12MaxDiff=1, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
disp = sgbm.compute(left, right).astype(np.float32) / 16.0
valid_stereo = disp > 0.5
print(f'  stereo valid: {valid_stereo.sum()}/{disp.size} ({100*valid_stereo.mean():.1f}%)')
if valid_stereo.sum() < 1000:
    print('FATAL: too few valid stereo pixels', file=sys.stderr); sys.exit(2)

stereo_depth_m = np.zeros_like(disp)
stereo_depth_m[valid_stereo] = baseline_m * fx_rectified / disp[valid_stereo]
valid_joint = valid_stereo & (stereo_depth_m > 0.05) & (stereo_depth_m < 3.0) & mvalid
n_joint = int(valid_joint.sum())
print(f'  joint valid : {n_joint} pixels')
if n_joint < 500:
    print('FATAL: too few joint-valid pixels', file=sys.stderr); sys.exit(2)

s_d = stereo_depth_m[valid_joint]
m_d = moge_depth_m[valid_joint]
ratios = s_d / m_d
sc_factor = float(np.median(ratios))
print(f'  stereo depth median (valid): {np.median(s_d):.3f} m')
print(f'  MoGe   depth median (joint): {np.median(m_d):.3f} m')
print(f'  ratio p25/median/p75       : {np.percentile(ratios,25):.3f} / {sc_factor:.3f} / {np.percentile(ratios,75):.3f}')
print(f'')
print(f'  sc_factor = stereo / MoGe  = {sc_factor:.4f}')
with open(f'{STAGED}/.sc_factor', 'w') as fh:
    fh.write(f'{sc_factor:.6f}\n')
PYEOF
    if [ ! -f "$STAGED/.sc_factor" ]; then
      echo "FATAL: stereo anchor failed for $NAME"
      exit 4
    fi
    SC_FACTOR=$(cat "$STAGED/.sc_factor")
  fi

  APPLY=$(python -c "
import math
s = $SC_FACTOR
print('1' if abs(math.log(s)) > 0.1 else '0')")
  if [ "$APPLY" = "1" ]; then
    echo "  sc_factor $SC_FACTOR is >10% off 1.0 -- patching configs/CRCD/${CONFIG}.yaml"
    Y="/content/DDS-SLAM/configs/CRCD/${CONFIG}.yaml"
    python - <<PYEOF
path = '$Y'
with open(path) as f: text = f.read()
lines = text.splitlines()
out = []
in_data = False
inserted = False
for line in lines:
    if line.startswith('data:'):
        in_data = True; out.append(line); continue
    if in_data and not inserted and line and not line.startswith(' '):
        out.append(f'  sc_factor: $SC_FACTOR   # stereo-anchor calibrated, Phase 1.6')
        inserted = True; in_data = False
    if 'sc_factor:' in line and in_data:
        out.append(f'  sc_factor: $SC_FACTOR   # stereo-anchor calibrated, Phase 1.6')
        inserted = True; continue
    out.append(line)
if not inserted:
    out.append(f'  sc_factor: $SC_FACTOR   # stereo-anchor calibrated, Phase 1.6')
with open(path, 'w') as f: f.write('\n'.join(out) + '\n')
print(f'  patched {path}')
PYEOF
    INCONSISTENCY=$(python -c "print(round(abs(1-$SC_FACTOR)*100,1))")
    echo "  WARN: F7 latent bug — only trunc fully scales (range_d/near/far/depth_trunc unscaled)."
    echo "        At sc_factor=$SC_FACTOR the inconsistency is ${INCONSISTENCY}% per threshold."
  else
    echo "  sc_factor $SC_FACTOR within 10% of 1.0 -- MoGe is near-metric on this snippet, no rescale needed"
  fi

  # --------------------------------------------------------------------------
  # PHASE 2 -- GT motion profile sanity (sentinel-flagged, non-blocking)
  # --------------------------------------------------------------------------
  phase "2.$NAME" "GT motion profile sanity"
  python - <<PYEOF
import numpy as np
GT = '$STAGED/groundtruth.txt'
rows=[]
with open(GT) as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        v=line.split()
        if len(v) >= 8:
            rows.append([float(v[1]), float(v[2]), float(v[3])])
xyz = np.array(rows)
d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
print(f'  GT poses    : {len(xyz)}')
print(f'  extent (mm) : x={(xyz[:,0].max()-xyz[:,0].min())*1000:.2f}  '
      f'y={(xyz[:,1].max()-xyz[:,1].min())*1000:.2f}  '
      f'z={(xyz[:,2].max()-xyz[:,2].min())*1000:.2f}')
print(f'  total path  : {d.sum():.2f} mm')
print(f'  per-frame   : median={np.median(d):.4f}  mean={d.mean():.4f}  max={d.max():.4f} mm/f')
active = d > 0.001
print(f'  active frac : {100*active.mean():.1f}%  ({active.sum()}/{len(d)})')
if active.sum() > 0:
    am = d[active]
    am_median = np.median(am)
    am_max = am.max()
    print(f'  active median: {am_median:.4f} mm/f  ({am_median/1.0:.2f}x noise floor)')
    print(f'  active max   : {am_max:.4f} mm/f  ({am_max/1.0:.2f}x noise floor)')
    if am_median < 0.5:
        print(f'  SENTINEL: active median < 0.5 mm/f -- sub-SNR (tracker quality untrustable)')
    if (1 - active.mean()) > 0.3:
        print(f'  SENTINEL: static fraction > 30% -- const_speed=True will amplify drift')
PYEOF

  # --------------------------------------------------------------------------
  # PHASE 3 -- SLAM
  # --------------------------------------------------------------------------
  phase "3.$NAME" "SLAM (config: $CONFIG)"
  if [ -d "$OUTPUT/demo" ] && [ -f "$OUTPUT/demo/est_c2w_data.txt" ]; then
    LINES=$(wc -l < "$OUTPUT/demo/est_c2w_data.txt")
    if [ "$LINES" -ge "$FRAMES" ]; then
      echo "  SLAM output already present ($LINES poses >= $FRAMES) -- skip"
    fi
  fi
  if [ ! -f "$OUTPUT/demo/est_c2w_data.txt" ] || [ "$(wc -l < "$OUTPUT/demo/est_c2w_data.txt" 2>/dev/null || echo 0)" -lt "$FRAMES" ]; then
    T0=$(date +%s)
    cd /content/DDS-SLAM
    if ! python ddsslam.py --config "configs/CRCD/${CONFIG}.yaml"; then
      echo "  $NAME SLAM crashed; check log.  Continuing to next snippet."
      echo "FAILED_SLAM_$NAME" >> "$DRIVE_ROOT/_failures.log"
      continue
    fi
    echo "  $NAME SLAM elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  fi

  # --------------------------------------------------------------------------
  # PHASE 5 -- render-quality eval
  # --------------------------------------------------------------------------
  phase "5.$NAME" "PSNR/SSIM/LPIPS eval"
  EVAL_OUT=$DRIVE_DST/render_eval.txt
  EVAL_CSV=$DRIVE_DST/render_eval.csv
  python Addons/eval/eval_rendering.py \
    --gt_dir "$STAGED/video_frames" \
    --render_dir "$OUTPUT" \
    --name "$NAME" \
    --output_csv "$EVAL_CSV" \
    --summary_csv "$DRIVE_ROOT/_render_summary.csv" \
    --sequence "CRCD (${NAME})" \
    2>&1 | tee "$EVAL_OUT" || echo "  WARN: render eval failed (likely no rendered frames)"

  # --------------------------------------------------------------------------
  # PHASE 6 -- extended trajectory metrics + ship
  # --------------------------------------------------------------------------
  phase "6.$NAME" "trajectory metrics + ship to Drive"
  SUMMARY=$DRIVE_DST/summary.txt
  python - > "$SUMMARY" 2>&1 <<PYEOF
import os, csv, numpy as np

GT = '$STAGED/groundtruth.txt'
EST = '$OUTPUT/demo/est_c2w_data.txt'
NAME = '$NAME'
FRAMES = $FRAMES

def parse_est(p):
    poses=[]
    if not os.path.isfile(p): return np.zeros((0,3))
    with open(p) as f:
        for line in f:
            v=line.strip().split()
            if not v or v[0].startswith('#'): continue
            if len(v) >= 12:
                poses.append(np.array(list(map(float,v[:12]))).reshape(3,4)[:3,3])
    return np.array(poses)

def parse_tum(p):
    poses=[]
    with open(p) as f:
        for line in f:
            v=line.strip().split()
            if not v or v[0].startswith('#'): continue
            if len(v) >= 8:
                poses.append([float(v[1]),float(v[2]),float(v[3])])
    return np.array(poses)

def horn(m, d, with_scale=False):
    mc=m.mean(0); dc=d.mean(0); mm=m-mc; dd=d-dc
    H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    ds=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,ds]); R=Vt.T@D@U.T
    s=(S*np.array([1,1,ds])).sum()/(mm*mm).sum() if with_scale else 1.0
    t=dc-s*R@mc
    return (s*(R@m.T)).T+t, s

g = parse_tum(GT)
e = parse_est(EST)
print(f'=== {NAME} summary ===')
print(f'  GT poses: {len(g)}  est poses: {len(e)}')
if len(e) < 10:
    print('  ERROR: too few est poses to evaluate')
else:
    n = min(len(e), len(g)); ee, gg = e[:n], g[:n]
    raw = np.linalg.norm(ee - gg, axis=1) * 1000
    ase3,_  = horn(ee, gg, False); ese3  = np.linalg.norm(ase3-gg, axis=1)*1000
    asim3,s = horn(ee, gg, True);  esim3 = np.linalg.norm(asim3-gg, axis=1)*1000
    epath = np.linalg.norm(np.diff(ee,axis=0), axis=1).sum()
    gpath = np.linalg.norm(np.diff(gg,axis=0), axis=1).sum()
    ratio = epath / max(gpath, 1e-12)
    pearson = []
    for axis in range(3):
        try:
            r = np.corrcoef(asim3[:, axis], gg[:, axis])[0, 1]
            pearson.append(r if np.isfinite(r) else 0.0)
        except Exception:
            pearson.append(0.0)
    print(f'  raw RMSE  : {np.sqrt((raw**2).mean()):.2f} mm')
    print(f'  SE3 ATE   : {ese3.mean():.3f} mm')
    print(f'  Sim3 ATE  : {esim3.mean():.3f} mm  (scale s={s:.4f})')
    print(f'  est_path  : {epath*1000:.2f} mm  /  GT_path: {gpath*1000:.2f} mm  -> ratio {ratio:.2f}')
    print(f'  Pearson xyz: ({pearson[0]:.3f}, {pearson[1]:.3f}, {pearson[2]:.3f})')
    if ratio > 3:
        print(f'  WARN: est_path/GT_path > 3 — tracker is jittering, not tracking')
    if max(abs(p) for p in pearson) < 0.1:
        print(f'  WARN: all |Pearson| < 0.1 — trajectory shape uncorrelated with GT')
PYEOF
  cat "$SUMMARY"
  # Ship: tar key outputs to drive.
  # IMPORTANT: ddsslam.py:806 saves rendered RGB as $OUTPUT/{frame:04d}.jpg
  # at the ROOT of the output dir, and depth at $OUTPUT/depth/{frame:04d}.png.
  # Previously we shipped only `demo/` + `ckpts/`, leaving renders behind.
  # User-validated 2026-06-05 on the overnight run: payload.tgz had 0
  # rendered images on all 3 completed snippets despite render_freq=1.
  # Fix: ship the renders explicitly.
  if [ ! -f "$DRIVE_DST/payload.tgz" ] && [ -d "$OUTPUT/demo" ]; then
    echo "  shipping payload to $DRIVE_DST..."
    # Build list of directories/items to include
    SHIP_ITEMS=(demo)
    [ -d "$OUTPUT/ckpts" ] && SHIP_ITEMS+=(ckpts)
    [ -d "$OUTPUT/depth" ] && SHIP_ITEMS+=(depth)
    # Also include all top-level .jpg renders if any exist
    if ls "$OUTPUT"/*.jpg >/dev/null 2>&1; then
      mkdir -p "$OUTPUT/renders_rgb"
      mv "$OUTPUT"/*.jpg "$OUTPUT/renders_rgb/" 2>/dev/null || true
      SHIP_ITEMS+=(renders_rgb)
    fi
    tar czf "$DRIVE_DST/payload.tgz.partial" \
      -C "$OUTPUT" "${SHIP_ITEMS[@]}" 2>/dev/null || true
    mv "$DRIVE_DST/payload.tgz.partial" "$DRIVE_DST/payload.tgz" 2>/dev/null || true
    echo "  shipped items: ${SHIP_ITEMS[*]}"
  fi
  mark_done "$DRIVE_DST"
  echo "## $NAME complete"
done

# ============================================================================
# PHASE 7 -- combined 4-snippet summary
# ============================================================================
phase 7 "combined summary"
COMBINED=$DRIVE_ROOT/COMBINED_SUMMARY.txt
{
  echo "=== DDS-SLAM CRCD 4-snippet preliminary ($(date -Iseconds)) ==="
  echo ""
  printf '%-10s %7s %9s %9s %9s %8s %12s %18s\n' \
    'snippet' 'frames' 'raw_mm' 'SE3_mm' 'Sim3_mm' 'sim3_s' 'est/GT_path' 'pearson_xyz'
  echo '---------------------------------------------------------------------------------------------------'
  for ROW in "${SNIPPETS[@]}"; do
    read -r NAME _ _ _ FRAMES <<< "$ROW"
    SFILE=$DRIVE_ROOT/$NAME/summary.txt
    if [ -f "$SFILE" ]; then
      grep -E 'raw RMSE|SE3 ATE|Sim3 ATE|est_path|Pearson' "$SFILE" | awk -v n="$NAME" -v f="$FRAMES" 'BEGIN{r="";se3="";si3="";s="";rt="";p=""}
        /raw RMSE/   { r = $4 }
        /SE3 ATE/    { se3 = $4 }
        /Sim3 ATE/   { si3 = $4; for(i=1;i<=NF;i++) if($i ~ /s=/) s=$i }
        /est_path/   { for(i=1;i<=NF;i++) if($i=="ratio") rt = $(i+1) }
        /Pearson xyz/{ p = $3" "$4" "$5 }
        END { printf "%-10s %7s %9s %9s %9s %8s %12s %18s\n", n, f, r, se3, si3, s, rt, p }'
    else
      printf '%-10s %7s   (no summary)\n' "$NAME" "$FRAMES"
    fi
  done
  echo ""
  echo "Per-snippet Drive payloads:"
  for ROW in "${SNIPPETS[@]}"; do
    read -r NAME _ _ _ _ <<< "$ROW"
    echo "  $DRIVE_ROOT/$NAME/payload.tgz"
  done
  echo ""
  echo "Interpretation hints:"
  echo "  - sim3_s should be near 1.0 if Phase 1.6 stereo anchor was applied"
  echo "  - est_path/GT_path ratio > 3 means tracker is jittering, not tracking"
  echo "  - Pearson per axis > 0.5 means trajectory shape correlates with GT"
  echo "  - Sub-SNR snippets (most CRCD) will have low Pearson regardless"
  echo "  - C_2/001 is the best-case snippet (29mm extent); others are sub-SNR"
} | tee "$COMBINED"
echo ""
echo "=== runbook done $(date -Iseconds) ==="
echo "Log:      $LOG"
echo "Summary:  $COMBINED"
