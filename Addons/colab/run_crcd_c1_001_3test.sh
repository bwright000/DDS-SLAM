#!/bin/bash
# ============================================================================
# DDS-SLAM CRCD C_1/001 — 3-config diagnostic sweep (~1-1.5 hr A100)
#
# Tests three hypotheses on the small C_1/001 snippet (360 frames) before
# committing to a config for the full 4-snippet sweep:
#
#   paper_faithful           — baseline: hash_size 19, default schedule
#   semsup_sched (Recipe A)  — under-training fix: iters=50, first_iters=1000,
#                              lr=1e-4, tracking iter=20
#   paper_faithful_depthx0p3 — depth scale fix: png_depth_scale 10000 -> 33333
#                              (equivalent to rescaling MoGe NPYs by 0.3)
#
# Paste into a Colab VS Code tunnel terminal. Six phases, resume-safe.
# Master log on Drive survives session death.
# ============================================================================
set -euo pipefail
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_crcd_c1_001_3test_${DATE}
mkdir -p "$DRIVE_ROOT/paper_faithful" "$DRIVE_ROOT/semsup_sched" "$DRIVE_ROOT/paper_faithful_depthx0p3"
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== runbook start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }

STAGED=/content/DDS-SLAM/data/CRCD/C1_001
DRIVE_SNIPPET=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
CALIB_PKL=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl

activate_dds_env() {
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    echo "modern stack missing -- full rebuild (~15 min)"
    bash /content/DDS-SLAM/Addons/env/colab_setup.sh --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { echo "env check FAIL"; exit 1; }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

ship_to_drive() {  # $1=src_dir  $2=dst_dir_on_drive
  local SRC=$1 DST=$2
  tar czf "$DST/payload.tgz.partial" -C "$SRC" .
  mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  mark_done "$DST"
}

# ============================================================================
# PHASE 0 -- env + repo sync + GPU gate
# ============================================================================
phase 0 "env check + repo sync"
[ -d /content/drive/MyDrive ] || { echo "Drive not mounted -- abort"; exit 1; }
cd /content/DDS-SLAM
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull, continuing with local edits"
else
  git fetch && git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD) \
    || { echo "non-FF on remote -- manual reconcile required, abort"; exit 1; }
fi
nvidia-smi -L
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "  GPU: $GPU"
if [[ ! "$GPU" =~ A100 ]]; then
  echo "WARN: non-A100 GPU ($GPU). Will run but expect ~3-5x slowdown."
fi
activate_dds_env

# ============================================================================
# PHASE 1 -- stage raw C_1/001 + preprocess to DDS-SLAM layout
# ============================================================================
phase 1 "C_1/001 preprocess"
if [ -f "$STAGED/.STAGED" ]; then
  echo "  staged dir complete -- skip preprocess"
else
  rm -rf "$STAGED"
  mkdir -p /content/crcd_raw "$(dirname "$STAGED")"
  if [ ! -d /content/crcd_raw/snippet_001 ]; then
    if [ -f "${DRIVE_SNIPPET}.tar" ]; then
      mkdir -p /content/crcd_raw/snippet.tmp
      tar xf "${DRIVE_SNIPPET}.tar" -C /content/crcd_raw/snippet.tmp
      if [ -d /content/crcd_raw/snippet.tmp/snippet_001 ]; then
        mv /content/crcd_raw/snippet.tmp/snippet_001 /content/crcd_raw/snippet_001
        rm -rf /content/crcd_raw/snippet.tmp
      else
        mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_001
      fi
    else
      cp -r "$DRIVE_SNIPPET" /content/crcd_raw/snippet.tmp
      mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_001
    fi
  fi
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir /content/crcd_raw/snippet_001 \
    --calib_pkl   "$CALIB_PKL" \
    --output_dir  "${STAGED}.tmp"
  NL=$(find "${STAGED}.tmp/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  NG=$(grep -cv '^#' "${STAGED}.tmp/groundtruth.txt" 2>/dev/null || echo 0)
  [ "$NL" -gt 100 ] && [ "$NG" -ge "$NL" ] || { echo "preprocess produced $NL frames / $NG gt rows -- abort"; exit 1; }
  mv "${STAGED}.tmp" "$STAGED"
  touch "$STAGED/.STAGED"
  rm -rf /content/crcd_raw
fi
echo "  staged: $(find $STAGED/video_frames -maxdepth 1 -name '*l.png' | wc -l) left frames"

# ============================================================================
# PHASE 2 -- MoGe-2 depth gen (reuse Drive cache if present + count matches)
# ============================================================================
phase 2 "MoGe-2 depth"
EXPECTED=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
ACTUAL=0
[ -d "$STAGED/depth" ] && ACTUAL=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)

DRIVE_DEPTH_DIR=$DRIVE_SNIPPET/depth
N_DRIVE_DEPTH=0
[ -d "$DRIVE_DEPTH_DIR" ] && N_DRIVE_DEPTH=$(find "$DRIVE_DEPTH_DIR" -maxdepth 1 -name '*.png' | wc -l)

if [ -f "$STAGED/depth/.DONE" ] && [ "$ACTUAL" -eq "$EXPECTED" ]; then
  echo "  depth/ complete ($ACTUAL/$EXPECTED) -- skip"
elif [ "$N_DRIVE_DEPTH" -eq "$EXPECTED" ]; then
  echo "  Drive depth cache hit ($N_DRIVE_DEPTH==$EXPECTED) -- copy + index-pair rename"
  mkdir -p "$STAGED/depth.tmp"
  python3 - <<PYEOF
import os, shutil
RAW_RGB = sorted(f for f in os.listdir('/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/rgb') if f.endswith('.png'))
DEPTH_FILES = sorted(f for f in os.listdir('${DRIVE_DEPTH_DIR}') if f.endswith('.png'))
assert len(RAW_RGB) == len(DEPTH_FILES), f"rgb={len(RAW_RGB)} depth={len(DEPTH_FILES)}"
for i, dname in enumerate(DEPTH_FILES):
    shutil.copy2(os.path.join('${DRIVE_DEPTH_DIR}', dname),
                 os.path.join('${STAGED}/depth.tmp', f'{i:06d}.png'))
print(f"copied {len(DEPTH_FILES)} depth PNGs from Drive cache")
PYEOF
  rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"
  sync; touch "$STAGED/depth/.DONE"; sync
else
  echo "  depth/ missing or count mismatch ($ACTUAL/$EXPECTED, drive=$N_DRIVE_DEPTH) -- regenerate"
  python3 -c 'import moge.model.v2' 2>/dev/null || \
    python3 -m pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub
  cd "$STAGED"
  mkdir -p _moge_in depth.tmp
  for f in video_frames/*l.png; do
    fid=$(basename "$f" l.png)
    [ -L "_moge_in/${fid}-left.png" ] || ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"
  done
  python3 /content/DDS-SLAM/Addons/depth/generate_depth_moge.py \
    --rgb _moge_in --out _moge_npy \
    --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0
  python3 - <<'PYEOF'
import numpy as np, cv2, glob, os
for p in sorted(glob.glob('_moge_npy/*-left_depth.npy')):
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    d = np.load(p).astype(np.float32)
    tmp = f'depth.tmp/.{fid}.tmp.png'
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, out)
print('npy_in', len(glob.glob('_moge_npy/*-left_depth.npy')),
      'png_out', len(glob.glob('depth.tmp/*.png')))
PYEOF
  NPY=$(find _moge_npy -maxdepth 1 -name '*-left_depth.npy' | wc -l)
  PNG=$(find depth.tmp -maxdepth 1 -name '*.png' | wc -l)
  [ "$PNG" -eq "$NPY" ] && [ "$PNG" -eq "$EXPECTED" ] \
    || { echo "depth count mismatch png=$PNG npy=$NPY expected=$EXPECTED -- abort"; exit 1; }
  rm -rf depth && mv depth.tmp depth
  sync; touch depth/.DONE; sync
  rm -rf _moge_in _moge_npy
  cd /content/DDS-SLAM
  # Sync depth back to Drive snippet under original frame_NNNNNN.png names so
  # tomorrow's SNI-SLAM runbook + future DDS-SLAM resumes can both reuse it.
  echo "  syncing depth back to Drive snippet"
  python3 - <<PYEOF
import os, shutil
STAGED = '${STAGED}'
DRIVE_SNIP = '${DRIVE_SNIPPET}'
DST = os.path.join(DRIVE_SNIP, 'depth')
os.makedirs(DST, exist_ok=True)
rgb_orig = sorted(f for f in os.listdir(os.path.join(DRIVE_SNIP, 'rgb')) if f.endswith('.png'))
copied = 0
for i, orig in enumerate(rgb_orig):
    src = os.path.join(STAGED, 'depth', f'{i:06d}.png')
    if not os.path.exists(src): continue
    dst = os.path.join(DST, orig)
    if not os.path.exists(dst):
        shutil.copy2(src, dst); copied += 1
print(f"Drive sync copied={copied}")
PYEOF
fi

# ============================================================================
# Helper: run one config and ship  ($1=cfg_name  $2=output_dir  $3=drive_dst)
# ============================================================================
run_and_ship() {
  local NAME=$1 OUT=$2 DST=$3
  if done_marker "$DST"; then
    echo "  $NAME already shipped -- skip"
    return 0
  fi
  echo "=== run $NAME ==="
  T0=$(date +%s)
  cd /content/DDS-SLAM
  python ddsslam.py --config "configs/CRCD/c1_001_${NAME}.yaml"
  echo "  $NAME elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  ship_to_drive "$OUT" "$DST"
  echo "  $NAME shipped -> $DST"
}

# ============================================================================
# PHASE 3 -- run paper_faithful (baseline / control, ~15-20 min A100)
# ============================================================================
phase 3 "c1_001 paper_faithful"
run_and_ship paper_faithful output/CRCD/C1_001_paper_faithful "$DRIVE_ROOT/paper_faithful"

# ============================================================================
# PHASE 4 -- run semsup_sched (Recipe A, ~30-40 min A100)
# ============================================================================
phase 4 "c1_001 semsup_sched (Recipe A)"
run_and_ship semsup_sched output/CRCD/C1_001_semsup_sched "$DRIVE_ROOT/semsup_sched"

# ============================================================================
# PHASE 5 -- run paper_faithful_depthx0p3 (depth scale hypothesis, ~15-20 min)
# ============================================================================
phase 5 "c1_001 paper_faithful_depthx0p3"
run_and_ship paper_faithful_depthx0p3 output/CRCD/C1_001_paper_faithful_depthx0p3 "$DRIVE_ROOT/paper_faithful_depthx0p3"

# ============================================================================
# PHASE 6 -- summary: SE3 + Sim3 ATE for each config
# ============================================================================
phase 6 "summary (SE3 + Sim3 ATE per config)"
SUMMARY=$DRIVE_ROOT/summary.txt
: > "$SUMMARY"
python3 - >> "$SUMMARY" 2>&1 <<'PYEOF'
import os, numpy as np
GT = '/content/DDS-SLAM/data/CRCD/C1_001/groundtruth.txt'

def parse_est(p):
    poses=[]
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

def horn(m,d,sc=False):
    mc=m.mean(0); dc=d.mean(0); mm=m-mc; dd=d-dc
    H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    ds=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,ds]); R=Vt.T@D@U.T
    s=(S*np.array([1,1,ds])).sum()/(mm*mm).sum() if sc else 1.0
    t=dc-s*R@mc
    return (s*(R@m.T)).T+t, s

g = parse_tum(GT)
gpath = np.linalg.norm(np.diff(g,axis=0),axis=1).sum()*1000
print(f"GT path length: {gpath:.2f} mm  (extent xyz mm: {(g.max(0)-g.min(0))*1000})")
print(f"{'config':<32} {'frames':>7} {'SE3_mean_mm':>12} {'Sim3_mean_mm':>13} {'sim3_s':>10} {'est_path_mm':>13}")
print("-" * 95)
for name in ['paper_faithful', 'semsup_sched', 'paper_faithful_depthx0p3']:
    est_path = f'/content/DDS-SLAM/output/CRCD/C1_001_{name}/demo/est_c2w_data.txt'
    if not os.path.isfile(est_path):
        print(f"{name:<32}   (no est_c2w_data.txt at {est_path})")
        continue
    e = parse_est(est_path)
    n = min(len(e), len(g))
    if n < 10:
        print(f"{name:<32}   (only {n} matched frames)"); continue
    ee, gg = e[:n], g[:n]
    ase3,_ = horn(ee,gg,False); ese3 = np.linalg.norm(ase3-gg,axis=1)*1000
    asim3,s = horn(ee,gg,True); esim3 = np.linalg.norm(asim3-gg,axis=1)*1000
    epath = np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()*1000
    print(f"{name:<32} {n:>7d} {ese3.mean():>12.2f} {esim3.mean():>13.3f} {s:>10.4f} {epath:>13.2f}")
PYEOF
echo ""
echo "=== summary ==="
cat "$SUMMARY"
echo ""
echo "=== runbook done $(date -Iseconds) ==="
echo "Logs: $LOG"
echo "Summary: $SUMMARY"
echo "Per-config payloads: $DRIVE_ROOT/{paper_faithful,semsup_sched,paper_faithful_depthx0p3}/payload.tgz"
