#!/usr/bin/env bash
# =====================================================================
# StereoMIS P2_1 MoGe-2 back-4000 — OVERNIGHT runbook (error-hardened)
# 2026-06-08
# =====================================================================
# PREREQS (run these FIRST, in order):
#   1. Mount Drive (Colab cell):  from google.colab import drive; drive.mount('/content/drive')
#   2. Clone+pull repo:           cd /content && (git clone https://github.com/bwright000/DDS-SLAM.git || true) && cd DDS-SLAM && git checkout main && git pull
#   3. Restore cached env:        bash Addons/env/colab_exact_env.sh --skip-data   (restores /tmp/dds_env + CUDA 11.3 from MyDrive/dds_cache)
#   4. Run THIS:                  nohup bash Addons/colab/overnight_stereomis_20260608.sh > /content/stereomis.out 2>&1 & tail -f /content/stereomis.out
#
# DESIGN (why each guard exists):
#   - Two-env split: MoGe-2 needs torch>=2 (SYSTEM python); DDS-SLAM needs the
#     paper-exact dds_env (torch 1.10). We never mix them.
#   - MoGe stage-1 pre-allocates a [N,H,W] buffer; 4000 frames OOMs -> empty
#     depth -> SLAM IndexError. So MoGe runs in 500-frame CHUNKS.
#   - FAIL-FAST: every stage verifies its output count and aborts with a clear
#     message instead of letting a later stage crash cryptically.
#   - Resumable: .DONE is written ONLY on full success; MoGe depth + ckpt are
#     cached to Drive so a re-run skips finished work.
#   - set -uo pipefail (NOT -e): non-fatal steps log a WARN and continue; only
#     genuinely unrecoverable states `exit 1`.
# =====================================================================
set -uo pipefail

DATE=20260608
REPO=/content/DDS-SLAM
STEREO_DS=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1
STEREO_CKPT_DIR=$STEREO_DS/checkpoints
STEREO_MOGE_DIR=$STEREO_DS/depth/MoGe2_p2-1-back4000_${DATE}      # depth as a dataset element
RUN_OUT=/content/drive/MyDrive/Outputs/DDS-SLAM_stereomis-overnight_${DATE}
SNAME=DDS-SLAM_stereomis-p2-1-moge-back4000_${DATE}
CFG=configs/StereoMIS/p2_1_moge_back4000.yaml                     # tracked; output dir = $SNAME
N_BACK=4000

mkdir -p "$STEREO_CKPT_DIR" "$RUN_OUT" "$STEREO_MOGE_DIR"
SDIR=$RUN_OUT/$SNAME; mkdir -p "$SDIR"
LOG=$RUN_OUT/runbook.log
say(){  echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }
fail(){ echo "[FATAL $(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== StereoMIS overnight start $(date -Iseconds) ==="
[ -f "$SDIR/.DONE" ] && { say "already complete (.DONE present) — nothing to do."; exit 0; }

# ---------------------------------------------------------------------
# 0. PRE-FLIGHT
# ---------------------------------------------------------------------
command -v nvidia-smi >/dev/null 2>&1 || { fail "no nvidia-smi — select a GPU runtime"; exit 1; }
say "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
[ -d /content/drive/MyDrive ] || { fail "Drive not mounted at /content/drive (run the mount cell first)"; exit 1; }
cd "$REPO" 2>/dev/null || { fail "$REPO missing — clone the repo first"; exit 1; }
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
  git fetch -q 2>/dev/null && git merge -q --ff-only origin/main 2>/dev/null && say "pulled main" || say "WARN: git pull skipped (offline/non-FF)"
else
  say "WARN: working tree dirty — skipping git pull"
fi
say "HEAD: $(git rev-parse --short HEAD 2>/dev/null)"
[ -f "$CFG" ] || { fail "config $CFG missing (git pull?)"; exit 1; }

# ---------------------------------------------------------------------
# 1. STAGE P2_1 (tarball-first; avoids Drive FUSE drops on long runs)
# ---------------------------------------------------------------------
if [ ! -d /content/p2_1_local/video_frames ] || [ -z "$(ls /content/p2_1_local/video_frames/*l.png 2>/dev/null | head -1)" ]; then
  mkdir -p /content/p2_1_local
  STAR=$STEREO_DS/P2_1_staging.tar
  if [ -f "$STAR" ]; then
    say "staging from $STAR ($(du -h "$STAR" 2>/dev/null | cut -f1))"; tar xf "$STAR" -C /content/p2_1_local
  else
    say "WARN: $STAR missing — per-file copy from Drive (slow)"
    ( cd "$STEREO_DS" && tar cf - video_frames groundtruth.txt StereoCalibration.ini masks 2>/dev/null | tar xf - -C /content/p2_1_local )
  fi
fi
( cd /content/p2_1_local/video_frames && ls *l.png 2>/dev/null | tail -$N_BACK ) > /content/_p2_1_frames.txt
NIMG=$(wc -l < /content/_p2_1_frames.txt)
[ "$NIMG" -ge 1 ] || { fail "no left frames staged (NIMG=$NIMG) — check $STEREO_DS/video_frames/*l.png"; exit 1; }
say "staged $NIMG left frames (last-$N_BACK)"

# ---------------------------------------------------------------------
# 2. MoGe-2 DEPTH  (SYSTEM python / torch>=2) — chunked, cached to Drive
# ---------------------------------------------------------------------
NDEP=$(ls "$STEREO_MOGE_DIR"/*.png 2>/dev/null | wc -l)
if [ "$NDEP" -ge "$NIMG" ]; then
  say "MoGe depth already on Drive ($NDEP png) -> skip generation"
else
  say "generating MoGe-2 depth (system python, chunked) ..."
  PYSYS=$(command -v python3)
  "$PYSYS" -c "import torch,sys; sys.exit(0 if torch.__version__>='2' else 1)" 2>/dev/null \
    || { fail "system python lacks torch>=2 (needed for MoGe-2). Do NOT activate dds_env before this step."; exit 1; }
  "$PYSYS" -c "import moge" 2>/dev/null || {
    say "installing moge into system python ..."
    "$PYSYS" -m pip install -q "git+https://github.com/microsoft/MoGe.git" 2>&1 | tail -3 | tee -a "$LOG"
  }
  "$PYSYS" -c "import moge" 2>/dev/null || { fail "moge still not importable — fix the MoGe install before re-running"; exit 1; }

  NPY=/content/p2_1_moge_npy; rm -rf "$NPY"; mkdir -p "$NPY"
  CHUNK=500; i=0
  while [ "$i" -lt "$NIMG" ]; do
    STAGE=/content/_moge_stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
    # generate_depth_moge globs *-left.png; StereoMIS frames are *l.png -> symlink-rename
    sed -n "$((i+1)),$((i+CHUNK))p" /content/_p2_1_frames.txt | while read f; do
      stem="${f%l.png}"; ln -sf "/content/p2_1_local/video_frames/$f" "$STAGE/${stem}-left.png"; done
    say "  MoGe chunk frames $i..$((i+CHUNK)): $(ls "$STAGE"/*-left.png 2>/dev/null | wc -l) staged"
    # --depth_scale 1 => npy holds METRES (png conversion multiplies once -> no double-scale)
    "$PYSYS" Addons/depth/generate_depth_moge.py --rgb "$STAGE" --out "$NPY" \
      --temporal_window 5 --depth_scale 1 2>&1 | tee -a "$LOG" || say "  WARN: MoGe chunk $i failed"
    i=$((i+CHUNK))
  done
  NNPY=$(ls "$NPY"/*-left_depth.npy 2>/dev/null | wc -l)
  say "MoGe NPYs produced: $NNPY/$NIMG"
  # npy(metres) -> uint16 png @ png_depth_scale=10000, named <stem>.png
  "$PYSYS" - "$NPY" "$STEREO_MOGE_DIR" <<'PYEOF'
import sys, glob, os, numpy as np, cv2
nd, pd = sys.argv[1], sys.argv[2]; n=0
for f in sorted(glob.glob(os.path.join(nd, '*-left_depth.npy'))):
    d = np.load(f).astype(np.float32)
    u = np.clip(d*10000, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(pd, os.path.basename(f).replace('-left_depth.npy','')+'.png'), u); n+=1
print('converted PNGs:', n)
PYEOF
fi

# ---------------------------------------------------------------------
# 3. DEPTH FAIL-FAST (before touching the SLAM)
# ---------------------------------------------------------------------
NDEP=$(ls "$STEREO_MOGE_DIR"/*.png 2>/dev/null | wc -l)
say "depth check: $NDEP png vs $NIMG frames"
[ "$NDEP" -ge "$NIMG" ] || { fail "MoGe depth incomplete ($NDEP/$NIMG) — aborting BEFORE SLAM. Inspect Phase-2 log above (moge import / OOM / HF model download)."; exit 1; }

# ---------------------------------------------------------------------
# 4. DDS-SLAM  (dds_env / torch 1.10) — ATE produced inline by ddsslam.py
# ---------------------------------------------------------------------
[ -f /tmp/dds_env/bin/activate ] || { fail "/tmp/dds_env missing — run Addons/env/colab_exact_env.sh --skip-data first"; exit 1; }
source /tmp/dds_env/bin/activate
# CUDA 11.3 at RUNTIME for tinycudann (set HERE so Phase-2 MoGe/torch2 kept default CUDA)
export CUDA_HOME=/usr/local/cuda-11.3
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10
python -c "import torch,tinycudann,marching_cubes; print('env OK', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | tee -a "$LOG" \
  || { fail "dds_env import failed (cache stale or GPU arch != T4). Rebuild: bash Addons/env/colab_exact_env.sh --skip-data --skip-cache"; exit 1; }

# StereoMISDataset reads {datadir}/depth/*.png — symlink the MoGe dir there.
# (rm -rf first: `ln -sf` into an existing dir nests the link INSIDE it.)
rm -rf /content/p2_1_local/depth
ln -sf "$STEREO_MOGE_DIR" /content/p2_1_local/depth
NLINK=$(ls /content/p2_1_local/depth/*.png 2>/dev/null | wc -l)
[ "$NLINK" -ge "$NIMG" ] || { fail "depth symlink resolves to only $NLINK png — aborting"; exit 1; }

say "SLAM: ddsslam.py --config $CFG  ($NIMG frames — this is the long pole, ~hours)"
python ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "WARN: ddsslam.py exited non-zero (see traceback above)"

OUT=$(python -c "from config import load_config; print(load_config('$CFG')['data']['output'])" 2>/dev/null)
CKPT=$(ls -t "$REPO/$OUT/demo"/checkpoint*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  fail "no checkpoint produced under $REPO/$OUT/demo — SLAM did not complete. NOT writing .DONE (will retry)."
  exit 1
fi
cp "$CKPT" "$STEREO_CKPT_DIR/$SNAME.pt"
say "saved checkpoint -> $STEREO_CKPT_DIR/$SNAME.pt"
# ATE + pose plots are written inline by ddsslam.py (tools.eval_ate.pose_evaluation)
cp "$REPO/$OUT/demo/output.txt" "$SDIR/ate_output.txt" 2>/dev/null && say "ATE -> $SDIR/ate_output.txt" || say "WARN: output.txt (ATE) not found"
cp "$REPO/$OUT/demo"/pose_*.png "$SDIR/" 2>/dev/null || true
# Full-frame render (PSNR/SSIM/LPIPS comparable + 6-panel video source)
python Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" \
  --output_dir "$SDIR/rendered_all" --save_depth --save_gt 2>&1 | tee -a "$LOG" || say "WARN: render_all_frames failed"
tar czf "$SDIR/payload.tgz" -C "$REPO/$OUT" . 2>/dev/null && say "payload -> $SDIR/payload.tgz" || true

touch "$SDIR/.DONE"   # success
say "=== DONE $(date -Iseconds) ==="
say "  checkpoint : $STEREO_CKPT_DIR/$SNAME.pt"
say "  depth      : $STEREO_MOGE_DIR  ($NDEP png)"
say "  artefacts  : $SDIR  (ate_output.txt, pose_*.png, rendered_all/, payload.tgz)"
say "  log        : $LOG"
