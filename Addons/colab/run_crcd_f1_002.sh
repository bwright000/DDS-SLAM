#!/bin/bash
# ============================================================================
# DDS-SLAM CRCD F1/002 runbook — paste into Colab VS Code tunnel terminal
# Six phases, resume-safe. Re-paste after session drop to pick up where left off.
# Master log: $DRIVE_ROOT/runbook.log (on Drive — survives session death)
# ============================================================================
set -euo pipefail   # -e + pipefail so partial failures don't mark DONE
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_crcd_${DATE}
mkdir -p "$DRIVE_ROOT/semsup_v2" "$DRIVE_ROOT/paper_faithful" "$DRIVE_ROOT/paper_faithful_v2"
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== runbook start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

# ---------- helpers ----------
phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }
STAGED=/content/DDS-SLAM/data/CRCD/F1_002

activate_dds_env() {
  # Modern stack: colab_setup.sh installs into the system Python 3.12 + torch 2.x.
  # No /tmp/dds_env venv. Memory project_session_20260523_late confirmed the modern
  # stack matches legacy LPIPS within 0.001 — numerically equivalent for paper recreation.
  # The exact-paper env (colab_exact_env.sh -> /tmp/dds_env) is only needed if we
  # want to track the paper's specific torch 1.10 / CUDA 11.3 stack.
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    echo "modern stack missing -- full rebuild (~15 min on T4)"
    # --skip-data: we have our own data; the bundled gdown download often 403s
    bash /content/DDS-SLAM/Addons/env/colab_setup.sh --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { echo "env check FAIL"; exit 1; }
  # Keep driver libs on LD_LIBRARY_PATH (Colab convention)
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

ship_to_drive() {  # $1=src_dir  $2=dst_dir_on_drive
  local SRC=$1 DST=$2
  tar czf "$DST/payload.tgz.partial" -C "$SRC" .
  mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  mark_done "$DST"
}

# ============================================================================
# PHASE 0 -- env sanity + repo sync.
# ============================================================================
phase 0 "env check + repo update"
[ -d /content/drive/MyDrive ] || { echo "Drive not mounted -- abort"; exit 1; }
cd /content/DDS-SLAM
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull, continuing with local edits"
else
  git fetch && git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD) \
    || { echo "non-FF on remote -- manual reconcile required, abort"; exit 1; }
fi
nvidia-smi -L
activate_dds_env

# Phase 3's MoGe install runs in /tmp/moge_env (its own venv inside a subshell)
# so it can't contaminate the system-python torch stack regardless of where
# dds-slam imports live. No pyvenv.cfg check needed for the modern stack.

# ============================================================================
# PHASE 1 -- SemSup trail_3 paper_faithful_v2 sanity run (~30 min A100)
# ============================================================================
phase 1 "SemSup paper_faithful_v2"
if done_marker "$DRIVE_ROOT/semsup_v2"; then
  echo "  already shipped -- skip"
else
  cd /content/DDS-SLAM
  # Drive canonical path: MyDrive/Datasets/SemSup/v2_data/trial_3 (i — correct spelling)
  # Repo config wants: data/Super/trail_3 (a — repo typo perpetuated through codebase)
  # Copy renames at destination so DDS-SLAM's globs hit.
  [ -d data/Super/trail_3 ] || {
    mkdir -p data/Super
    cp -r /content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3 data/Super/trail_3
  }
  python ddsslam.py --config configs/Super/trail3_paper_faithful_v2.yaml
  ship_to_drive output/trail3_paper_faithful_v2/demo "$DRIVE_ROOT/semsup_v2"
fi

# ============================================================================
# PHASE 2 -- CRCD F_1/002 preprocess. Sentinel-gated.
# ============================================================================
phase 2 "CRCD F_1/002 preprocess"
if [ -f "$STAGED/.STAGED" ]; then
  echo "  staged dir complete -- skip preprocess"
else
  rm -rf "$STAGED"
  mkdir -p /content/crcd_raw
  if [ ! -d /content/crcd_raw/snippet_002 ]; then
    if [ -f /content/drive/MyDrive/Datasets/CRCD-Published/F_1/snippet_002.tar ]; then
      mkdir -p /content/crcd_raw/snippet.tmp
      tar xf /content/drive/MyDrive/Datasets/CRCD-Published/F_1/snippet_002.tar -C /content/crcd_raw/snippet.tmp
      if [ -d /content/crcd_raw/snippet.tmp/snippet_002 ]; then
        mv /content/crcd_raw/snippet.tmp/snippet_002 /content/crcd_raw/snippet_002
        rm -rf /content/crcd_raw/snippet.tmp
      else
        mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_002
      fi
    else
      cp -r /content/drive/MyDrive/Datasets/CRCD-Published/F_1/snippet_002 /content/crcd_raw/snippet.tmp
      mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_002
    fi
  fi
  cp -n /content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl /content/crcd_raw/
  cd /content/DDS-SLAM
  mkdir -p "$(dirname $STAGED)"
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir /content/crcd_raw/snippet_002 \
    --calib_pkl   /content/crcd_raw/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl \
    --output_dir  "${STAGED}.tmp"
  NL=$(ls "${STAGED}.tmp/video_frames"/*l.png 2>/dev/null | wc -l)
  NG=$(grep -cv '^#' "${STAGED}.tmp/groundtruth.txt" 2>/dev/null || echo 0)
  [ "$NL" -gt 100 ] && [ "$NG" -ge "$NL" ] || { echo "preprocess produced $NL frames / $NG gt rows -- abort"; exit 1; }
  mv "${STAGED}.tmp" "$STAGED"
  touch "$STAGED/.STAGED"
  rm -rf /content/crcd_raw
fi

# ============================================================================
# PHASE 3 -- MoGe-2 depth gen. Runs in an isolated venv inside a SUBSHELL so
# env mutations cannot leak into dds_env. Atomic PNG writes + count-matched sentinel.
# ============================================================================
phase 3 "MoGe-2 depth gen (isolated venv, resumable)"
# Robust counts that survive missing dirs without `set -e` killing the script.
# NOTE: `find $DIR | wc -l` UNDER set -o pipefail EXITS NON-ZERO if $DIR doesn't
# exist (find errors out; pipefail propagates; set -e kills). 2>/dev/null only
# mutes stderr -- the exit code still propagates. Use an explicit dir-exists
# guard so missing dirs default the count to 0 without killing the script.
EXPECTED=0
ACTUAL=0
if [ -d "$STAGED/video_frames" ]; then
  EXPECTED=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
fi
if [ -d "$STAGED/depth" ]; then
  ACTUAL=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
fi
if [ -f "$STAGED/depth/.DONE" ] && [ "$ACTUAL" -eq "$EXPECTED" ]; then
  echo "  depth/ complete ($ACTUAL/$EXPECTED) -- skip MoGe"
else
  echo "  depth/ incomplete ($ACTUAL/$EXPECTED) -- resuming MoGe"
  # Modern stack: dds-slam imports already live in system python, so there's no
  # /tmp/dds_env to protect. The earlier venv-isolation design (commit 6541845)
  # was overkill and broke because Colab Python 3.12's `python3 -m venv` fails
  # to bootstrap pip via ensurepip. Install MoGe directly into system python.
  python3 -c 'import moge.model.v2' 2>/dev/null || \
    python3 -m pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub
  cd "$STAGED"
  mkdir -p _moge_in depth.tmp
  for f in video_frames/*l.png; do
    fid=$(basename "$f" l.png)
    [ -L "_moge_in/${fid}-left.png" ] || ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"
  done
  echo "  symlinks: $(ls _moge_in/ 2>/dev/null | wc -l)"
  # --temporal_window 1 disables temporal smoothing. Memory budget on T4 is
  # ~12 GB RAM and the pre-allocated raw depth stack alone is ~4.7 GB for
  # 1287 frames at 1280x720; smoothing would allocate a SECOND 4.7 GB buffer
  # and OOM. Per memory project_minmax_results_20260531 temporal smoothing
  # produced metric-neutral renders anyway (DeltaPSNR < 0.1).
  python3 /content/DDS-SLAM/Addons/depth/generate_depth_moge.py \
    --rgb _moge_in --out _moge_npy \
    --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0
  (
    cd "$STAGED"
    python3 - <<'PYEOF'
import numpy as np, cv2, glob, os
n_in = sorted(glob.glob('_moge_npy/*-left_depth.npy'))
for p in n_in:
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/{fid}.png'
    if os.path.exists(out):
        continue
    d = np.load(p).astype(np.float32)
    tmp = out + '.part'
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, out)
print('npy_in', len(n_in), 'png_out', len(glob.glob('depth.tmp/*.png')))
PYEOF
    NPY=$(ls _moge_npy/*-left_depth.npy 2>/dev/null | wc -l)
    PNG=$(ls depth.tmp/*.png | wc -l)
    [ "$PNG" -eq "$NPY" ] && [ "$PNG" -eq "$EXPECTED" ] \
      || { echo "depth count mismatch png=$PNG npy=$NPY expected=$EXPECTED -- keep intermediates, abort"; exit 1; }
    rm -rf depth && mv depth.tmp depth
    sync; touch depth/.DONE; sync
    rm -rf _moge_in _moge_npy
  )

  # ----- Persist depth back to the canonical CRCD-Published snippet -----
  # CRCD ships snippet_002/depth/ empty (depth_placeholder: true). Fill it with
  # our MoGe-2 maps so future runs reuse the same depth without regenerating.
  # Re-map our staged 000000..001286.png back to the snippet's frame_NNNNNN.png
  # naming (frame_011159..012445 for F_1/002) by listing the snippet's rgb/.
  echo "  syncing depth back to CRCD-Published snippet on Drive..."
  python3 - <<'PYEOF'
import os, shutil
STAGED = '/content/DDS-SLAM/data/CRCD/F1_002'
SRC_SNIPPET = '/content/drive/MyDrive/Datasets/CRCD-Published/F_1/snippet_002'
DST_DEPTH = f'{SRC_SNIPPET}/depth'
os.makedirs(DST_DEPTH, exist_ok=True)
rgb_orig = sorted(f for f in os.listdir(f'{SRC_SNIPPET}/rgb') if f.endswith('.png'))
copied = skipped = 0
for i, orig in enumerate(rgb_orig):
    src_png = f'{STAGED}/depth/{i:06d}.png'
    if not os.path.exists(src_png):
        continue
    dst_png = f'{DST_DEPTH}/{orig}'  # match the snippet's frame_NNNNNN.png convention
    if os.path.exists(dst_png):
        skipped += 1; continue
    shutil.copy2(src_png, dst_png)
    copied += 1
print(f'depth -> Drive snippet: copied={copied} skipped={skipped} total_on_drive={len(os.listdir(DST_DEPTH))}')
PYEOF
  sync
fi

# ============================================================================
# PHASE 4 -- CRCD paper_faithful run (~45-60 min A100)
# ============================================================================
phase 4 "CRCD F1_002 paper_faithful"
if done_marker "$DRIVE_ROOT/paper_faithful"; then
  echo "  already shipped -- skip"
else
  cd /content/DDS-SLAM
  python ddsslam.py --config configs/CRCD/f1_002.yaml
  ship_to_drive output/CRCD/F1_002_paper_faithful/demo "$DRIVE_ROOT/paper_faithful"
fi

# ============================================================================
# PHASE 5 -- CRCD paper_faithful_v2 run (~45-60 min A100). Reuses depth/.
# ============================================================================
phase 5 "CRCD F1_002 paper_faithful_v2"
if done_marker "$DRIVE_ROOT/paper_faithful_v2"; then
  echo "  already shipped -- skip"
else
  cd /content/DDS-SLAM
  python ddsslam.py --config configs/CRCD/f1_002_v2.yaml
  ship_to_drive output/CRCD/F1_002_paper_faithful_v2/demo "$DRIVE_ROOT/paper_faithful_v2"
fi

echo ""
echo "=== runbook DONE $(date -Iseconds) -- outputs in $DRIVE_ROOT ==="
