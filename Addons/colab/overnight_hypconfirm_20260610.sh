#!/bin/bash
# ============================================================================
# DDS-SLAM GAUGE / DEAD-FIELD HYPOTHESIS CONFIRMATION  (T4-friendly)
#
# 3 single-element tests x 2 datasets (SemSup trail_3 + CRCD C_1/001):
#   baseline    - field on, pose tracked (control)
#   deformoff   - H2: deformation_off:true (field bypassed)   -> field INERT?
#   posefrozen  - H3: pose pinned (4 knobs)                   -> field REVIVES?
#
# Per run, 4 GPU diagnostics:
#   timenet_weight_audit  (H1 dead-weight verdict)
#   dx_hook               (H1/H3 Δx stats)
#   dx_norm_heatmap       (H3 DECIDER: spatial Δx revival)
#   render_all_frames     (--save_gt --save_depth: 6-panel ingredients)
# The canonical 6-panel video is assembled LOCALLY afterward from the shipped
# renders (proven path; seg/traj live on the local F: drive).
#
# Reliability: set -uo pipefail (NOT -e) so one failed test never kills the
# night; SemSup is the must-have, CRCD is best-effort (skipped if staging fails).
# Resume-safe via per-test .DONE markers. Paste into a Colab VS Code tunnel term.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_hypconfirm_${DATE}
LWORK=/content/hypconfirm
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

STAGED=$REPO/data/CRCD/C1_001
DRIVE_SNIPPET=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
CALIB_PKL=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl

say "=== hyp-confirm start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }

activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "modern stack missing -- full rebuild (~15 min)"
    bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { say "env check FAIL"; exit 1; }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  if [ -d "$REPO/data/Super/trail_3/rgb" ]; then say "SemSup already staged"; return 0; fi
  [ -d "$SRC/rgb" ] || { say "FATAL: SemSup source missing at $SRC"; return 1; }
  say "staging SemSup (trial_3 -> trail_3)"
  mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"
  [ -d "$REPO/data/Super/trail_3/depth/variant_a_stereo" ] \
    || say "  WARN: depth/variant_a_stereo missing -- paperfaith depth_subdir will fail"
}

# CRCD preprocess + MoGe (proven block from run_crcd_c1_001_3test.sh; exit->return for graceful skip)
stage_crcd(){
  if [ -f "$STAGED/.STAGED" ] && [ -f "$STAGED/depth/.DONE" ]; then say "CRCD already staged"; return 0; fi
  [ -d "$DRIVE_SNIPPET" ] || { say "  CRCD snippet missing at $DRIVE_SNIPPET"; return 1; }
  say "staging CRCD C_1/001 (preprocess)"
  rm -rf "$STAGED"; mkdir -p /content/crcd_raw "$(dirname "$STAGED")"
  if [ ! -d /content/crcd_raw/snippet_001 ]; then
    cp -r "$DRIVE_SNIPPET" /content/crcd_raw/snippet.tmp && mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_001
  fi
  python "$REPO/Addons/preprocess/preprocess_crcd_published.py" \
    --snippet_dir /content/crcd_raw/snippet_001 --calib_pkl "$CALIB_PKL" --output_dir "${STAGED}.tmp" || return 1
  local NL; NL=$(find "${STAGED}.tmp/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  [ "$NL" -gt 100 ] || { say "  preprocess produced $NL frames -- CRCD abort"; return 1; }
  mv "${STAGED}.tmp" "$STAGED"; touch "$STAGED/.STAGED"; rm -rf /content/crcd_raw
  say "staging CRCD depth (MoGe-2; Drive cache if present)"
  local EXPECTED; EXPECTED=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  local NDRIVE=0; [ -d "$DRIVE_SNIPPET/depth" ] && NDRIVE=$(find "$DRIVE_SNIPPET/depth" -maxdepth 1 -name '*.png' | wc -l)
  if [ "$NDRIVE" -eq "$EXPECTED" ]; then
    say "  Drive depth cache hit ($NDRIVE) -- copy + index-rename"
    mkdir -p "$STAGED/depth.tmp"
    python3 - <<PYEOF || return 1
import os, shutil
RGB=sorted(f for f in os.listdir('${DRIVE_SNIPPET}/rgb') if f.endswith('.png'))
DEP=sorted(f for f in os.listdir('${DRIVE_SNIPPET}/depth') if f.endswith('.png'))
assert len(RGB)==len(DEP), f'rgb={len(RGB)} depth={len(DEP)}'
for i,d in enumerate(DEP): shutil.copy2(os.path.join('${DRIVE_SNIPPET}/depth',d), os.path.join('${STAGED}/depth.tmp', f'{i:06d}.png'))
print('copied', len(DEP))
PYEOF
    rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"; touch "$STAGED/depth/.DONE"
  else
    say "  regenerating MoGe depth"
    python3 -c 'import moge.model.v2' 2>/dev/null || python3 -m pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub
    cd "$STAGED"; mkdir -p _moge_in depth.tmp
    for f in video_frames/*l.png; do fid=$(basename "$f" l.png); ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"; done
    python3 "$REPO/Addons/depth/generate_depth_moge.py" --rgb _moge_in --out _moge_npy \
      --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 || { cd "$REPO"; return 1; }
    python3 - <<'PYEOF' || { cd "$REPO"; return 1; }
import numpy as np, cv2, glob, os
for p in sorted(glob.glob('_moge_npy/*-left_depth.npy')):
    fid=os.path.basename(p).split('-')[0]; out=f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    cv2.imwrite(out, np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('png_out', len(glob.glob('depth.tmp/*.png')))
PYEOF
    rm -rf depth && mv depth.tmp depth; touch depth/.DONE; rm -rf _moge_in _moge_npy; cd "$REPO"
  fi
  return 0
}

# --- per-test driver: run + 4 diagnostics + ship -------------------------------
# $1=cfg  $2=out_demo_dir  $3=rgb_dir  $4=rgb_pattern  $5=label
run_test(){
  local CFG=$1 OUT=$2 RGB=$3 PAT=$4 LBL=$5
  local DST="$DRIVE/$LBL" LW="$LWORK/$LBL"
  done_marker "$DST" && { say "  $LBL already shipped -- skip"; return 0; }
  rm -rf "$LW"; mkdir -p "$LW" "$DST"
  say "=== TEST $LBL  (cfg=$CFG) ==="
  cd "$REPO"
  local T0; T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN: $LBL run nonzero exit"
  say "  $LBL run elapsed $(( ($(date +%s)-T0)/60 )) min"
  local CKPT; CKPT=$(ls -t "$OUT"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  ERROR: no checkpoint in $OUT -- skip diagnostics for $LBL"; return 1; }
  say "  ckpt: $CKPT"
  # H1 dead-weight verdict
  python diagnosis/infra/timenet_weight_audit.py --ckpt "$CKPT" --json "$LW/timenet_audit.json" 2>&1 | tee -a "$LOG" || say "  WARN timenet_audit"
  # H1/H3 Δx stats
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
  # H3 DECIDER: spatial Δx-norm heatmap
  python diagnosis/infra/dx_norm_heatmap.py --config "$CFG" --checkpoint "$CKPT" --rgb_dir "$RGB" --rgb_pattern "$PAT" --output_dir "$LW/dx_heatmap" 2>&1 | tee -a "$LOG" || say "  WARN dx_norm_heatmap"
  # 6-panel ingredients (rectified rgb + gt + depth)
  python -W ignore Addons/viz/render_all_frames.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/renders" --skip 2 --save_gt --save_depth 2>&1 | tee -a "$LOG" || say "  WARN render_all_frames"
  cp "$OUT"/est_c2w_data.txt "$OUT"/groundtruth.txt "$OUT"/output_relative.txt "$LW/" 2>/dev/null || true
  cp "$OUT"/output.txt "$OUT"/ate_output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; sync
  say "  $LBL shipped -> $DST/payload.tgz"
}

# ---------------------------------------------------------------------------
nvidia-smi -L || true
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo unknown)
say "GPU: $GPU  (T4 fine for these small runs; A100 faster)"
cd "$REPO"; activate_dds_env

# === SemSup arm (must-have; SemSup-posefrozen is the decisive H3 test) ===
if stage_semsup; then
  SRGB="$REPO/data/Super/trail_3/rgb"
  run_test configs/Super/hyp_super_baseline.yaml   output/hyp_super_baseline/demo   "$SRGB" '*-left.png' super_baseline
  run_test configs/Super/hyp_super_deformoff.yaml  output/hyp_super_deformoff/demo  "$SRGB" '*-left.png' super_deformoff
  run_test configs/Super/hyp_super_posefrozen.yaml output/hyp_super_posefrozen/demo "$SRGB" '*-left.png' super_posefrozen
else say "SemSup staging failed -- ABORT (must-have)"; fi

# === CRCD arm (best-effort; posefrozen here is confounded by real camera motion) ===
if stage_crcd; then
  CRGB="$STAGED/video_frames"
  run_test configs/CRCD/hyp_crcd_baseline.yaml   output/hyp_crcd_baseline/demo   "$CRGB" '*l.png' crcd_baseline
  run_test configs/CRCD/hyp_crcd_deformoff.yaml  output/hyp_crcd_deformoff/demo  "$CRGB" '*l.png' crcd_deformoff
  run_test configs/CRCD/hyp_crcd_posefrozen.yaml output/hyp_crcd_posefrozen/demo "$CRGB" '*l.png' crcd_posefrozen
else say "CRCD staging failed -- SKIPPING CRCD arm (SemSup results stand)"; fi

say "=== hyp-confirm DONE $(date -Iseconds) ==="
say "Per test on Drive: $DRIVE/<label>/payload.tgz  (renders, dx, dx_heatmap, timenet_audit, est/gt poses)"
say "READ-OUT: timenet_audit.json (dead?) + dx/ + dx_heatmap/ (Δx>0 & on tissue? = H3 revival)"
say "Assemble 6-panel videos LOCALLY from each payload's renders/ (seg+traj on F:)."
# close the runtime to stop compute (guarded; no-op off Colab)
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab / already free -- stop the runtime manually)"
