#!/bin/bash
# ============================================================================
# DDS-SLAM OVERNIGHT — MAP-ABSORPTION GATE + 3D FIELD DIAGNOSTICS (T4/A100). 2026-06-14.
#
# Battery-7 falsified stabilise-first (field surface-DEAD across 3 seeds, pose frozen). Architecture
# read: the static map fits a blurry TIME-AVERAGE and starves the field. BEFORE building Inc-1 (the
# learnt uncertainty head) we must answer: is there a real photometric RESIDUAL at the moving tissue
# for the uncertainty to fire on?  This overnight answers that across all 3 SemSup seeds + (best-effort)
# CRCD large-motion, persists the checkpoints (battery-7 did NOT), runs the full field-diagnostic suite,
# and validates the Inc-0 plumbing.
#
# PHASES (each resume-safe via .DONE; ships to Drive incrementally):
#   A  SemSup s0/s1/s2  -- re-train (frozen pose) + PERSIST ckpt + map_absorption_probe
#                          + render_eval_attrib + dx_hook -> field_viz3d (3D field) + timestamp-sweep
#   B  Inc-0 regression -- golden + check (proves Arm-2 config plumbing == base, bit-identical)
#   C  CRCD C_2/001     -- BEST-EFFORT (set +e, auto-SKIP if not staged): large motion + FREE pose
#                          (both failure channels). Cannot harm A/B (they ship first).
#
# Robust for unattended: SemSup core completes first and is never blocked by CRCD.
# ~15min env + ~37min x3 SemSup + ~5min/diag + (CRCD ~50min). Budget ~3-4h. set -uo.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
DRIVE=/content/drive/MyDrive/Outputs/dds_overnight_mapprobe_${DATE}
LWORK=/content/mapprobe
mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }

say "=== overnight map-probe start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD 2>/dev/null) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }

activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "modern stack missing -- full rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" || { say "env FAIL"; exit 1; }
  python -c "import lpips" 2>/dev/null || pip install -q lpips || say "  WARN lpips (LPIPS will be skipped)"
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  [ -d "$REPO/data/Super/trail_3/rgb" ] && { say "SemSup staged"; return 0; }
  [ -d "$SRC/rgb" ] || { say "FATAL: SemSup source missing"; return 1; }
  mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"
}

# full field-diagnostic suite on one trained checkpoint --------------------------------
diagnose(){
  local CFG=$1 CKPT=$2 LW=$3
  python diagnosis/infra/map_absorption_probe.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/map_absorb.json" --max_frames 40 --frame_stride 3 2>&1 | tee -a "$LOG" || say "  WARN probe"
  python diagnosis/infra/render_eval_attrib.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/field_attrib.json" --max_frames 40 --frame_stride 3 2>&1 | tee -a "$LOG" || say "  WARN attrib"
  python diagnosis/infra/dx_hook.py --config "$CFG" --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
  python diagnosis/infra/field_liveness.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/liveness.json" 2>&1 | tee -a "$LOG" || say "  WARN liveness"
  python diagnosis/infra/dx_seg_localise.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/seg_localise.json" --max_frames 30 --frame_stride 5 2>&1 | tee -a "$LOG" || say "  WARN seg_localise"
  # gradient-attribution: WHO gets the update during deformation (the assume-nothing probe).
  # AFTER the others so a grad crash can't starve them; writes to the LWORK mirror so it ships + is summarised.
  python diagnosis/infra/grad_attribution_probe.py --config "$CFG" --checkpoint "$CKPT" \
    --json "$LW/grad_attrib.json" --max_frames 12 --frame_stride 3 2>&1 | tee -a "$LOG" || say "  WARN grad_attrib"
  local BOUND; BOUND=$(python -c "import json; from config import load_config; print(json.dumps(load_config('$CFG')['mapping']['bound']))" 2>/dev/null)
  if [ -n "$BOUND" ]; then
    python diagnosis/infra/field_viz3d.py --dx_dir "$LW/dx" --out "$LW/field3d.png" \
      --bound "$BOUND" --max_frames 40 2>&1 | tee -a "$LOG" || say "  WARN field_viz3d"
  else
    say "  WARN field_viz3d skipped (could not read bound from $CFG)"
  fi
}

nvidia-smi -L || true
cd "$REPO"; activate_dds_env
stage_semsup || { say "stage failed -- abort"; exit 1; }

# ---- PHASE A: SemSup 3 seeds (the gate + persisted checkpoints + 3D field) ----
run_semsup(){
  local LBL=$1; local CFG=configs/Super/pb7_redesign_${LBL}.yaml
  local OUTB=output/pb7_redesign_${LBL}; local DEMO="$OUTB/demo"
  local DST="$DRIVE/$LBL"; local LW="$LWORK/$LBL"
  done_marker "$DST" && { say "  $LBL shipped -- skip"; return 0; }
  mkdir -p "$LW" "$DST"
  if ! ls "$DEMO"/checkpoint*.pt >/dev/null 2>&1; then
    say "=== TRAIN SemSup $LBL ($CFG) ==="; local T0=$(date +%s)
    python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN $LBL nonzero exit"
    say "  $LBL train $(( ($(date +%s)-T0)/60 )) min"
  fi
  local CKPT; CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  ERROR $LBL no ckpt"; return 1; }
  say "=== DIAGNOSE SemSup $LBL ==="; diagnose "$CFG" "$CKPT" "$LW"
  cp "$CKPT" "$LW/checkpoint150.pt" 2>/dev/null || say "  WARN ckpt copy"   # PERSIST (battery-7 didn't)
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  $LBL shipped (ckpt persisted)"
}
say "########## PHASE A: SemSup 3-seed gate ##########"
run_semsup s0; run_semsup s1; run_semsup s2

# ---- PHASE B: Inc-0 regression (Arm-2 plumbing == base) ----
say "########## PHASE B: Inc-0 regression ##########"
INC0="$DRIVE/inc0"; mkdir -p "$INC0"
if ! done_marker "$INC0"; then
  python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml \
    --write-golden 2>&1 | tee -a "$LOG" || say "  WARN golden"
  cp Addons/regression/golden_inc0.json "$INC0/" 2>/dev/null || true
  python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml \
    2>&1 | tee "$INC0/inc0_check.txt" || say "  WARN inc0 check"
  sync; touch "$INC0/.DONE"
fi

# ---- PHASE C: CRCD C_2/001 (best-effort, large motion + FREE pose) ----
say "########## PHASE C: CRCD C_2/001 (best-effort) ##########"
set +e   # isolate: a CRCD failure must NOT abort the shipped SemSup core
crcd_c2(){
  local DST="$DRIVE/crcd_c2_001"; local LW="$LWORK/crcd_c2_001"
  done_marker "$DST" && { say "  CRCD c2 shipped -- skip"; return 0; }
  local CFG=configs/CRCD/c2_001_paperfaith_lrfix.yaml
  local STAGED="$REPO/data/CRCD/C2_001"
  local SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/C_2/snippet_001
  local CALIB=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
  local MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_2/snippet_001/depth
  if [ ! -d "$STAGED/video_frames" ]; then
    [ -d "$SNIP" ] && [ -f "$CALIB" ] && [ -d "$MOGE" ] || { say "  CRCD c2 prereqs MISSING (snippet/calib/MoGe depth) -- SKIP"; echo SKIP_PREREQ > "$DST/.SKIP" 2>/dev/null; mkdir -p "$DST"; echo SKIP_PREREQ > "$DST/.SKIP"; return 0; }
    say "  staging C_2/001 ..."; mkdir -p "$LW"
    python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SNIP" --calib_pkl "$CALIB" --output_dir "${STAGED}.tmp" 2>&1 | tee -a "$LOG" \
      && mv "${STAGED}.tmp" "$STAGED" || { say "  CRCD preprocess FAILED -- SKIP"; mkdir -p "$DST"; echo FAIL_PREPROCESS > "$DST/.SKIP"; return 0; }
    mkdir -p "$STAGED/depth.tmp"; python3 - "$MOGE" "$STAGED/depth.tmp" <<'PYEOF'
import os,sys,shutil
src,dst=sys.argv[1],sys.argv[2]
fs=sorted(f for f in os.listdir(src) if f.endswith('.png'))
for i,f in enumerate(fs): shutil.copy2(os.path.join(src,f),os.path.join(dst,f'{i:06d}.png'))
print('copied',len(fs),'MoGe depth')
PYEOF
    rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"
  fi
  mkdir -p "$LW" "$DST"
  local OUTB=output/CRCD/C2_001_paperfaith_lrfix; local DEMO="$OUTB/demo"
  if ! ls "$DEMO"/checkpoint*.pt >/dev/null 2>&1; then
    say "  === TRAIN CRCD c2 ==="; local T0=$(date +%s)
    python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG"
    say "  CRCD c2 train $(( ($(date +%s)-T0)/60 )) min"
  fi
  local CKPT; CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "  CRCD c2 no ckpt -- SKIP diag"; echo NO_CKPT > "$DST/.SKIP"; return 0; }
  say "  === DIAGNOSE CRCD c2 ==="; diagnose "$CFG" "$CKPT" "$LW"
  cp "$CKPT" "$LW/checkpoint.pt" 2>/dev/null
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  sync; touch "$DST/.DONE"; say "  CRCD c2 shipped"
}
crcd_c2
set -uo pipefail

# ---- FINAL readout ----
say "########## SUMMARY ##########"
python3 - "$DRIVE" <<'PY' 2>&1 | tee -a "$LOG"
import sys, json, os, glob
DR = sys.argv[1]
print(f"\n{'run':<14}{'signal?':<9}{'A ratio':<9}{'out/in':<9}{'moves_t':<9}{'grad f/map_eff':<15}{'grad verdict':<26}")
print('-'*92)
for d in sorted(glob.glob(f'{DR}/*/')):
    name = os.path.basename(d.rstrip('/'))
    lw = f'/content/mapprobe/{name}'   # LWORK mirror holds the json (ship dir has the tgz)
    mp = os.path.join(lw, 'map_absorb.json')
    if not os.path.exists(mp): continue
    m = json.load(open(mp))
    f = json.load(open(os.path.join(lw, 'field3d.json'))) if os.path.exists(os.path.join(lw, 'field3d.json')) else {}
    g = json.load(open(os.path.join(lw, 'grad_attrib.json'))) if os.path.exists(os.path.join(lw, 'grad_attrib.json')) else {}
    print(f"{name:<14}{str(m.get('VERDICT_signal_exists')):<9}{str(m.get('A_moving_over_static_ratio')):<9}"
          f"{str(f.get('out_over_in_ratio','?')):<9}{str(m.get('D_render_moves_with_t')):<9}"
          f"{str(g.get('RATIO_field_over_map_effective','?')):<15}{str(g.get('VERDICT_tag','?')):<26}")
print("\nGATE 1 (is there signal?): signal?=True on any run -> GO build Inc-1. All False -> motion sub-SNR / map tracks it -> rethink.")
print("GATE 2 (who eats it?): grad verdict 'map-absorbs'/'map-favoured' -> THROTTLE-MAP is the fix; 'signal-not-reaching-field'")
print("  -> plumbing/anchor (not a race); 'race-or-shared' -> field gets the gradient, deadness is optimisation/ill-posedness.")
print("Per run: map_absorb.json (A/B/C/D), field3d.png (3D field), *_sweep.png (render-moves-with-t), grad_attrib.json (per-component grad).")
PY
say "=== overnight DONE $(date -Iseconds) ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
