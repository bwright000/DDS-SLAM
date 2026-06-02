#!/bin/bash
# ============================================================
# generate_6panel_sweep.sh — Generate the 6-panel comparison video for
# every DDS-SLAM Super sweep run.
#
# For each run we need:
#   Panel 1: Input RGB              (data/Super/trail_3/rgb/*left.png)
#   Panel 2: Rendered RGB           (<run_drive>/0000.jpg .. 0150.jpg)
#   Panel 3: Input Depth            (data/Super/trail_3/<depth_subdir>/*left_depth.npy)
#   Panel 4: Output Depth           (<run_drive>/depth/*.png)
#   Panel 5: Trajectory aligned     (<run_drive>/demo/est_c2w_data.txt + GT)
#   Panel 6: Trajectory raw         (same, no Horn alignment)
#
# Pre-flights each input before firing the video — skips runs with missing
# pieces and reports them at the end. Saves each .mp4 alongside the run on
# Drive AND in a single _videos/ collection folder for easy review.
#
# Usage (on Colab in dds_env):
#   cd /content/DDS-SLAM
#   source /tmp/dds_env/bin/activate
#   bash Addons/viz/generate_6panel_sweep.sh
#
# Override paths:
#   DDS_DATA_ROOT=/content/DDS-SLAM/data/Super/trail_3 \
#   DDS_SWEEP_ROOT=/content/drive/MyDrive/Outputs/ddsslam_super_depthsweep_20260602 \
#   bash Addons/viz/generate_6panel_sweep.sh
#
#   RUNS="trail3_paper_faithful" bash Addons/viz/generate_6panel_sweep.sh   # subset
# ============================================================
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------
# Paths (override via env)
# ---------------------------------------------------------------------
DDS_DATA_ROOT=${DDS_DATA_ROOT:-/content/DDS-SLAM/data/Super/trail_3}
# Auto-find latest sweep dir if not set
if [ -z "${DDS_SWEEP_ROOT:-}" ]; then
  DDS_SWEEP_ROOT=$(ls -d /content/drive/MyDrive/Outputs/ddsslam_super_depthsweep_* 2>/dev/null | sort | tail -1)
fi
if [ -z "$DDS_SWEEP_ROOT" ] || [ ! -d "$DDS_SWEEP_ROOT" ]; then
  echo "FATAL: DDS_SWEEP_ROOT not set or not a dir."
  echo "  Tried: $DDS_SWEEP_ROOT"
  echo "  Set explicitly: DDS_SWEEP_ROOT=/content/drive/MyDrive/Outputs/ddsslam_super_depthsweep_YYYYMMDD bash $0"
  exit 1
fi
VIDEO_COLLECTION="$DDS_SWEEP_ROOT/_videos"
mkdir -p "$VIDEO_COLLECTION"

GT_TRAJ="$DDS_DATA_ROOT/groundtruth.txt"

echo "=== 6-panel video sweep ==="
echo "  data root:  $DDS_DATA_ROOT"
echo "  sweep root: $DDS_SWEEP_ROOT"
echo "  videos to: $VIDEO_COLLECTION"
echo ""

# ---------------------------------------------------------------------
# Run -> depth_subdir map (from configs/Super/<run>.yaml depth_subdir field)
# ---------------------------------------------------------------------
declare -A DEPTH_SUBDIR
DEPTH_SUBDIR[trail3_variant_b_ep9]="depth/variant_b_ep9"
DEPTH_SUBDIR[trail3_variant_a_stereo]="depth/variant_a_stereo"
DEPTH_SUBDIR[trail3_variant_c_stereo]="depth/variant_c_stereo"
DEPTH_SUBDIR[trail3_moge2]="depth/moge2"
DEPTH_SUBDIR[trail3_variant_b_ep9_hash19]="depth/variant_b_ep9"
DEPTH_SUBDIR[trail3_paper_faithful]="depth/variant_a_stereo"

RUNS_DEFAULT="trail3_variant_b_ep9 trail3_variant_a_stereo trail3_variant_c_stereo trail3_moge2 trail3_variant_b_ep9_hash19 trail3_paper_faithful"
RUNS=${RUNS:-$RUNS_DEFAULT}

# ---------------------------------------------------------------------
# Per-run pre-flight: every input must exist with sane frame counts
# ---------------------------------------------------------------------
preflight_run() {
  local run="$1"
  local run_dir="$DDS_SWEEP_ROOT/$run"
  local depth_subdir="${DEPTH_SUBDIR[$run]:-}"

  if [ ! -d "$run_dir" ]; then
    echo "  MISS run dir not found: $run_dir"
    return 1
  fi
  if [ -z "$depth_subdir" ]; then
    echo "  MISS no depth_subdir mapping for $run (unknown config)"
    return 1
  fi

  local rgb_in="$DDS_DATA_ROOT/rgb"
  local depth_in="$DDS_DATA_ROOT/$depth_subdir"
  local rgb_out="$run_dir"
  local depth_out="$run_dir/depth"
  local traj_est="$run_dir/demo/est_c2w_data.txt"

  local n_rgb_in=$(ls "$rgb_in"/*left.png 2>/dev/null | wc -l)
  local n_depth_in=$(ls "$depth_in"/*left_depth.npy 2>/dev/null | wc -l)
  local n_rgb_out=$(ls "$rgb_out"/[0-9]*.jpg 2>/dev/null | wc -l)
  local n_depth_out=$(ls "$depth_out"/*.png 2>/dev/null | wc -l)
  local has_traj=0; [ -f "$traj_est" ] && has_traj=1
  local has_gt=0;   [ -f "$GT_TRAJ" ]  && has_gt=1

  echo "  rgb_in=$n_rgb_in depth_in=$n_depth_in rgb_out=$n_rgb_out depth_out=$n_depth_out traj=$has_traj gt=$has_gt"

  # All four image dirs must be non-empty AND est trajectory must exist
  if [ "$n_rgb_in" -eq 0 ] || [ "$n_depth_in" -eq 0 ] || [ "$n_rgb_out" -eq 0 ] || [ "$n_depth_out" -eq 0 ] || [ "$has_traj" -eq 0 ]; then
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------------
# Generate one video. Lays out 6 panels as 2x3 grid (auto by script).
# ---------------------------------------------------------------------
generate_one() {
  local run="$1"
  local run_dir="$DDS_SWEEP_ROOT/$run"
  local depth_subdir="${DEPTH_SUBDIR[$run]}"

  local rgb_in="$DDS_DATA_ROOT/rgb"
  local depth_in="$DDS_DATA_ROOT/$depth_subdir"
  local rgb_out="$run_dir"
  local depth_out="$run_dir/depth"
  local traj_est="$run_dir/demo/est_c2w_data.txt"

  local out_video="$run_dir/${run}_6panel.mp4"

  echo "  generating -> $out_video"
  python Addons/viz/generate_video.py \
    --rgb_input_dir   "$rgb_in"   --rgb_input_pattern '*left.png' \
    --rgb_output_dir  "$rgb_out"  --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$depth_in" \
    --depth_output_dir "$depth_out" \
    --trajectory_est  "$traj_est" \
    --trajectory_gt   "$GT_TRAJ" \
    --trajectory_raw \
    --output "$out_video" \
    --fps 15 \
    --panel_height 360 --panel_width 480
  local rc=$?

  if [ "$rc" -eq 0 ] && [ -f "$out_video" ] && [ "$(stat -c%s "$out_video" 2>/dev/null || echo 0)" -gt 100000 ]; then
    cp "$out_video" "$VIDEO_COLLECTION/${run}_6panel.mp4"
    echo "  DONE -> $out_video (also copied to $VIDEO_COLLECTION/)"
    return 0
  else
    echo "  FAIL rc=$rc, video missing or tiny"
    return 1
  fi
}

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
PASS=""; FAIL=""; SKIP=""
for run in $RUNS; do
  echo "--- $run ---"
  if preflight_run "$run"; then
    if generate_one "$run"; then
      PASS="$PASS $run"
    else
      FAIL="$FAIL $run"
    fi
  else
    SKIP="$SKIP $run"
  fi
  echo ""
done

echo "=== summary ==="
echo "  PASS:$PASS"
echo "  FAIL:$FAIL"
echo "  SKIP (preflight):$SKIP"
echo ""
echo "All videos collected at: $VIDEO_COLLECTION"
ls -la "$VIDEO_COLLECTION" 2>/dev/null
