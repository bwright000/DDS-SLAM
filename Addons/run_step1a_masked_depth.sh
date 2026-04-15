#!/usr/bin/env bash
# Step 1a of troubleshooting plan elegant-fluttering-valley:
# regenerate StereoMIS P2_1 depth with specularity + instrument masking,
# at mm quantization (scale=1000), then run DDS-SLAM and print ATE.
#
# Run on Colab from /content/DDS-SLAM after activating the venv:
#   bash Addons/run_step1a_masked_depth.sh
#
# Assumes /content/p2_1_local/ already populated per CLAUDE.local.md
# (video_frames/, depth/, masks/, groundtruth.txt, StereoCalibration.ini).

set -euo pipefail

SRC=/content/p2_1_local
DST=/content/p2_1_masked
EXP=demo_masked_mm

# 1. Build masked basedir: regen depth at mm scale, symlink the rest.
mkdir -p "$DST/depth"
for sub in video_frames masks groundtruth.txt StereoCalibration.ini; do
  [ -e "$DST/$sub" ] || ln -s "$SRC/$sub" "$DST/$sub"
done

echo "=== Regenerating depth (scale 100 -> 1000, masked) ==="
python Addons/regenerate_stereomis_depth.py \
  --depth_in  "$SRC/depth" \
  --rgb_dir   "$SRC/video_frames" \
  --mask_dir  "$SRC/masks" \
  --depth_out "$DST/depth" \
  --in_scale  100 \
  --out_scale 1000

# 2. Point dataset at the new basedir + bump png_depth_scale to 1000.
#    Use a throwaway config copy so the original yamls stay clean.
mkdir -p configs/StereoMIS/_step1a
cp configs/StereoMIS/stereomis.yaml configs/StereoMIS/_step1a/stereomis.yaml
cp configs/StereoMIS/p2_1.yaml      configs/StereoMIS/_step1a/p2_1.yaml
sed -i 's|^  png_depth_scale: 100.*|  png_depth_scale: 1000|' \
       configs/StereoMIS/_step1a/stereomis.yaml
sed -i "s|^inherit_from: .*|inherit_from: configs/StereoMIS/_step1a/stereomis.yaml|" \
       configs/StereoMIS/_step1a/p2_1.yaml
sed -i "s|^  datadir: .*|  datadir: $DST|"     configs/StereoMIS/_step1a/p2_1.yaml
sed -i "s|^  exp_name: .*|  exp_name: $EXP|"   configs/StereoMIS/_step1a/p2_1.yaml

# 3. Run DDS-SLAM.
echo "=== Running DDS-SLAM with masked mm depth ==="
python ddsslam.py --config configs/StereoMIS/_step1a/p2_1.yaml

# 4. Print ATE summary.
OUT="output/StereoMIS/P2_1/$EXP"
echo "=== Result: $OUT ==="
[ -f "$OUT/output.txt" ] && tail -n 30 "$OUT/output.txt"
ls "$OUT"/pose_r_*.png 2>/dev/null | tail -n 3
