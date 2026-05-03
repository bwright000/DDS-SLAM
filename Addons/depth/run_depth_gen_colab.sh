#!/bin/bash
# =============================================================================
# Paper-faithful StereoMIS P2_1 depth generation — Colab runner
# =============================================================================
# Generates RAFT-based depth maps at scale=100 AND scale=10000 in one pass.
# Both pipelines use identical config (RPE PoseNet.flow2depth + spec/instrument
# masks); only the uint16 PNG quantization differs. See PIPELINE_REFERENCE.md.
#
# Usage on Colab:
#   bash run_depth_gen_colab.sh              # uses VERSION=v1
#   VERSION=v2 bash run_depth_gen_colab.sh   # overrides for re-run
#
# Output naming scheme (simple, sortable, version-bumpable):
#   /content/drive/MyDrive/Datasets/StereoMisPP/p2_1_depth_<VERSION>/depth_s100/
#   /content/drive/MyDrive/Datasets/StereoMisPP/p2_1_depth_<VERSION>/depth_s10000/
#   /content/drive/MyDrive/Datasets/StereoMisPP/p2_1_depth_<VERSION>/MANIFEST.txt
# =============================================================================

set -euo pipefail

VERSION="${VERSION:-v1}"
SRC_DRIVE="/content/drive/MyDrive/Datasets/StereoMisPP/P2_1"
DST_DRIVE="/content/drive/MyDrive/Datasets/StereoMisPP/p2_1_depth_${VERSION}"
LOCAL_DATA="/content/p2_1_local"
RPE_DIR="/tmp/robust-pose-estimator"
DDS_DIR="/content/DDS-SLAM"
GEN="${DDS_DIR}/Addons/depth/generate_depth_stereo.py"

echo "============================================================"
echo "Depth Pipeline Run: VERSION=${VERSION}"
echo "  Source: ${SRC_DRIVE}"
echo "  Output: ${DST_DRIVE}"
echo "============================================================"

# ----- 1. Pre-flight checks -----
if [ -d "${DST_DRIVE}" ]; then
  echo "ERROR: ${DST_DRIVE} already exists. Bump VERSION or delete the existing dir."
  exit 1
fi
[ -d "${SRC_DRIVE}" ] || { echo "ERROR: source ${SRC_DRIVE} not mounted/found"; exit 1; }
[ -f "${GEN}" ]      || { echo "ERROR: ${GEN} not found (DDS-SLAM repo missing?)"; exit 1; }

# ----- 2. Stage data locally (avoid Drive FUSE drops) -----
if [ ! -f "${LOCAL_DATA}/StereoCalibration.ini" ]; then
  echo ">>> Staging P2_1 data to ${LOCAL_DATA}..."
  mkdir -p "${LOCAL_DATA}"
  cd "${SRC_DRIVE}"
  tar cf - video_frames masks StereoCalibration.ini groundtruth.txt | tar xf - -C "${LOCAL_DATA}"
  cd /content
else
  echo ">>> Local data already staged at ${LOCAL_DATA}, skipping copy"
fi

NUM_FRAMES=$(ls "${LOCAL_DATA}/video_frames"/*l.png 2>/dev/null | wc -l)
NUM_MASKS=$(ls "${LOCAL_DATA}/masks"/*.png 2>/dev/null | wc -l)
echo "    Frames: ${NUM_FRAMES}    Masks: ${NUM_MASKS}"

# ----- 3. Clone robust-pose-estimator with submodules -----
if [ ! -f "${RPE_DIR}/core/RAFT/core/raft.py" ]; then
  echo ">>> Cloning robust-pose-estimator with submodules..."
  rm -rf "${RPE_DIR}"
  git clone --recurse-submodules https://github.com/aimi-lab/robust-pose-estimator.git "${RPE_DIR}"
else
  echo ">>> RPE already at ${RPE_DIR}, skipping clone"
fi

CKPT="${RPE_DIR}/trained/poseNet_2xf8up4b.pth"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint missing at ${CKPT}"; exit 1; }
echo "    Checkpoint: $(ls -la ${CKPT} | awk '{print $5}') bytes"

# ----- 4. Generate scale=100 (coarse, 10mm/level) -----
OUT_S100="${LOCAL_DATA}/depth_s100"
echo ""
echo ">>> [1/2] Generating depth at scale=100 ..."
echo "    Output: ${OUT_S100}"
python "${GEN}" \
    --datadir "${LOCAL_DATA}" \
    --depth_scale 100 \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_S100}" \
  2>&1 | tee "/content/depth_gen_s100_${VERSION}.log"

# ----- 5. Generate scale=10000 (fine, 0.1mm/level) -----
OUT_S10K="${LOCAL_DATA}/depth_s10000"
echo ""
echo ">>> [2/2] Generating depth at scale=10000 ..."
echo "    Output: ${OUT_S10K}"
python "${GEN}" \
    --datadir "${LOCAL_DATA}" \
    --depth_scale 10000 \
    --checkpoint "${CKPT}" \
    --output_dir "${OUT_S10K}" \
  2>&1 | tee "/content/depth_gen_s10000_${VERSION}.log"

# ----- 6. Verify both outputs -----
echo ""
echo ">>> Verifying outputs..."
python <<PYEOF
import cv2, glob, numpy as np, sys
ok = True
for d, scale in [("${OUT_S100}", 100), ("${OUT_S10K}", 10000)]:
    files = sorted(glob.glob(f"{d}/*.png"))
    if len(files) != ${NUM_FRAMES}:
        print(f"FAIL {d}: {len(files)} files, expected ${NUM_FRAMES}")
        ok = False
        continue
    sample = files[len(files)//2]
    depth = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
    valid = depth[depth > 0]
    if len(valid) == 0:
        print(f"FAIL {d}: empty depth in {sample}")
        ok = False
        continue
    metres_min = valid.min() / scale
    metres_max = valid.max() / scale
    unique = len(np.unique(valid))
    zero_pct = (depth == 0).sum() / depth.size * 100
    expected_metres_max = 0.30  # endo depth typically ≤ 0.25 m
    if not (0.01 < metres_min < expected_metres_max and 0.01 < metres_max < expected_metres_max):
        print(f"WARN {d}: metres range [{metres_min:.4f}, {metres_max:.4f}] outside expected band")
    print(f"OK   {d}: files={len(files)}, sample={sample.split('/')[-1]}, "
          f"metres=[{metres_min:.4f}, {metres_max:.4f}], unique={unique}, zero%={zero_pct:.1f}")
sys.exit(0 if ok else 1)
PYEOF

# ----- 7. Archive to Drive -----
echo ""
echo ">>> Archiving to Drive: ${DST_DRIVE}"
mkdir -p "${DST_DRIVE}"
cp -r "${OUT_S100}" "${DST_DRIVE}/"
cp -r "${OUT_S10K}" "${DST_DRIVE}/"
cp "/content/depth_gen_s100_${VERSION}.log" "${DST_DRIVE}/"
cp "/content/depth_gen_s10000_${VERSION}.log" "${DST_DRIVE}/"

# Manifest documents what was generated, for future reference
cat > "${DST_DRIVE}/MANIFEST.txt" <<EOF
Paper-faithful StereoMIS P2_1 depth maps
Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Version: ${VERSION}
Source: ${SRC_DRIVE}
Frames: ${NUM_FRAMES}
Masks (instrument): ${NUM_MASKS}

Pipeline:
  - robust-pose-estimator (aimi-lab/robust-pose-estimator @ HEAD)
  - PoseNet.flow2depth() with poseNet_2xf8up4b.pth checkpoint
  - depth_clipping = [1, 250] mm (scale = 1/250)
  - Specularity mask: img.sum < 3*255*0.96, dilate 11x11
  - Instrument mask: from masks/<frame>l.png (mask>0 = valid)
  - Output: uint16 PNG, value = depth_meters * depth_scale

Contents:
  depth_s100/   — 26 distinct values per frame, 10 mm resolution
  depth_s10000/ — ~2500 levels per frame, 0.1 mm resolution

Checkpoint: $(md5sum ${CKPT} | awk '{print $1}')
RPE commit: $(cd ${RPE_DIR} && git rev-parse --short HEAD)
DDS commit: $(cd ${DDS_DIR} 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "n/a")

Use in Co-SLAM/DDS-SLAM yaml:
  cam.png_depth_scale: 100   (with depth_s100/)
  cam.png_depth_scale: 10000 (with depth_s10000/)
EOF

echo ""
echo "============================================================"
echo "DONE. Both depth datasets generated and archived."
echo "  ${DST_DRIVE}/depth_s100/    ($(du -sh ${DST_DRIVE}/depth_s100 | awk '{print $1}'))"
echo "  ${DST_DRIVE}/depth_s10000/  ($(du -sh ${DST_DRIVE}/depth_s10000 | awk '{print $1}'))"
echo "  ${DST_DRIVE}/MANIFEST.txt"
echo "============================================================"
