#!/bin/bash
# ============================================================================
# manual_phase3_finish.sh — finish Phase 3 manually when MoGe-2's [3/3] save
# loop completed but the script OOM-killed during its post-save stats print.
#
# Idempotent. Picks up from NPYs already on disk; skips already-converted PNGs.
# After this completes, re-fire the main runbook and it skips Phases 1/2/3 via
# sentinels and goes straight to Phase 4 (CRCD paper_faithful SLAM).
# ============================================================================
set -u
S=/content/DDS-SLAM/data/CRCD/F1_002
EXPECTED=1287

echo "=== Step 1: verify NPYs ==="
NPY=$(ls $S/_moge_npy/*-left_depth.npy 2>/dev/null | wc -l)
echo "  _moge_npy NPY count: $NPY / $EXPECTED"
if [ "$NPY" -ne "$EXPECTED" ]; then
  echo "  ABORT: NPY count mismatch -- re-run MoGe inference"
  exit 1
fi

echo "=== Step 2: NPY -> uint16 PNG conversion ==="
cd $S
mkdir -p depth.tmp
python3 - <<'PYEOF'
import numpy as np, cv2, glob, os
n_in = sorted(glob.glob('_moge_npy/*-left_depth.npy'))
print(f'  converting {len(n_in)} NPYs -> PNGs')
done_count = 0
for i, p in enumerate(n_in):
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/{fid}.png'
    if os.path.exists(out):
        done_count += 1
        continue
    d = np.load(p).astype(np.float32)
    # Atomic write needs the SAME extension as the final file (cv2.imwrite
    # picks the encoder by extension). Use a hidden temp filename ending
    # in .png, then os.replace to the final visible name.
    tmp = f'depth.tmp/.{fid}.tmp.png'
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, out)
    done_count += 1
    if (i + 1) % 100 == 0:
        print(f'  {i+1}/{len(n_in)}')
print(f'  done: png_out={len(glob.glob("depth.tmp/*.png"))}')
PYEOF

echo "=== Step 3: atomic promote depth.tmp -> depth ==="
PNG=$(ls depth.tmp/*.png | wc -l)
echo "  png_count=$PNG expected=$EXPECTED"
if [ "$PNG" -ne "$EXPECTED" ]; then
  echo "  ABORT: PNG count mismatch"
  exit 1
fi
rm -rf depth
mv depth.tmp depth
sync
touch depth/.DONE
sync
rm -rf _moge_in _moge_npy
echo "  depth/ now has $(ls depth/*.png | wc -l) PNGs + .DONE sentinel"

echo "=== Step 4: persist depth to CRCD-Published snippet on Drive ==="
python3 - <<'PYEOF'
import os, shutil
STAGED = '/content/DDS-SLAM/data/CRCD/F1_002'
SRC = '/content/drive/MyDrive/Datasets/CRCD-Published/F_1/snippet_002'
DST = f'{SRC}/depth'
os.makedirs(DST, exist_ok=True)
rgb_orig = sorted(f for f in os.listdir(f'{SRC}/rgb') if f.endswith('.png'))
copied = skipped = 0
for i, orig in enumerate(rgb_orig):
    src = f'{STAGED}/depth/{i:06d}.png'
    if not os.path.exists(src):
        continue
    dst = f'{DST}/{orig}'
    if os.path.exists(dst):
        skipped += 1
        continue
    shutil.copy2(src, dst)
    copied += 1
    if (i + 1) % 200 == 0:
        print(f'  {i+1}/{len(rgb_orig)} (copied={copied} skipped={skipped})')
print(f'  final: copied={copied} skipped={skipped} total_on_drive={len(os.listdir(DST))}')
PYEOF

echo ""
echo "=== Phase 3 manual finish complete ==="
echo "  Next: fire the runbook to start Phase 4 (CRCD paper_faithful SLAM):"
echo "    nohup bash /content/DDS-SLAM/Addons/colab/run_crcd_f1_002.sh > /content/runbook_nohup.out 2>&1 &"
echo "    disown \$!"
