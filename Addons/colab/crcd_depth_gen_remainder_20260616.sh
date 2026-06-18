#!/bin/bash
# ============================================================================
# CRCD depth-gen — REMAINDER snippets (the 16 not yet done). 2026-06-16.
# DEPTH ONLY (no SLAM). Reuses the EXACT methodology that produced the 4 existing
# snippets (run_crcd_4snippets.sh Phases 1-1.6): MoGe-2 metric depth PNG (x10000)
# + per-snippet stereo anchor = StereoSGBM on frame 0 -> sc_factor = median(stereo/MoGe).
# Output cached to Drive CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth/*.png,.sc_factor}
# so future SLAM runs rehydrate it (same cache the 4-snippet runbook reads).
#
# Run in a 3rd terminal alongside the SLAM (GPU mem has room; it competes for compute,
# so threads are capped). Resume-safe (.DONE per snippet cache). Smallest-first.
#   PARALLEL not used (sequential MoGe; polite to the running SLAM). Override threads: THREADS=2 bash ...
# ============================================================================
set -uo pipefail
# 🚨 DEPRECATED 2026-06-18: this RECTIFIES CRCD frames for depth. Policy = RAW-LEFT for ALL CRCD.
[ "${ALLOW_RECTIFIED:-0}" = 1 ] || { echo "🚨 crcd_depth_gen_remainder is DEPRECATED (rectified depth). RAW-LEFT replacement: crcd_depth_rawleft_all_20260618.sh. ALLOW_RECTIFIED=1 to force a deliberate rectified run." >&2; exit 1; }
DATE=$(date +%Y%m%d)
THREADS=${THREADS:-2}
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
DDS_DIR=/content/DDS-SLAM
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
DRIVE_CACHE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2   # where the 4-snippet runbook rehydrates
LOG=/content/drive/MyDrive/Outputs/crcd_depth_gen_${DATE}.log; mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
say "=== CRCD depth-gen (remainder) start $(date -Iseconds)  THREADS=$THREADS ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
[ -f "$CALIB_PKL" ] || { say "FATAL: calib pickle missing at $CALIB_PKL"; exit 1; }

# Snippet table: NAME EP SID FRAMES  (smallest-first for resilience). F_3/007 + C/F done already.
SNIPPETS=(
  "F3_004 F_3 004 200"   "F3_005 F_3 005 200"   "E3_003 E_3 003 263"  "E3_005 E_3 005 265"
  "E3_002 E_3 002 278"   "F3_002 F_3 002 312"   "E3_004 E_3 004 360"  "F3_006 F_3 006 387"
  "F3_003 F_3 003 400"   "E3_001 E_3 001 480"   "F3_001 F_3 001 602"  "B2_001 B_2 001 1321"
  "G2_003 G_2 003 1321"  "C3_001 C_3 001 1527"  "G3_001 G_3 001 1987" "E1_001 E_1 001 2108"
)

activate(){
  cd "$DDS_DIR"
  python -c "import torch; assert torch.cuda.is_available()" || { say "FATAL: env not ready (run the SLAM env first)"; exit 1; }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
  if ! python -c 'from moge.model.v2 import MoGeModel' 2>/dev/null; then
    say "installing MoGe-2..."; pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -3
    python -c 'from moge.model.v2 import MoGeModel' || { say "FATAL: MoGe-2 not importable"; exit 1; }
  fi
}
copy_item(){ local SRC=$1 DST=$2 L=$3
  [ -e "$SRC" ] || { echo "FATAL missing $SRC"; return 1; }
  [ -e "$DST" ] && { echo "  $L exists -> skip"; return 0; }
  cp -r "$SRC" "$DST" || rsync -a --partial "$SRC" "$DST" || { echo "FATAL copy $L"; return 1; }
}

activate

for ROW in "${SNIPPETS[@]}"; do
  read -r NAME EP SID FRAMES <<< "$ROW"
  echo ""; echo "############ $NAME ($EP/snippet_$SID, $FRAMES frames) ############"
  RAW=/content/crcd_raw/${EP}_snippet_${SID}
  STAGED=$DDS_DIR/data/CRCD/${NAME}
  CACHE=$DRIVE_CACHE_ROOT/$EP/snippet_$SID
  # resume: cached depth complete?
  if [ -f "$CACHE/depth/.DONE" ] && [ -f "$CACHE/.sc_factor" ] \
     && [ "$(ls "$CACHE/depth"/*.png 2>/dev/null | wc -l)" -ge "$FRAMES" ]; then
    say "  $NAME cache complete on Drive -> skip"; continue; fi

  # ---- stage raw ----
  say "[$NAME 1] stage raw"
  if [ ! -f "$RAW/.STAGED" ]; then
    DRIVE_TAR=$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar
    mkdir -p "$RAW"
    if [ -f "$DRIVE_TAR" ]; then tar xf "$DRIVE_TAR" -C "$RAW" || { say "  tar extract FAIL -> skip $NAME"; rm -rf "$RAW"; continue; }
    else DSRC=$DRIVE_CRCD/$EP/snippet_$SID
      [ -d "$DSRC" ] || { say "  raw NOT on Drive ($DSRC) -> skip $NAME"; continue; }
      copy_item "$DSRC/rgb" "$RAW/rgb" rgb && copy_item "$DSRC/rgbright" "$RAW/rgbright" rgbright \
        && copy_item "$DSRC/semantic_instance" "$RAW/semantic_instance" sem \
        && copy_item "$DSRC/groundtruth.txt" "$RAW/groundtruth.txt" gt \
        && copy_item "$DSRC/intrinsics.yaml" "$RAW/intrinsics.yaml" intr || { say "  stage FAIL -> skip $NAME"; continue; }
    fi
    touch "$RAW/.STAGED"
  fi

  # ---- preprocess (rectify) ----
  say "[$NAME 2] preprocess (rectify)"
  if [ ! -f "$STAGED/.PREPROCESSED" ]; then
    rm -rf "$STAGED"; mkdir -p "$STAGED"
    python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$RAW" --calib_pkl "$CALIB_PKL" --output_dir "$STAGED" \
      || { say "  preprocess FAIL -> skip $NAME"; continue; }
    [ -f "$STAGED/groundtruth.txt" ] || cp "$RAW/groundtruth.txt" "$STAGED/groundtruth.txt"
    touch "$STAGED/.PREPROCESSED"
  fi
  [ -f "$STAGED/rectified_calib.txt" ] || { say "  rectified_calib.txt missing -> skip $NAME"; continue; }
  N_L=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  say "  rectified left frames: $N_L"

  # ---- MoGe-2 metric depth (PNG x10000) ----
  say "[$NAME 3] MoGe-2 depth gen ($N_L frames)"
  if [ ! -f "$STAGED/depth/.DONE" ] || [ "$(ls "$STAGED/depth"/*.png 2>/dev/null | wc -l)" -lt "$N_L" ]; then
    cd "$STAGED"; rm -rf _moge_in _moge_npy depth.tmp; mkdir -p _moge_in _moge_npy depth.tmp
    for f in video_frames/*l.png; do fid=$(basename "$f" l.png); ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"; done
    python "$DDS_DIR/Addons/depth/generate_depth_moge.py" --rgb _moge_in --out _moge_npy \
      --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 || { say "  MoGe gen FAIL -> skip $NAME"; cd "$DDS_DIR"; continue; }
    python - <<'PYEOF'
import numpy as np, cv2, glob, os
for p in sorted(glob.glob('_moge_npy/*-left_depth.npy')):
    fid=os.path.basename(p).split('-')[0]; out=f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    cv2.imwrite(out, np.clip(np.load(p).astype(np.float32),0,65535).astype(np.uint16))
print('npy->png:', len(glob.glob('depth.tmp/*.png')))
PYEOF
    PNG=$(ls depth.tmp/*.png 2>/dev/null | wc -l)
    [ "$PNG" -ge "$N_L" ] || { say "  depth count $PNG < $N_L -> skip $NAME"; cd "$DDS_DIR"; continue; }
    rm -rf depth && mv depth.tmp depth; sync; touch depth/.DONE
    rm -rf _moge_in _moge_npy; cd "$DDS_DIR"
  fi

  # ---- stereo anchor: StereoSGBM on frame 0 -> sc_factor (median stereo/MoGe) ----
  say "[$NAME 4] stereo-anchor sc_factor (frame-0 SGBM)"
  if [ ! -f "$STAGED/.sc_factor" ]; then
    python - "$STAGED" <<'PYEOF'
import cv2, numpy as np, os, sys, glob
STAGED=sys.argv[1]
calib={}
for line in open(f'{STAGED}/rectified_calib.txt'):
    k,v=line.strip().split(); calib[k]=float(v)
baseline_m=calib['baseline_m']; fx=calib['fx']
left_path=sorted(glob.glob(f'{STAGED}/video_frames/*l.png'))[0]
fid=os.path.basename(left_path).replace('l.png','')
left=cv2.imread(left_path,cv2.IMREAD_GRAYSCALE)
right=cv2.imread(f'{STAGED}/video_frames/{fid}r.png',cv2.IMREAD_GRAYSCALE)
moge=cv2.imread(f'{STAGED}/depth/{fid}.png',cv2.IMREAD_UNCHANGED)
if left is None or right is None or moge is None: print('FATAL read frame0',file=sys.stderr); sys.exit(1)
moge_m=moge.astype(np.float32)/10000.0; mvalid=moge_m>0.01
sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=7,P1=8*49,P2=32*49,
    disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=32,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp=sgbm.compute(left,right).astype(np.float32)/16.0; vs=disp>0.5
if vs.sum()<1000: print('FATAL few stereo px',file=sys.stderr); sys.exit(2)
sd=np.zeros_like(disp); sd[vs]=baseline_m*fx/disp[vs]
vj=vs&(sd>0.05)&(sd<3.0)&mvalid
if vj.sum()<500: print('FATAL few joint px',file=sys.stderr); sys.exit(2)
sc=float(np.median(sd[vj]/moge_m[vj]))
print(f'  stereo median={np.median(sd[vj]):.3f}m MoGe median={np.median(moge_m[vj]):.3f}m -> sc_factor={sc:.4f}')
open(f'{STAGED}/.sc_factor','w').write(f'{sc:.6f}\n')
PYEOF
    [ -f "$STAGED/.sc_factor" ] || { say "  stereo anchor FAIL -> skip $NAME"; continue; }
  fi
  SC=$(cat "$STAGED/.sc_factor"); say "  sc_factor = $SC"

  # ---- cache depth + sc_factor to Drive (future SLAM rehydrates this) ----
  say "[$NAME 5] cache to Drive: $CACHE"
  mkdir -p "$CACHE/depth"
  cp -n "$STAGED/depth"/*.png "$CACHE/depth/" 2>/dev/null
  cp "$STAGED/.sc_factor" "$CACHE/.sc_factor"
  sync; touch "$CACHE/depth/.DONE"; sync
  say "  $NAME DONE: $(ls "$CACHE/depth"/*.png | wc -l) PNGs + sc_factor=$SC cached"
done

say "=== CRCD depth-gen (remainder) DONE $(date -Iseconds) ==="
echo "Cached under: $DRIVE_CACHE_ROOT/<EP>/snippet_<SID>/{depth,.sc_factor}"
echo "sc_factor summary:"
for ROW in "${SNIPPETS[@]}"; do read -r NAME EP SID _ <<< "$ROW"
  F=$DRIVE_CACHE_ROOT/$EP/snippet_$SID/.sc_factor
  [ -f "$F" ] && printf '  %-8s sc_factor=%s\n' "$NAME" "$(cat "$F")" || printf '  %-8s (not done)\n' "$NAME"
done
