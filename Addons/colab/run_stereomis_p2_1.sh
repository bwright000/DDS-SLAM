#!/bin/bash
# ============================================================================
# DDS-SLAM StereoMIS P2_1 last-4000 — two-variant run on the paper benchmark.
#
# Variants:
#   T0_literal = literal upstream stereomis.yaml + p2_1.yaml (hash=16,
#                voxel_sdf=0.002, iters=10, first_iters=200, lr=1e-3).
#                Matches DDS-SLAM-Base verbatim aside from our LOCAL camera
#                intrinsics (fx=560.07 vs upstream 516.95; we use local
#                because data was rectified with the local calibration).
#   T1_hash19  = T0 + grid.hash_size 16->19 ONLY.  Tests whether the SemSup
#                paper-faithful hash-bump transfers to StereoMIS.  voxel_sdf
#                INTENTIONALLY kept at 0.002 (StereoMIS bound is too large
#                for finer voxels per playbook).
#
# Paper target: ATE 8.3 mm (Table II).  Previous attempts hit ~50 mm.
# This run gates whether the gap is config-fixable.
#
# Wall on A100: ~80 min/variant + ~5 min eval each = ~3 hr total.
# Wall on T4:   ~5-6 hr/variant (not recommended for paper target).
#
# Data source on Drive: MyDrive/Datasets/StereoMisPP/P2_1/  (per CLAUDE.local.md).
# Sentinel-gated, resumable.
# ============================================================================
set -euo pipefail
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_stereomis_p2_1_${DATE}
mkdir -p "$DRIVE_ROOT"/{T0_literal,T1_hash19,_eval}
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== runbook start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }

STAGED=/content/DDS-SLAM/data/P2_1
DRIVE_DATA=/content/drive/MyDrive/Datasets/StereoMisPP/P2_1

activate_dds_env() {
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    echo "modern stack missing -- rebuild"
    bash /content/DDS-SLAM/Addons/env/colab_setup.sh --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { echo "env check FAIL"; exit 1; }
  if ! python -c "import lpips" 2>/dev/null; then
    echo "  installing lpips..."
    pip install -q lpips || echo "  WARN: lpips install failed; LPIPS will be missing from summary"
    python -c "import lpips" 2>/dev/null || echo "  WARN: lpips still not importable; LPIPS empty"
  fi
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

ship_to_drive() {
  local SRC=$1 DST=$2
  tar czf "$DST/payload.tgz.partial" -C "$SRC" .
  mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  mark_done "$DST"
}

# Returns 0 on full success, 2 on SLAM crash, 3 on ship-only failure.
run_variant() {
  local NAME=$1 OUT=$2 DST=$3
  if done_marker "$DST"; then
    echo "  $NAME already shipped -- skip"
    return 0
  fi
  echo "=== run $NAME ==="
  T0=$(date +%s)
  cd /content/DDS-SLAM
  if ! python ddsslam.py --config "configs/StereoMIS/p2_1_${NAME}.yaml"; then
    echo "  $NAME SLAM exited non-zero. Likely OOM or config error."
    return 2
  fi
  echo "  $NAME SLAM elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  if ! ship_to_drive "$OUT" "$DST"; then
    echo "  $NAME ship failed. SLAM output kept at $OUT."
    return 3
  fi
  return 0
}

# ============================================================================
# PHASE 0 -- env + repo sync + GPU report
# ============================================================================
phase 0 "env check + repo sync"
[ -d /content/drive/MyDrive ] || { echo "Drive not mounted -- abort"; exit 1; }
cd /content/DDS-SLAM
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "  dirty tree -- skipping pull"
else
  git fetch && git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD) \
    || { echo "non-FF on remote -- abort"; exit 1; }
fi
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "  GPU: $GPU"
if [[ ! "$GPU" =~ A100 ]]; then
  echo "  WARN: non-A100 GPU. Expect ~5-6x slowdown on T4 (would push total wall to ~15 hr)."
  echo "        Continuing anyway."
fi
activate_dds_env

# ============================================================================
# PHASE 1 -- stage StereoMIS P2_1 rgb + masks + GT from Drive
# (depth/ NOT expected — we generate via MoGe-2 in Phase 1.5)
# ============================================================================
phase 1 "stage StereoMIS P2_1 rgb + masks + GT from Drive (depth gen separately)"
if [ -f "$STAGED/.STAGED" ]; then
  echo "  already staged"
else
  [ -d "$DRIVE_DATA" ] || { echo "FATAL: $DRIVE_DATA not on Drive (per CLAUDE.local.md path: MyDrive/Datasets/StereoMisPP/P2_1/)"; exit 3; }
  mkdir -p "$STAGED"
  echo "  copying from $DRIVE_DATA ..."
  # Stage rgb + masks + GT + calib.  DO NOT expect depth/ — we generate via MoGe.
  (cd "$DRIVE_DATA" && tar cf - video_frames masks groundtruth.txt StereoCalibration.ini 2>/dev/null) \
    | (cd "$STAGED" && tar xf -) \
    || { echo "FATAL: tar pipe failed (check $DRIVE_DATA has video_frames/, masks/, groundtruth.txt)"; exit 3; }
  touch "$STAGED/.STAGED"
fi
N_L=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' 2>/dev/null | wc -l)
N_M=$(find "$STAGED/masks" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
N_G=$(grep -cv '^#' "$STAGED/groundtruth.txt" 2>/dev/null || echo 0)
echo "  staged: left=$N_L  masks=$N_M  GT=$N_G"
if [ "$N_L" -lt 4000 ] || [ "$N_G" -lt 4000 ]; then
  echo "FATAL: insufficient frames staged (need >=4000 rgb left + GT, got L=$N_L G=$N_G)"
  exit 3
fi

# ============================================================================
# PHASE 1.5 -- generate MoGe-2 depth on the left rgb frames
# ============================================================================
phase 1.5 "MoGe-2 depth generation (4000 frames at 640x512, ~30-40 min on A100)"
EXPECTED=$N_L
ACTUAL_DEPTH=0
[ -d "$STAGED/depth" ] && ACTUAL_DEPTH=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
if [ -f "$STAGED/depth/.DONE" ] && [ "$ACTUAL_DEPTH" -ge "$EXPECTED" ]; then
  echo "  depth/ complete ($ACTUAL_DEPTH/$EXPECTED) -- skip MoGe gen"
else
  echo "  depth/ incomplete ($ACTUAL_DEPTH/$EXPECTED) -- running MoGe-2"
  # MoGe-2 requires torch>=2.0 (we have it via colab_setup.sh modern stack)
  python3 -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || {
    echo "  installing MoGe-2..."
    pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub 2>&1 | tail -5
    python3 -c 'from moge.model.v2 import MoGeModel' \
      || { echo "FATAL: MoGe-2 not importable after install"; exit 5; }
  }
  cd "$STAGED"
  mkdir -p _moge_in _moge_npy depth.tmp
  # Symlink: video_frames/NNNNNNl.png -> _moge_in/NNNNNN-left.png
  for f in video_frames/*l.png; do
    fid=$(basename "$f" l.png)
    [ -L "_moge_in/${fid}-left.png" ] || ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"
  done
  echo "  symlinks: $(ls _moge_in/ | wc -l)"
  # Run MoGe-2 (no temporal smoothing — memory: 4000 frames at 640x512 float32
  # = 5 GB peak; --temporal_window 1 keeps it under 4 GB).
  # No --ref provided -> MoGe-2's native metric depth saved at scale 10000.
  python3 /content/DDS-SLAM/Addons/depth/generate_depth_moge.py \
    --rgb _moge_in --out _moge_npy \
    --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 \
    || { echo "FATAL: MoGe-2 generation failed"; exit 5; }
  # Convert NPY -> uint16 PNG, named {fid}.png so dataset.py:121 glob hits
  python3 - <<'PYEOF'
import numpy as np, cv2, glob, os
n_in = sorted(glob.glob('_moge_npy/*-left_depth.npy'))
written = 0
for p in n_in:
    fid = os.path.basename(p).split('-')[0]
    out = f'depth.tmp/{fid}.png'
    if os.path.exists(out): continue
    d = np.load(p).astype(np.float32)
    tmp = f'depth.tmp/.{fid}.tmp.png'
    cv2.imwrite(tmp, np.clip(d, 0, 65535).astype(np.uint16))
    os.replace(tmp, out)
    written += 1
print(f'npy->png wrote {written} (of {len(n_in)} npy files)')
PYEOF
  NPY=$(ls _moge_npy/*-left_depth.npy 2>/dev/null | wc -l)
  PNG=$(ls depth.tmp/*.png 2>/dev/null | wc -l)
  if [ "$PNG" -ne "$NPY" ] || [ "$PNG" -lt "$EXPECTED" ]; then
    echo "FATAL: depth count mismatch (png=$PNG npy=$NPY expected>=$EXPECTED). Keeping intermediates."
    cd /content/DDS-SLAM
    exit 5
  fi
  rm -rf depth && mv depth.tmp depth
  sync; touch depth/.DONE; sync
  rm -rf _moge_in _moge_npy
  cd /content/DDS-SLAM
  # Optional: persist depth back to Drive for future re-use
  echo "  syncing depth back to Drive for future re-use"
  mkdir -p "$DRIVE_DATA/depth"
  cp -n "$STAGED/depth/"*.png "$DRIVE_DATA/depth/" 2>/dev/null || true
fi
N_D=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
echo "  depth ready: $N_D files"

# ============================================================================
# PHASE 2 -- pre-flight GT motion sanity check
# ============================================================================
phase 2 "GT motion profile sanity"
python3 - <<PYEOF
import numpy as np
GT = '$STAGED/groundtruth.txt'
rows=[]
with open(GT) as f:
    for line in f:
        line=line.strip()
        if not line or line.startswith('#'): continue
        v=line.split()
        if len(v) >= 8:
            rows.append([float(v[1]), float(v[2]), float(v[3])])
xyz = np.array(rows)[-4000:]   # match dataset.py:120 slice
d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
print(f'last-4000: {len(xyz)} GT poses')
print(f'  extent (mm): x={(xyz[:,0].max()-xyz[:,0].min())*1000:.1f}  '
      f'y={(xyz[:,1].max()-xyz[:,1].min())*1000:.1f}  '
      f'z={(xyz[:,2].max()-xyz[:,2].min())*1000:.1f}')
print(f'  total path: {d.sum():.1f} mm')
print(f'  per-frame: median={np.median(d):.3f}  mean={d.mean():.3f}  max={d.max():.3f} mm/frame')
active = d > 0.001
print(f'  active fraction: {100*active.mean():.1f}%')
print(f'  active median:  {np.median(d[active]):.3f} mm/frame  (vs 1mm noise = {np.median(d[active])/1.0:.2f}x)')
print(f'  active max:     {d[active].max():.3f} mm/frame  ({d[active].max()/1.0:.2f}x noise)')
PYEOF

# ============================================================================
# PHASE 3 -- T0_literal SLAM
# ============================================================================
phase 3 "T0_literal (upstream stereomis.yaml + p2_1.yaml verbatim, local intrinsics)"
T0_RC=0
run_variant T0_literal output/StereoMIS/P2_1_T0_literal "$DRIVE_ROOT/T0_literal" || T0_RC=$?
case "$T0_RC" in
  2) echo "FAILED_SLAM" > "$DRIVE_ROOT/T0_literal/.FAILED" ;;
  3) echo "FAILED_SHIP" > "$DRIVE_ROOT/T0_literal/.FAILED" ;;
esac

# ============================================================================
# PHASE 4 -- T1_hash19 SLAM
# ============================================================================
phase 4 "T1_hash19 (T0 + grid.hash_size 16->19 only)"
T1_RC=0
run_variant T1_hash19 output/StereoMIS/P2_1_T1_hash19 "$DRIVE_ROOT/T1_hash19" || T1_RC=$?
case "$T1_RC" in
  2) echo "FAILED_SLAM" > "$DRIVE_ROOT/T1_hash19/.FAILED" ;;
  3) echo "FAILED_SHIP" > "$DRIVE_ROOT/T1_hash19/.FAILED" ;;
esac

# ============================================================================
# PHASE 5 -- PSNR/SSIM/LPIPS eval on both
# ============================================================================
phase 5 "render-quality eval (PSNR / SSIM / LPIPS)"
for V in T0_literal T1_hash19; do
  LOCAL_OUT=/content/DDS-SLAM/output/StereoMIS/P2_1_$V
  if [ ! -d "$LOCAL_OUT" ]; then
    echo "  $V: no local output -- skip"; continue
  fi
  if [ -f "$DRIVE_ROOT/$V/.FAILED" ] && grep -q FAILED_SLAM "$DRIVE_ROOT/$V/.FAILED"; then
    echo "  $V: SLAM crashed -- skip eval"; continue
  fi
  EVAL_OUT=$DRIVE_ROOT/_eval/${V}_render.txt
  EVAL_CSV=$DRIVE_ROOT/_eval/${V}_render.csv
  python Addons/eval/eval_rendering.py \
    --gt_dir "$STAGED/video_frames" \
    --render_dir "$LOCAL_OUT" \
    --name "$V" \
    --output_csv "$EVAL_CSV" \
    --summary_csv "$DRIVE_ROOT/_eval/summary.csv" \
    --sequence "StereoMIS (P2_1)" \
    --gt_offset 4465 \
    2>&1 | tee "$EVAL_OUT"
done

# ============================================================================
# PHASE 6 -- combined summary (raw + Sim3 ATE + est_path/GT + Pearson + render)
# ============================================================================
phase 6 "combined summary"
if [ ! -d "$DRIVE_ROOT/_eval" ]; then
  echo "FATAL: $DRIVE_ROOT/_eval missing"; exit 1
fi
SUMMARY=$DRIVE_ROOT/summary.txt
: > "$SUMMARY"
python3 - >> "$SUMMARY" 2>&1 <<PYEOF
import os, csv, glob, numpy as np
GT_FULL = '$STAGED/groundtruth.txt'
DR = '$DRIVE_ROOT'

def parse_est(p):
    poses=[]
    if not os.path.isfile(p): return np.zeros((0,3))
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

def horn(m, d, with_scale=False):
    mc=m.mean(0); dc=d.mean(0); mm=m-mc; dd=d-dc
    H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    ds=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,ds]); R=Vt.T@D@U.T
    s=(S*np.array([1,1,ds])).sum()/(mm*mm).sum() if with_scale else 1.0
    t=dc-s*R@mc
    return (s*(R@m.T)).T+t, s

render = {}
csv_path = f'{DR}/_eval/summary.csv'
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            render[row['method']] = row

# Apply same last-4000 slice as dataset.py:120 for GT
g_full = parse_tum(GT_FULL)
g = g_full[-4000:]
gpath_m = np.linalg.norm(np.diff(g,axis=0),axis=1).sum()
print(f'StereoMIS P2_1 last-4000:')
print(f'  GT poses     : {len(g)}')
print(f'  GT path      : {gpath_m*1000:.1f} mm = {gpath_m:.3f} m')
print(f'  GT extent mm : {(g.max(0)-g.min(0))*1000}')
print(f'  Paper target : 8.3 mm raw ATE')
print()
print(f"{'variant':<14} {'frames':>7} {'raw_mm':>9} {'SE3_mm':>9} {'Sim3_mm':>9} {'sim3_s':>9} {'est/GT_path':>11} {'pearson_xyz':>16} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-" * 130)

for V in ['T0_literal', 'T1_hash19']:
    p = f'/content/DDS-SLAM/output/StereoMIS/P2_1_{V}/demo/est_c2w_data.txt'
    e = parse_est(p)
    if len(e) < 10:
        p2 = p.replace('/demo/', '/')
        if os.path.isfile(p2):
            e = parse_est(p2)
    if len(e) < 10:
        row_str = f"{V:<14} {'(no est)':>7}"
    else:
        n = min(len(e), len(g)); ee, gg = e[:n], g[:n]
        # Raw (no alignment) — Euclidean distance frame-by-frame in world coords
        raw = np.linalg.norm(ee - gg, axis=1) * 1000
        ase3,_  = horn(ee, gg, False); ese3  = np.linalg.norm(ase3-gg, axis=1)*1000
        asim3,s = horn(ee, gg, True);  esim3 = np.linalg.norm(asim3-gg, axis=1)*1000
        epath = np.linalg.norm(np.diff(ee,axis=0), axis=1).sum()
        gpath = np.linalg.norm(np.diff(gg,axis=0), axis=1).sum()
        ratio = epath / max(gpath, 1e-12)
        # Pearson per axis (between Sim3-aligned est and gt)
        pearson = []
        for axis in range(3):
            try:
                r = np.corrcoef(asim3[:, axis], gg[:, axis])[0, 1]
                pearson.append(r if np.isfinite(r) else 0.0)
            except Exception:
                pearson.append(0.0)
        pearson_str = f"({pearson[0]:.2f},{pearson[1]:.2f},{pearson[2]:.2f})"
        row_str = (f"{V:<14} {n:>7d} {np.sqrt((raw**2).mean()):>9.2f} "
                   f"{ese3.mean():>9.2f} {esim3.mean():>9.3f} {s:>9.4f} {ratio:>11.2f} "
                   f"{pearson_str:>16}")
    if V in render:
        r = render[V]
        try:
            row_str += f" {float(r['psnr_mean']):>7.2f} {float(r['ssim_mean']):>7.3f}"
            if 'lpips_mean' in r and r['lpips_mean']:
                row_str += f" {float(r['lpips_mean']):>7.3f}"
        except (ValueError, KeyError):
            pass
    print(row_str)

print()
print("Interpretation hints:")
print("  - Sim3_mm < 10 = approaching paper target (8.3 mm).")
print("  - sim3_s should be near 1.0 (calibrated stereo depth is metric).")
print("  - est/GT_path ratio: ideal 1.0; >2 indicates significant jitter.")
print("  - Pearson xyz: should all be > 0.5 if tracker actually follows GT shape.")
print("  - PSNR > 22 dB is paper-tier render quality.")
PYEOF
cat "$SUMMARY"
echo ""
echo "=== runbook done $(date -Iseconds) ==="
echo "Logs:    $LOG"
echo "Summary: $SUMMARY"
echo "Per-variant payloads: $DRIVE_ROOT/{T0_literal,T1_hash19}/payload.tgz"
echo "Per-frame render CSVs: $DRIVE_ROOT/_eval/"
