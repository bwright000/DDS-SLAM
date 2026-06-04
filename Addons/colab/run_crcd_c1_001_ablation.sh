#!/bin/bash
# ============================================================================
# DDS-SLAM CRCD C_1/001 — incremental ablation from upstream (~6-8 hr A100, slower on T4)
#
# Six SLAM runs walking from upstream DDS-SLAM-Base values to our best-shot
# config in single-knob steps. Plus retroactive PSNR/SSIM/LPIPS on the three
# 3-test payloads from this morning's run.
#
# Chain:
#   T0_SM = upstream StereoMIS knobs + CRCD intrinsics + depth ×1.0       (literal baseline)
#   T0_SS = upstream Super knobs + CRCD intrinsics + depth ×1.0           (literal aggressive baseline)
#   T1    = T0_SM + hash 19 + voxel_sdf 0.001                              (paper representation)
#   T2    = T1   + Recipe A (iters 20 + first_iters 1000)                  (schedule)
#   T3    = T2   + depth ×0.16                                              (scale fix, bound mismatched)
#   T4    = T3   + corrected bound z=[-0.05,0.22]                          (scale + bound)
#
# Per-variant metrics: SE3 ATE, Sim3 ATE, sim3 scale, est path, PSNR, SSIM, LPIPS.
#
# Pre-req: this morning's run_crcd_c1_001_3test.sh has staged data at
#   /content/DDS-SLAM/data/CRCD/C1_001/{video_frames,depth,masks,groundtruth.txt}
# AND shipped the 3-test payloads to
#   MyDrive/Outputs/dds_crcd_c1_001_3test_<DATE>/{paper_faithful,semsup_sched,paper_faithful_depthx0p3}/payload.tgz
# If either is missing the runbook re-stages from Drive (uses cached MoGe depth,
# ~5-10 min on T4).
#
# Paste into a Colab VS Code tunnel terminal. Sentinel-gated, resumable.
# ============================================================================
set -euo pipefail
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_crcd_c1_001_ablation_${DATE}
PRIOR_3TEST_GLOB=/content/drive/MyDrive/Outputs/dds_crcd_c1_001_3test_*
mkdir -p "$DRIVE_ROOT"/{T0_SM,T0_SS,T1_paperrep,T2_recipeA,T3_depthx16,T4_bound,_eval}
LOG="$DRIVE_ROOT/runbook.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== runbook start $(date -Iseconds) -- DRIVE_ROOT=$DRIVE_ROOT ==="

phase() { echo ""; echo "[PHASE $1] $(date +%H:%M:%S) -- $2"; }
done_marker() { [ -f "$1/.DONE" ]; }
mark_done() { sync; touch "$1/.DONE"; sync; }

STAGED=/content/DDS-SLAM/data/CRCD/C1_001
DRIVE_SNIPPET=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
CALIB_PKL=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl

activate_dds_env() {
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    echo "modern stack missing -- full rebuild"
    bash /content/DDS-SLAM/Addons/env/colab_setup.sh --skip-data --skip-tunnel
  fi
  python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()" \
    || { echo "env check FAIL"; exit 1; }
  python -c "import lpips" 2>/dev/null || {
    echo "  installing lpips for rendering eval..."
    pip install -q lpips
  }
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}

ship_to_drive() {
  local SRC=$1 DST=$2
  tar czf "$DST/payload.tgz.partial" -C "$SRC" .
  mv "$DST/payload.tgz.partial" "$DST/payload.tgz"
  mark_done "$DST"
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
activate_dds_env

# ============================================================================
# PHASE 1 -- ensure staged data is present (recover from Drive if needed)
# ============================================================================
phase 1 "verify or restage C_1/001 data"
if [ -f "$STAGED/.STAGED" ] && [ -f "$STAGED/depth/.DONE" ]; then
  N_RGB=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  N_DEPTH=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
  echo "  staged data present: rgb=$N_RGB depth=$N_DEPTH"
else
  echo "  staged data missing -- restaging from Drive (uses cached MoGe depth)"
  rm -rf "$STAGED"
  mkdir -p /content/crcd_raw "$(dirname "$STAGED")"
  if [ ! -d /content/crcd_raw/snippet_001 ]; then
    cp -r "$DRIVE_SNIPPET" /content/crcd_raw/snippet.tmp
    mv /content/crcd_raw/snippet.tmp /content/crcd_raw/snippet_001
  fi
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir /content/crcd_raw/snippet_001 \
    --calib_pkl   "$CALIB_PKL" \
    --output_dir  "${STAGED}.tmp"
  mv "${STAGED}.tmp" "$STAGED"
  touch "$STAGED/.STAGED"
  # Copy cached depth from Drive snippet
  EXPECTED=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
  N_DRIVE_DEPTH=$(find "$DRIVE_SNIPPET/depth" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l || echo 0)
  if [ "$N_DRIVE_DEPTH" -eq "$EXPECTED" ]; then
    mkdir -p "$STAGED/depth.tmp"
    python3 - <<PYEOF
import os, shutil
RAW = '/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001'
STG = '$STAGED'
rgb_orig = sorted(f for f in os.listdir(os.path.join(RAW, 'rgb')) if f.endswith('.png'))
depth_files = sorted(f for f in os.listdir(os.path.join(RAW, 'depth')) if f.endswith('.png'))
for i, dname in enumerate(depth_files):
    shutil.copy2(os.path.join(RAW, 'depth', dname),
                 os.path.join(STG, 'depth.tmp', f'{i:06d}.png'))
print(f'restaged {len(depth_files)} depth pngs')
PYEOF
    rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"
    sync; touch "$STAGED/depth/.DONE"; sync
  else
    echo "  Drive depth cache missing or mismatch ($N_DRIVE_DEPTH/$EXPECTED) -- need fresh MoGe gen"
    echo "  Please re-run scripts/run_crcd_c1_001_3test.sh first"
    exit 1
  fi
fi

# ============================================================================
# Helper: run one ablation variant and ship
# ============================================================================
run_variant() {
  local NAME=$1 OUT=$2 DST=$3
  if done_marker "$DST"; then
    echo "  $NAME already shipped -- skip"
    return 0
  fi
  echo "=== run $NAME ==="
  T0=$(date +%s)
  cd /content/DDS-SLAM
  python ddsslam.py --config "configs/CRCD/c1_001_ABL_${NAME}.yaml"
  echo "  $NAME elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  ship_to_drive "$OUT" "$DST"
}

# ============================================================================
# PHASE 2-7 -- six ablation SLAM runs
# ============================================================================
phase 2 "T0_SM upstream StereoMIS literal"
run_variant T0_SM        output/CRCD/C1_001_ABL_T0_SM        "$DRIVE_ROOT/T0_SM"

phase 3 "T0_SS upstream Super literal (may OOM on T4)"
if ! run_variant T0_SS   output/CRCD/C1_001_ABL_T0_SS        "$DRIVE_ROOT/T0_SS"; then
  echo "  T0_SS FAILED (likely OOM on T4 from voxel_sdf=0.0002 + hash=16). Continuing."
  echo "FAILED_OOM_OR_OTHER" > "$DRIVE_ROOT/T0_SS/.FAILED"
fi

phase 4 "T1 + hash 19 + voxel_sdf 0.001 (paper representation)"
run_variant T1_paperrep  output/CRCD/C1_001_ABL_T1_paperrep  "$DRIVE_ROOT/T1_paperrep"

phase 5 "T2 + Recipe A (iters=20, first_iters=1000)"
run_variant T2_recipeA   output/CRCD/C1_001_ABL_T2_recipeA   "$DRIVE_ROOT/T2_recipeA"

phase 6 "T3 + depth x0.16 (bound still original = mismatched)"
run_variant T3_depthx16  output/CRCD/C1_001_ABL_T3_depthx16  "$DRIVE_ROOT/T3_depthx16"

phase 7 "T4 + corrected bound z=[-0.05, 0.22] (best-shot)"
run_variant T4_bound     output/CRCD/C1_001_ABL_T4_bound     "$DRIVE_ROOT/T4_bound"

# ============================================================================
# PHASE 8 -- retroactive PSNR/SSIM/LPIPS on this morning's 3-test payloads
# ============================================================================
phase 8 "retroactive eval on 3-test payloads"
PRIOR=$(ls -d $PRIOR_3TEST_GLOB 2>/dev/null | head -1 || echo "")
if [ -z "$PRIOR" ]; then
  echo "  no 3-test payloads on Drive at $PRIOR_3TEST_GLOB -- skipping retroactive eval"
else
  echo "  3-test root: $PRIOR"
  mkdir -p /content/_retro_3test
  for V in paper_faithful semsup_sched paper_faithful_depthx0p3; do
    SRC=$PRIOR/$V/payload.tgz
    DST=/content/_retro_3test/$V
    if [ ! -f "$SRC" ]; then echo "  missing $SRC -- skip"; continue; fi
    if [ ! -d "$DST" ]; then
      mkdir -p "$DST" && tar xzf "$SRC" -C "$DST"
    fi
    EVAL_OUT=$DRIVE_ROOT/_eval/3test_${V}_render.txt
    EVAL_CSV=$DRIVE_ROOT/_eval/3test_${V}_render.csv
    RENDER_DIR=$DST/demo
    [ -d "$RENDER_DIR" ] || RENDER_DIR=$DST   # some payloads have demo/ flattened
    if find "$RENDER_DIR" -maxdepth 1 -name '????.jpg' | head -1 | grep -q .; then
      python Addons/eval/eval_rendering.py \
        --gt_dir "$STAGED/video_frames" \
        --render_dir "$RENDER_DIR" \
        --name "3test_${V}" \
        --output_csv "$EVAL_CSV" \
        --summary_csv "$DRIVE_ROOT/_eval/summary.csv" \
        --sequence "Lab1 (trail3)" \
        2>&1 | tee "$EVAL_OUT"
    else
      echo "  $V: no rendered ????.jpg in payload -- skip"
    fi
  done
fi

# ============================================================================
# PHASE 9 -- PSNR/SSIM/LPIPS on the six ablation variants
# ============================================================================
phase 9 "PSNR/SSIM/LPIPS on T0_SM .. T4"
for V in T0_SM T0_SS T1_paperrep T2_recipeA T3_depthx16 T4_bound; do
  LOCAL_OUT=/content/DDS-SLAM/output/CRCD/C1_001_ABL_$V
  if [ ! -d "$LOCAL_OUT" ]; then
    echo "  $V: no local output -- skip"; continue
  fi
  if [ -f "$DRIVE_ROOT/$V/.FAILED" ]; then
    echo "  $V: marked FAILED earlier -- skip"; continue
  fi
  EVAL_OUT=$DRIVE_ROOT/_eval/${V}_render.txt
  EVAL_CSV=$DRIVE_ROOT/_eval/${V}_render.csv
  python Addons/eval/eval_rendering.py \
    --gt_dir "$STAGED/video_frames" \
    --render_dir "$LOCAL_OUT" \
    --name "ABL_${V}" \
    --output_csv "$EVAL_CSV" \
    --summary_csv "$DRIVE_ROOT/_eval/summary.csv" \
    --sequence "Lab1 (trail3)" \
    2>&1 | tee "$EVAL_OUT"
done

# ============================================================================
# PHASE 10 -- combined summary (Sim3 ATE + render metrics, all 9 variants)
# ============================================================================
phase 10 "combined summary"
SUMMARY=$DRIVE_ROOT/summary.txt
: > "$SUMMARY"
python3 - >> "$SUMMARY" 2>&1 <<'PYEOF'
import os, csv, glob, numpy as np
GT = '/content/DDS-SLAM/data/CRCD/C1_001/groundtruth.txt'
DRIVE = sorted(glob.glob('/content/drive/MyDrive/Outputs/dds_crcd_c1_001_ablation_*'))
if not DRIVE:
    print("(no ablation drive root found)"); raise SystemExit
DR = DRIVE[-1]
PRIOR = sorted(glob.glob('/content/drive/MyDrive/Outputs/dds_crcd_c1_001_3test_*'))
PR = PRIOR[-1] if PRIOR else None

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

def horn(m,d,sc=False):
    mc=m.mean(0); dc=d.mean(0); mm=m-mc; dd=d-dc
    H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    ds=np.sign(np.linalg.det(Vt.T@U.T))
    D=np.diag([1,1,ds]); R=Vt.T@D@U.T
    s=(S*np.array([1,1,ds])).sum()/(mm*mm).sum() if sc else 1.0
    t=dc-s*R@mc
    return (s*(R@m.T)).T+t, s

# Read render summary CSV (Phase 8+9 appended rows)
render = {}
csv_path = f'{DR}/_eval/summary.csv'
if os.path.isfile(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            render[row['method']] = row

g = parse_tum(GT)
gpath = np.linalg.norm(np.diff(g,axis=0),axis=1).sum()*1000
print(f"GT path length: {gpath:.2f} mm  (extent xyz mm: {(g.max(0)-g.min(0))*1000})")
print()
print(f"{'variant':<32} {'frames':>7} {'SE3_mm':>9} {'Sim3_mm':>9} {'sim3_s':>9} {'est_path_mm':>13} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-" * 117)

# Order: 3-test results then ablation chain
order = [
    ('3test/paper_faithful',           '3test_paper_faithful'),
    ('3test/semsup_sched',             '3test_semsup_sched'),
    ('3test/paper_faithful_depthx0p3', '3test_paper_faithful_depthx0p3'),
    ('ABL T0_SM',                      'ABL_T0_SM'),
    ('ABL T0_SS',                      'ABL_T0_SS'),
    ('ABL T1_paperrep',                'ABL_T1_paperrep'),
    ('ABL T2_recipeA',                 'ABL_T2_recipeA'),
    ('ABL T3_depthx16',                'ABL_T3_depthx16'),
    ('ABL T4_bound',                   'ABL_T4_bound'),
]

# Map variant name -> est_c2w_data.txt path
def est_path_for(name):
    if name.startswith('3test/'):
        v = name.split('/')[1]
        return f'/content/_retro_3test/{v}/demo/est_c2w_data.txt'
    elif name.startswith('ABL '):
        v = name.split(' ')[1]
        return f'/content/DDS-SLAM/output/CRCD/C1_001_ABL_{v}/demo/est_c2w_data.txt'
    return None

for label, key in order:
    p = est_path_for(label)
    e = parse_est(p)
    if len(e) < 10:
        # Try without /demo/
        p2 = p.replace('/demo/', '/') if p else None
        if p2 and os.path.isfile(p2):
            e = parse_est(p2)
    if len(e) < 10:
        row_str = f"{label:<32} {'(no est)':>7}"
    else:
        n = min(len(e), len(g)); ee, gg = e[:n], g[:n]
        ase3,_  = horn(ee,gg,False); ese3  = np.linalg.norm(ase3-gg,axis=1)*1000
        asim3,s = horn(ee,gg,True);  esim3 = np.linalg.norm(asim3-gg,axis=1)*1000
        epath = np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()*1000
        row_str = f"{label:<32} {n:>7d} {ese3.mean():>9.2f} {esim3.mean():>9.3f} {s:>9.4f} {epath:>13.2f}"
    if key in render:
        r = render[key]
        try:
            row_str += f" {float(r['psnr_mean']):>7.2f} {float(r['ssim_mean']):>7.3f}"
            if 'lpips_mean' in r and r['lpips_mean']:
                row_str += f" {float(r['lpips_mean']):>7.3f}"
        except (ValueError, KeyError):
            pass
    print(row_str)
PYEOF
echo ""
echo "=== summary ==="
cat "$SUMMARY"
echo ""
echo "=== runbook done $(date -Iseconds) ==="
echo "Logs:    $LOG"
echo "Summary: $SUMMARY"
echo "Per-variant payloads: $DRIVE_ROOT/{T0_SM,T0_SS,T1_paperrep,T2_recipeA,T3_depthx16,T4_bound}/payload.tgz"
echo "Per-variant render metrics: $DRIVE_ROOT/_eval/"
