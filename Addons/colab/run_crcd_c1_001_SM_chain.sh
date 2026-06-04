#!/bin/bash
# ============================================================================
# DDS-SLAM CRCD C_1/001 — SM_v1/v2/v3 chain (incremental 100%-good-idea
# improvements from T0_SM upstream baseline; bounds kept at literal upstream)
#
# Three SLAM runs walking from T0_SM in single-knob steps.  Each step is an
# evidence-backed change; the chain endpoint (v3) is the proposed generalised
# CRCD-on-MoGe-2 config (modulo bound, which is a separate experiment).
#
# Chain (single change per step):
#   T0_SM (already run by run_crcd_c1_001_T0.sh) = literal upstream baseline
#   SM_v1 = T0_SM + hash 16->19 + voxel_sdf 0.002->0.001    (paper representation)
#   SM_v2 = SM_v1 + iters 10->20 + first_iters 200->1000    (Recipe A)
#   SM_v3 = SM_v2 + depth x0.16 (png_depth_scale 10000->62500) (MoGe correction)
#
# Bound for all three = literal upstream p2_1 [[-1.8,2.1],[-2,1.2],[-0.1,2.7]].
# This holds the bound knob constant so each step's delta isolates one change.
#
# Per-variant metrics: SE3 ATE, Sim3-aligned ATE, sim3 scale, est path,
#                      PSNR, SSIM, LPIPS.
#
# Final summary also pulls T0_SM + T0_SS from MyDrive/Outputs/dds_crcd_c1_001_T0_*
# so we have a 5-row comparison: T0_SM | T0_SS | SM_v1 | SM_v2 | SM_v3.
#
# Pre-req:
#   * /content/DDS-SLAM/data/CRCD/C1_001/ staged from a prior runbook (T0 or 3test)
#   * T0 runbook has already shipped T0_SM + T0_SS payloads to Drive (for
#     retroactive eval).  If they're missing the runbook still proceeds with
#     just SM_v1/v2/v3.
#
# Sentinel-gated, resumable.  Paste into a Colab VS Code tunnel terminal.
# ============================================================================
set -euo pipefail
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/dds_crcd_c1_001_SM_chain_${DATE}
PRIOR_T0_GLOB=/content/drive/MyDrive/Outputs/dds_crcd_c1_001_T0_*
mkdir -p "$DRIVE_ROOT"/{SM_v1,SM_v2,SM_v3,_eval}
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
  if ! python -c "import lpips" 2>/dev/null; then
    echo "  installing lpips for rendering eval..."
    if ! pip install -q lpips; then
      echo "  ERROR: lpips pip install failed -- LPIPS metric will be missing from summary"
      echo "  Continue anyway; PSNR + SSIM + ATE will still be computed."
    fi
    # Verify the install actually succeeded — pip can silently leave a broken package
    python -c "import lpips" 2>/dev/null || \
      echo "  WARN: lpips still not importable after install; LPIPS metric will be empty"
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
  if ! python ddsslam.py --config "configs/CRCD/c1_001_${NAME}.yaml"; then
    echo "  $NAME SLAM exited non-zero. Likely config error or OOM."
    return 2
  fi
  echo "  $NAME SLAM elapsed: $(( ($(date +%s) - T0) / 60 )) min"
  if ! ship_to_drive "$OUT" "$DST"; then
    echo "  $NAME ship failed (Drive or disk issue). SLAM output kept at $OUT."
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
activate_dds_env

# ============================================================================
# PHASE 1 -- verify staged data
# ============================================================================
phase 1 "verify C_1/001 data is staged"
if [ ! -f "$STAGED/.STAGED" ] || [ ! -f "$STAGED/depth/.DONE" ]; then
  echo "  ERROR: $STAGED missing .STAGED or .DONE sentinel."
  echo "         Run run_crcd_c1_001_T0.sh (or run_crcd_c1_001_3test.sh) first."
  exit 1
fi
N_RGB=$(find "$STAGED/video_frames" -maxdepth 1 -name '*l.png' | wc -l)
N_DEPTH=$(find "$STAGED/depth" -maxdepth 1 -name '*.png' | wc -l)
echo "  staged data: rgb=$N_RGB depth=$N_DEPTH"
# Defensive: a stale .STAGED from a prior partial preprocess could leave
# fewer-than-expected frames behind.  C_1/001 has 360 frames per the 3-test
# summary; threshold of 250 catches gross truncation while tolerating minor
# variations.
if [ "$N_RGB" -lt 250 ] || [ "$N_DEPTH" -lt 250 ]; then
  echo "  ERROR: insufficient staged data (rgb=$N_RGB depth=$N_DEPTH; expected ~360 each)"
  echo "         Likely a partial / stale .STAGED.  Re-run the T0 / 3test runbook to refresh."
  exit 1
fi
if [ "$N_RGB" -ne "$N_DEPTH" ]; then
  echo "  ERROR: rgb count ($N_RGB) does not match depth count ($N_DEPTH)"
  exit 1
fi

# ============================================================================
# PHASE 2 -- SM_v1 (paper representation upgrade)
# ============================================================================
phase 2 "SM_v1 (T0_SM + hash 19 + voxel_sdf 0.001)"
SM_v1_RC=0
run_variant SM_v1 output/CRCD/C1_001_SM_v1 "$DRIVE_ROOT/SM_v1" || SM_v1_RC=$?
case "$SM_v1_RC" in
  2) echo "FAILED_SLAM" > "$DRIVE_ROOT/SM_v1/.FAILED" ;;
  3) echo "FAILED_SHIP" > "$DRIVE_ROOT/SM_v1/.FAILED" ;;
esac

# ============================================================================
# PHASE 3 -- SM_v2 (+ Recipe A)
# ============================================================================
phase 3 "SM_v2 (SM_v1 + Recipe A: iters 20, first_iters 1000)"
SM_v2_RC=0
run_variant SM_v2 output/CRCD/C1_001_SM_v2 "$DRIVE_ROOT/SM_v2" || SM_v2_RC=$?
case "$SM_v2_RC" in
  2) echo "FAILED_SLAM" > "$DRIVE_ROOT/SM_v2/.FAILED" ;;
  3) echo "FAILED_SHIP" > "$DRIVE_ROOT/SM_v2/.FAILED" ;;
esac

# ============================================================================
# PHASE 4 -- SM_v3 (+ depth x0.16)
# ============================================================================
phase 4 "SM_v3 (SM_v2 + depth x0.16: png_depth_scale 62500)"
SM_v3_RC=0
run_variant SM_v3 output/CRCD/C1_001_SM_v3 "$DRIVE_ROOT/SM_v3" || SM_v3_RC=$?
case "$SM_v3_RC" in
  2) echo "FAILED_SLAM" > "$DRIVE_ROOT/SM_v3/.FAILED" ;;
  3) echo "FAILED_SHIP" > "$DRIVE_ROOT/SM_v3/.FAILED" ;;
esac

# ============================================================================
# PHASE 5 -- PSNR/SSIM/LPIPS on SM_v1/v2/v3
# ============================================================================
phase 5 "render-quality eval (PSNR / SSIM / LPIPS) on SM_v1/v2/v3"
for V in SM_v1 SM_v2 SM_v3; do
  LOCAL_OUT=/content/DDS-SLAM/output/CRCD/C1_001_$V
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
    --sequence "Lab1 (trail3)" \
    2>&1 | tee "$EVAL_OUT"
done

# ============================================================================
# PHASE 6 -- retroactive PSNR/SSIM/LPIPS on prior T0_SM + T0_SS payloads
# ============================================================================
phase 6 "retroactive eval on prior T0_SM + T0_SS payloads"
PRIOR=$(ls -d $PRIOR_T0_GLOB 2>/dev/null | tail -1 || echo "")
if [ -z "$PRIOR" ]; then
  echo "  no T0 payloads on Drive at $PRIOR_T0_GLOB -- skipping retroactive eval"
else
  echo "  T0 root: $PRIOR"
  mkdir -p /content/_retro_T0
  for V in T0_SM T0_SS; do
    # Skip retroactive eval if the prior T0 variant SLAM crashed — its
    # payload.tgz won't contain meaningful renders.  (Mirrors the Phase 5
    # guard for the new variants.)
    if [ -f "$PRIOR/$V/.FAILED" ] && grep -q FAILED_SLAM "$PRIOR/$V/.FAILED"; then
      echo "  $V: prior SLAM crashed (per .FAILED) -- skip retroactive eval"
      continue
    fi
    SRC=$PRIOR/$V/payload.tgz
    DST=/content/_retro_T0/$V
    if [ ! -f "$SRC" ]; then echo "  missing $SRC -- skip"; continue; fi
    if [ ! -d "$DST" ]; then
      mkdir -p "$DST" && tar xzf "$SRC" -C "$DST"
    fi
    EVAL_OUT=$DRIVE_ROOT/_eval/T0_${V}_render.txt
    EVAL_CSV=$DRIVE_ROOT/_eval/T0_${V}_render.csv
    RENDER_DIR=$DST/demo
    [ -d "$RENDER_DIR" ] || RENDER_DIR=$DST
    if find "$RENDER_DIR" -maxdepth 1 -name '????.jpg' | head -1 | grep -q .; then
      python Addons/eval/eval_rendering.py \
        --gt_dir "$STAGED/video_frames" \
        --render_dir "$RENDER_DIR" \
        --name "$V" \
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
# PHASE 7 -- combined 5-row summary
# ============================================================================
phase 7 "combined summary (T0_SM, T0_SS, SM_v1, SM_v2, SM_v3)"
if [ ! -d "$DRIVE_ROOT/_eval" ]; then
  echo "  FATAL: $DRIVE_ROOT/_eval missing"; exit 1
fi
SUMMARY=$DRIVE_ROOT/summary.txt
: > "$SUMMARY"
python3 - >> "$SUMMARY" 2>&1 <<PYEOF
import os, csv, glob, numpy as np
GT = '/content/DDS-SLAM/data/CRCD/C1_001/groundtruth.txt'
DR = '$DRIVE_ROOT'
PRIOR_GLOB = '$PRIOR_T0_GLOB'

if not os.path.isdir(DR):
    raise FileNotFoundError(f"DRIVE_ROOT missing: {DR}")

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

g = parse_tum(GT)
gpath = np.linalg.norm(np.diff(g,axis=0),axis=1).sum()*1000
print(f"GT path length: {gpath:.2f} mm  (extent xyz mm: {(g.max(0)-g.min(0))*1000})")
print()
print(f"{'variant':<10} {'frames':>7} {'SE3_mm':>9} {'Sim3_mm':>9} {'sim3_s':>9} {'est_path_mm':>13} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
print("-" * 95)

def est_path_for(label):
    if label in ('T0_SM', 'T0_SS'):
        prior_dirs = sorted(glob.glob(PRIOR_GLOB))
        if not prior_dirs:
            return None
        # Try retroactively extracted path first, then in-place
        return f'/content/_retro_T0/{label}/demo/est_c2w_data.txt'
    else:
        return f'/content/DDS-SLAM/output/CRCD/C1_001_{label}/demo/est_c2w_data.txt'

for V in ['T0_SM', 'T0_SS', 'SM_v1', 'SM_v2', 'SM_v3']:
    p = est_path_for(V)
    e = parse_est(p) if p else np.zeros((0,3))
    if len(e) < 10 and p:
        p2 = p.replace('/demo/', '/')
        if os.path.isfile(p2):
            e = parse_est(p2)
    if len(e) < 10:
        row_str = f"{V:<10} {'(no est)':>7}"
    else:
        n = min(len(e), len(g)); ee, gg = e[:n], g[:n]
        ase3,_  = horn(ee,gg,False); ese3  = np.linalg.norm(ase3-gg,axis=1)*1000
        asim3,s = horn(ee,gg,True);  esim3 = np.linalg.norm(asim3-gg,axis=1)*1000
        epath = np.linalg.norm(np.diff(ee,axis=0),axis=1).sum()*1000
        row_str = f"{V:<10} {n:>7d} {ese3.mean():>9.2f} {esim3.mean():>9.3f} {s:>9.4f} {epath:>13.2f}"
    if V in render:
        r = render[V]
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
echo "=== SM chain runbook done $(date -Iseconds) ==="
echo "Logs:    $LOG"
echo "Summary: $SUMMARY"
echo "Per-variant payloads: $DRIVE_ROOT/{SM_v1,SM_v2,SM_v3}/payload.tgz"
