#!/bin/bash
# ============================================================================
# run_sgsslam.sh <phase> [scene]   -  Arm 4 Stage 1: SGS-SLAM (ECCV 2024) onboarding.
#   phase: env | repro | crcd | eval | all
#
# PHASE-A (repro) = Replica via the repo's OWN eval (paper-faithful), gate REPORT-ONLY
#   vs verified paper averages (SGS-SLAM_eval_spec.md):
#     PSNR 34.66 | MS-SSIM 0.973 | LPIPS 0.096 | Depth-L1 0.356cm | ATE 0.412cm | mIoU 92.72%
#   Metrics are parsed from scripts/slam.py STDOUT, which prints (eval_helpers.py:770,793-798):
#     "Average PSNR: X"  "Average MS-SSIM: X"  "Average LPIPS: X"  "Average Depth L1: X cm"
#     "Final Average ATE RMSE: X cm"  "Average mIoU: X"   (mIoU as fraction 0-1)
#   Scope (user 2026-06-18): validate room0 first (`repro room0`), then full 8 (`repro`).
#
# CRCD (Phase B) is STUBBED here - needs CRCDGradSLAMDataset + semantic_ids/colors emission
#   + npz->est & render-rename adapters (ARM4 A4-1.3/1.4). Runs once those land.
#
# RUNS ON COLAB/A100. The `env` phase (3DGS rasterizer build) is the fragile part and may
# need on-instance iteration - it is best-effort and fails LOUD, it does not fake success.
#
# Usage (on the tunnel):
#   bash Addons/colab/run_sgsslam.sh env
#   bash Addons/colab/run_sgsslam.sh repro room0      # validate one scene
#   bash Addons/colab/run_sgsslam.sh repro            # full 8-scene gate
# ============================================================================
set -uo pipefail
PHASE=${1:-all}; SCENE_ARG=${2:-}
SGS=${SGS:-/content/SGS-SLAM}
SGS_URL=${SGS_URL:-https://github.com/IRMVLab/SGS-SLAM.git}
REPLICA=${REPLICA:-/content/data/Replica}          # Replica-with-GT-semantics (README Dropbox)
DRIVE_REPLICA=${DRIVE_REPLICA:-/content/drive/MyDrive/Datasets/Replica}
DATE=$(date +%Y%m%d)
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/SGS-SLAM_repro_${DATE}}
PY=${PY:-python}
ALL_SCENES="room0 room1 room2 office0 office1 office2 office3 office4"

echo "=== run_sgsslam.sh phase=$PHASE scene=${SCENE_ARG:-<all>} $(date -Iseconds) ==="

# --- env -------------------------------------------------------------------------------------
build_env(){
  [ -d "$SGS/.git" ] || git clone --recursive "$SGS_URL" "$SGS"
  cd "$SGS"
  # The diff-gaussian-rasterization-w-depth (SplaTAM fork, pin cb65e4b) + deps. This is the
  # fragile 3DGS step; requirements.txt pulls the rasterizer via pip VCS.
  $PY -c "import diff_gaussian_rasterization" 2>/dev/null && { echo "[env] rasterizer present"; } || {
    echo "[env] installing requirements + rasterizer (slow; may need cuda-11.8 toolkit)"
    pip install -q -r requirements.txt || echo "[env] WARN requirements.txt had failures"
  }
  echo "[env] smoke import:"
  $PY - <<'PY'
import importlib, sys
ok = True
for m in ("torch", "diff_gaussian_rasterization"):
    try:
        importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e:
        ok = False; print(f"  {m}: FAIL -> {e}")
import torch; print("  torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
sys.exit(0 if ok else 30)
PY
  local rc=$?
  [ "$rc" -eq 0 ] || { echo "FATAL[env]: rasterizer/torch import failed (exit 30). Likely a torch/CUDA"\
                       " mismatch - the repo wants torch 2.0.1 / cu11.8 (README). Build cuda-11.8 toolkit"\
                       " and rebuild the rasterizer for sm_80, then re-run."; exit 30; }
  echo "[env] OK"
}

# --- stage Replica (if not already local) ----------------------------------------------------
stage_replica(){
  [ -d "$REPLICA/$1" ] && { echo "[$1] Replica scene present"; return 0; }
  if [ -d "$DRIVE_REPLICA/$1" ]; then
    mkdir -p "$REPLICA"; echo "[$1] copying Replica scene from Drive (local SSD for speed)"
    cp -rn "$DRIVE_REPLICA/$1" "$REPLICA/"; return 0
  fi
  echo "[$1] Replica scene not found at $REPLICA/$1 or $DRIVE_REPLICA/$1 -> stage the"\
       " Replica-with-GT-semantics download (README Dropbox) to one of those. skipping."; return 1
}

# --- one scene -------------------------------------------------------------------------------
run_scene(){
  local s=$1 dst="$DRIVE/$s"
  stage_replica "$s" || return 1
  mkdir -p "$dst"
  # per-scene config: set scene_name, disable wandb, point basedir at the local Replica
  local cfg="/content/sgs_cfg_${s}.py"
  sed -e "s/^scene_name = .*/scene_name = \"$s\"/" \
      -e "s/use_wandb=True/use_wandb=False/" \
      -e "s#basedir=\"./data/Replica\"#basedir=\"$REPLICA\"#" \
      "$SGS/configs/replica/slam.py" > "$cfg"
  echo "[$s] running SGS-SLAM (this is a full SLAM pass; minutes-to-hours on A100)"
  ( cd "$SGS" && $PY scripts/slam.py "$cfg" ) 2>&1 | tee "$dst/slam.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { echo "[$s] FAIL rc=$rc" | tee "$dst/status.txt"; return 1; }
  # parse the 6 metrics from the eval stdout
  $PY - "$dst/slam.log" "$dst/metrics.json" "$s" <<'PY'
import sys, re, json
log, out, scene = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(log, encoding='utf-8', errors='ignore').read()
def g(pat):
    m = re.search(pat, t)
    return float(m.group(1)) if m else None
psnr  = g(r'Average PSNR:\s*([0-9.]+)')
ssim  = g(r'Average MS-SSIM:\s*([0-9.]+)')
lpips = g(r'Average LPIPS:\s*([0-9.]+)')
dl1   = g(r'Average Depth L1:\s*([0-9.]+)\s*cm')          # already cm
ate   = g(r'Final Average ATE RMSE:\s*([0-9.]+)\s*cm')    # already cm
miou  = g(r'Average mIoU:\s*([0-9.]+)')                   # fraction 0-1
miou_pct = miou * 100 if miou is not None else None
d = dict(scene=scene, psnr=psnr, ssim=ssim, lpips=lpips, depth_l1_cm=dl1, ate_rmse_cm=ate, miou_pct=miou_pct)
json.dump(d, open(out, 'w'), indent=2)
print(f"[{scene}] " + " ".join(f"{k}={v}" for k, v in d.items() if k != 'scene'))
if ate == 100.0:
    print(f"[{scene}] WARN ATE=100.0 sentinel -> trajectory alignment FAILED (tracking diverged)")
PY
  echo "PASS" > "$dst/status.txt"
}

# --- aggregate vs paper (REPORT-ONLY gate) ---------------------------------------------------
aggregate(){
  $PY - "$DRIVE" <<'PY'
import sys, os, json, glob
root = sys.argv[1]
paper = dict(psnr=34.66, ssim=0.973, lpips=0.096, depth_l1_cm=0.356, ate_rmse_cm=0.412, miou_pct=92.72)
band  = dict(psnr=('>=',33.5), ssim=('>=',0.96), lpips=('<=',0.12),
             depth_l1_cm=('<=',0.6), ate_rmse_cm=('<=',0.6), miou_pct=('>=',90.0))
ms = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, '*', 'metrics.json')))]
if not ms:
    print("no per-scene metrics yet"); raise SystemExit
keys = ['psnr','ssim','lpips','depth_l1_cm','ate_rmse_cm','miou_pct']
def mean(k):
    vs = [m[k] for m in ms if m.get(k) is not None]
    return sum(vs)/len(vs) if vs else None
lines = [f"SGS-SLAM Replica repro - {len(ms)} scene(s): {[m['scene'] for m in ms]}", ""]
lines.append(f"{'metric':<12}{'repro':>10}{'paper':>10}{'band':>12}{'verdict':>9}")
allpass = True
for k in keys:
    r = mean(k); p = paper[k]; op, th = band[k]
    if r is None: lines.append(f"{k:<12}{'--':>10}{p:>10}{op+str(th):>12}{'n/a':>9}"); continue
    ok = (r >= th) if op == '>=' else (r <= th)
    allpass = allpass and ok
    lines.append(f"{k:<12}{r:>10.3f}{p:>10.3f}{(op+str(th)):>12}{('PASS' if ok else 'MISS'):>9}")
lines += ["", f"GATE (report-only): {'PASS' if allpass else 'BELOW-BAND'} "
              f"(headline ATE-RMSE + PSNR; does NOT block CRCD - CONTRACT s8)"]
txt = "\n".join(lines); print("\n"+txt)
open(os.path.join(root, 'COMBINED.txt'), 'w').write(txt+"\n")
PY
}

case "$PHASE" in
  env)   build_env ;;
  repro|all)
    [ "$PHASE" = all ] && build_env
    mkdir -p "$DRIVE"
    SCENES="${SCENE_ARG:-$ALL_SCENES}"
    for s in $SCENES; do run_scene "$s" || echo "[$s] skipped/failed (see $DRIVE/$s)"; done
    aggregate
    echo "DONE repro -> $DRIVE (COMBINED.txt + per-scene metrics.json/slam.log)" ;;
  crcd)
    echo "CRCD (Phase B) NOT WIRED yet: needs CRCDGradSLAMDataset + semantic_ids/colors"\
         " emission + npz->est & render-rename adapters (ARM4 A4-1.3/1.4). Stub - exiting." ; exit 0 ;;
  eval)  aggregate ;;
  *) echo "usage: run_sgsslam.sh env|repro|crcd|eval|all [scene]"; exit 2 ;;
esac
