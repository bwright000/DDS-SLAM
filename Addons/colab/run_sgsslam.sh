#!/bin/bash
# ============================================================================
# run_sgsslam.sh <phase> [scene]   -  Arm 4 Stage 1: SGS-SLAM (ECCV 2024) onboarding.
#   phase: env | repro | crcd | eval | all
#
# ENV: SGS-SLAM is a py3.9 / torch2.0.1 / cu11.8 repo (README). Colab ships py3.12 /
#   torch2.11 / cu12.8, where its pinned deps (open3d==0.16.0 -> numpy 1.21.6) DO NOT install
#   and the rasterizer won't match. So we build the README's CONDA env (miniconda -> py3.9 ->
#   cuda-toolkit 11.8 -> torch 2.0.1+cu118 -> requirements.txt incl. the JonathonLuiten
#   diff-gaussian-rasterization-w-depth @cb65e4b). repro runs through that env's python.
#
# PHASE-A (repro) = Replica via the repo's OWN eval (paper-faithful), gate REPORT-ONLY vs
#   verified paper avgs (SGS-SLAM_eval_spec.md): PSNR 34.66 | MS-SSIM 0.973 | LPIPS 0.096 |
#   Depth-L1 0.356cm | ATE 0.412cm | mIoU 92.72%. Metrics parsed from slam.py STDOUT.
#   Scope: validate room0 first (`repro room0`), then full 8 (`repro`).
#
# CRCD (Phase B) STUBBED (needs CRCDGradSLAMDataset + adapters, A4-1.3/1.4).
# RUNS ON COLAB/A100. The conda build is slow (~20-30 min, one-time) and fails LOUD.
#
# Usage:  bash Addons/colab/run_sgsslam.sh env
#         bash Addons/colab/run_sgsslam.sh repro room0
#         bash Addons/colab/run_sgsslam.sh repro
# ============================================================================
set -uo pipefail
PHASE=${1:-all}; SCENE_ARG=${2:-}
SGS=${SGS:-/content/SGS-SLAM}
SGS_URL=${SGS_URL:-https://github.com/ShuhongLL/SGS-SLAM}     # NOT IRMVLab (that is SemGauss/SNI/DDS)
REPLICA=${REPLICA:-/content/data/Replica}                     # local staged scenes (room0, ...)
DRIVE_REPLICA_ZIPS=${DRIVE_REPLICA_ZIPS:-/content/drive/MyDrive/Datasets/Replica/SGS}  # the 2 zips
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
ENV_NAME=${ENV_NAME:-sgs-slam}
ENV_PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
DATE=$(date +%Y%m%d)
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/SGS-SLAM_repro_${DATE}}
ALL_SCENES="room0 room1 room2 office0 office1 office2 office3 office4"

echo "=== run_sgsslam.sh phase=$PHASE scene=${SCENE_ARG:-<all>} $(date -Iseconds) ==="

# --- env: clone + README conda stack + rasterizer (idempotent) ------------------------------
build_env(){
  [ -d "$SGS/.git" ] || git clone --recursive "$SGS_URL" "$SGS" || { echo "FATAL clone $SGS_URL"; exit 30; }
  if [ "${REBUILD_RAST:-0}" != 1 ] && [ -x "$ENV_PY" ] && PYTHONPATH= "$ENV_PY" -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "[env] $ENV_NAME ready (rasterizer imports; REBUILD_RAST=1 to recompile for this GPU)"; return 0; fi
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo "[env] installing miniconda -> $CONDA_ROOT"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_ROOT" || { echo "FATAL miniconda"; exit 30; }
  fi
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda config --set channel_priority flexible 2>/dev/null || true
  # accept Anaconda ToS (defaults channels) if the gate is present (known Colab trap)
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" -c conda-forge python=3.9 \
     || { echo "FATAL conda create"; exit 30; }
  # Install BY NAME / via "$ENV_PY -m pip" - do NOT rely on `conda activate` (on Colab the PATH
  # never switches and system py3.12 site-packages leak in via PYTHONPATH). PYTHONPATH= isolates
  # the env's py3.9 from Colab's py3.12 packages (mixing them segfaults / ModuleNotFound).
  local ENV_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
  echo "[env] cuda-toolkit 11.8 (nvcc) into $ENV_NAME"
  conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-11.8.0" cuda-toolkit || echo "[env] WARN cuda-toolkit"
  echo "[env] torch 2.0.1 + cu118 via conda (+ mkl/numpy pins: torch2.0.1 breaks on mkl>=2025 & numpy>=2)"
  conda install -y -n "$ENV_NAME" -c pytorch -c nvidia \
     pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 "mkl=2023.1.0" "numpy=1.26.4" \
     || { echo "FATAL torch install (conda pytorch channel)"; exit 30; }
  # torch MUST import before building the rasterizer (the build imports torch for CUDA info)
  PYTHONPATH= "$ENV_PY" -c "import torch,numpy; print('[env] torch',torch.__version__,'numpy',numpy.__version__,'OK')" \
     || { echo "FATAL: torch import broken (mkl/numpy ABI) - see error above"; exit 30; }
  echo "[env] pip deps (numpy<2 constrained; requirements minus the rasterizer)"
  printf 'numpy<2\n' > /tmp/sgs_constraints.txt        # pip otherwise pulls numpy 2.x -> torch ABI break
  grep -v 'diff-gaussian-rasterization' "$SGS/requirements.txt" > /tmp/sgs_reqs.txt
  PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/sgs_constraints.txt ninja wheel setuptools || true
  PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/sgs_constraints.txt -r /tmp/sgs_reqs.txt || {
    echo "[env] bulk reqs failed (likely open3d/cyclonedds) -> installing slam.py runtime deps individually"
    PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/sgs_constraints.txt pytorch-msssim torchmetrics lpips \
       opencv-python imageio matplotlib kornia natsort pyyaml plyfile tqdm pandas wandb || echo "[env] WARN some deps failed"; }
  PYTHONPATH= "$ENV_PY" -m pip install -q "numpy<2" || true   # re-assert: a dep may have bumped it
  # Build the rasterizer for THIS GPU's compute capability (+PTX). sm_80-only -> garbage kernel
  # -> "numel: integer multiplication overflow" on a non-A100 (L4 8.9 / H100 9.0 / 4090 8.9).
  local CC ARCH
  CC=$(PYTHONPATH= "$ENV_PY" -c "import torch;print('%d.%d'%torch.cuda.get_device_capability())" 2>/dev/null)
  [ -n "$CC" ] || CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$CC" ] || CC=7.5      # safe floor: 7.5+PTX SASS also JIT-runs on newer GPUs
  ARCH="${ARCH_OVERRIDE:-${CC}+PTX}"
  echo "[env] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) compute_cap=$CC -> arch $ARCH"
  echo "[env] rasterizer build @cb65e4b (clone --recursive + VERBOSE; arch=$ARCH)"
  local RAST=/content/diff-gaussian-rasterization-w-depth
  if [ ! -d "$RAST/.git" ]; then
    git clone https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git "$RAST" \
      && git -C "$RAST" checkout -q cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110 \
      && git -C "$RAST" submodule update --init --recursive || echo "[env] WARN rasterizer clone/submodule issue"
  fi
  # VERBOSE (no -q, -v) + tee FULL build log so the real nvcc/g++ error is visible, not "No available output"
  PYTHONPATH= CUDA_HOME="$ENV_ROOT" PATH="$ENV_ROOT/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" \
     "$ENV_PY" -m pip install --no-build-isolation --force-reinstall --no-deps -v "$RAST" 2>&1 | tee /content/rasterizer_build.log
  PYTHONPATH= "$ENV_PY" -c "import diff_gaussian_rasterization" 2>/dev/null \
     && echo "[env] rasterizer import OK" \
     || echo "[env] WARN rasterizer build FAILED -> FULL log at /content/rasterizer_build.log (paste the nvcc/g++ error lines)"
  echo "[env] smoke import:"
  PYTHONPATH= "$ENV_PY" - <<'PY'
import importlib, sys
ok = True
for m in ("torch", "diff_gaussian_rasterization", "pytorch_msssim", "torchmetrics"):
    try: importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e: ok = False; print(f"  {m}: FAIL -> {e}")
import torch; print("  torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
sys.exit(0 if ok else 30)
PY
  [ $? -eq 0 ] || { echo "FATAL[env]: smoke import failed (exit 30). Inspect the rasterizer build log above."; exit 30; }
  echo "[env] OK -> $ENV_PY"
}

# --- stage Replica: unzip the 2 SGS zips once, locate scene dirs -----------------------------
stage_replica_all(){
  [ -d "$REPLICA/room0/frames" ] && { echo "[replica] already staged"; return 0; }
  # The 2-zip set is two VARIANTS under different top dirs: Replica/ (full, the paper dataset)
  # and Replica_900/ (900-frame). Use exactly ONE - NEVER merge (mixing frame counts is what
  # caused "color != depth"). Prefer the non-900 (full) zip; override with REPLICA_ZIP=<path>.
  local full="" v900="" z
  for z in "$DRIVE_REPLICA_ZIPS"/*.zip; do [ -e "$z" ] || continue
    case "$(basename "$z")" in *900*) v900="$z";; *) full="$z";; esac; done
  local use="${REPLICA_ZIP:-${full:-$v900}}"
  [ -n "$use" ] || { echo "[replica] no zips at $DRIVE_REPLICA_ZIPS"; return 1; }
  echo "[replica] unzipping ONE zip (full Replica = paper): $(basename "$use")"
  rm -rf /content/data/_rep_tmp "$REPLICA"; mkdir -p /content/data/_rep_tmp
  unzip -q "$use" -d /content/data/_rep_tmp || { echo "[replica] unzip failed"; return 1; }
  local fr; fr=$(find /content/data/_rep_tmp -type d -name frames | head -1)
  [ -n "$fr" ] || { echo "[replica] FATAL: no <scene>/frames/ inside $(basename "$use") (depths-only half?) -> set REPLICA_ZIP to the complete zip"; return 1; }
  local root; root=$(dirname "$(dirname "$fr")")    # .../<top>/<scene>/frames -> <top>
  mkdir -p "$(dirname "$REPLICA")"; mv "$root" "$REPLICA"; rm -rf /content/data/_rep_tmp
  echo "[replica] scenes: $(ls -d "$REPLICA"/*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
}

# --- one scene -------------------------------------------------------------------------------
run_scene(){
  local s=$1 dst="$DRIVE/$s"
  [ -d "$REPLICA/$s" ] || { echo "[$s] scene dir missing under $REPLICA -> skip"; return 1; }
  local nf nd
  nf=$(ls "$REPLICA/$s/frames/"frame*.jpg 2>/dev/null | wc -l)
  nd=$(ls "$REPLICA/$s/depths/"depth*.png 2>/dev/null | wc -l)
  [ "$nf" -gt 0 ] && [ "$nf" -eq "$nd" ] || { echo "[$s] frames=$nf depths=$nd (mismatch/empty - zip incomplete or wrong) -> skip"; return 1; }
  [ -d "$REPLICA/$s/semantic_ids" ] || echo "[$s] WARN no semantic_ids/ (load_semantics=True expects it; semantic may be in the other zip)"
  mkdir -p "$dst"
  local cfg="/content/sgs_cfg_${s}.py"
  sed -e "s/^scene_name = .*/scene_name = \"$s\"/" \
      -e "s/use_wandb=True/use_wandb=False/" \
      -e "s#basedir=\"./data/Replica\"#basedir=\"$REPLICA\"#" \
      "$SGS/configs/replica/slam.py" > "$cfg"
  [ -n "${NUM_FRAMES:-}" ] && { sed -i "s/num_frames=-1/num_frames=$NUM_FRAMES/" "$cfg"; \
     echo "[$s] NUM_FRAMES=$NUM_FRAMES (quick T4 validation - NOT a paper-comparable gate run)"; }
  echo "[$s] running SGS-SLAM (full SLAM pass; minutes-to-hours on A100)"
  ( cd "$SGS" && PYTHONPATH= "$ENV_PY" scripts/slam.py "$cfg" ) 2>&1 | tee "$dst/slam.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[$s] FAIL" | tee "$dst/status.txt"; return 1; }
  PYTHONPATH= "$ENV_PY" - "$dst/slam.log" "$dst/metrics.json" "$s" <<'PY'
import sys, re, json
log, out, scene = sys.argv[1], sys.argv[2], sys.argv[3]
t = open(log, encoding='utf-8', errors='ignore').read()
def g(p):
    m = re.search(p, t); return float(m.group(1)) if m else None
miou = g(r'Average mIoU:\s*([0-9.]+)')
d = dict(scene=scene, psnr=g(r'Average PSNR:\s*([0-9.]+)'), ssim=g(r'Average MS-SSIM:\s*([0-9.]+)'),
         lpips=g(r'Average LPIPS:\s*([0-9.]+)'), depth_l1_cm=g(r'Average Depth L1:\s*([0-9.]+)\s*cm'),
         ate_rmse_cm=g(r'Final Average ATE RMSE:\s*([0-9.]+)\s*cm'), miou_pct=miou*100 if miou is not None else None)
json.dump(d, open(out, 'w'), indent=2)
print(f"[{scene}] " + " ".join(f"{k}={v}" for k, v in d.items() if k != 'scene'))
if d['ate_rmse_cm'] == 100.0: print(f"[{scene}] WARN ATE=100 sentinel -> tracking diverged")
PY
  echo "PASS" > "$dst/status.txt"
}

# --- aggregate vs paper (REPORT-ONLY gate) ---------------------------------------------------
aggregate(){
  local py="$ENV_PY"; [ -x "$py" ] || py=python
  PYTHONPATH= "$py" - "$DRIVE" <<'PY'
import sys, os, json, glob
root = sys.argv[1]
paper = dict(psnr=34.66, ssim=0.973, lpips=0.096, depth_l1_cm=0.356, ate_rmse_cm=0.412, miou_pct=92.72)  # 8-scene avg
band  = dict(psnr=('>=',33.5), ssim=('>=',0.96), lpips=('<=',0.12),
             depth_l1_cm=('<=',0.6), ate_rmse_cm=('<=',0.6), miou_pct=('>=',90.0))
# PER-SCENE paper (Table 1 PSNR/SSIM/LPIPS; Table 3 mIoU for 4 scenes). ATE/Depth-L1 = avg only.
PS = {'room0': dict(psnr=32.50, ssim=0.976, lpips=0.070, miou_pct=92.95),
      'room1': dict(psnr=34.25, ssim=0.978, lpips=0.094, miou_pct=92.91),
      'room2': dict(psnr=35.10, ssim=0.982, lpips=0.070, miou_pct=92.10),
      'office0': dict(psnr=38.54, ssim=0.984, lpips=0.086, miou_pct=92.90),
      'office1': dict(psnr=39.20, ssim=0.980, lpips=0.087),
      'office2': dict(psnr=32.90, ssim=0.965, lpips=0.101),
      'office3': dict(psnr=32.05, ssim=0.966, lpips=0.115),
      'office4': dict(psnr=32.75, ssim=0.949, lpips=0.148)}
ms = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, '*', 'metrics.json')))]
if not ms:
    print("no per-scene metrics yet"); raise SystemExit
keys = ['psnr','ssim','lpips','depth_l1_cm','ate_rmse_cm','miou_pct']
full = len(ms) >= 8
L = [f"SGS-SLAM Replica repro - {len(ms)} scene(s): {[m['scene'] for m in ms]}", "",
     "per-scene vs PER-SCENE paper (PSNR/SSIM/LPIPS/mIoU):"]
for m in ms:
    ref = PS.get(m['scene'], {}); parts = [f"  {m['scene']:<8}"]
    for k, lab in [('psnr','PSNR'),('ssim','SSIM'),('lpips','LPIPS'),('miou_pct','mIoU')]:
        r = m.get(k)
        if r is None: continue
        p = ref.get(k); parts.append(f"{lab} {r:.3f}" + (f"/p{p:.3f}" if p is not None else "/p?"))
    parts.append(f"ATE {m.get('ate_rmse_cm')}cm DepthL1 {m.get('depth_l1_cm')}cm (paper avg 0.412/0.356)")
    L.append("  ".join(parts))
def mean(k):
    vs = [m[k] for m in ms if m.get(k) is not None]; return sum(vs)/len(vs) if vs else None
L += ["", f"subset MEAN vs 8-scene-avg gate{'' if full else '  (NOTE: avg targets are exact only at full 8; a single scene differs by scene difficulty - judge it on the per-scene line above)'}:",
      f"{'metric':<12}{'repro':>10}{'paperAvg':>10}{'band':>12}{'verdict':>9}"]
allpass = True
for k in keys:
    r = mean(k); p = paper[k]; op, th = band[k]
    if r is None: L.append(f"{k:<12}{'--':>10}{p:>10}{op+str(th):>12}{'n/a':>9}"); continue
    ok = (r >= th) if op == '>=' else (r <= th); allpass = allpass and ok
    L.append(f"{k:<12}{r:>10.3f}{p:>10.3f}{(op+str(th)):>12}{('PASS' if ok else 'MISS'):>9}")
L += ["", f"GATE (report-only): {'PASS' if allpass else 'BELOW-BAND'} - report-only, does NOT block CRCD (CONTRACT s8)"
      + ("" if full else ". PARTIAL run -> use the per-scene line, not the avg gate.")]
txt = "\n".join(L); print("\n"+txt); open(os.path.join(root, 'COMBINED.txt'), 'w').write(txt+"\n")
PY
}

case "$PHASE" in
  env)   build_env ;;
  repro|all)
    [ "$PHASE" = all ] && build_env
    [ -x "$ENV_PY" ] || { echo "FATAL: env not built ($ENV_PY missing) - run 'env' first"; exit 30; }
    stage_replica_all || { echo "FATAL: Replica not staged"; exit 1; }
    mkdir -p "$DRIVE"
    for s in ${SCENE_ARG:-$ALL_SCENES}; do run_scene "$s" || echo "[$s] skipped/failed (see $DRIVE/$s)"; done
    aggregate
    echo "DONE repro -> $DRIVE (COMBINED.txt + per-scene metrics.json/slam.log)" ;;
  crcd)
    echo "CRCD (Phase B) NOT WIRED yet: needs CRCDGradSLAMDataset + semantic_ids/colors emission"\
         " + npz->est & render-rename adapters (ARM4 A4-1.3/1.4). Stub - exiting."; exit 0 ;;
  eval)  aggregate ;;
  *) echo "usage: run_sgsslam.sh env|repro|crcd|eval|all [scene]"; exit 2 ;;
esac
