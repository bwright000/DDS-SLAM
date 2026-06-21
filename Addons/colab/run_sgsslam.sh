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
# CRCD (Phase B) WIRED: 'crcd [snippet|bench5]' (default c1_001). RECTIFIED is the DEFAULT
#   (user 06-20: rectified on ALL CRCD methods — CRCD rgb is pre-rectification/distorted + SGS is
#   a pinhole rasterizer; depth/ is the raw-space MoGe corpus -> the rectified assembly REMAPS it).
#   raw-left is DEPRECATED, behind RAW_LEFT=1 (distorted frames, pinhole-imperfect).
#   Per snippet: stage raw+MoGe -> preprocess(rectify) + assemble SGS layout + crcd.yaml -> patch
#   eval() raw-depth dump -> scripts/slam.py -> npz->est + render-rename -> DDS eval (sim3_ate
#   [CANONICAL] + eval_rendering + depth_l1 + 6-panel video) -> metrics.json/status.txt to Drive.
#   n=1, native res, replica hyperparams (swap only H/W + n_classes + intrinsics).
# RUNS ON COLAB/A100. The conda build is slow (~20-30 min, one-time) and fails LOUD.
#
# Usage:  bash Addons/colab/run_sgsslam.sh env
#         bash Addons/colab/run_sgsslam.sh repro room0
#         bash Addons/colab/run_sgsslam.sh repro
#         bash Addons/colab/run_sgsslam.sh crcd                 # c1_001, rectified (default)
#         bash Addons/colab/run_sgsslam.sh crcd c2_001
#         bash Addons/colab/run_sgsslam.sh crcd bench5          # all 5 benchmark snippets
#         RAW_LEFT=1 DEPTH_SCALE=10000 bash Addons/colab/run_sgsslam.sh crcd c1_001   # deprecated raw-left
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
  # conda torch does NOT bundle pybind11 headers and --no-build-isolation skips pip's auto build-deps,
  # so install pybind11 + put its include on CPATH (else: "fatal error: pybind11/pybind11.h: No such file").
  PYTHONPATH= "$ENV_PY" -m pip install -q pybind11 || echo "[env] WARN pybind11 install"
  local PYBIND_INC; PYBIND_INC=$(PYTHONPATH= "$ENV_PY" -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
  echo "[env] pybind11 include: $PYBIND_INC"
  # VERBOSE (no -q, -v) + tee FULL build log so the real nvcc/g++ error is visible, not "No available output"
  PYTHONPATH= CUDA_HOME="$ENV_ROOT" PATH="$ENV_ROOT/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" \
     CPATH="${PYBIND_INC}${CPATH:+:$CPATH}" \
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
  pkill -f 'unzip.*Replica' 2>/dev/null || true   # kill any stray unzip from a prior killed run
  rm -rf /content/data/_rep_tmp "$REPLICA"; mkdir -p /content/data/_rep_tmp
  # -o = overwrite, never prompt (under nohup there's no stdin -> a prompt = EOF -> "unzip failed")
  unzip -o -q "$use" -d /content/data/_rep_tmp || { echo "[replica] unzip failed"; return 1; }
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

# ============================================================================
# CRCD (Phase B) — wire SGS-SLAM onto the 5-snippet CRCD benchmark.
#   RECTIFIED is the DEFAULT (user 06-20: rectified on ALL CRCD; rgb is distorted pre-rect + SGS is
#   pinhole; depth/ is raw-space -> remapped). raw-left DEPRECATED behind RAW_LEFT=1. n=1, native
#   resolution, replica hyperparams (swap only H/W + n_classes + intrinsics). CANONICAL tracking
#   metric = sim3_ate (NEVER the rigid output.txt).
# ============================================================================
REPO=${REPO:-/content/DDS-SLAM}                            # this repo (DDS-SLAM working copy)
DDS_PY=${DDS_PY:-python}                                   # torch2 env python for DDS eval CLIs
DEPTH_SCALE=${DEPTH_SCALE:-10000}                          # MoGe png value/scale=metres; verify via Sim3 path-ratio
RAW_LEFT=${RAW_LEFT:-0}                                    # default 0 = RECTIFIED (user 06-20: rectified on ALL CRCD); 1 = DEPRECATED raw-left
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
DRIVE_CRCD_MOGE=${DRIVE_CRCD_MOGE:-/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2}
CALIB_PKL=${CALIB_PKL:-$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}
CRCD_LOCAL=${CRCD_LOCAL:-/content/data/CRCD}               # local staging root
CRCD_STAGED=${CRCD_STAGED:-/content/data/CRCD_staged}      # rectified preprocess outputs (guarded)
CRCD_DRIVE=${CRCD_DRIVE:-/content/drive/MyDrive/Outputs/SGS-SLAM_CRCD_${DATE}}
BENCH5="c1_001 c2_001 e3_005 c3_001 g3_001"               # the 5 benchmark snippets

# NAME (e.g. c1_001 / C1_001) -> "EP SID" (e.g. "C_1 001"); reuse the LOSO parser convention.
crcd_ep_sid(){
  local n; n=$(echo "$1" | tr 'a-z' 'A-Z')
  [[ "$n" =~ ^[A-Z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }
  echo "${n:0:1}_${n:1:1} ${n:3}"
}

# generate MoGe-2 depth into the published depth/ dir IF it's absent (user 06-21). MoGe runs in the
# SYSTEM torch2 python ($MOGE_PY) with the `moge` package (NOT the SGS conda env). Raw left frames,
# up-to-scale, scale=DEPTH_SCALE, written as {i:06d}.png to match the existing c1/c2 depth/ corpus
# (the rectified assembly remaps it; per-snippet Sim3 absorbs the up-to-scale factor).
MOGE_PY=${MOGE_PY:-$DDS_PY}
crcd_ensure_depth(){
  local NAME=$1 EP SID; read -r EP SID <<< "$(crcd_ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$NAME] bad name"; return 1; }
  local DD="$DRIVE_CRCD_MOGE/$EP/snippet_$SID/depth"
  [ "$(ls "$DD"/*.png 2>/dev/null | wc -l)" -gt 0 ] && { echo "[$NAME] MoGe depth present ($(ls "$DD"/*.png 2>/dev/null | wc -l) png)"; return 0; }
  echo "[$NAME] MoGe depth MISSING in $DD -> generating (MoGe-2 on raw left, scale=$DEPTH_SCALE)"
  local SRC="$DRIVE_CRCD/$EP/snippet_$SID"
  [ -d "$SRC/rgb" ] || { echo "[$NAME] no raw rgb on Drive ($SRC/rgb) -> cannot gen depth"; return 1; }
  "$MOGE_PY" -c 'from moge.model.v2 import MoGeModel' 2>/dev/null || {
    echo "[$NAME] FATAL MoGe not importable in '$MOGE_PY' (pip install git+https://github.com/microsoft/MoGe.git)"; return 1; }
  local W="/content/crcd_depthgen/$NAME"; rm -rf "$W"; mkdir -p "$W/_in" "$W/_npy"
  local f st; for f in "$SRC/rgb"/*.png; do st=$(basename "$f" .png); ln -sf "$f" "$W/_in/${st}-left.png"; done
  PYTHONPATH= "$MOGE_PY" "$REPO/Addons/depth/generate_depth_moge.py" --rgb "$W/_in" --out "$W/_npy" \
     --depth_scale "$DEPTH_SCALE" --temporal_window 1 --max_depth_m 5.0 --resolution_level 9 \
     || { echo "[$NAME] MoGe gen FAILED"; return 1; }
  mkdir -p "$DD"
  PYTHONPATH= "$MOGE_PY" - "$W/_npy" "$DD" <<'PY'
import numpy as np, cv2, glob, os, sys
npy_dir, out = sys.argv[1], sys.argv[2]
ns = sorted(glob.glob(os.path.join(npy_dir, '*-left_depth.npy')))
for i, p in enumerate(ns):
    cv2.imwrite(os.path.join(out, f'{i:06d}.png'),
                np.clip(np.load(p).astype(np.float32), 0, 65535).astype(np.uint16))
print(f'[depthgen] wrote {len(ns)} png -> {out}')
PY
  rm -rf "$W"
  [ "$(ls "$DD"/*.png 2>/dev/null | wc -l)" -gt 0 ] || { echo "[$NAME] depth-gen produced no png"; return 1; }
  echo "[$NAME] MoGe depth generated -> $DD ($(ls "$DD"/*.png 2>/dev/null | wc -l) png)"
}

# stage the raw snippet (rgb, semantic_instance, groundtruth.txt, intrinsics.yaml) + MoGe depth
# from Drive to local. Returns 0 on success; the dirs land under $CRCD_LOCAL/<NAME>/.
crcd_stage(){
  local NAME=$1 EP SID
  read -r EP SID <<< "$(crcd_ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$NAME] bad snippet name (want e.g. c1_001)"; return 1; }
  local SRC="$DRIVE_CRCD/$EP/snippet_$SID" MOGE="$DRIVE_CRCD_MOGE/$EP/snippet_$SID/depth"
  [ -d "$SRC/rgb" ] && [ -d "$SRC/semantic_instance" ] || {
    echo "[$NAME] raw snippet missing on Drive: $SRC (rgb/ + semantic_instance/)"; return 1; }
  [ -d "$MOGE" ] || { echo "[$NAME] MoGe depth missing on Drive: $MOGE"; return 1; }
  local dst="$CRCD_LOCAL/$NAME"
  rm -rf "$dst"; mkdir -p "$dst/rgb" "$dst/semantic_instance" "$dst/depth"
  cp -rn "$SRC/rgb/." "$dst/rgb/"
  cp -rn "$SRC/semantic_instance/." "$dst/semantic_instance/"
  cp -rn "$MOGE/." "$dst/depth/"
  cp -f "$SRC/groundtruth.txt" "$dst/groundtruth.txt" 2>/dev/null || {
    echo "[$NAME] FATAL no groundtruth.txt at $SRC"; return 1; }
  cp -f "$SRC/intrinsics.yaml" "$dst/intrinsics.yaml" 2>/dev/null || \
    echo "[$NAME] WARN no intrinsics.yaml (raw-K will fall back to calib pickle)"
  echo "[$NAME] staged rgb=$(ls "$dst/rgb"/*.png 2>/dev/null | wc -l) "\
       "sem=$(ls "$dst/semantic_instance"/*.png 2>/dev/null | wc -l) "\
       "depth=$(ls "$dst/depth"/*.png 2>/dev/null | wc -l)"
}

# idempotently patch the cloned SGS eval() to ALSO dump raw uint16 rendered depth (for depth_l1).
# WHY — utils/eval_helpers.py saves only a JET-colormapped png; depth_l1 needs metric uint16.
crcd_patch_raw_depth(){
  PYTHONPATH= "$ENV_PY" - "$SGS/utils/eval_helpers.py" "$DEPTH_SCALE" <<'PY'
import sys, io
path, scale = sys.argv[1], float(sys.argv[2])
lines = io.open(path, encoding='utf-8').read().split('\n')
MARK = "# [DDS] raw-depth dump"
if any(MARK in l for l in lines):
    print("[patch] raw-depth already present -> skip"); sys.exit(0)
if not any('viz_render_depth' in l for l in lines):
    print("[patch] FATAL 'viz_render_depth' not in eval_helpers.py - raw-depth var renamed; "
          "inspect the cloned source + update the patch."); sys.exit(1)
def ind(s): return s[:len(s) - len(s.lstrip())]
out, did_mk, did_save = [], False, False
for l in lines:
    out.append(l)
    # 1) make rendered_depth_raw/ right after render_depth_dir is assigned
    if (not did_mk) and ('render_depth_dir =' in l) and ('os.path.join' in l):
        sp = ind(l)
        out.append(f"{sp}render_depth_raw_dir = os.path.join(eval_dir, 'rendered_depth_raw')  {MARK}")
        out.append(f"{sp}os.makedirs(render_depth_raw_dir, exist_ok=True)")
        did_mk = True
    # 2) dump raw uint16 depth right after the JET imwrite (same loop var time_idx, same indent)
    if (not did_save) and ('cv2.imwrite(' in l) and ('render_depth_dir' in l):
        sp = ind(l)
        out.append(f"{sp}cv2.imwrite(os.path.join(render_depth_raw_dir, 'gs_{{:04d}}.png'.format(time_idx)), "
                   f"np.clip(viz_render_depth * {scale}, 0, 65535).astype(np.uint16))  {MARK}")
        did_save = True
if not (did_mk and did_save):
    print(f"[patch] FATAL anchors not found (mk={did_mk} save={did_save}) - eval_helpers.py "
          "structure changed; aborting (fix the anchors)."); sys.exit(1)
io.open(path, 'w', encoding='utf-8').write('\n'.join(out))
print(f"[patch] raw-depth dump injected (scale={scale})")
PY
}

# write the CRCD slam config by sed-ing the replica template (native H/W, 4 classes, no wandb)
crcd_write_cfg(){
  local NAME=$1 scene_dir=$2 H=$3 W=$4 cfg=$5
  sed -e "s/^scene_name = .*/scene_name = \"$NAME\"/" \
      -e "s/use_wandb=True/use_wandb=False/" \
      -e "s/num_semantic_classes=101/num_semantic_classes=4/" \
      -e "s#group_name=\"[^\"]*\"#group_name=\"CRCD\"#" \
      -e "s#basedir=\"./data/Replica\"#basedir=\"$(dirname "$scene_dir")\"#" \
      -e "s#gradslam_data_cfg=\"[^\"]*\"#gradslam_data_cfg=\"./configs/data/crcd.yaml\"#" \
      -e "s/desired_image_height=[0-9]*/desired_image_height=$H/" \
      -e "s/desired_image_width=[0-9]*/desired_image_width=$W/" \
      "$SGS/configs/replica/slam.py" > "$cfg"
}

# run ONE CRCD snippet end-to-end (assemble -> patch -> SGS -> npz->est -> rename -> DDS eval)
run_crcd_one(){
  local NAME; NAME=$(echo "$1" | tr 'A-Z' 'a-z')
  local UP; UP=$(echo "$NAME" | tr 'a-z' 'A-Z')             # C1_001 for paths/run_name
  local EP SID; read -r EP SID <<< "$(crcd_ep_sid "$NAME")"
  local OUT="$CRCD_DRIVE/$UP"; mkdir -p "$OUT"
  local scene_dir="$SGS/data/crcd/$UP"
  local local_snip="$CRCD_LOCAL/$NAME"

  crcd_ensure_depth "$NAME" || { echo "FAILED depth-gen (no published depth + MoGe gen failed)" > "$OUT/status.txt"; return 1; }
  crcd_stage "$NAME" || { echo "FAILED stage" > "$OUT/status.txt"; return 1; }

  # ---- assemble into the SGS ReplicaDataset layout + emit crcd.yaml ----
  rm -rf "$scene_dir"
  if [ "$RAW_LEFT" != 1 ]; then
    echo "[$NAME] RECTIFIED assembly (DEFAULT; rectified-on-all; depth/ is raw-space -> remapped via left map)"
    local staged="$CRCD_STAGED/$UP"; rm -rf "$staged"; mkdir -p "$staged"
    PYTHONPATH= "$DDS_PY" "$REPO/Addons/preprocess/preprocess_crcd_published.py" \
       --snippet_dir "$local_snip" --calib_pkl "$CALIB_PKL" --output_dir "$staged" \
       || { echo "FAILED preprocess" > "$OUT/status.txt"; return 1; }
    PYTHONPATH= "$DDS_PY" "$REPO/Addons/colab/crcd_assemble_sgs.py" --mode rectified \
       --staged "$staged" --moge_depth "$local_snip/depth" --calib_pkl "$CALIB_PKL" \
       --out "$scene_dir" --depth_scale "$DEPTH_SCALE" --n_classes 4 \
       --emit_yaml "$SGS/configs/data/crcd.yaml" \
       || { echo "FAILED assemble(rect)" > "$OUT/status.txt"; return 1; }
  else
    echo "[$NAME] RAW_LEFT=1 -> DEPRECATED raw-left assembly (distorted frames; pinhole-imperfect; non-default)"
    PYTHONPATH= "$DDS_PY" "$REPO/Addons/colab/crcd_assemble_sgs.py" --mode rawleft \
       --rgb_dir "$local_snip/rgb" --sem_dir "$local_snip/semantic_instance" \
       --moge_depth "$local_snip/depth" --groundtruth "$local_snip/groundtruth.txt" \
       --intrinsics_yaml "$local_snip/intrinsics.yaml" --calib_pkl "$CALIB_PKL" \
       --out "$scene_dir" --depth_scale "$DEPTH_SCALE" --n_classes 4 \
       --emit_yaml "$SGS/configs/data/crcd.yaml" \
       || { echo "FAILED assemble(rawleft)" > "$OUT/status.txt"; return 1; }
  fi

  # native res = the assembled frame size (read from crcd.yaml we just wrote)
  local H W
  H=$(grep -E '^\s*image_height:' "$SGS/configs/data/crcd.yaml" | grep -oE '[0-9]+' | head -1)
  W=$(grep -E '^\s*image_width:'  "$SGS/configs/data/crcd.yaml" | grep -oE '[0-9]+' | head -1)
  [ -n "$H" ] && [ -n "$W" ] || { echo "FAILED no H/W from crcd.yaml" > "$OUT/status.txt"; return 1; }
  echo "[$NAME] native res H=$H W=$W"

  local cfg="/content/sgs_crcd_${NAME}.py"
  crcd_write_cfg "$UP" "$scene_dir" "$H" "$W" "$cfg"
  crcd_patch_raw_depth || { echo "FAILED depth-patch (eval_helpers anchors changed)" > "$OUT/status.txt"; return 1; }

  # ---- run SGS-SLAM ----
  echo "[$NAME] running SGS-SLAM (full SLAM pass)"
  ( cd "$SGS" && PYTHONPATH= "$ENV_PY" scripts/slam.py "$cfg" ) 2>&1 | tee "$OUT/slam.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED slam.py" > "$OUT/status.txt"; return 1; }

  # SGS output_dir = experiments/<group_name>/<run_name>; run_name=<scene>_<seed=0>
  local output_dir="$SGS/experiments/CRCD/${UP}_0"
  [ -f "$output_dir/params.npz" ] || {
    # fall back: find the newest params.npz under experiments/CRCD
    output_dir=$(dirname "$(ls -t "$SGS"/experiments/CRCD/*/params.npz 2>/dev/null | head -1)")
  }
  [ -f "$output_dir/params.npz" ] || { echo "FAILED no params.npz" > "$OUT/status.txt"; return 1; }
  echo "[$NAME] SGS output_dir=$output_dir"
  [ "$(ls "$output_dir/eval/rendered_depth_raw"/*.png 2>/dev/null | wc -l)" -gt 0 ] || \
     echo "[$NAME] WARN rendered_depth_raw empty -> depth_l1 will be nan (patch fired? eval save_frames on?)"

  # ---- npz -> est_c2w_data.txt ----
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/sgsslam_npz_to_est.py" \
     --npz "$output_dir/params.npz" --out "$OUT/est_c2w_data.txt" \
     || { echo "FAILED npz->est" > "$OUT/status.txt"; return 1; }
  [ -s "$OUT/est_c2w_data.txt" ] || { echo "FAILED empty est" > "$OUT/status.txt"; return 1; }

  # ---- rename renders for eval_rendering Mode-1: render gs_{idx}.png -> $OUT/{idx}.jpg,
  #      and copy the scene GT frame -> $OUT/{idx}_gt.png (pairs by stem). Renders are STRIDED
  #      by config eval_every (default 5) so only those indices get a GT sibling. ----
  PYTHONPATH= "$DDS_PY" - "$output_dir/eval/rendered_rgb" "$scene_dir/frames" "$OUT" <<'PY'
import sys, os, glob, re, shutil
ren_dir, frames_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
n = 0
for p in sorted(glob.glob(os.path.join(ren_dir, 'gs_*.png'))):
    m = re.search(r'gs_(\d+)\.png$', os.path.basename(p))
    if not m: continue
    idx = int(m.group(1))
    shutil.copy(p, os.path.join(out, f'{idx}.jpg'))                       # render
    gt = os.path.join(frames_dir, f'frame{idx:06d}.jpg')
    if os.path.exists(gt):
        shutil.copy(gt, os.path.join(out, f'{idx}_gt.png'))              # GT sibling (Mode-1)
    n += 1
print(f"[rename] {n} renders -> {out}/{{idx}}.jpg (+ {{idx}}_gt.png)")
PY

  # ---- DDS eval: sim3 ATE (canonical) + render PSNR/SSIM/LPIPS + depth L1 + 6-panel video ----
  # sim3_ate --out and eval_rendering --summary_csv both APPEND; clear stale files so a re-run is fresh.
  rm -f "$OUT/sim3_metrics.txt" "$OUT/render_eval.txt" "$OUT/render_eval.csv"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/sim3_ate.py" \
     --est "$OUT/est_c2w_data.txt" --gt "$local_snip/groundtruth.txt" \
     --name "CRCD $UP" --out "$OUT/sim3_metrics.txt" || echo "[$NAME] WARN sim3_ate failed"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/eval_rendering.py" \
     --gt_dir "$OUT" --render_dir "$OUT" --sequence "CRCD ($UP)" \
     --output_csv "$OUT/render_eval.csv" --summary_csv "$OUT/render_eval.txt" \
     || echo "[$NAME] WARN eval_rendering failed"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/depth_l1.py" \
     --render_depth_dir "$output_dir/eval/rendered_depth_raw" \
     --input_depth_dir "$scene_dir/depths" \
     --render_scale "$DEPTH_SCALE" --input_scale "$DEPTH_SCALE" --sc_factor 1.0 \
     --out "$OUT/depth_l1.txt" || echo "[$NAME] WARN depth_l1 failed"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/viz/generate_video.py" \
     --rgb_input_dir "$OUT" --rgb_output_dir "$OUT" \
     --depth_input_dir "$scene_dir/depths" \
     --depth_output_dir "$output_dir/eval/rendered_depth_raw" \
     --trajectory_est "$OUT/est_c2w_data.txt" --trajectory_gt "$local_snip/groundtruth.txt" \
     --png_depth_scale "$DEPTH_SCALE" --output "$OUT/video.mp4" \
     || echo "[$NAME] WARN generate_video failed"

  # ---- parse metrics.json + PASS ----
  PYTHONPATH= "$DDS_PY" - "$OUT" "$UP" <<'PY'
import sys, os, re, json, csv
out, name = sys.argv[1], sys.argv[2]
def _txt(fn):
    p = os.path.join(out, fn)
    return open(p, encoding='utf-8', errors='ignore').read() if os.path.isfile(p) else ''
def grab(s, pat):
    m = re.search(pat, s); return float(m.group(1)) if m else None
# --- sim3_metrics.txt: "Sim3 ATE  rmse/mean/median/max : R / M / Md / Mx mm" ---
sim = _txt('sim3_metrics.txt')
ate_mean = None
m = re.search(r'Sim3 ATE.*?:\s*[0-9.]+\s*/\s*([0-9.]+)\s*/', sim)
if m: ate_mean = float(m.group(1))
# --- render_eval.txt is a CSV (header + 1 data row); pull psnr_mean/ssim_mean/lpips_mean ---
psnr = ssim = lpips = None
rp = os.path.join(out, 'render_eval.txt')
if os.path.isfile(rp):
    try:
        rows = list(csv.DictReader(open(rp)))
        if rows:
            r = rows[-1]
            psnr = float(r['psnr_mean']) if r.get('psnr_mean') else None
            ssim = float(r['ssim_mean']) if r.get('ssim_mean') else None
            lpips = float(r['lpips_mean']) if r.get('lpips_mean') else None
    except Exception as e:
        print(f"[metrics] WARN render_eval parse: {e}")
d = dict(scene=name,
         sim3_ate_mean_mm=ate_mean,
         sim3_ate_max_mm=grab(sim, r'Sim3 ATE.*?:\s*[0-9.]+\s*/\s*[0-9.]+\s*/\s*[0-9.]+\s*/\s*([0-9.]+)\s*mm'),
         path_ratio=grab(sim, r'est/GT path ratio\s*:\s*([0-9.]+)'),
         pearson_dom=grab(sim, r'\|Pearson\| dom axis\s*:\s*([0-9.]+)'),
         psnr=psnr, ssim=ssim, lpips=lpips,
         depth_l1_mm=grab(_txt('depth_l1.txt'), r'mean=([0-9.]+)'))
json.dump(d, open(os.path.join(out, 'metrics.json'), 'w'), indent=2)
print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in d.items() if k != 'scene'))
PY
  echo "PASS" > "$OUT/status.txt"
  echo "[$NAME] DONE -> $OUT"
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
    [ -x "$ENV_PY" ] || { echo "FATAL: SGS env not built ($ENV_PY missing) - run 'env' first"; exit 30; }
    [ -d "$SGS/.git" ] || { echo "FATAL: SGS clone missing at $SGS - run 'env' first"; exit 30; }
    mkdir -p "$CRCD_DRIVE"
    SNIPS=${SCENE_ARG:-c1_001}                  # default single snippet; pass 'bench5' for all 5
    [ "$SNIPS" = bench5 ] && SNIPS="$BENCH5"
    echo "=== CRCD phase: snippets='$SNIPS'  mode=$([ "$RAW_LEFT" = 1 ] && echo rawleft-DEPRECATED || echo rectified)"\
         " depth_scale=$DEPTH_SCALE -> $CRCD_DRIVE ==="
    for s in $SNIPS; do run_crcd_one "$s" || echo "[$s] FAILED (see $CRCD_DRIVE/$(echo "$s" | tr a-z A-Z)/status.txt)"; done
    echo "DONE crcd -> $CRCD_DRIVE (per-snippet metrics.json/status.txt/video.mp4)" ;;
  eval)  aggregate ;;
  *) echo "usage: run_sgsslam.sh env|repro|crcd|eval|all [scene]"; exit 2 ;;
esac
