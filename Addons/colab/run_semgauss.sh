#!/bin/bash
# ============================================================================
# run_semgauss.sh <phase> [snippet]  -  Arm-4 method #3: SemGauss-SLAM onboarding
#   phase: env | crcd [snippet|bench5] | eval
#
# WHAT — wire SemGauss-SLAM (dense semantic 3DGS SLAM, IRMVLab @8e8db0a) onto the
#   RECTIFIED 5-snippet CRCD benchmark and judge it with the SAME DDS harness battery
#   used for SGS/SNI/DDS: Sim3 ATE (CANONICAL, NEVER rigid) + render PSNR/SSIM/LPIPS +
#   Depth-L1 + the canonical 6-panel video. We do NOT improve SemGauss — only configure
#   it faithfully for CRCD. Method dynamics (iters/lrs/loss-weights/keyframe cadence/BA)
#   are kept BYTE-for-BYTE from configs/replica/replica.py; only the CRCD data config
#   (H/W, 4 classes, intrinsics, seg head) + 2 mechanical patches + 1 preflight change.
#
# SemGauss is SplaTAM-family (exactly like SGS-SLAM): per-frame poses live inside
#   params.npz as cam_unnorm_rots (1,4,N) + cam_trans (1,3,N) [w2c rel. to frame 0] -> the
#   SAME sgsslam_npz_to_est.py converts them. Renders land at eval/rendered_rgb/gs_%04d.png.
#
# ENV — SemGauss is a py3.10 / torch1.12.1 / cu11.6 repo (requirements.txt is a conda
#   `conda create --file` export, so pip cannot read it -> deps installed explicitly).
#   Colab's system stack won't satisfy its pins, so we build the README conda env
#   (miniconda -> py3.10 -> cuda-toolkit 11.6 -> torch 1.12.1+cu116 -> the IN-REPO
#   diff-gaussian-rasterization-w-depth_sem_gauss, glm vendored -> NO --recursive).
#   ⚠️ SemGauss's rasterizer installs under module `diff_gaussian_rasterization` (same
#   name as SGS-SLAM's, INCOMPATIBLE build) -> use a DISTINCT conda env `sem_gauss`;
#   NEVER share it with the sgs-slam env.
#
# GPU — cu11.6 covers T4 (sm_75) + A100 (sm_80). Do NOT schedule on L4 (sm_89) or H100
#   (sm_90): cu11.6 nvcc cannot target those arches (build fails / PTX-JIT only). T4 3DGS
#   OOMs on long snippets (~>450 frames) -> run long snippets on A100, or port a
#   NUM_FRAMES cap for a T4 smoke. Shortest snippet (e3_005) runs FIRST as the de-facto smoke.
#
# FAITHFUL-BENCHMARK ADJUSTMENTS (each deliberate; none touch method dynamics):
#   - data = rectified assembly (crcd_assemble_sgs.py --mode semgauss): rgb/rgb_%06d.png +
#     depth/depth_%06d.png (rectified with the left map) + semantic_remap/semantic_%06d.png +
#     traj.txt + a byte-compatible configs/data/crcd.yaml. CRCD rgb is pre-rectification /
#     distorted and SemGauss is a pinhole rasterizer -> rectified is mandatory.
#   - use_gt_semantic=False (METHOD-FAITHFUL DINO self-supervision). The seg net is OUR
#     in-domain 4-class DINOv2/14 head (loso_v2_max/dinov2_crcd_<UP>.pth, snippet HELD OUT).
#   - n_classes=4 (bg/Liver/Gallbladder/Tool). H/W = the rectified frame size (from crcd.yaml).
#   - mIoU dropped on CRCD (benchmark = the 5 metrics + video). GT masks are loader INPUT
#     only (1 PNG per rgb), never supervision under use_gt_semantic=False.
#
# THE 2 MECHANICAL PATCHES + 1 PREFLIGHT (only these touch SemGauss source):
#   (a) utils/dinov2_seg.py:9  hardcoded sys.path.append('/data0/.../facebookresearch_dinov2_main')
#       -> the in-repo vendored backbone (segmentation/facebookresearch_dinov2_main). idempotent+loud.
#   (b) utils/eval_utils.py    the depth-viz save hands cv2 a float64 `viz_render_depth*1000.0`
#       (PNG saturate-casts -> WRONG on a ~0.1m surgical scene). Inject a sibling raw uint16 dump
#       rendered_depth_raw/gs_%04d.png = clip(viz_render_depth*SCALE,0,65535) for depth_l1. idempotent+loud.
#   (preflight) dry-load the loso head into DINO2SEG(H,W,cls=4,edge=0,dim=16) BEFORE the run.
#       SemGauss loads strict=True mid-run and would hard-crash on a mismatched head; the preflight
#       turns that into an isolated .FAILED + skip.
#
# Usage (fresh Colab, Drive mounted):
#   bash Addons/colab/run_semgauss.sh env                 # sem_gauss env + clone + patch (one-time)
#   bash Addons/colab/run_semgauss.sh crcd                # bench5 (all 5, shortest-first)
#   bash Addons/colab/run_semgauss.sh crcd e3_005         # one snippet (T4 smoke)
#   NUM_FRAMES=200 bash Addons/colab/run_semgauss.sh crcd c1_001   # T4 frame cap
#   bash Addons/colab/run_semgauss.sh eval                # (re)aggregate
# ============================================================================
set -uo pipefail
VERB=${1:-}; ARG=${2:-}
DATE=${DATE:-$(date +%Y%m%d)}

# ---- repos / pythons -------------------------------------------------------
SEMGAUSS=${SEMGAUSS:-/content/SemGauss-SLAM}                 # IRMVLab clone (NOT ShuhongLL = SGS)
SEMGAUSS_URL=${SEMGAUSS_URL:-https://github.com/IRMVLab/SemGauss-SLAM}
REPO=${REPO:-/content/DDS-SLAM}                              # this repo (DDS working copy)
DDS_PY=${DDS_PY:-python}                                     # torch2 MoGe python: depth-gen + DDS eval CLIs
MOGE_PY=${MOGE_PY:-$DDS_PY}                                  # MoGe-2 import lives in the same torch2 python
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
ENV_NAME=${ENV_NAME:-sem_gauss}                             # DISTINCT env (rasterizer name-clashes with sgs-slam)
ENV_PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"              # the sem_gauss conda python (runs sem_gauss.py)

# ---- seg heads (OUR in-domain DINOv2/14 heads; snippet HELD OUT) -----------
SEG_DIR_DRIVE=${SEG_DIR_DRIVE:-/content/drive/MyDrive/Outputs/seg}
SEG_PTH=${SEG_PTH:-}                                        # explicit override wins

# ---- CRCD data -------------------------------------------------------------
DEPTH_SCALE=${DEPTH_SCALE:-10000}                           # MoGe png value/scale=metres (verify via Sim3 path-ratio)
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
DRIVE_CRCD_MOGE=${DRIVE_CRCD_MOGE:-/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2}
CALIB_PKL=${CALIB_PKL:-$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}
CRCD_LOCAL=${CRCD_LOCAL:-/content/data/CRCD}                # local raw+MoGe staging root
CRCD_STAGED=${CRCD_STAGED:-/content/data/CRCD_staged}       # rectified preprocess outputs

# ---- output ----------------------------------------------------------------
DRIVE_OUT=${DRIVE_OUT:-/content/drive/MyDrive/Outputs/SemGauss-SLAM_bench_$DATE}
SEED=${SEED:-2027}                                          # pass-through (replica default)
FORCE=${FORCE:-0}                                           # 1 = ignore .DONE, redo
NUM_FRAMES=${NUM_FRAMES:-}                                  # T4 frame cap (empty = -1 = all)

BENCH5="e3_005 c1_001 c2_001 c3_001 g3_001"                # SHORTEST-FIRST (e3_005 = T4 smoke)

# ⚠️ CONFIRM-IF-UNSURE knobs (grepped from sem_gauss.py/eval_utils.py; change here if the repo moves):
#   render RGB subdir  = eval/rendered_rgb   (eval_utils.py:186 -> gs_%04d.png)
#   render depth RAW   = eval/rendered_depth_raw   (injected by patch (b), sibling of eval/rendered_depth)
#   params.npz         = experiments/<group_name>/<run_name>/params.npz   (common_utils.py save_params)
SEMGAUSS_REN_RGB_SUB=${SEMGAUSS_REN_RGB_SUB:-eval/rendered_rgb}
SEMGAUSS_REN_DEPTH_SUB=${SEMGAUSS_REN_DEPTH_SUB:-eval/rendered_depth_raw}

say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }

# NAME (c1_001 / C1_001) -> "EP SID" (e.g. "C_1 001") — shared LOSO/CRCD parser convention.
crcd_ep_sid(){
  local n; n=$(echo "$1" | tr 'a-z' 'A-Z')
  [[ "$n" =~ ^[A-Z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }
  echo "${n:0:1}_${n:1:1} ${n:3}"
}

# per-snippet in-domain seg head. Precedence (canonical 2026-07-04): explicit SEG_PTH ->
# loso_v2_max fold (CANONICAL: the complete LOSO set -> one recipe for all snippets) ->
# loso_v2_b2 -> flat 15-snippet head -> loso_ref. dinov3 heads are patch-16 and can NOT load
# into SemGauss's /14 DINO2SEG -> never resolved here. Arg = UPPERCASE name (C1_001).
seg_head_for(){ local UP=$1 c
  for c in "${SEG_PTH:-}" \
           "$SEG_DIR_DRIVE/loso_v2_max/dinov2_crcd_${UP}.pth" \
           "$SEG_DIR_DRIVE/loso_v2_b2/dinov2_crcd_${UP}.pth" \
           "$SEG_DIR_DRIVE/dinov2_crcd.pth" \
           "$SEG_DIR_DRIVE/loso_ref/dinov2_crcd_${UP}.pth"; do
    [ -n "$c" ] && [ -f "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# ---------------------------------------------------------------- ENV -------
# patch (a): repoint dinov2_seg.py's hardcoded backbone sys.path to the in-repo vendored copy.
apply_dinov2_path_patch(){
  local F="$SEMGAUSS/utils/dinov2_seg.py"
  [ -f "$F" ] || { echo "[env] FATAL $F missing (clone incomplete)"; return 1; }
  if grep -q "os.path.dirname(__file__), '..', 'segmentation'" "$F"; then
    echo "[env] dinov2_seg sys.path patch already applied"; return 0; fi
  grep -q "sys.path.append('/data0/3dg/splatam/segmentation/facebookresearch_dinov2_main')" "$F" || {
    echo "[env] FATAL dinov2_seg.py sys.path anchor gone (hardcoded /data0 append) -> inspect + update patch (a)"; return 1; }
  python3 - "$F" <<'PY' || return 1
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
old = "sys.path.append('/data0/3dg/splatam/segmentation/facebookresearch_dinov2_main')"
new = "sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'segmentation', 'facebookresearch_dinov2_main'))"
assert old in s, "anchor gone (should have been caught by the grep)"
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print("[env] patched dinov2_seg.py sys.path -> in-repo segmentation/facebookresearch_dinov2_main")
PY
}

build_env(){
  [ -d "$SEMGAUSS/.git" ] || git clone "$SEMGAUSS_URL" "$SEMGAUSS" || { echo "FATAL clone $SEMGAUSS_URL"; exit 30; }
  apply_dinov2_path_patch || exit 30
  if [ "${REBUILD_RAST:-0}" != 1 ] && [ -x "$ENV_PY" ] \
     && PYTHONPATH= "$ENV_PY" -c "import diff_gaussian_rasterization" 2>/dev/null; then
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
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" -c conda-forge python=3.10 pip setuptools wheel \
     || { echo "FATAL conda create"; exit 30; }
  # conda-forge's python does NOT bundle pip -> every `$ENV_PY -m pip` = "No module named pip".
  # Ensure it explicitly (idempotent; ALSO repairs an env created before this line existed).
  PYTHONPATH= "$ENV_PY" -m pip --version >/dev/null 2>&1 \
     || conda install -y -n "$ENV_NAME" -c conda-forge pip setuptools wheel \
     || { echo "FATAL: pip not installable into $ENV_NAME"; exit 30; }
  # Install BY NAME / via "$ENV_PY -m pip" - do NOT rely on `conda activate` (on Colab the PATH
  # never switches and system py site-packages leak in via PYTHONPATH). PYTHONPATH= isolates the
  # env's py3.10 from Colab's system packages (mixing them segfaults / ModuleNotFound).
  local ENV_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
  echo "[env] cuda-toolkit 11.6 (nvcc) into $ENV_NAME"
  conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-11.6.0" cuda-toolkit || echo "[env] WARN cuda-toolkit"
  echo "[env] torch 1.12.1 + cu116 via conda (+ mkl/numpy pins: torch1.12 breaks on mkl>=2024 & numpy>=2)"
  conda install -y -n "$ENV_NAME" -c pytorch -c conda-forge \
     pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 "numpy<2" "mkl<2024" \
     || { echo "FATAL torch install (conda pytorch channel)"; exit 30; }
  PYTHONPATH= "$ENV_PY" -c "import torch,numpy; print('[env] torch',torch.__version__,'numpy',numpy.__version__,'OK')" \
     || { echo "FATAL: torch import broken (mkl/numpy ABI) - see error above"; exit 30; }
  echo "[env] pip deps (explicit; requirements.txt is a conda-export -> pip cannot read it)"
  # 🚨 PIN torch + numpy in the constraints. timm/kornia/torchmetrics/lpips all `install_require` torch,
  # and pip WILL upgrade the conda torch 1.12.1/cu116 to a cu130 torch-2.x wheel to satisfy their LATEST
  # release -> that mismatches cuda-toolkit 11.6 and the rasterizer build dies ("detected CUDA 11.6
  # mismatches PyTorch 13.0"). Pinning torch here makes pip BACKTRACK to torch-1.12-compatible dep
  # versions instead of bumping torch. (numpy pin: a dep otherwise pulls numpy 2.x -> torch ABI break.)
  printf 'numpy<2\ntorch==1.12.1\ntorchvision==0.13.1\ntorchaudio==0.12.1\n' > /tmp/semgauss_constraints.txt
  PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/semgauss_constraints.txt ninja wheel setuptools pybind11 || true
  PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/semgauss_constraints.txt \
     "numpy<2" timm kornia opencv-python lpips pytorch-msssim torchmetrics open3d==0.16.0 \
     plyfile imageio natsort matplotlib wandb trimesh \
     || echo "[env] WARN some pip deps failed (inspect above - open3d 0.16.0 / py3.10 is the likely landmine)"
  PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/semgauss_constraints.txt "numpy<2" || true
  # DEFENSIVE drift-guard: if torch still drifted off 1.12/cu116 (unpinned dep, or an env corrupted by a
  # prior run), force conda back -> else the cu116 rasterizer build fails. cu116 wheel comes from conda,
  # NOT pip (pip's torch==1.12.1 is cpu/cu102). --force-reinstall overrides conda's "already installed".
  PYTHONPATH= "$ENV_PY" -c "import torch;v=torch.__version__;c=str(torch.version.cuda);assert v.startswith('1.12') and c.startswith('11'),v+'/'+c" 2>/dev/null \
     || { echo "[env] torch drifted off 1.12.1/cu116 -> conda --force-reinstall (repairs a corrupted env)";
          conda install -y -n "$ENV_NAME" -c pytorch -c conda-forge --force-reinstall \
             pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 "mkl<2024" \
             || { echo "FATAL: could not restore torch 1.12.1/cu116"; exit 30; }
          # re-run the pip deps now that torch is pinned-correct (they were built against the wrong torch)
          PYTHONPATH= "$ENV_PY" -m pip install -q -c /tmp/semgauss_constraints.txt --force-reinstall --no-deps \
             timm kornia torchmetrics || echo "[env] WARN dep re-pin partial"; }
  # Build the IN-REPO rasterizer for THIS GPU's compute capability (+PTX). glm is vendored under
  # third_party/glm (setup.py -I's it) -> NO --recursive clone. sm_80-only kernels give
  # "numel: integer multiplication overflow" off an A100 -> match the live GPU.
  local CC ARCH
  CC=$(PYTHONPATH= "$ENV_PY" -c "import torch;print('%d.%d'%torch.cuda.get_device_capability())" 2>/dev/null)
  [ -n "$CC" ] || CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ -n "$CC" ] || CC=7.5      # safe floor: 7.5+PTX SASS also JIT-runs on newer GPUs
  case "$CC" in
    8.9|9.0) echo "[env] WARN GPU cc=$CC (L4/H100): cu11.6 nvcc cannot target sm_$CC -> build may fail / PTX-JIT only. Schedule on T4(sm_75) or A100(sm_80).";;
  esac
  ARCH="${ARCH_OVERRIDE:-${CC}+PTX}"
  echo "[env] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) compute_cap=$CC -> arch $ARCH"
  local PYBIND_INC; PYBIND_INC=$(PYTHONPATH= "$ENV_PY" -c "import pybind11; print(pybind11.get_include())" 2>/dev/null)
  echo "[env] pybind11 include: $PYBIND_INC"
  # CUDA 11.6 nvcc supports host GCC <=10, but Colab's default is GCC 11 -> the rasterizer's
  # rasterizer_impl.cu fails ("parameter packs not expanded with '...'" in std_function.h). Install
  # gcc/g++-10 and route nvcc's host compiler (CUDAHOSTCXX) + the C++ steps (CC/CXX) through it.
  # (SGS builds fine on GCC 11 because it's cu118, which accepts GCC 11; cu116 does not.)
  local HOSTENV=""
  if ! command -v g++-10 >/dev/null 2>&1; then
    echo "[env] installing gcc-10/g++-10 (CUDA 11.6 nvcc needs host GCC<=10; Colab default is 11)"
    sudo apt-get -qq update >/dev/null 2>&1 && sudo apt-get -qq install -y gcc-10 g++-10 >/dev/null 2>&1 || true
  fi
  if command -v g++-10 >/dev/null 2>&1; then
    HOSTENV="CUDAHOSTCXX=$(command -v g++-10) CC=$(command -v gcc-10) CXX=$(command -v g++-10)"
    echo "[env] nvcc host compiler -> $(command -v g++-10)"
  else
    echo "[env] WARN g++-10 unavailable -> rasterizer build may fail (GCC11 vs CUDA11.6)"
  fi
  echo "[env] in-repo rasterizer build (VERBOSE; arch=$ARCH) -> $SEMGAUSS/diff-gaussian-rasterization-w-depth_sem_gauss"
  PYTHONPATH= CUDA_HOME="$ENV_ROOT" PATH="$ENV_ROOT/bin:$PATH" TORCH_CUDA_ARCH_LIST="$ARCH" \
     CPATH="${PYBIND_INC}${CPATH:+:$CPATH}" $HOSTENV \
     "$ENV_PY" -m pip install --no-build-isolation --force-reinstall --no-deps -v \
     "$SEMGAUSS/diff-gaussian-rasterization-w-depth_sem_gauss" 2>&1 | tee /content/semgauss_rasterizer_build.log
  PYTHONPATH= "$ENV_PY" -c "import diff_gaussian_rasterization" 2>/dev/null \
     && echo "[env] rasterizer import OK" \
     || echo "[env] WARN rasterizer build FAILED -> FULL log at /content/semgauss_rasterizer_build.log (paste the nvcc/g++ error lines)"
  echo "[env] smoke import:"
  PYTHONPATH= "$ENV_PY" - <<'PY'
import importlib, sys
ok = True
for m in ("torch", "diff_gaussian_rasterization", "timm", "pytorch_msssim", "torchmetrics", "open3d"):
    try: importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e: ok = False; print(f"  {m}: FAIL -> {e}")
try:
    from diff_gaussian_rasterization import GaussianRasterizer; print("  GaussianRasterizer: OK")
except Exception as e: ok = False; print(f"  GaussianRasterizer: FAIL -> {e}")
import torch; print("  torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
sys.exit(0 if ok else 30)
PY
  [ $? -eq 0 ] || { echo "FATAL[env]: smoke import failed (exit 30). Inspect the rasterizer build log above."; exit 30; }
  echo "[env] OK -> $ENV_PY"
}

# ---------------------------------------------------- CRCD depth / stage ----
# generate MoGe-2 depth into the published depth/ dir IF absent (MoGe runs in $MOGE_PY torch2).
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
# from Drive to local -> $CRCD_LOCAL/<NAME>/. groundtruth.txt is the CRCD-Published 360-row GT.
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

# patch (b): idempotently add a raw uint16 rendered-depth dump to SemGauss's eval() (for depth_l1).
# The stock save writes cv2.imwrite(..., viz_render_depth * 1000.0) -> float64 saturate-cast (wrong on
# a ~0.1m scene). Inject a sibling rendered_depth_raw/gs_%04d.png = clip(viz_render_depth*SCALE,0,65535).
# Anchors (all CONFIRMED present in utils/eval_utils.py): `render_depth_dir = os.path.join(...)`,
# `cv2.imwrite(...render_depth_dir...)`, and the `viz_render_depth` var. Fails loud if any anchor moved.
crcd_patch_raw_depth(){
  PYTHONPATH= "$ENV_PY" - "$SEMGAUSS/utils/eval_utils.py" "$DEPTH_SCALE" <<'PY'
import sys, io
path, scale = sys.argv[1], float(sys.argv[2])
lines = io.open(path, encoding='utf-8').read().split('\n')
MARK = "# [DDS] raw-depth dump"
if any(MARK in l for l in lines):
    print("[patch] raw-depth already present -> skip"); sys.exit(0)
if not any('viz_render_depth' in l for l in lines):
    print("[patch] FATAL 'viz_render_depth' not in eval_utils.py - raw-depth var renamed; "
          "inspect the source + update the patch."); sys.exit(1)
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
    # 2) dump raw uint16 depth right after the depth-viz imwrite (same loop var time_idx, same indent)
    if (not did_save) and ('cv2.imwrite(' in l) and ('render_depth_dir' in l):
        sp = ind(l)
        out.append(f"{sp}cv2.imwrite(os.path.join(render_depth_raw_dir, 'gs_{{:04d}}.png'.format(time_idx)), "
                   f"np.clip(viz_render_depth * {scale}, 0, 65535).astype(np.uint16))  {MARK}")
        did_save = True
if not (did_mk and did_save):
    print(f"[patch] FATAL anchors not found (mk={did_mk} save={did_save}) - eval_utils.py "
          "structure changed; aborting (fix the anchors)."); sys.exit(1)
io.open(path, 'w', encoding='utf-8').write('\n'.join(out))
print(f"[patch] raw-depth dump injected into eval_utils.py (scale={scale})")
PY
}

# write the CRCD experiment config from a TEMPLATE (nested python dict -> heredoc, NOT sed).
# Starts from configs/replica/replica.py; overrides ONLY the CRCD-specific keys. Every method
# dynamic (map/keyframe/BA cadence, all iters/sil_thres/loss_weights/lrs, pruning_dict) is
# BYTE-for-BYTE the replica value. H/W stamped from crcd.yaml -> model.H==desired==yaml (mandatory).
semgauss_write_cfg(){
  local UP=$1 H=$2 W=$3 HEADABS=$4 BASEDIR=$5 cfg=$6 NF=$7
  mkdir -p "$(dirname "$cfg")"
  cat > "$cfg" <<PYCFG
# AUTO-GENERATED by run_semgauss.sh for CRCD snippet $UP (faithful; method dynamics == replica.py).
import os
from os.path import join as p_join

primary_device = "cuda:0"
seed = $SEED
scene_name = "$UP"

map_every = 8
keyframe_every = 5

mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "CRCD"
run_name = f"{scene_name}_{seed}"

config = dict(
    workdir=f"./experiments/{group_name}",
    group_name=group_name,
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every,
    keyframe_every=keyframe_every,
    BA_every=32,
    BA_iters=15,
    mapping_window_size=mapping_window_size,
    report_global_progress_every=1000,
    eval_every=1,                       # CRCD: dense renders (every frame) for the harness
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=1499,
    save_checkpoints=True,
    checkpoint_interval=300,
    use_gt_semantic=False,              # FAITHFUL: DINO self-supervision (NOT GT masks)
    model=dict(
        c_dim=16,
        pretrained_model_path="$HEADABS",   # OUR in-domain loso 4-class DINOv2/14 head (snippet held out)
        n_classes=4,                    # bg / Liver / Gallbladder / Tool
        crop_edge=0,
        H=$H,                           # == data.desired_image_height == crcd.yaml image_height
        W=$W,                           # == data.desired_image_width  == crcd.yaml image_width
    ),
    data=dict(
        basedir="$BASEDIR",             # parent of data/crcd/$UP
        gradslam_data_cfg="./configs/data/crcd.yaml",
        sequence="$UP",
        desired_image_height=$H,
        desired_image_width=$W,
        start=0,
        end=-1,
        stride=1,
        num_frames=$NF,
    ),
    BA=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=40,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        visualize_tracking_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            se=0.004,
        ),
        lrs=dict(
            means3D=0.000001,
            rgb_colors=0.000025,
            sem_labels=0.000025,
            unnorm_rotations=0.00001,
            logit_opacities=0.0005,
            log_scales=0.00001,
            cam_unnorm_rots=0.0000004,
            cam_trans=0.000002,
        ),
    ),
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        visualize_tracking_loss=True,    # VERBATIM replica (never fires: it needs frame_idx%1000==0 with a >0
                                         # tracking frame; CRCD snippets are <1000 frames -> no functional effect)
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            se=0,
            se_fe=0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            sem_labels=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            se=0.01,
            se_fe=0.01,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            sem_labels=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),
    ),
)
PYCFG
}

# ------------------------------------------------- run ONE CRCD snippet -----
run_crcd_one(){
  local NAME; NAME=$(echo "$1" | tr 'A-Z' 'a-z')
  local UP; UP=$(echo "$NAME" | tr 'a-z' 'A-Z')
  local EP SID; read -r EP SID <<< "$(crcd_ep_sid "$NAME")"
  [ -n "$EP" ] || { echo "[$1] bad snippet name (want e.g. c1_001)"; return 1; }
  local OUT="$DRIVE_OUT/$UP"; mkdir -p "$OUT"
  local scene_dir="$SEMGAUSS/data/crcd/$UP"
  local basedir="$SEMGAUSS/data/crcd"
  local local_snip="$CRCD_LOCAL/$NAME"

  # 1) .DONE idempotency gate
  [ -f "$OUT/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { say "$UP done -> skip (FORCE=1 to redo)"; return 0; }
  rm -f "$OUT/.FAILED"

  # 2) resolve the in-domain seg head (WARN + .FAILED + skip if absent)
  local HEAD; HEAD=$(seg_head_for "$UP") || {
    echo "[$UP] no in-domain seg head ($SEG_DIR_DRIVE/loso_v2_max/dinov2_crcd_${UP}.pth) -> SKIP (isolated). Train the fold or set SEG_PTH."
    echo "FAILED no seg head" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  mkdir -p "$SEMGAUSS/seg"
  local HEADABS="$SEMGAUSS/seg/dinov2_crcd_${UP}.pth"       # copy local so a Drive FUSE drop mid-run is safe
  cp -f "$HEAD" "$HEADABS" || { echo "FAILED head copy" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  echo "$HEAD" > "$OUT/seg_head_provenance.txt"; say "$UP seg head: $HEAD -> $HEADABS"

  # 3) depth + stage
  crcd_ensure_depth "$NAME" || { echo "FAILED depth-gen" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  crcd_stage "$NAME"        || { echo "FAILED stage"     > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 4) rectify (preprocess_crcd_published) -> 5) assemble SemGauss tree + emit crcd.yaml
  rm -rf "$scene_dir"
  local staged="$CRCD_STAGED/$UP"; rm -rf "$staged"; mkdir -p "$staged"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/preprocess/preprocess_crcd_published.py" \
     --snippet_dir "$local_snip" --calib_pkl "$CALIB_PKL" --output_dir "$staged" \
     || { echo "FAILED preprocess" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/colab/crcd_assemble_sgs.py" --mode semgauss \
     --staged "$staged" --moge_depth "$local_snip/depth" --calib_pkl "$CALIB_PKL" \
     --out "$scene_dir" --depth_scale "$DEPTH_SCALE" --n_classes 4 \
     --emit_yaml "$SEMGAUSS/configs/data/crcd.yaml" > "$OUT/assemble.log" 2>&1 \
     || { echo "[$UP] assemble FAILED:"; tail -20 "$OUT/assemble.log"; echo "FAILED assemble" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  cat "$OUT/assemble.log"
  grep -q "semantic unique ids" "$OUT/assemble.log" || {
    echo "[$UP] assembler did not print the semantic-unique-ids line -> aborting (assembly incomplete)"
    echo "FAILED assemble (no sem-ids line)" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 6) native res = the assembled frame size (read back from the crcd.yaml we just wrote)
  local H W
  H=$(grep -E '^\s*image_height:' "$SEMGAUSS/configs/data/crcd.yaml" | grep -oE '[0-9]+' | head -1)
  W=$(grep -E '^\s*image_width:'  "$SEMGAUSS/configs/data/crcd.yaml" | grep -oE '[0-9]+' | head -1)
  [ -n "$H" ] && [ -n "$W" ] || { echo "FAILED no H/W from crcd.yaml" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  echo "[$UP] native res H=$H W=$W"

  # 7) PREFLIGHT: dry-load the head into DINO2SEG (SemGauss loads strict=True mid-run -> would hard-crash).
  ( cd "$SEMGAUSS" && PYTHONPATH= "$ENV_PY" - "$HEADABS" "$H" "$W" <<'PY'
import sys, torch
from utils.dinov2_seg import DINO2SEG
head, H, W = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = DINO2SEG(img_h=H, img_w=W, num_cls=4, edge=0, dim=16)
sd = torch.load(head, map_location='cpu')
assert any(k.startswith('backbone.') for k in sd) and any('segmentation_conv' in k for k in sd), \
    'head is not a full DINO2SEG state_dict (missing backbone.* / segmentation_conv.* keys)'
m.load_state_dict(sd)
print(f"[preflight] seg head OK: loads into DINO2SEG(H={H},W={W},cls=4,edge=0,dim=16)")
PY
  ) || { echo "[$UP] PREFLIGHT FAILED: head incompatible with DINO2SEG(H=$H,W=$W,cls=4,edge=0,dim=16) -> SKIP (isolated)"
         echo "FAILED preflight (bad seg head)" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 8) write the experiment config
  local NF=${NUM_FRAMES:--1}
  local cfg="$SEMGAUSS/configs/crcd/bench_${UP}.py"
  semgauss_write_cfg "$UP" "$H" "$W" "$HEADABS" "$basedir" "$cfg" "$NF"
  [ -n "${NUM_FRAMES:-}" ] && echo "[$UP] NUM_FRAMES=$NUM_FRAMES (T4 hardware cap; NOT a full-snippet gate run)"

  # 9) retarget the raw-depth dump patch (idempotent; SemGauss eval_utils.py)
  crcd_patch_raw_depth || { echo "FAILED depth-patch (eval_utils anchors changed)" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 10) background VRAM watcher (SemGauss 3DGS CRCD VRAM only proven on A100)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 30; done \
    > "$OUT/vram_samples.txt" ) & local VPID=$!

  # 11) run SemGauss-SLAM
  say "$UP: SemGauss sem_gauss.py (frames=$(ls "$scene_dir/rgb"/*.png 2>/dev/null | wc -l), num_frames=$NF)"
  ( cd "$SEMGAUSS" && "$ENV_PY" sem_gauss.py "configs/crcd/bench_${UP}.py" ) 2>&1 | tee "$OUT/run.log"
  local RC=${PIPESTATUS[0]}; kill $VPID 2>/dev/null
  sort -rn "$OUT/vram_samples.txt" 2>/dev/null | head -1 | xargs -I{} echo "[peak VRAM] {} MiB" | tee -a "$OUT/run.log"
  [ "$RC" -eq 0 ] || { echo "FAILED sem_gauss.py rc=$RC" > "$OUT/status.txt"; touch "$OUT/.FAILED"; say "$UP FAILED (isolated) -> next"; return 1; }

  # 12) find params.npz (seed=$SEED in the run-dir name). Fallback: periodic params{t}.npz if eval()
  #     crashed AFTER writing a checkpoint (guards the trajectory export).
  local NPZ; NPZ=$(ls -t "$SEMGAUSS"/experiments/*/${UP}_*/params.npz 2>/dev/null | head -1)
  [ -n "$NPZ" ] || NPZ=$(ls -t "$SEMGAUSS"/experiments/*/${UP}_*/params*.npz 2>/dev/null | head -1)
  [ -n "$NPZ" ] && [ -f "$NPZ" ] || { echo "FAILED no params.npz (experiments/*/${UP}_*)" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  local RUNDIR; RUNDIR=$(dirname "$NPZ")
  echo "[$UP] SemGauss run dir=$RUNDIR (npz=$(basename "$NPZ"))"
  local REN_RGB="$RUNDIR/$SEMGAUSS_REN_RGB_SUB"
  local REN_DEPTH_RAW="$RUNDIR/$SEMGAUSS_REN_DEPTH_SUB"
  [ "$(ls "$REN_DEPTH_RAW"/*.png 2>/dev/null | wc -l)" -gt 0 ] || \
     echo "[$UP] WARN $REN_DEPTH_RAW empty -> depth_l1 will be nan (patch (b) fired? eval save_frames on?)"

  # 13) params.npz -> est_c2w_data.txt (SplaTAM serialization; reuse VERBATIM)
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/sgsslam_npz_to_est.py" \
     --npz "$NPZ" --out "$OUT/est_c2w_data.txt" \
     || { echo "FAILED npz->est" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  [ -s "$OUT/est_c2w_data.txt" ] || { echo "FAILED empty est" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 14) render-rename: gs_%04d.png -> $OUT/{idx}.jpg ; GT sibling rgb/rgb_%06d.png -> $OUT/{idx}_gt.png
  PYTHONPATH= "$DDS_PY" - "$REN_RGB" "$scene_dir/rgb" "$OUT" <<'PY'
import sys, os, glob, re, shutil
ren_dir, rgb_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
n = 0
for p in sorted(glob.glob(os.path.join(ren_dir, 'gs_*.png'))):
    m = re.search(r'gs_(\d+)\.png$', os.path.basename(p))
    if not m: continue
    idx = int(m.group(1))
    shutil.copy(p, os.path.join(out, f'{idx}.jpg'))                       # render
    gt = os.path.join(rgb_dir, f'rgb_{idx:06d}.png')
    if os.path.exists(gt):
        shutil.copy(gt, os.path.join(out, f'{idx}_gt.png'))              # GT sibling (Mode-1)
    n += 1
print(f"[rename] {n} renders -> {out}/{{idx}}.jpg (+ {{idx}}_gt.png)")
PY

  # 15) DDS eval battery (reuse verbatim; only paths differ). Clear stale APPEND-mode outputs.
  rm -f "$OUT/sim3_metrics.txt" "$OUT/render_eval.txt" "$OUT/render_eval.csv"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/sim3_ate.py" \
     --est "$OUT/est_c2w_data.txt" --gt "$local_snip/groundtruth.txt" \
     --name "CRCD $UP" --out "$OUT/sim3_metrics.txt" || echo "[$UP] WARN sim3_ate failed"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/eval_rendering.py" \
     --gt_dir "$OUT" --render_dir "$OUT" --sequence "CRCD ($UP)" \
     --output_csv "$OUT/render_eval.csv" --summary_csv "$OUT/render_eval.txt" \
     || echo "[$UP] WARN eval_rendering failed"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/depth_l1.py" \
     --render_depth_dir "$REN_DEPTH_RAW" \
     --input_depth_dir "$scene_dir/depth" \
     --render_scale "$DEPTH_SCALE" --input_scale "$DEPTH_SCALE" --sc_factor 1.0 \
     --out "$OUT/depth_l1.txt" || echo "[$UP] WARN depth_l1 failed"
  local nseg; nseg=$(ls "$scene_dir/semantic_remap"/*.png 2>/dev/null | wc -l)
  echo "[$UP] video inputs: renders=$(ls "$OUT"/*.jpg 2>/dev/null | wc -l) gt=$(ls "$OUT"/*_gt.png 2>/dev/null | wc -l) seg=$nseg depth=$(ls "$scene_dir/depth"/*.png 2>/dev/null | wc -l)"
  [ "$nseg" -gt 0 ] || echo "[$UP] WARN semantic_remap EMPTY ($scene_dir/semantic_remap) -> seg panel will be MISSING from the video"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/viz/generate_video.py" \
     --rgb_input_dir "$OUT" --rgb_output_dir "$OUT" \
     --depth_input_dir "$scene_dir/depth" \
     --depth_output_dir "$REN_DEPTH_RAW" \
     --seg_dir "$scene_dir/semantic_remap" --seg_pattern '*.png' --seg_classmap \
     --trajectory_est "$OUT/est_c2w_data.txt" --trajectory_gt "$local_snip/groundtruth.txt" \
     --png_depth_scale "$DEPTH_SCALE" --output "$OUT/video.mp4" \
     || echo "[$UP] WARN generate_video failed"

  # 16) metrics.json parse + PASS + .DONE (reuse run_sgsslam.sh's parse verbatim)
  PYTHONPATH= "$DDS_PY" - "$OUT" "$UP" <<'PY'
import sys, os, re, json, csv
out, name = sys.argv[1], sys.argv[2]
def _txt(fn):
    p = os.path.join(out, fn)
    return open(p, encoding='utf-8', errors='ignore').read() if os.path.isfile(p) else ''
def grab(s, pat):
    m = re.search(pat, s); return float(m.group(1)) if m else None
sim = _txt('sim3_metrics.txt')
ate_mean = None
m = re.search(r'Sim3 ATE.*?:\s*[0-9.]+\s*/\s*([0-9.]+)\s*/', sim)
if m: ate_mean = float(m.group(1))
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
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"; sync
  say "$UP DONE -> $OUT"
}

# ------------------------------------------------------------- aggregate ----
aggregate(){
  say "=== SemGauss-SLAM CRCD bench summary ($DRIVE_OUT) ==="
  PYTHONPATH= "$DDS_PY" - "$DRIVE_OUT" <<'PY'
import sys, os, json, glob
root = sys.argv[1]
rows = []
for mj in sorted(glob.glob(os.path.join(root, '*', 'metrics.json'))):
    try: rows.append(json.load(open(mj)))
    except Exception: pass
def f(v, p=2):
    return ('%.*f' % (p, v)) if isinstance(v, (int, float)) else '--'
hdr = (f"{'snippet':<9}{'ATEmean':>9}{'ATEmax':>9}{'pathR':>7}{'|Pr|dom':>8}"
       f"{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'DL1mm':>8}")
print(hdr); print('-' * len(hdr))
for d in rows:
    print(f"{str(d.get('scene','?')):<9}{f(d.get('sim3_ate_mean_mm')):>9}{f(d.get('sim3_ate_max_mm')):>9}"
          f"{f(d.get('path_ratio')):>7}{f(d.get('pearson_dom')):>8}{f(d.get('psnr')):>7}"
          f"{f(d.get('ssim'),3):>7}{f(d.get('lpips'),3):>7}{f(d.get('depth_l1_mm')):>8}")
if not rows: print("(no metrics.json yet)")
PY
  local s UP
  for s in $BENCH5; do UP=$(echo "$s" | tr 'a-z' 'A-Z')
    [ -f "$DRIVE_OUT/$UP/.DONE" ] || echo "  $UP: $(cat "$DRIVE_OUT/$UP/status.txt" 2>/dev/null || echo 'not run')"
  done
}

# ------------------------------------------------------------- dispatch -----
case "$VERB" in
  env) build_env ;;
  crcd)
    [ -x "$ENV_PY" ] || { echo "FATAL: sem_gauss env not built ($ENV_PY missing) - run 'env' first"; exit 30; }
    [ -d "$SEMGAUSS/.git" ] || { echo "FATAL: SemGauss clone missing at $SEMGAUSS - run 'env' first"; exit 30; }
    mkdir -p "$DRIVE_OUT"; exec > >(tee -a "$DRIVE_OUT/runbook.log") 2>&1
    SNIPS=${ARG:-bench5}
    { [ "$SNIPS" = bench5 ] || [ "$SNIPS" = all ]; } && SNIPS="$BENCH5"
    say "=== SemGauss CRCD bench  repo=$(cd "$SEMGAUSS" 2>/dev/null && git rev-parse --short HEAD)  snippets='$SNIPS' seed=$SEED depth_scale=$DEPTH_SCALE -> $DRIVE_OUT ==="
    for s in $SNIPS; do run_crcd_one "$s" || echo "[$s] FAILED (isolated) -> next"; done
    aggregate ;;
  eval) aggregate ;;
  *) echo "usage: run_semgauss.sh env | crcd [snippet|bench5] | eval"; exit 2 ;;
esac
