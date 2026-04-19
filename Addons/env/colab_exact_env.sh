#!/bin/bash
# ============================================================
# DDS-SLAM Colab Setup — EXACT Paper Environment
#
# Installs Python 3.7 + PyTorch 1.10.1+cu113 + CUDA 11.3 +
# tinycudann v1.5 + PyTorch3D 0.7.2 on Google Colab.
#
# First run: ~20 min (downloads + compilation)
# Cached run: ~5 min (restores venv from Google Drive)
#
# Usage (in Colab terminal or notebook cell):
#   bash Addons/env/colab_exact_env.sh [--skip-data] [--skip-cache] [--no-drive]
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CACHE_DIR="/content/drive/MyDrive/dds_cache"
VENV_DIR="/tmp/dds_env"
CUDA_INSTALL_DIR="/usr/local/cuda-11.3"
CUDA_RUNFILE="cuda_11.3.1_465.19.01_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/${CUDA_RUNFILE}"

SKIP_DATA=false
SKIP_CACHE=false
NO_DRIVE=false
for arg in "$@"; do
    case $arg in
        --skip-data)  SKIP_DATA=true ;;
        --skip-cache) SKIP_CACHE=true ;;
        --no-drive)   NO_DRIVE=true ;;
    esac
done

echo "============================================================"
echo "DDS-SLAM Colab Setup — EXACT Paper Environment"
echo "  Python 3.7 | PyTorch 1.10.1+cu113 | CUDA 11.3"
echo "============================================================"

# Record OS version for cache validation
OS_VERSION=$(lsb_release -rs 2>/dev/null || echo "unknown")
echo "  Colab OS: Ubuntu $OS_VERSION"

# --- 0. Verify GPU ---
echo ""
echo "[0/8] Verifying GPU..."
python3 -c "
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
print('  ' + result.stdout.strip())
"

# ============================================================
# Phase A: System Setup
# ============================================================

# --- 1. Install GCC 10 (CUDA 11.3 nvcc rejects GCC > 10) ---
echo ""
echo "[1/8] Installing GCC 10..."
if dpkg -l gcc-10 &>/dev/null; then
    echo "  GCC 10 already installed"
else
    sudo apt-get update -qq
    sudo apt-get install -y -qq gcc-10 g++-10
    echo "  GCC 10 installed"
fi
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10
echo "  CC=$CC  CXX=$CXX  CUDAHOSTCXX=$CUDAHOSTCXX"

# --- 2. Install Python 3.7 ---
echo ""
echo "[2/8] Installing Python 3.7..."
if command -v python3.7 &>/dev/null; then
    echo "  Python 3.7 already installed: $(python3.7 --version)"
else
    # Try deadsnakes PPA first
    if sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null && \
       sudo apt-get update -qq && \
       sudo apt-get install -y -qq python3.7 python3.7-dev python3.7-distutils python3.7-venv 2>/dev/null; then
        echo "  Installed Python 3.7 via deadsnakes PPA"
    else
        # Fallback: miniconda
        echo "  deadsnakes failed, falling back to miniconda..."
        if [ ! -f /tmp/miniconda/bin/conda ]; then
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
            bash /tmp/miniconda.sh -b -p /tmp/miniconda
        fi
        /tmp/miniconda/bin/conda create -y -n dds37 python=3.7 2>/dev/null
        # Create a symlink so python3.7 is available
        ln -sf /tmp/miniconda/envs/dds37/bin/python3.7 /usr/local/bin/python3.7
        echo "  Installed Python 3.7 via miniconda"
    fi
fi
echo "  $(python3.7 --version)"

# --- 3. Install CUDA 11.3.1 toolkit ---
echo ""
echo "[3/8] Installing CUDA 11.3.1 toolkit..."
if [ -f "$CUDA_INSTALL_DIR/bin/nvcc" ]; then
    echo "  CUDA 11.3 already installed"
else
    # Check for cached runfile on Google Drive
    RUNFILE_PATH="/tmp/${CUDA_RUNFILE}"
    if [ "$NO_DRIVE" = false ] && [ -f "$CACHE_DIR/${CUDA_RUNFILE}" ]; then
        echo "  Copying CUDA runfile from Drive cache..."
        cp "$CACHE_DIR/${CUDA_RUNFILE}" "$RUNFILE_PATH"
    elif [ ! -f "$RUNFILE_PATH" ]; then
        echo "  Downloading CUDA 11.3.1 (~2.6GB)..."
        wget -q --show-progress "$CUDA_URL" -O "$RUNFILE_PATH"
        # Cache the runfile to Drive for next session
        if [ "$NO_DRIVE" = false ] && [ -d "/content/drive/MyDrive" ]; then
            mkdir -p "$CACHE_DIR"
            echo "  Caching CUDA runfile to Google Drive..."
            cp "$RUNFILE_PATH" "$CACHE_DIR/${CUDA_RUNFILE}"
        fi
    fi
    echo "  Installing toolkit (no driver)..."
    sudo sh "$RUNFILE_PATH" --silent --toolkit --toolkitpath="$CUDA_INSTALL_DIR" --no-opengl-libs --override
    echo "  CUDA 11.3.1 installed to $CUDA_INSTALL_DIR"
fi

# --- 4. Set CUDA environment ---
export CUDA_HOME="$CUDA_INSTALL_DIR"
export PATH="$CUDA_INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_INSTALL_DIR/lib64:${LD_LIBRARY_PATH:-}"
echo "  nvcc: $($CUDA_INSTALL_DIR/bin/nvcc --version | grep release)"

# ============================================================
# Phase B: Python Environment
# ============================================================

# --- 5. Check for cached venv ---
CACHE_RESTORED=false
if [ "$SKIP_CACHE" = false ] && [ "$NO_DRIVE" = false ] && [ -f "$CACHE_DIR/dds_env.tar.gz" ]; then
    echo ""
    echo "[5/8] Restoring cached Python environment from Google Drive..."
    # Validate cache was built on same OS
    CACHED_OS=$(cat "$CACHE_DIR/dds_env_os.txt" 2>/dev/null || echo "none")
    if [ "$CACHED_OS" = "$OS_VERSION" ]; then
        tar xzf "$CACHE_DIR/dds_env.tar.gz" -C /tmp
        CACHE_RESTORED=true
        echo "  Restored venv from cache (OS match: $OS_VERSION)"
    else
        echo "  Cache OS mismatch ($CACHED_OS vs $OS_VERSION), rebuilding..."
    fi
else
    echo ""
    echo "[5/8] No cache found, building from scratch..."
fi

if [ "$CACHE_RESTORED" = false ]; then
    # --- Create venv ---
    echo "  Creating Python 3.7 virtual environment..."
    python3.7 -m pip install --upgrade pip setuptools wheel 2>/dev/null || \
        wget -q https://bootstrap.pypa.io/pip/3.7/get-pip.py -O /tmp/get-pip.py && python3.7 /tmp/get-pip.py
    python3.7 -m venv "$VENV_DIR" || python3.7 -m virtualenv "$VENV_DIR" 2>/dev/null
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel

    # --- 6. Install PyTorch 1.10.1+cu113 ---
    echo ""
    echo "[6/8] Installing PyTorch 1.10.1+cu113..."
    pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
        -f https://download.pytorch.org/whl/cu113/torch_stable.html
    python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"

    # --- 7. Install tinycudann ---
    echo ""
    echo "[7/8] Building tinycudann + pytorch3d + deps (~15 min)..."
    pip install ninja
    export TCNN_CUDA_ARCHITECTURES=75  # T4 = compute capability 7.5
    # Pin to commit known to work with PyTorch 1.10 + CUDA 11.3
    pip install "git+https://github.com/NVlabs/tiny-cuda-nn/@91ee479d275d322a65726435040fc20b56b9c991#subdirectory=bindings/torch"
    python -c "import tinycudann; print('  tinycudann: OK')"

    # --- Install pytorch3d v0.7.2 ---
    echo "  Building pytorch3d v0.7.2 (~10 min)..."
    pip install fvcore==0.1.5.post20210915 iopath==0.1.9
    export CUB_HOME="$CUDA_INSTALL_DIR/include"
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
    python -c "from pytorch3d.transforms import matrix_to_quaternion; print('  pytorch3d: OK')"

    # --- Install remaining requirements (exact versions from requirements.txt) ---
    echo "  Installing pip requirements..."
    pip install Cython numpy==1.21.6
    pip install -r "$REPO_ROOT/requirements.txt"

    # --- Build marching cubes ---
    echo "  Building marching cubes C++ extension..."
    cd "$REPO_ROOT/external/NumpyMarchingCubes"
    rm -f marching_cubes/src/_mcubes.cpp  # Force Cython regeneration
    python setup.py install 2>&1 | tail -3
    cd "$REPO_ROOT"

    # --- Cache the venv to Google Drive ---
    if [ "$NO_DRIVE" = false ] && [ -d "/content/drive/MyDrive" ]; then
        echo "  Caching venv to Google Drive (~1 min)..."
        mkdir -p "$CACHE_DIR"
        tar czf "$CACHE_DIR/dds_env.tar.gz" -C /tmp dds_env
        echo "$OS_VERSION" > "$CACHE_DIR/dds_env_os.txt"
        echo "  Cache saved."
    fi
else
    source "$VENV_DIR/bin/activate"
    echo "[6-7/8] Skipped (using cached environment)"
fi

# ============================================================
# Phase C: Verification
# ============================================================
echo ""
echo "[8/8] Verifying environment..."
python -c "
import sys
print(f'  Python:       {sys.version}')

import torch
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA (torch): {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0)}')
print(f'  GPU avail:    {torch.cuda.is_available()}')

import numpy as np
print(f'  NumPy:        {np.__version__}')

import tinycudann as tcnn
print(f'  tinycudann:   OK')

from pytorch3d.transforms import matrix_to_quaternion
print(f'  pytorch3d:    OK')

import marching_cubes
print(f'  marching_cubes: OK')

import mathutils
print(f'  mathutils:    OK')

import cv2, yaml, trimesh, tqdm, scipy
print(f'  Other deps:   OK')

# Quick TCNN functional test
e = tcnn.Encoding(3, {
    'otype': 'HashGrid', 'n_levels': 1, 'n_features_per_level': 2,
    'log2_hashmap_size': 4, 'base_resolution': 4, 'per_level_scale': 1.0
})
x = torch.rand(1, 3, device='cuda')
y = e(x)
print(f'  TCNN test:    OK (output shape {y.shape})')

print()
print('  All checks passed!')
"

# ============================================================
# Phase D: Dataset Download (optional)
# ============================================================
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "Downloading datasets..."
    pip install -q gdown 2>/dev/null
    mkdir -p "$DATA_DIR"

    # Semantic-Super dataset
    if [ ! -d "$DATA_DIR/Super" ]; then
        echo "  Downloading Semantic-Super dataset..."
        gdown --id 1ZxWw2kNmgeMhBXAGyovL2icXHzn2OCVV -O "$DATA_DIR/Super.zip" && \
            unzip -q "$DATA_DIR/Super.zip" -d "$DATA_DIR" && \
            rm -f "$DATA_DIR/Super.zip" && \
            echo "  Semantic-Super downloaded." || \
            echo "  WARNING: Semantic-Super download failed (quota/permissions)"
    else
        echo "  Semantic-Super already exists, skipping."
    fi

    # StereoMIS dataset
    if [ ! -d "$DATA_DIR/P2_1" ]; then
        echo "  Downloading StereoMIS dataset..."
        pip install -q zenodo_get 2>/dev/null
        mkdir -p "$DATA_DIR/stereomis_download"
        cd "$DATA_DIR/stereomis_download"
        zenodo_get 7727692 2>/dev/null && \
            echo "  StereoMIS downloaded to $DATA_DIR/stereomis_download/" || \
            echo "  WARNING: StereoMIS download failed"
        cd "$REPO_ROOT"
    else
        echo "  StereoMIS P2_1 already exists, skipping."
    fi
else
    echo ""
    echo "Skipping dataset download (--skip-data)"
fi

# ============================================================
# Done
# ============================================================
echo ""
echo "============================================================"
echo "Setup complete! Exact paper environment ready."
echo ""
echo "IMPORTANT: Activate the environment before running:"
echo "  export CUDA_HOME=$CUDA_INSTALL_DIR"
echo "  export PATH=$CUDA_INSTALL_DIR/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=$CUDA_INSTALL_DIR/lib64:\$LD_LIBRARY_PATH"
echo "  export CC=/usr/bin/gcc-10 CXX=/usr/bin/g++-10 CUDAHOSTCXX=/usr/bin/g++-10"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run DDS-SLAM:"
echo "  cd $REPO_ROOT"
echo "  python ddsslam.py --config ./configs/Super/trail3.yaml"
echo ""
echo "Available configs:"
echo "  ./configs/Super/trail3.yaml   (151 frames, fastest)"
echo "  ./configs/Super/trail4.yaml"
echo "  ./configs/Super/trail8.yaml"
echo "  ./configs/Super/trail9.yaml"
echo "  ./configs/StereoMIS/p2_1.yaml (4000 frames)"
echo "  ./configs/StereoMIS/p2_0.yaml (13825 frames)"
echo "============================================================"
