#!/bin/bash
# ============================================================
# DDS-SLAM Colab Setup Script (Conda — exact original env)
#
# Replicates the paper's environment:
#   Python 3.7 | PyTorch 1.10.1+cu113 | tinycudann | pytorch3d
#
# Usage (in Colab terminal):
#   bash Addons/colab_setup.sh [--skip-data] [--skip-tunnel]
# ============================================================
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
CONDA_ENV=ddsslam

SKIP_DATA=false
SKIP_TUNNEL=false
for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-tunnel) SKIP_TUNNEL=true ;;
    esac
done

echo "============================================================"
echo "DDS-SLAM Colab Setup (exact environment reproduction)"
echo "============================================================"

# --- 0. Verify GPU ---
echo ""
echo "[0/6] Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No GPU found! Select a GPU runtime in Colab.'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'  Driver CUDA: {torch.version.cuda}')
"

# --- 1. Install Miniconda ---
echo ""
echo "[1/6] Setting up Miniconda..."
if ! command -v conda &> /dev/null; then
    echo "  Downloading Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm /tmp/miniconda.sh
    echo "  Miniconda installed."
else
    echo "  Conda already available."
fi

# Make conda available in this script
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"

# --- 2. Create conda environment with exact versions ---
echo ""
echo "[2/6] Creating conda environment '$CONDA_ENV' (Python 3.7 + PyTorch 1.10.1+cu113)..."
if conda env list | grep -q "$CONDA_ENV"; then
    echo "  Environment '$CONDA_ENV' already exists, activating..."
else
    conda create -n $CONDA_ENV python=3.7 -y -q
fi
conda activate $CONDA_ENV

echo "  Python: $(python --version)"
echo "  Installing PyTorch 1.10.1+cu113..."
pip install -q torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo "  Installing requirements.txt..."
pip install -q -r "$REPO_ROOT/requirements.txt"

# --- 3. Install tinycudann ---
echo ""
echo "[3/6] Installing tinycudann..."
python -c "import tinycudann" 2>/dev/null && echo "  tinycudann already installed" || {
    echo "  Building tinycudann from source (this takes ~5-10 min)..."
    # Set CUDA arch for T4 (compute capability 7.5)
    export TCNN_CUDA_ARCHITECTURES=75
    pip install ninja
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
}

# --- 4. Install pytorch3d ---
echo ""
echo "[4/6] Installing pytorch3d..."
python -c "import pytorch3d" 2>/dev/null && echo "  pytorch3d already installed" || {
    echo "  Installing pytorch3d (building from source for PyTorch 1.10.1)..."
    conda install -y -q -c fvcore -c iopath -c conda-forge fvcore iopath 2>/dev/null || true
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"
}

# --- 5. Build marching cubes extension ---
echo ""
echo "[5/6] Building marching cubes extension..."
cd "$REPO_ROOT/external/NumpyMarchingCubes"
python setup.py install --quiet
cd "$REPO_ROOT"
echo "  Done."

# --- Verify all imports ---
echo ""
echo "  Verifying all imports..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPU available: {torch.cuda.is_available()}')
import tinycudann as tcnn
print('  tinycudann: OK')
from pytorch3d.transforms import matrix_to_quaternion
print('  pytorch3d: OK')
import marching_cubes
print('  marching_cubes: OK')
from mathutils import Matrix
print('  mathutils: OK')
import cv2, yaml, scipy, trimesh, tqdm
print('  All imports OK!')
"

# --- 6. Download datasets ---
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "[6/6] Downloading datasets..."
    mkdir -p "$DATA_DIR"
    pip install -q gdown

    # Semantic-Super dataset (Google Drive)
    if [ ! -d "$DATA_DIR/Super" ]; then
        echo "  Downloading Semantic-Super dataset..."
        gdown --id 1ZxWw2kNmgeMhBXAGyovL2icXHzn2OCVV -O "$DATA_DIR/Super.zip"
        unzip -q "$DATA_DIR/Super.zip" -d "$DATA_DIR"
        rm -f "$DATA_DIR/Super.zip"
        echo "  Semantic-Super downloaded."
    else
        echo "  Semantic-Super already exists, skipping."
    fi

    # StereoMIS dataset (Zenodo)
    if [ ! -d "$DATA_DIR/P2_1" ]; then
        echo "  Downloading StereoMIS dataset..."
        pip install -q zenodo_get
        mkdir -p "$DATA_DIR/stereomis_download"
        cd "$DATA_DIR/stereomis_download"
        zenodo_get 7727692
        cd "$REPO_ROOT"
        echo "  StereoMIS downloaded to $DATA_DIR/stereomis_download/"
        echo "  >>> You may need to extract and reorganize files into $DATA_DIR/P2_1/ <<<"
        echo "  Expected structure: P2_1/{video_frames/, depth/, masks/, pose/}"
    else
        echo "  StereoMIS P2_1 already exists, skipping."
    fi

    # Check what we have
    echo ""
    echo "  Dataset status:"
    SUPER_RGB=$(find "$DATA_DIR/Super/rgb" -name '*left.png' 2>/dev/null | wc -l)
    SUPER_DEPTH=$(find "$DATA_DIR/Super/rgb" -name '*_depth.npy' 2>/dev/null | wc -l)
    echo "    Super: $SUPER_RGB RGB images, $SUPER_DEPTH depth maps"

    STEREO_RGB=$(find "$DATA_DIR/P2_1/video_frames" -name '*l.png' 2>/dev/null | wc -l)
    STEREO_DEPTH=$(find "$DATA_DIR/P2_1/depth" -name '*.png' 2>/dev/null | wc -l)
    echo "    StereoMIS P2_1: $STEREO_RGB RGB images, $STEREO_DEPTH depth maps"

    if [ "$SUPER_DEPTH" -eq 0 ] && [ "$STEREO_DEPTH" -eq 0 ]; then
        echo ""
        echo "  WARNING: No depth maps found!"
        echo "  You need to run depth estimation before DDS-SLAM can work."
        echo "  - Semantic-Super: https://github.com/ucsdarclab/Python-SuPer"
        echo "  - StereoMIS: https://github.com/aimi-lab/robust-pose-estimator"
    fi
else
    echo ""
    echo "[6/6] Skipping dataset download (--skip-data)"
fi

# --- VS Code tunnel ---
if [ "$SKIP_TUNNEL" = false ]; then
    if ! command -v code &> /dev/null; then
        echo ""
        echo "Installing VS Code CLI..."
        curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" -o /tmp/vscode_cli.tar.gz
        tar -xzf /tmp/vscode_cli.tar.gz -C /usr/local/bin
        rm /tmp/vscode_cli.tar.gz
    fi
fi

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "IMPORTANT: Always activate the conda env before running:"
echo "  eval \"\$(/opt/miniconda3/bin/conda shell.bash hook)\""
echo "  conda activate $CONDA_ENV"
echo ""
echo "Then run DDS-SLAM:"
echo "  cd $REPO_ROOT"
echo "  python ddsslam.py --config ./configs/Super/trail3.yaml"
echo ""
echo "Available configs:"
echo "  ./configs/Super/trail3.yaml   (151 frames, fastest)"
echo "  ./configs/Super/trail4.yaml   (151 frames)"
echo "  ./configs/Super/trail8.yaml   (151 frames)"
echo "  ./configs/Super/trail9.yaml   (151 frames)"
echo "  ./configs/StereoMIS/p2_1.yaml (4000 frames)"
echo "  ./configs/StereoMIS/p2_0.yaml (13825 frames)"
echo "============================================================"

if [ "$SKIP_TUNNEL" = false ]; then
    echo ""
    echo "Starting VS Code tunnel..."
    code tunnel
fi
