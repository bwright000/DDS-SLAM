#!/bin/bash
# ============================================================
# DDS-SLAM Colab Setup Script
# Sets up environment, downloads data, starts VS Code tunnel
#
# Usage (in Colab terminal):
#   bash colab_setup.sh [--skip-data] [--skip-tunnel]
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DDS_SLAM_DIR="$SCRIPT_DIR/DDS-SLAM"
DATA_DIR="$DDS_SLAM_DIR/data"

SKIP_DATA=false
SKIP_TUNNEL=false
for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --skip-tunnel) SKIP_TUNNEL=true ;;
    esac
done

echo "============================================================"
echo "DDS-SLAM Colab Setup"
echo "============================================================"

# --- 0. Verify GPU ---
echo ""
echo "[0/5] Verifying GPU..."
python3 -c "
import torch
assert torch.cuda.is_available(), 'No GPU found! Select a GPU runtime in Colab.'
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
"

# --- 1. Install Python dependencies ---
echo ""
echo "[1/5] Installing Python dependencies..."
pip install -q PyYAML scipy trimesh matplotlib opencv-contrib-python tqdm yacs mathutils Cython ninja gdown

# Install tinycudann
python3 -c "import tinycudann" 2>/dev/null && echo "  tinycudann already installed" || {
    echo "  Building tinycudann from source (this takes ~5 min)..."
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
}

# Install pytorch3d
python3 -c "import pytorch3d" 2>/dev/null && echo "  pytorch3d already installed" || {
    echo "  Installing pytorch3d..."
    PYT_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0].replace('.',''))")
    PY_VERSION=$(python3 -c "import sys; print(f'py3{sys.version_info.minor}')")
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda.replace('.',''))")
    WHEEL_TAG="${PY_VERSION}_cu${CUDA_VERSION}_pyt${PYT_VERSION}"
    echo "  Trying prebuilt wheel: $WHEEL_TAG"
    pip install --no-index --no-cache-dir pytorch3d \
        -f "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/${WHEEL_TAG}/download.html" 2>/dev/null || {
        echo "  Prebuilt wheel not found, building from source..."
        pip install git+https://github.com/facebookresearch/pytorch3d.git
    }
}

# --- 2. Build marching cubes extension ---
echo ""
echo "[2/5] Building marching cubes extension..."
cd "$DDS_SLAM_DIR/external/NumpyMarchingCubes"
python3 setup.py install --quiet
cd "$SCRIPT_DIR"
echo "  Done."

# --- 3. Verify all imports ---
echo ""
echo "[3/5] Verifying imports..."
cd "$DDS_SLAM_DIR"
python3 -c "
import tinycudann as tcnn
from pytorch3d.transforms import matrix_to_quaternion
import marching_cubes
import cv2, yaml, scipy, trimesh, tqdm
print('  All imports OK!')
"
cd "$SCRIPT_DIR"

# --- 4. Download datasets ---
if [ "$SKIP_DATA" = false ]; then
    echo ""
    echo "[4/5] Downloading datasets..."
    mkdir -p "$DATA_DIR"

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
        cd "$SCRIPT_DIR"
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
    echo "[4/5] Skipping dataset download (--skip-data)"
fi

# --- 5. VS Code tunnel ---
if [ "$SKIP_TUNNEL" = false ]; then
    echo ""
    echo "[5/5] Setting up VS Code tunnel..."

    if ! command -v code &> /dev/null; then
        echo "  Installing VS Code CLI..."
        curl -fsSL "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" -o /tmp/vscode_cli.tar.gz
        tar -xzf /tmp/vscode_cli.tar.gz -C /usr/local/bin
        rm /tmp/vscode_cli.tar.gz
    fi

    echo ""
    echo "============================================================"
    echo "Setup complete! Starting VS Code tunnel..."
    echo ""
    echo "To run DDS-SLAM after connecting:"
    echo "  cd $DDS_SLAM_DIR"
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
    echo ""

    code tunnel
else
    echo ""
    echo "[5/5] Skipping VS Code tunnel (--skip-tunnel)"
    echo ""
    echo "============================================================"
    echo "Setup complete!"
    echo ""
    echo "To run DDS-SLAM:"
    echo "  cd $DDS_SLAM_DIR"
    echo "  python ddsslam.py --config ./configs/Super/trail3.yaml"
    echo "============================================================"
fi
