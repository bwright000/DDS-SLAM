#!/bin/bash
# Run DDS-SLAM in Docker with the exact paper environment
# (Python 3.7, PyTorch 1.10.1+cu113, TCNN compiled against CUDA 11.3)
#
# Usage on Colab:
#   cd /root/DDS-SLAM
#   bash Addons/run_docker.sh build          # Build Docker image (~20-30 min first time)
#   bash Addons/run_docker.sh semsup         # Run Semantic-SuPer trail3
#   bash Addons/run_docker.sh stereomis      # Run StereoMIS P2_1
#   bash Addons/run_docker.sh shell          # Interactive shell inside container

set -e

IMAGE_NAME="ddsslam-original"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Detect data directories
DATA_SUPER="${REPO_DIR}/data/Super"
DATA_P2_1="${REPO_DIR}/data/P2_1"
DRIVE_DIR="/content/drive"

build() {
    echo "Building Docker image: ${IMAGE_NAME}"
    echo "This will take ~20-30 minutes (compiling TCNN + pytorch3d from source)..."
    docker build -t "${IMAGE_NAME}" -f "${REPO_DIR}/Addons/Dockerfile" "${REPO_DIR}"
    echo "Build complete!"
}

run_container() {
    # Build marching cubes inside container, then run the command
    local CMD="$1"

    docker run --rm --gpus all \
        -v "${REPO_DIR}:/workspace" \
        -v "${DRIVE_DIR}:/drive" 2>/dev/null \
        "${IMAGE_NAME}" \
        bash -c "
            cd /workspace/external/NumpyMarchingCubes && \
            python3 setup.py install 2>&1 | tail -1 && \
            cd /workspace && \
            echo '=== Environment ===' && \
            python3 -c \"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
import tinycudann; print('tinycudann: OK')
from pytorch3d.transforms import matrix_to_quaternion; print('pytorch3d: OK')
import marching_cubes; print('marching_cubes: OK')
\" && \
            echo '==================' && \
            ${CMD}
        "
}

case "${1}" in
    build)
        build
        ;;

    semsup)
        echo "Running Semantic-SuPer trail3 in Docker (paper environment)..."

        # Check data exists
        if [ ! -d "${DATA_SUPER}/rgb" ]; then
            echo "ERROR: Data not found at ${DATA_SUPER}/rgb"
            echo "Symlink or copy trial_3 data to ${DATA_SUPER}"
            exit 1
        fi

        # Check depth maps exist
        DEPTH_COUNT=$(ls "${DATA_SUPER}/rgb/"*_depth.npy 2>/dev/null | wc -l)
        if [ "${DEPTH_COUNT}" -eq 0 ]; then
            echo "WARNING: No depth maps found. Generate them first outside Docker:"
            echo "  python3 Addons/generate_depth.py --datadir data/Super --method depth_anything"
            exit 1
        fi
        echo "Found ${DEPTH_COUNT} depth maps"

        run_container "python3 ddsslam.py --config ./configs/Super/trail3.yaml"
        ;;

    stereomis)
        echo "Running StereoMIS P2_1 in Docker (paper environment)..."

        if [ ! -d "${DATA_P2_1}/video_frames" ]; then
            echo "ERROR: Data not found at ${DATA_P2_1}/video_frames"
            exit 1
        fi

        DEPTH_COUNT=$(ls "${DATA_P2_1}/depth/"*.png 2>/dev/null | wc -l)
        if [ "${DEPTH_COUNT}" -eq 0 ]; then
            echo "WARNING: No depth maps found in ${DATA_P2_1}/depth/"
            exit 1
        fi
        echo "Found ${DEPTH_COUNT} depth maps"

        run_container "python3 ddsslam.py --config ./configs/StereoMIS/p2_1.yaml"
        ;;

    shell)
        echo "Starting interactive shell in Docker container..."
        docker run --rm -it --gpus all \
            -v "${REPO_DIR}:/workspace" \
            -v "${DRIVE_DIR}:/drive" 2>/dev/null \
            "${IMAGE_NAME}" \
            bash -c "
                cd /workspace/external/NumpyMarchingCubes && \
                python3 setup.py install 2>&1 | tail -1 && \
                cd /workspace && \
                exec bash
            "
        ;;

    verify)
        echo "Verifying Docker environment..."
        run_container "python3 -c \"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')

# Check TCNN
import tinycudann as tcnn
m = tcnn.Encoding(3, {'otype':'HashGrid','n_levels':1,'n_features_per_level':2,'log2_hashmap_size':4,'base_resolution':4,'per_level_scale':1.0})
print(f'TCNN encoding params dtype: {m.params.dtype}')

# Check all imports work
from pytorch3d.transforms import matrix_to_quaternion
import marching_cubes
import mathutils
print('All imports OK!')
\""
        ;;

    *)
        echo "Usage: bash Addons/run_docker.sh {build|semsup|stereomis|shell|verify}"
        echo ""
        echo "Commands:"
        echo "  build      Build Docker image with paper's exact environment"
        echo "  semsup     Run DDS-SLAM on Semantic-SuPer trail3"
        echo "  stereomis  Run DDS-SLAM on StereoMIS P2_1"
        echo "  shell      Interactive shell inside container"
        echo "  verify     Verify environment (versions, imports)"
        ;;
esac
