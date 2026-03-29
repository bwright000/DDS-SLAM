FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TCNN_CUDA_ARCHITECTURES=75

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.7 python3.7-dev python3-pip python3.7-distutils \
    git cmake build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyTorch 1.10.1+cu113 (exact paper version)
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html

# tinycudann (compiled against CUDA 11.3 + PyTorch 1.10)
RUN pip install ninja
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# pytorch3d
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.2"

# Marching cubes build deps
RUN pip install Cython numpy==1.21.6

# Other dependencies (from requirements.txt)
RUN pip install \
    scipy \
    opencv-contrib-python==4.6.0.66 \
    trimesh \
    matplotlib==3.5.3 \
    mathutils==2.81.2 \
    PyYAML \
    tqdm \
    colorama==0.4.6 \
    fvcore==0.1.5.post20210915 \
    iopath==0.1.9

WORKDIR /workspace

# Build marching cubes at container startup (needs the repo mounted)
CMD ["bash", "-c", "\
    cd /workspace/external/NumpyMarchingCubes && \
    python3 setup.py install 2>/dev/null && \
    cd /workspace && \
    echo '=== Environment ===' && \
    python3 -c \"import torch; print(f'PyTorch {torch.__version__}'); import tinycudann; print('TCNN OK'); from pytorch3d.transforms import matrix_to_quaternion; print('pytorch3d OK'); import marching_cubes; print('marching_cubes OK')\" && \
    echo '=== Ready ===' && \
    exec bash"]
