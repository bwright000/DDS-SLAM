"""
Depth map generation for DDS-SLAM Semantic-Super dataset.

Supports 3 methods:
  - depth_anything : Depth Anything V2 Metric Indoor (monocular, HuggingFace)
  - monodepth2     : Monodepth2 mono+stereo (monocular, auto-download weights)
  - raft_stereo    : RAFT-Stereo (stereo pairs, Dropbox weights)

Output: {datadir}/rgb/{frame}-left_depth.npy (480x640 float32)
Values are pre-multiplied by depth_scale (default 8.0) so DDS-SLAM loader
divides by png_depth_scale to get meters.

Usage:
  python Addons/generate_depth.py --datadir data/Super --method depth_anything
  python Addons/generate_depth.py --datadir data/Super --method monodepth2
  python Addons/generate_depth.py --datadir data/Super --method raft_stereo --baseline 0.0055
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm


def get_left_images(datadir):
    """Find all left RGB images sorted by name."""
    files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*-left.png')))
    if not files:
        # Try underscore variant
        files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*_left.png')))
    if not files:
        raise FileNotFoundError(f"No left images found in {datadir}/rgb/")
    print(f"Found {len(files)} left images")
    return files


def save_depth(left_path, depth, depth_scale):
    """Save depth map as .npy alongside the RGB image."""
    base = left_path.replace('-left.png', '-left_depth.npy').replace('_left.png', '_left_depth.npy')
    depth_scaled = (depth * depth_scale).astype(np.float32)
    np.save(base, depth_scaled)
    return base


# =============================================================================
# Method 1: Depth Anything V2 (Metric Indoor)
# =============================================================================
def run_depth_anything(datadir, depth_scale, target_h=480, target_w=640):
    """Generate depth using Depth Anything V2 Metric Indoor model."""
    from transformers import pipeline

    print("Loading Depth Anything V2 Metric Indoor model...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        device=0 if torch.cuda.is_available() else -1,
    )

    left_images = get_left_images(datadir)
    for img_path in tqdm(left_images, desc="Depth Anything V2"):
        from PIL import Image
        image = Image.open(img_path).convert("RGB")

        result = pipe(image)
        depth_pil = result["depth"]
        # Convert PIL depth to numpy array
        depth = np.array(result["predicted_depth"].squeeze().cpu().numpy(), dtype=np.float32)

        # Resize to target resolution
        if depth.shape != (target_h, target_w):
            depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        save_depth(img_path, depth, depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps.")


# =============================================================================
# Method 2: Monodepth2
# =============================================================================
def setup_monodepth2():
    """Clone monodepth2 repo and return path."""
    repo_dir = '/tmp/monodepth2'
    if not os.path.exists(repo_dir):
        print("Cloning monodepth2...")
        os.system(f'git clone --depth 1 https://github.com/nianticlabs/monodepth2.git {repo_dir}')
    return repo_dir


def run_monodepth2(datadir, depth_scale, target_h=480, target_w=640):
    """Generate depth using Monodepth2 mono+stereo pretrained model."""
    repo_dir = setup_monodepth2()

    # Save cwd and change to repo dir (monodepth2 expects to run from its root)
    original_cwd = os.getcwd()
    os.chdir(repo_dir)
    sys.path.insert(0, repo_dir)

    import networks
    # monodepth2's utils.py has download_model_if_doesnt_exist
    from utils import download_model_if_doesnt_exist

    model_name = "mono+stereo_640x192"
    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)

    print(f"Loading Monodepth2 model: {model_name}")
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(
        os.path.join(model_path, "encoder.pth"), map_location="cpu", weights_only=False
    )
    filtered_dict_enc = {
        k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(
        os.path.join(model_path, "depth.pth"), map_location="cpu", weights_only=False
    )
    depth_decoder.load_state_dict(loaded_dict)

    encoder.eval()
    depth_decoder.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    depth_decoder.to(device)

    # Model input size
    feed_height = loaded_dict_enc.get("height", 192)
    feed_width = loaded_dict_enc.get("width", 640)

    # Switch back to original dir for file paths
    os.chdir(original_cwd)

    left_images = get_left_images(datadir)
    with torch.no_grad():
        for img_path in tqdm(left_images, desc="Monodepth2"):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = image.shape[:2]

            # Resize for model
            input_image = cv2.resize(image, (feed_width, feed_height))
            input_image = input_image.astype(np.float32) / 255.0
            input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).to(device)

            features = encoder(input_tensor)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_np = disp.squeeze().cpu().numpy()

            # Resize disparity to original resolution
            disp_resized = cv2.resize(disp_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

            # Convert disparity to depth
            min_disp = 1e-3
            disp_resized = np.maximum(disp_resized, min_disp)
            depth = 1.0 / disp_resized

            # Resize to target
            if depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            save_depth(img_path, depth, depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps.")
    sys.path.remove(repo_dir)


# =============================================================================
# Method 3: RAFT-Stereo
# =============================================================================
def setup_raft_stereo():
    """Clone RAFT-Stereo repo and download weights."""
    repo_dir = '/tmp/RAFT-Stereo'
    if not os.path.exists(repo_dir):
        print("Cloning RAFT-Stereo...")
        os.system(f'git clone --depth 1 https://github.com/princeton-vl/RAFT-Stereo.git {repo_dir}')

    weights_dir = os.path.join(repo_dir, 'models')
    if not os.path.exists(weights_dir) or not os.listdir(weights_dir):
        print("Downloading RAFT-Stereo weights from Dropbox...")
        os.makedirs(weights_dir, exist_ok=True)
        zip_path = os.path.join(repo_dir, 'models.zip')
        os.system(f'wget -q "https://www.dropbox.com/s/ftveifyqcomiwaq/models.zip" -O {zip_path}')
        os.system(f'unzip -q -o {zip_path} -d {repo_dir}')
        os.remove(zip_path)

    return repo_dir


def run_raft_stereo(datadir, depth_scale, baseline=0.0055, fx=768.99,
                    target_h=480, target_w=640):
    """Generate depth using RAFT-Stereo with stereo image pairs."""
    repo_dir = setup_raft_stereo()
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, os.path.join(repo_dir, 'core'))

    from raft_stereo import RAFTStereo
    from utils.utils import InputPadder

    # Load model
    class Args:
        restore_ckpt = os.path.join(repo_dir, 'models', 'raftstereo-middlebury.pth')
        shared_backbone = False
        n_downsample = 2
        n_gru_layers = 3
        slow_fast_gru = False
        valid_iters = 32
        hidden_dims = [128] * 3
        corr_implementation = 'reg'
        corr_levels = 4
        corr_radius = 4
        mixed_precision = True
        context_norm = 'batch'

    args = Args()
    print(f"Loading RAFT-Stereo model: {args.restore_ckpt}")
    model = torch.nn.DataParallel(RAFTStereo(args))
    state_dict = torch.load(args.restore_ckpt, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.module
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    left_images = get_left_images(datadir)
    with torch.no_grad():
        for left_path in tqdm(left_images, desc="RAFT-Stereo"):
            # Find matching right image
            right_path = left_path.replace('-left.png', '-right.png').replace('_left.png', '_right.png')
            if not os.path.exists(right_path):
                print(f"  Warning: no right image for {os.path.basename(left_path)}, skipping")
                continue

            # Load images
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

            left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float().unsqueeze(0).to(device)
            right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float().unsqueeze(0).to(device)

            # Pad to multiple of 8
            padder = InputPadder(left_tensor.shape, divis_by=32)
            left_tensor, right_tensor = padder.pad(left_tensor, right_tensor)

            # Run model
            _, flow_up = model(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
            # RAFT-Stereo outputs negative disparity
            disp = -flow_up.squeeze().cpu().numpy()

            # Unpad
            disp = padder.unpad(torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)).squeeze().numpy()

            # Convert disparity to depth: Z = fx * baseline / disparity
            min_disp = 0.1  # Avoid division by zero
            disp = np.maximum(disp, min_disp)
            depth = (fx * baseline) / disp

            # Resize to target
            if depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            # Clamp to valid range (0, depth_trunc=5m)
            depth = np.clip(depth, 0.0, 5.0)

            save_depth(left_path, depth, depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps.")
    sys.path.remove(repo_dir)
    sys.path.remove(os.path.join(repo_dir, 'core'))

# =============================================================================
# Method 4: MoGe
# =============================================================================
def run_moge(datadir, depth_scale, target_h=480, target_w=640):
    """Generate depth using MoGe-2 (Monocular Geometry Estimation v2, metric)."""
    from moge.model.v2 import MoGeModel

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading MoGe-2 metric depth model...")
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)
    model.eval()

    left_images = get_left_images(datadir)
    with torch.no_grad():
        for img_path in tqdm(left_images, desc="MoGe-2"):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            input_tensor = torch.tensor(
                image / 255.0, dtype=torch.float32, device=device
            ).permute(2, 0, 1)  # (3, H, W)

            output = model.infer(input_tensor, resolution_level=9)
            depth = output["depth"].cpu().numpy().astype(np.float32)

            # Replace invalid pixels (inf) with zero
            depth = np.where(np.isfinite(depth), depth, 0.0)

            # Clamp to valid range
            depth = np.clip(depth, 0.0, 5.0)

            # Resize to target resolution
            if depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            save_depth(img_path, depth, depth_scale)

    print(f"Done. Generated {len(left_images)} depth maps.")


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate depth maps for DDS-SLAM')
    parser.add_argument('--datadir', type=str, required=True,
                        help='Path to dataset (e.g., data/Super)')
    parser.add_argument('--method', type=str, required=True,
                        choices=['depth_anything', 'monodepth2', 'raft_stereo', 'moge'],
                        help='Depth estimation method')
    parser.add_argument('--depth_scale', type=float, default=8.0,
                        help='Depth scale factor (default: 8.0, matches png_depth_scale in config)')
    parser.add_argument('--baseline', type=float, default=0.0055,
                        help='Stereo baseline in meters (default: 0.0055m for da Vinci)')
    parser.add_argument('--fx', type=float, default=768.99,
                        help='Focal length in pixels (default: 768.99 from Super config)')
    args = parser.parse_args()

    print(f"Method: {args.method}")
    print(f"Data dir: {args.datadir}")
    print(f"Depth scale: {args.depth_scale}")

    if args.method == 'depth_anything':
        run_depth_anything(args.datadir, args.depth_scale)
    elif args.method == 'monodepth2':
        run_monodepth2(args.datadir, args.depth_scale)
    elif args.method == 'raft_stereo':
        print(f"Baseline: {args.baseline}m, fx: {args.fx}px")
        run_raft_stereo(args.datadir, args.depth_scale,
                        baseline=args.baseline, fx=args.fx)
    elif args.method == 'moge':
        run_moge(args.datadir, args.depth_scale)


if __name__ == '__main__':
    main()
