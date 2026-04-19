"""
Depth map generation for DDS-SLAM datasets (Semantic-Super and StereoMIS).

Supports 4 methods:
  - depth_anything : Depth Anything V2 Metric Indoor (monocular, HuggingFace)
  - monodepth2     : Monodepth2 mono+stereo (monocular, auto-download weights)
  - raft_stereo    : RAFT-Stereo (stereo pairs, Dropbox weights)
  - moge           : MoGe-2 (monocular, metric depth, CVPR'25)

Output formats:
  npy (default): {datadir}/rgb/{frame}-left_depth.npy (float32, values = meters * depth_scale)
  png:           {output_dir}/{frame}.png (uint16, values = meters * depth_scale)

Usage:
  # Semantic-Super (npy output, depth_scale=8)
  python Addons/generate_depth.py --datadir data/Super --method moge

  # StereoMIS (png output, depth_scale=100)
  python Addons/generate_depth.py --datadir data/P2_1 --method moge \
    --output_format png --depth_scale 100 --output_dir data/P2_1/depth
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
    """Find all left RGB images sorted by name.

    Supports both dataset conventions:
      Semantic-Super: {datadir}/rgb/*-left.png  or  *_left.png
      StereoMIS:      {datadir}/video_frames/*l.png
    """
    # Semantic-Super patterns
    files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*-left.png')))
    if not files:
        files = sorted(glob.glob(os.path.join(datadir, 'rgb', '*_left.png')))
    # StereoMIS pattern
    if not files:
        files = sorted(glob.glob(os.path.join(datadir, 'video_frames', '*l.png')))
    if not files:
        raise FileNotFoundError(
            f"No left images found in {datadir}/rgb/ or {datadir}/video_frames/")
    print(f"Found {len(files)} left images")
    return files


def save_depth(left_path, depth, depth_scale, output_format='npy', output_dir=None):
    """Save depth map in the specified format.

    Args:
        left_path: path to the source left RGB image
        depth: depth in meters, shape (H, W) float32
        depth_scale: multiplier (8.0 for Super, 100 for StereoMIS)
        output_format: 'npy' (Semantic-Super) or 'png' (StereoMIS)
        output_dir: override output directory (default: alongside RGB for npy,
                    or {datadir}/depth/ for png)
    """
    if output_format == 'png':
        # 16-bit PNG: values = depth_meters * depth_scale
        depth_scaled = np.clip(depth * depth_scale, 0, 65535).astype(np.uint16)
        # Derive output filename from input frame name
        frame_name = os.path.splitext(os.path.basename(left_path))[0]
        # Strip 'l' suffix for StereoMIS (e.g., '000001l' -> '000001')
        if frame_name.endswith('l'):
            frame_name = frame_name[:-1]
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(left_path)), 'depth')
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f'{frame_name}.png')
        cv2.imwrite(out_path, depth_scaled)
        return out_path
    else:
        # NPY format (Semantic-Super default)
        base = left_path.replace('-left.png', '-left_depth.npy').replace(
            '_left.png', '_left_depth.npy')
        depth_scaled = (depth * depth_scale).astype(np.float32)
        np.save(base, depth_scaled)
        return base


# =============================================================================
# Method 1: Depth Anything V2 (Metric Indoor)
# =============================================================================
def run_depth_anything(datadir, depth_scale, target_h=480, target_w=640,
                       output_format='npy', output_dir=None):
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

        save_depth(img_path, depth, depth_scale, output_format, output_dir)

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


def run_monodepth2(datadir, depth_scale, target_h=480, target_w=640,
                   output_format='npy', output_dir=None):
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

            save_depth(img_path, depth, depth_scale, output_format, output_dir)

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
                    output_format='npy', output_dir=None,
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

            save_depth(left_path, depth, depth_scale, output_format, output_dir)

    print(f"Done. Generated {len(left_images)} depth maps.")
    sys.path.remove(repo_dir)
    sys.path.remove(os.path.join(repo_dir, 'core'))

# =============================================================================
# Method 4: MoGe
# =============================================================================
def run_moge(datadir, depth_scale, target_h=480, target_w=640,
             output_format='npy', output_dir=None):
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

            # Clamp to valid range (use depth_trunc from config via depth_scale heuristic)
            max_depth = 10.0 if depth_scale >= 100 else 5.0
            depth = np.clip(depth, 0.0, max_depth)

            # Resize to target resolution
            if depth.shape != (target_h, target_w):
                depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

            save_depth(img_path, depth, depth_scale, output_format, output_dir)

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
    parser.add_argument('--output_format', type=str, default='npy',
                        choices=['npy', 'png'],
                        help='Output format: npy (Semantic-Super) or png (StereoMIS)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory (default: auto from dataset structure)')
    parser.add_argument('--target_h', type=int, default=None,
                        help='Target height (default: auto from first image)')
    parser.add_argument('--target_w', type=int, default=None,
                        help='Target width (default: auto from first image)')
    args = parser.parse_args()

    print(f"Method: {args.method}")
    print(f"Data dir: {args.datadir}")
    print(f"Depth scale: {args.depth_scale}")
    print(f"Output format: {args.output_format}")

    fmt = args.output_format
    odir = args.output_dir

    # Auto-detect target resolution from first image if not specified
    if args.target_h is None or args.target_w is None:
        left_imgs = get_left_images(args.datadir)
        first_img = cv2.imread(left_imgs[0])
        th = args.target_h or first_img.shape[0]
        tw = args.target_w or first_img.shape[1]
        print(f"Target resolution: {tw}x{th} (from {'args' if args.target_h else 'first image'})")
    else:
        th, tw = args.target_h, args.target_w
        print(f"Target resolution: {tw}x{th}")

    if args.method == 'depth_anything':
        run_depth_anything(args.datadir, args.depth_scale, target_h=th, target_w=tw,
                           output_format=fmt, output_dir=odir)
    elif args.method == 'monodepth2':
        run_monodepth2(args.datadir, args.depth_scale, target_h=th, target_w=tw,
                       output_format=fmt, output_dir=odir)
    elif args.method == 'raft_stereo':
        print(f"Baseline: {args.baseline}m, fx: {args.fx}px")
        run_raft_stereo(args.datadir, args.depth_scale,
                        baseline=args.baseline, fx=args.fx,
                        output_format=fmt, output_dir=odir,
                        target_h=th, target_w=tw)
    elif args.method == 'moge':
        run_moge(args.datadir, args.depth_scale, target_h=th, target_w=tw,
                 output_format=fmt, output_dir=odir)


if __name__ == '__main__':
    main()
