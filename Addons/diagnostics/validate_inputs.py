"""
Validate DDS-SLAM's inputs before trusting any tracking run.

Audits what the StereoMISDataset at dataset.py:107 actually loads:

  [1] RGB frames            — count, range, missing-file gaps
  [2] Depth maps            — existence, scale, unique-value quantization,
                              range vs config's png_depth_scale
  [3] Ground-truth poses    — TUM parse, motion stats, verify gt_idx mapping
  [4] Intrinsics            — StereoCalibration.ini parse vs config's cam block
  [5] Masks                 — half-rate alignment to RGB frames (off-by-one check)
  [6] Computed edge_semantic — range, NaN, visual spot-check saved to PNG

Results are printed and optional sample images saved to --out_dir.

  python Addons/diagnostics/validate_inputs.py \\
      --basedir "F:/Datasets/StereoMIS/StereoMIS/P2_1" \\
      --config configs/StereoMIS/p2_1.yaml \\
      [--depth_dir /alt/path/to/depth]  # override if depth not in basedir
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import yaml

# run from any cwd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --- config loader (matches config.py but only pulls keys we need) ---
def load_merged_config(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    parent = cfg.get('inherit_from')
    if parent:
        parent_path = os.path.join(REPO_ROOT, parent)
        if not os.path.isfile(parent_path):
            parent_path = parent
        with open(parent_path) as f:
            base = yaml.safe_load(f)
        # shallow-merge
        for k, v in base.items():
            if isinstance(v, dict):
                base_v = cfg.get(k, {}) or {}
                merged = dict(v)
                merged.update(base_v)
                cfg[k] = merged
            else:
                cfg.setdefault(k, v)
    return cfg


def parse_tum_groundtruth(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts = float(parts[0])
            xyz = np.array(parts[1:4], float)
            qxyzw = np.array(parts[4:8], float)
            rows.append((ts, xyz, qxyzw))
    return rows


def quat_to_R(qx, qy, qz, qw):
    # standard quat->R; avoids scipy dep
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 0.0 if n < 1e-12 else 2.0 / n
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array([
        [1 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1 - (xx + yy)]
    ])


def geodesic(R1, R2):
    ct = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1.0, 1.0)
    return float(np.arccos(ct))


def cv2_write(path, img):
    cv2.imwrite(path, img)


def compute_edge_semantic_local(semantic_data):
    """Replicates dataset.py:compute_edge_semantic (UseInstance=False path)."""
    edges_semantic = cv2.Canny(semantic_data, 1, 1)
    edges = np.where(edges_semantic == 255, 0, 1).astype(np.uint8)
    dist_transform = cv2.distanceTransform(edges, cv2.DIST_L2, 0, dstType=cv2.CV_32F)
    edge_data = np.exp(-dist_transform / 10)
    return edge_data


def frame_num_of_img(path):
    base = os.path.splitext(os.path.basename(path))[0]
    # tolerate both 'NNNNNNl' and 'NNNNNN'
    import re
    n = re.sub(r'[^0-9]', '', base)
    return int(n) if n else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--basedir', required=True, help='dataset root (video_frames/, masks/, groundtruth.txt)')
    ap.add_argument('--config', required=True, help='StereoMIS/p2_1.yaml')
    ap.add_argument('--depth_dir', default=None,
                    help='override depth path if not under basedir/depth/')
    ap.add_argument('--out_dir', default='input_validation',
                    help='where to save sample panels')
    ap.add_argument('--n_samples', type=int, default=8,
                    help='number of frames to visualise')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cfg = load_merged_config(args.config)
    cam = cfg.get('cam', {})
    png_depth_scale = cam.get('png_depth_scale', 100.0)

    print(f'basedir     : {args.basedir}')
    print(f'config      : {args.config}   (png_depth_scale = {png_depth_scale})')
    print()

    # === [1] RGB frames =====================================================
    print('[1] RGB frames')
    rgb_all = sorted(glob.glob(os.path.join(args.basedir, 'video_frames', '*l.png')))
    print(f'    total *l.png files           : {len(rgb_all)}')
    if not rgb_all:
        print('    ABORT — no RGB frames found')
        return
    nums = np.array([frame_num_of_img(p) for p in rgb_all])
    gaps = np.diff(nums)
    print(f'    frame range                  : {nums.min()} .. {nums.max()}')
    print(f'    consecutive? (max gap = 1)   : max gap = {gaps.max()}  (1 = ok)')
    rgb_back4000 = rgb_all[-4000:]
    back_nums = np.array([frame_num_of_img(p) for p in rgb_back4000])
    print(f'    back-4000 slice              : frames {back_nums.min()} .. {back_nums.max()}  '
          f'(expect {nums.max() - 3999} .. {nums.max()})')
    print()

    # === [2] Depth maps =====================================================
    print('[2] Depth maps')
    depth_dir = args.depth_dir or os.path.join(args.basedir, 'depth')
    depth_all = sorted(glob.glob(os.path.join(depth_dir, '*.png')))
    print(f'    depth dir                    : {depth_dir}')
    print(f'    total .png files             : {len(depth_all)}')
    if depth_all:
        depth_back = depth_all[-4000:]
        dn = np.array([frame_num_of_img(p) for p in depth_back])
        print(f'    back-4000 depth range        : frames {dn.min()} .. {dn.max()}')
        # Scale / quantization audit on a sample of 20 frames
        idxs = np.linspace(0, len(depth_back) - 1, min(20, len(depth_back))).astype(int)
        samples = []
        for i in idxs:
            d = cv2.imread(depth_back[i], cv2.IMREAD_UNCHANGED)
            if d is None:
                continue
            samples.append(d)
        if samples:
            arr = np.stack(samples)
            nz = arr[arr > 0]
            scale = png_depth_scale
            print(f'    sample frames audited        : {len(samples)}')
            print(f'    raw uint16 range             : [{arr.min()}, {arr.max()}]')
            print(f'    nonzero-only min..max        : [{nz.min()}, {nz.max()}]')
            print(f'    /= png_depth_scale           : depth in m [{nz.min() / scale:.4f}, {nz.max() / scale:.4f}]')
            print(f'    valid fraction               : {(arr > 0).mean() * 100:.2f}%')
            uniques_per_frame = [np.unique(s[s > 0]).size for s in samples]
            print(f'    unique values per frame      : min={min(uniques_per_frame)}  '
                  f'median={int(np.median(uniques_per_frame))}  max={max(uniques_per_frame)}')
            flag = ' [!] QUANTISED' if min(uniques_per_frame) < 50 else ''
            print(f'    (fewer than ~50 uniques usually means scale is wrong){flag}')
            # StereoMIS depths are ~0.03-0.25m typically
            if nz.min() / scale > 1.0:
                print(f'    [!] min depth {nz.min()/scale:.2f}m seems too LARGE — scale likely too small')
            if nz.max() / scale < 0.01:
                print(f'    [!] max depth {nz.max()/scale:.4f}m seems too SMALL — scale likely too large')
    else:
        print('    (depth not local — skip quantisation audit)')
    print()

    # === [3] Ground-truth poses =============================================
    print('[3] Ground-truth poses')
    gt_path = os.path.join(args.basedir, 'groundtruth.txt')
    if os.path.isfile(gt_path):
        rows = parse_tum_groundtruth(gt_path)
        print(f'    groundtruth.txt              : {gt_path}')
        print(f'    rows parsed                  : {len(rows)}')
        ts = np.array([r[0] for r in rows])
        xyz = np.array([r[1] for r in rows])
        quat = np.array([r[2] for r in rows])
        print(f'    timestamp range              : {ts.min():.3f} .. {ts.max():.3f}')
        print(f'    translation origin (row 0)   : {xyz[0]*1000} mm   (expect ~0)')
        print(f'    |xyz| range                  : {np.linalg.norm(xyz, axis=1).min()*1000:.3f} .. '
              f'{np.linalg.norm(xyz, axis=1).max()*1000:.3f} mm')
        motions = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
        print(f'    frame-to-frame motion (mm)   : median={np.median(motions):.3f}  '
              f'mean={motions.mean():.3f}  p95={np.quantile(motions, 0.95):.3f}  '
              f'max={motions.max():.3f}')
        # Rotation from quaternion — check quat norm
        quat_norms = np.linalg.norm(quat, axis=1)
        print(f'    quat norms (should be ~1)    : mean={quat_norms.mean():.6f}  '
              f'std={quat_norms.std():.6f}')
        # Frame-to-frame rotation
        R_all = np.stack([quat_to_R(q[0], q[1], q[2], q[3]) for q in quat])
        rots = np.array([geodesic(R_all[i], R_all[i - 1]) for i in range(1, len(R_all))])
        print(f'    frame-to-frame rotation (deg): median={np.degrees(np.median(rots)):.4f}  '
              f'p95={np.degrees(np.quantile(rots, 0.95)):.4f}  max={np.degrees(rots.max()):.4f}')
        # GT index lookup check — dataset.py:238-239 uses frame_num - 1 as gt_idx.
        # Back-4000 frames are img_files[-4000:] with frame numbers nums.max()-3999 .. nums.max().
        first_back_num = back_nums.min()
        last_back_num = back_nums.max()
        print(f'    GT idx for first back frame  : gt[{first_back_num - 1}]   '
              f'xyz = {xyz[first_back_num - 1]*1000} mm')
        print(f'    GT idx for last back frame   : gt[{last_back_num - 1}]    '
              f'xyz = {xyz[last_back_num - 1]*1000} mm')
        print(f'    (N_rows = {len(rows)}, back range indices = {first_back_num - 1}..{last_back_num - 1})')
    else:
        print(f'    no groundtruth.txt at {gt_path} — tracker will run on identity GT')
    print()

    # === [4] Intrinsics =====================================================
    print('[4] Intrinsics')
    calib_path = os.path.join(args.basedir, 'StereoCalibration.ini')
    print(f'    calib file                   : {calib_path}   exists={os.path.isfile(calib_path)}')
    if cam:
        print(f'    config cam.fx / fy / cx / cy : {cam.get("fx")} / {cam.get("fy")} / '
              f'{cam.get("cx")} / {cam.get("cy")}')
        print(f'    config H / W                 : {cam.get("H")} / {cam.get("W")}')
        # Sanity: cx ~ W/2, cy ~ H/2
        if cam.get('W') and cam.get('cx'):
            pct = abs(cam['cx'] - cam['W'] / 2) / cam['W'] * 100
            print(f'    |cx - W/2| / W = {pct:.1f}%   (principal point offset)')
    print()

    # === [5] Masks + alignment ==============================================
    print('[5] Masks')
    masks_all = sorted(glob.glob(os.path.join(args.basedir, 'masks', '*.png')))
    print(f'    total mask files             : {len(masks_all)}')
    if masks_all:
        mn = np.array([frame_num_of_img(p) for p in masks_all])
        print(f'    mask frame range             : {mn.min()} .. {mn.max()}')
        mask_gaps = np.diff(mn)
        print(f'    mask stride                  : {int(np.median(mask_gaps))} '
              f'(min={mask_gaps.min()}, max={mask_gaps.max()})')
        masks_back2000 = masks_all[-2000:]
        mnb = np.array([frame_num_of_img(p) for p in masks_back2000])
        print(f'    back-2000 mask range         : {mnb.min()} .. {mnb.max()}')
        # Check dataset.py:145  semantic_paths[index//2] alignment for back-4000
        # RGB index 0 = frame back_nums[0];  mask requested = masks_back2000[0]
        # Are these the same frame number?
        print(f'    RGB back[0]    frame         : {back_nums[0]}')
        print(f'    mask back[0]   frame         : {mnb[0]}')
        off = int(mnb[0]) - int(back_nums[0])
        print(f'    offset (mask - RGB)          : {off} frame(s)')
        if off != 0:
            print(f'    [!] mask for RGB frame {back_nums[0]} will actually come from '
                  f'mask frame {mnb[0]}  (off by {off})')
        # Check several examples
        print(f'    --- dataset.py:145 alignment audit (RGB idx -> mask idx) ---')
        for rgb_idx in [0, 1, 2, 3, 100, 500, 1000, 3999]:
            rgb_frame = back_nums[rgb_idx] if rgb_idx < len(back_nums) else -1
            mask_idx = rgb_idx // 2
            mask_frame = mnb[mask_idx] if mask_idx < len(mnb) else -1
            print(f'       rgb_idx={rgb_idx:>4d}  rgb_frame={rgb_frame:>5d}  '
                  f'mask_idx={mask_idx:>4d}  mask_frame={mask_frame:>5d}  '
                  f'drift={int(mask_frame) - int(rgb_frame):+d}f')
    print()

    # === [6] Computed edge_semantic =========================================
    print('[6] edge_semantic field (computed per dataset.py:162)')
    if masks_all:
        # Sample a few back masks, compute edge field, report range + save viz
        sample_idxs = np.linspace(0, len(masks_back2000) - 1,
                                  min(args.n_samples, len(masks_back2000))).astype(int)
        for i_panel, mi in enumerate(sample_idxs):
            m_path = masks_back2000[mi]
            m_img = cv2.imread(m_path)
            if m_img is None:
                continue
            edge = compute_edge_semantic_local(m_img)
            # Paired RGB (the frame DDS-SLAM would associate via rgb_idx = 2*mi or 2*mi+1)
            rgb_idx_a, rgb_idx_b = 2 * mi, 2 * mi + 1
            if rgb_idx_a >= len(rgb_back4000):
                continue
            rgb_path = rgb_back4000[rgb_idx_a]
            rgb_img = cv2.imread(rgb_path)
            rgb_frame = frame_num_of_img(rgb_path)
            mask_frame = frame_num_of_img(m_path)
            # Visualise: [RGB | mask | edge_field]
            H, W = m_img.shape[:2]
            if rgb_img.shape[:2] != (H, W):
                rgb_img = cv2.resize(rgb_img, (W, H))
            edge_u8 = (edge * 255).astype(np.uint8)
            edge_rgb = cv2.applyColorMap(edge_u8, cv2.COLORMAP_TURBO)
            panel = np.hstack([rgb_img, m_img, edge_rgb])
            # Label
            cv2.putText(panel, f'RGB f{rgb_frame}', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panel, f'Mask f{mask_frame} (drift {mask_frame - rgb_frame:+d})',
                        (W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(panel, f'edge_semantic  [{edge.min():.3f},{edge.max():.3f}]',
                        (2 * W + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            out = os.path.join(args.out_dir, f'panel_{i_panel:02d}_rgb{rgb_frame:06d}.png')
            cv2_write(out, panel)
        print(f'    saved {len(sample_idxs)} panels -> {args.out_dir}/')
        print(f'    edge range should be [~0, 1.0] — 1 at edges, decays to 0 away')
    else:
        print('    skipped (no masks)')
    print()
    print('input validation complete.')


if __name__ == '__main__':
    main()
