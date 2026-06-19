#!/usr/bin/env python3
"""Assemble a CRCD snippet into the SGS-SLAM (ReplicaDataset) scene layout (Arm-4 A4-1.4).

SGS-SLAM's ReplicaDataset globs, per scene dir:
  frames/frame{i:06d}.jpg  depths/depth{i:06d}.png  semantic_ids/semantic_id{i:06d}.png
  semantic_colors/semantic_color{i:06d}.png  traj.txt (4x4 c2w/line)
so we build exactly that from the preprocessed CRCD outputs + the raw MoGe-2 depth, reusing the
Replica loader (configs/data/crcd.yaml sets dataset_name:'replica'). Everything is kept RECTIFIED
(consistent with the DDS-SLAM CRCD runs): rgb + semantic come from preprocess_crcd_published
(already rectified); the raw-space MoGe depth is rectified here with the SAME left map.

Pairing is BY SORTED INDEX (depth files are 000000.png... sequential; rgb is 000000l.png...; they
share order, not basenames). Prints np.unique(semantic) so the class count is confirmed.

Inputs:
  --staged       data/CRCD/<NAME>  (video_frames/*l.png, semantic_class/*.png, groundtruth.txt, rectified_calib.txt)
  --moge_depth   raw MoGe-2 depth dir (uint16, raw image space)  e.g. .../CRCD-Published-MoGe-2/<EP>/snippet_<SID>/depth
  --calib_pkl    ECM_STEREO ...opencv.pkl  (left rectification map, to rectify the depth)
  --out          SGS scene dir to create
  --depth_scale  png value / scale = metres (default 10000; verify via Sim3 path-ratio)
  --n_classes    expected semantic classes (default 4: 0=bg 1=Liver 2=Gallbladder 3=Tool)
  --emit_yaml    optional path to write configs/data/crcd.yaml (intrinsics from rectified_calib)

Usage:
  python Addons/colab/crcd_assemble_sgs.py --staged data/CRCD/C1_001 \
     --moge_depth /content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth \
     --calib_pkl <CALIB>.pkl --out /content/SGS-SLAM/data/crcd/C1_001 \
     --emit_yaml /content/SGS-SLAM/configs/data/crcd.yaml
"""
import argparse
import glob
import os
import pickle

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    print(f"[assemble] need cv2: {e}"); raise

# 4-class colour LUT (RGB): bg grey, Liver red, Gallbladder green, Tool blue
LUT = np.array([[60, 60, 60], [220, 40, 40], [40, 200, 40], [40, 90, 235]], np.uint8)


def load_left_map(calib_pkl):
    with open(calib_pkl, 'rb') as f:
        c = pickle.load(f)
    for kx, ky in (('ecm_map_left_x', 'ecm_map_left_y'), ('map_left_x', 'map_left_y'),
                   ('mapLx', 'mapLy'), ('left_map_x', 'left_map_y')):
        if kx in c and ky in c:
            return np.asarray(c[kx]), np.asarray(c[ky])
    raise RuntimeError(f"{calib_pkl}: no left rectification map found (keys: {list(c)[:20]})")


def read_rectified_calib(path):
    kv = {}
    for line in open(path):
        p = line.split()
        if len(p) >= 2:
            try:
                kv[p[0]] = float(p[1])
            except ValueError:
                pass
    return kv


def tum_to_c2w(tx, ty, tz, qx, qy, qz, qw):
    """TUM quaternion is [qx,qy,qz,qw] (xyzw). Returns 4x4 c2w."""
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5 + 1e-12
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [tx, ty, tz]
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--staged', required=True)
    ap.add_argument('--moge_depth', required=True)
    ap.add_argument('--calib_pkl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--depth_scale', type=float, default=10000.0)
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--emit_yaml', default=None)
    a = ap.parse_args()

    rgbs = sorted(glob.glob(os.path.join(a.staged, 'video_frames', '*l.png')))
    deps = sorted(glob.glob(os.path.join(a.moge_depth, '*.png')))
    sems = sorted(glob.glob(os.path.join(a.staged, 'semantic_class', '*.png')))
    gt = [ln.split() for ln in open(os.path.join(a.staged, 'groundtruth.txt')) if ln.strip() and not ln.startswith('#')]
    n = min(len(rgbs), len(deps), len(sems), len(gt))
    if not n:
        raise RuntimeError(f"empty pairing: rgb={len(rgbs)} depth={len(deps)} sem={len(sems)} gt={len(gt)}")
    if len({len(rgbs), len(deps), len(sems), len(gt)}) > 1:
        print(f"[assemble] WARN count mismatch rgb={len(rgbs)} depth={len(deps)} sem={len(sems)} "
              f"gt={len(gt)} -> pairing first {n} by index")

    mlx, mly = load_left_map(a.calib_pkl)
    for sub in ('frames', 'depths', 'semantic_ids', 'semantic_colors'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    uniq = set()
    traj = []
    for i in range(n):
        # rgb (already rectified) -> frames/frame{i}.jpg
        rgb = cv2.imread(rgbs[i], cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(a.out, 'frames', f'frame{i:06d}.jpg'), rgb)
        # raw MoGe depth -> rectify with the SAME left map (NEAREST preserves depth values) -> depths/depth{i}.png
        dep = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED)
        if dep.shape[:2] != mlx.shape[:2]:
            dep = cv2.resize(dep, (mlx.shape[1], mlx.shape[0]), interpolation=cv2.INTER_NEAREST)
        dep_rect = cv2.remap(dep, mlx, mly, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(a.out, 'depths', f'depth{i:06d}.png'), dep_rect.astype(np.uint16))
        # rectified 4-class semantic -> semantic_ids (id) + semantic_colors (LUT)
        sem = cv2.imread(sems[i], cv2.IMREAD_UNCHANGED)
        if sem.ndim == 3:
            sem = sem[..., 0]
        sem = sem.astype(np.uint8)
        uniq.update(np.unique(sem).tolist())
        cv2.imwrite(os.path.join(a.out, 'semantic_ids', f'semantic_id{i:06d}.png'), sem)
        col = LUT[np.clip(sem, 0, len(LUT) - 1)]                       # RGB
        cv2.imwrite(os.path.join(a.out, 'semantic_colors', f'semantic_color{i:06d}.png'),
                    cv2.cvtColor(col, cv2.COLOR_RGB2BGR))
        # TUM pose -> 4x4 c2w line
        g = gt[i]
        tx, ty, tz, qx, qy, qz, qw = (float(g[1]), float(g[2]), float(g[3]),
                                      float(g[4]), float(g[5]), float(g[6]), float(g[7]))
        traj.append(' '.join(f'{v:.9f}' for v in tum_to_c2w(tx, ty, tz, qx, qy, qz, qw).reshape(-1)))
    open(os.path.join(a.out, 'traj.txt'), 'w').write('\n'.join(traj) + '\n')

    print(f"[assemble] {n} frames -> {a.out}")
    print(f"[assemble] semantic unique ids = {sorted(uniq)}  (expected 0..{a.n_classes - 1}; "
          f"{'bg present' if 0 in uniq else 'NO bg class - consider n_classes=3'})")

    if a.emit_yaml:
        kv = read_rectified_calib(os.path.join(a.staged, 'rectified_calib.txt'))
        h = int(kv.get('height', rgb.shape[0])); w = int(kv.get('width', rgb.shape[1]))
        os.makedirs(os.path.dirname(os.path.abspath(a.emit_yaml)), exist_ok=True)
        with open(a.emit_yaml, 'w') as f:
            f.write("dataset_name: 'replica'\n")  # reuse the Replica loader (layout matches)
            f.write("camera_params:\n")
            f.write(f"  image_height: {h}\n  image_width: {w}\n")
            f.write(f"  fx: {kv['fx']}\n  fy: {kv['fy']}\n  cx: {kv['cx']}\n  cy: {kv['cy']}\n")
            f.write(f"  png_depth_scale: {a.depth_scale}\n  crop_edge: 0\n")
        print(f"[assemble] wrote {a.emit_yaml} (fx={kv['fx']} fy={kv['fy']} cx={kv['cx']} cy={kv['cy']} "
              f"{w}x{h} scale={a.depth_scale})")


if __name__ == '__main__':
    main()
