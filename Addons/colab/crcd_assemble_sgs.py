#!/usr/bin/env python3
"""Assemble a CRCD snippet into the SGS-SLAM (ReplicaDataset) scene layout (Arm-4 A4-1.4).

SGS-SLAM's ReplicaDataset globs, per scene dir:
  frames/frame{i:06d}.jpg  depths/depth{i:06d}.png  semantic_ids/semantic_id{i:06d}.png
  semantic_colors/semantic_color{i:06d}.png  traj.txt (4x4 c2w/line)
so we build exactly that, reusing the Replica loader (configs/data/crcd.yaml sets
dataset_name:'replica').

Two MODES (--mode):
  rawleft  (DEFAULT, policy [[feedback_crcd_rawleft_only_20260618]]) — NO rectification.
           rgb from raw rgb/*.png, depth COPIED as-is from the MoGe dir (already raw-space
           aligned -> NO depth rectification), semantic from raw semantic_instance/*.png.
           Intrinsics = RAW-left K (from intrinsics.yaml / calib pickle fallback).
           CAVEAT: SGS-SLAM is a PINHOLE rasterizer; raw-left frames carry lens distortion,
           so rectified is geometrically cleaner for a pinhole model. Policy default is
           raw-left anyway (consistency across ALL CRCD work; MoGe depth is raw-aligned so
           rawleft avoids a depth-remap step). Rectified is GUARDED behind ALLOW_RECTIFIED=1
           in the runbook.
  rectified — the ORIGINAL path: rgb + semantic from preprocess_crcd_published (already
           rectified), MoGe depth rectified here with the SAME left map, intrinsics from
           rectified_calib.txt. Selected by the runbook when ALLOW_RECTIFIED=1.
  semgauss — SAME rectified inputs + transforms as `rectified`, but writes the SemGauss
           gradslam ReplicaDataset tree instead: rgb/rgb_{i:06d}.png (PNG, NOT jpg),
           depth/depth_{i:06d}.png (underscore mandatory), semantic_remap/semantic_{i:06d}.png,
           traj.txt, + emit_yaml. Requires --staged + --calib_pkl. Used by run_semgauss.sh
           (Arm-4 method #3). One semantic PNG per rgb -> the loader's unconditional
           semantic_paths[index] stays 1:1 by construction.

Pairing is BY SORTED INDEX in both modes (depth files are 000000.png... sequential; rgb/sem
share order, not basenames). Prints np.unique(semantic) so the class count is confirmed.

Inputs (rawleft):
  --rgb_dir         raw rgb dir          .../CRCD-Published/<EP>/snippet_<SID>/rgb
  --sem_dir         raw semantic dir     .../CRCD-Published/<EP>/snippet_<SID>/semantic_instance
  --moge_depth      raw MoGe-2 depth dir .../CRCD-Published-MoGe-2/<EP>/snippet_<SID>/depth
  --groundtruth     TUM groundtruth.txt  .../CRCD-Published/<EP>/snippet_<SID>/groundtruth.txt
  --intrinsics_yaml raw-left intrinsics.yaml (camera.fx/fy/cx/cy) ; calib_pkl is the fallback

Inputs (rectified):
  --staged          data/CRCD/<NAME> (video_frames/*l.png, semantic_class/*.png,
                    groundtruth.txt, rectified_calib.txt)
  --moge_depth      raw MoGe-2 depth dir (rectified here with the left map)
  --calib_pkl       ECM_STEREO ...opencv.pkl  (left rectification map)

Common:
  --out          SGS scene dir to create
  --depth_scale  png value / scale = metres (default 10000; verify via Sim3 path-ratio)
  --n_classes    expected semantic classes (default 4: 0=bg 1=Liver 2=Gallbladder 3=Tool)
  --emit_yaml    optional path to write configs/data/crcd.yaml (intrinsics per mode)

Usage (rawleft, DEFAULT):
  python Addons/colab/crcd_assemble_sgs.py --mode rawleft \
     --rgb_dir  /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/rgb \
     --sem_dir  /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/semantic_instance \
     --moge_depth /content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth \
     --groundtruth /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/groundtruth.txt \
     --intrinsics_yaml /content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001/intrinsics.yaml \
     --calib_pkl <CALIB>.pkl \
     --out /content/SGS-SLAM/data/crcd/C1_001 \
     --emit_yaml /content/SGS-SLAM/configs/data/crcd.yaml

Usage (rectified, guarded):
  python Addons/colab/crcd_assemble_sgs.py --mode rectified --staged data/CRCD/C1_001 \
     --moge_depth /content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth \
     --calib_pkl <CALIB>.pkl --out /content/SGS-SLAM/data/crcd/C1_001 \
     --emit_yaml /content/SGS-SLAM/configs/data/crcd.yaml
"""
import argparse
import glob
import os
import pickle
import re

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


def _K_from_3x3(M):
    """Pull fx,fy,cx,cy out of a 3x3 intrinsics matrix (list/ndarray)."""
    M = np.asarray(M, dtype=np.float64).reshape(3, 3)
    return dict(fx=float(M[0, 0]), fy=float(M[1, 1]), cx=float(M[0, 2]), cy=float(M[1, 2]))


def read_raw_left_K(intrinsics_yaml, calib_pkl):
    """Parse the RAW-left intrinsics DEFENSIVELY.

    Try, in order:
      1) intrinsics.yaml  -> camera.{fx,fy,cx,cy}  (the CRCD-Published schema; NOTE: this
         file may already hold the RECTIFIED K depending on how the snippet was packaged —
         flagged loud below, verify fx/cx against a known raw value on first run).
      2) intrinsics.yaml  -> top-level fx/fy/cx/cy or a K/camera_matrix 3x3.
      3) calib pickle     -> cameraMatrix_left / M1 / K_left / KL (3x3, the true raw-left K).
    Fail LOUD if none yields a usable K.
    Returns (K dict {fx,fy,cx,cy}, width, height, source_str).
    """
    # ---- 1/2: intrinsics.yaml ----
    if intrinsics_yaml and os.path.isfile(intrinsics_yaml):
        try:
            import yaml
            with open(intrinsics_yaml) as f:
                y = yaml.safe_load(f) or {}
        except Exception as e:  # pragma: no cover
            print(f"[assemble] WARN could not parse {intrinsics_yaml}: {e}"); y = {}
        cam = y.get('camera', y) if isinstance(y, dict) else {}
        # CRCD-Published intrinsics.yaml carries the RECTIFIED K (is_rectified_K: true,
        # frame_state: pre_rectification) — WRONG for the distorted raw frames rawleft writes.
        # Skip it and fall through to the calib pickle's true raw-left K (fail loud if none).
        rect_flagged = bool((cam.get('is_rectified_K') if isinstance(cam, dict) else None)
                            or (y.get('is_rectified_K') if isinstance(y, dict) else None)
                            or str((cam.get('frame_state', '') if isinstance(cam, dict) else '')
                                   or (y.get('frame_state', '') if isinstance(y, dict) else '')).startswith('pre_rect'))
        if rect_flagged:
            print("[assemble] intrinsics.yaml is flagged RECTIFIED/pre_rectification -> NOT using it "
                  "for raw-left distorted frames; falling through to the calib-pickle raw-left K.")
        elif isinstance(cam, dict) and all(k in cam for k in ('fx', 'fy', 'cx', 'cy')):
            w = int(cam.get('width', 1280)); h = int(cam.get('height', 720))
            return (dict(fx=float(cam['fx']), fy=float(cam['fy']),
                         cx=float(cam['cx']), cy=float(cam['cy'])), w, h,
                    f"intrinsics.yaml:camera ({intrinsics_yaml})")
        for kk in ('K', 'camera_matrix', 'cameraMatrix', 'M1', 'intrinsic_matrix'):
            if isinstance(y, dict) and kk in y:
                try:
                    K = _K_from_3x3(y[kk])
                    w = int(y.get('width', 1280)); h = int(y.get('height', 720))
                    return K, w, h, f"intrinsics.yaml:{kk}"
                except Exception:
                    pass
    # ---- 3: calib pickle fallback (TRUE raw-left K) ----
    if calib_pkl and os.path.isfile(calib_pkl):
        with open(calib_pkl, 'rb') as f:
            c = pickle.load(f)
        for kk in ('cameraMatrix_left', 'M1', 'K_left', 'KL', 'left_camera_matrix',
                   'mtxL', 'cameraMatrixL'):
            if kk in c:
                try:
                    K = _K_from_3x3(c[kk])
                    print(f"[assemble] raw-left K from calib pickle '{kk}' (true raw-left K)")
                    return K, 1280, 720, f"calib_pkl:{kk}"
                except Exception:
                    pass
        raise RuntimeError(
            f"raw-left K not found: intrinsics.yaml had no usable camera.{{fx,fy,cx,cy}} and "
            f"calib pickle has none of cameraMatrix_left/M1/K_left/KL (keys: {list(c)[:20]})")
    raise RuntimeError(
        f"raw-left K not found and no calib pickle to fall back to "
        f"(intrinsics_yaml={intrinsics_yaml!r}, calib_pkl={calib_pkl!r})")


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


def _read_gt(path):
    return [ln.split() for ln in open(path)
            if ln.strip() and not ln.startswith('#')]


def _nsort(paths):
    """Sort by the trailing integer in the basename (robust to non-zero-padded MoGe names);
    each input list ends up frame-order-ascending so zip() pairs by position (= index pairing)."""
    def key(p):
        m = re.findall(r'\d+', os.path.basename(p))
        return (int(m[-1]) if m else 0, p)
    return sorted(paths, key=key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['rawleft', 'rectified', 'semgauss'], default='rawleft',
                    help='rawleft (DEFAULT) | rectified (SGS ReplicaDataset layout) | '
                         'semgauss (SemGauss ReplicaDataset layout: rgb/rgb_*.png depth/depth_*.png '
                         'semantic_remap/semantic_*.png). semgauss uses the SAME rectified inputs as '
                         'rectified (--staged + --calib_pkl), only the output tree differs.')
    # rawleft inputs
    ap.add_argument('--rgb_dir', help='[rawleft] raw rgb dir (frame_*.png)')
    ap.add_argument('--sem_dir', help='[rawleft] raw semantic_instance dir (frame_*.png uint16)')
    ap.add_argument('--groundtruth', help='[rawleft] TUM groundtruth.txt')
    ap.add_argument('--intrinsics_yaml', help='[rawleft] raw-left intrinsics.yaml')
    # rectified inputs
    ap.add_argument('--staged', help='[rectified] data/CRCD/<NAME> (preprocessed)')
    # common
    ap.add_argument('--moge_depth', required=True, help='raw MoGe-2 depth dir (uint16)')
    ap.add_argument('--calib_pkl', help='ECM_STEREO opencv.pkl (left map for rectified; '
                                        'raw-K fallback for rawleft)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--depth_scale', type=float, default=10000.0)
    ap.add_argument('--n_classes', type=int, default=4)
    ap.add_argument('--emit_yaml', default=None)
    a = ap.parse_args()

    # SemGauss ReplicaDataset globs a DIFFERENT tree than SGS (rgb/rgb_*.png, depth/depth_*.png,
    # semantic_remap/*.png — CONFIRMED SemGauss datasets/gradslam_datasets/replica.py:44-52).
    _dirs = ('rgb', 'depth', 'semantic_remap') if a.mode == 'semgauss' \
        else ('frames', 'depths', 'semantic_ids', 'semantic_colors')
    for sub in _dirs:
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    deps = _nsort(glob.glob(os.path.join(a.moge_depth, '*.png')))

    if a.mode in ('rectified', 'semgauss'):
        if not a.staged or not a.calib_pkl:
            ap.error(f"--mode {a.mode} requires --staged and --calib_pkl")
        rgbs = _nsort(glob.glob(os.path.join(a.staged, 'video_frames', '*l.png')))
        sems = _nsort(glob.glob(os.path.join(a.staged, 'semantic_class', '*.png')))
        gt = _read_gt(os.path.join(a.staged, 'groundtruth.txt'))
        mlx, mly = load_left_map(a.calib_pkl)
    else:  # rawleft (DEFAULT)
        if not (a.rgb_dir and a.sem_dir and a.groundtruth):
            ap.error("--mode rawleft requires --rgb_dir, --sem_dir, --groundtruth")
        rgbs = _nsort(glob.glob(os.path.join(a.rgb_dir, '*.png')))
        sems = _nsort(glob.glob(os.path.join(a.sem_dir, '*.png')))
        gt = _read_gt(a.groundtruth)
        mlx = mly = None  # no rectification

    n = min(len(rgbs), len(deps), len(sems), len(gt))
    if not n:
        raise RuntimeError(f"empty pairing ({a.mode}): rgb={len(rgbs)} depth={len(deps)} "
                           f"sem={len(sems)} gt={len(gt)}")
    if len({len(rgbs), len(deps), len(sems), len(gt)}) > 1:
        print(f"[assemble] WARN count mismatch rgb={len(rgbs)} depth={len(deps)} sem={len(sems)} "
              f"gt={len(gt)} -> pairing first {n} by index")

    uniq = set()
    traj = []
    rgb = None
    for i in range(n):
        # ---- rgb ----  SGS: frames/frame{i}.jpg   SemGauss: rgb/rgb_{i}.png (PNG mandatory)
        rgb = cv2.imread(rgbs[i], cv2.IMREAD_COLOR)
        if a.mode == 'semgauss':
            cv2.imwrite(os.path.join(a.out, 'rgb', f'rgb_{i:06d}.png'), rgb)
        else:
            cv2.imwrite(os.path.join(a.out, 'frames', f'frame{i:06d}.jpg'), rgb)

        # ---- depth ----  SGS: depths/depth{i}.png   SemGauss: depth/depth_{i}.png (underscore)
        dep = cv2.imread(deps[i], cv2.IMREAD_UNCHANGED)
        if a.mode in ('rectified', 'semgauss'):
            # rectify with the SAME left map (NEAREST preserves depth values)
            if dep.shape[:2] != mlx.shape[:2]:
                dep = cv2.resize(dep, (mlx.shape[1], mlx.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            dep = cv2.remap(dep, mlx, mly, interpolation=cv2.INTER_NEAREST)
        # rawleft: COPY as-is — MoGe depth is already raw-space aligned. Still resize to the
        # rgb frame size defensively in case depth was generated at a different resolution.
        elif dep.shape[:2] != rgb.shape[:2]:
            dep = cv2.resize(dep, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        if a.mode == 'semgauss':
            cv2.imwrite(os.path.join(a.out, 'depth', f'depth_{i:06d}.png'), dep.astype(np.uint16))
        else:
            cv2.imwrite(os.path.join(a.out, 'depths', f'depth{i:06d}.png'), dep.astype(np.uint16))

        # ---- semantic ----  SGS: semantic_ids + semantic_colors   SemGauss: semantic_remap (id only)
        sem = cv2.imread(sems[i], cv2.IMREAD_UNCHANGED)
        if sem.ndim == 3:
            sem = sem[..., 0]
        sem = sem.astype(np.uint8)
        if a.mode == 'rawleft' and sem.shape[:2] != rgb.shape[:2]:
            sem = cv2.resize(sem, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        uniq.update(np.unique(sem).tolist())
        if a.mode == 'semgauss':
            # SemGauss uses GT masks as loader INPUT only (not supervision under use_gt_semantic=False);
            # one PNG per rgb -> the loader's unconditional semantic_paths[index] stays 1:1 by construction.
            cv2.imwrite(os.path.join(a.out, 'semantic_remap', f'semantic_{i:06d}.png'), sem)
        else:
            cv2.imwrite(os.path.join(a.out, 'semantic_ids', f'semantic_id{i:06d}.png'), sem)
            col = LUT[np.clip(sem, 0, len(LUT) - 1)]                       # RGB
            cv2.imwrite(os.path.join(a.out, 'semantic_colors', f'semantic_color{i:06d}.png'),
                        cv2.cvtColor(col, cv2.COLOR_RGB2BGR))

        # ---- TUM pose -> 4x4 c2w line ----
        g = gt[i]
        tx, ty, tz, qx, qy, qz, qw = (float(g[1]), float(g[2]), float(g[3]),
                                      float(g[4]), float(g[5]), float(g[6]), float(g[7]))
        traj.append(' '.join(f'{v:.9f}' for v in tum_to_c2w(tx, ty, tz, qx, qy, qz, qw).reshape(-1)))
    open(os.path.join(a.out, 'traj.txt'), 'w').write('\n'.join(traj) + '\n')

    print(f"[assemble] mode={a.mode}: {n} frames -> {a.out}")
    print(f"[assemble] semantic unique ids = {sorted(uniq)}  (expected 0..{a.n_classes - 1}; "
          f"{'bg present' if 0 in uniq else 'NO bg class - consider n_classes=3'})")

    if a.emit_yaml:
        if a.mode in ('rectified', 'semgauss'):
            kv = read_rectified_calib(os.path.join(a.staged, 'rectified_calib.txt'))
            fx, fy, cx, cy = kv['fx'], kv['fy'], kv['cx'], kv['cy']
            h = int(kv.get('height', rgb.shape[0])); w = int(kv.get('width', rgb.shape[1]))
            ksrc = 'rectified_calib.txt'
        else:
            K, w, h, ksrc = read_raw_left_K(a.intrinsics_yaml, a.calib_pkl)
            fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
            # trust the actual frame size we just wrote over any header value
            h, w = rgb.shape[0], rgb.shape[1]
        os.makedirs(os.path.dirname(os.path.abspath(a.emit_yaml)), exist_ok=True)
        with open(a.emit_yaml, 'w') as f:
            f.write("dataset_name: 'replica'\n")  # reuse the Replica loader (layout matches)
            f.write("camera_params:\n")
            f.write(f"  image_height: {h}\n  image_width: {w}\n")
            f.write(f"  fx: {fx}\n  fy: {fy}\n  cx: {cx}\n  cy: {cy}\n")
            f.write(f"  png_depth_scale: {a.depth_scale}\n  crop_edge: 0\n")
        print(f"[assemble] wrote {a.emit_yaml} (K src={ksrc}; fx={fx} fy={fy} cx={cx} cy={cy} "
              f"{w}x{h} scale={a.depth_scale})")


if __name__ == '__main__':
    main()
