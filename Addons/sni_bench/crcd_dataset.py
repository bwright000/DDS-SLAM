"""CRCD dataloader for BASE SNI-SLAM (drop-in for src/utils/datasets.py).

Input adaptation ONLY — presents CRCD's native files in the contract BaseDataset expects,
so NO SNI internals are touched. Mirrors the proven DDS-SLAM StereoMISDataset reader:
  color    : {input}/video_frames/*l.png   (left of the rectified stereo pair)
  depth    : {input}/depth/*l.png          (uint16 / png_depth_scale=10000 -> metres)
  semantic : {input}/masks/*l.png          (CRCD {0 bg,1 liver,2 gallbladder,3 tool} -> {0,1,1,2})
  poses    : {input}/groundtruth.txt       (TUM: ts tx ty tz qx qy qz qw), ABSOLUTE metric

POSE FRAME = frame-0-relative (T0^-1 @ Ti), matching DDS (identity-init) / SGS (relative_pose):
frame 0 -> identity, camera starts at origin, tissue at z~+0.1. Then the authors' col-1,2 flip
(OpenCV -> OpenGL) exactly as Replica.load_poses. SNI uses gt only for the frame-0 anchor + eval
(Tracker.py:279 `if idx==0`); frames 1..N are tracked.
"""
import glob
import os
import re

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# BaseDataset is defined above this block in datasets.py; this module is appended there.


def _fid(p):
    m = re.findall(r'\d+', os.path.basename(p))
    return int(m[0]) if m else -1


class CRCD(BaseDataset):  # noqa: F821  (BaseDataset from the enclosing datasets.py)
    # CRCD colorectal masks: 0=bg, 1=liver, 2=gallbladder, 3=tool  ->  0=bg, 1=tissue, 2=tool
    _MASK_REMAP = {0: 0, 1: 1, 2: 1, 3: 2}
    semantic_classes = np.array([0, 1, 2], dtype=np.uint8)
    num_semantic_class = 3

    def __init__(self, cfg, args, scale, device='cuda:0'):
        super(CRCD, self).__init__(cfg, args, scale, device)

        # TWO on-disk layouts, same content (pairing is by frame id, so naming may differ per modality):
        #   rect_staged v2 tar (THE benchmark inputs, what every other method consumed):
        #     video_frames/NNNNNNl.png | depth/NNNNNN.png (metric stereo-anchored MoGe) |
        #     semantic_class/NNNNNN.png = RAW 4-class {0 bg,1 liver,2 gallbladder,3 tool}.
        #     masks/ there is a BINARY tool mask (0/255) -- NEVER read it as semantics.
        #   local repo data/CRCD/<name> (raw-left era): depth/NNNNNNl.png, masks/ = 4-class {0..3}.
        colors = sorted(glob.glob(f'{self.input_folder}/video_frames/*l.png'), key=_fid)
        depths = (sorted(glob.glob(f'{self.input_folder}/depth/*l.png'), key=_fid)
                  or sorted(glob.glob(f'{self.input_folder}/depth/[0-9]*.png'), key=_fid))
        masks = (sorted(glob.glob(f'{self.input_folder}/semantic_class/*.png'), key=_fid)
                 or sorted(glob.glob(f'{self.input_folder}/masks/*.png'), key=_fid))

        # pair the modalities BY FRAME ID (video_frames also holds *r; depth/masks are left-only) so
        # rgb[i]/depth[i]/mask[i] are the SAME frame -- a positional zip would silently misalign.
        cby, dby, mby = ({_fid(p): p for p in xs} for xs in (colors, depths, masks))
        ids = sorted(set(cby) & set(dby) & set(mby))
        assert ids, f"CRCD: no common frame ids across rgb/depth/masks in {self.input_folder}"
        self.color_paths = [cby[i] for i in ids]
        self.depth_paths = [dby[i] for i in ids]
        self.semantic_paths = [mby[i] for i in ids]
        self.frame_ids = ids
        self.n_img = len(ids)

        # HARD GUARD: semantics must be the RAW 4-class map. This trips loudly if the glob ever
        # lands on the binary 0/255 tool mask (rect staging's masks/) or a pre-remapped source.
        _m0 = cv2.imread(self.semantic_paths[0], cv2.IMREAD_UNCHANGED)
        if _m0 is None:
            raise FileNotFoundError(self.semantic_paths[0])
        if _m0.ndim == 3:
            _m0 = _m0[..., 0]
        _ids = set(np.unique(_m0).tolist())
        assert _ids <= {0, 1, 2, 3}, (
            f"CRCD semantics: raw ids {sorted(_ids)} in {self.semantic_paths[0]} are not the "
            f"4-class map {{0,1,2,3}} (binary tool mask or pre-remapped source?)")

        self.path = cfg['model']['path']
        self.load_poses(f'{self.input_folder}/groundtruth.txt')
        assert len(self.poses) == self.n_img, \
            f"CRCD: {len(self.poses)} poses vs {self.n_img} frames"

    # The base getitem's class-list remap is a bijection (one raw value -> one index) so it CANNOT
    # merge liver(1)+gallbladder(2) into a single tissue class. With semantic_classes=[0,1,2] it leaves
    # CRCD's raw values {0,1,2,3} unchanged (3 is unmatched -> kept), so we finish the 4->3 collapse here.
    def __getitem__(self, index):
        index, color, depth, pose, semantic = super(CRCD, self).__getitem__(index)
        s = semantic.clone()
        for k, v in self._MASK_REMAP.items():
            if k != v:
                s[semantic == k] = v
        return index, color, depth, pose, s

    def load_poses(self, path):
        """TUM groundtruth -> frame-0-relative c2w, then flip cols 1,2 (OpenCV->OpenGL)."""
        rows = []
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith('#'):
                    continue
                v = ln.split()
                if len(v) < 8:
                    continue
                rows.append([float(x) for x in v[1:8]])   # tx ty tz qx qy qz qw
        assert len(rows) >= self.n_img, \
            f"groundtruth.txt has {len(rows)} usable rows < n_img {self.n_img}"

        abs_c2w = []
        for tx, ty, tz, qx, qy, qz, qw in rows[:self.n_img]:
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            abs_c2w.append(T)

        inv0 = np.linalg.inv(abs_c2w[0])            # frame-0-relative: T0^-1 @ Ti
        self.poses = []
        for T in abs_c2w:
            c2w = inv0 @ T
            c2w[:3, 1] *= -1                        # OpenCV -> OpenGL (authors' convention)
            c2w[:3, 2] *= -1
            self.poses.append(torch.from_numpy(c2w).float())
