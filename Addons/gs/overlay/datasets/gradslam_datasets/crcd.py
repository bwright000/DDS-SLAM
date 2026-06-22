"""CRCD (colorectal surgery) loader for the GS migration Phase-0.

CRCD frames + MoGe-2 monocular depth + TUM ground-truth, in the format EndoGSLAM's GradSLAMDataset wants.
Deliberately mirrors the DDS-SLAM (NeRF) CRCD conventions so the GS run is measured on the SAME data:
  - RAW-LEFT frames only (standing policy),
  - MoGe-2 depth as FLOAT .npy (no uint16 quantisation),
  - poses parsed from groundtruth.txt (TUM) into c2w EXACTLY as DDS-SLAM/datasets/dataset.py:350-355
    (scipy Rotation.from_quat([qx,qy,qz,qw])), so Sim3-ATE is comparable across NeRF and GS.

File globs are read from the data config's optional `paths:` block (configs/data/crcd.yaml) so an unverified
staged layout is a one-line config change, never a code edit.
"""
import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted
from scipy.spatial.transform import Rotation

from .basedataset import GradSLAMDataset


class CRCDDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = 1,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 720,
        desired_width: Optional[int] = 1280,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        _paths = config_dict.get("paths", {}) if isinstance(config_dict, dict) else {}
        self._color_glob = _paths.get("color_glob", "video_frames/*l.png")
        self._depth_glob = _paths.get("depth_glob", "depth/moge2/*.npy")
        self.pose_path = os.path.join(self.input_folder, _paths.get("pose_file", "groundtruth.txt"))
        # NB: we deliberately do NOT take a train/test split here (unlike c3vd.py). SLAM + full-360 ATE
        # need every frame; the base train_test_split with stride=1 keeps them all.
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(os.path.join(self.input_folder, self._color_glob)))
        depth_paths = natsorted(glob.glob(os.path.join(self.input_folder, self._depth_glob)))
        if len(color_paths) == 0:
            raise FileNotFoundError(
                f"No CRCD frames matched {self._color_glob} under {self.input_folder} "
                f"(check the staged layout / override `paths.color_glob` in configs/data/crcd.yaml)."
            )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """Parse CRCD groundtruth.txt (TUM: `timestamp tx ty tz qx qy qz qw`) -> list of c2w [4,4].
        Identical convention to DDS-SLAM/datasets/dataset.py:350-355 so ATE matches the NeRF runs."""
        poses = []
        with open(self.pose_path, "r") as f:
            lines = [ln for ln in f.readlines() if ln.strip() and not ln.lstrip().startswith("#")]
        for ln in lines:
            v = list(map(float, ln.split()))
            tx, ty, tz = v[1:4]
            qx, qy, qz, qw = v[4:8]
            c2w = np.eye(4, dtype=np.float32)
            c2w[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            c2w[:3, 3] = [tx, ty, tz]
            poses.append(torch.from_numpy(c2w).float())
        # Guard the documented 271-vs-360 mispairing: GT row count MUST equal the staged frame count.
        if len(poses) != len(self.color_paths):
            raise ValueError(
                f"CRCD GT poses ({len(poses)}) != frames ({len(self.color_paths)}). "
                f"You are almost certainly pairing the STALE 271-row local GT against 360 Published frames "
                f"(or vice-versa). Stage CRCD-Published 360-GT. pose_file={self.pose_path}"
            )
        return poses

    def read_embedding_from_file(self, embedding_path: str):
        # Phase-3 (seg/DINO): a precomputed [1, C, H, W] feature/mask tensor -> [1, H, W, C].
        embedding = torch.load(embedding_path)
        return embedding.permute(0, 2, 3, 1)
