"""CRCD loader for the GS migration Phase-0 — reads the ARM-4 ASSEMBLED scene layout.

We do NOT re-stage CRCD here. The canonical, tested pipeline already exists:
  DDS-SLAM/Addons/preprocess/preprocess_crcd_published.py   (rectify raw rgb + masks + K)
  DDS-SLAM/Addons/colab/crcd_assemble_sgs.py --mode rectified  (index-pair + remap MoGe depth)
which emit a gradslam Replica-style scene (the SAME layout SGS-SLAM consumes), so the GS-migration
run is measured on EXACTLY the same rectified data as the SGS benchmark (SGS c1_001 = Sim3-ATE 3.31mm).

This loader reads that assembled scene:
  <seq>/frames/frame{i:06d}.jpg          rectified RGB
  <seq>/depths/depth{i:06d}.png          uint16 MoGe-2 depth, png_depth_scale=10000 (set in the data yaml)
  <seq>/semantic_ids/semantic_id{i:06d}.png   class ids {0 bg,1 Liver,2 Gallbladder,3 Tool}  (Phase-3)
  <seq>/traj.txt                         one 4x4 c2w (row-major, 16 floats) per line (GT, TUM->c2w)

🚨 RECTIFIED is the locked policy for pinhole CRCD SLAM methods (incl. this one) — see
project_benchmark_onboarding_audit_20260617. Intrinsics + depth scale come from the staging's emitted
crcd.yaml, NOT hardcoded.
"""
import glob
import os
from typing import Optional

import numpy as np
import torch
from natsort import natsorted

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
        embedding_dir: Optional[str] = "semantic_ids",
        embedding_dim: Optional[int] = 1,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
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
        color_paths = natsorted(glob.glob(f"{self.input_folder}/frames/*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depths/*.png"))
        if len(color_paths) == 0:
            raise FileNotFoundError(
                f"No assembled CRCD frames under {self.input_folder}/frames/ — run the staging first "
                f"(preprocess_crcd_published.py + crcd_assemble_sgs.py --mode rectified)."
            )
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.png"))
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        """traj.txt: one 4x4 c2w (16 space-separated floats, row-major) per line — as written by
        crcd_assemble_sgs.py (tum_to_c2w(...).reshape(-1))."""
        poses = []
        with open(self.pose_path, "r") as f:
            for ln in f:
                if not ln.strip() or ln.lstrip().startswith("#"):
                    continue
                v = list(map(float, ln.split()))
                if len(v) != 16:
                    raise ValueError(f"traj.txt line has {len(v)} values, expected 16 (4x4 c2w): {self.pose_path}")
                poses.append(torch.from_numpy(np.array(v, dtype=np.float32).reshape(4, 4)).float())
        if len(poses) != len(self.color_paths):
            raise ValueError(
                f"CRCD GT poses ({len(poses)}) != frames ({len(self.color_paths)}) in {self.input_folder}. "
                f"Re-run the assembly (index-paired); do not mix a stale GT."
            )
        return poses

    def read_embedding_from_file(self, embedding_path: str):
        # Phase-3: class-id PNG -> [1, H, W, 1] long (canonical {0 bg,1 Liver,2 GB,3 Tool}).
        import cv2
        sem = cv2.imread(embedding_path, cv2.IMREAD_UNCHANGED)
        if sem is None:
            raise FileNotFoundError(embedding_path)
        if sem.ndim == 3:
            sem = sem[..., 0]
        return torch.from_numpy(sem.astype(np.int64))[None, ..., None]
