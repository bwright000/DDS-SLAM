#!/usr/bin/env python3
"""PERSEUS/DROID-SLAM traj_est.npy -> DDS est_c2w_data.txt (16 floats/line, 4x4 row-major c2w).

Row layout of traj_est (N,7) — derived from the PERSEUS repo's OWN code, not assumed:

  * c2w, positions first: SegmentedSLAM/evaluation_scripts/test_tum.py:109-112 builds
        PoseTrajectory3D(positions_xyz=traj_est[:,:3], orientations_quat_wxyz=traj_est[:,3:], ...)
    and compares it DIRECTLY (align=True) against the TUM groundtruth read by
    evo.file_interface.read_tum_trajectory_file (test_tum.py:114-119). TUM GT poses are
    camera-to-world, so the authors treat traj_est rows as C2W [tx ty tz | quat].
  * The producer agrees: upstream droid.py terminate() returns
        self.traj_filler(stream)  ->  camera_trajectory.inv().data.cpu().numpy()
    where video.poses are w2c (lietorch SE3); .inv() makes them c2w. (In the PERSEUS fork
    those two lines are shipped commented-out at droid_slam/droid.py:92-93; the runbook
    restores them verbatim — see run_perseus.sh apply_terminate_restore_patch.)
  * Quaternion order: lietorch SE3 .data is w-LAST (tx,ty,tz,qx,qy,qz,qw) — proven inside
    the repo by droid_slam/depth_video.py:44, which initialises the identity pose as
    [0, 0, 0, 0, 0, 0, 1] (w in the LAST slot). test_tum.py:111 labels cols 3: as
    'quat_wxyz'; that is the well-known upstream DROID-SLAM eval mislabel and is harmless
    there because evo's ATE(translation_part) consumes only positions_xyz. We use the TRUE
    order (xyzw). NOTE our consumer (Addons/eval/sim3_ate.py load_est) also reads ONLY the
    translations [3,7,11], so the quaternion-order choice cannot affect any reported metric;
    it only matters if these matrices are ever reused for rendering/warping.

Usage:
  python Addons/eval/perseus_traj_to_est.py --traj traj_est.npy --out est_c2w_data.txt \
      --expected_frames 360
"""
import argparse

import numpy as np


def quat_xyzw_to_R(q):
    """unit quaternion (x,y,z,w) -> 3x3 rotation matrix (pure numpy; no scipy needed)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traj', required=True, help='traj_est.npy from PERSEUS demo.py (N,7)')
    ap.add_argument('--out', required=True, help='est_c2w_data.txt (16 floats/line, 4x4 c2w)')
    ap.add_argument('--expected_frames', type=int, default=None,
                    help='assert row count == number of streamed frames (stride-1 => = frame count)')
    a = ap.parse_args()

    traj = np.load(a.traj, allow_pickle=True)
    # the un-patched PERSEUS terminate() returns None -> np.save writes a 0-d object array.
    assert traj is not None and getattr(traj, 'ndim', 0) == 2 and traj.shape[1] == 7, (
        f"traj_est has shape/dtype {getattr(traj, 'shape', None)}/{getattr(traj, 'dtype', None)} "
        f"— expected (N,7) float [tx ty tz qx qy qz qw]. A 0-d object array means demo.py ran with "
        f"the fork's gutted terminate() (returns None); the runbook's terminate-restore patch did "
        f"not apply — inspect droid_slam/droid.py.")
    traj = traj.astype(float)
    if a.expected_frames is not None:
        assert len(traj) == a.expected_frames, (
            f"traj rows ({len(traj)}) != streamed frames ({a.expected_frames}). With --stride 1 the "
            f"trajectory filler emits one pose per input frame; a mismatch means a stride!=1 run, a "
            f"crashed fill, or a stale traj_est.npy was picked up.")

    with open(a.out, 'w') as f:
        for row in traj:
            t, q = row[:3], row[3:7]
            n = np.linalg.norm(q)
            assert n > 1e-8, f"zero quaternion in row {row}"
            c2w = np.eye(4)
            c2w[:3, :3] = quat_xyzw_to_R(q / n)
            c2w[:3, 3] = t
            f.write(' '.join(f'{v:.10f}' for v in c2w.reshape(-1)) + '\n')
    print(f"[perseus_traj_to_est] {len(traj)} poses (c2w, quat xyzw) -> {a.out}")


if __name__ == '__main__':
    main()
