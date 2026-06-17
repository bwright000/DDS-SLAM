#!/usr/bin/env python3
"""Export a SNI-SLAM checkpoint's estimated trajectory to est_c2w_data.txt.

WHY THIS EXISTS — the DDS-SLAM harness (Addons/eval/sim3_ate.py:load_est and
Addons/viz/generate_video.py) consume the estimated trajectory as a plain text file
'<RUN>/est_c2w_data.txt' with ONE pose per line, row-major, translation at indices [3,7,11]
(12 floats = 3x4 c2w, or 16 floats = 4x4 c2w). SNI-SLAM does NOT write that file: it keeps
per-frame estimated poses inside its checkpoint .tar as the tensor `estimate_c2w_list`
([N,4,4] c2w). kitti_to_tum.py does the OPPOSITE conversion (it READS est_c2w_data.txt and
WRITES TUM pairs) and cannot create this file. This exporter bridges the gap.

It loads the .tar with torch.load, finds the per-frame c2w stack under one of the known keys,
flattens each 4x4 to a 16-float row-major line, and writes est_c2w_data.txt.

NOTE (OQ-10): SNI-SLAM only fills estimate_c2w_list for KEYFRAMES / processed frames; rows
that are still identity/zero are dropped UNLESS --keep-all is passed. If the count of kept
poses is far below the frame count, the harness pairing degrades to resampling — the script
prints a loud warning so the agent can escalate rather than silently publish a sparse traj.

Usage:
  python Addons/eval/sni_export_traj.py \
      --ckpt <SNI_OUT>/ckpts/<latest>.tar \
      --out  <RUN>/est_c2w_data.txt
"""
import argparse
import os

import numpy as np
import torch

CANDIDATE_KEYS = ['estimate_c2w_list', 'est_c2w_list', 'estimate_c2w', 'c2w_list', 'poses']


def _find_pose_stack(obj):
    """Return an [N,4,4] numpy array of c2w poses found in a (possibly nested) ckpt dict."""
    if isinstance(obj, dict):
        for k in CANDIDATE_KEYS:
            if k in obj:
                t = obj[k]
                arr = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
                return arr, k
        # one level of nesting (some repos wrap in {'state':{...}} or {'model':{...}})
        for v in obj.values():
            if isinstance(v, dict):
                try:
                    return _find_pose_stack(v)
                except KeyError:
                    continue
    raise KeyError(f"no pose stack under keys {CANDIDATE_KEYS}; ckpt top-level keys="
                   f"{list(obj.keys()) if isinstance(obj, dict) else type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='SNI-SLAM checkpoint .tar')
    ap.add_argument('--out', required=True, help='destination est_c2w_data.txt')
    ap.add_argument('--keep-all', action='store_true',
                    help='keep every row even if identity/zero (default drops empty rows)')
    a = ap.parse_args()

    ckpt = torch.load(a.ckpt, map_location='cpu')
    arr, key = _find_pose_stack(ckpt)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[1:] == (4, 4):
        mats = arr
    elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
        eye = np.tile(np.array([0, 0, 0, 1.0]), (arr.shape[0], 1, 1))
        mats = np.concatenate([arr, eye], axis=1)
    else:
        raise SystemExit(f"unexpected pose stack shape {arr.shape} under key '{key}'")

    kept, dropped = [], 0
    for m in mats:
        is_identity = np.allclose(m, np.eye(4))
        is_zero = np.allclose(m, 0.0)
        if not a.keep_all and (is_identity or is_zero):
            dropped += 1
            continue
        kept.append(m.reshape(-1))  # 16 floats, row-major; translation at 3,7,11

    if not kept:
        raise SystemExit("ERROR: 0 valid poses exported — check the ckpt / key")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    with open(a.out, 'w') as fh:
        for row in kept:
            fh.write(' '.join(f'{x:.9f}' for x in row) + '\n')

    n_total = len(mats)
    print(f"[export] key='{key}'  total={n_total}  written={len(kept)}  dropped(empty)={dropped}")
    print(f"[export] wrote {a.out}")
    if dropped > 0 and len(kept) < 0.9 * n_total:
        print(f"[WARN] only {len(kept)}/{n_total} frames have a non-trivial estimated pose. "
              f"sim3_ate.py will RESAMPLE to pair against GT. If this is unexpected (i.e. "
              f"SNI-SLAM does not keep per-frame poses for all frames), ESCALATE (OQ-10) "
              f"before headlining ATE.")


if __name__ == '__main__':
    main()
