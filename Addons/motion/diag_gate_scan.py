#!/usr/bin/env python3
"""Scan the flow_agree gate across a sequence vs GT camera motion — diagnostic for WHY track/fix.

For each (subsampled) frame t it runs the EXACT gate input (Addons.motion.flow_track.agreement_gate,
causal ref = t-stride): cam_mag = |median flow| (the "is the camera moving" proxy) and disagree_frac =
fraction of DINO k-means regions whose median Sampson residual exceeds the deadband (the "is the scene
moving incoherently" signal = what the DINO features 'think'). Decision = cam_mag>cam_thresh AND
disagree<=disagree_thresh. Overlays GT per-frame camera motion. Where GT moves AND disagree is high =
"scene moving at the same time as the camera -> gate FIXes -> misses the camera motion".
"""
import argparse
import glob
import os
import sys

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'eval'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from sim3_ate import load_gt_tum  # noqa: E402
from Addons.motion.flow_track import load_raft, load_dino, agreement_gate, dino_grid  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--every', type=int, default=4)
    ap.add_argument('--cam_thresh', type=float, default=2.0)
    ap.add_argument('--disagree_thresh', type=float, default=0.2)
    ap.add_argument('--deadband', type=float, default=3.0)
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--small', action='store_true', help='RAFT-small (fast; run used large)')
    a = ap.parse_args()

    import torch
    dev = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    print('device:', dev, 'RAFT-', 'small' if a.small else 'large')
    raft, tf = load_raft(dev, small=a.small)
    dino = load_dino(dev)
    files = sorted(glob.glob(a.frames_dir + '/*.png') + glob.glob(a.frames_dir + '/*.jpg'))
    files = [f for f in files if 'right' not in os.path.basename(f).lower()]
    GT = load_gt_tum(a.gt)
    gstep = np.linalg.norm(np.diff(GT, axis=0), axis=1) * 1000.0

    T, C, D, K, G = [], [], [], [], []
    for t in range(a.stride, len(files), a.every):
        cur = cv2.imread(files[t]); ref = cv2.imread(files[t - a.stride])
        dg = dino_grid(cur, dino, dev)
        cam, dis = agreement_gate(ref, cur, dg, raft, tf, dev,
                                  n_groups=a.n_groups, ransac_thresh=1.0, deadband=a.deadband)
        dt = (cam > a.cam_thresh) and (dis <= a.disagree_thresh)
        g = float(gstep[min(t, len(gstep) - 1)])
        T.append(t); C.append(cam); D.append(dis); K.append(1.0 if dt else 0.0); G.append(g)
        print(f"f{t} cam_mag={cam:.2f} disagree={dis:.2f} track={int(dt)} GT={g:.3f}")

    T, C, D, K, G = map(np.asarray, (T, C, D, K, G))
    still = G < 0.05
    # correctness vs GT: TRACK when GT moving, FIX when GT still
    gt_moving = (G >= 0.05).astype(float)
    agree_gt = (K == gt_moving).mean()

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax[0].fill_between(T, 0, 1, where=still, transform=ax[0].get_xaxis_transform(), color='0.88', label='GT still')
    ax[0].plot(T, G, 'k', lw=2, label='GT camera motion (mm/f)')
    ax[0].set_ylabel('GT mm/f'); ax[0].set_title('GT camera motion'); ax[0].grid(alpha=.3); ax[0].legend(loc='upper right', fontsize=8)

    ax[1].fill_between(T, 0, 1, where=still, transform=ax[1].get_xaxis_transform(), color='0.88')
    ax[1].plot(T, C, 'b', lw=1.4, label='cam_mag |median flow|')
    ax[1].axhline(a.cam_thresh, ls='--', c='b', lw=1, label=f'cam_thresh {a.cam_thresh}')
    ax[1].set_ylabel('cam_mag (px)', color='b'); ax[1].grid(alpha=.3); ax[1].legend(loc='upper left', fontsize=8)
    axb = ax[1].twinx()
    axb.plot(T, D, 'r', lw=1.4, label='disagree_frac (DINO regions)')
    axb.axhline(a.disagree_thresh, ls='--', c='r', lw=1, label=f'disagree_thresh {a.disagree_thresh}')
    axb.set_ylabel('disagree_frac', color='r'); axb.set_ylim(0, 1); axb.legend(loc='upper right', fontsize=8)
    ax[1].set_title('Gate inputs: cam_mag (blue, camera proxy) vs disagree_frac (red, scene-incoherence)')

    ax[2].fill_between(T, 0, 1, where=still, transform=ax[2].get_xaxis_transform(), color='0.88')
    ax[2].plot(T, K, drawstyle='steps-mid', color='purple', lw=1.8, label='gate: 1=TRACK 0=FIX')
    ax[2].plot(T, gt_moving, 'k', lw=1.2, alpha=.6, label='GT: 1=moving 0=still')
    ax[2].set_ylim(-0.1, 1.1); ax[2].set_ylabel('decision'); ax[2].set_xlabel('frame')
    ax[2].set_title(f'Gate decision vs GT (match={agree_gt*100:.0f}%) -- INVERTED where purple != black'); ax[2].grid(alpha=.3); ax[2].legend(loc='upper right', fontsize=8)

    fig.suptitle(f'E3_005 flow-gate scan (RAFT-{"small" if a.small else "large"}, every {a.every}) '
                 f'-- gate/GT agreement {agree_gt*100:.0f}%', fontsize=13)
    fig.tight_layout(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches='tight'); print('wrote', a.out)


if __name__ == '__main__':
    main()
