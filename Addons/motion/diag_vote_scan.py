#!/usr/bin/env python3
"""GATE v2 OFFLINE BENCH -- scan the region-VOTE detector (and the OLD agreement_gate side-by-side)
across a staged snippet vs GT camera motion. No SLAM: flow+DINO+depth only, so the detector is judged
BEFORE it touches the tracker.

PASS BAR (from the C1/E3 freeze-confusion analysis of the old gate):
  E3_005 : the old gate froze 117 GT-MOVING frames (28% precision) -- the vote must NOT (still-call
           precision must be high on both windows).
  C1_001 : the old gate's 185/198 correct freezes (93% precision) -- the vote must KEEP that recall.
Per scanned frame both detectors run on the SAME flow; the vote additionally consumes the staged depth.
Outputs: <out>.png (GT + inputs + decisions), <out>.json (confusions old vs new), <out>.csv (per frame).

  python Addons/motion/diag_vote_scan.py --frames_dir data/CRCD/E3_005/video_frames \
      --depth_dir data/CRCD/E3_005/depth --gt data/CRCD/E3_005/groundtruth.txt \
      --out /content/drive/MyDrive/Outputs/vote_scan/E3_005 [--every 3] [--small]
"""
import argparse
import glob
import json
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
from Addons.motion.flow_track import (load_raft, load_dino, agreement_gate,  # noqa: E402
                                      dino_grid, region_vote)


def load_depth(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d.astype(np.float32) / 10000.0            # staged uint16 @ png_depth_scale 10000


def confusion(dec_moving, gt_moving):
    """still-call confusion: how good are the FREEZE decisions."""
    dec_still = ~dec_moving; gt_still = ~gt_moving
    n_frz = int(dec_still.sum())
    ok_frz = int((dec_still & gt_still).sum())
    wrong_frz = int((dec_still & gt_moving).sum())
    prec = 100.0 * ok_frz / max(n_frz, 1)
    rec = 100.0 * ok_frz / max(int(gt_still.sum()), 1)
    agree = 100.0 * float((dec_moving == gt_moving).mean())
    return dict(n_freeze=n_frz, freeze_ok=ok_frz, freeze_wrong=wrong_frz,
                freeze_precision=round(prec, 1), still_recall=round(rec, 1), gt_agree=round(agree, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames_dir', required=True)
    ap.add_argument('--depth_dir', required=True)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--out', required=True, help='output stem (writes .png/.json/.csv)')
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--every', type=int, default=3)
    ap.add_argument('--still_floor_px', type=float, default=0.5)
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--cam_thresh', type=float, default=2.0)     # old gate, for the side-by-side
    ap.add_argument('--disagree_thresh', type=float, default=0.2)
    ap.add_argument('--deadband', type=float, default=3.0)
    ap.add_argument('--gt_still_mm', type=float, default=0.05)
    ap.add_argument('--small', action='store_true', help='RAFT-small (fast)')
    ap.add_argument('--video', action='store_true',
                    help='ALSO write <out>_votes.mp4: the DEMOCRACY overlay -- district boundaries + '
                         'per-district vote arrows (color=trust) + tally banner, on the real frames')
    ap.add_argument('--k_list', default='',
                    help='K-SWEEP: comma list of extra district counts (e.g. "6,24"). RAFT flow + DINO '
                         'grid are computed ONCE per frame and shared; only KMeans+pooling repeat, so '
                         'extra k are nearly free. Each k dumps <out>_votes_k<K>.npz for offline replay. '
                         'The primary n_groups (12) keeps the json/csv/video outputs.')
    a = ap.parse_args()

    import torch
    dev = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    print('device:', dev, '| RAFT-', 'small' if a.small else 'large')
    raft, tf = load_raft(dev, small=a.small)
    dino = load_dino(dev)
    # LEFT frames only -- staged video_frames/ holds BOTH eyes (000000l.png + 000000r.png; the runbook
    # everywhere globs '*l.png'). A bare *.png doubles the count and breaks the depth pairing.
    files = sorted(glob.glob(a.frames_dir + '/*l.png'))
    if not files:                                       # raw-left / other stagings: fall back, drop rights
        files = sorted(glob.glob(a.frames_dir + '/*.png') + glob.glob(a.frames_dir + '/*.jpg'))
        files = [f for f in files if not os.path.basename(f).lower().endswith(('r.png', 'right.png'))]
    dfiles = sorted(glob.glob(a.depth_dir + '/*.png') + glob.glob(a.depth_dir + '/*.npy'))
    assert len(dfiles) >= len(files), f"depth({len(dfiles)}) < frames({len(files)})"
    GT = load_gt_tum(a.gt)
    gstep = np.linalg.norm(np.diff(GT, axis=0), axis=1) * 1000.0

    rows = []
    NG = a.n_groups
    dump = dict(frame=[], gt_mm=[], flow=[], depth=[], centroid=[], ok=[], resid=[], trust=[], sampson=[])
    vw = None   # --video writer, opened lazily on the first frame
    from Addons.motion.flow_track import _raft_flow
    k_extra = [int(k) for k in a.k_list.split(',') if k.strip()] if a.k_list else []
    kdump = {k: dict(frame=[], gt_mm=[], flow=[], depth=[], centroid=[], ok=[], resid=[], trust=[])
             for k in k_extra}
    for t in range(a.stride, len(files), a.every):
        cur = cv2.imread(files[t]); ref = cv2.imread(files[t - a.stride])
        depth = load_depth(dfiles[t - a.stride])                       # REF-frame depth (matches ref pixels)
        dg = dino_grid(cur, dino, dev)
        # OLD gate (identical inputs to the champion runs) + per-region Sampson detail
        cam, dis, samp = agreement_gate(ref, cur, dg, raft, tf, dev, n_groups=NG,
                                        ransac_thresh=1.0, deadband=a.deadband, return_detail=True)
        old_track = bool(cam > a.cam_thresh and dis <= a.disagree_thresh)
        # NEW vote. flow computed ONCE and shared across the k-sweep (depth = ref frame).
        _flow = _raft_flow(raft, tf, ref, cur, dev)
        info, w, lab = region_vote(ref, cur, depth, dg, raft, tf, dev,
                                   n_groups=NG, still_floor_px=a.still_floor_px, flow=_flow)
        for kk in k_extra:
            ik, _, _ = region_vote(ref, cur, depth, dg, raft, tf, dev,
                                   n_groups=kk, still_floor_px=a.still_floor_px, flow=_flow)
            kd = kdump[kk]
            kd['frame'].append(t); kd['gt_mm'].append(float(gstep[min(t, len(gstep) - 1)]))
            if ik is None:
                z2 = np.zeros((kk, 2)).tolist(); z1 = np.zeros(kk).tolist()
                kd['flow'].append(z2); kd['depth'].append(z1); kd['centroid'].append(z2)
                kd['ok'].append(np.zeros(kk, bool).tolist()); kd['resid'].append(z1); kd['trust'].append(np.ones(kk).tolist())
            else:
                kd['flow'].append(ik['region_flow']); kd['depth'].append(ik['region_depth'])
                kd['centroid'].append(ik['region_centroid']); kd['ok'].append(ik['region_ok'])
                kd['resid'].append(ik['region_resid']); kd['trust'].append(ik['region_trust'])
        if info is None:
            info = dict(moving=True, confidence=0.0, mag=0.0, turn=0.0, slide=0.0, zoom=0.0,
                        n_valid=0, n_inliers=0,                        # degenerate -> track (never freeze blind)
                        region_flow=np.zeros((NG, 2)).tolist(), region_depth=np.zeros(NG).tolist(),
                        region_centroid=np.zeros((NG, 2)).tolist(), region_ok=np.zeros(NG, bool).tolist(),
                        region_resid=np.zeros(NG).tolist(), region_trust=np.ones(NG).tolist())
        g = float(gstep[min(t, len(gstep) - 1)])
        rows.append(dict(frame=t, gt_mm=g, old_cam=cam, old_dis=dis, old_track=int(old_track),
                         vote_moving=int(info['moving']), vote_mag=info['mag'], vote_conf=info['confidence'],
                         turn=info['turn'], slide=info['slide'], zoom=info['zoom'],
                         n_inl=info['n_inliers'], n_val=info['n_valid']))
        # raw VOTES -> npz: replay ANY candidate rule offline (no GPU) against these frames
        dump['frame'].append(t); dump['gt_mm'].append(g)
        dump['flow'].append(info['region_flow']); dump['depth'].append(info['region_depth'])
        dump['centroid'].append(info['region_centroid']); dump['ok'].append(info['region_ok'])
        dump['resid'].append(info['region_resid']); dump['trust'].append(info['region_trust'])
        dump['sampson'].append(samp.tolist())
        print(f"f{t:4d} GT={g:.3f}mm | old cam={cam:5.2f} dis={dis:.2f} track={int(old_track)} | "
              f"vote mag={info['mag']:5.2f} turn={info['turn']:4.2f} slide={info['slide']:4.2f} "
              f"zoom={info['zoom']:4.2f} moving={int(info['moving'])} inl={info['n_inliers']}/{info['n_valid']}")
        # --- THE DEMOCRACY OVERLAY: district boundaries + per-district vote arrows on the real frame ---
        if a.video and 'region_flow' in info:
            vis = cur.copy()
            Hh, Ww = vis.shape[:2]
            edges = (cv2.Laplacian(lab.astype(np.float32), cv2.CV_32F) != 0)
            vis[edges] = (255, 255, 255)                                   # district boundaries
            rf = np.asarray(info['region_flow']); rc = np.asarray(info['region_centroid'])
            rt = np.asarray(info['region_trust']); rok = np.asarray(info['region_ok'])
            for k in range(NG):
                if not rok[k]: continue
                x, y = int(rc[k, 0]), int(rc[k, 1])
                u, v = rf[k]; m = float(np.hypot(u, v))
                L = min(20 + m * 6, 120)                                   # arrow length ~ vote size
                ex, ey = (int(x + u / m * L), int(y + v / m * L)) if m > 1e-6 else (x, y)
                c = (int(60 + 195 * (1 - rt[k])), int(60 + 195 * rt[k]), 40)   # BGR: green=trusted, red=dissident
                cv2.arrowedLine(vis, (x, y), (ex, ey), c, 3, tipLength=0.3)
                cv2.putText(vis, f"{m:.1f}", (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
            gt_moving = g >= a.gt_still_mm
            banner = (f"f{t}  GT:{'MOVING' if gt_moving else 'STILL'} {g:.2f}mm | vote q10-era mag={info['mag']:.1f} "
                      f"turn={info['turn']:.1f} slide={info['slide']:.1f} zoom={info['zoom']:.1f} "
                      f"-> {'MOVING' if info['moving'] else 'STILL'} | old cam={cam:.1f} dis={dis:.2f}")
            col = (0, 200, 0) if (bool(info['moving']) == bool(gt_moving)) else (0, 0, 255)
            cv2.rectangle(vis, (0, 0), (Ww, 26), (0, 0, 0), -1)
            cv2.putText(vis, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            if vw is None:
                vw = cv2.VideoWriter(a.out + '_votes.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (Ww, Hh))
            vw.write(vis)

    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    gt_moving = R['gt_mm'] >= a.gt_still_mm
    res = dict(name=os.path.basename(a.out), n=len(rows), every=a.every, stride=a.stride,
               still_floor_px=a.still_floor_px,
               old=confusion(R['old_track'].astype(bool), gt_moving),
               vote=confusion(R['vote_moving'].astype(bool), gt_moving))
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out + '.json', 'w'), indent=2)
    import csv as _csv
    with open(a.out + '.csv', 'w', newline='') as fh:
        wcsv = _csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wcsv.writeheader(); wcsv.writerows(rows)
    np.savez_compressed(a.out + '_votes.npz', stride=a.stride, every=a.every,
                        wh=np.array(cur.shape[:2][::-1]),               # [W,H] for centroid normalisation
                        **{k: np.asarray(v) for k, v in dump.items()})
    for kk, kd in kdump.items():
        np.savez_compressed(a.out + f'_votes_k{kk}.npz', stride=a.stride, every=a.every,
                            wh=np.array(cur.shape[:2][::-1]),
                            **{k: np.asarray(v) for k, v in kd.items()})
        print('wrote', a.out + f'_votes_k{kk}.npz')
    if vw is not None:
        vw.release(); print('wrote', a.out + '_votes.mp4')

    # ---- plot: GT + vote components + decisions (old vs new) ----
    T = R['frame']; still = ~gt_moving
    fig, ax = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for axx in ax:
        axx.fill_between(T, 0, 1, where=still, transform=axx.get_xaxis_transform(), color='0.88')
    ax[0].plot(T, R['gt_mm'], 'k', lw=2); ax[0].set_ylabel('GT mm/f'); ax[0].set_title('GT camera motion (grey = GT still)')
    ax[1].plot(T, R['old_cam'], 'b', lw=1.2, label='old cam_mag (px)')
    ax[1].axhline(a.cam_thresh, ls='--', c='b', lw=1)
    axb = ax[1].twinx(); axb.plot(T, R['old_dis'], 'r', lw=1.2, label='old disagree'); axb.axhline(a.disagree_thresh, ls='--', c='r', lw=1); axb.set_ylim(0, 1)
    ax[1].set_ylabel('cam_mag', color='b'); axb.set_ylabel('disagree', color='r'); ax[1].set_title('OLD gate inputs (raw px, hardcoded thresholds)')
    ax[2].plot(T, R['vote_mag'], 'purple', lw=1.6, label='vote consensus mag (px @ median plane)')
    ax[2].plot(T, R['turn'], 'C0', lw=1, alpha=.8, label='turn')
    ax[2].plot(T, R['slide'], 'C2', lw=1, alpha=.8, label='slide')
    ax[2].plot(T, R['zoom'], 'C3', lw=1, alpha=.8, label='zoom')
    ax[2].axhline(a.still_floor_px, ls='--', c='purple', lw=1, label=f'still floor {a.still_floor_px}px')
    ax[2].set_ylabel('px @ median plane'); ax[2].legend(fontsize=8, ncol=5); ax[2].set_title('VOTE: consensus magnitude + turn/slide/zoom readout')
    ax[3].plot(T, R['old_track'], drawstyle='steps-mid', color='b', lw=1.2, alpha=.7, label=f"old gate ({res['old']['freeze_precision']}% frz-prec)")
    ax[3].plot(T, R['vote_moving'], drawstyle='steps-mid', color='purple', lw=1.8, label=f"vote ({res['vote']['freeze_precision']}% frz-prec)")
    ax[3].plot(T, gt_moving.astype(float), 'k', lw=1, alpha=.6, label='GT moving')
    ax[3].set_ylim(-0.1, 1.1); ax[3].set_ylabel('1=TRACK/MOVING'); ax[3].set_xlabel('frame'); ax[3].legend(fontsize=8)
    ax[3].set_title(f"decisions -- GT agreement: old {res['old']['gt_agree']}% vs VOTE {res['vote']['gt_agree']}%")
    fig.suptitle(f"{os.path.basename(a.out)} vote-scan (stride {a.stride}, every {a.every})", fontsize=13)
    fig.tight_layout(); fig.savefig(a.out + '.png', dpi=130, bbox_inches='tight')
    print('wrote', a.out + '.png/.json/.csv')


if __name__ == '__main__':
    main()
