#!/usr/bin/env python3
"""Gate PREVIEW (causal): run the flow_agree gate's EXACT per-frame decision in the CAUSAL direction
(ref = t-ref_stride -> cur = t), WITHOUT the SLAM. Shows which frames the SLAM gate will TRACK (camera:
features agree on one rigid motion) vs FIX (still, or scene: features disagree). Validates the causal
direction + the thresholds cheaply before a SLAM run. Reuses the IDENTICAL agreement_gate the SLAM uses.

  python Addons/motion/gate_preview.py --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png' \
    --out_dir output/gate_preview_c1 --ref_stride 8 --cam_thresh 2.0 --disagree_thresh 0.2 --n_groups 12
"""
import os, sys, glob, argparse, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root -> import Addons.*
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rgb_dir', required=True); ap.add_argument('--rgb_glob', default='*l.png')
    ap.add_argument('--out_dir', default='output/gate_preview')
    ap.add_argument('--ref_stride', type=int, default=8)
    ap.add_argument('--cam_thresh', type=float, default=2.0)
    ap.add_argument('--disagree_thresh', type=float, default=0.2)
    ap.add_argument('--deadband', type=float, default=3.0)
    ap.add_argument('--n_groups', type=int, default=12)
    ap.add_argument('--ransac_thresh', type=float, default=1.0)
    ap.add_argument('--raft_small', action='store_true')
    ap.add_argument('--downscale', type=float, default=1.0)
    ap.add_argument('--stride', type=int, default=1, help='evaluate every Nth current-frame t')
    ap.add_argument('--max_frames', type=int, default=400)
    args = ap.parse_args()

    import cv2, torch
    from Addons.motion.flow_track import load_raft, load_dino, dino_grid, agreement_gate
    os.makedirs(args.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    R = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    assert len(R) > args.ref_stride, f'need > ref_stride frames (got {len(R)})'
    raft, tf = load_raft(dev, args.raft_small); dino = load_dino(dev)
    print(f"[gate_preview] {len(R)} frames | CAUSAL t-{args.ref_stride}->t | dev {dev} | "
          f"cam_thresh {args.cam_thresh} disagree_thresh {args.disagree_thresh} deadband {args.deadband} K={args.n_groups}")

    def rd(i):
        im = cv2.imread(R[i])
        return cv2.resize(im, None, fx=args.downscale, fy=args.downscale) if args.downscale != 1.0 else im

    rows = []
    ts = list(range(args.ref_stride, len(R), args.stride))[:args.max_frames]
    for t in ts:
        a, b = rd(t - args.ref_stride), rd(t)            # CAUSAL: ref = past (t-stride), cur = present (t)
        cam_mag, dis = agreement_gate(a, b, dino_grid(b, dino, dev), raft, tf, dev,
                                      n_groups=args.n_groups, ransac_thresh=args.ransac_thresh,
                                      deadband=args.deadband)
        track = (cam_mag > args.cam_thresh) and (dis <= args.disagree_thresh)
        reason = 'TRACK(camera)' if track else ('FIX(still)' if cam_mag <= args.cam_thresh else 'FIX(scene)')
        rows.append((t, round(cam_mag, 3), round(dis, 3), int(track), reason))
        if len(rows) % 20 == 1 or t == ts[-1]:
            print(f"  f{t:04d} cam_mag {cam_mag:5.2f} disagree {dis:4.2f} -> {reason}")

    n = len(rows); ntr = sum(r[3] for r in rows)
    nstill = sum(1 for r in rows if r[4] == 'FIX(still)'); nscene = sum(1 for r in rows if r[4] == 'FIX(scene)')
    print(f"\n=== {n} frames: TRACK {ntr} ({100*ntr/max(n,1):.0f}%) | FIX-still {nstill} | FIX-scene {nscene} ===")
    with open(os.path.join(args.out_dir, 'gate_preview.csv'), 'w', newline='') as fp:
        w = csv.writer(fp); w.writerow(['frame', 'cam_mag', 'disagree_frac', 'track', 'reason']); w.writerows(rows)

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fr = [r[0] for r in rows]; cm = [r[1] for r in rows]; dg = [r[2] for r in rows]
        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax[0].plot(fr, cm, '-o', ms=3); ax[0].axhline(args.cam_thresh, c='k', ls='--', lw=.7)
        ax[0].set_ylabel('cam_mag'); ax[0].set_title('camera motion |median flow|  (dashed = cam_thresh; >thresh = moving)')
        ax[1].plot(fr, dg, '-o', ms=3, c='tab:red'); ax[1].axhline(args.disagree_thresh, c='k', ls='--', lw=.7)
        ax[1].set_ylabel('disagree_frac'); ax[1].set_xlabel('frame')
        ax[1].set_title('per-region disagreement  (dashed = disagree_thresh; >thresh = scene)')
        for r in rows:                                   # shade FIX frames (gray=still, orange=scene)
            if not r[3]:
                c = 'gray' if r[4] == 'FIX(still)' else 'tab:orange'
                for a_ in ax:
                    a_.axvspan(r[0] - 0.5, r[0] + 0.5, color=c, alpha=0.18)
        out = os.path.join(args.out_dir, 'gate_preview.png'); plt.tight_layout(); plt.savefig(out, dpi=90)
        print(f"figure -> {out}  (green=TRACK, gray=FIX-still, orange=FIX-scene)")
    except Exception as e:
        print(f"(no figure: {e})")
    print(f"csv -> {os.path.join(args.out_dir, 'gate_preview.csv')}")


if __name__ == '__main__':
    main()
