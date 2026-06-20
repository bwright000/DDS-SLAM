#!/usr/bin/env python3
"""GT-timing check: does the flow's predicted CAMERA motion track the real GT camera motion over time?

Validates the camera/scene split (feature_flow_probe) against GT poses by TIMING, not magnitude
(monocular flow is up-to-scale; the question is whether the PROFILES line up). For each pair we have:
  cam_mag  = |median flow| (the dominant rigid motion = camera proxy; deformation is a minority so the
             MEDIAN sits on the static/camera majority)
  resid    = mean Sampson residual (the SCENE-motion the camera can't explain)
GT camera motion per pair -> predicted camera pixel-flow gt_camflow = fx * rot_rad (rotation dominates
the surgical pixel flow; translation needs depth so it's reported but not the headline).

PREDICTION (the validation):
  Pearson(cam_mag, gt_camflow)  HIGH  -> the flow's camera estimate tracks the real camera motion's timing.
  Pearson(resid,   gt_camflow)  LOW   -> the scene residual is orthogonal to camera motion (as it should be).

  python Addons/motion/gt_timing.py --pairs_csv output/flowprobe_C_1_001_k12/feature_flow_pairs.csv \
     --gt data/CRCD/C1_001/groundtruth.txt --fx 1096.7 --out_fig output/flowprobe_C_1_001_k12/gt_timing.png
  python Addons/motion/gt_timing.py --selftest
"""
import os, sys, csv, argparse, numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def load_gt(path):
    P = []
    for ln in open(path):
        if ln.startswith('#') or not ln.strip():
            continue
        v = ln.split()
        try:
            P.append(list(map(float, v[1:8])))   # tx ty tz qx qy qz qw
        except Exception:
            pass
    P = np.array(P, float)
    q = P[:, 3:7] / np.linalg.norm(P[:, 3:7], axis=1, keepdims=True)
    return P[:, :3], q


def gt_pair_motion(t, q, a, b):
    """trans (mm) + rotation (deg) between GT rows a and b."""
    a = min(a, len(t) - 1); b = min(b, len(t) - 1)
    dt = np.linalg.norm(t[b] - t[a]) * 1000.0
    dr = np.degrees(2 * np.arccos(min(1.0, abs(float(np.dot(q[a], q[b]))))))
    return dt, dr


def run_selftest():
    rng = np.random.RandomState(0); n = 120
    gt = np.abs(rng.randn(n)) * 5                      # GT camera pixel-flow profile (>=0)
    cam = gt + rng.randn(n) * 0.5                      # camera proxy tracks GT
    scene = np.random.RandomState(1).randn(n) * 5      # scene residual: INDEPENDENT of GT (zero-mean)
    rc, rs = pearson(cam, gt), pearson(scene, gt)
    ok = rc > 0.9 and abs(rs) < 0.3
    print(f"SELFTEST Pearson(cam,gt)={rc:.2f} (>0.9) Pearson(scene,gt)={rs:.2f} (~0) -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pairs_csv'); ap.add_argument('--gt')
    ap.add_argument('--fx', type=float, default=1096.7)
    ap.add_argument('--out_fig', default='')
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())

    rows = list(csv.DictReader(open(args.pairs_csv)))
    t, q = load_gt(args.gt)
    fa = np.array([int(r['frame_a']) for r in rows]); fb = np.array([int(r['frame_b']) for r in rows])
    cam_mag = np.array([float(r['cam_mag']) for r in rows])
    flow_mag = np.array([float(r['flow_mag']) for r in rows])
    resid = np.array([float(r['resid_mean']) for r in rows])
    gtt = np.array([gt_pair_motion(t, q, a, b) for a, b in zip(fa, fb)])        # [N,2] mm,deg
    gt_trans, gt_rot = gtt[:, 0], gtt[:, 1]
    gt_camflow = args.fx * np.radians(gt_rot)                                    # predicted camera px-flow (rotation)
    held = (gt_trans == 0) & (gt_rot == 0)                                       # exactly-0 GT = held/dropped rows

    print(f"[gt_timing] {len(rows)} pairs | GT {len(t)} poses | held(exactly-0 GT) {held.mean():.0%}")
    print(f"  GT camera motion: rot med {np.median(gt_rot):.2f}deg max {gt_rot.max():.2f} | trans med {np.median(gt_trans):.2f}mm max {gt_trans.max():.2f}")
    print(f"\n=== TIMING correlations (whole sequence) ===")
    print(f"  Pearson(cam_mag, gt_camflow)  = {pearson(cam_mag, gt_camflow):+.3f}   <- camera proxy vs GT camera (want HIGH)")
    print(f"  Pearson(flow_mag, gt_camflow) = {pearson(flow_mag, gt_camflow):+.3f}   (mean flow; scene-contaminated)")
    print(f"  Pearson(resid,   gt_camflow)  = {pearson(resid, gt_camflow):+.3f}   <- scene residual vs GT camera (want LOW)")
    nz = ~held
    rc = pearson(cam_mag, gt_camflow)                                            # whole-sequence (diluted by held GT)
    if nz.sum() >= 4:
        rc = pearson(cam_mag[nz], gt_camflow[nz])                                # HEADLINE: GT-reliable pairs only
        print(f"\n=== on GT-RELIABLE pairs only (exclude exactly-0 GT, n={nz.sum()}) — THE headline ===")
        print(f"  Pearson(cam_mag, gt_camflow)  = {rc:+.3f}   <- the camera-timing number that matters")
        print(f"  Pearson(resid,   gt_camflow)  = {pearson(resid[nz], gt_camflow[nz]):+.3f}")
    else:
        print(f"\n  (only {int(nz.sum())} GT-reliable pairs — too few to isolate; using whole-sequence)")
    print(f"\n  VERDICT: GT held/dropout {held.mean():.0%}. "
          f"{'TIMING MATCHES — the flow camera estimate tracks GT camera motion on the reliable pairs.' if rc > 0.5 else 'WEAK match — inspect the overlay (the held GT limits the test).'}")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        x = fa
        fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        def nrm(z): z = np.asarray(z, float); return (z - z.min()) / (np.ptp(z) + 1e-9)
        ax[0].plot(x, nrm(cam_mag), '-o', ms=3, label='cam_mag (flow camera proxy)')
        ax[0].plot(x, nrm(gt_camflow), '-s', ms=3, label='GT camera (fx*rot)')
        ax[0].plot(x, nrm(flow_mag), ':', alpha=.5, label='mean flow (scene-contaminated)')
        ax[0].scatter(x[held], np.zeros(held.sum()), c='k', marker='|', s=80, label='held/0 GT')
        ax[0].set_title(f'CAMERA timing — Pearson(cam_mag,GT)={rc:+.2f}'); ax[0].legend(fontsize=8); ax[0].set_ylabel('normalised')
        ax[1].plot(x, nrm(resid), '-o', ms=3, c='tab:red', label='scene residual (Sampson)')
        ax[1].plot(x, nrm(gt_camflow), '-s', ms=3, c='tab:green', alpha=.6, label='GT camera')
        ax[1].set_title(f'SCENE residual vs GT camera — Pearson={pearson(resid,gt_camflow):+.2f} (want ~0)')
        ax[1].legend(fontsize=8); ax[1].set_xlabel('frame'); ax[1].set_ylabel('normalised')
        out = args.out_fig or os.path.splitext(args.pairs_csv)[0] + '_gt_timing.png'
        plt.tight_layout(); plt.savefig(out, dpi=90); print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"\n(no figure: {e})")


if __name__ == '__main__':
    main()
