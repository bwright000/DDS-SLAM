#!/usr/bin/env python3
"""Per-run flow diagnostics -- the two checks that decide whether a flow-supervisor arm works:
  D-1 CAMERA-ACTIVATION TIMING: does the estimated camera move WHEN GT moves and hold still WHEN GT is still?
      Spearman rho(est per-frame step, GT per-frame step) over Sim3-aligned trajectories + a moving/still ratio.
      PASS: rho >= 0.5 AND moving/still ratio >= 2.0  (lag-tolerant; ranks -> scale-free, robust on sub-SNR GT).
  D-2 OVER-TRAVEL / JITTER: does the est stop exaggerating motion? path-ratio (est*s / GT) toward 1, and no
      phantom motion on GT-still frames.  PASS: path-ratio in [0.7,1.4] AND still-frame est step <= 3x GT-still.
Works for ANY arm (needs only est_c2w_data.txt + groundtruth.txt); optionally overlays the solved motion from
trust_log.csv (solve_pnp t_norm). Writes <out>.json (pass flags + numbers) and <plot>.png.
  python Addons/eval/flow_diag.py --est RUN/est_c2w_data.txt --gt DD/groundtruth.txt [--trust OUT/trust_log.csv] \
      --name CELL --out DST/flow_diag.json --plot DST/flow_diag.png
"""
import argparse, json, os
import numpy as np


def load_est(p):
    return np.loadtxt(p)[:, [3, 7, 11]]


def load_gt(p):
    d = np.loadtxt(p, comments='#')
    return (d[:, 1:4] if d.ndim == 2 else d[None, 1:4])


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
    C = D.T @ S / len(src); U, Dd, Vt = np.linalg.svd(C); E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt; s = np.trace(np.diag(Dd) @ E) / ((S ** 2).sum() / len(src))
    return s, R, (md - s * R @ ms)


def _rank(x):
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() < 1e-9 or rb.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', required=True); ap.add_argument('--gt', required=True)
    ap.add_argument('--trust', default=None); ap.add_argument('--name', default='')
    ap.add_argument('--out', default=None); ap.add_argument('--plot', default=None)
    a = ap.parse_args()
    est, gt = load_est(a.est), load_gt(a.gt)
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    s, R, t = umeyama(est, gt); ea = (s * (R @ est.T).T + t)
    es = np.linalg.norm(np.diff(ea, axis=0), axis=1) * 1000.0   # est per-frame step (mm, aligned)
    gs = np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000.0   # GT per-frame step (mm)
    # D-1 activation timing
    rho = abs(spearman(es, gs))
    thr = np.median(gs); mv, st = gs > thr, gs <= thr
    r_act = float(np.median(es[mv]) / (np.median(es[st]) + 1e-9)) if mv.any() and st.any() else 0.0
    d1 = bool(rho >= 0.5 and r_act >= 2.0)
    # D-2 over-travel: path-ratio is the robust (scale-corrected) disease measure; est still-frame jitter (mm)
    # is reported as INFO (a ratio to GT-still is ill-defined here since GT-still motion ~= 0).
    pl = lambda x: float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    pathr = pl(est) * s / (pl(gt) + 1e-12)
    est_still_mm = float(np.median(es[st])) if st.any() else 0.0
    gt_still_mm = float(np.median(gs[st])) if st.any() else 0.0
    d2 = bool(0.7 <= pathr <= 1.4)
    res = dict(name=a.name, n=int(n), sim3_scale=round(float(s), 4), activation_rho=round(rho, 3),
               moving_still_ratio=round(r_act, 2), path_ratio=round(float(pathr), 2),
               est_still_step_mm=round(est_still_mm, 3), gt_still_step_mm=round(gt_still_mm, 3),
               D1_timing_pass=d1, D2_overtravel_pass=d2, flow_ok=bool(d1 and d2))
    print(f"[flow_diag] {a.name}: activation rho={rho:.2f} moving/still={r_act:.1f} (D1 {'PASS' if d1 else 'FAIL'}) | "
          f"path-ratio={pathr:.2f} est-still-jitter={est_still_mm:.3f}mm (GT-still {gt_still_mm:.3f}mm) (D2 {'PASS' if d2 else 'FAIL'})")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=2)
    if a.plot:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ax[0].plot(gs, 'k-', lw=2, label='GT step'); ax[0].plot(es, 'C1-', lw=1, alpha=.8, label='est step (aligned)')
            ax[0].set_ylabel('per-frame step (mm)'); ax[0].legend(fontsize=8)
            ax[0].set_title(f"{a.name}  D1 timing rho={rho:.2f} mv/st={r_act:.1f} [{'PASS' if d1 else 'FAIL'}]  |  "
                            f"D2 path-ratio={pathr:.2f} phantom={phantom:.1f}x [{'PASS' if d2 else 'FAIL'}]")
            if a.trust and os.path.exists(a.trust):
                import csv
                rows = list(csv.DictReader(open(a.trust)))
                if rows and 't_norm' in rows[0]:
                    fr = [int(r['frame']) for r in rows if r.get('t_norm', '') not in ('', None)]
                    tn = [float(r['t_norm']) for r in rows if r.get('t_norm', '') not in ('', None)]
                    if fr:
                        ax[1].plot(fr, tn, 'C2-', lw=1, label='solved |t| (PnP, up-to-scale)')
            ax[1].plot(gs / (np.median(gs) + 1e-9), 'k-', lw=1, alpha=.5, label='GT step (norm)')
            ax[1].set_xlabel('frame'); ax[1].set_ylabel('solved/GT motion'); ax[1].legend(fontsize=8)
            plt.tight_layout(); plt.savefig(a.plot, dpi=85)
            print(f"[flow_diag] saved {a.plot}")
        except Exception as e:
            print(f"[flow_diag] plot skipped: {e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
