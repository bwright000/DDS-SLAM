#!/usr/bin/env python3
"""Per-run flow diagnostics -- the two checks that decide whether a flow-supervisor arm works:
  D-1 CAMERA-ACTIVATION TIMING: does the estimated camera move WHEN GT moves and hold still WHEN GT is still?
      Spearman rho(est per-frame step, GT per-frame step) over Sim3-aligned trajectories + a moving/still ratio.
      PASS: rho >= 0.5 AND moving/still ratio >= 2.0  (lag-tolerant; ranks -> scale-free, robust on sub-SNR GT).
  D-2 OVER-TRAVEL / JITTER: does the est stop exaggerating motion? path-ratio (est*s / GT) toward 1, and the est
      goes quiet when the camera stops.  PASS: path-ratio in [0.7,1.4] AND est-still-step <= 0.5*est-moving-step
      (GT-still ~= 0 so the jitter is gated est-still vs est-moving, not vs GT).
Works for ANY arm (needs only est_c2w_data.txt + groundtruth.txt); optionally overlays the solved motion from
trust_log.csv (solve_pnp t_norm). Writes <out>.json (pass flags + numbers) and <plot>.png.
  python Addons/eval/flow_diag.py --est RUN/est_c2w_data.txt --gt DD/groundtruth.txt [--trust OUT/trust_log.csv] \
      --name CELL --out DST/flow_diag.json --plot DST/flow_diag.png
"""
import argparse, json, os
import numpy as np


def load_est(p):
    return np.loadtxt(p)[:, [3, 7, 11]]


def load_est_rot(p):
    d = np.loadtxt(p)
    if d.ndim != 2 or d.shape[1] < 12:
        return None
    return d[:, :12].reshape(-1, 3, 4)[:, :, :3]


def load_gt(p):
    d = np.loadtxt(p, comments='#')
    return (d[:, 1:4] if d.ndim == 2 else d[None, 1:4])


def load_gt_quat(p):
    d = np.loadtxt(p, comments='#')
    if d.ndim != 2 or d.shape[1] < 8:
        return None
    q = d[:, 4:8]
    return q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)


def rot_steps_est(Rm):
    # relative-rotation angle per frame pair (deg); gauge-free, so NO Sim3 alignment is involved
    rel = np.einsum('nij,nik->njk', Rm[:-1], Rm[1:])   # R_i^T @ R_{i+1}
    tr = np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def rot_steps_gt(q):
    dot = np.clip(np.abs((q[:-1] * q[1:]).sum(1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); S, D = src - ms, dst - md
    C = D.T @ S / len(src); U, Dd, Vt = np.linalg.svd(C); E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt; s = np.trace(np.diag(Dd) @ E) / ((S ** 2).sum() / len(src))
    return s, R, (md - s * R @ ms)


def _rank(x):
    # AVERAGE ranks for ties: CRCD GT is 44-58% bit-identical frames (step exactly 0); arbitrary distinct
    # ranks on those ties made rho index-order-dependent. Average-rank = the standard Spearman treatment.
    o = np.argsort(x, kind='mergesort'); r = np.empty(len(x)); r[o] = np.arange(len(x))
    xs = np.asarray(x)[o]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


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
    if len(est) != len(gt):
        print(f"[flow_diag] WARN: length mismatch est={len(est)} gt={len(gt)} -> head-truncating to min; "
              f"if the difference is not a trailing crop the D1/D2 verdicts are judging MISPAIRED frames")
    n = min(len(est), len(gt)); est, gt = est[:n], gt[:n]
    s, R, t = umeyama(est, gt); ea = (s * (R @ est.T).T + t)
    es = np.linalg.norm(np.diff(ea, axis=0), axis=1) * 1000.0   # est per-frame step (mm, aligned)
    gs = np.linalg.norm(np.diff(gt, axis=0), axis=1) * 1000.0   # GT per-frame step (mm)
    # D-1 activation timing
    rho = abs(spearman(es, gs))
    # still/moving split: median floor at 0.1um -- CRCD GT is >50% bit-identical (step exactly 0), where
    # thr=median=0 made the split degenerate (mv = every nonzero frame incl. sub-noise ones). With the
    # floor, exact-still frames (gs<=eps) are 'still' and the split survives majority-static sequences.
    thr = max(float(np.median(gs)), 1e-4); mv, st = gs > thr, gs <= thr
    r_act = float(np.median(es[mv]) / (np.median(es[st]) + 1e-9)) if mv.any() and st.any() else 0.0
    d1 = bool(rho >= 0.5 and r_act >= 2.0)
    # D-2 over-travel: path-ratio is the robust (scale-corrected) disease measure; est still-frame jitter (mm)
    # is reported as INFO (a ratio to GT-still is ill-defined here since GT-still motion ~= 0).
    pl = lambda x: float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    pathr = pl(est) * s / (pl(gt) + 1e-12)
    est_still_mm = float(np.median(es[st])) if st.any() else 0.0
    gt_still_mm = float(np.median(gs[st])) if st.any() else 0.0
    est_moving_mm = float(np.median(es[mv])) if mv.any() else 0.0
    # D2 = over-travel (path-ratio) AND a still-frame JITTER guard: the est must be >=2x quieter on GT-still than
    # on GT-moving frames (est_still <= 0.5*est_moving). NB a ratio to GT-still is ill-defined (GT-still ~= 0), so
    # we gate est_still-vs-est_moving -- the meaningful "does the camera go quiet when it actually stops".
    d2 = bool(0.7 <= pathr <= 1.4 and est_still_mm <= 0.5 * est_moving_mm)
    # FREEZE CONFUSION (the metric that vindicated the C1 gate): frozen frame = est step ~ 0 (pose
    # copied). precision = P(GT-still | frozen)  [C1 old gate: 93.4% = right freezes; E3: 28.2% =
    # inverted]. recall = coverage of the GT-still frames. Zeroes for arms that never freeze.
    frz = es < 1e-6
    gstill = gs <= 1e-4
    n_frz = int(frz.sum())
    frz_prec = round(100.0 * float((frz & gstill).sum()) / n_frz, 1) if n_frz else None
    frz_rec = round(100.0 * float((frz & gstill).sum()) / max(int(gstill.sum()), 1), 1) if n_frz else None
    # ROTATION CHANNEL (report-only, no pass bar yet): the whole battery above reads camera POSITIONS,
    # but this RCM platform is rotation-dominant (~5.7x the image flow on E3). Relative-rotation steps
    # are gauge-free (no Sim3), and rotation has no scale ambiguity -> rot_path_ratio is an ABSOLUTE
    # over/under-travel measure, unlike the translation path-ratio which rides on the Sim3 scale.
    rot = {}
    Rm, gq = load_est_rot(a.est), load_gt_quat(a.gt)
    if Rm is not None and gq is not None:
        er, gr = rot_steps_est(Rm[:n]), rot_steps_gt(gq[:n])
        rrho = abs(spearman(er, gr))
        rthr = max(float(np.median(gr)), 1e-4); rmv, rst = gr > rthr, gr <= rthr
        r_ract = float(np.median(er[rmv]) / (np.median(er[rst]) + 1e-9)) if rmv.any() and rst.any() else 0.0
        rot = dict(rot_rho=round(rrho, 3), rot_path_ratio=round(float(er.sum() / (gr.sum() + 1e-12)), 2),
                   rot_moving_still_ratio=round(r_ract, 2),
                   est_rot_still_deg=round(float(np.median(er[rst])) if rst.any() else 0.0, 4),
                   est_rot_moving_deg=round(float(np.median(er[rmv])) if rmv.any() else 0.0, 4))
    res = dict(name=a.name, n=int(n), sim3_scale=round(float(s), 4), activation_rho=round(rho, 3),
               moving_still_ratio=round(r_act, 2), path_ratio=round(float(pathr), 2),
               est_still_step_mm=round(est_still_mm, 3), est_moving_step_mm=round(est_moving_mm, 3),
               gt_still_step_mm=round(gt_still_mm, 3), n_frozen=n_frz,
               freeze_precision=frz_prec, freeze_still_recall=frz_rec,
               D1_timing_pass=d1, D2_overtravel_pass=d2,
               flow_ok=bool(d1 and d2), **rot)
    print(f"[flow_diag] {a.name}: activation rho={rho:.2f} moving/still={r_act:.1f} (D1 {'PASS' if d1 else 'FAIL'}) | "
          f"path-ratio={pathr:.2f} still/moving jitter={est_still_mm:.3f}/{est_moving_mm:.3f}mm (D2 {'PASS' if d2 else 'FAIL'})"
          + (f" | frozen={n_frz} prec={frz_prec}% recall={frz_rec}%" if n_frz else "")
          + (f" | ROT rho={rot['rot_rho']:.2f} path={rot['rot_path_ratio']:.2f} mv/st={rot['rot_moving_still_ratio']:.1f}"
             if rot else ""))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=2)
    if a.plot:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
            ax[0].plot(gs, 'k-', lw=2, label='GT step'); ax[0].plot(es, 'C1-', lw=1, alpha=.8, label='est step (aligned)')
            ax[0].set_ylabel('per-frame step (mm)'); ax[0].legend(fontsize=8)
            ax[0].set_title(f"{a.name}  D1 timing rho={rho:.2f} mv/st={r_act:.1f} [{'PASS' if d1 else 'FAIL'}]  |  "
                            f"D2 path-ratio={pathr:.2f} still-jitter={est_still_mm:.2f}mm [{'PASS' if d2 else 'FAIL'}]")
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
