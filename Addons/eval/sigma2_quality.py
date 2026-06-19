#!/usr/bin/env python3
"""Arm-1 σ²-quality judge — the DIRECT uncertainty metric Arm-1 lacked (mirrors Arm-2's pin-EPE).

Until now Arm-1's only judge was the downstream Sim3 ATE (indirect + sub-SNR). This scores σ² DIRECTLY,
on the two things σ² is USED for — one metric per use:

  (1) MAPPING competence (Inc-1) — the map is down-weighted by σ², so σ² SHOULD rank pixels by RENDER
      ERROR. Metric = SPARSIFICATION / AUSE: progressively drop the highest-σ² pixels; does the remaining
      error fall as fast as dropping by TRUE error (the oracle)? AUSE = normalised area between the
      σ²-ranked curve and the oracle curve. We also report frac_oracle = 1 − AUSE/AUSE_random ∈ (−∞,1]:
      1 = perfectly calibrated, 0 = no better than random, <0 = worse than random (anti-calibrated).

  (2) TRACKING correctness (Inc-2) — the pose is down-weighted by σ², and a pixel is pose-harmful when it
      MOVES, regardless of its render error. Metric = Pearson(σ², inter-frame motion). (Partner analysis
      A lives in sigma2_diagnostics.py; reproduced here so the judge is self-contained.)

🚨 THE TWO PULL APART. Deformation is appearance-preserving (tissue moves, colour ~unchanged) → LOW render
error → a σ² perfectly calibrated to render error (great AUSE) is LOW on deforming tissue (BAD tracking).
That tension is exactly why one photometric σ² cannot serve both, and why the COMBINE (Arm-2 motion → σ²)
is the real fix. This judge MEASURES the tension so we can read the teacher A/B (l2 vs depth vs rgb_depth)
on σ² QUALITY directly, instead of only through the sub-SNR ATE.

NUMERICAL (AUSE + frac_oracle + correlations → JSON) + VISUAL (sparsification curves + σ²-vs-motion) per
the two-diagnostic-sets rule. Pure numpy + cv2 (matplotlib optional).

  # rgb-error calibration (l2 cells) + deformation-awareness:
  python Addons/eval/sigma2_quality.py --name geo_rd_crcd_s0 \
    --uncert_dir output/.../uncert --render_dir output/... --render_glob '[0-9]*.jpg' \
    --rgb_dir data/CRCD/C1_001/video_frames --rgb_glob '*l.png'
  # depth-error calibration (depth-teacher cells), if rendered depth is dumped:
  python Addons/eval/sigma2_quality.py --name geo_rd_crcd_s0 --error_mode depth \
    --uncert_dir .../uncert --render_dir .../depth --gt_depth_dir data/.../depth/moge2 ...
  # verify the AUSE/Pearson math, no data needed:
  python Addons/eval/sigma2_quality.py --selftest
"""
import os, sys, glob, json, argparse, numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')   # σ² glyphs -> don't crash a cp1252 Windows console
except Exception:
    pass

try:
    import cv2
except Exception:
    cv2 = None

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))   # numpy 2.x renamed trapz -> trapezoid


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def sparsification(err, unc, n_steps=40, seed=0):
    """Drop the top-fraction by each ranking, return remaining-RMSE curves.
    err, unc: 1-D, same length, err>=0. Returns (fracs, c_unc, c_oracle, c_random)."""
    err = err.ravel().astype(np.float64); unc = unc.ravel().astype(np.float64)
    n = len(err)
    o_unc = np.argsort(-unc)                       # remove highest-σ² first
    o_or = np.argsort(-err)                        # oracle: remove highest-error first
    o_rand = np.random.RandomState(seed).permutation(n)
    fracs = np.linspace(0.0, 0.95, n_steps)

    def curve(order):
        out = np.empty(len(fracs))
        for j, f in enumerate(fracs):
            k = int(f * n)
            keep = order[k:]
            out[j] = np.sqrt(np.mean(err[keep] ** 2)) if len(keep) else 0.0
        return out

    return fracs, curve(o_unc), curve(o_or), curve(o_rand)


def ause(fracs, c_unc, c_oracle, c_random):
    """Normalise each curve to start at 1.0; AUSE = area between σ²-curve and oracle.
    frac_oracle = 1 − AUSE/AUSE_random : 1=perfect, 0=random, <0=anti-calibrated."""
    def norm(c):
        return c / (c[0] + 1e-12)
    n_unc, n_or, n_rand = norm(c_unc), norm(c_oracle), norm(c_random)
    a_unc = float(_trapz(np.abs(n_unc - n_or), fracs))
    a_rand = float(_trapz(np.abs(n_rand - n_or), fracs))
    frac = 1.0 - a_unc / a_rand if a_rand > 1e-9 else 0.0
    return a_unc, a_rand, frac, (n_unc, n_or, n_rand)


def _read_gray(path, shape_hw=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[..., 0]
    img = img.astype(np.float32)
    if shape_hw is not None and img.shape != shape_hw:
        img = cv2.resize(img, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return img


def _read_rgb(path, shape_hw):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[:2] != shape_hw:
        img = cv2.resize(img, (shape_hw[1], shape_hw[0]))
    return img.astype(np.float32)


def run_selftest():
    """Synthetic data with KNOWN σ²-error relationships → verify the metrics behave."""
    rng = np.random.RandomState(1)
    n = 120000
    err = np.abs(rng.randn(n)) + 0.05                      # true per-pixel error
    sig_perfect = err.copy()                               # σ² == error
    sig_good = err + 0.6 * rng.randn(n)                    # noisy but correlated
    sig_rand = rng.permutation(err)                        # no information
    sig_anti = -err + err.max()                            # anti-calibrated

    print("=== SELFTEST: sparsification / AUSE (lower AUSE, higher frac_oracle = better) ===")
    results = {}
    for nm, sg in [('perfect', sig_perfect), ('good', sig_good), ('random', sig_rand), ('anti', sig_anti)]:
        fr, cu, co, cr = sparsification(err, np.clip(sg, 1e-6, None))
        a, ar, frac, _ = ause(fr, cu, co, cr)
        results[nm] = (a, frac)
        print(f"  {nm:8s}: AUSE {a:.4f}  frac_oracle {frac:+.3f}")
    ok = (results['perfect'][0] < results['good'][0] < results['random'][0]
          and results['anti'][0] >= results['random'][0] - 1e-6
          and results['perfect'][1] > 0.9 and abs(results['random'][1]) < 0.15)
    # motion correlation sanity
    motion = np.abs(rng.randn(n))
    sig_mot = 0.8 * motion + 0.2 * rng.randn(n)
    r_mot = pearson(sig_mot, motion); r_none = pearson(rng.randn(n), motion)
    print(f"\n=== SELFTEST: Pearson(σ², motion) ===")
    print(f"  correlated   r={r_mot:+.3f} (expect ~+0.97)   uncorrelated r={r_none:+.3f} (expect ~0)")
    ok = ok and r_mot > 0.9 and abs(r_none) < 0.05
    print(f"\nSELFTEST {'PASS ✓' if ok else 'FAIL ✗'} — "
          f"AUSE orders perfect<good<random<=anti, frac_oracle perfect>0.9 random~0, motion r tracks.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true', help='verify the math on synthetic data, no I/O')
    ap.add_argument('--name', default='cell')
    ap.add_argument('--uncert_dir', default=''); ap.add_argument('--uncert_glob', default='*.png')
    ap.add_argument('--error_mode', choices=['rgb', 'depth'], default='rgb')
    ap.add_argument('--render_dir', default=''); ap.add_argument('--render_glob', default='[0-9]*.jpg')
    ap.add_argument('--rgb_dir', default=''); ap.add_argument('--rgb_glob', default='*.png')
    ap.add_argument('--gt_depth_dir', default=''); ap.add_argument('--gt_depth_glob', default='*.npy')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--px_stride', type=int, default=2)
    ap.add_argument('--out_json', default=''); ap.add_argument('--out_fig', default='')
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(run_selftest())

    assert cv2 is not None, 'cv2 required for the data path (selftest needs only numpy)'
    assert args.uncert_dir and args.render_dir, 'need --uncert_dir and --render_dir (or --selftest)'
    U = sorted(glob.glob(os.path.join(args.uncert_dir, args.uncert_glob)))
    Rn = sorted(glob.glob(os.path.join(args.render_dir, args.render_glob)))
    GT = (sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob))) if args.error_mode == 'rgb'
          else sorted(glob.glob(os.path.join(args.gt_depth_dir, args.gt_depth_glob))))
    N = min(len(U), len(Rn), len(GT))
    assert N >= 2, f'need aligned frames (uncert {len(U)} render {len(Rn)} gt {len(GT)})'
    print(f"[{args.name}] frames {N} | error_mode {args.error_mode}")

    sig_all, err_all, mot_all = [], [], []
    prev_gray = None
    for i in range(0, N, args.stride):
        sg = _read_gray(U[i])
        if sg is None:
            continue
        H, W = sg.shape
        # --- per-pixel render error ---
        if args.error_mode == 'rgb':
            rend = _read_rgb(Rn[i], (H, W)); gt = _read_rgb(GT[i], (H, W))
            if rend is None or gt is None:
                continue
            err = np.abs(rend - gt).mean(-1)                      # [H,W]
            gray = gt.mean(-1)
        else:
            rend = _read_gray(Rn[i]);
            gt = np.load(GT[i]).astype(np.float32)
            if rend is None:
                continue
            rend = cv2.resize(rend, (W, H)) if rend.shape != (H, W) else rend
            gt = cv2.resize(gt, (W, H)) if gt.shape != (H, W) else gt
            err = np.abs(rend - gt) / (np.abs(gt) + 1e-6)          # scale-invariant depth error
            gray = gt
        # --- inter-frame motion proxy (deformation-awareness) ---
        mot = np.abs(gray - prev_gray) if (prev_gray is not None and prev_gray.shape == gray.shape) else np.zeros_like(gray)
        prev_gray = gray
        s = slice(None, None, args.px_stride)
        sig_all.append(sg[s, s].ravel()); err_all.append(err[s, s].ravel()); mot_all.append(mot[s, s].ravel())

    sig = np.concatenate(sig_all); err = np.concatenate(err_all); mot = np.concatenate(mot_all)
    fr, cu, co, cr = sparsification(err, np.clip(sig, 1e-6, None))
    a_unc, a_rand, frac, (n_unc, n_or, n_rand) = ause(fr, cu, co, cr)
    r_mot = pearson(sig, mot)

    print(f"\n=== (1) MAPPING calibration — sparsification / AUSE (n={len(sig)} px, {args.error_mode} error) ===")
    print(f"  AUSE            : {a_unc:.4f}   (lower = σ² ranks error better; 0 = oracle)")
    print(f"  AUSE_random     : {a_rand:.4f}")
    print(f"  frac_oracle     : {frac:+.3f}   (1=perfect, 0=random, <0=anti-calibrated)")
    print(f"\n=== (2) TRACKING correctness — deformation-awareness ===")
    print(f"  Pearson(σ², inter-frame motion): {r_mot:+.3f}")
    print(f"  NOTE: high AUSE-calibration + low motion-corr = σ² is a render-error/contrast detector,")
    print(f"        NOT deformation-aware -> the COMBINE (Arm-2 motion -> σ²) is what lifts (2).")

    out = {'name': args.name, 'error_mode': args.error_mode, 'n_px': int(len(sig)),
           'ause': a_unc, 'ause_random': a_rand, 'frac_oracle': frac, 'pearson_sigma2_motion': r_mot}
    oj = args.out_json or (os.path.join(os.path.dirname(args.uncert_dir.rstrip('/\\')), f'{args.name}_sigma2_quality.json'))
    with open(oj, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\njson -> {oj}")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.3)); fig.suptitle(f"Arm-1 σ²-quality — {args.name}", fontsize=12)
        ax[0].plot(fr, n_or, 'g-', label='oracle (by true error)')
        ax[0].plot(fr, n_unc, 'b-', label=f'σ²-ranked (AUSE {a_unc:.3f})')
        ax[0].plot(fr, n_rand, color='gray', ls='--', label='random')
        ax[0].set_xlabel('fraction of highest-σ² pixels removed'); ax[0].set_ylabel('remaining error (norm.)')
        ax[0].set_title(f'(1) calibration  frac_oracle={frac:+.2f}'); ax[0].legend(fontsize=8)
        idx = np.random.RandomState(0).permutation(len(sig))[:20000]
        ax[1].scatter(mot[idx], sig[idx], s=2, alpha=.15, c='tab:orange')
        ax[1].set_xlabel('inter-frame motion |I_t−I_{t-1}|'); ax[1].set_ylabel('σ²')
        ax[1].set_title(f'(2) deformation-awareness  r={r_mot:+.2f}')
        of = args.out_fig or (os.path.join(os.path.dirname(args.uncert_dir.rstrip('/\\')), f'{args.name}_sigma2_quality.png'))
        plt.tight_layout(); plt.savefig(of, dpi=90); print(f"figure -> {of}")
    except Exception as e:
        print(f"(no figure: {e})")


if __name__ == '__main__':
    main()
