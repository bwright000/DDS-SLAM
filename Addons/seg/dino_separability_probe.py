"""DE-RISK PROBE for the feature-group attribution engine (ARM1_4 spec).

THE QUESTION (answered with ZERO model code / no GPU): do the FROZEN baked DINO features carry
SEPARABLE {tissue, tool, bg} with a real gap over a shuffled-label control? If yes, the learned
what-kind head (slot-attention v1b) is de-risked. If true ~= shuffled, frozen DINO can't separate
them -> slot-attention won't conjure it -> SKIP v1b, ship v1a (sigma^2-stabilisation) only.

Method: pair each DINO patch feature with its majority seg label (canonicalised to bg/tissue/tool),
held-out split, then compare held-out 3-way mIoU of:
  (1) TRUE      - logistic-regression head on DINO -> seg label
  (2) SHUFFLED  - same, train labels permuted (the chance control; the gate is TRUE >> SHUFFLED)
  (3) KMEANS    - unsupervised k=3 on DINO, Hungarian-matched to labels (does DINO cluster them WITHOUT labels)

  python Addons/seg/dino_separability_probe.py \
    --dino_dir data/CRCD/C1_001/dino_reg --dino_glob '*_dino.npy' \
    --seg_dir  data/CRCD/C1_001/masks   --seg_glob '*.png' \
    --tool_labels 3 --bg_labels 0     # set from the histogram it prints
  python Addons/seg/dino_separability_probe.py --selftest    # verify the mIoU/gap math, no data
"""
import os, sys, glob, argparse, numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def miou(y_true, y_pred, n_cls=3):
    ious = []
    for c in range(n_cls):
        inter = np.sum((y_true == c) & (y_pred == c))
        union = np.sum((y_true == c) | (y_pred == c))
        ious.append(inter / union if union > 0 else np.nan)
    return float(np.nanmean(ious)), [round(float(x), 3) if not np.isnan(x) else None for x in ious]


def hungarian_match(y_true, y_clusters, n_cls=3):
    """map cluster ids -> labels by max-overlap (greedy Hungarian on the 3x3 confusion)."""
    from itertools import permutations
    best, best_map = -1, None
    for perm in permutations(range(n_cls)):
        mapped = np.array([perm[c] for c in y_clusters])
        acc = np.mean(mapped == y_true)
        if acc > best:
            best, best_map = acc, perm
    return np.array([best_map[c] for c in y_clusters])


def run_selftest():
    rng = np.random.RandomState(0)
    n, C = 6000, 32
    y = rng.randint(0, 3, n)
    # features that ENCODE the label (separable) + noise
    feat = np.eye(3)[y] @ rng.randn(3, C) * 3.0 + rng.randn(n, C)
    from sklearn.linear_model import LogisticRegression
    tr = np.arange(n) % 2 == 0
    lr = LogisticRegression(max_iter=200).fit(feat[tr], y[tr])
    m_true, _ = miou(y[~tr], lr.predict(feat[~tr]))
    lr_s = LogisticRegression(max_iter=200).fit(feat[tr], rng.permutation(y[tr]))
    m_shuf, _ = miou(y[~tr], lr_s.predict(feat[~tr]))
    print(f"SELFTEST: separable-feature mIoU true={m_true:.3f} shuffled={m_shuf:.3f}  gap={m_true-m_shuf:+.3f}")
    ok = m_true > 0.7 and m_shuf < 0.4 and (m_true - m_shuf) > 0.3
    # non-separable control: features independent of label -> true ~= shuffled
    feat2 = rng.randn(n, C)
    m2t, _ = miou(y[~tr], LogisticRegression(max_iter=200).fit(feat2[tr], y[tr]).predict(feat2[~tr]))
    m2s, _ = miou(y[~tr], LogisticRegression(max_iter=200).fit(feat2[tr], rng.permutation(y[tr])).predict(feat2[~tr]))
    print(f"SELFTEST: noise-feature   mIoU true={m2t:.3f} shuffled={m2s:.3f}  gap={m2t-m2s:+.3f} (expect ~0)")
    ok = ok and abs(m2t - m2s) < 0.15
    print(f"SELFTEST {'PASS' if ok else 'FAIL'} - gap>>0 for separable, ~0 for noise.")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--dino_dir', default=''); ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--seg_dir', default=''); ap.add_argument('--seg_glob', default='*.png')
    ap.add_argument('--tool_labels', default='3'); ap.add_argument('--bg_labels', default='0')
    ap.add_argument('--tissue_labels', default='', help='empty = all non-tool non-bg')
    ap.add_argument('--stride', type=int, default=3, help='use every Nth frame')
    ap.add_argument('--px_stride', type=int, default=2, help='subsample patches')
    ap.add_argument('--test_frac', type=float, default=0.3, help='last frac of frames = held-out')
    ap.add_argument('--max_samples', type=int, default=200000)
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(run_selftest())

    import cv2
    from sklearn.linear_model import LogisticRegression
    from sklearn.cluster import KMeans
    tool = set(int(x) for x in args.tool_labels.split(',') if x != '')
    bg = set(int(x) for x in args.bg_labels.split(',') if x != '')
    tis = set(int(x) for x in args.tissue_labels.split(',') if x != '')

    D = sorted(glob.glob(os.path.join(args.dino_dir, args.dino_glob)))
    S = sorted(glob.glob(os.path.join(args.seg_dir, args.seg_glob)))
    N = min(len(D), len(S))
    assert N >= 4, f'need aligned dino+seg (dino {len(D)} seg {len(S)})'
    print(f"dino {len(D)} | seg {len(S)} | pairing {N} | tool={tool} bg={bg}")
    # label histogram of the first mask (so the mapping can be verified — like the seg-B fix)
    _s0 = cv2.imread(S[0], cv2.IMREAD_GRAYSCALE)
    print(f"  seg label histogram ({os.path.basename(S[0])}): {dict(zip(*[a.tolist() for a in np.unique(_s0, return_counts=True)]))}")

    feats, labs, frames = [], [], []
    for i in range(0, N, args.stride):
        dino = np.load(D[i]).astype(np.float32)         # [gh, gw, C]
        if dino.ndim != 3:
            continue
        gh, gw, C = dino.shape
        seg = cv2.imread(S[i], cv2.IMREAD_GRAYSCALE)
        if seg is None:
            continue
        seg_g = cv2.resize(seg, (gw, gh), interpolation=cv2.INTER_NEAREST)   # patch-grid majority(approx via nearest)
        lab = np.full((gh, gw), 1, np.int64)            # default 1 = tissue
        lab[np.isin(seg_g, list(bg))] = 0               # 0 = bg
        if tis:
            _tis = np.isin(seg_g, list(tis)); lab[~_tis & ~np.isin(seg_g, list(bg)) & ~np.isin(seg_g, list(tool))] = 0
        lab[np.isin(seg_g, list(tool))] = 2             # 2 = tool
        f = dino[::args.px_stride, ::args.px_stride].reshape(-1, C)
        l = lab[::args.px_stride, ::args.px_stride].reshape(-1)
        feats.append(f); labs.append(l); frames.append(np.full(len(l), i))
    X = np.concatenate(feats); Y = np.concatenate(labs); F = np.concatenate(frames)
    # cap
    if len(X) > args.max_samples:
        idx = np.random.RandomState(0).permutation(len(X))[:args.max_samples]
        X, Y, F = X[idx], Y[idx], F[idx]
    # L2-normalise (DINO lives in direction)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    cls_present = {int(c): int((Y == c).sum()) for c in (0, 1, 2)}
    print(f"  samples {len(X)} | class counts bg/tissue/tool = {cls_present}")
    if cls_present.get(2, 0) < 50:
        print("  🚨 fewer than 50 TOOL samples -> tool_labels likely wrong (check the histogram above) — re-run with the right --tool_labels.")

    # held-out split by FRAME (last test_frac of frames)
    fcut = np.quantile(F, 1 - args.test_frac)
    tr, te = F <= fcut, F > fcut
    print(f"  train {tr.sum()} / test {te.sum()} (held-out last {args.test_frac:.0%} of frames)")

    rng = np.random.RandomState(0)
    def fit_eval(ytr):
        lr = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
        lr.fit(X[tr], ytr); return miou(Y[te], lr.predict(X[te]))
    m_true, iou_true = fit_eval(Y[tr])
    m_shuf, _ = fit_eval(rng.permutation(Y[tr]))
    km = KMeans(n_clusters=3, n_init=4, random_state=0).fit(X[tr])
    cl_te = km.predict(X[te]); cl_mapped = hungarian_match(Y[te], cl_te)
    m_km, _ = miou(Y[te], cl_mapped)

    print(f"\n=== DINO SEPARABILITY (held-out 3-way mIoU) ===")
    print(f"  (1) TRUE  (supervised LR on DINO) : {m_true:.3f}   per-class bg/tissue/tool {iou_true}")
    print(f"  (2) SHUFFLED (chance control)     : {m_shuf:.3f}")
    print(f"  (3) KMEANS (unsupervised)         : {m_km:.3f}")
    gap = m_true - m_shuf
    print(f"  GAP (true - shuffled)             : {gap:+.3f}")
    print(f"\n  VERDICT: {'GO — DINO separates tissue/tool/bg, the learned what-kind (v1b) is de-risked.' if gap > 0.2 else 'NO-GO — true ~= shuffled; frozen DINO does NOT separate them -> SKIP v1b, ship v1a (sigma^2-stabilisation) only.'}")
    print(f"  (kmeans high => DINO clusters them WITHOUT labels => the pure-DINO/unsupervised arm is viable too.)")


if __name__ == '__main__':
    main()
