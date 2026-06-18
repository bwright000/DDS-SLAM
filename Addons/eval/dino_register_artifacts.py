#!/usr/bin/env python3
"""Arm-1 analysis C — do registers actually remove DINOv2 artifact tokens on OUR vits14? (may KILL dino_reg)

Darcet ICLR-2024 ("ViTs Need Registers"): plain DINOv2 emits high-NORM artifact tokens in low-info
regions (≈2.37%, norm>150 on ViT-g); registers drive this ≈0%. BUT vits14 is small/less-trained — its
artifacts may ALREADY be ≈0%, making dino_reg a no-op. This measures the artifact-token fraction on
OUR baked grids (plain vs _reg) BEFORE any dino_reg A/B, and (if σ² given) whether those tokens
polluted σ². If plain ≈ reg ≈ 0% -> kill dino_reg honestly. NUMERICAL + VISUAL per the rule.

  python Addons/eval/dino_register_artifacts.py \
    --dino_dir data/Super/trail_3/dino --dino_reg_dir data/Super/trail_3/dino_reg \
    [--uncert_dir output/dino_super_s0/uncert]  --name trail3
"""
import os, glob, argparse, numpy as np


def load_grid(p):
    g = np.load(p).astype(np.float32)
    known = {384, 768, 1024, 1536}
    ax = next((i for i, s in enumerate(g.shape) if s in known), 2)
    return np.moveaxis(g, ax, 2)                      # (Hp,Wp,C)


def token_norms(paths, stride):
    norms = []
    for p in paths[::stride]:
        g = load_grid(p)
        norms.append(np.linalg.norm(g, axis=2).ravel())   # per-token L2 norm
    return np.concatenate(norms) if norms else np.array([])


def speckle_tokens(paths, stride):
    """Per-token spatial incoherence = the PCA blotchiness. ||feature - mean(4-neighbours)|| / ||feature||
    (scale-invariant). The BLOTCHES are the TAIL of this (a minority of bad tokens), NOT the mean -- and
    the tail is what matters because σ² is heavy-tailed. Returns the per-token values so we read the tail."""
    vals = []
    for p in paths[::stride]:
        g = load_grid(p).astype(np.float32)
        nb = (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1)) / 4.0
        rough = np.linalg.norm(g - nb, axis=2) / (np.linalg.norm(g, axis=2) + 1e-6)
        vals.append(rough[1:-1, 1:-1].ravel())             # drop the wrap-edge border
    return np.concatenate(vals) if vals else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cell')
    ap.add_argument('--dino_dir', required=True, help='plain vits14 grids')
    ap.add_argument('--dino_reg_dir', default='', help='vits14_reg grids (the comparison)')
    ap.add_argument('--dino_glob', default='*_dino.npy')
    ap.add_argument('--uncert_dir', default=''); ap.add_argument('--uncert_glob', default='*.png')
    ap.add_argument('--norm_thresh', type=float, default=150.0, help='Darcet abs threshold (ViT-g); we ALSO report a data-driven knee')
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--out_fig', default='')
    ap.add_argument('--pca_n', type=int, default=0, help='ALSO render first-3-PCA-as-RGB for N sample frames (plain vs reg) = the literal blotch view; the visual that motivated dino_reg')
    args = ap.parse_args()

    P = sorted(glob.glob(os.path.join(args.dino_dir, args.dino_glob)))
    PR = sorted(glob.glob(os.path.join(args.dino_reg_dir, args.dino_glob))) if args.dino_reg_dir else []
    assert P, f'no grids in {args.dino_dir}'
    n_plain = token_norms(P, args.stride)
    n_reg = token_norms(PR, args.stride) if PR else np.array([])
    # data-driven knee: median + 6*MAD (robust outlier line), reported alongside the absolute 150
    med = np.median(n_plain); mad = np.median(np.abs(n_plain - med)) + 1e-6
    knee = med + 6 * 1.4826 * mad

    def frac(n, t): return float((n > t).mean()) if len(n) else float('nan')
    # vits14 has NO Darcet-style dramatic (>150) artifacts; its "blotches" are a MILD upper tail.
    # Measure at vits14's OWN scale: relative spread (max/med, p99/med) + fraction above the PLAIN p95
    # applied to BOTH backbones (so a tightening shows up). Absolute 150 reported only for reference.
    rel = float(np.percentile(n_plain, 95))
    print(f"[{args.name}] tokens: plain {len(n_plain)}  reg {len(n_reg)}")
    print(f"\n=== (C) DINO token L2-norm (vits14 has no >150 outliers — measure relative spread) ===")
    print(f"  plain vits14 : med {med:.1f}  p98 {np.percentile(n_plain,98):.1f}  max {n_plain.max():.1f}  (max/med {n_plain.max()/med:.2f}, p99/med {np.percentile(n_plain,99)/med:.2f})")
    if len(n_reg):
        rmed = np.median(n_reg)
        print(f"  vits14_reg   : med {rmed:.1f}  p98 {np.percentile(n_reg,98):.1f}  max {n_reg.max():.1f}  (max/med {n_reg.max()/rmed:.2f}, p99/med {np.percentile(n_reg,99)/rmed:.2f})")
        print(f"  upper-tail spread (max/med): plain {n_plain.max()/med:.3f} -> reg {n_reg.max()/rmed:.3f}  ({'TIGHTER (registers clean the mild tail = the PCA blotches)' if n_reg.max()/rmed < n_plain.max()/med else 'no tightening'})")
        print(f"  frac above plain-p95 ({rel:.1f}): plain {100*frac(n_plain,rel):.1f}%  reg {100*frac(n_reg,rel):.1f}%  (reg<plain => mild artifact reduction, real but small)")
    print(f"  (ref: Darcet >150 abs: plain {100*frac(n_plain,args.norm_thresh):.2f}% — ~0 by design, vits14 too small for dramatic artifacts)")
    # the metric that ACTUALLY matches the PCA: spatial speckle. The BLOTCHES are the TAIL, not the mean.
    sp_p = speckle_tokens(P, args.stride); sp_r = speckle_tokens(PR, args.stride) if PR else np.array([])
    print(f"\n  *** spatial SPECKLE — the BLOTCHES are the TAIL (a minority of bad tokens), not the mean ***")
    print(f"      plain: mean {sp_p.mean():.3f}  p95 {np.percentile(sp_p,95):.3f}  p99 {np.percentile(sp_p,99):.3f}  max {sp_p.max():.3f}")
    if len(sp_r):
        t_p, t_r = np.percentile(sp_p, 99), np.percentile(sp_r, 99)
        print(f"      reg  : mean {sp_r.mean():.3f}  p95 {np.percentile(sp_r,95):.3f}  p99 {t_r:.3f}  max {sp_r.max():.3f}")
        print(f"      BLOTCH tail p99: {t_p:.3f} -> {t_r:.3f} ({100*(1-t_r/max(t_p,1e-9)):+.0f}%)  worst max: {sp_p.max():.3f} -> {sp_r.max():.3f} ({100*(1-sp_r.max()/max(sp_p.max(),1e-9)):+.0f}%)  <- the blotch removal the mean hides")

    # σ²-on-artifact: did high-norm tokens pollute σ²?
    bias = None; corr_fn = None
    if args.uncert_dir:
        import cv2
        U = sorted(glob.glob(os.path.join(args.uncert_dir, args.uncert_glob)))
        ins, outs = [], []; sig_s, nrm_s = [], []
        for i, p in list(enumerate(P))[::args.stride]:
            if i >= len(U): break
            g = load_grid(p); nm = np.linalg.norm(g, axis=2)              # (Hp,Wp) DINO feature norm
            sg = cv2.imread(U[i], cv2.IMREAD_UNCHANGED)
            if sg is None: continue
            sg = sg.astype(np.float32);  sg = sg[..., 0] if sg.ndim == 3 else sg
            nm_up = cv2.resize(nm, (sg.shape[1], sg.shape[0]), interpolation=cv2.INTER_NEAREST)
            art = nm_up > rel                                            # vits14-scale tail (plain p95)
            if art.any() and (~art).any(): ins.append(sg[art].mean()); outs.append(sg[~art].mean())
            sig_s.append(sg[::4, ::4].ravel()); nrm_s.append(nm_up[::4, ::4].ravel())
        if sig_s:
            ss = np.concatenate(sig_s); nn = np.concatenate(nrm_s)
            sz = ss - ss.mean(); nz = nn - nn.mean(); dd = np.sqrt((sz * sz).sum() * (nz * nz).sum())
            corr_fn = float((sz * nz).sum() / dd) if dd > 1e-12 else 0.0
            print(f"\n  *** Pearson(σ², DINO feature-norm) = {corr_fn:+.3f} ***  THE DECISIVE TEST:")
            print(f"      |r|≳0.3 => feature blotches DO drive σ² -> cleaner reg features help σ² -> KEEP dino_reg.")
            print(f"      |r|~0   => feature blotches do NOT drive σ² (it's contrast/motion, see analysis A) -> dino_reg can't help σ².")
        if ins:
            bias = (float(np.mean(ins)), float(np.mean(outs)))
            print(f"  σ² inside feature-tail vs outside: {bias[0]:.1f} vs {bias[1]:.1f}  ({bias[0]/max(bias[1],1e-6):.2f}x)")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3 if bias else 2, figsize=(15 if bias else 10, 4)); fig.suptitle(f"DINO register artifacts — {args.name}")
        ax[0].hist(n_plain, bins=120, alpha=.6, label='plain', log=True)
        if len(n_reg): ax[0].hist(n_reg, bins=120, alpha=.6, label='reg', log=True)
        ax[0].axvline(rel, c='k', ls=':', lw=1, label='plain p95'); ax[0].set_xlabel('token L2 norm'); ax[0].set_ylabel('count (log)'); ax[0].legend(); ax[0].set_title('token-norm dist (vits14: no >150 outliers)')
        bars = [100*frac(n_plain, rel)] + ([100*frac(n_reg, rel)] if len(n_reg) else [])
        ax[1].bar(['plain'] + (['reg'] if len(n_reg) else []), bars, color=['tab:blue', 'tab:green'][:len(bars)]); ax[1].set_ylabel('% tokens > plain-p95'); ax[1].set_title('mild-tail fraction (the blotches)')
        if bias: ax[2].bar(['outside', 'inside-artifact'], [bias[1], bias[0]], color=['gray', 'tab:red']); ax[2].set_ylabel('mean σ²'); ax[2].set_title('σ² pollution by artifacts')
        out = args.out_fig or os.path.join(os.path.dirname(args.dino_dir.rstrip('/')), f'{args.name}_dino_registers.png')
        plt.tight_layout(); plt.savefig(out, dpi=90); print(f"figure -> {out}")
    except Exception as e:
        print(f"(no figure: {e})")

    # ---- PCA blotch view (the literal thing the user observed) ----
    if args.pca_n > 0:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            def pca_rgb(g):                                  # (Hp,Wp,C) -> first-3-PCA as RGB in [0,1]
                H, W, C = g.shape; X = g.reshape(-1, C).astype(np.float32); X = X - X.mean(0)
                Vt = np.linalg.svd(X, full_matrices=False)[2]
                Y = X @ Vt[:3].T
                Y = (Y - Y.min(0)) / (np.ptp(Y, 0) + 1e-9)
                return Y.reshape(H, W, 3)
            idx = np.linspace(0, len(P) - 1, args.pca_n).astype(int)
            cols = 2 if PR else 1
            fig, ax = plt.subplots(args.pca_n, cols, figsize=(4.5 * cols, 3 * args.pca_n), squeeze=False)
            for r, i in enumerate(idx):
                ax[r][0].imshow(pca_rgb(load_grid(P[i]))); ax[r][0].set_title(f'plain f{i}'); ax[r][0].axis('off')
                if PR: ax[r][1].imshow(pca_rgb(load_grid(PR[i]))); ax[r][1].set_title(f'reg f{i}'); ax[r][1].axis('off')
            outp = os.path.join(os.path.dirname(args.dino_dir.rstrip('/')), f'{args.name}_dino_PCA.png')
            plt.suptitle(f'DINO PCA (first 3 comps as RGB) — {args.name}: do registers clean the blotches?'); plt.tight_layout(); plt.savefig(outp, dpi=90); print(f"PCA figure -> {outp}")
        except Exception as e:
            print(f"(no PCA figure: {e})")


if __name__ == '__main__':
    main()
