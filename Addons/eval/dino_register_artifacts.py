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
    print(f"[{args.name}] tokens: plain {len(n_plain)}  reg {len(n_reg)}")
    print(f"\n=== (C) DINO token L2-norm (artifact = high norm) ===")
    print(f"  plain vits14 : med {med:.1f}  p98 {np.percentile(n_plain,98):.1f}  max {n_plain.max():.1f}")
    if len(n_reg): print(f"  vits14_reg   : med {np.median(n_reg):.1f}  p98 {np.percentile(n_reg,98):.1f}  max {n_reg.max():.1f}")
    print(f"  artifact frac (norm>{args.norm_thresh:.0f}, Darcet abs): plain {100*frac(n_plain,args.norm_thresh):.2f}%" + (f"  reg {100*frac(n_reg,args.norm_thresh):.2f}%" if len(n_reg) else ""))
    print(f"  artifact frac (norm>{knee:.1f}, data-driven knee)    : plain {100*frac(n_plain,knee):.2f}%" + (f"  reg {100*frac(n_reg,knee):.2f}%" if len(n_reg) else ""))
    if frac(n_plain, args.norm_thresh) < 0.001 and frac(n_plain, knee) < 0.01:
        print("  ⚠️ VERDICT: plain vits14 artifact fraction ≈ 0 -> registers have little to remove -> dino_reg likely a NO-OP. Kill the arm unless (D)/render says otherwise.")
    elif len(n_reg):
        red = 100 * (1 - frac(n_reg, knee) / max(frac(n_plain, knee), 1e-9))
        print(f"  registers cut the knee-artifact fraction by {red:.0f}% -> dino_reg JUSTIFIED if σ² (below) confirms pollution.")

    # σ²-on-artifact: did high-norm tokens pollute σ²?
    bias = None
    if args.uncert_dir:
        import cv2
        U = sorted(glob.glob(os.path.join(args.uncert_dir, args.uncert_glob)))
        ins, outs = [], []
        for i, p in list(enumerate(P))[::args.stride]:
            if i >= len(U): break
            g = load_grid(p); nm = np.linalg.norm(g, axis=2)              # (Hp,Wp)
            sg = cv2.imread(U[i], cv2.IMREAD_UNCHANGED)
            if sg is None: continue
            sg = sg.astype(np.float32);  sg = sg[..., 0] if sg.ndim == 3 else sg
            nm_up = cv2.resize(nm, (sg.shape[1], sg.shape[0]), interpolation=cv2.INTER_NEAREST)
            art = nm_up > knee
            if art.any() and (~art).any(): ins.append(sg[art].mean()); outs.append(sg[~art].mean())
        if ins:
            bias = (float(np.mean(ins)), float(np.mean(outs)))
            print(f"\n  σ² inside artifact tokens {bias[0]:.1f}  vs outside {bias[1]:.1f}  ({'POLLUTED: '+f'{bias[0]/max(bias[1],1e-6):.2f}x' if bias[0]>bias[1] else 'no pollution'})")

    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3 if bias else 2, figsize=(15 if bias else 10, 4)); fig.suptitle(f"DINO register artifacts — {args.name}")
        ax[0].hist(n_plain, bins=120, alpha=.6, label='plain', log=True)
        if len(n_reg): ax[0].hist(n_reg, bins=120, alpha=.6, label='reg', log=True)
        ax[0].axvline(args.norm_thresh, c='r', ls='--', lw=1, label='150'); ax[0].axvline(knee, c='k', ls=':', lw=1, label='knee'); ax[0].set_xlabel('token L2 norm'); ax[0].set_ylabel('count (log)'); ax[0].legend(); ax[0].set_title('token-norm dist')
        bars = [100*frac(n_plain, knee)] + ([100*frac(n_reg, knee)] if len(n_reg) else [])
        ax[1].bar(['plain'] + (['reg'] if len(n_reg) else []), bars, color=['tab:blue', 'tab:green'][:len(bars)]); ax[1].set_ylabel('% artifact tokens (>knee)'); ax[1].set_title('artifact fraction')
        if bias: ax[2].bar(['outside', 'inside-artifact'], [bias[1], bias[0]], color=['gray', 'tab:red']); ax[2].set_ylabel('mean σ²'); ax[2].set_title('σ² pollution by artifacts')
        out = args.out_fig or os.path.join(os.path.dirname(args.dino_dir.rstrip('/')), f'{args.name}_dino_registers.png')
        plt.tight_layout(); plt.savefig(out, dpi=90); print(f"figure -> {out}")
    except Exception as e:
        print(f"(no figure: {e})")


if __name__ == '__main__':
    main()
