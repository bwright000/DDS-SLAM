#!/usr/bin/env python3
"""Arm-1 σ² diagnostics — is the uncertainty detecting CONTRAST/MOTION, not deformation? (analyses A+B)

Tests the literature finding (NeRF-On-the-go's σ∝‖error‖ degeneracy ⇒ a raw-residual σ² is a contrast/
edge/specular/motion readout) on OUR σ² maps, turning the session observation into a number.

  A) correlation: per-pixel Pearson of σ² vs image-gradient |∇I| (contrast), vs inter-frame motion
     |I_t-I_{t-1}|, and point-biserial vs a specular mask. Predicted: σ² ~ |∇I| strong, ~ motion
     moderate. (No clean deformation signal here — that's the whole point; deformation needs Arm-2.)
  B) segment-stratified mean σ² per class {tool, tissue, bg}. Predicted: high on tool/specular.

NUMERICAL (correlations + per-class means) + VISUAL (a 4-panel PNG) per the two-diagnostic-sets rule.
Run once per cell (geo/dino/dino_reg uncert dir) and compare. Pure numpy + cv2.

  python Addons/eval/sigma2_diagnostics.py --name geo_super_s0 \
    --uncert_dir output/geo_super_s0/uncert --rgb_dir data/Super/trail_3/rgb --rgb_glob '*left.png' \
    --seg_dir data/Super/trail_3/seg/png_masks --seg_glob '*left.png' --tool_labels 3
"""
import os, glob, argparse, numpy as np, cv2


def pearson(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='cell')
    ap.add_argument('--uncert_dir', required=True); ap.add_argument('--uncert_glob', default='*.png')
    ap.add_argument('--rgb_dir', required=True); ap.add_argument('--rgb_glob', default='*.png')
    ap.add_argument('--seg_dir', default=''); ap.add_argument('--seg_glob', default='*.png')
    ap.add_argument('--tool_labels', default='', help='csv of seg pixel-values that are TOOL (e.g. 3)')
    ap.add_argument('--tissue_labels', default='', help='csv of TISSUE labels; empty=all non-bg non-tool')
    ap.add_argument('--bg_labels', default='0', help='csv of BACKGROUND labels')
    ap.add_argument('--stride', type=int, default=3, help='use every Nth frame')
    ap.add_argument('--px_stride', type=int, default=3, help='subsample pixels for the correlation')
    ap.add_argument('--out_fig', default='', help='PNG path; default <uncert_dir>/../<name>_sigma2_diag.png')
    args = ap.parse_args()

    U = sorted(glob.glob(os.path.join(args.uncert_dir, args.uncert_glob)))
    R = sorted(glob.glob(os.path.join(args.rgb_dir, args.rgb_glob)))
    S = sorted(glob.glob(os.path.join(args.seg_dir, args.seg_glob))) if args.seg_dir else []
    N = min(len(U), len(R)) if not S else min(len(U), len(R), len(S))
    assert N >= 2, f'need aligned frames (uncert {len(U)} rgb {len(R)} seg {len(S)})'
    tool = set(int(x) for x in args.tool_labels.split(',') if x != '')
    bg = set(int(x) for x in args.bg_labels.split(',') if x != '')
    tis = set(int(x) for x in args.tissue_labels.split(',') if x != '')
    print(f"[{args.name}] frames {N} | seg {'yes' if S else 'no'} | tool={tool or '-'} bg={bg or '-'}")

    sig_all, grad_all, mot_all, spec_all = [], [], [], []
    cls_sig = {'tool': [], 'tissue': [], 'bg': []}; sp_sig = []
    prev_gray = None
    for i in range(0, N, args.stride):
        sg = cv2.imread(U[i], cv2.IMREAD_UNCHANGED)
        if sg is None: continue
        sg = sg.astype(np.float32)
        if sg.ndim == 3: sg = sg[..., 0]
        H, W = sg.shape
        _nb = (np.roll(sg, 1, 0) + np.roll(sg, -1, 0) + np.roll(sg, 1, 1) + np.roll(sg, -1, 1)) / 4.0
        sp_sig.append((np.abs(sg - _nb) / (sg + 1.0))[1:-1, 1:-1].mean())   # σ² spatial speckle
        rgb = cv2.imread(R[i]); rgb = cv2.resize(rgb, (W, H))
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
        grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 3))
        spec = (gray > np.percentile(gray, 99)).astype(np.float32)
        if prev_gray is not None and prev_gray.shape == gray.shape:
            mot = np.abs(gray - prev_gray)
        else:
            mot = np.zeros_like(gray)
        prev_gray = gray
        ss, gg, mm, pp = (a[::args.px_stride, ::args.px_stride].ravel() for a in (sg, grad, mot, spec))
        sig_all.append(ss); grad_all.append(gg); mot_all.append(mm); spec_all.append(pp)
        if S:
            seg = cv2.imread(S[i], cv2.IMREAD_GRAYSCALE)
            if seg is not None:
                seg = cv2.resize(seg, (W, H), interpolation=cv2.INTER_NEAREST)
                m_tool = np.isin(seg, list(tool)) if tool else np.zeros_like(seg, bool)
                m_bg = np.isin(seg, list(bg)) if bg else np.zeros_like(seg, bool)
                m_tis = np.isin(seg, list(tis)) if tis else (~m_tool & ~m_bg)
                for nm, msk in [('tool', m_tool), ('tissue', m_tis), ('bg', m_bg)]:
                    if msk.any(): cls_sig[nm].append(sg[msk])
    sig = np.concatenate(sig_all); grad = np.concatenate(grad_all); mot = np.concatenate(mot_all); spec = np.concatenate(spec_all)
    r_grad, r_mot = pearson(sig, grad), pearson(sig, mot)
    # point-biserial = pearson with the binary specular mask
    r_spec = pearson(sig, spec)
    print(f"\n=== (A) σ² correlations (n={len(sig)} px) ===")
    print(f"  Pearson(σ², |∇I| contrast) : {r_grad:+.3f}")
    print(f"  Pearson(σ², |I_t−I_t-1| motion): {r_mot:+.3f}")
    print(f"  point-biserial(σ², specular): {r_spec:+.3f}")
    print(f"  => σ² is dominated by {'CONTRAST' if abs(r_grad)>=abs(r_mot) else 'MOTION'} (lit prediction: contrast/edge/specular detector, NOT deformation)")
    print(f"  σ² spatial speckle : {np.mean(sp_sig):.3f}  (the dino_reg decider — compare dino vs dino_reg: lower on dino_reg => the register feature-smoothing reached σ²)")
    means = {}
    if S:
        print(f"\n=== (B) segment-stratified σ² ===")
        for nm in ('tool', 'tissue', 'bg'):
            if cls_sig[nm]:
                v = np.concatenate(cls_sig[nm]); means[nm] = (float(v.mean()), float(v.std()), len(v))
                print(f"  {nm:7s}: mean {v.mean():.1f}  std {v.std():.1f}  (n={len(v)})")
        if 'tool' in means and 'tissue' in means:
            print(f"  tool/tissue σ² ratio: {means['tool'][0]/max(means['tissue'][0],1e-6):.2f}x  (>>1 => σ² on tools not tissue)")

    # ---- VISUAL ----
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.2)); fig.suptitle(f"Arm-1 σ² diagnostics — {args.name}", fontsize=13)
        idx = np.random.RandomState(0).permutation(len(sig))[:20000]
        ax[0].scatter(grad[idx], sig[idx], s=2, alpha=.15); ax[0].set_xlabel('|∇I| contrast'); ax[0].set_ylabel('σ²'); ax[0].set_title(f'σ² vs contrast  r={r_grad:+.2f}')
        ax[1].scatter(mot[idx], sig[idx], s=2, alpha=.15, c='tab:orange'); ax[1].set_xlabel('|I_t−I_t-1| motion'); ax[1].set_title(f'σ² vs motion  r={r_mot:+.2f}')
        ax[2].bar(['contrast', 'motion', 'specular'], [r_grad, r_mot, r_spec], color=['tab:blue', 'tab:orange', 'tab:red']); ax[2].axhline(0, c='k', lw=.5); ax[2].set_ylabel('Pearson r'); ax[2].set_title('σ² correlations')
        if means:
            ks = [k for k in ('bg', 'tissue', 'tool') if k in means]
            ax[3].bar(ks, [means[k][0] for k in ks], yerr=[means[k][1] for k in ks], color=['gray', 'tab:green', 'tab:red'][:len(ks)]); ax[3].set_ylabel('mean σ²'); ax[3].set_title('σ² by segment class')
        else:
            ax[3].text(.5, .5, 'no seg', ha='center'); ax[3].axis('off')
        out = args.out_fig or os.path.join(os.path.dirname(args.uncert_dir.rstrip('/')), f'{args.name}_sigma2_diag.png')
        plt.tight_layout(); plt.savefig(out, dpi=90); print(f"\nfigure -> {out}")
    except Exception as e:
        print(f"\n(no figure: {e})")


if __name__ == '__main__':
    main()
