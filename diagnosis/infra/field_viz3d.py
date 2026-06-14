"""
3D deformation-field visualiser (a thesis deliverable + a diagnostic).

Shows, in 3D, WHERE the deformation field puts its motion and HOW BIG it is. Two modes:
  --dx_dir   : from dx_hook NPZ dumps (x_canonical + delta_x) — runs anywhere, no GPU/checkpoint.
  (future)   : --checkpoint grid mode (query TimeNet on a regular grid) — needs CUDA/tcnn, on Colab.

The diagnostic question this answers visually: is the field's energy AT the tissue surface (in the
scene bound) where it belongs, or OFF-surface in empty space (the Battery-7 pathology)?

Panels:
  A  3D scatter of sample points, coloured by ||Δx||  (subsampled)
  B  ||Δx|| vs depth z, with the scene bound marked  (in-bound = tissue; out-of-bound = empty space)
  C  histogram of ||Δx||, in-bound vs out-of-bound
Prints the decisive ratio: mean ||Δx|| out-of-bound / in-bound  (>>1 = field wastes motion off-surface).

Usage:
  python diagnosis/infra/field_viz3d.py --dx_dir <dir of frame_*.npz> --out field3d.png \
      [--bound "[[-0.7,0.7],[-0.7,0.7],[0.7,1.2]]"] [--max_frames 30 --scatter_frame frame_0072.npz]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_dx(dx_dir, max_frames):
    files = sorted(glob.glob(os.path.join(dx_dir, '*.npz')))[:max_frames]
    if not files:
        raise SystemExit(f'no NPZ in {dx_dir}')
    X, D = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X.append(d['x_canonical'].reshape(-1, 3)); D.append(d['delta_x'].reshape(-1, 3))
    return np.concatenate(X), np.concatenate(D), files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dx_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bound', default='[[-0.7,0.7],[-0.7,0.7],[0.7,1.2]]')
    ap.add_argument('--max_frames', type=int, default=30)
    ap.add_argument('--scatter_frame', default=None, help='one NPZ name for the 3D scatter (default: a mid frame)')
    args = ap.parse_args()
    bound = np.array(json.loads(args.bound))                       # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]

    X, D, files = load_dx(args.dx_dir, args.max_frames)
    mag = np.linalg.norm(D, axis=-1)                              # ||Δx|| per point
    inb = ((X >= bound[:, 0]) & (X <= bound[:, 1])).all(-1)       # inside the scene bounding box

    in_mean = float(mag[inb].mean()) if inb.any() else float('nan')
    out_mean = float(mag[~inb].mean()) if (~inb).any() else float('nan')
    ratio = out_mean / (in_mean + 1e-12)
    summary = {
        'dx_dir': args.dx_dir, 'n_points': int(mag.size), 'frac_in_bound': float(inb.mean()),
        'mean_dx_in_bound': in_mean, 'mean_dx_out_of_bound': out_mean,
        'out_over_in_ratio': round(ratio, 3), 'max_dx': float(mag.max()),
        'verdict': ('FIELD WASTES MOTION OFF-SURFACE (out-of-bound >> in-bound)'
                    if ratio > 3 else 'field motion concentrated near/in the scene')
    }

    # scatter from one frame for clarity
    sf = args.scatter_frame
    if sf is None:
        sf = os.path.basename(files[len(files) // 2])
    sd = np.load(os.path.join(args.dx_dir, sf), allow_pickle=True)
    sx = sd['x_canonical'].reshape(-1, 3); sm = np.linalg.norm(sd['delta_x'].reshape(-1, 3), axis=-1)
    if sx.shape[0] > 6000:
        idx = np.random.default_rng(0).choice(sx.shape[0], 6000, replace=False); sx, sm = sx[idx], sm[idx]

    fig = plt.figure(figsize=(18, 5.5))
    axA = fig.add_subplot(1, 3, 1, projection='3d')
    p = axA.scatter(sx[:, 0], sx[:, 1], sx[:, 2], c=sm, s=4, cmap='inferno')
    fig.colorbar(p, ax=axA, shrink=0.6, label='||Δx||')
    axA.set_title(f'A: 3D field ({sf})\ncolour = deformation magnitude'); axA.set_xlabel('x'); axA.set_ylabel('y'); axA.set_zlabel('z')

    axB = fig.add_subplot(1, 3, 2)
    axB.scatter(X[:, 2], mag, s=2, alpha=0.15, color='#333')
    axB.axvspan(bound[2, 0], bound[2, 1], color='#7fbf7f', alpha=0.3, label=f'scene bound z∈[{bound[2,0]},{bound[2,1]}]')
    axB.set_xlabel('sample depth z'); axB.set_ylabel('||Δx||'); axB.set_title('B: deformation vs depth'); axB.legend()

    axC = fig.add_subplot(1, 3, 3)
    axC.hist(mag[inb], bins=60, alpha=0.6, label=f'in-bound (mean {in_mean:.2e})', color='#2a7d2a', density=True)
    axC.hist(mag[~inb], bins=60, alpha=0.6, label=f'out-of-bound (mean {out_mean:.2e})', color='#b03030', density=True)
    axC.set_xlabel('||Δx||'); axC.set_ylabel('density'); axC.set_yscale('log')
    axC.set_title(f'C: in vs out of scene\nout/in ratio = {ratio:.1f}'); axC.legend()

    fig.suptitle(f'Deformation field — {args.dx_dir}', fontsize=13)
    plt.tight_layout(); plt.savefig(args.out, dpi=120, bbox_inches='tight')
    json.dump(summary, open(args.out.replace('.png', '.json'), 'w'), indent=2)
    print(json.dumps(summary, indent=2)); print('wrote', args.out)


if __name__ == '__main__':
    main()
