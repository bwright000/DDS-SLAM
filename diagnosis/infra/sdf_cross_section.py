"""
SDF cross-section visualization (diagnoses marching_cubes failures).

For a given checkpoint, query the SDF at a regular 2D grid on an axis-aligned
slice through the scene bound.  Render as a heatmap with iso-contours.

Three patterns to look for:
  - Clean sign changes at a curve  → surface forming (mc bug elsewhere if it fails)
  - Uniform same-sign values        → SDF didn't converge (mushy)
  - Saturated alternating values    → numerical instability

Usage:
  python diagnosis/infra/sdf_cross_section.py \
    --config configs/CRCD/c1_001_paperfaith_lrfix.yaml \
    --checkpoint output/CRCD/c1_001_paperfaith_lrfix/demo/checkpoint359.pt \
    --output_dir diagnosis/report/sdf_slice_C1_001 \
    --axes xy,xz,yz
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--axes', type=str, default='xy,xz,yz',
                        help='Comma-separated planes to slice on')
    parser.add_argument('--resolution', type=int, default=200,
                        help='Slice resolution per axis')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from config import load_config
    from model.scene_rep import JointEncoding

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    bounding_box = torch.from_numpy(np.array(cfg['mapping']['bound'])).to(device)
    model = JointEncoding(cfg, bounding_box).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    bound = np.array(cfg['mapping']['bound'])  # (3, 2): xmin,xmax / ymin,ymax / zmin,zmax
    xmin, xmax = bound[0]
    ymin, ymax = bound[1]
    zmin, zmax = bound[2]

    # Define center point (midpoint of bound)
    cx_, cy_, cz_ = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2

    axes_to_slice = args.axes.split(',')

    fig, axarr = plt.subplots(1, len(axes_to_slice), figsize=(6 * len(axes_to_slice), 6))
    if len(axes_to_slice) == 1:
        axarr = [axarr]

    for ax_idx, plane in enumerate(axes_to_slice):
        if plane == 'xy':
            U = torch.linspace(xmin, xmax, args.resolution, device=device)
            V = torch.linspace(ymin, ymax, args.resolution, device=device)
            UU, VV = torch.meshgrid(U, V, indexing='ij')
            pts = torch.stack([UU.flatten(), VV.flatten(), torch.full_like(UU.flatten(), cz_)], dim=-1)
            u_label, v_label, slice_label = 'x (m)', 'y (m)', f'z={cz_:.3f}'
            extent = [xmin, xmax, ymin, ymax]
        elif plane == 'xz':
            U = torch.linspace(xmin, xmax, args.resolution, device=device)
            V = torch.linspace(zmin, zmax, args.resolution, device=device)
            UU, VV = torch.meshgrid(U, V, indexing='ij')
            pts = torch.stack([UU.flatten(), torch.full_like(UU.flatten(), cy_), VV.flatten()], dim=-1)
            u_label, v_label, slice_label = 'x (m)', 'z (m)', f'y={cy_:.3f}'
            extent = [xmin, xmax, zmin, zmax]
        elif plane == 'yz':
            U = torch.linspace(ymin, ymax, args.resolution, device=device)
            V = torch.linspace(zmin, zmax, args.resolution, device=device)
            UU, VV = torch.meshgrid(U, V, indexing='ij')
            pts = torch.stack([torch.full_like(UU.flatten(), cx_), UU.flatten(), VV.flatten()], dim=-1)
            u_label, v_label, slice_label = 'y (m)', 'z (m)', f'x={cx_:.3f}'
            extent = [ymin, ymax, zmin, zmax]
        else:
            print(f'  unknown plane: {plane}'); continue

        # Query SDF (no time/deformation — sample static canonical scene)
        with torch.no_grad():
            sdf = model.query_sdf(pts.unsqueeze(0), return_geo=False)  # (1, N)
            sdf = sdf.squeeze(0).cpu().numpy()
        sdf_grid = sdf.reshape(args.resolution, args.resolution)

        ax = axarr[ax_idx]
        # Heatmap with diverging cmap (sign matters for SDF)
        vlim = max(abs(sdf_grid.min()), abs(sdf_grid.max()))
        im = ax.imshow(sdf_grid.T, origin='lower', extent=extent, cmap='RdBu_r',
                       vmin=-vlim, vmax=vlim, aspect='auto')
        # Zero-crossing contour (this is what marching_cubes extracts as the surface)
        try:
            contour = ax.contour(UU.cpu().numpy(), VV.cpu().numpy(), sdf_grid,
                                 levels=[0], colors='black', linewidths=1.5)
        except Exception as e:
            print(f'  contour failed on {plane}: {e}')
        ax.set_xlabel(u_label)
        ax.set_ylabel(v_label)
        ax.set_title(f'SDF on {plane.upper()}-plane ({slice_label})\n'
                     f'min={sdf_grid.min():.3f} max={sdf_grid.max():.3f}  '
                     f'frac<0 = {(sdf_grid<0).mean():.1%}')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f'SDF cross-section diagnostic\n'
                 f'config: {os.path.basename(args.config)}',
                 fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(args.output_dir, 'sdf_cross_section.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
