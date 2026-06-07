"""
TEST 3 — Per-segment tracker health.

Goal: localise tracker decoupling to a CLASS or regime.
Method: compute estimated-vs-GT per-frame motion-rate Pearson, bucketed by
which class dominates the frame (liver / gallbladder / tool pixel fraction).
Prediction: worst on gallbladder/tool-dominated frames, least bad on
liver-bed-dominated frames.

Per Wyrd's plan (workflow wx3zjzfyh, 2026-06-05):
  - PURE LOCAL CPU — no GPU needed
  - Runs on existing checkpoints + groundtruth + semantic masks
  - Output: Pearson + RPE as a function of dominant class

Usage:
  python diagnosis/phase1/test3_per_segment_health.py \
    --est est_c2w_data.txt \
    --gt groundtruth.txt \
    --semantic_dir <CRCD-Published>/<EP>/snippet_<SID>/semantic_instance/ \
    --out diagnosis/report/test3_<snippet>.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import infra
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from diagnosis.infra.motion_rate import (
    load_est_c2w_data, load_groundtruth_tum, per_frame_motion_rate,
)
from diagnosis.infra.frame_select import class_fractions_per_frame


def per_frame_rpe(est_c2w, gt_c2w, slice_range=None):
    """Per-frame Relative Pose Error (RPE).

    Standard ATE/RPE definition: error in inter-frame motion between est and gt.
    For frame i: rpe[i] = || relative_motion(est, i, i+1) - relative_motion(gt, i, i+1) ||

    Returns rpe_trans_mm (N-1,) and rpe_rot_deg (N-1,).
    """
    if slice_range is not None:
        s, e = slice_range
        est_c2w = est_c2w[s:e]
        gt_c2w = gt_c2w[s:e]
    est_motion = per_frame_motion_rate(est_c2w)
    gt_motion = per_frame_motion_rate(gt_c2w)
    return {
        'rpe_trans_mm': est_motion['trans_mm'] - gt_motion['trans_mm'],
        'rpe_rot_deg': est_motion['rot_deg'] - gt_motion['rot_deg'],
        'est_trans': est_motion['trans_mm'],
        'gt_trans': gt_motion['trans_mm'],
        'est_rot': est_motion['rot_deg'],
        'gt_rot': gt_motion['rot_deg'],
    }


def class_dominant_per_frame(class_fracs):
    """For each frame, return which class has the highest pixel fraction.
    Returns array of strings ('bg', 'liver', 'gallbladder', 'tool').
    """
    classes = list(class_fracs.keys())
    frac_matrix = np.stack([class_fracs[c] for c in classes], axis=1)  # (N, num_classes)
    dom_idx = np.argmax(frac_matrix, axis=1)
    return np.array([classes[i] for i in dom_idx])


def bucketed_pearson(est_motion, gt_motion, buckets, label):
    """For each unique value in `buckets`, compute Pearson(est, gt).
    Returns dict: {bucket_value: {pearson_r, p_value, spearman_r, n_frames}}
    """
    out = {}
    for b in np.unique(buckets):
        mask = (buckets == b)
        if mask.sum() < 3:
            continue
        est_b = est_motion[mask]
        gt_b = gt_motion[mask]
        r, p = pearsonr(est_b, gt_b)
        rho, _ = spearmanr(est_b, gt_b)
        out[str(b)] = {
            'pearson_r': float(r),
            'p_value': float(p),
            'spearman_r': float(rho),
            'n_frames': int(mask.sum()),
            'est_mean': float(est_b.mean()),
            'gt_mean': float(gt_b.mean()),
            'est_std': float(est_b.std()),
            'gt_std': float(gt_b.std()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--est', type=str, required=True, help='est_c2w_data.txt')
    ap.add_argument('--gt', type=str, required=True, help='groundtruth.txt (TUM format)')
    ap.add_argument('--semantic_dir', type=str, required=True,
                    help='dir of semantic_instance/frame_*.png (or semantic_class/*.png)')
    ap.add_argument('--out_csv', type=str, default='test3_per_segment_health.csv')
    ap.add_argument('--out_fig', type=str, default='test3_per_segment_health.png')
    ap.add_argument('--name', type=str, default='snippet')
    args = ap.parse_args()

    print(f'=== TEST 3 — Per-segment tracker health ({args.name}) ===')

    print(f'Loading est trajectory from {args.est}...')
    est_c2w = load_est_c2w_data(args.est)
    print(f'  {len(est_c2w)} estimated poses')

    print(f'Loading GT trajectory from {args.gt}...')
    gt_c2w = load_groundtruth_tum(args.gt)
    print(f'  {len(gt_c2w)} GT poses')

    # Align lengths
    n = min(len(est_c2w), len(gt_c2w))
    est_c2w = est_c2w[:n]
    gt_c2w = gt_c2w[:n]
    print(f'  Aligned to {n} frames')

    print(f'Loading semantic masks from {args.semantic_dir}...')
    class_fracs = class_fractions_per_frame(args.semantic_dir)
    # Align with poses
    n_sem = len(class_fracs['bg'])
    if n_sem < n:
        print(f'  WARN: only {n_sem} semantic masks but {n} poses; truncating')
        est_c2w = est_c2w[:n_sem]
        gt_c2w = gt_c2w[:n_sem]
        n = n_sem
    else:
        for k in class_fracs:
            class_fracs[k] = class_fracs[k][:n]

    # Per-frame motion rate
    print('Computing per-frame motion rate...')
    motion_data = per_frame_rpe(est_c2w, gt_c2w)
    est_trans = motion_data['est_trans']
    gt_trans = motion_data['gt_trans']
    est_rot = motion_data['est_rot']
    gt_rot = motion_data['gt_rot']

    # Class fractions are per-frame; motion is per-INCREMENT (N-1).  Align:
    # use the SECOND frame's class fraction for each increment.
    aligned_class_fracs = {k: v[1:] for k, v in class_fracs.items()}
    dom_classes = class_dominant_per_frame(aligned_class_fracs)

    # GLOBAL stats (whole snippet)
    global_r_trans, global_p_trans = pearsonr(est_trans, gt_trans)
    global_r_rot, _ = pearsonr(est_rot, gt_rot)
    print(f'\n== GLOBAL ==')
    print(f'  Pearson(est, gt) trans: r={global_r_trans:.3f} (p={global_p_trans:.2e})')
    print(f'  Pearson(est, gt) rot  : r={global_r_rot:.3f}')
    print(f'  est mean trans  : {est_trans.mean():.3f} mm/frame')
    print(f'  gt  mean trans  : {gt_trans.mean():.3f} mm/frame')

    # PER-DOMINANT-CLASS stats
    print(f'\n== PER-DOMINANT-CLASS Pearson(est, gt) trans ==')
    per_class = bucketed_pearson(est_trans, gt_trans, dom_classes, 'dominant_class')
    for cls in ['bg', 'liver', 'gallbladder', 'tool']:
        if cls in per_class:
            d = per_class[cls]
            print(f'  {cls:12s}: r={d["pearson_r"]:+.3f}  rho={d["spearman_r"]:+.3f}  '
                  f'n={d["n_frames"]:3d}  est_mean={d["est_mean"]:.2f}  gt_mean={d["gt_mean"]:.2f} mm/f')

    # PER-CLASS-FRACTION-DECILE stats (more granular than dominant-only)
    print(f'\n== Pearson(est, gt) trans by tool pixel fraction (deciles) ==')
    tool_frac = aligned_class_fracs['tool']
    tool_decile = np.clip((tool_frac * 10).astype(int), 0, 9)
    per_decile = bucketed_pearson(est_trans, gt_trans, tool_decile, 'tool_decile')
    for d in sorted(per_decile.keys(), key=int):
        x = per_decile[d]
        print(f'  decile {d}: r={x["pearson_r"]:+.3f}  n={x["n_frames"]:3d}')

    # Save CSV
    df = pd.DataFrame({
        'frame': np.arange(n - 1),
        'est_trans_mm': est_trans,
        'gt_trans_mm': gt_trans,
        'est_rot_deg': est_rot,
        'gt_rot_deg': gt_rot,
        'rpe_trans_mm': motion_data['rpe_trans_mm'],
        'rpe_rot_deg': motion_data['rpe_rot_deg'],
        'dominant_class': dom_classes,
        'bg_frac': aligned_class_fracs['bg'],
        'liver_frac': aligned_class_fracs['liver'],
        'gb_frac': aligned_class_fracs['gallbladder'],
        'tool_frac': aligned_class_fracs['tool'],
    })
    df.to_csv(args.out_csv, index=False)
    print(f'\nSaved per-frame data: {args.out_csv}')

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter trans by dominant class
    ax = axes[0, 0]
    colors = {'bg': 'gray', 'liver': 'darkred', 'gallbladder': 'darkgreen', 'tool': 'steelblue'}
    for cls in ['bg', 'liver', 'gallbladder', 'tool']:
        mask = (dom_classes == cls)
        if mask.sum() == 0:
            continue
        ax.scatter(gt_trans[mask], est_trans[mask], c=colors[cls], label=f'{cls} (n={mask.sum()})',
                   alpha=0.6, s=15)
    lim = max(est_trans.max(), gt_trans.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('GT translation magnitude (mm/frame)')
    ax.set_ylabel('Est translation magnitude (mm/frame)')
    ax.set_title(f'Per-frame motion: est vs GT  (global r={global_r_trans:.3f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Pearson per class bar
    ax = axes[0, 1]
    classes_order = ['bg', 'liver', 'gallbladder', 'tool']
    r_vals = [per_class.get(c, {}).get('pearson_r', np.nan) for c in classes_order]
    n_vals = [per_class.get(c, {}).get('n_frames', 0) for c in classes_order]
    bars = ax.bar(classes_order, r_vals, color=[colors[c] for c in classes_order])
    for bar, r, n in zip(bars, r_vals, n_vals):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.02 if r > 0 else r - 0.05,
                    f'r={r:+.2f}\nn={n}', ha='center', fontsize=10)
    ax.axhline(global_r_trans, color='red', linestyle='--', label=f'global r={global_r_trans:.3f}')
    ax.set_ylabel('Pearson(est trans, GT trans)')
    ax.set_title('Tracker health by dominant class')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='k', alpha=0.3)
    ax.legend()
    ax.grid(alpha=0.3)

    # RPE per class
    ax = axes[1, 0]
    rpe_trans = motion_data['rpe_trans_mm']
    rpe_by_class = {c: rpe_trans[dom_classes == c] for c in classes_order if (dom_classes == c).sum() > 0}
    ax.boxplot([rpe_by_class[c] for c in rpe_by_class.keys()],
               tick_labels=list(rpe_by_class.keys()))
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('RPE translation error (mm/frame, est - gt)')
    ax.set_title('Per-frame RPE by dominant class')
    ax.grid(alpha=0.3)

    # Pearson vs tool fraction (deciles)
    ax = axes[1, 1]
    deciles_sorted = sorted(per_decile.keys(), key=int)
    decile_r = [per_decile[d]['pearson_r'] for d in deciles_sorted]
    decile_n = [per_decile[d]['n_frames'] for d in deciles_sorted]
    ax.bar([int(d) / 10 for d in deciles_sorted], decile_r, width=0.08)
    for i, d in enumerate(deciles_sorted):
        ax.text(int(d) / 10, decile_r[i] + 0.02, f'n={decile_n[i]}', ha='center', fontsize=8)
    ax.set_xlabel('Tool pixel fraction (decile)')
    ax.set_ylabel('Pearson(est, gt)')
    ax.set_title('Tracker health vs tool pixel fraction')
    ax.set_ylim(-1, 1)
    ax.axhline(0, color='k', alpha=0.3)
    ax.axhline(global_r_trans, color='red', linestyle='--', alpha=0.5, label='global r')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'TEST 3 — Per-segment tracker health: {args.name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150, bbox_inches='tight')
    print(f'Saved figure: {args.out_fig}')

    # One-line verdict
    print(f'\n=== VERDICT ===')
    worst_cls = min(per_class.items(), key=lambda kv: kv[1]['pearson_r'])
    best_cls = max(per_class.items(), key=lambda kv: kv[1]['pearson_r'])
    print(f'  Worst-tracked class: {worst_cls[0]} (r={worst_cls[1]["pearson_r"]:.3f})')
    print(f'  Best-tracked class : {best_cls[0]} (r={best_cls[1]["pearson_r"]:.3f})')
    print(f'  Tracker-health gap : {best_cls[1]["pearson_r"] - worst_cls[1]["pearson_r"]:.3f}')
    if worst_cls[0] in ('tool', 'gallbladder') and best_cls[0] == 'liver':
        print(f'  PREDICTION CONFIRMED: worst on tool/GB-dominated, best on liver-bed.')
    elif best_cls[1]['pearson_r'] - worst_cls[1]['pearson_r'] < 0.1:
        print(f'  No clear class-dependent decoupling — tracker quality uniform across classes.')
    else:
        print(f'  Unexpected pattern — investigate.')


if __name__ == '__main__':
    main()
