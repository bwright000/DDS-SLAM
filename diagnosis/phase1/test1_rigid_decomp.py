"""
TEST 1 — Rigid-component decomposition of Δx (the SMOKING GUN).

Reads dx_hook.py output NPZ dump.  For each frame, fits the best global
SE(3) (Procrustes/Kabsch) to {x_i → x_i + Δx_i}, splitting Δx into:
  G_t  = global rigid component (4×4 matrix)
  res  = non-rigid residual = Δx - rigid(x)

Compares {G_t} trajectory shape against {T_t^GT} (the GT camera trajectory).
If they correlate, the deformation field is carrying ego-motion — gauge
absorption CONFIRMED.

Pure CPU.  Per workflow wx3zjzfyh verification: 271 frames × 10K samples
in 0.11 s on Windows.

Usage:
  python diagnosis/phase1/test1_rigid_decomp.py \
    --dx_dir diagnosis/report/dx_dump_C1_001 \
    --gt groundtruth.txt \
    --out_csv diagnosis/report/test1_C1_001.csv \
    --out_fig diagnosis/report/test1_C1_001.png \
    --name C_1/001
"""

import argparse
import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
from diagnosis.infra.motion_rate import (
    load_groundtruth_tum, per_frame_motion_rate, pose_magnitude,
)


def kabsch(P, Q):
    """Best rigid SE(3) such that R @ P + t ≈ Q  (rows of P, Q are 3D points).
    Returns (R, t).
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    # Handle reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = centroid_Q - R @ centroid_P
    return R, t


def decompose_frame(x_canonical, delta_x):
    """For one frame: fit rigid SE(3) such that R @ x + t ≈ (x + Δx).
    Returns (R, t, residual) where residual has shape (N, 3).
    """
    # Flatten N rays × S samples → (N*S, 3)
    P = x_canonical.reshape(-1, 3)
    Q = (x_canonical + delta_x).reshape(-1, 3)
    R, t = kabsch(P, Q)
    Q_pred = (R @ P.T).T + t
    residual = Q - Q_pred
    return R, t, residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dx_dir', type=str, required=True, help='dx_hook output dir')
    parser.add_argument('--gt', type=str, required=True, help='groundtruth.txt (TUM)')
    parser.add_argument('--out_csv', type=str, required=True)
    parser.add_argument('--out_fig', type=str, required=True)
    parser.add_argument('--name', type=str, default='snippet')
    args = parser.parse_args()

    # Load dump summary
    summary_path = os.path.join(args.dx_dir, 'summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print(f'Loaded dump summary: {summary["n_frames"]} frames, {summary["rays_per_frame"]} rays')

    # Load all NPZ files
    files = sorted(glob.glob(os.path.join(args.dx_dir, 'frame_*.npz')))
    if not files:
        print(f'FATAL: no frame_*.npz in {args.dx_dir}'); sys.exit(1)
    n = len(files)
    print(f'Loading {n} frames from {args.dx_dir}')

    # Allocate G_t = per-frame rigid SE(3) (4x4)
    G = np.zeros((n, 4, 4))
    residual_norms = np.zeros(n)
    delta_x_norms = np.zeros(n)
    rigid_to_total_ratio = np.zeros(n)
    rigid_trans_mm = np.zeros(n)
    rigid_rot_deg = np.zeros(n)

    for i, f in enumerate(files):
        data = np.load(f)
        x = data['x_canonical']
        dx = data['delta_x']
        R, t, res = decompose_frame(x, dx)
        G[i, :3, :3] = R
        G[i, :3, 3] = t
        G[i, 3, 3] = 1.0
        residual_norms[i] = float(np.linalg.norm(res, axis=-1).mean())
        delta_x_norms[i] = float(np.linalg.norm(dx, axis=-1).mean())
        rigid_pred = (R @ x.reshape(-1, 3).T).T + t - x.reshape(-1, 3)
        rigid_to_total_ratio[i] = float(np.linalg.norm(rigid_pred, axis=-1).mean() / max(delta_x_norms[i], 1e-9))
        rigid_trans_mm[i] = float(np.linalg.norm(t) * 1000)
        # Geodesic rotation angle of R
        trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        rigid_rot_deg[i] = float(np.degrees(np.arccos(trace)))

    print(f'Per-frame Δx stats:')
    print(f'  Δx norm     : mean={delta_x_norms.mean():.4f}, max={delta_x_norms.max():.4f} m')
    print(f'  residual    : mean={residual_norms.mean():.4f}, max={residual_norms.max():.4f} m')
    print(f'  rigid/total : mean={rigid_to_total_ratio.mean():.3f}  (1.0 = pure rigid; 0 = pure deformation)')

    # Load GT camera trajectory
    gt_c2w = load_groundtruth_tum(args.gt)
    gt_c2w = gt_c2w[:n]  # align lengths
    n_aligned = min(n, len(gt_c2w))

    # Compare G_t to GT camera trajectory shape (translation increments)
    # G_t is canonical→frame-t in deformation field space; compare ΔG_t to camera motion increments.
    # ΔG_t = G_{t+1} - G_t (translation deltas)
    G_t_trans = G[:n_aligned, :3, 3]
    GT_t_trans = gt_c2w[:n_aligned, :3, 3]

    G_inc = np.linalg.norm(np.diff(G_t_trans, axis=0), axis=1) * 1000  # mm
    GT_inc = np.linalg.norm(np.diff(GT_t_trans, axis=0), axis=1) * 1000

    # Smooth length match
    n_inc = min(len(G_inc), len(GT_inc))
    G_inc = G_inc[:n_inc]
    GT_inc = GT_inc[:n_inc]
    pear_r, pear_p = pearsonr(G_inc, GT_inc)
    spear_rho, _ = spearmanr(G_inc, GT_inc)

    print(f'\n== GAUGE ABSORPTION CHECK ==')
    print(f'  Pearson(ΔG_t, ΔT_camera) on translation increments: r={pear_r:+.3f} (p={pear_p:.2e})')
    print(f'  Spearman(ΔG_t, ΔT_camera)                          : ρ={spear_rho:+.3f}')

    # Trajectory-shape comparison: subtract first pose, compare relative shapes
    # G is canonical-relative; treat its translation column as a "pose-like" trajectory.
    G_rel = G_t_trans - G_t_trans[0]
    GT_rel = GT_t_trans - GT_t_trans[0]
    # Compare via Procrustes alignment + Pearson on aligned coords
    R_align, t_align = kabsch(G_rel, GT_rel)
    G_aligned = (R_align @ G_rel.T).T + t_align
    shape_r_xyz = [pearsonr(G_aligned[:, k], GT_rel[:, k])[0] for k in range(3)]
    print(f'  Per-axis Pearson(G_aligned, GT_camera) after Procrustes: x={shape_r_xyz[0]:+.3f} y={shape_r_xyz[1]:+.3f} z={shape_r_xyz[2]:+.3f}')

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        'frame': np.arange(n),
        'dx_norm_m': delta_x_norms,
        'residual_norm_m': residual_norms,
        'rigid_to_total': rigid_to_total_ratio,
        'rigid_trans_mm': rigid_trans_mm,
        'rigid_rot_deg': rigid_rot_deg,
        'Gx': G_t_trans[:n, 0],
        'Gy': G_t_trans[:n, 1],
        'Gz': G_t_trans[:n, 2],
        'GTx': GT_t_trans[:n_aligned, 0].tolist() + [np.nan] * (n - n_aligned),
        'GTy': GT_t_trans[:n_aligned, 1].tolist() + [np.nan] * (n - n_aligned),
        'GTz': GT_t_trans[:n_aligned, 2].tolist() + [np.nan] * (n - n_aligned),
    })
    df.to_csv(args.out_csv, index=False)
    print(f'\nSaved CSV: {args.out_csv}')

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (1) Rigid translation vs GT camera translation — overlay
    ax = axes[0, 0]
    ax.plot(G_t_trans[:, 0] * 1000, label='G_t.x', color='red', alpha=0.7)
    ax.plot(GT_rel[:, 0] * 1000, label='GT.x', color='black', linestyle='--')
    ax.set_title('Rigid component trans X vs GT camera X (mm, frame 0 zeroed)')
    ax.set_xlabel('frame'); ax.set_ylabel('translation (mm)')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(G_t_trans[:, 1] * 1000, label='G_t.y', color='green', alpha=0.7)
    ax.plot(GT_rel[:, 1] * 1000, label='GT.y', color='black', linestyle='--')
    ax.set_title('Y component')
    ax.set_xlabel('frame'); ax.set_ylabel('mm')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(G_t_trans[:, 2] * 1000, label='G_t.z', color='blue', alpha=0.7)
    ax.plot(GT_rel[:, 2] * 1000, label='GT.z', color='black', linestyle='--')
    ax.set_title(f'Z component  (Pearson xyz: {shape_r_xyz[0]:+.2f}, {shape_r_xyz[1]:+.2f}, {shape_r_xyz[2]:+.2f})')
    ax.set_xlabel('frame'); ax.set_ylabel('mm')
    ax.legend(); ax.grid(alpha=0.3)

    # (2) Increment correlation scatter
    ax = axes[1, 0]
    ax.scatter(GT_inc, G_inc, alpha=0.4, s=8)
    lim = max(G_inc.max(), GT_inc.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('GT camera Δ (mm/frame)')
    ax.set_ylabel('Rigid Δ (mm/frame)')
    ax.set_title(f'Increment correlation: Pearson r={pear_r:+.3f}, ρ={spear_rho:+.3f}')
    ax.legend(); ax.grid(alpha=0.3)

    # (3) Rigid-to-total energy ratio over time
    ax = axes[1, 1]
    ax.plot(rigid_to_total_ratio, color='steelblue')
    ax.set_xlabel('frame')
    ax.set_ylabel('|rigid| / |Δx|')
    ax.set_title(f'Rigid-to-total Δx ratio (mean={rigid_to_total_ratio.mean():.2f})')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(alpha=0.3)

    # (4) Δx magnitude vs residual magnitude over time
    ax = axes[1, 2]
    ax.plot(delta_x_norms * 1000, label='|Δx| (mm)', color='orange')
    ax.plot(residual_norms * 1000, label='|non-rigid residual| (mm)', color='purple')
    ax.set_xlabel('frame'); ax.set_ylabel('mean magnitude (mm)')
    ax.set_title('Δx norm vs non-rigid residual norm')
    ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle(f'TEST 1 — Rigid-component decomposition (gauge absorption check): {args.name}',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=150, bbox_inches='tight')
    print(f'Saved figure: {args.out_fig}')

    # Test 1 verdict (Direction A — field absorbs ego-motion)
    print(f'\n=== TEST 1 VERDICT (Direction A: field absorbs ego-motion) ===')
    if abs(pear_r) > 0.6 and max(map(abs, shape_r_xyz)) > 0.7:
        direction_A = 'CONFIRMED'
        print(f'  >>> Direction A CONFIRMED <<<')
        print(f'  Δx rigid component tracks GT camera motion (Pearson > 0.7 on dominant axis)')
    elif abs(pear_r) > 0.3 or max(map(abs, shape_r_xyz)) > 0.5:
        direction_A = 'PARTIAL'
        print(f'  >>> Direction A PARTIAL <<<  (Pearson {pear_r:+.2f}, max axis {max(map(abs, shape_r_xyz)):.2f})')
    else:
        direction_A = 'NOT SUPPORTED'
        print(f'  Direction A NOT SUPPORTED (Pearson {pear_r:+.2f})')

    # =======================================================================
    # TEST 1' (Direction B — POSE absorbs deformation)
    # =======================================================================
    # The gauge mode is symmetric: a large deformation can be "explained" as
    # fake camera motion just as ego-motion can be "explained" as fake
    # deformation.  Wyrd's plan tests Direction A; we add B for completeness.
    #
    # Signal: est trajectory has motion BEYOND what GT says.  If that extra
    # motion correlates with deformation events (non-rigid Δx residual), the
    # tracker is misclassifying real deformation as camera motion.
    print(f'\n=== TEST 1\' VERDICT (Direction B: pose absorbs deformation) ===')

    # Need est_c2w trajectory.  Load from ckpt if dx_dir has frame_*.npz with
    # c2w field saved per frame (which dx_hook.py does).
    c2w_est = []
    for f in files:
        d = np.load(f)
        if 'c2w' in d.files:
            c2w_est.append(d['c2w'])
    if not c2w_est:
        print(f'  Could not extract est c2w from NPZ — skipping Test 1\'')
        return
    c2w_est = np.stack(c2w_est, axis=0)[:n_aligned]

    # Compute Sim3 alignment of est to GT (use a few-point Horn-style)
    # For the trajectory POSITIONS only (4x4 → just take [:3,3]).
    est_xyz = c2w_est[:, :3, 3]
    gt_xyz = gt_c2w[:n_aligned, :3, 3]
    # Procrustes (rotation+translation only — Sim3 minus scale for simplicity)
    R_a, t_a = kabsch(est_xyz, gt_xyz)
    est_aligned = (R_a @ est_xyz.T).T + t_a
    pose_residual = np.linalg.norm(est_aligned - gt_xyz, axis=1) * 1000  # mm per frame

    # Magnitude of non-rigid Δx per frame = "real deformation magnitude"
    # We have it from the loop above: residual_norms (mean non-rigid residual per frame)
    # Length-align
    pr = pose_residual[:n_aligned]
    rn = residual_norms[:n_aligned]

    # Correlate per-frame pose residual with per-frame non-rigid Δx magnitude
    pear_pose_def, p_pose_def = pearsonr(pr, rn * 1000)  # both in mm
    spear_pose_def, _ = spearmanr(pr, rn * 1000)
    print(f'  Pearson(|est - GT|, |non-rigid Δx|) = {pear_pose_def:+.3f} (p={p_pose_def:.2e})')
    print(f'  Spearman                            = {spear_pose_def:+.3f}')
    print(f'  Mean est-vs-GT pose residual : {pr.mean():.3f} mm/frame')
    print(f'  Mean non-rigid Δx (deformation): {rn.mean()*1000:.3f} mm')

    if abs(pear_pose_def) > 0.5:
        direction_B = 'CONFIRMED'
        print(f'  >>> Direction B CONFIRMED <<<')
        print(f'  Tracker pose drift correlates with deformation events.')
        print(f'  → Tracker is absorbing scene deformation as fake camera motion.')
    elif abs(pear_pose_def) > 0.25:
        direction_B = 'PARTIAL'
        print(f'  >>> Direction B PARTIAL <<<')
    else:
        direction_B = 'NOT SUPPORTED'
        print(f'  Direction B NOT SUPPORTED.')

    # =======================================================================
    # OVERALL VERDICT (combine A + B)
    # =======================================================================
    print(f'\n=== OVERALL GAUGE MODE VERDICT ===')
    print(f'  Direction A (field absorbs ego-motion) : {direction_A}')
    print(f'  Direction B (pose absorbs deformation): {direction_B}')
    if direction_A != 'NOT SUPPORTED' and direction_B != 'NOT SUPPORTED':
        print(f'  >>> BIDIRECTIONAL GAUGE ABSORPTION — both modes present')
        print(f'  → Combined effect explains observed tracker decoupling.')
    elif direction_A == 'NOT SUPPORTED' and direction_B != 'NOT SUPPORTED':
        print(f'  >>> Direction B DOMINANT — pose-absorbs-deformation is the failure mode')
        print(f'  → Tracker, not deformation field, is the gauge sink.')
        print(f'  → Fix priority: constrain tracker (e.g. pose regulariser) rather than field.')
    elif direction_A != 'NOT SUPPORTED' and direction_B == 'NOT SUPPORTED':
        print(f'  >>> Direction A DOMINANT — original gauge-absorption hypothesis')
        print(f'  → Fix priority: constrain deformation field (Phase 2 Test D capacity probe).')
    else:
        print(f'  >>> NEITHER direction confirmed — gauge mode unlikely.')
        print(f'  → Tracker decoupling stems from something else (sub-SNR? depth scale?).')

    # Append additional column to existing CSV
    df['pose_residual_mm'] = pose_residual[:n]
    df.to_csv(args.out_csv, index=False)


if __name__ == '__main__':
    main()
