#!/usr/bin/env python
"""figures/ba_noise_floor.png — the two-panel evidence figure for sec:ba_floor.
Left: per-cycle keyframe displacement (mean |dt| per BA cycle) across runs, grouped by the
bundle-adjustment pose learning rate — flat at the Adam floor regardless of arm/sequence,
tenfold lower under the corrected step. Right: the trajectory fingerprint — tracked step size
by frame phase relative to the 5-frame keyframe cadence, normalised by each run's interior
mean; the accumulated keyframe drift lands on phase 4 and the correction removes it.
Data harvested from local result drops (run.log [ba] lines + est_c2w_data.txt)."""
import os, re, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#202124'; MUT = '#5f6368'; GRID = '#e8eaed'
BLUE = '#4a72b0'   # corrected (1e-5)
WARM = '#c2571a'   # released (1e-4)
plt.rcParams.update({'font.family': 'Arial', 'text.color': INK, 'axes.edgecolor': MUT,
                     'axes.labelcolor': INK, 'xtick.color': MUT, 'ytick.color': MUT})

DROPS = [
    # (glob, lr_group)  -- released 1e-4 arms
    (r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_basen3_20260707\E3_005_abl_base_s*", '1e-4'),
    (r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_uncvotefba_20260707\*_uncvote_fba_s0", '1e-4'),
    (r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_extra_20260707\C2_001_abl_base_s0", '1e-4'),
    (r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_extra_20260707\G3_001_abl_base_s0", '1e-4'),
    # corrected 1e-5 arms
    (r"C:\Users\benli\OneDrive\Desktop\Results\rect_bestbase_calm_20260708-20260708T201452Z-3-001\rect_bestbase_calm_20260708\*_s0", '1e-5'),
    (r"C:\Users\benli\OneDrive\Desktop\Results\drive-download-20260711T053215Z-2-001\rect_bestbase_bdds_final_20260709\*_s0", '1e-5'),
]

def kfdt(run_log):
    try: t = open(run_log, encoding='utf-8', errors='ignore').read()
    except OSError: return None
    v = [float(x) for x in re.findall(r'kf\|dt\| mean=([\d.]+)', t)]
    return float(np.mean(v)) if len(v) >= 10 else None

pts = {'1e-4': [], '1e-5': []}
for pat, grp in DROPS:
    for d in sorted(glob.glob(pat)):
        m = kfdt(os.path.join(d, 'run.log'))
        if m: pts[grp].append((os.path.basename(d), m))
print("harvested:", {k: len(v) for k, v in pts.items()})
for k, v in pts.items():
    for n, m in v: print(f"  [{k}] {n}: {m:.3f}")

# phase profiles (step/interior-mean by phase), representative pairs
GT = {'C1': r"F:\Datasets\CRCD-Published\C_1\snippet_001\groundtruth.txt",
      'E3': r"F:\Datasets\CRCD-Published\E_3\snippet_005\groundtruth.txt"}
def phases(est, sc=1.0):
    P = np.loadtxt(est)[:, [3, 7, 11]]
    s = np.linalg.norm(np.diff(P, axis=0), axis=1) * 1000 * sc
    tr = np.where(s > 1e-6)[0]
    ph = [s[tr[(tr % 5) == p]].mean() for p in range(5)]
    interior = np.mean(ph[:4])
    return [x / interior for x in ph]

PROF = [
    ('E3 base (released)', r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_basen3_20260707\E3_005_abl_base_s1\est_c2w_data.txt", WARM, '-'),
    ('C1 (released)', r"F:\Datasets\Resi;ts\drive-download-20260708T062006Z-3-001\rect_bestbase_uncvotefba_20260707\C1_001_uncvote_fba_s0\est_c2w_data.txt", WARM, '--'),
    ('E3 (corrected)', r"C:\Users\benli\OneDrive\Desktop\Results\rect_bestbase_calm_20260708-20260708T201452Z-3-001\rect_bestbase_calm_20260708\E3_005_uncvote_fba_calm_s0\est_c2w_data.txt", BLUE, '-'),
    ('C1 (corrected)', r"C:\Users\benli\OneDrive\Desktop\Results\rect_bestbase_calm_20260708-20260708T201452Z-3-001\rect_bestbase_calm_20260708\C1_001_uncvote_fba_calm_s0\est_c2w_data.txt", BLUE, '--'),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0), dpi=200)
# --- left: the floor ---
for grp, col, xoff in [('1e-4', WARM, 0), ('1e-5', BLUE, 1)]:
    ys = [m for _, m in pts[grp]]
    xs = np.full(len(ys), xoff) + np.linspace(-0.13, 0.13, max(len(ys), 2))[:len(ys)]
    ax1.scatter(xs, ys, s=42, color=col, zorder=3, edgecolors='white', linewidths=1)
    ax1.text(xoff + 0.2, np.median(ys), f"median {np.median(ys):.3f}", ha='left', va='center',
             fontsize=9, color=col, fontweight='bold')
ax1.axhline(1e-4 * 1e3 * np.sqrt(3), color=WARM, lw=1, ls=':', alpha=0.7)
ax1.text(0.52, 1e-4 * 1e3 * np.sqrt(3) * 1.08, r'$\eta\sqrt{3}$ (single Adam step, $\eta=10^{-4}$)',
         fontsize=8, color=MUT, ha='center')
ax1.set_yscale('log'); ax1.set_xticks([0, 1])
ax1.set_xticklabels(['released\npose lr $10^{-4}$', 'corrected\npose lr $10^{-5}$'], fontsize=9)
ax1.set_ylabel('keyframe displacement per BA cycle (mm)', fontsize=9)
ax1.set_title('The optimiser floor: every run, every sequence', fontsize=10, color=INK)
ax1.grid(axis='y', color=GRID, lw=0.7); ax1.set_axisbelow(True)
for s in ('top', 'right'): ax1.spines[s].set_visible(False)
# --- right: the fingerprint ---
ends = {WARM: [], BLUE: []}
for lab, est, col, ls in PROF:
    y = phases(est)
    ax2.plot(range(5), y, ls, color=col, lw=2, marker='o', ms=5,
             markeredgecolor='white', markeredgewidth=0.8)
    ends[col].append(y[4])
ax2.text(4.15, np.mean(ends[WARM]), 'released\n(C1 dashed, E3 solid)', fontsize=8.5,
         color=WARM, va='center', fontweight='bold')
ax2.text(4.15, np.mean(ends[BLUE]) - 0.12, 'corrected', fontsize=8.5, color=BLUE,
         va='top', fontweight='bold')
ax2.axhline(1.0, color=MUT, lw=0.8, ls=':')
ax2.set_xticks(range(5)); ax2.set_xticklabels(['0', '1', '2', '3', '4\n(onto keyframe)'], fontsize=9)
ax2.set_xlim(-0.3, 6.3)
ax2.set_xlabel('frame phase relative to the 5-frame keyframe cadence', fontsize=9)
ax2.set_ylabel('tracked step / interior mean', fontsize=9)
ax2.set_title('The trajectory fingerprint and its removal', fontsize=10, color=INK)
ax2.grid(axis='y', color=GRID, lw=0.7); ax2.set_axisbelow(True)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)
fig.tight_layout()
out = r"c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\figures\ba_noise_floor.png"
fig.savefig(out, bbox_inches='tight', facecolor='white')
print("wrote", out)
