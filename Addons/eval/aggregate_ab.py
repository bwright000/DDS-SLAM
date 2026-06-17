#!/usr/bin/env python3
"""Cross-shard aggregator for the base / geo(v1) / dino(v2) A/B/C.

When the matrix is split across multiple T4 Colab instances (SHARD=i/N), every instance writes the
SAME Drive results dir but each instance's end-of-run summary only sees its own slice. Run this ONCE
after all shards finish to combine them: it reads every <cell>_s<seed>/payload.tgz, extracts
sim3_metrics.txt (CRCD Sim3 ATE + |Pearson|) and render_metrics.txt (CRCD + SemSup PSNR/SSIM/LPIPS),
and prints the full n=3 table.

    python Addons/eval/aggregate_ab.py /content/drive/MyDrive/Outputs/a100_dino_ab_20260617

Parsing is frame-count aware: SemSup seeds with <100 rendered frames (partial/old-bug runs) are
EXCLUDED and flagged, so a 3-frame or 84-frame run can't silently skew the mean.
"""
import sys
import os
import glob
import re
import tarfile
import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'


def member_text(tgz, suffix):
    """Return the text of the first archived file ending in `suffix`, or None."""
    try:
        with tarfile.open(tgz, 'r:gz') as t:
            ms = [n for n in t.getnames() if n.endswith(suffix)]
            return t.extractfile(ms[0]).read().decode('utf-8', 'ignore') if ms else None
    except Exception:
        return None


def num(s, key):
    # 'KEY: <digit>' — the colon+space+digit requirement skips '[LPIPS]', 'v[0.1]', 'LPIPS: available'.
    m = re.search(key + r':\s+([0-9][0-9.]*)', s) if s else None
    return float(m.group(1)) if m else None


def sim3_vals(s):
    """(ate_mean, ate_max, |pearson|) from sim3_ate.py output, else (None,None,None)."""
    if not s:
        return (None, None, None)
    m = re.search(r'Sim3 ATE[^:]*:\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)', s)
    pe = re.search(r'Pearson\|?\s*dom axis\s*:\s*([0-9.]+)', s)
    return (float(m.group(2)) if m else None,
            float(m.group(4)) if m else None,
            float(pe.group(1)) if pe else None)


def seeds(cell):
    return sorted(glob.glob(os.path.join(ROOT, cell + '_s*')))


def stat(xs):
    xs = [v for v in xs if v is not None]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)


def fmt(xs):
    m, sd, _ = stat(xs)
    return f"{m:.2f}+/-{sd:.2f}" if m is not None else "--"


def fmt3(xs):
    m, _, _ = stat(xs)
    return f"{m:.3f}" if m is not None else "--"


CRCD = [('base', 'c1_001_canon_base'), ('geo(v1)', 'c1_001_canon_uncert'),
        ('dino(v2)', 'c1_001_canon_uncert_dino')]
SEM = [('base', 'trail3_moge2_uncert_base'), ('geo(v1)', 'trail3_moge2_uncert'),
       ('dino(v2)', 'trail3_moge2_uncert_dino')]

print(f"\nAggregating shards under: {ROOT}")

print("\n=== CRCD c1_001 — Sim3 ATE (mm) + |Pearson|dom (scale-free) + render ===")
for tag, cell in CRCD:
    am, ax, pe, ps, ss, lp, nf = ([] for _ in range(7))
    for d in seeds(cell):
        p = os.path.join(d, 'payload.tgz')
        if not os.path.isfile(p):
            continue
        a, x, pr = sim3_vals(member_text(p, 'sim3_metrics.txt'))
        am.append(a); ax.append(x); pe.append(pr)
        rm = member_text(p, 'render_metrics.txt')
        ps.append(num(rm, 'PSNR')); ss.append(num(rm, 'SSIM'))
        lp.append(num(rm, 'LPIPS')); nf.append(num(rm, 'Rendered'))
    frames = int(stat(nf)[0]) if stat(nf)[0] else 0
    print(f"  {tag:<9} seeds={len(seeds(cell))}  ATE_mean={fmt(am)}  ATE_max={fmt(ax)}  "
          f"|Pear|={fmt3(pe)}  | render PSNR={fmt(ps)} SSIM={fmt3(ss)} LPIPS={fmt3(lp)} (frames~{frames})")

print("\n=== SemSup trail3 — render PSNR/SSIM/LPIPS (partial seeds excluded) ===")
for tag, cell in SEM:
    ps, ss, lp, part = [], [], [], []
    for d in seeds(cell):
        rm = member_text(os.path.join(d, 'payload.tgz'), 'render_metrics.txt')
        n = num(rm, 'Rendered')
        if rm and num(rm, 'PSNR') is not None and (n or 0) >= 100:
            ps.append(num(rm, 'PSNR')); ss.append(num(rm, 'SSIM')); lp.append(num(rm, 'LPIPS'))
        elif n is not None and n < 100:
            part.append(f"{os.path.basename(d)}={int(n)}f")
    flag = f"   [excluded partial: {', '.join(part)}]" if part else ""
    print(f"  {tag:<9} n={stat(ps)[2]}  PSNR={fmt(ps)}  SSIM={fmt3(ss)}  LPIPS={fmt3(lp)}{flag}")
print()
