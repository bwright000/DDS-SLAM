#!/usr/bin/env python3
"""General CRCD cross-snippet aggregator (method-agnostic, layout = <NAME>/payload.tgz).

WHY THIS EXISTS — aggregate_ab.py is HARDCODED to the DDS-SLAM A/B/C study:
  CRCD=[('base','c1_001_canon_base'),...], SEM=[('base','trail3_moge2_uncert_base'),...],
  globs '<cell>_s<seed>/payload.tgz', and parses 'render_metrics.txt'.
  None of that matches a per-method run-book layout where each snippet is one directory
  '<NAME>/payload.tgz' (NAME in C1_001,E3_005,C3_001,G3_001,C2_001) and the emitted files
  are 'render_eval.txt'/'render_eval.csv' (eval_rendering.py output) and 'sim3_metrics.txt'.
  This aggregator is parameterized over the snippet names and reads the files the run book
  actually emits, so the final cross-snippet (and, when --method-roots is given, cross-method)
  mean+/-std table is actually produced.

Inputs (in priority order, per snippet dir <ROOT>/<NAME>/):
  - render_eval.csv  (preferred; summary row written by eval_rendering.py --output_csv)
  - render_eval.txt  (fallback; stdout teed by the run book)
  - sim3_metrics.txt (Sim3 ATE rmse/mean/median/max mm + |Pearson| dom axis)
  - payload.tgz      (last resort: extract the above by suffix if loose files are absent)

Usage:
  # single method, cross-snippet table:
  python Addons/eval/aggregate_crcd_generic.py \
      --root /content/drive/MyDrive/Outputs/SNI-SLAM/crcd_<DATE> \
      --names C1_001 E3_005 C3_001 G3_001 C2_001 \
      --out  /content/drive/MyDrive/Outputs/SNI-SLAM/crcd_<DATE>/COMBINED_SUMMARY.txt

  # cross-method table (master orchestrator), one root per method:
  python Addons/eval/aggregate_crcd_generic.py \
      --method-roots SNI-SLAM=/content/drive/MyDrive/Outputs/SNI-SLAM/crcd_<DATE> \
                     SemGauss=/content/drive/MyDrive/Outputs/SemGauss-SLAM/crcd_<DATE> \
      --names C1_001 E3_005 C3_001 G3_001 C2_001 \
      --out /content/drive/MyDrive/Outputs/CROSS_METHOD_TABLE.txt
"""
import argparse
import glob
import os
import re
import tarfile

import numpy as np

DEFAULT_NAMES = ['C1_001', 'E3_005', 'C3_001', 'G3_001', 'C2_001']


def _read_loose_or_tar(snip_dir, basename):
    """Return text of <snip_dir>/<basename>, else first payload.tgz member ending in basename."""
    loose = os.path.join(snip_dir, basename)
    if os.path.isfile(loose):
        return open(loose, 'r', encoding='utf-8', errors='ignore').read()
    tgz = os.path.join(snip_dir, 'payload.tgz')
    if os.path.isfile(tgz):
        try:
            with tarfile.open(tgz, 'r:gz') as t:
                ms = [n for n in t.getnames() if n.endswith(basename)]
                if ms:
                    return t.extractfile(ms[0]).read().decode('utf-8', 'ignore')
        except Exception:
            pass
    return None


def render_vals(snip_dir):
    """(psnr, ssim, lpips, n_frames). Prefer render_eval.csv summary row, else parse render_eval.txt."""
    csv = _read_loose_or_tar(snip_dir, 'render_eval.csv')
    if csv:
        # eval_rendering.py CSV: per-frame rows + summary; pull the MEAN/avg row if present,
        # else average the numeric PSNR/SSIM/LPIPS columns over all data rows.
        import csv as _csv
        import io
        rows = list(_csv.reader(io.StringIO(csv)))
        if rows:
            hdr = [h.strip().lower() for h in rows[0]]
            def col(name):
                return hdr.index(name) if name in hdr else None
            ip, iss, il = col('psnr'), col('ssim'), col('lpips')
            if ip is not None:
                ps, ssv, lp = [], [], []
                for r in rows[1:]:
                    try:
                        ps.append(float(r[ip]))
                        if iss is not None:
                            ssv.append(float(r[iss]))
                        if il is not None:
                            lp.append(float(r[il]))
                    except (ValueError, IndexError):
                        continue
                if ps:
                    return (float(np.mean(ps)),
                            float(np.mean(ssv)) if ssv else None,
                            float(np.mean(lp)) if lp else None,
                            len(ps))
    # Fallback: parse the human summary lines from render_eval.txt.
    txt = _read_loose_or_tar(snip_dir, 'render_eval.txt')
    if txt:
        def g(key):
            # 'KEY: <digit>' — colon+space+digit REQUIRED, else 'LPIPS' first matches the availability
            # banner 'LPIPS: available' and greedily eats to the next digit -> LPIPS column ~1.0 garbage
            # in every SUMMARY table (same bug aggregate_ab.py:36 already fixed; this regressed it).
            m = re.search(key + r':\s+([0-9][0-9.]*)', txt)
            return float(m.group(1)) if m else None
        nf = g('Rendered') or g('frames')
        return (g('PSNR'), g('SSIM'), g('LPIPS'), int(nf) if nf else None)
    return (None, None, None, None)


def sim3_vals(snip_dir):
    """(ate_rmse, ate_mean, ate_max, |pearson|) from sim3_metrics.txt (mm), else Nones."""
    s = _read_loose_or_tar(snip_dir, 'sim3_metrics.txt')
    if not s:
        return (None, None, None, None)
    m = re.search(r'Sim3 ATE[^:]*:\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)\s*/\s*([0-9.]+)', s)
    pe = re.search(r'Pearson\|?\s*dom axis\s*:\s*([0-9.]+)', s)
    return ((float(m.group(1)), float(m.group(2)), float(m.group(4))) if m else (None, None, None)) + \
           ((float(pe.group(1)),) if pe else (None,))


def _stat(xs):
    xs = [v for v in xs if v is not None]
    return (np.mean(xs), np.std(xs), len(xs)) if xs else (None, None, 0)


def _fmt(xs, prec=2):
    m, sd, _ = _stat(xs)
    return f"{m:.{prec}f}+/-{sd:.{prec}f}" if m is not None else "--"


def aggregate_one(root, names):
    rows = []
    cols = {k: [] for k in ('psnr', 'ssim', 'lpips', 'ate_rmse', 'ate_mean', 'ate_max', 'pearson')}
    for nm in names:
        d = os.path.join(root, nm)
        if not os.path.isdir(d):
            rows.append((nm, 'MISSING', None, None, None, None, None, None))
            continue
        ps, ss, lp, nf = render_vals(d)
        ar, am, ax, pe = sim3_vals(d)
        rows.append((nm, nf, ps, ss, lp, ar, am, pe))
        for k, v in (('psnr', ps), ('ssim', ss), ('lpips', lp),
                     ('ate_rmse', ar), ('ate_mean', am), ('ate_max', ax), ('pearson', pe)):
            cols[k].append(v)
    return rows, cols


def render_table(title, rows, cols):
    L = [f"\n=== {title} ==="]
    L.append(f"  {'snippet':<10} {'frames':>7} {'PSNR':>7} {'SSIM':>6} {'LPIPS':>6} "
             f"{'ATE_rmse':>9} {'ATE_mean':>9} {'|Pear|':>7}")
    for nm, nf, ps, ss, lp, ar, am, pe in rows:
        def c(v, p=2):
            return f"{v:.{p}f}" if isinstance(v, float) else ('--' if v is None else str(v))
        L.append(f"  {nm:<10} {c(nf,0) if nf is not None else '--':>7} "
                 f"{c(ps):>7} {c(ss,3):>6} {c(lp,3):>6} {c(ar):>9} {c(am):>9} {c(pe,3):>7}")
    L.append(f"  {'MEAN+/-STD':<10} {'':>7} {_fmt(cols['psnr']):>7} {_fmt(cols['ssim'],3):>6} "
             f"{_fmt(cols['lpips'],3):>6} {_fmt(cols['ate_rmse']):>9} {_fmt(cols['ate_mean']):>9} "
             f"{_fmt(cols['pearson'],3):>7}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', help='single-method results root containing <NAME>/ dirs')
    ap.add_argument('--method-roots', nargs='+', default=None,
                    help='cross-method: METHOD=PATH entries')
    ap.add_argument('--names', nargs='+', default=DEFAULT_NAMES)
    ap.add_argument('--out', default=None, help='also write the table to this file')
    a = ap.parse_args()

    out_chunks = []
    if a.method_roots:
        for entry in a.method_roots:
            method, _, path = entry.partition('=')
            rows, cols = aggregate_one(path, a.names)
            out_chunks.append(render_table(f"{method}  ({path})", rows, cols))
    elif a.root:
        rows, cols = aggregate_one(a.root, a.names)
        out_chunks.append(render_table(os.path.basename(a.root.rstrip('/')), rows, cols))
    else:
        ap.error('provide --root or --method-roots')

    text = "\n".join(out_chunks) + "\n"
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w') as fh:
            fh.write(text)
        print(f"[aggregate] wrote {a.out}")


if __name__ == '__main__':
    main()
