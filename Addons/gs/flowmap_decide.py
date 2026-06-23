#!/usr/bin/env python3
"""v1 flow_map cross-arm DECISION. Reads base/flowmap/unictrl metrics_split.json (n=3 seeds) and applies
the spec's gate: KEEP v1 IFF flowmap HELD-OUT∩DYNAMIC PSNR beats BOTH base AND the uniform-control
(n=3 non-overlapping) AND the global-blur guard holds (flowmap held-out-static PSNR >= base - tau).
A win over base but NOT over the uniform control = a global Adam LR effect, NOT localized catch-up.

  python Addons/gs/flowmap_decide.py experiments/CRCD_base [C2_001]
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "experiments/CRCD_base"
SNIP = sys.argv[2] if len(sys.argv) > 2 else None
TAU = float(os.environ.get("FM_TAU", 0.2))


def collect(arm):
    pat = f"{ROOT}/{SNIP + '_' if SNIP else '*_'}{arm}_s*/metrics_split.json"
    return [json.load(open(j, encoding="utf-8")) for j in sorted(glob.glob(pat))]


def stat(rows, sub, key):
    v = [r[sub][key] for r in rows
         if r.get(sub, {}).get('n', 0) > 0 and r[sub][key] == r[sub][key]]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (float('nan'), float('nan'), 0)


def main():
    arms = {a: collect(a) for a in ('base', 'flowmap', 'unictrl')}
    print("=" * 72); print("  v1 flow_map DECISION  (HEADLINE = HELD-OUT ∩ DYNAMIC render)"); print("=" * 72)
    for a in ('base', 'flowmap', 'unictrl'):
        dm, ds, k = stat(arms[a], 'heldout_dynamic', 'PSNR')
        sm, _, _ = stat(arms[a], 'heldout_static', 'PSNR')
        print(f"  {a:9s} dynPSNR {dm:7.3f} ± {ds:.3f} (n={k})   staticPSNR {sm:7.3f}")
    bd, bds, _ = stat(arms['base'], 'heldout_dynamic', 'PSNR')
    fd, fds, _ = stat(arms['flowmap'], 'heldout_dynamic', 'PSNR')
    ud, uds, _ = stat(arms['unictrl'], 'heldout_dynamic', 'PSNR')
    bs, _, _ = stat(arms['base'], 'heldout_static', 'PSNR')
    fs, _, _ = stat(arms['flowmap'], 'heldout_static', 'PSNR')
    beats_base = (fd - fds) > (bd + bds)          # n=3 non-overlapping
    beats_uni = (fd - fds) > (ud + uds)
    guard_ok = (fs == fs) and (bs == bs) and (fs >= bs - TAU)
    print("-" * 72)
    print(f"  flowmap dynPSNR beats base (non-overlap) : {beats_base}   ({fd:.3f} vs {bd:.3f})")
    print(f"  flowmap dynPSNR beats UNIFORM-control     : {beats_uni}    ({fd:.3f} vs {ud:.3f})  <- specificity")
    print(f"  GUARD flowmap static >= base - {TAU:.2f}       : {guard_ok}    ({fs:.3f} vs {bs:.3f})")
    keep = bool(beats_base and beats_uni and guard_ok)
    print("=" * 72)
    print("  VERDICT: " + ("KEEP v1 — localized catch-up is real."
          if keep else "NOT a v1 win -> static-chase ceiling: escalate to v2 (deform field) or retune lam/deadbands."))
    print("=" * 72)


if __name__ == '__main__':
    main()
