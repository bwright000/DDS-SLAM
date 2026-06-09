"""timenet_weight_audit.py — settle 'dead weights' vs 'trained-silent'.

Audit intervention #1 (diagnosis/report/failure_mode_audit.md): the dead field
(dx==0) could be (a) DEAD/ZERO TimeNet weights, or (b) normal weights that the
trained MLP maps to ~0 output ('trained-silent'). dx_hook proves the OUTPUT is 0;
this proves which side of the WEIGHTS we're on — which discriminates the cause
(decay/weight-collapse vs gauge/no-reward). Bit-exact dx==0.0 (per test1 CSV)
strongly predicts (a); this confirms it.

Loads a DDS-SLAM checkpoint, finds every TimeNet (deformation MLP) parameter, and
reports per-layer L2 norm, max|.|, mean|.|, and a DEAD flag (max|.| < --dead_thresh).
Also reports the color/sdf decoder layers as a LIVE reference (those should be
healthy), so a dead TimeNet stands out against a live backbone in the same ckpt.

Usage:
  python diagnosis/infra/timenet_weight_audit.py --ckpt path/to/checkpointNNN.pt
  python diagnosis/infra/timenet_weight_audit.py --ckpt ckpt.pt --dead_thresh 1e-6 --json out.json
"""
import argparse
import json
import re

import torch


def _flatten(obj, prefix=""):
    """Yield (dotted_key, tensor) for every tensor anywhere in a nested
    dict/list checkpoint, so we don't depend on DDS-SLAM's exact save layout."""
    if torch.is_tensor(obj):
        if obj.is_floating_point():
            yield prefix.strip("."), obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}{i}.")


def _stats(t):
    t = t.detach().float().flatten()
    a = t.abs()
    return {
        "n": int(t.numel()),
        "l2": float(t.norm().item()),
        "max_abs": float(a.max().item()) if t.numel() else 0.0,
        "mean_abs": float(a.mean().item()) if t.numel() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dead_thresh", type=float, default=1e-6,
                    help="a param group is 'dead' if its max|.| < this")
    ap.add_argument("--time_re", default="time_net|timenet|deform",
                    help="regex (case-insensitive) selecting deformation params")
    ap.add_argument("--ref_re", default="color_net|sdf_net|colornet|sdfnet",
                    help="regex for healthy backbone params (LIVE reference)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    tensors = dict(_flatten(ck))
    if not tensors:
        raise SystemExit("No float tensors found in checkpoint — unexpected format.")

    time_re = re.compile(args.time_re, re.I)
    ref_re = re.compile(args.ref_re, re.I)
    time_keys = [k for k in tensors if time_re.search(k)]
    ref_keys = [k for k in tensors if ref_re.search(k)]

    def report(title, keys):
        print(f"\n=== {title} ({len(keys)} tensors) ===")
        rows = []
        for k in sorted(keys):
            s = _stats(tensors[k])
            dead = s["max_abs"] < args.dead_thresh
            print(f"  {'DEAD ' if dead else '     '}{k:55s} "
                  f"l2={s['l2']:.3e} max|.|={s['max_abs']:.3e} mean|.|={s['mean_abs']:.3e} n={s['n']}")
            rows.append({"key": k, **s, "dead": dead})
        return rows

    print(f"checkpoint: {args.ckpt}")
    print(f"top-level keys: {list(ck.keys()) if isinstance(ck, dict) else type(ck).__name__}")
    time_rows = report("TimeNet / deformation params", time_keys)
    ref_rows = report("Backbone reference (color/sdf) — expect LIVE", ref_keys)

    n_dead = sum(r["dead"] for r in time_rows)
    print("\n" + "=" * 60)
    if not time_rows:
        print("VERDICT: no TimeNet params matched — check --time_re against the key list above.")
    elif n_dead == len(time_rows):
        print(f"VERDICT: TimeNet is DEAD-WEIGHTS ({n_dead}/{len(time_rows)} layers max|.|<{args.dead_thresh:.0e}).")
        print("  => cause is weight collapse (decay / starvation), NOT 'trained-silent gauge'.")
    elif n_dead > 0:
        print(f"VERDICT: TimeNet PARTIALLY dead ({n_dead}/{len(time_rows)} layers). Inspect which layer zeroed.")
    else:
        ma = max((r["max_abs"] for r in time_rows), default=0.0)
        print(f"VERDICT: TimeNet weights are LIVE (max|.|={ma:.3e}) but output is ~0 (per dx_hook)")
        print("  => 'TRAINED-SILENT': the MLP maps healthy weights to ~0 => gauge / no-reward, not dead weights.")
    if ref_rows:
        ref_dead = sum(r["dead"] for r in ref_rows)
        print(f"  backbone reference: {ref_dead}/{len(ref_rows)} dead (expect 0 — confirms the ckpt itself is fine).")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"ckpt": args.ckpt, "dead_thresh": args.dead_thresh,
                       "timenet": time_rows, "backbone": ref_rows}, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
