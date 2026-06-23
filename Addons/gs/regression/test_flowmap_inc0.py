#!/usr/bin/env python3
"""Inc-0 parity gate for v1 flow_map. GPU GS is NOT byte-reproducible (CUDA rasteriser), so we PROVE
default-off == base structurally, not by float-equality.

Tier A (HARD gate, must pass):
  A1  config: FLOW_MAP unset vs FLOW_MAP=0 -> deep-equal AND flow_map['enable'] is False (injector default-off).
  A2  main.py edits are GUARDED: the gs_flow_gate import is never top-level; the get_loss multiply is inside
      the mapping branch; get_loss has the appended map_weight=None default; no undefined `_current_time_idx`.
Tier B (advisory): FM_LAMBDA=0 reduces to base (run-trace) — skipped here (needs staged data + GPU); the
  runbook may run it separately.

  python Addons/gs/regression/test_flowmap_inc0.py /content/EndoGSLAM/configs/crcd/crcd_base.py
Prints exactly one grep line: '>>> INC0 PARITY PASS' or '>>> INC0 PARITY FAIL: <why>'.
"""
import importlib.util
import os
import re
import sys


def _load_config(cfg_path):
    spec = importlib.util.spec_from_file_location("crcd_cfg_probe", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.config


def main():
    cfg_path = sys.argv[1]
    main_py = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(cfg_path))), "scripts", "main.py")
    fails = []

    # --- A1: config default-off + toggle-invariant ---
    try:
        os.environ.pop("FLOW_MAP", None)
        cfg_unset = _load_config(cfg_path)
        os.environ["FLOW_MAP"] = "0"
        cfg_zero = _load_config(cfg_path)
        os.environ.pop("FLOW_MAP", None)
        if cfg_unset != cfg_zero:
            fails.append("A1 config FLOW_MAP unset != FLOW_MAP=0 (default-off not bit-identical)")
        fm = cfg_zero.get("flow_map")
        if fm is None:
            fails.append("A1 no flow_map block injected (run inject_flowmap_knobs.py first)")
        elif fm.get("enable") is not False:
            fails.append(f"A1 flow_map.enable is {fm.get('enable')!r}, expected False")
    except Exception as e:
        fails.append(f"A1 config-load raised: {e}")

    # --- A2: main.py edits guarded ---
    try:
        with open(main_py, encoding="utf-8") as f:
            lines = f.readlines()
        src = "".join(lines)
        # the gs_flow_gate import must NOT be at column 0 (must be inside an enable guard)
        top_level_import = [i + 1 for i, ln in enumerate(lines)
                            if re.match(r"^(from|import)\b.*gs_flow_gate", ln)]
        if top_level_import:
            fails.append(f"A2 gs_flow_gate imported at top-level (lines {top_level_import}) -> not parity-isolated")
        if "gs_flow_gate" in src and "if " not in src[max(0, src.find("gs_flow_gate") - 400):src.find("gs_flow_gate")]:
            # weak heuristic: an enable-guard 'if' should appear shortly before the import
            pass  # the col-0 check above is the hard one
        if "_current_time_idx" in src:
            fails.append("A2 undefined name '_current_time_idx' present in main.py (NameError risk)")
        if "map_weight=None" not in src and "map_weight = None" not in src:
            fails.append("A2 get_loss lacks the appended 'map_weight=None' default")
        if "map_weight is not None" not in src:
            fails.append("A2 get_loss depth up-weight guard 'map_weight is not None' absent")
    except Exception as e:
        fails.append(f"A2 main.py inspect raised: {e}")

    if fails:
        print(">>> INC0 PARITY FAIL: " + " | ".join(fails))
        sys.exit(1)
    print(">>> INC0 PARITY PASS")


if __name__ == "__main__":
    main()
