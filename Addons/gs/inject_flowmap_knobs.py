#!/usr/bin/env python3
"""Idempotently append the v1 flow_map config block to crcd_base.py (env-driven; ALL defaults = base, so
flow_map-off == base regardless of injection state). Anchor-asserted; running twice is a no-op.
Mirrors the overnight_crcd.sh injector pattern.

  python Addons/gs/inject_flowmap_knobs.py /content/EndoGSLAM/configs/crcd/crcd_base.py
"""
import sys

MARKER = "# >>> v1 flow_map block (inject_flowmap_knobs) >>>"
BLOCK = '''

# >>> v1 flow_map block (inject_flowmap_knobs) >>>
# v1 MAPPING CATCH-UP: per-pixel depth up-weight where the scene moved. Every env knob defaults to base
# -> FLOW_MAP unset/0 => enable False => bit-identical to base (main.py reads via .get and no-ops).
config["flow_map"] = dict(
    enable        = bool(int(os.environ.get("FLOW_MAP", 0))),       # MASTER (0 = base; no import, no RAFT)
    lam           = float(os.environ.get("FM_LAMBDA", 1.0)),        # peak extra depth weight (w in [1,1+lam])
    deadband      = float(os.environ.get("FM_DEADBAND", 3.0)),      # Sampson px below = camera/noise
    soft_scale    = float(os.environ.get("FM_SOFT", 5.0)),          # flow ramp width (px)
    depth_deadband= float(os.environ.get("FM_DEPTH_DB", 2.0)),      # 3D surface-change deadband (metric)
    depth_soft    = float(os.environ.get("FM_DEPTH_SOFT", 4.0)),    # depth ramp width
    require_depth = bool(int(os.environ.get("FM_REQ_DEPTH", 1))),   # AND the depth gate (0 -> Sampson only)
    ref_stride    = int(os.environ.get("FM_REF_STRIDE", 8)),        # ref = max(0, t - ref_stride)
    raft_small    = bool(int(os.environ.get("FM_RAFT_SMALL", 0))),
    w_max         = float(os.environ.get("FM_WMAX", 3.0)),          # hard per-pixel clamp
    uniform_ctrl  = bool(int(os.environ.get("FM_UNIFORM", 0))),     # CONTROL arm: constant up-weight, no gating
    eval_holdout_every = int(os.environ.get("FM_HOLDOUT_EVERY", 0)),# 0 = no holdout (base); 5 = clean metric
)
# <<< v1 flow_map block <<<
'''


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        src = f.read()
    if MARKER in src:
        print(f"[inject] flow_map block already present in {path} -> no-op")
        return
    assert "config = dict(" in src, f"[inject] no 'config = dict(' anchor in {path} (wrong file?)"
    assert "import os" in src, f"[inject] crcd_base.py must import os (block uses os.environ): {path}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(src.rstrip() + "\n" + BLOCK)
    print(f"[inject] appended flow_map block to {path}")


if __name__ == "__main__":
    main()
