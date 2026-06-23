#!/usr/bin/env bash
# Fresh-box bootstrap: clone EndoGSLAM @ pinned base + lay down the GS-migration CRCD adapter (Phase-0).
# Idempotent. Stores NO authors' code in our repo — our 4 files are copied in, the 3 edits are PATCHED at
# runtime (EndoGSLAM has no LICENSE; caution C1). Run on the box that will train (Colab T4/A100).
#
#   bash Addons/gs/setup_endogslam_crcd.sh          # default ENDO_DIR=/content/EndoGSLAM
#   ENDO_DIR=/path/to/EndoGSLAM bash Addons/gs/setup_endogslam_crcd.sh
set -euo pipefail

ENDO_REPO="${ENDO_REPO:-https://github.com/Loping151/EndoGSLAM}"
ENDO_PIN="${ENDO_PIN:-6338c643f832f92ae161086bd26892994c761c5c}"   # base we derived the patches against
ENDO_DIR="${ENDO_DIR:-/content/EndoGSLAM}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OV="$SELF_DIR/overlay"

echo ">>> EndoGSLAM -> $ENDO_DIR  (pin $ENDO_PIN)"
if [ ! -d "$ENDO_DIR/.git" ]; then
  git clone "$ENDO_REPO" "$ENDO_DIR"
fi
cd "$ENDO_DIR"
git fetch -q origin || true
git checkout -q "$ENDO_PIN"

echo ">>> lay down CRCD adapter (our new files)"
install -D -m644 "$OV/datasets/gradslam_datasets/crcd.py" datasets/gradslam_datasets/crcd.py
install -D -m644 "$OV/configs/data/crcd.yaml"             configs/data/crcd.yaml
install -D -m644 "$OV/configs/crcd/crcd_base.py"          configs/crcd/crcd_base.py
install -D -m644 "$OV/scripts/eval_sim3_crcd.py"          scripts/eval_sim3_crcd.py
install -D -m644 "$OV/scripts/gs_eval.py"                 scripts/gs_eval.py

echo ">>> apply in-place edits (patch, not vendor)"
python3 "$SELF_DIR/apply_patches.py" "$ENDO_DIR"

echo ">>> lay down v1 flow_map (sensor + adapter + injector + parity test + runbook)"
install -D -m644 "$SELF_DIR/../motion/flow_track.py"        "$ENDO_DIR/Addons/motion/flow_track.py"
install -D -m644 "$SELF_DIR/../motion/gs_flow_gate.py"      "$ENDO_DIR/Addons/motion/gs_flow_gate.py"
install -D -m644 "$SELF_DIR/inject_flowmap_knobs.py"        "$ENDO_DIR/Addons/gs/inject_flowmap_knobs.py"
install -D -m644 "$SELF_DIR/apply_patches_flowmap.py"       "$ENDO_DIR/Addons/gs/apply_patches_flowmap.py"
install -D -m644 "$SELF_DIR/regression/test_flowmap_inc0.py" "$ENDO_DIR/Addons/gs/regression/test_flowmap_inc0.py"
install -D -m644 "$SELF_DIR/flow_map_ab_20260623.sh"        "$ENDO_DIR/Addons/gs/flow_map_ab_20260623.sh"
# package __init__ so `from Addons.motion.gs_flow_gate import ...` resolves (EndoGSLAM root is on sys.path)
for d in Addons Addons/motion Addons/gs Addons/gs/regression; do touch "$ENDO_DIR/$d/__init__.py"; done

echo ">>> apply v1 flow_map patches (parity-safe; no-op when flow_map disabled)"
python3 "$SELF_DIR/apply_patches_flowmap.py" "$ENDO_DIR"

# Upstream typo: datasets/ ships `_init_.py` (single underscores) not `__init__.py`, so the local
# `datasets` is only a namespace package -> on Colab the installed HuggingFace `datasets` package
# shadows it (ModuleNotFoundError: datasets.gradslam_datasets). Give it a real __init__.py so the
# local package wins at sys.path[0].
echo ">>> fix datasets package init (upstream _init_.py typo; HF 'datasets' shadows it on Colab)"
[ -f "$ENDO_DIR/datasets/__init__.py" ] || cp "$ENDO_DIR/datasets/_init_.py" "$ENDO_DIR/datasets/__init__.py"

echo ">>> verify (syntax)"
python3 -m py_compile \
  datasets/gradslam_datasets/crcd.py datasets/gradslam_datasets/__init__.py \
  datasets/gradslam_datasets/basedataset.py scripts/main.py \
  configs/crcd/crcd_base.py scripts/eval_sim3_crcd.py \
  Addons/motion/gs_flow_gate.py Addons/motion/flow_track.py Addons/gs/inject_flowmap_knobs.py
echo ">>> OK: EndoGSLAM + CRCD adapter ready at $ENDO_DIR"
