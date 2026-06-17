#!/bin/bash
# run_cell.sh <config.yaml> <name> [seed]
# ONE manual cell, done right = the runbook's run_one WITHOUT the smoke gate / parallelism:
#   train (TF32 off + seed, no model edit) -> 6-panel inline video -> render PSNR/SSIM/LPIPS
#   (+ scale-corrected Sim3 ATE for CRCD) -> ship est + metrics + video to Drive.
# Use this for ANY manual run so metrics + video are NEVER forgotten -- a bare `python ddsslam.py`
# produces neither. Auto-detects CRCD vs SemSup from the config's datadir.
#   bash Addons/colab/run_cell.sh configs/CRCD/c1_001_canon_base.yaml base_v0
set -uo pipefail
CFG=$1; NAME=$2; SEED=${3:-0}
REPO=/content/DDS-SLAM; cd "$REPO"
OUT="output/$NAME"; RUN="$OUT/demo"; OVR="/content/_cell_${NAME}.yaml"
DST="/content/drive/MyDrive/Outputs/manual_cells/$NAME"; mkdir -p "$DST"

# --- resolve datadir + type + paper-ref sequence from the inherit chain (3 lines) ---
{ read DDIR; read VT; read SEQ; } < <(python - "$CFG" <<'PY'
import yaml, sys, os
def load(p):
    d = yaml.safe_load(open(p)); b = load(d['inherit_from']) if 'inherit_from' in d else {}
    def m(a, b):
        o = dict(a)
        for k, v in b.items(): o[k] = m(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
        return o
    return m(b, d)
c = load(sys.argv[1]); dd = c.get('data', {}).get('datadir', '').rstrip('/'); snip = os.path.basename(dd)
if 'CRCD' in dd:   print(dd); print('crcd');  print(f'CRCD ({snip})')
elif 'Super' in dd or 'trail' in dd: print(dd); print('super'); print('Lab1 (trail3)')
else:              print(dd); print('stereomis'); print('StereoMIS (P2_1)')
PY
)
echo "[run_cell] $NAME | cfg=$CFG | datadir=$DDIR | type=$VT | seq=$SEQ | seed=$SEED"
[ -n "$DDIR" ] && [ -d "$DDIR" ] || { echo "[run_cell] FATAL: datadir '$DDIR' not found"; exit 1; }

# --- 1. train (TF32 OFF + seed via override config; repo model untouched) ---
cat > "$OVR" <<YML
inherit_from: $CFG
seed: $SEED
data:
  output: $OUT
  exp_name: demo
YML
python -W ignore - "$OVR" <<'PY'
import sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
sys.argv = ['ddsslam.py', '--config', sys.argv[1]]; runpy.run_path('ddsslam.py', run_name='__main__')
PY
[ -f "$RUN/est_c2w_data.txt" ] || { echo "[run_cell] FATAL: no est_c2w_data.txt -> train failed"; exit 1; }

# --- 2. 6-panel video (uncert panel auto-included if the run wrote sigma^2) ---
UNC=""; [ -d "$OUT/uncert" ] && UNC="--uncert_dir $OUT/uncert"
NJPG=$(ls "$OUT"/[0-9]*.jpg 2>/dev/null | wc -l); echo "[run_cell] rendered frames on disk: $NJPG"
if [ "$VT" = "crcd" ]; then
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$DDIR/video_frames" --rgb_input_pattern '*l.png' \
    --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DDIR/depth" --depth_output_dir "$OUT/depth" --depth_norm robust \
    --seg_dir "$DDIR/semantic_class" --seg_pattern '*.png' --skip_raw_seg --seg_classmap $UNC \
    --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DDIR/groundtruth.txt" --trajectory_raw \
    --output "$DST/${NAME}_6panel.mp4" --fps 15 2>&1 | tail -3 || echo "WARN video"
else
  python Addons/viz/generate_video.py \
    --rgb_input_dir "$DDIR/rgb" --rgb_input_pattern '*left.png' \
    --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' \
    --depth_input_dir "$DDIR/depth/moge2" --depth_output_dir "$OUT/depth" --depth_norm robust \
    --seg_dir "$DDIR/seg/png_masks" --seg_pattern '*left.png' --skip_raw_seg --skip_horn_traj $UNC \
    --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DDIR/groundtruth.txt" --trajectory_raw \
    --output "$DST/${NAME}_6panel.mp4" --fps 15 2>&1 | tail -3 || echo "WARN video"
fi

# --- 3. metrics: render PSNR/SSIM/LPIPS always; Sim3 ATE for CRCD (real GT) ---
echo "===== METRICS: $NAME ====="
GTDIR="$DDIR/rgb"; [ "$VT" = "crcd" ] && GTDIR="$DDIR/video_frames"
python Addons/eval/eval_rendering.py --gt_dir "$GTDIR" --render_dir "$OUT" --name "$NAME" --sequence "$SEQ" \
  > "$DST/render_metrics.txt" 2>&1 || echo "WARN render-eval"
grep -E "Rendered:|PSNR:|SSIM:|LPIPS:" "$DST/render_metrics.txt" || cat "$DST/render_metrics.txt" | tail -3
if [ "$VT" = "crcd" ]; then
  python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DDIR/groundtruth.txt" --name "$NAME" \
    | tee "$DST/sim3_metrics.txt"
fi

# --- 4. ship the small artefacts to Drive (renders stay on /content) ---
cp "$RUN/est_c2w_data.txt" "$RUN/output.txt" "$DST/" 2>/dev/null || true
CK=$(ls -t "$RUN"/checkpoint*.pt 2>/dev/null | head -1); [ -n "$CK" ] && cp "$CK" "$DST/checkpoint.pt"
echo "[run_cell] DONE -> $DST  (6panel.mp4 + render_metrics.txt + sim3_metrics.txt + est)"
