#!/bin/bash
# ============================================================================
# T4 — ARM-2 Stage-1 TEACHER A/B (deformation-field revival). 2026-06-19.
#
# Fresh-instance, FULLY Drive-persisted (the fix for "runtime died -> lost the bake"):
#   env -> stage SemSup -> restore-OR-bake DINO + Δx* targets (persist BOTH to the dataset on
#   Drive for reuse) -> integration smoke -> teacher_off vs teacher_on -> field-warped pin-EPE
#   JUDGE (the arbiter) -> ship EVERYTHING to Drive.
#
#   bash Addons/colab/teacher_ab_t4_20260619.sh                 # n=1 smoke (seed 0): does the field revive?
#   SEEDS="0 1 2" bash Addons/colab/teacher_ab_t4_20260619.sh   # full n=3 (after the smoke confirms)
#
# Persists for reuse (so a dead runtime never costs the bake again):
#   DINO  -> MyDrive/Datasets/SemSup/v2_data/trial_3/dino/*_dino.npy
#   Δx*   -> MyDrive/Datasets/SemSup/v2_data/trial_3/deform/*_deform.npz
#   runs  -> MyDrive/Outputs/manual_cells/teacher_{off,on}/  (est+ckpt+6panel+render+pin_epe+figure)
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM; cd "$REPO"
SEEDS="${SEEDS:-0}"
GRID_SCALE="${GRID_SCALE:-3}"
DATASET=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3      # source-of-truth: dino/ + deform/ persist HERE
DD=$REPO/data/Super/trail_3
DRIVE=/content/drive/MyDrive/Outputs/teacher_ab_t4_${DATE}; mkdir -p "$DRIVE"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
say "=== TEACHER A/B (T4) start $(date -Iseconds) HEAD=$(git rev-parse --short HEAD) seeds=$SEEDS scale=$GRID_SCALE ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted. In a Colab cell: from google.colab import drive; drive.mount('/content/drive')"; exit 1; }
nvidia-smi -L || true

# torch>=2 python for DINO baking + the numpy Δx* bake (Colab system python is torch2 on the modern stack)
DINO_PY=python3
for p in python3 /usr/bin/python3 python; do "$p" -c "import torch,sys;sys.exit(0 if int(torch.__version__.split('.')[0])>=2 else 1)" 2>/dev/null && { DINO_PY=$p; break; }; done
say "torch>=2 python (DINO + Δx* bake) = $DINO_PY"

# ---------------------------------------------------------------------------
activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "env build (~15 min, colab_setup modern torch2 stack; T4=sm_75)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; fi
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
  python -c "import torch; assert torch.cuda.is_available()" || { say "env FAIL (no CUDA)"; exit 1; }
  local CC=$(python -c "import torch;print('%d%d'%torch.cuda.get_device_capability())" 2>/dev/null || echo 75)
  say "GPU sm_$CC"
  python - <<'PY' 2>/dev/null || { say "  tcnn probe failed on sm_$CC -> rebuild (~10 min)"; TCNN_CUDA_ARCHITECTURES=$CC pip install -q --force-reinstall --no-deps "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" || { say "TCNN rebuild FAIL"; exit 1; }; }
import torch, tinycudann as tcnn
e=tcnn.Encoding(3,{"otype":"HashGrid","n_levels":2,"n_features_per_level":2,"log2_hashmap_size":15,"base_resolution":16,"per_level_scale":1.5})
_=e(torch.rand(8,3,device='cuda'))
PY
  python -c "import lpips" 2>/dev/null || pip install -q lpips || true
}

stage_semsup(){
  if [ ! -d "$DD/rgb" ]; then
    [ -d "$DATASET/rgb" ] || { say "FATAL: SemSup source $DATASET/rgb missing on Drive"; exit 1; }
    mkdir -p "$REPO/data/Super"; cp -r "$DATASET" "$DD"; say "staged SemSup -> $DD ($(ls "$DD/rgb"/*left.png 2>/dev/null|wc -l) frames)"
  fi
  local MOG="$DD/depth/moge2"
  [ -d "$MOG" ] && [ "$(ls "$MOG"/*left_depth.npy 2>/dev/null|wc -l)" -ge 151 ] && { say "moge2 depth present"; return 0; }
  mkdir -p "$MOG"
  for c in "$DATASET/depth/MoGe2_trail3_20260608" "$DATASET/depth/moge2" /content/drive/MyDrive/Datasets/SemSup/MoGe2_trail3_20260608; do
    [ -d "$c" ] && { cp "$c"/*left_depth.npy "$MOG"/ 2>/dev/null && break; }; done
  local n=$(ls "$MOG"/*left_depth.npy 2>/dev/null|wc -l); say "moge2 depth: $n npy"
  [ "$n" -ge 151 ] || { say "FATAL: moge2 depth not found on Drive (need MoGe2_trail3_20260608)"; exit 1; }
}

ensure_dino(){
  local OUT="$DD/dino"
  [ "$(ls "$OUT"/*_dino.npy 2>/dev/null|wc -l)" -ge 151 ] && { say "DINO present ($(ls "$OUT"/*_dino.npy|wc -l))"; return 0; }
  if [ "$(ls "$DATASET/dino"/*_dino.npy 2>/dev/null|wc -l)" -ge 151 ]; then
    mkdir -p "$OUT"; cp "$DATASET/dino"/*_dino.npy "$OUT"/; say "DINO RESTORED from Drive ($(ls "$OUT"/*_dino.npy|wc -l))"; return 0; fi
  say "baking DINOv2 (C=384) -> $OUT (~few min)"
  $DINO_PY "$REPO/Addons/dino/generate_dino_features.py" --rgb_dir "$DD/rgb" --rgb_glob '*left.png' --out_dir "$OUT" --backbone dinov2_vits14 --fp32 2>&1 | tail -6
  [ "$(ls "$OUT"/*_dino.npy 2>/dev/null|wc -l)" -ge 151 ] || { say "FATAL DINO bake incomplete"; exit 1; }
  mkdir -p "$DATASET/dino"; cp "$OUT"/*_dino.npy "$DATASET/dino"/ && say "DINO PERSISTED to Drive dataset (reuse next time)"
}

ensure_deform(){
  local OUT="$DD/deform"
  [ "$(ls "$OUT"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ] && { say "Δx* targets present ($(ls "$OUT"/*_deform.npz|wc -l))"; return 0; }
  if [ "$(ls "$DATASET/deform"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ]; then
    mkdir -p "$OUT"; cp "$DATASET/deform"/*_deform.npz "$OUT"/; say "Δx* targets RESTORED from Drive ($(ls "$OUT"/*_deform.npz|wc -l)) — no re-bake"; return 0; fi
  say "baking Δx* targets (grid_scale $GRID_SCALE, fast matmul ~10-26 min) -> $OUT"
  $DINO_PY "$REPO/Addons/deform/generate_deform_targets.py" \
    --dino_dir "$DD/dino" --dino_glob '*_dino.npy' \
    --depth_dir "$DD/depth/moge2" --depth_glob '*left_depth.npy' \
    --out_dir "$OUT" --grid_scale "$GRID_SCALE" 2>&1 | tail -6
  [ "$(ls "$OUT"/*_deform.npz 2>/dev/null|wc -l)" -ge 151 ] || { say "FATAL Δx* bake incomplete"; exit 1; }
  say "validate Δx* vs held-out pins (expect valid-only reduction >=72%, cos>0.85):"
  $DINO_PY "$REPO/Addons/deform/validate_deform_targets.py" \
    --pts "$REPO/Addons/eval/gt_pins/trial_3_l_pts.npy" --deform_dir "$OUT" \
    --depth_dir "$DD/depth/moge2" --depth_glob '*left_depth.npy' 2>&1 | tee "$DRIVE/deform_validate.txt"
  mkdir -p "$DATASET/deform"; cp "$OUT"/*_deform.npz "$DATASET/deform"/ && say "Δx* targets PERSISTED to Drive dataset (a dead runtime never costs the bake again)"
}

# integration smoke: code couldn't be tested locally (no tcnn). PASS = constructs + trains + no crash/NaN.
smoke_teacher(){
  say "########## SMOKE: teacher_on path constructs+trains (~8 min timeout) ##########"
  local SM=output/_teacher_smoke OVR=/content/_teacher_smoke.yaml SLOG="$DRIVE/teacher_smoke.log"
  cat > "$OVR" <<YML
inherit_from: configs/Super/trail3_teacher_on.yaml
seed: 0
data:
  output: ${SM}
  exp_name: demo
YML
  timeout 480 python -W ignore - "$OVR" > "$SLOG" 2>&1 <<'PY'
import sys, runpy, torch
torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
cfg=sys.argv[1]; sys.argv=['ddsslam.py','--config',cfg]; runpy.run_path('ddsslam.py', run_name='__main__')
PY
  local TB=$(grep -c "Traceback" "$SLOG" 2>/dev/null); TB=${TB:-0}
  local NAN=$(grep -ciw "nan" "$SLOG" 2>/dev/null); NAN=${NAN:-0}
  local KF=$(grep -c "add keyframe" "$SLOG" 2>/dev/null); KF=${KF:-0}
  say "  smoke: tracebacks=$TB nan=$NAN keyframes=$KF"
  rm -rf "$SM"
  if [ "$TB" -eq 0 ] && [ "$NAN" -eq 0 ] && [ "$KF" -ge 1 ]; then say "  >>> SMOKE PASS: teacher path runs. Proceeding."; return 0; fi
  say "  >>> SMOKE FAIL — last 40 lines:"; tail -40 "$SLOG"; return 1
}

judge_one(){  # NAME CFG -> field-warped pin EPE (the arbiter) + 6-panel figure to Drive
  local NAME=$1 CFG=$2 RUN="output/$NAME/demo" DST="/content/drive/MyDrive/Outputs/manual_cells/$NAME"
  local CK=$(ls -t "$RUN"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CK" ] && [ -f "$RUN/est_c2w_data.txt" ] || { say "  JUDGE skip $NAME (no ckpt/est -> train failed?)"; return 0; }
  mkdir -p "$DST"
  say "  JUDGE $NAME (field-warped pin EPE; |Δx|>0 + reduction>>shuffled + cos>0 = ALIVE)"
  python "$REPO/Addons/eval/field_warped_pin_epe.py" --config "$CFG" \
    --checkpoint "$CK" --est_c2w "$RUN/est_c2w_data.txt" \
    --pts "$REPO/Addons/eval/gt_pins/trial_3_l_pts.npy" \
    --depth_dir "$DD/depth/moge2" --depth_glob '*left_depth.npy' \
    --fig_dir "$DST" --tag "$NAME" 2>&1 | tee "$DST/pin_epe.txt"
}

# ---- pipeline ----
stage_semsup
ensure_dino            # torch2; restore-or-bake + persist
ensure_deform          # numpy; restore-or-bake + VALIDATE + persist (the autosave that was missing)
activate_dds_env       # ddsslam + judge need torch2 + tcnn(sm_75)
smoke_teacher || { say "########## ABORTED at teacher smoke. Fix + re-launch (bake is now persisted). ##########"; exit 1; }

for S in $SEEDS; do
  for ARM in off on; do
    NAME="teacher_${ARM}"; [ "$S" != "0" ] && NAME="teacher_${ARM}_s${S}"
    CFG="configs/Super/trail3_teacher_${ARM}.yaml"
    say "########## CELL $NAME (seed $S, $ARM) ##########"
    bash "$REPO/Addons/colab/run_cell.sh" "$CFG" "$NAME" "$S"   # train -> 6-panel video -> render PSNR -> ship to Drive
    judge_one "$NAME" "$CFG"
  done
done

say "########## SUMMARY ##########"
for S in $SEEDS; do for ARM in off on; do
  NAME="teacher_${ARM}"; [ "$S" != "0" ] && NAME="teacher_${ARM}_s${S}"
  echo "--- $NAME ---"; grep -E "reduction|cos\(|Δx\| field|VERDICT|PSNR:" "/content/drive/MyDrive/Outputs/manual_cells/$NAME/pin_epe.txt" "/content/drive/MyDrive/Outputs/manual_cells/$NAME/render_metrics.txt" 2>/dev/null | sed 's#.*/##'
done; done
say "=== TEACHER A/B DONE. Pin-EPE+figures: MyDrive/Outputs/manual_cells/teacher_*  |  Δx* bake persisted: $DATASET/deform ==="
