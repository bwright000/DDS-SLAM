#!/bin/bash
# ============================================================================
# run_semanticsuper.sh - Arm-4 method #4 (Semantic-SuPer) onboarding.
#   PHASE A (this file): reproduce the method on ITS OWN dataset (the "Super" dVRK
#   ex-vivo trails) via the UPSTREAM tracker Python-SuPer/run_semantic_super.py ->
#   native Table I 2D PIXEL REPROJECTION ERROR (all-points / edge-points, mean(std)).
#   This is "Path A" - the ONLY path that supports a Semantic-SuPer reproduction claim
#   (the DDS render-PSNR data-port is NOT a reproduction + walks on these static-camera
#   trails). Camera is STATIC per trail -> NO ATE, NO PSNR.
#   PHASE B (crcd): STUB - deferred (ontology {tool,beef,chicken} vs CRCD 4-class +
#   CRCD has no reproj GT; both are user decisions).
#
#   Mirrors run_sgsslam.sh + CONTRACT.md (CLI verbs env|repro|crcd|eval|all). The older
#   SemanticSuPer.md plumbing (--phase A|B, render_metrics.txt, aggregate_ab.py) is DEPRECATED.
#
#   ENV-SS = an ISOLATED old stack (py3.8/torch1.11+cu113/pytorch3d0.6.2 Pulsar), NEVER with
#   the DDS modern torch2 env. Pinned wheels. T4 (sm_75) is the SAFE GPU (cu113 predates A100).
#
#   Usage:  bash Addons/colab/run_semanticsuper.sh env            # build ENV-SS (one-time, iterates)
#           MONO2_CKPT=<dir> bash Addons/colab/run_semanticsuper.sh repro            # trail_3 (Lab1, only local)
#           MONO2_CKPT=<dir> bash Addons/colab/run_semanticsuper.sh repro trail_8    # one trail (needs Drive data)
#           MONO2_CKPT=<dir> bash Addons/colab/run_semanticsuper.sh repro all4       # all 4 trails (need Drive data)
#           bash Addons/colab/run_semanticsuper.sh eval                              # aggregate reproj vs Table I
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=${REPO:-/content/DDS-SLAM}
SS_REPO=${SS_REPO:-/content/Python-SuPer}
SS_URL=https://github.com/ucsdarclab/Python-SuPer
SS_SHA=${SS_SHA:-be244fa}
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
SS_ENV=${SS_ENV:-super-ss}; SS_ENV_PY="$CONDA_ROOT/envs/$SS_ENV/bin/python"
DATA_ROOT=${DATA_ROOT:-/content/Super}          # staged LOCAL (Drive FUSE drops on long runs); ss_stage copies here
# Canonical Drive source for the Super trails (memory reference_drive_dataset_paths): Drive uses
# 'trial_N' (i), repo wants 'trail_N' (a) -> ss_stage renames on copy.
DRIVE_SUPER=${DRIVE_SUPER:-/content/drive/MyDrive/Datasets/SemSup/v2_data}
MONO2_CKPT=${MONO2_CKPT:-/content/drive/MyDrive/Datasets/depthmodels/semsup_variant_a_stereo_paper-faithful}  # Monodepth2-stereo ckpt
DRIVE=${DRIVE:-/content/drive/MyDrive/Outputs/SemanticSuPer_phaseA_$DATE}
PHASE=${1:-}; TRAIL_ARG=${2:-}
ALL4="trail_3 trail_4 trail_8 trail_9"

# ---- per-trail knobs (Python-SuPer/README.md:63/76/89/102) ---------------------------------
mesh_step(){ case "$1" in *_3) echo 32;; *_4) echo 32;; *_8) echo 25;; *_9) echo 18;; *) echo 32;; esac; }
edge_ids(){ case "$1" in
  *_3) echo "5 10 11 13 14 17 20 23 24 25 26 27 28 29 30 31 32";;
  *_4) echo "1 3 5 6 12 13 14 17 20 22 25";;
  *_8) echo "3 6 7 9 11 12 13 16 19 20 21 22 23 24 25 26 27 30 31 32 34 35 36";;
  *_9) echo "3 4 5 7 8 9 11 13 17 24 25 29 31 36 37 46 50 51";; esac; }
# paper Table I (Lin et al. ICRA'23, arXiv:2210.16674): "Lab  all_mean(std)  edge_mean(std)"
paper_ref(){ case "$1" in
  *_3) echo "Lab1 7.5(6.1) 6.7(5.7)";; *_4) echo "Lab2 8.6(7.6) 9.2(7.8)";;
  *_8) echo "Lab3 6.0(4.9) 5.9(4.8)";; *_9) echo "Lab4 4.3(3.8) 4.3(3.4)";; esac; }

# ---- ENV-SS: isolated py3.8 old-stack (pinned wheels) + Pulsar smoke ------------------------
build_env(){
  echo "[env] ENV-SS (py3.8/torch1.11+cu113/pytorch3d0.6.2 Pulsar) - isolated old stack"
  [ -d "$SS_REPO/.git" ] || git clone "$SS_URL" "$SS_REPO" || { echo "[env] FATAL clone Python-SuPer"; return 30; }
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      && bash /tmp/mc.sh -b -p "$CONDA_ROOT" || { echo "[env] FATAL miniconda"; return 30; }
  fi
  "$CONDA_ROOT/bin/conda" create -y -n "$SS_ENV" python=3.8 >/dev/null 2>&1 || true
  [ -x "$SS_ENV_PY" ] || { echo "[env] FATAL conda env not created ($SS_ENV_PY)"; return 30; }
  local PIP="PYTHONPATH= $SS_ENV_PY -m pip install -q"
  eval $PIP --upgrade pip
  eval $PIP torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
    -f https://download.pytorch.org/whl/cu113/torch_stable.html || { echo "[env] FATAL torch1.11+cu113"; return 30; }
  eval $PIP https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/pytorch3d-0.6.2-cp38-cp38-linux_x86_64.whl \
    || { echo "[env] FATAL pytorch3d 0.6.2 wheel"; return 30; }
  eval $PIP torch-scatter==2.0.9 torch-sparse==0.6.14 \
    -f https://data.pyg.org/whl/torch-1.11.0+cu113.html || { echo "[env] FATAL torch-scatter/sparse"; return 30; }
  eval $PIP torch-geometric==2.0.4 segmentation-models-pytorch==0.3.0 open3d==0.15.2 \
    "opencv-python==4.5.3.56" numpy==1.23.1 kornia==0.6.6 cupy-cuda113 filterpy moviepy tensorboard tqdm pyyaml scipy \
    || { echo "[env] FATAL deps"; return 30; }
  # MANDATORY Pulsar smoke (the >512-track crash + ABI check)
  PYTHONPATH= "$SS_ENV_PY" -c "from pytorch3d.renderer.points.pulsar import Renderer; Renderer(64,64,n_channels=3,n_track=512); print('[env] Pulsar import OK')" \
    || { echo "[env] FATAL Pulsar smoke (pytorch3d/torch ABI)"; return 30; }
  echo "[env] ENV-SS ready -> $SS_ENV_PY"
}

# ---- idempotently patch the upstream to ALSO dump the reproj number to a txt -----------------
# WHY: log_trackpts_err writes reprojerr/pythonsuper_{mean,std}[_edge_pts] ONLY to TensorBoard
# (utils/utils.py:499-511) - no stdout/txt. We dump all/edge mean(std) -> {logdir}/reproj_err.txt.
ss_patch_harvest(){
  PYTHONPATH= "$SS_ENV_PY" - "$SS_REPO/utils/utils.py" <<'PY'
import sys, io
p = sys.argv[1]; s = io.open(p, encoding='utf-8').read()
MARK = '# [DDS] reproj txt-dump'
if MARK in s:
    print('[patch] reproj harvest already present'); sys.exit(0)
anchor = "summary_writer.add_scalar('reprojerr/pythonsuper_std'"
i = s.find(anchor)
if i < 0:
    print('[patch] FATAL reproj anchor not found in utils.py - upstream changed; fix the patch'); sys.exit(1)
ls = s.rfind('\n', 0, i) + 1; ind = s[ls:i]            # match the anchor line's indent
eol = s.find('\n', i)
blk = ("\n" + ind + MARK + "\n"
       + ind + "import numpy as _np, os as _os\n"
       + ind + "_am=float(_np.mean(err_array)); _astd=float(_np.std(err_array)); _em=_es=float('nan')\n"
       + ind + "if len(edge_ids)>0:\n"
       + ind + "    _sel=_np.zeros(err_array.shape[1],dtype=bool); _sel[_np.array(edge_ids)-1]=True\n"
       + ind + "    _e=err_array[:,_sel]; _em=float(_np.mean(_e)); _es=float(_np.std(_e))\n"
       + ind + "open(_os.path.expanduser(f'{logdir}/reproj_err.txt'),'w').write("
       + "f'all_mean {_am:.4f}\\nall_std {_astd:.4f}\\nedge_mean {_em:.4f}\\nedge_std {_es:.4f}\\n')")
s = s[:eol] + blk + s[eol:]
io.open(p, 'w', encoding='utf-8').write(s)
print('[patch] reproj txt-dump injected after the TB scalars')
PY
}

# ---- stage a Super trail from Drive (trial_N -> trail_N) + verify the UPSTREAM layout --------
# Drive SemSup/v2_data may hold only the DDS-SLAM SUBSET (left + png_masks + moge depth). The
# upstream tracker needs MORE: stereo (-left + -right), seg/DeepLabV3+/*.npy soft logits, and the
# green-pin rgb/*_l_pts.npy. Verify all four and fail LOUD (exit 20) if the subset was shipped.
ss_stage(){
  local T=$1 SID=${1##*_} dst="$DATA_ROOT/$T"
  if [ ! -d "$dst/rgb" ] || [ "$(ls "$dst"/rgb/*-left.png 2>/dev/null | wc -l)" -eq 0 ]; then
    local src="$DRIVE_SUPER/trial_$SID"
    [ -d "$src" ] || { echo "[$T] BLOCKED: Super source not on Drive: $src"; return 20; }
    echo "[$T] staging $src -> $dst (local copy; trial_$SID -> $T)"
    mkdir -p "$DATA_ROOT"; rm -rf "$dst"; cp -r "$src" "$dst" || { echo "[$T] FAILED copy"; return 1; }
  fi
  local nl nr ns np
  nl=$(ls "$dst"/rgb/*-left.png 2>/dev/null | wc -l); nr=$(ls "$dst"/rgb/*-right.png 2>/dev/null | wc -l)
  ns=$(ls "$dst"/seg/DeepLabV3+/*-left.npy 2>/dev/null | wc -l); np=$(ls "$dst"/rgb/*_l_pts.npy 2>/dev/null | wc -l)
  echo "[$T] layout: left=$nl right=$nr seg/DeepLabV3+=$ns pins=$np"
  { [ "$nl" -ge 2 ] && [ "$nr" -ge 2 ] && [ "$ns" -ge 2 ] && [ "$np" -ge 1 ]; } || {
    echo "[$T] BLOCKED: incomplete UPSTREAM layout. The Drive $DRIVE_SUPER/trial_$SID looks like the"
    echo "[$T]   DDS-SLAM subset (left+masks+moge). Upload the FULL Super trail (stereo + seg/DeepLabV3+/*.npy"
    echo "[$T]   logits + rgb/*_l_pts.npy) to $DRIVE_SUPER/trial_$SID and re-run."; return 20; }
}

# ---- run ONE trail end-to-end -> reproj_err.txt + report-only Table I gate ------------------
run_trail(){
  local T=$1 OUT="$DRIVE/$1"; mkdir -p "$OUT"
  [ -f "$OUT/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { echo "[$T] already done -> skip"; return 0; }
  ss_stage "$T" || { echo "BLOCKED: staging/upstream-layout (see log)" > "$OUT/status.txt"; echo "[$T] BLOCKED stage"; return 20; }
  local DD="$DATA_ROOT/$T"
  [ -e "$MONO2_CKPT" ] || { echo "BLOCKED: Monodepth2 ckpt missing at $MONO2_CKPT (set MONO2_CKPT=<dir>)" > "$OUT/status.txt"; echo "[$T] BLOCKED no MONO2_CKPT"; return 20; }
  [ -d "$MONO2_CKPT" ] && { [ -f "$MONO2_CKPT/encoder.pth" ] && [ -f "$MONO2_CKPT/depth.pth" ]; } || echo "[$T] WARN $MONO2_CKPT may lack encoder.pth/depth.pth -> Monodepth2 load could fail (verify ckpt layout)"
  local gt; gt=$(cd "$DD" && ls rgb/*_l_pts.npy 2>/dev/null | head -1)
  [ -n "$gt" ] || { echo "BLOCKED: no rgb/*_l_pts.npy GT in $DD" > "$OUT/status.txt"; echo "[$T] BLOCKED no GT pts"; return 20; }
  local NF; NF=$(ls "$DD/rgb"/*-left.png 2>/dev/null | wc -l)
  [ "$NF" -ge 2 ] || { echo "BLOCKED: <2 left frames in $DD/rgb" > "$OUT/status.txt"; echo "[$T] BLOCKED too few frames"; return 20; }
  ss_patch_harvest || { echo "FAILED reproj-harvest patch (utils.py anchor)" > "$OUT/status.txt"; return 1; }
  local MN="ss_$T"
  echo "[$T] running upstream tracker (frames=$NF mesh_step=$(mesh_step "$T") gt=$gt)"
  ( cd "$SS_REPO" && PYTHONPATH= "$SS_ENV_PY" run_semantic_super.py \
      --model_name "$MN" --data_dir "$DD" --data superv2 --start_id 0 --end_id "$NF" \
      --tracking_gt_file "$gt" --num_layers 50 \
      --pretrained_encoder_checkpoint_dir "$MONO2_CKPT" \
      --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir "$MONO2_CKPT" --post_process \
      --load_seg --seg_dir seg/DeepLabV3+ --seg_ext .npy --num_classes 3 \
      --mesh_step_size "$(mesh_step "$T")" --edge_ids $(edge_ids "$T") \
      --sf_soft_seg_point_plane --sf_bn_morph --mesh_rot --mesh_face --render_loss ) 2>&1 | tee "$OUT/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED run_semantic_super.py (see run.log)" > "$OUT/status.txt"; return 1; }
  local RES="$SS_REPO/results/$MN/reproj_err.txt"
  [ -f "$RES" ] || RES=$(ls -t "$SS_REPO"/results/*/reproj_err.txt 2>/dev/null | head -1)
  [ -f "$RES" ] || { echo "FAILED no reproj_err.txt (did the harvest patch fire? --edge_ids?)" > "$OUT/status.txt"; return 1; }
  cp -f "$RES" "$OUT/reproj_err.txt"
  cp -f "$SS_REPO/results/$MN"/reprojerr_*.png "$OUT/" 2>/dev/null || true
  # ---- report-only gate vs paper Table I (NEVER blocks) ----
  PYTHONPATH= "$SS_ENV_PY" - "$OUT/reproj_err.txt" "$(paper_ref "$T")" "$OUT/summary.txt" "$T" <<'PY'
import sys, re
txt = open(sys.argv[1]).read(); ref = sys.argv[2].split(); out = sys.argv[3]; T = sys.argv[4]
def g(k):
    m = re.search(k + r'\s+([0-9.nan]+)', txt);
    try: return float(m.group(1)) if m else None
    except: return None
am, astd, em, es = g('all_mean'), g('all_std'), g('edge_mean'), g('edge_std')
def pm(x):
    m = re.match(r'([0-9.]+)\(([0-9.]+)\)', x); return (float(m.group(1)), float(m.group(2)))
lab = ref[0] if ref else T; pam, pas = pm(ref[1]); pem, pes = pm(ref[2])
def ok(v, m, s): return (v is not None) and (abs(v - m) <= s or v <= 1.3 * m)
verdict = 'PASS' if (ok(am, pam, pas) and ok(em, pem, pes)) else 'BELOW-BAND'
line = (f"{T} ({lab}): reproj px  all={am}({astd}) edge={em}({es})  |  paper all={pam}({pas}) "
        f"edge={pem}({pes})  ->  {verdict} (report-only, does NOT block)")
print(line); open(out, 'w').write(line + "\n")
PY
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"; sync
  echo "[$T] DONE -> $OUT (reproj_err.txt + summary.txt + plots)"
}

aggregate(){
  echo "=== Semantic-SuPer Phase-A reproj-error vs paper Table I ==="
  cat "$DRIVE"/*/summary.txt 2>/dev/null || echo "no per-trail summaries yet under $DRIVE"
}

case "$PHASE" in
  env)   build_env ;;
  repro)
    [ -x "$SS_ENV_PY" ] || { echo "FATAL: ENV-SS not built ($SS_ENV_PY) - run 'env' first"; exit 30; }
    [ -d "$SS_REPO/.git" ] || { echo "FATAL: Python-SuPer clone missing ($SS_REPO) - run 'env'"; exit 30; }
    mkdir -p "$DRIVE"
    TRS=${TRAIL_ARG:-trail_3}; [ "$TRS" = all4 ] && TRS="$ALL4"
    echo "=== Semantic-SuPer Phase-A: trails='$TRS'  (Path A: upstream reproj-err vs Table I) -> $DRIVE"
    rc=0; for t in $TRS; do run_trail "$t" || { c=$?; [ "$c" = 20 ] && rc=20 || rc=1; echo "[$t] -> $c (see $DRIVE/$t/status.txt)"; }; done
    aggregate
    echo "DONE repro (exit hint: $rc; 20=BLOCKED needs Drive data/ckpt)" ;;
  crcd)
    echo "Phase-B (CRCD) is a STUB - DEFERRED. Open decisions: (Q2) ontology {tool,beef,chicken} absent on"
    echo "CRCD {bg,Liver,Gallbladder,Tool}; (metric) CRCD has no green-pin reproj GT -> native reproj-err"
    echo "cannot be reproduced on CRCD; a DDS render-PSNR/Sim3 adaptation would NOT be a Semantic-SuPer"
    echo "reproduction. Resolve with the user before wiring. Exiting 0 (no-op)."; exit 0 ;;
  eval)  aggregate ;;
  all)   build_env && { TRAIL_ARG=${TRAIL_ARG:-trail_3} "$0" repro "${TRAIL_ARG:-trail_3}"; } ;;
  *) echo "usage: run_semanticsuper.sh env|repro [trail_3|trail_4|trail_8|trail_9|all4]|crcd|eval|all"; exit 2 ;;
esac
