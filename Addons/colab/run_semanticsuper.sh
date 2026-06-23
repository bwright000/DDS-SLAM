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
  echo "[env] ENV-SS from authors' environment.yaml (conda: pytorch3d 0.6.2 binary, no fbai wheels)"
  [ -d "$SS_REPO/.git" ] || git clone "$SS_URL" "$SS_REPO" || { echo "[env] FATAL clone Python-SuPer"; return 30; }
  local YAML="$SS_REPO/resources/environment.yaml"
  [ -f "$YAML" ] || { echo "[env] FATAL no $YAML"; return 30; }
  # already built? Pulsar imports -> skip (REBUILD_ENV=1 to force a clean rebuild)
  if [ "${REBUILD_ENV:-0}" != 1 ] && [ -x "$SS_ENV_PY" ] \
     && PYTHONPATH= "$SS_ENV_PY" -c "from pytorch3d.renderer.points.pulsar import Renderer" 2>/dev/null; then
    echo "[env] $SS_ENV ready (Pulsar imports; REBUILD_ENV=1 to rebuild)"; return 0; fi
  if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      && bash /tmp/mc.sh -b -p "$CONDA_ROOT" || { echo "[env] FATAL miniconda"; return 30; }
  fi
  # environment.yaml lists 'defaults'+'anaconda' channels -> accept their TOS non-interactively
  "$CONDA_ROOT/bin/conda" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
  "$CONDA_ROOT/bin/conda" config --set channel_priority flexible 2>/dev/null || true
  # create the EXACT authors' env: conda installs torch/torchvision/pytorch-scatter/-sparse/pytorch3d
  # with their pinned build strings (py38_cu113_pyt1110) + runs the pip: section. Wipe any half-built env.
  "$CONDA_ROOT/bin/conda" env remove -y -n "$SS_ENV" 2>/dev/null || true
  "$CONDA_ROOT/bin/conda" env create -n "$SS_ENV" -f "$YAML" \
    || { echo "[env] FATAL conda env create from environment.yaml (see error above)"; return 30; }
  # MANDATORY Pulsar smoke (the >512-track crash + ABI check)
  # Pulsar Renderer signature (pytorch3d 0.6.2, per Python-SuPer/renderer/renderer.py:63-67):
  # Renderer(width, height, max_num_balls, n_track=..., ...). 3rd positional is REQUIRED.
  PYTHONPATH= "$SS_ENV_PY" -c "from pytorch3d.renderer.points.pulsar import Renderer; Renderer(64, 64, 512, n_track=64); print('[env] Pulsar import OK')" \
    || { echo "[env] FATAL Pulsar smoke (pytorch3d/torch ABI)"; return 30; }
  # run-time extras not pinned in the yaml (defensive; no-op if already present)
  PYTHONPATH= "$SS_ENV_PY" -m pip install -q tqdm pyyaml 2>/dev/null || true
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

# ---- normalize a Monodepth2 ckpt for the upstream loader -------------------------------------
# load_checkpoints (shared_functions.py:136-148) does torch.load(path) then state["encoder"]/
# state["depth"] -> it needs a SINGLE .pth with those top-level keys (authors' wrapped format).
# A standard Monodepth2 weights_N/ ships RAW encoder.pth + depth.pth state-dicts -> repackage into
# the wrapped form. Echoes a usable .pth path; returns 1 if neither shape is found.
ss_prep_ckpt(){
  local ck="$1" out=/content/ss_mono2_combined.pth
  if [ -f "$ck" ] && PYTHONPATH= "$SS_ENV_PY" -c "import torch,sys; s=torch.load(sys.argv[1],map_location='cpu'); sys.exit(0 if isinstance(s,dict) and 'encoder' in s else 1)" "$ck" 2>/dev/null; then
    echo "$ck"; return 0; fi
  local enc dep
  enc=$(ls "$ck"/encoder.pth "$ck"/weights_*/encoder.pth 2>/dev/null | head -1)
  dep=$(ls "$ck"/depth.pth   "$ck"/weights_*/depth.pth   2>/dev/null | head -1)
  if [ -f "$enc" ] && [ -f "$dep" ]; then
    PYTHONPATH= "$SS_ENV_PY" -c "import torch,sys; torch.save({'encoder':torch.load(sys.argv[1],map_location='cpu'),'depth':torch.load(sys.argv[2],map_location='cpu')}, sys.argv[3])" "$enc" "$dep" "$out" >/dev/null 2>&1 \
      && { echo "$out"; return 0; }
  fi
  return 1
}

# ---- detect the Monodepth2 encoder depth (resnet18/34/50) from the ckpt's encoder.fc.weight ----
# (the authors' ckpt is resnet50; the DDS variant_a_stereo re-train is resnet18 -> must match
# --num_layers or load_state_dict size-mismatches). Echoes 18/34/50; defaults 50 if unknown.
ss_num_layers(){
  PYTHONPATH= "$SS_ENV_PY" - "$1" <<'PY' 2>/dev/null || echo 50
import torch, sys
s = torch.load(sys.argv[1], map_location='cpu')
enc = s.get('encoder', s) if isinstance(s, dict) else s
fc = enc.get('encoder.fc.weight') if isinstance(enc, dict) else None
if fc is None or fc.shape[1] == 2048:
    print(50)
elif fc.shape[1] == 512:
    idx = [int(k.split('.')[2]) for k in enc if k.startswith('encoder.layer1.') and k.split('.')[2].isdigit()]
    print(34 if (max(idx) if idx else 0) >= 2 else 18)
else:
    print(50)
PY
}

# ---- pin the authors' numpy/scikit-image (the yaml left them UNPINNED -> conda took NEWER versions
# that break the old upstream: np.bool removed in numpy>=1.24 (utils.py:503); skimage>=0.20 ssim
# requires data_range (data_loader.py:367)). Idempotent: only pins if the versions are wrong. ----
ss_fix_versions(){
  PYTHONPATH= "$SS_ENV_PY" -c "import numpy,skimage,sys; sys.exit(0 if numpy.__version__.startswith('1.23') and skimage.__version__.startswith('0.19') else 1)" 2>/dev/null \
    || { echo "[env] pinning authors' numpy==1.23.1 + scikit-image==0.19.3 (upstream needs np.bool / old ssim)"; \
         PYTHONPATH= "$SS_ENV_PY" -m pip install -q "numpy==1.23.1" "scikit-image==0.19.3"; }
}

# ---- run ONE trail end-to-end -> reproj_err.txt + report-only Table I gate ------------------
run_trail(){
  local T=$1 OUT="$DRIVE/$1"; mkdir -p "$OUT"
  [ -f "$OUT/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { echo "[$T] already done -> skip"; return 0; }
  ss_stage "$T" || { echo "BLOCKED: staging/upstream-layout (see log)" > "$OUT/status.txt"; echo "[$T] BLOCKED stage"; return 20; }
  local DD="$DATA_ROOT/$T"
  [ -e "$MONO2_CKPT" ] || { echo "BLOCKED: Monodepth2 ckpt missing at $MONO2_CKPT" > "$OUT/status.txt"; echo "[$T] BLOCKED no MONO2_CKPT"; return 20; }
  local CKPT; CKPT=$(ss_prep_ckpt "$MONO2_CKPT") || { echo "BLOCKED: cannot prep Monodepth2 ckpt from $MONO2_CKPT (need a wrapped .pth with 'encoder' key, OR a Monodepth2 weights dir with encoder.pth+depth.pth)" > "$OUT/status.txt"; echo "[$T] BLOCKED ckpt-format"; return 20; }
  echo "[$T] Monodepth2 ckpt -> $CKPT  (NOTE: weights_N = a DDS re-train, not the authors' released ckpt -> faithful-except-depth)"
  local NL=${NUM_LAYERS:-}; [ -n "$NL" ] || NL=$(ss_num_layers "$CKPT"); [ -n "$NL" ] || NL=50
  echo "[$T] Monodepth2 encoder -> resnet$NL (auto-detected; authors'=50, DDS variant_a_stereo=18)"
  local gt; gt=$(cd "$DD" && ls rgb/*_l_pts.npy 2>/dev/null | head -1)
  [ -n "$gt" ] || { echo "BLOCKED: no rgb/*_l_pts.npy GT in $DD" > "$OUT/status.txt"; echo "[$T] BLOCKED no GT pts"; return 20; }
  local NF; NF=$(ls "$DD/rgb"/*-left.png 2>/dev/null | wc -l)
  [ "$NF" -ge 2 ] || { echo "BLOCKED: <2 left frames in $DD/rgb" > "$OUT/status.txt"; echo "[$T] BLOCKED too few frames"; return 20; }
  ss_patch_harvest || { echo "FAILED reproj-harvest patch (utils.py anchor)" > "$OUT/status.txt"; return 1; }
  ss_fix_versions
  local MN="ss_$T"
  echo "[$T] running upstream tracker (frames=$NF mesh_step=$(mesh_step "$T") gt=$gt)"
  ( cd "$SS_REPO" && PYTHONPATH= "$SS_ENV_PY" run_semantic_super.py \
      --model_name "$MN" --data_dir "$DD" --data superv2 --start_id 0 --end_id "$NF" \
      --tracking_gt_file "$gt" --num_layers "$NL" \
      --pretrained_encoder_checkpoint_dir "$CKPT" \
      --depth_model monodepth2_stereo --pretrained_depth_checkpoint_dir "$CKPT" --post_process \
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

# ============================================================================
# PHASE B — CRCD: run the UPSTREAM tracker on CRCD with SGS-comparable metrics.
#   depth=MoGe (metric, loaded); resize 1280x720 -> IMG_WxIMG_H; seg=GT 4-class; NO green-pins
#   (--tracking_gt_file omitted -> evaluate_tracking=False -> pin path off). STEP-1 metric =
#   render PSNR/SSIM/LPIPS + video. STEP-2 (todo) = Sim3 ATE via exported global T_g + depth-L1.
# ============================================================================
DDS_PY=${DDS_PY:-python3}                          # system torch2 python for the DDS eval tools
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
DRIVE_CRCD_MOGE=${DRIVE_CRCD_MOGE:-/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2}
DRIVE_CRCD_RECT=${DRIVE_CRCD_RECT:-/content/drive/MyDrive/Datasets/CRCD-Published-rectified}  # rectify ONCE, cache here
CALIB_PKL=${CALIB_PKL:-$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl}
CRCD_DRIVE=${CRCD_DRIVE:-/content/drive/MyDrive/Outputs/SemanticSuPer_crcd_$DATE}
BENCH5_CRCD="c1_001 c2_001 e3_005 c3_001 g3_001"
IMG_W=${IMG_W:-640}; IMG_H=${IMG_H:-360}; CRCD_SEG=${CRCD_SEG:-GT}; CRCD_MESH_STEP=${CRCD_MESH_STEP:-32}
crcd_ep_sid(){ local n; n=$(echo "$1"|tr 'a-z' 'A-Z'); [[ "$n" =~ ^[A-Z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }; echo "${n:0:1}_${n:1:1} ${n:3}"; }

# idempotent upstream patches for the CRCD path (3): get_K reads crcd_K.txt; get_depth treats the
# loaded .npy as METRIC depth (skip disp_to_depth); render_img works without pins + dumps the clean
# render to results/<model>/render/<t>.png for PSNR (the upstream only logs renders to TensorBoard).
ss_patch_crcd(){
  ( cd "$SS_REPO" && git checkout -- utils/data_loader.py super/nodes.py super/deform_mesh.py utils/labels.py 2>/dev/null ) || true  # clean slate -> idempotent
  PYTHONPATH= "$SS_ENV_PY" - "$SS_REPO" <<'PY'
import io, os, sys
R = sys.argv[1]
def edit(path, mark, anchor, ins=None, replace=None):
    p = os.path.join(R, path); s = io.open(p, encoding='utf-8').read()
    if mark in s: print(f"[patch-crcd] {path}: already ({mark})"); return
    if anchor not in s: print(f"[patch-crcd] FATAL anchor missing in {path}: {anchor!r}"); sys.exit(1)
    if replace is not None: s = s.replace(anchor, replace, 1)
    else: i = s.find(anchor) + len(anchor); s = s[:i] + ins + s[i:]
    io.open(p, 'w', encoding='utf-8').write(s); print(f"[patch-crcd] {path}: patched ({mark})")
# target SuPerDataset.get_K (the override at line 201, used by __getitem__) — NOT the base
# GeneralDataset.get_K (raise NotImplementedError). Anchor = its signature + first 'superv1' line.
edit('utils/data_loader.py', '[DDS-crcd-K]',
     "    def get_K(self):\n        if self.opt.data == 'superv1':\n",
     replace=("    def get_K(self):\n"
              "        if self.opt.data == 'crcd':  # [DDS-crcd-K]\n"
              "            _kv = {}\n"
              "            for _l in open(os.path.join(self.opt.data_dir, 'crcd_K.txt')):\n"
              "                _p = _l.split()\n"
              "                if len(_p) >= 2 and _p[0] in ('fx','fy','cx','cy'): _kv[_p[0]] = float(_p[1])\n"
              "            return np.array([[_kv['fx'],0,_kv['cx'],0],[0,_kv['fy'],_kv['cy'],0],[0,0,1,0],[0,0,0,1]], dtype=np.float32)\n"
              "        if self.opt.data == 'superv1':\n"))
edit('utils/data_loader.py', '[DDS-crcd-depth]',
     "            disp, depth = disp_to_depth(disp, self.min_depth, self.max_depth)\n",
     replace=("            if self.opt.data == 'crcd':  # [DDS-crcd-depth] loaded .npy is METRIC depth (metres)\n"
              "                depth = disp.clone(); disp = 1.0 / (depth + 1e-6)\n"
              "            else:\n"
              "                disp, depth = disp_to_depth(disp, self.min_depth, self.max_depth)\n"))
edit('super/nodes.py', '[DDS-crcd-render]',
     "        render_img = (255*render_img).type(torch.uint8)\n",
     ins=("        render_img_keypoints = render_img  # [DDS-crcd-render] default when no pins\n"
          "        try:\n"
          "            import os as _os, cv2 as _cv2\n"
          "            _rd = _os.path.join(self.output_dir, 'render'); _os.makedirs(_rd, exist_ok=True)\n"
          "            _cv2.imwrite(_os.path.join(_rd, '%06d.png' % self.time), render_img.permute(1,2,0).cpu().numpy()[:, :, ::-1])\n"
          "        except Exception: pass\n"))
# evaluate() (the reproj-err eval) is called unconditionally (super.py:81) but accesses pin-only
# attrs (track_rsts @775); with no green-pins (evaluate_tracking=False) -> AttributeError. Early-return.
edit('super/nodes.py', '[DDS-crcd-noeval]',
     "    def evaluate(self):\n",
     ins="        if not getattr(self, 'evaluate_tracking', False): return  # [DDS-crcd-noeval] no green-pins on CRCD\n")

# per-class kernel lists are hardcoded for 3 classes ([3,3,3]) but indexed by range(num_classes) ->
# overflow at CRCD's 4 classes. Size them to num_classes.
edit('utils/data_loader.py', '[DDS-crcd-kernels]', "            kernels = [3, 3, 3]\n",
     replace="            kernels = [3] * opt.num_classes  # [DDS-crcd-kernels]\n")
edit('super/deform_mesh.py', '[DDS-crcd-kernels]', "                        kernels = [3, 3, 3]\n",
     replace="                        kernels = [3] * self.opt.num_classes  # [DDS-crcd-kernels]\n")

# id2color palette is sized for 3 Super classes (Beef/Chicken/Tool) -> id2color[3] crashes for CRCD's
# 4 classes. Replace with an >=8-row palette (viz only; tracking/metrics unaffected).
edit('utils/labels.py', '[DDS-crcd-id2color]',
     "id2color        = torch.zeros((3,3))\n",
     replace=("id2color        = torch.tensor([[0,0,0],[230,60,60],[40,200,60],[60,120,230],"
              "[230,230,60],[230,60,230],[60,230,230],[150,150,150]], dtype=torch.float32)  # [DDS-crcd-id2color]\n"))

# alias the GENERIC superv2 opt.data branches (in BOTH data_loader.py and nodes.py) to also accept
# crcd so it reuses superv2 logic (invalid mask, disparity-viz scale, seg-viz else-branch). get_K's
# crcd branch is FIRST -> still returns crcd_K; nodes.py seg-viz else already routes crcd -> id2color.
for _f in ('utils/data_loader.py', 'super/nodes.py'):
    _p = os.path.join(R, _f); _s = io.open(_p, encoding='utf-8').read()
    if "[DDS-crcd-alias]" in _s: continue
    _s = _s.replace("opt.data == 'superv2'", "opt.data in ('superv2', 'crcd')")
    _s = _s.replace('opt.data == "superv2"', 'opt.data in ("superv2", "crcd")')
    io.open(_p, 'w', encoding='utf-8').write(_s + "\n# [DDS-crcd-alias]\n")
    print(f"[patch-crcd] {_f}: superv2->crcd alias")
print("[patch-crcd] done")
PY
}

run_crcd_one(){
  local NAME; NAME=$(echo "$1"|tr 'A-Z' 'a-z'); local UP; UP=$(echo "$NAME"|tr 'a-z' 'A-Z')
  local EP SID; read -r EP SID <<< "$(crcd_ep_sid "$NAME")"; [ -n "$EP" ] || { echo "[$NAME] bad name"; return 1; }
  local OUT="$CRCD_DRIVE/$UP"; mkdir -p "$OUT"
  [ -f "$OUT/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { echo "[$NAME] done -> skip"; return 0; }
  local SRC="$DRIVE_CRCD/$EP/snippet_$SID" MOGE="$DRIVE_CRCD_MOGE/$EP/snippet_$SID/depth"
  [ -d "$SRC/rgb" ] && [ -d "$SRC/rgbright" ] || { echo "BLOCKED: CRCD stereo missing ($SRC: rgb+rgbright)" > "$OUT/status.txt"; echo "[$NAME] BLOCKED no stereo"; return 20; }
  [ "$(ls "$MOGE"/*.png 2>/dev/null|wc -l)" -gt 0 ] || { echo "BLOCKED: MoGe depth missing ($MOGE)" > "$OUT/status.txt"; echo "[$NAME] BLOCKED no MoGe"; return 20; }
  local STAGED=/content/CRCD_staged/$UP SUPER=/content/Super_crcd/$UP
  local nraw; nraw=$(ls "$SRC/rgb"/*.png 2>/dev/null | wc -l)
  # ---- rectify ONCE, cached (Drive-persisted + local fast-path) ----
  if [ "$nraw" -gt 0 ] && [ "$(ls "$STAGED/video_frames"/*l.png 2>/dev/null | wc -l)" = "$nraw" ]; then
    echo "[$NAME] rectified: local cache hit ($nraw frames)"
  elif [ "$nraw" -gt 0 ] && [ "$(ls "$DRIVE_CRCD_RECT/$UP/video_frames"/*l.png 2>/dev/null | wc -l)" = "$nraw" ]; then
    echo "[$NAME] rectified: Drive cache hit -> copying local"; rm -rf "$STAGED"; mkdir -p "$STAGED"; cp -rn "$DRIVE_CRCD_RECT/$UP/." "$STAGED/"
  else
    echo "[$NAME] rectifying (no cache) -> $STAGED, then caching to $DRIVE_CRCD_RECT/$UP"; rm -rf "$STAGED"
    PYTHONPATH= "$DDS_PY" "$REPO/Addons/preprocess/preprocess_crcd_published.py" \
       --snippet_dir "$SRC" --calib_pkl "$CALIB_PKL" --output_dir "$STAGED" \
       || { echo "FAILED rectify/preprocess" > "$OUT/status.txt"; return 1; }
    mkdir -p "$DRIVE_CRCD_RECT/$UP"; cp -rn "$STAGED/." "$DRIVE_CRCD_RECT/$UP/" 2>/dev/null \
       || echo "[$NAME] WARN Drive rect-cache write failed (kept local)"
  fi
  # ---- assemble to SuPer (skip if complete + !FORCE; depends on resize/seg params) ----
  local nrect; nrect=$(ls "$STAGED/video_frames"/*l.png 2>/dev/null | wc -l)
  if [ "$(ls "$SUPER/rgb"/*-left.png 2>/dev/null | wc -l)" = "$nrect" ] && [ -f "$SUPER/crcd_K.txt" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "[$NAME] SuPer-assembled: cache hit (FORCE=1 to redo)"
  else
    rm -rf "$SUPER"
    PYTHONPATH= "$DDS_PY" "$REPO/Addons/colab/crcd_assemble_super.py" \
       --staged "$STAGED" --moge_depth "$MOGE" --calib "$STAGED/rectified_calib.txt" \
       --out "$SUPER" --img_w "$IMG_W" --img_h "$IMG_H" --n_classes 4 --seg_src "$CRCD_SEG" \
       || { echo "FAILED assemble-super" > "$OUT/status.txt"; return 1; }
  fi
  ss_patch_crcd || { echo "FAILED crcd patches" > "$OUT/status.txt"; return 1; }
  ss_fix_versions
  local NF; NF=$(ls "$SUPER/rgb"/*-left.png 2>/dev/null|wc -l); local MN="ss_crcd_$UP"
  echo "[$NAME] running upstream tracker on CRCD (frames=$NF res=${IMG_W}x${IMG_H} seg=$CRCD_SEG mesh_step=$CRCD_MESH_STEP, NO pins)"
  ( cd "$SS_REPO" && PYTHONPATH= "$SS_ENV_PY" run_semantic_super.py \
      --model_name "$MN" --data crcd --data_dir "$SUPER" --start_id 0 --end_id "$NF" \
      --height "$IMG_H" --width "$IMG_W" \
      --load_depth --depth_dir depth --depth_ext .npy \
      --load_seg --seg_dir "seg/$CRCD_SEG" --seg_ext .npy --num_classes 4 \
      --phase test --save_sample_freq 1 --mesh_step_size "$CRCD_MESH_STEP" \
      --sf_soft_seg_point_plane --sf_bn_morph --mesh_rot --mesh_face --render_loss ) 2>&1 | tee "$OUT/run.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAILED run_semantic_super.py (see run.log)" > "$OUT/status.txt"; return 1; }
  local RD="$SS_REPO/results/$MN/render"
  [ "$(ls "$RD"/*.png 2>/dev/null|wc -l)" -gt 0 ] || { echo "FAILED no renders in $RD (render-save patch?)" > "$OUT/status.txt"; return 1; }
  PYTHONPATH= "$DDS_PY" - "$RD" "$SUPER/rgb" "$OUT" <<'PY'
import sys, os, glob, re, shutil
rd, rgb, out = sys.argv[1:4]
n = 0
for p in sorted(glob.glob(os.path.join(rd, '*.png')), key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1))):
    idx = int(re.search(r'(\d+)', os.path.basename(p)).group(1))
    shutil.copy(p, os.path.join(out, f'{idx}.jpg'))
    gt = os.path.join(rgb, f'{idx:06d}-left.png')
    if os.path.exists(gt): shutil.copy(gt, os.path.join(out, f'{idx}_gt.png'))
    n += 1
print(f"[rename] {n} render/gt pairs -> {out}")
PY
  "$DDS_PY" -c "import lpips" 2>/dev/null || "$DDS_PY" -m pip install -q lpips 2>/dev/null
  rm -f "$OUT/render_eval.txt" "$OUT/render_eval.csv"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/eval_rendering.py" --gt_dir "$OUT" --render_dir "$OUT" \
     --sequence "CRCD-SuPer ($UP)" --output_csv "$OUT/render_eval.csv" --summary_csv "$OUT/render_eval.txt" \
     || echo "[$NAME] WARN eval_rendering"
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/viz/generate_video.py" --rgb_input_dir "$OUT" --rgb_output_dir "$OUT" \
     --output "$OUT/video.mp4" 2>/dev/null || echo "[$NAME] WARN video"
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"; sync
  echo "[$NAME] DONE -> $OUT (render_eval + video; ATE/depth-L1 = step-2 via T_g + rendered-depth)"
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
    [ -x "$SS_ENV_PY" ] || { echo "FATAL: ENV-SS not built ($SS_ENV_PY) - run 'env' first"; exit 30; }
    [ -d "$SS_REPO/.git" ] || { echo "FATAL: Python-SuPer clone missing ($SS_REPO) - run 'env'"; exit 30; }
    mkdir -p "$CRCD_DRIVE"
    SN=${TRAIL_ARG:-bench5}; { [ "$SN" = bench5 ] || [ "$SN" = all ]; } && SN="$BENCH5_CRCD"
    echo "=== Semantic-SuPer Phase-B CRCD: snippets='$SN' depth=MoGe seg=$CRCD_SEG res=${IMG_W}x${IMG_H} -> $CRCD_DRIVE"
    rc=0; for s in $SN; do run_crcd_one "$s" || { c=$?; [ "$c" = 20 ] && rc=20 || rc=1; echo "[$s] -> $c (see $CRCD_DRIVE/$(echo "$s"|tr a-z A-Z)/status.txt)"; }; done
    echo "DONE crcd (exit hint: $rc) -> $CRCD_DRIVE  (render PSNR/SSIM/LPIPS + video; ATE/depth-L1 = step-2)" ;;
  eval)  aggregate ;;
  all)   build_env && { TRAIL_ARG=${TRAIL_ARG:-trail_3} "$0" repro "${TRAIL_ARG:-trail_3}"; } ;;
  *) echo "usage: run_semanticsuper.sh env|repro [trail_3|trail_4|trail_8|trail_9|all4]|crcd|eval|all"; exit 2 ;;
esac
