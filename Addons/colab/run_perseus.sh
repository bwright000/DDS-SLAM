#!/bin/bash
# ============================================================================
# run_perseus.sh <phase> [snippet]  -  PERSEUS onboarding (TRACKING-ONLY arm)
#   phase: env | crcd [snippet|bench5] | eval
#
# WHAT — wire PERSEUS (vu-maple-lab, "Perception with Semantic Endoscopic Understanding
#   and SLAM": a DROID-SLAM fork + semantic seg + monocular depth, IPCAI'26) onto the
#   RECTIFIED 5-snippet CRCD benchmark as a TRACKING-ONLY row. We do NOT improve the
#   method — only configure it faithfully. All DROID hyperparameters (beta/filter_thresh/
#   warmup/keyframe_thresh/frontend*/backend*) stay at the authors' demo.py defaults.
#   --stride 1 is DATASET-NECESSARY (default 3 would emit N/3 poses vs N GT rows), and
#   --disable_vis because Colab is headless.
#
# TRACKING-ONLY RATIONALE — PERSEUS renders nothing (no novel-view synthesis) and its
#   monocular depth head is display/point-cloud only (never consumed by tracking/BA), so
#   the battery is Sim3 ATE mean/max + est/GT path ratio + Sim3-aligned |Pearson| dom via
#   Addons/eval/sim3_ate.py vs the staged 360-row groundtruth.txt. NO eval_rendering /
#   depth_l1 / 6-panel video. metrics.json carries nulls for the render fields so the
#   aggregate table prints "--" in those columns.
#
# SEG IS CLOUD-LABELLING ONLY — in this fork the seg mask is stored in video.masks and
#   consumed EXCLUSIVELY by visualization.py (verified: grep -rn masks droid_slam/ ->
#   depth_video.py:28 definition + visualization.py:151-156 read; NO consumer in
#   motion_filter/droid_frontend/droid_backend/factor_graph). Tracking is provably
#   seg-independent. Domain footnote for the table: the shipped seg/MDE weights are
#   airway-trained (CAO); on colorectal CRCD their labels would be out-of-domain anyway.
#
# MECHANICAL PATCHES (idempotent + fail-loud; NEITHER touches tracking/BA math):
#   (t) terminate-restore [ALWAYS]: the fork's droid_slam/droid.py:92-96 ships upstream
#       DROID-SLAM's trajectory fill COMMENTED OUT, returns None, and unconditionally
#       kills self.visualizer (which does not exist under --disable_vis -> AttributeError).
#       As committed, demo.py can NEVER produce traj_est.npy. We restore the authors' own
#       commented-out upstream lines (traj_filler + .inv().data return — the exact
#       behaviour their evaluation_scripts/test_tum.py:92 depends on), guard the visualizer
#       kill behind disable_vis, and adapt the 7-tuple demo stream to the filler's
#       (t, image, intrinsics) 3-tuple. backend(7)/backend(12) global BA above the patch
#       is stock and untouched.
#   (s) NOSEG stub [ONLY when PERSEUS_NOSEG=1; runtime-gated so it is inert otherwise]:
#       replaces demo.py's SegMDEInference import with a null object returning
#       (zeros mask, zeros depth). TRACKING-IDENTICAL fallback (see cloud-labelling note
#       above: masks feed visualization only; seg_mask/seg_depth otherwise feed only
#       show_image_cleaned, dead under --disable_vis). Default = FAITHFUL (weights
#       required). ⚠️ KNOWN UPSTREAM SHAPE HAZARD: SegMDEInference.create_circle_mask
#       returns a fixed (1080,1080) mask multiplied against an (H,W) prediction — on
#       frames that are not 1080x1080 the faithful seg path may crash with a numpy
#       broadcast error. If that happens the run .FAILEDs in isolation; re-run with
#       PERSEUS_NOSEG=1 (tracking-identical) and footnote it.
#
# ENV — authors' recipe (README): conda py3.12 + cuda-toolkit 11.8 + torch 2.5.1+cu118
#   (requirements_torch_118.txt carries its own -i cu118 index; honored verbatim) +
#   requirements_others.txt + torch-scatter from the pyg cu118 wheel index + in-repo
#   `python setup.py install` (droid_backends + lietorch CUDA extensions; the setup.py
#   HARDCODES -gencode sm_60..86, covering T4 sm_75 + A100 sm_80 natively — do NOT
#   schedule on H100 sm_90). cu118 nvcc accepts Colab's gcc-11 -> no gcc-10 dance.
#   crypt.h build failure -> README's `cp /usr/include/crypt.h $ENV/include/` retry.
#   A constraints file pins torch==2.5.1/cu118 so no later pip step upgrades it.
#
# WEIGHTS — droid.pth (upstream DROID-SLAM): $WEIGHTS_DRIVE/droid.pth, else gdown
#   1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh; FATAL if neither. Seg/MDE weights: the Box link is
#   manual-download-only -> must be pre-staged in $WEIGHTS_DRIVE (filenames below, read
#   from the MDE submodule's SegMDEInference defaults). The CODE loads them from
#   droid_slam/cao_seg/src/models/ (test_one_line_import.py:27 default Paths, relative to
#   cwd=SegmentedSLAM) while the README says droid_slam/MDE/application/cao_seg/src/models
#   — we symlink droid_slam/cao_seg -> droid_slam/MDE/application/cao_seg so both are the
#   same dir (also required for its internal `droid_slam.cao_seg.*` imports).
#
# DATA — rect_bench stage cache, exactly like run_snislam.sh:
#   $STAGE_CACHE_DIR/<UP>_v2.tar -> /content/rect_staged/<UP>. --imagedir points at
#   <staged>/video_frames DIRECTLY (rectified left %06dl.png, zero-padded ->
#   demo.py's lexicographic sorted(os.listdir()) == frame order; the r.png frames are
#   tar-excluded and we fail-loud preflight that ONLY *l.png files are present, because
#   demo.py streams EVERY file in the dir). NO assembler. CALIB.txt per snippet from
#   rectified_calib.txt: single line `0 0 W H fx fy cx cy 0 0 0 0 0` (no crop; rectified
#   => zero distortion; W/H via cv2 from the first staged frame).
#
# Usage (fresh Colab, Drive mounted):
#   bash Addons/colab/run_perseus.sh env                    # conda env + clone + patch + weights (one-time)
#   bash Addons/colab/run_perseus.sh crcd                   # bench5 (all 5, shortest-first)
#   bash Addons/colab/run_perseus.sh crcd e3_005            # one snippet (smoke)
#   PERSEUS_NOSEG=1 bash Addons/colab/run_perseus.sh env    # env without seg/MDE weights
#   PERSEUS_NOSEG=1 bash Addons/colab/run_perseus.sh crcd   # tracking-identical noseg arm (outputs *_noseg)
#   PERSEUS_RECON=1 bash Addons/colab/run_perseus.sh crcd c1_001  # also save reconstructions/<UP> to Drive
#   PERSEUS_BUFFER=768 bash Addons/colab/run_perseus.sh crcd      # bigger keyframe buffer if it overflows
#   bash Addons/colab/run_perseus.sh eval                   # (re)aggregate
# ============================================================================
set -uo pipefail
VERB=${1:-}; ARG=${2:-}
DATE=${DATE:-$(date +%Y%m%d)}

# ---- repos / pythons -------------------------------------------------------
PERSEUS=${PERSEUS:-/content/perseus}                         # vu-maple-lab clone (--recursive: MDE submodule)
PERSEUS_URL=${PERSEUS_URL:-https://github.com/vu-maple-lab/perseus}
SLAM="$PERSEUS/SegmentedSLAM"                                # demo.py cwd (traj_est.npy + droid.pth live here)
REPO=${REPO:-/content/DDS-SLAM}                              # this repo (DDS working copy)
DDS_PY=${DDS_PY:-python}                                     # system torch2 python: DDS eval CLIs (mirror run_semgauss)
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
ENV_NAME=${ENV_NAME:-perseus}
ENV_PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

# ---- weights ---------------------------------------------------------------
WEIGHTS_DRIVE=${WEIGHTS_DRIVE:-/content/drive/MyDrive/Datasets/perseus_weights}
DROID_GDOWN_ID=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh             # public droid.pth (README: from princeton-vl/DROID-SLAM)
SEG_BOX_URL="https://vanderbilt.app.box.com/s/f1kkupr9ppj7z4ga50xap33dw86k2f5v"   # manual download only (README)
# ⚠️ CONFIRM-IF-UNSURE: seg/MDE weight filenames = the SegMDEInference default checkpoint Paths
# (MDE submodule application/cao_seg/src/models/test_one_line_import.py:27, @427605c). Change here if the
# submodule moves them.
SEG_WEIGHT_FILES=${SEG_WEIGHT_FILES:-"spie_cao_tumor_segmentation.pth mde_cao_518.pth"}
SEG_MODELS_DIR="$SLAM/droid_slam/cao_seg/src/models"         # where the CODE loads from (via our symlink)

# ---- data / output ---------------------------------------------------------
STAGE_CACHE_DIR=${STAGE_CACHE_DIR:-/content/drive/MyDrive/dds_cache/rect_staged}; STAGEVER=v2
DRIVE_OUT=${DRIVE_OUT:-/content/drive/MyDrive/Outputs/PERSEUS_bench_$DATE}
FORCE=${FORCE:-0}                                            # 1 = ignore .DONE, redo
PERSEUS_NOSEG=${PERSEUS_NOSEG:-0}                            # 1 = stub seg/MDE (tracking-identical; header note)
PERSEUS_RECON=${PERSEUS_RECON:-0}                            # 1 = --reconstruction_path <UP> (auto-enables --upsample -> more VRAM)
PERSEUS_BUFFER=${PERSEUS_BUFFER:-512}                        # demo.py default; keyframe buffer size

BENCH5="e3_005 c1_001 c2_001 c3_001 g3_001"                 # SHORTEST-FIRST (e3_005 = smoke)

say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }

# --------------------------------------------------------------- patches ----
# patch (t): restore upstream DROID-SLAM terminate() (see header). Idempotent (marker) + fail-loud (anchor).
apply_terminate_restore_patch(){
  local F="$SLAM/droid_slam/droid.py"
  [ -f "$F" ] || { echo "[patch] FATAL $F missing (clone incomplete)"; return 1; }
  python3 - "$F" <<'PY' || return 1
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
MARK = "[DDS] PERSEUS_TERMINATE_RESTORE"
if MARK in s:
    print("[patch] terminate-restore already applied"); sys.exit(0)
# exact fork block (droid_slam/droid.py:92-96 @cf43e3e): upstream fill commented out, returns None,
# unconditional visualizer.kill() (crashes under --disable_vis).
old = ("        # camera_trajectory = self.traj_filler(stream)\n"
      "        # return camera_trajectory.inv().data.cpu().numpy()\n"
      "        time.sleep(2)\n"
      "        self.visualizer.kill()\n"
      "        return None")
assert old in s, ("[patch] FATAL: droid.py terminate() anchor block not found (fork moved?) — "
                  "inspect droid_slam/droid.py terminate() and update apply_terminate_restore_patch.")
new = (
"        # ==== [DDS] PERSEUS_TERMINATE_RESTORE (run_perseus.sh; mechanical, no method change) ====\n"
"        # Restores the authors' own commented-out UPSTREAM DROID-SLAM behaviour (the return value\n"
"        # their evaluation_scripts/test_tum.py:92 consumes). Tracking/BA math untouched.\n"
"        if not self.disable_vis and hasattr(self, 'visualizer'):\n"
"            time.sleep(2)\n"
"            self.visualizer.kill()\n"
"        def _first3(gen):  # demo.py streams yield 7-tuples; the filler wants (t, image, intrinsics)\n"
"            for tup in gen:\n"
"                yield tup[0], tup[1], tup[2]\n"
"        camera_trajectory = self.traj_filler(_first3(stream))\n"
"        return camera_trajectory.inv().data.cpu().numpy()")
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print("[patch] droid.py terminate() restored to upstream semantics (+ disable_vis guard)")
PY
}

# patch (s): NOSEG stub — runtime-gated on env PERSEUS_NOSEG, so once applied it is INERT for
# faithful runs (unset/0 -> the original import executes). Idempotent (marker) + fail-loud (anchor).
apply_noseg_stub_patch(){
  local F="$SLAM/demo.py"
  [ -f "$F" ] || { echo "[patch] FATAL $F missing (clone incomplete)"; return 1; }
  python3 - "$F" <<'PY' || return 1
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
MARK = "[DDS] PERSEUS_NOSEG_STUB"
if MARK in s:
    print("[patch] NOSEG stub already applied (runtime-gated)"); sys.exit(0)
old = "from droid_slam.MDE.application.cao_seg.src.models.test_one_line_import import SegMDEInference"
assert old in s, ("[patch] FATAL: demo.py SegMDEInference import anchor not found — "
                  "inspect demo.py and update apply_noseg_stub_patch.")
new = '''# ==== [DDS] PERSEUS_NOSEG_STUB (run_perseus.sh; runtime-gated, TRACKING-IDENTICAL) ====
# video.masks is consumed ONLY by visualization.py (verified: grep -rn masks droid_slam/ ->
# depth_video.py:28 definition + visualization.py:151-156 read; NO tracking/BA consumer), and
# seg_mask/seg_depth otherwise feed only show_image_cleaned (dead under --disable_vis).
import os as _dds_os
if _dds_os.environ.get('PERSEUS_NOSEG', '0') == '1':
    import numpy as _dds_np
    class SegMDEInference:
        def __init__(self, *a, **k):
            print('[DDS] PERSEUS_NOSEG=1 -> SegMDEInference stubbed (zeros mask/depth; viz-only path)')
        def inference_single(self, frame_rgb):
            _z = _dds_np.zeros(frame_rgb.shape[:2], dtype=_dds_np.float32)
            return _z, _z
else:
    from droid_slam.MDE.application.cao_seg.src.models.test_one_line_import import SegMDEInference'''
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new))
print("[patch] demo.py NOSEG stub injected (gated on env PERSEUS_NOSEG=1)")
PY
}

# --------------------------------------------------------------- weights ----
fetch_droid_weights(){
  [ -f "$SLAM/droid.pth" ] && { echo "[env] droid.pth present"; return 0; }
  if [ -f "$WEIGHTS_DRIVE/droid.pth" ]; then
    cp -f "$WEIGHTS_DRIVE/droid.pth" "$SLAM/droid.pth" && { echo "[env] droid.pth <- $WEIGHTS_DRIVE"; return 0; }
  fi
  echo "[env] droid.pth not on Drive -> gdown $DROID_GDOWN_ID"
  ( cd "$SLAM" && { PYTHONPATH= "$ENV_PY" -m gdown "$DROID_GDOWN_ID" -O droid.pth 2>/dev/null \
                    || python3 -m gdown "$DROID_GDOWN_ID" -O droid.pth; } )
  [ -s "$SLAM/droid.pth" ] || { echo "FATAL[env]: droid.pth unavailable (Drive: $WEIGHTS_DRIVE/droid.pth; gdown id $DROID_GDOWN_ID)"; return 1; }
  echo "[env] droid.pth fetched ($(du -h "$SLAM/droid.pth" | cut -f1))"
}

ensure_seg_weights(){
  if [ "$PERSEUS_NOSEG" = 1 ]; then
    echo "[env] PERSEUS_NOSEG=1 -> seg/MDE weights NOT required (stubbed at runtime)"; return 0; fi
  # MDE submodule must be checked out (test_one_line_import.py lives inside it)
  if [ ! -f "$SLAM/droid_slam/MDE/application/cao_seg/src/models/test_one_line_import.py" ]; then
    echo "[env] MDE submodule empty -> git submodule update --init"
    ( cd "$PERSEUS" && git submodule update --init --recursive )
    [ -f "$SLAM/droid_slam/MDE/application/cao_seg/src/models/test_one_line_import.py" ] || {
      echo "FATAL[env]: MDE submodule (MedICL-VU/MDE) not checkout-able -> faithful seg impossible."
      echo "  Either fix the submodule or run with PERSEUS_NOSEG=1 (tracking-identical)."; return 1; }
  fi
  # the code imports droid_slam.cao_seg.* and loads droid_slam/cao_seg/src/models/*.pth
  # (test_one_line_import.py:21-27) but the tree ships it at droid_slam/MDE/application/cao_seg
  # (README placement) -> alias them with a symlink (env wiring, not a method change).
  [ -e "$SLAM/droid_slam/cao_seg" ] || ln -sfn "$SLAM/droid_slam/MDE/application/cao_seg" "$SLAM/droid_slam/cao_seg"
  # patch (m) [CONFIRMED vs MDE@427605c source]: inference_single multiplies the frame-sized prediction
  # by create_circle_mask's FIXED np.ones((1080,1080)) (the circle logic is already stubbed to all-ones
  # upstream; frame_rgb arg ignored). Works on their rig only because their calib crops to 1080x1080;
  # on CRCD 720x1280 it is (720,1280)*(1080,1080) -> numpy broadcast crash at frame 0. Fix = size the
  # ones-mask to the frame. NUMERICALLY A NO-OP (x*1 either way) -> pure shape repair, method untouched.
  local MFILE="$SLAM/droid_slam/MDE/application/cao_seg/src/models/test_one_line_import.py"
  python3 - "$MFILE" <<'PY' || { echo "FATAL[env]: mask-shape patch anchor missing (MDE submodule moved?) -> inspect $MFILE"; return 1; }
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
MARK = "[DDS-crcd-maskshape]"
if MARK in s:
    print("[env] mask-shape patch already applied"); sys.exit(0)
old = "mask = np.ones((height, width), dtype=np.uint8)"
new = "mask = np.ones(frame_rgb.shape[:2], dtype=np.uint8)  # [DDS-crcd-maskshape] all-ones (upstream stub) sized to the frame; x*1 no-op"
assert old in s, "anchor gone: " + old
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print("[env] patched create_circle_mask -> frame-shaped ones mask (broadcast-crash fix, numeric no-op)")
PY
  # RECURSIVE lookup: the Box share is a FOLDER (CAO_BPH_MDE_Segmentation) with the lab's OWN
  # subfolders/filenames (e.g. CAO/Seg/..., CAO/MDE/..., plus BPH variants we don't use). Resolution
  # order per expected file: (1) explicit override SEG_SRC_SEG / SEG_SRC_MDE (absolute path), (2) exact
  # expected filename anywhere below WEIGHTS_DRIVE, (3) role heuristic: a UNIQUE *.pth on a CAO path
  # matching the role (*seg* / *mde*) -- ambiguity (>1 candidate) fails loud, never guesses.
  # Whatever source is found is COPIED TO the expected name (the code hardcodes its load paths).
  resolve_weight_src(){ local dest=$1 pat="" ov=""
    case "$dest" in
      spie_cao_tumor_segmentation.pth) ov="${SEG_SRC_SEG:-}"; pat='*seg*';;
      mde_cao_518.pth)                 ov="${SEG_SRC_MDE:-}"; pat='*mde*';;
    esac
    [ -n "$ov" ] && [ -f "$ov" ] && { echo "$ov"; return 0; }
    local hit; hit=$(find "$WEIGHTS_DRIVE" -type f -name "$dest" 2>/dev/null | head -1)
    [ -n "$hit" ] && { echo "$hit"; return 0; }
    [ -n "$pat" ] || return 0
    local cands n
    cands=$(find "$WEIGHTS_DRIVE" -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.ckpt' \) -ipath '*cao*' -ipath "$pat" 2>/dev/null)
    n=$(printf '%s' "$cands" | grep -c . || true)
    if [ "$n" = 1 ]; then echo "$cands"; return 0; fi
    [ "$n" -gt 1 ] && { echo "[env] AMBIGUOUS candidates for $dest (set SEG_SRC_SEG/SEG_SRC_MDE):" >&2
                        printf '%s\n' "$cands" | sed 's/^/    /' >&2; }
    return 0
  }
  local missing=0 f SRC
  for f in $SEG_WEIGHT_FILES; do
    if [ -f "$SEG_MODELS_DIR/$f" ]; then echo "[env] seg weight present: $f"; continue; fi
    SRC=$(resolve_weight_src "$f")
    if [ -n "$SRC" ]; then
      mkdir -p "$SEG_MODELS_DIR"; cp -f "$SRC" "$SEG_MODELS_DIR/$f" \
        && echo "[env] seg weight $f <- $SRC" || missing=1
    else missing=1; echo "[env] seg weight MISSING: $f (searched $WEIGHTS_DRIVE recursively + CAO role heuristic)"; fi
  done
  [ "$missing" = 0 ] || {
    echo "FATAL[env]: seg/MDE weights absent and the Box link is manual-download-only."
    echo "  1) Download from: $SEG_BOX_URL"
    echo "  2) Save under Drive anywhere below: $WEIGHTS_DRIVE (searched recursively)"
    echo "  3) Re-run: bash Addons/colab/run_perseus.sh env"
    echo "  OR run tracking-identical without them: PERSEUS_NOSEG=1 bash Addons/colab/run_perseus.sh env"
    echo "  checkpoint files actually present under $WEIGHTS_DRIVE:"
    find "$WEIGHTS_DRIVE" -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.ckpt' \) 2>/dev/null | head -15 | sed 's/^/    /'
    echo "  -> map explicitly with: SEG_SRC_SEG=<abs path to the CAO seg ckpt> SEG_SRC_MDE=<abs path to the CAO MDE ckpt>"
    return 1; }
}

# ------------------------------------------------------------------- ENV ----
# patch (f): guard the fork's confidence-map tail in factor_graph.update() for the FILLER graph.
# The fork appends a confidence-viz block that picks the edge between the two most-recent keyframes
# (all_nodes topk-2) and upsamples its BA weight into video.confidence_maps. In the trajectory
# filler's temporary graph EVERY edge is keyframe->fill-frame (trajectory_filler.py:68-69) -> no
# KF-KF edge exists -> edge_indices empty -> edge_indices[0] IndexError. This is WHY the authors
# shipped terminate() disabled: their own viz addition breaks their own filler. The block runs AFTER
# the BA solve and only writes confidence_maps (viz/telemetry; read only by the display path) ->
# skipping the empty case is POSE-NEUTRAL. Idempotent + fail-loud.
apply_filler_confidence_guard(){
  local F="$PERSEUS/SegmentedSLAM/droid_slam/factor_graph.py"
  [ -f "$F" ] || { echo "[env] FATAL $F missing"; return 1; }
  python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding='utf-8').read()
MARK = "[DDS-fillguard]"
if MARK in s:
    print("[env] filler-confidence guard already applied"); sys.exit(0)
old = "            edge_index = edge_indices[0]"
assert s.count(old) == 1, f"anchor not unique/missing ({s.count(old)}x): edge_index = edge_indices[0]"
new = ("            if edge_indices.numel() == 0:  # [DDS-fillguard] filler graph has no KF-KF edge ->\n"
       "                self.age += 1              # confidence-viz n/a there; BA solve already done above\n"
       "                return                     # (pose-neutral: this tail only writes confidence_maps)\n"
       "            edge_index = edge_indices[0]")
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print("[env] patched factor_graph.update -> empty-edge guard on the confidence-viz tail (pose-neutral)")
PY
}

build_env(){
  [ -d "$PERSEUS/.git" ] || git clone --recursive "$PERSEUS_URL" "$PERSEUS" || { echo "FATAL clone $PERSEUS_URL"; exit 30; }
  apply_terminate_restore_patch || exit 30
  apply_filler_confidence_guard || exit 30
  [ "$PERSEUS_NOSEG" = 1 ] && { apply_noseg_stub_patch || exit 30; }
  if [ "${REBUILD_EXT:-0}" != 1 ] && [ -x "$ENV_PY" ] \
     && PYTHONPATH= "$ENV_PY" -c "import droid_backends, lietorch" 2>/dev/null; then
    echo "[env] $ENV_NAME ready (droid_backends imports; REBUILD_EXT=1 to recompile)"
  else
    if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
      echo "[env] installing miniconda -> $CONDA_ROOT"
      wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
      bash /tmp/mc.sh -b -p "$CONDA_ROOT" || { echo "FATAL miniconda"; exit 30; }
    fi
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda config --set channel_priority flexible 2>/dev/null || true
    # accept Anaconda ToS (defaults channels) if the gate is present (known Colab trap)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" -c conda-forge python=3.12 pip setuptools wheel \
       || { echo "FATAL conda create"; exit 30; }
    # conda-forge python may not bundle pip -> ensure explicitly (idempotent)
    PYTHONPATH= "$ENV_PY" -m pip --version >/dev/null 2>&1 \
       || conda install -y -n "$ENV_NAME" -c conda-forge pip setuptools wheel \
       || { echo "FATAL: pip not installable into $ENV_NAME"; exit 30; }
    local ENV_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
    echo "[env] cuda-toolkit 11.8 (nvcc) into $ENV_NAME"
    conda install -y -n "$ENV_NAME" -c "nvidia/label/cuda-11.8.0" cuda-toolkit || echo "[env] WARN cuda-toolkit"
    # constraints: NOTHING may upgrade the cu118 torch (a dep's `install_requires: torch` would pull a
    # cu12x torch-2.x wheel -> extension build/runtime mismatch). setuptools<81 keeps pkg_resources
    # available for legacy `setup.py install` paths.
    printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\nnumpy<2\nsetuptools<81\n' > /tmp/perseus_constraints.txt
    echo "[env] torch 2.5.1+cu118 via authors' requirements_torch_118.txt (embedded -i cu118 index, honored verbatim)"
    PYTHONPATH= "$ENV_PY" -m pip install -r "$SLAM/requirements_torch_118.txt" \
       || { echo "FATAL torch install (requirements_torch_118.txt)"; exit 30; }
    PYTHONPATH= "$ENV_PY" -c "import torch;v=torch.__version__;c=str(torch.version.cuda);assert v.startswith('2.5.1') and c.startswith('11.8'),(v,c);print('[env] torch',v,'cuda',c,'OK')" \
       || { echo "FATAL: torch is not 2.5.1/cu118 after install"; exit 30; }
    echo "[env] requirements_others.txt (constrained; sam2/smp/albumentations only feed the seg path)"
    PYTHONPATH= "$ENV_PY" -m pip install -c /tmp/perseus_constraints.txt -r "$SLAM/requirements_others.txt" \
       || echo "[env] WARN some requirements_others deps failed (fatal only for FAITHFUL seg; smoke below decides)"
    echo "[env] torch-scatter from the pyg cu118 wheel index (README verbatim)"
    PYTHONPATH= "$ENV_PY" -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html --no-index \
       || echo "[env] WARN torch-scatter failed (visualization-only dep; tracking unaffected under --disable_vis)"
    # build droid_backends + lietorch (setup.py HARDCODES -gencode sm_60..86 -> TORCH_CUDA_ARCH_LIST is
    # ignored for these exts; export it anyway as a defensive no-op for any stray sub-build).
    local CC; CC=$(PYTHONPATH= "$ENV_PY" -c "import torch;print('%d.%d'%torch.cuda.get_device_capability())" 2>/dev/null)
    [ -n "$CC" ] || CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$CC" ] || CC=7.5
    case "$CC" in 9.0) echo "[env] WARN GPU cc=9.0 (H100): setup.py's hardcoded gencode list tops out at sm_86 with NO PTX -> kernels won't run. Use T4/A100.";; esac
    echo "[env] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) compute_cap=$CC (setup.py gencodes sm_60..86)"
    echo "[env] building droid_backends + lietorch (python setup.py install; VERBOSE log -> /content/perseus_ext_build.log)"
    # `env` prefix: expanded VAR=val tokens are not parsed as assignments by bash -> env accepts them as args.
    ( cd "$SLAM" && env PYTHONPATH= CUDA_HOME="$ENV_ROOT" PATH="$ENV_ROOT/bin:$PATH" \
        TORCH_CUDA_ARCH_LIST="${CC}+PTX" "$ENV_PY" setup.py install ) 2>&1 | tee /content/perseus_ext_build.log
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      # README's crypt.h note: py<=3.11-era pyconfig chains may include crypt.h absent from the env
      if grep -q "crypt.h" /content/perseus_ext_build.log && [ -f /usr/include/crypt.h ]; then
        echo "[env] build failed on crypt.h -> cp /usr/include/crypt.h into the env (README remedy) + retry"
        mkdir -p "$ENV_ROOT/include"; cp -f /usr/include/crypt.h "$ENV_ROOT/include/"
        ( cd "$SLAM" && env PYTHONPATH= CUDA_HOME="$ENV_ROOT" PATH="$ENV_ROOT/bin:$PATH" \
            TORCH_CUDA_ARCH_LIST="${CC}+PTX" "$ENV_PY" setup.py install ) 2>&1 | tee -a /content/perseus_ext_build.log
        [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FATAL[env]: extension build failed after crypt.h retry (log: /content/perseus_ext_build.log)"; exit 30; }
      else
        echo "FATAL[env]: extension build failed (log: /content/perseus_ext_build.log)"; exit 30
      fi
    fi
  fi
  fetch_droid_weights || exit 30
  ensure_seg_weights  || exit 30
  echo "[env] smoke import (core tracking chain; exit 30 on fail):"
  ( cd "$SLAM" && env PYTHONPATH="$SLAM/droid_slam" "$ENV_PY" - <<'PY'
import importlib, sys
ok = True
for m in ("torch", "lietorch", "droid_backends", "cv2"):
    try: importlib.import_module(m); print(f"  {m}: OK")
    except Exception as e: ok = False; print(f"  {m}: FAIL -> {e}")
try:
    from droid import Droid; print("  droid.Droid: OK")
except Exception as e: ok = False; print(f"  droid.Droid: FAIL -> {e}")
try:
    import torch_scatter; print("  torch_scatter: OK (viz-only)")
except Exception as e: print(f"  torch_scatter: WARN -> {e} (visualization-only; fine under --disable_vis)")
import torch; print("  torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
sys.exit(0 if ok else 30)
PY
  ) || { echo "FATAL[env]: smoke import failed (exit 30)"; exit 30; }
  if [ "$PERSEUS_NOSEG" != 1 ]; then
    echo "[env] faithful-mode smoke: SegMDEInference import chain (sam2/smp/albumentations/MDE submodule):"
    ( cd "$SLAM" && env PYTHONPATH="$SLAM" "$ENV_PY" -c \
        "from droid_slam.MDE.application.cao_seg.src.models.test_one_line_import import SegMDEInference; print('  SegMDEInference import: OK')" ) \
      || { echo "FATAL[env]: SegMDEInference import chain broken -> fix deps/submodule, or use PERSEUS_NOSEG=1"; exit 30; }
  fi
  echo "[env] OK -> $ENV_PY"
}

# ------------------------------------------------- stage restore (per UP) ---
stage_restore(){ local UP=$1 DD="/content/rect_staged/$UP"
  if [ ! -f "$DD/.STAGED" ]; then
    local TGZ="$STAGE_CACHE_DIR/${UP}_$STAGEVER.tar"
    [ -f "$TGZ" ] || { say "  FATAL: no stage cache $TGZ -- run the rect_bench staging for $UP first (rect_bench_best_vs_base_20260626.sh stages+caches it)"; return 1; }
    say "  restoring rect stage cache ($(du -h "$TGZ" | cut -f1))"
    mkdir -p /content/rect_staged && tar -xf "$TGZ" -C /content/rect_staged || { say "  FATAL untar"; return 1; }
    [ -f "$DD/.STAGED" ] || { say "  FATAL: restored tar lacks .STAGED"; return 1; }
  fi
  # demo.py streams EVERY file in --imagedir via sorted(os.listdir()) -> the dir must contain
  # ONLY the rectified-left frames (r.png are tar-excluded; enforce, don't assume).
  local NF NBAD
  NF=$(ls "$DD/video_frames"/*l.png 2>/dev/null | wc -l)
  NBAD=$(ls "$DD/video_frames" 2>/dev/null | grep -v 'l\.png$' | wc -l)
  [ "$NF" -gt 0 ] || { say "  FATAL: no *l.png in $DD/video_frames"; return 1; }
  [ "$NBAD" -eq 0 ] || { say "  FATAL: $NBAD non-left files in $DD/video_frames (demo.py would stream them as frames) -> clean the staging"; return 1; }
  [ -f "$DD/groundtruth.txt" ] || { say "  FATAL: $DD/groundtruth.txt missing"; return 1; }
  [ -f "$DD/rectified_calib.txt" ] || { say "  FATAL: $DD/rectified_calib.txt missing"; return 1; }
  say "  $UP staged: $NF left frames, GT rows=$(grep -v '^#' "$DD/groundtruth.txt" | grep -cve '^\s*$')"
}

# CALIB.txt (13-value single line demo.py:153-154 unpacks): no crop (cr = full frame),
# rectified => zero distortion. W/H via cv2 from the first staged frame (mirror mk_sni_cfg).
mk_calib(){ local UP=$1 DD="/content/rect_staged/$UP" CAL=$2
  PYTHONPATH= "$DDS_PY" - "$DD" "$CAL" <<'PY' || return 1
import glob, os, sys
import cv2
DD, CAL = sys.argv[1], sys.argv[2]
fr = sorted(glob.glob(os.path.join(DD, "video_frames", "*l.png")))
assert fr, f"no frames in {DD}/video_frames"
img = cv2.imread(fr[0]); assert img is not None, f"unreadable frame {fr[0]}"
H, W = img.shape[:2]
intr = {}
for ln in open(os.path.join(DD, "rectified_calib.txt")):
    p = ln.split()
    if len(p) >= 2 and p[0] in ("fx", "fy", "cx", "cy"):
        intr[p[0]] = float(p[1])
assert all(k in intr for k in ("fx", "fy", "cx", "cy")), f"rectified_calib.txt incomplete: {intr}"
os.makedirs(os.path.dirname(CAL), exist_ok=True)
with open(CAL, "w") as f:
    f.write(f"0 0 {W} {H} {intr['fx']} {intr['fy']} {intr['cx']} {intr['cy']} 0 0 0 0 0\n")
print(f"[calib] {CAL}: cr=(0,0,{W},{H}) fx={intr['fx']} fy={intr['fy']} cx={intr['cx']} cy={intr['cy']} dist=0 (rectified)")
PY
}

# ------------------------------------------------- run ONE CRCD snippet -----
run_crcd_one(){
  local NAME; NAME=$(echo "$1" | tr 'A-Z' 'a-z')
  local UP; UP=$(echo "$NAME" | tr 'a-z' 'A-Z')
  # NOSEG segregation: claimed tracking-identical, but it gets its OWN dir/row label so it can
  # never masquerade as the faithful row without the footnote.
  local VS=""; [ "$PERSEUS_NOSEG" = 1 ] && VS="_noseg"
  local OUT="$DRIVE_OUT/${UP}${VS}"; mkdir -p "$OUT"
  local DD="/content/rect_staged/$UP"

  # 1) .DONE idempotency gate
  [ -f "$OUT/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { say "$UP$VS done -> skip (FORCE=1 to redo)"; return 0; }
  rm -f "$OUT/.FAILED"

  # 2) stage + calib
  stage_restore "$UP" || { echo "FAILED stage" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  local CAL="$SLAM/calib/CRCD_${UP}.txt"
  mk_calib "$UP" "$CAL" || { echo "FAILED calib" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  local IMDIR="$DD/video_frames"
  local NFRAMES; NFRAMES=$(ls "$IMDIR"/*l.png | wc -l)

  # 3) patches must be in place (env applies them; re-assert here so a stale clone fails loud)
  grep -q "PERSEUS_TERMINATE_RESTORE" "$SLAM/droid_slam/droid.py" || { apply_terminate_restore_patch \
     || { echo "FAILED terminate patch" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }; }
  [ "$PERSEUS_NOSEG" = 1 ] && ! grep -q "PERSEUS_NOSEG_STUB" "$SLAM/demo.py" && { apply_noseg_stub_patch \
     || { echo "FAILED noseg patch" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }; }

  # 4) run demo.py from $SLAM (traj_est.npy saves to cwd -> clear stale FIRST, capture after).
  #    FAITHFUL flags only: --stride 1 (dataset-necessary: N poses vs N GT rows), --disable_vis
  #    (headless), --buffer (author-exposed capacity knob). NO DROID hyperparameter overrides.
  rm -f "$SLAM/traj_est.npy"
  local RECON_ARGS=""
  [ "$PERSEUS_RECON" = 1 ] && RECON_ARGS="--reconstruction_path ${UP}${VS}" \
     && echo "[$UP] PERSEUS_RECON=1 -> --reconstruction_path ${UP}${VS} (demo.py auto-enables --upsample: more VRAM)"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 30; done \
    > "$OUT/vram_samples.txt" ) & local VPID=$!
  say "$UP$VS: PERSEUS demo.py (frames=$NFRAMES stride=1 buffer=$PERSEUS_BUFFER noseg=$PERSEUS_NOSEG)"
  ( cd "$SLAM" && env PYTHONPATH= PERSEUS_NOSEG="$PERSEUS_NOSEG" \
      "$ENV_PY" demo.py --imagedir "$IMDIR" --calib "$CAL" --stride 1 --disable_vis \
      --buffer "$PERSEUS_BUFFER" $RECON_ARGS ) 2>&1 | tee "$OUT/run.log"
  local RC=${PIPESTATUS[0]}; kill $VPID 2>/dev/null
  sort -rn "$OUT/vram_samples.txt" 2>/dev/null | head -1 | xargs -I{} echo "[peak VRAM] {} MiB" | tee -a "$OUT/run.log"
  [ "$RC" -eq 0 ] || { echo "FAILED demo.py rc=$RC" > "$OUT/status.txt"; touch "$OUT/.FAILED"; say "$UP$VS FAILED (isolated) -> next"; return 1; }
  [ -s "$SLAM/traj_est.npy" ] || { echo "FAILED no traj_est.npy" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  cp -f "$SLAM/traj_est.npy" "$OUT/traj_est.npy"
  [ "$PERSEUS_RECON" = 1 ] && [ -d "$SLAM/reconstructions/${UP}${VS}" ] \
     && { cp -rf "$SLAM/reconstructions/${UP}${VS}" "$OUT/reconstruction" && echo "[$UP] reconstruction -> $OUT/reconstruction"; }

  # 5) traj_est.npy (N,7 c2w, quat xyzw) -> est_c2w_data.txt (16-float 4x4 c2w; layout derivation
  #    + test_tum.py citations live in Addons/eval/perseus_traj_to_est.py's docstring)
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/perseus_traj_to_est.py" \
     --traj "$OUT/traj_est.npy" --out "$OUT/est_c2w_data.txt" --expected_frames "$NFRAMES" \
     || { echo "FAILED traj->est" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }
  [ -s "$OUT/est_c2w_data.txt" ] || { echo "FAILED empty est" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 6) GT prefix-truncation guard (verbatim from run_semgauss.sh): a capped/partial run covers a
  #    PREFIX of the sequence; sim3_ate would otherwise uniform-resample the full GT onto it
  #    (WRONG pairing). Full runs: counts equal -> untouched.
  rm -f "$OUT/sim3_metrics.txt"
  local GT_EVAL="$DD/groundtruth.txt"
  local n_est n_gt; n_est=$(grep -cve '^\s*$' "$OUT/est_c2w_data.txt")
  n_gt=$(grep -v '^#' "$GT_EVAL" | grep -cve '^\s*$')
  if [ "$n_est" -lt "$n_gt" ]; then
    grep -v '^#' "$GT_EVAL" | awk 'NF' | head -n "$n_est" > "$OUT/groundtruth_prefix.txt"
    GT_EVAL="$OUT/groundtruth_prefix.txt"
    echo "[$UP] est=$n_est < gt=$n_gt -> Sim3 eval vs the first $n_est GT rows (prefix; PARTIAL-SEQUENCE, flag in the table)"
  fi

  # 7) Sim3 ATE (the CANONICAL CRCD tracking metric; NEVER rigid)
  PYTHONPATH= "$DDS_PY" "$REPO/Addons/eval/sim3_ate.py" \
     --est "$OUT/est_c2w_data.txt" --gt "$GT_EVAL" \
     --name "CRCD $UP PERSEUS$VS" --out "$OUT/sim3_metrics.txt" \
     || { echo "FAILED sim3_ate" > "$OUT/status.txt"; touch "$OUT/.FAILED"; return 1; }

  # 8) metrics.json — tracking fields + nulls for render fields (aggregate prints "--")
  PYTHONPATH= "$DDS_PY" - "$OUT" "$UP$VS" "$PERSEUS_NOSEG" "$PERSEUS_BUFFER" <<'PY'
import json, os, re, sys
out, name, noseg, buf = sys.argv[1], sys.argv[2], sys.argv[3] == "1", int(sys.argv[4])
sim = open(os.path.join(out, 'sim3_metrics.txt'), encoding='utf-8', errors='ignore').read()
def grab(pat):
    m = re.search(pat, sim); return float(m.group(1)) if m else None
ate_mean = grab(r'Sim3 ATE.*?:\s*[0-9.]+\s*/\s*([0-9.]+)\s*/')
d = dict(scene=name,
         sim3_ate_mean_mm=ate_mean,
         sim3_ate_max_mm=grab(r'Sim3 ATE.*?:\s*[0-9.]+\s*/\s*[0-9.]+\s*/\s*[0-9.]+\s*/\s*([0-9.]+)\s*mm'),
         path_ratio=grab(r'est/GT path ratio\s*:\s*([0-9.]+)'),
         pearson_dom=grab(r'\|Pearson\| dom axis\s*:\s*([0-9.]+)'),
         psnr=None, ssim=None, lpips=None, depth_l1_mm=None,   # PERSEUS renders nothing (tracking-only arm)
         method='PERSEUS', noseg=noseg, stride=1, buffer=buf)
json.dump(d, open(os.path.join(out, 'metrics.json'), 'w'), indent=2)
print(f"[{name}] " + " ".join(f"{k}={v}" for k, v in d.items() if k != 'scene'))
PY
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"; sync
  say "$UP$VS DONE -> $OUT"
}

# ------------------------------------------------------------- aggregate ----
aggregate(){
  say "=== PERSEUS CRCD bench summary (TRACKING-ONLY; render columns are structurally --) ($DRIVE_OUT) ==="
  PYTHONPATH= "$DDS_PY" - "$DRIVE_OUT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
rows = []
for mj in sorted(glob.glob(os.path.join(root, '*', 'metrics.json'))):
    try: rows.append(json.load(open(mj)))
    except Exception: pass
def f(v, p=2):
    return ('%.*f' % (p, v)) if isinstance(v, (int, float)) else '--'
hdr = (f"{'snippet':<13}{'ATEmean':>9}{'ATEmax':>9}{'pathR':>7}{'|Pr|dom':>8}"
       f"{'PSNR':>7}{'SSIM':>7}{'LPIPS':>7}{'DL1mm':>8}")
print(hdr); print('-' * len(hdr))
for d in rows:
    print(f"{str(d.get('scene','?')):<13}{f(d.get('sim3_ate_mean_mm')):>9}{f(d.get('sim3_ate_max_mm')):>9}"
          f"{f(d.get('path_ratio')):>7}{f(d.get('pearson_dom')):>8}{f(d.get('psnr')):>7}"
          f"{f(d.get('ssim'),3):>7}{f(d.get('lpips'),3):>7}{f(d.get('depth_l1_mm')):>8}")
if not rows: print("(no metrics.json yet)")
PY
  local s UP VS=""
  [ "$PERSEUS_NOSEG" = 1 ] && VS="_noseg"
  for s in $BENCH5; do UP=$(echo "$s" | tr 'a-z' 'A-Z')
    [ -f "$DRIVE_OUT/${UP}${VS}/.DONE" ] || echo "  ${UP}${VS}: $(cat "$DRIVE_OUT/${UP}${VS}/status.txt" 2>/dev/null || echo 'not run')"
  done
}

# ------------------------------------------------------------- dispatch -----
case "$VERB" in
  env) build_env ;;
  crcd)
    [ -x "$ENV_PY" ] || { echo "FATAL: perseus env not built ($ENV_PY missing) - run 'env' first"; exit 30; }
    [ -d "$PERSEUS/.git" ] || { echo "FATAL: PERSEUS clone missing at $PERSEUS - run 'env' first"; exit 30; }
    [ -f "$SLAM/droid.pth" ] || { echo "FATAL: $SLAM/droid.pth missing - run 'env' first"; exit 30; }
    mkdir -p "$DRIVE_OUT"; exec > >(tee -a "$DRIVE_OUT/runbook.log") 2>&1
    SNIPS=${ARG:-bench5}
    { [ "$SNIPS" = bench5 ] || [ "$SNIPS" = all ]; } && SNIPS="$BENCH5"
    say "=== PERSEUS CRCD bench  repo=$(cd "$PERSEUS" 2>/dev/null && git rev-parse --short HEAD)  snippets='$SNIPS' noseg=$PERSEUS_NOSEG buffer=$PERSEUS_BUFFER -> $DRIVE_OUT ==="
    for s in $SNIPS; do run_crcd_one "$s" || echo "[$s] FAILED (isolated) -> next"; done
    aggregate ;;
  eval) aggregate ;;
  *) echo "usage: run_perseus.sh env | crcd [snippet|bench5] | eval"; exit 2 ;;
esac
