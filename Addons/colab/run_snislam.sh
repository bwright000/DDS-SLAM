#!/bin/bash
# ============================================================================
# run_snislam.sh — SNI-SLAM on the RECTIFIED 5-snippet CRCD benchmark (Arm-4).
#
#   SNI-SLAM's OWN model (our fork bwright000/SNI-SLAM, which carries the 16
#   proven CRCD fixes: CRCD dataset class, surgical intrinsics, sub-SNR lr,
#   CRCD-scaled tri-planes 60978a1, pose-freeze during mapping, torch-1.11
#   ckpt load, post-hoc renderer 7467371) run on the SAME staged data as the
#   DDS/SGS/SemSup arms — restored from the rect_bench Drive stage cache
#   (dds_cache/rect_staged/<NAME>_v2.tar: rectified frames + every-120
#   stereo-anchored METRIC MoGe depth (sc=1) + semantic_class + GT + bound.yaml
#   + rectified_calib.txt) — and judged with the SAME DDS harness battery:
#     1. Sim3 ATE (rmse/mean/median/max, mm) + recovered scale s
#        + est/GT path-ratio + Sim3-ALIGNED |Pearson| (dom axis + mean-xyz)
#        [Addons/eval/sim3_ate.py — NEVER the rigid output]
#     2. PSNR   3. SSIM   4. LPIPS(alex)   [Addons/eval/eval_rendering.py]
#     5. Depth-L1 input-vs-output self-consistency, mm [Addons/eval/depth_l1.py]
#     6. canonical 6-panel video [Addons/viz/generate_video.py]
#
#   BENCHMARK ADJUSTMENTS vs the fork's June configs (each deliberate):
#     - data = the rect_bench staging (rectified + anchored METRIC depth), NOT
#       the raw-left frame-0-anchor flow -> per-snippet cam.* from
#       rectified_calib.txt; bounds from the SAME bound.yaml as the DDS arm.
#     - truncation 0.01 hard-set (depth is already metric sc=1 -> the fork's
#       Phase-3.6/3.7 sc_factor rescale/trunc-scale machinery is NOT needed).
#     - semantics = OUR in-domain 4-class head seg/dinov2_crcd.pth (trained on
#       the 15 NON-benchmark snippets 2026-06-18 -> held-out on all 5) with
#       use_gt_semantic=False (METHOD-FAITHFUL DINO self-supervision).
#       GT_SEM=1 flips to GT-mask supervision (ablation, label it).
#     - mIoU dropped on CRCD (benchmark = the 5 metrics + video).
#
#   GPU: T4 OK for the sni env (cu113/sm_75). CRCD VRAM was only ever proven
#   on A100 -> shortest snippet runs FIRST as the de-facto smoke; per-snippet
#   peak VRAM is logged; a snippet failing (OOM) is isolated, not fatal.
#
#   Usage (fresh Colab T4, Drive mounted):
#     bash Addons/colab/run_snislam.sh env      # sni env + fork + seg assets (one-time)
#     nohup bash Addons/colab/run_snislam.sh crcd &> /content/sni_crcd.out & disown
#     bash Addons/colab/run_snislam.sh eval     # (re)aggregate
# ============================================================================
set -uo pipefail
DATE=${DATE:-$(date +%Y%m%d)}
REPO=${REPO:-/content/DDS-SLAM}
SNI_REPO=${SNI_REPO:-/content/sni-slam}
SNI_URL=https://github.com/bwright000/SNI-SLAM
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
SNI_ENV=${SNI_ENV:-sni}; SNI_PY="$CONDA_ROOT/envs/$SNI_ENV/bin/python"
ENV_CACHE=/content/drive/MyDrive/dds_cache/sni_env_bench.tar.gz
STAGE_CACHE_DIR=/content/drive/MyDrive/dds_cache/rect_staged; STAGEVER=v2
SEG_PTH=${SEG_PTH:-}                              # explicit override wins; else per-snippet resolution below
SEG_DIR_DRIVE=${SEG_DIR_DRIVE:-/content/drive/MyDrive/Outputs/seg}
SNI_GDRIVE_ID=1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h   # authors' folder (dinov2 backbone zip fallback)
DRIVE_OUT=${DRIVE_OUT:-/content/drive/MyDrive/Outputs/SNI-SLAM_bench_$DATE}
GT_SEM=${GT_SEM:-0}
SNIPPETS=${SNIPPETS:-"E3_005 C1_001 C2_001 C3_001 G3_001"}   # shortest-first (E3 = T4 smoke)
VERB=${1:-}

say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }

# ---- per-snippet in-domain seg head (OUR trained DINOv2/14 heads; snippet HELD OUT) ----
# Precedence (user-set canonical 2026-07-04): explicit SEG_PTH -> loso_v2_max fold (CANONICAL:
# the complete LOSO fold set, so all snippets use ONE recipe -> a consistent benchmark) ->
# loso_v2_b2 fold -> flat 15-snippet head -> loso_ref fold. dinov3 heads are patch-16 and
# can NOT load into SNI's /14 DINO2SEG -> never resolved here.
seg_head_for(){ local NAME=$1 c
  for c in "${SEG_PTH:-}" \
           "$SEG_DIR_DRIVE/loso_v2_max/dinov2_crcd_${NAME}.pth" \
           "$SEG_DIR_DRIVE/loso_v2_b2/dinov2_crcd_${NAME}.pth" \
           "$SEG_DIR_DRIVE/dinov2_crcd.pth" \
           "$SEG_DIR_DRIVE/loso_ref/dinov2_crcd_${NAME}.pth"; do
    [ -n "$c" ] && [ -f "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# ---------------------------------------------------------------- env ----
build_env(){
  [ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; return 1; }
  [ -d "$SNI_REPO/.git" ] || git clone -q "$SNI_URL" "$SNI_REPO" || { say "FATAL clone $SNI_URL"; return 1; }
  ( cd "$SNI_REPO" && git pull -q ) || true
  if [ "${REBUILD_ENV:-0}" != 1 ] && [ -x "$SNI_PY" ] \
     && PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d;assert torch.__version__.startswith('1.11') and pytorch3d.__version__=='0.7.1'" 2>/dev/null; then
    say "sni env ready (REBUILD_ENV=1 to rebuild)"
  else
    if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
      wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        && bash /tmp/mc.sh -b -p "$CONDA_ROOT" || { say "FATAL miniconda"; return 1; }
    fi
    "$CONDA_ROOT/bin/conda" tos accept --override-channels \
      --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    if [ -f "$ENV_CACHE" ]; then
      say "restoring sni env from Drive cache ($(du -h "$ENV_CACHE"|cut -f1))"
      mkdir -p "$CONDA_ROOT/envs/$SNI_ENV" && tar -xzf "$ENV_CACHE" -C "$CONDA_ROOT/envs/$SNI_ENV" \
        && PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d" 2>/dev/null \
        || { say "cache restore failed -> full build"; rm -rf "$CONDA_ROOT/envs/$SNI_ENV"; }
    fi
    if ! PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d" 2>/dev/null; then
      sudo apt-get -qq update && sudo apt-get -qq install -y libopenexr-dev 2>/dev/null || true
      say "conda env create (~25-40 min; the pytorch3d 0.7.1/cu113/py37 pin is the known landmine)"
      "$CONDA_ROOT/bin/conda" env create -n "$SNI_ENV" -f "$SNI_REPO/environment.yaml" \
        || { say "FATAL conda env create (check pytorch3d resolution; see fork scripts/colab_setup_sni.sh)"; return 1; }
      say "caching env to Drive (one-time)"
      tar -czf /tmp/sni_env.tar.gz -C "$CONDA_ROOT/envs/$SNI_ENV" . && mv -f /tmp/sni_env.tar.gz "$ENV_CACHE" || true
    fi
  fi
  PYTHONPATH= "$SNI_PY" - <<'PY' || { say "FATAL env smoke"; return 1; }
import torch, pytorch3d
assert torch.__version__.startswith('1.11'), torch.__version__
assert torch.cuda.is_available(), 'no CUDA'
print('[env] torch', torch.__version__, '| pytorch3d', pytorch3d.__version__, '| GPU', torch.cuda.get_device_name(0))
PY
  # ---- seg assets ----
  # DINOv2 backbone CODE (always -> sys.path for DINO2SEG) + the Replica head weights
  # (only the GT_SEM=1 path torch.loads seg/dinov2_replica.pth; the trained-head path carries its
  # own backbone in dinov2_crcd.pth). Fetch+place BOTH so either path works.
  if [ ! -d "$SNI_REPO/seg/facebookresearch_dinov2_main" ] || { [ "$GT_SEM" = 1 ] && [ ! -f "$SNI_REPO/seg/dinov2_replica.pth" ]; }; then
    say "fetching DINOv2 backbone code$([ "$GT_SEM" = 1 ] && echo ' + Replica head') (authors' Drive folder)"
    pip install -q gdown 2>/dev/null
    gdown --folder "https://drive.google.com/drive/folders/$SNI_GDRIVE_ID" -O /tmp/snidl --remaining-ok 2>/dev/null || true
    Z=$(find /tmp/snidl -name 'facebookresearch_dinov2_main.zip' | head -1)
    [ -n "$Z" ] && unzip -qo "$Z" -d "$SNI_REPO/seg/"
    for f in dinov2_replica.pth semantic_classes.pkl num_semantic_class.pkl; do
      S=$(find /tmp/snidl -name "$f" | head -1); [ -n "$S" ] && cp -f "$S" "$SNI_REPO/seg/$f"
    done
    [ -d "$SNI_REPO/seg/facebookresearch_dinov2_main" ] || { say "FATAL: DINOv2 backbone code not obtained (gdown quota? fetch manually into $SNI_REPO/seg/)"; return 1; }
    { [ "$GT_SEM" = 1 ] && [ ! -f "$SNI_REPO/seg/dinov2_replica.pth" ]; } && { say "FATAL: dinov2_replica.pth not obtained but GT_SEM=1 needs it (fetch manually into $SNI_REPO/seg/)"; return 1; } || true
  fi
  if [ "$GT_SEM" != 1 ]; then
    local MISS="" FOUND="" H
    for NAME in $SNIPPETS; do
      if H=$(seg_head_for "$NAME"); then say "seg head [$NAME]: $H"; FOUND="$FOUND $NAME"; else MISS="$MISS $NAME"; fi
    done
    [ -n "$MISS" ] && say "WARN: no in-domain seg head for:$MISS -> those snippets will be SKIPPED (isolated). Train the fold or run them GT_SEM=1. Canonical source: $SEG_DIR_DRIVE/loso_v2_max/ ."
    [ -n "$FOUND" ] || { say "FATAL: no seg head for ANY snippet -> check $SEG_DIR_DRIVE/loso_v2_max/, or GT_SEM=1 for the GT-mask ablation."; return 1; }
  fi
  # DDS-harness eval deps in the SYSTEM python (no tinycudann needed for SNI eval)
  python3 -c "import lpips" 2>/dev/null || pip install -q lpips
  python3 -c "import cv2, matplotlib, imageio" 2>/dev/null || pip install -q opencv-contrib-python matplotlib imageio imageio-ffmpeg
  say "env DONE"
}

# ------------------------------------------------- stage bridge (per snippet)
# rect_bench tar -> the fork CRCD loader layout (rgb_%06d / depth_%06d /
# semantic_class_%06d re-indexed 0..N-1 in lockstep + traj.txt 16-float c2w).
stage_bridge(){ local NAME=$1 DD="/content/rect_staged/$NAME" SD="$SNI_REPO/data/CRCD/$NAME"
  if [ -f "$SD/.BRIDGED" ]; then say "  $NAME already bridged"; return 0; fi
  if [ ! -f "$DD/.STAGED" ]; then
    local TGZ="$STAGE_CACHE_DIR/${NAME}_$STAGEVER.tar"
    [ -f "$TGZ" ] || { say "  FATAL: no stage cache $TGZ -- run the rect_bench staging for $NAME first (rect_bench_best_vs_base_20260626.sh stages+caches it)"; return 1; }
    say "  restoring rect stage cache ($(du -h "$TGZ"|cut -f1))"
    mkdir -p /content/rect_staged && tar -xf "$TGZ" -C /content/rect_staged || { say "  FATAL untar"; return 1; }
    [ -f "$DD/.STAGED" ] || { say "  FATAL: restored tar lacks .STAGED"; return 1; }
  fi
  say "  bridging -> SNI layout ($SD)"
  PYTHONPATH= "$SNI_PY" - "$DD" "$SD" <<'PY' || return 1
import glob, os, sys
import numpy as np
from scipy.spatial.transform import Rotation as R
DD, SD = sys.argv[1], sys.argv[2]
import re
def _fid(p):  # frame-id = trailing digit run in the basename (robust to unpadded names)
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[-1]) if m else -1
rgb  = sorted(glob.glob(f"{DD}/video_frames/*l.png"), key=_fid)   # numeric sort (lexicographic breaks on unpadded ids)
dep  = sorted(glob.glob(f"{DD}/depth/[0-9]*.png"), key=_fid)
sem  = sorted(glob.glob(f"{DD}/semantic_class/*.png"), key=_fid)
gt   = [l.split() for l in open(f"{DD}/groundtruth.txt") if l.strip() and not l.startswith('#')]
n = min(len(rgb), len(dep), len(sem), len(gt))
assert n > 0, f"empty staging: rgb={len(rgb)} dep={len(dep)} sem={len(sem)} gt={len(gt)}"
if not (len(rgb) == len(dep) == len(sem) == len(gt)):
    print(f"[bridge] WARN count mismatch rgb={len(rgb)} dep={len(dep)} sem={len(sem)} gt={len(gt)} -> truncating to {n}")
# CRITICAL (audit 2026-07-04): modalities are paired BY POSITION below. If depth/semantic are not 1:1
# per-frame with rgb (e.g. depth sampled sparsely), rgb[i]/dep[i]/sem[i] would be DIFFERENT frames and the
# loader's own equal-count assert still passes -> SILENT rgb-vs-depth-vs-pose misalignment. Guard it:
_rid, _did, _sid = [_fid(p) for p in rgb[:n]], [_fid(p) for p in dep[:n]], [_fid(p) for p in sem[:n]]
_mis = [(i, _rid[i], _did[i], _sid[i]) for i in range(n) if not (_rid[i] == _did[i] == _sid[i])]
assert not _mis, (f"[bridge] FRAME-ID MISALIGNMENT at {len(_mis)} idx (first: idx {_mis[0][0]} "
                  f"rgb#{_mis[0][1]} dep#{_mis[0][2]} sem#{_mis[0][3]}). depth/semantic not 1:1 per-frame "
                  f"with rgb -> re-stage {DD} with one depth+semantic per rgb frame.")
for sub in ("rgb", "depth", "semantic_class"):
    os.makedirs(f"{SD}/{sub}", exist_ok=True)
for i in range(n):
    for src, dst in ((rgb[i], f"{SD}/rgb/rgb_{i:06d}.png"),
                     (dep[i], f"{SD}/depth/depth_{i:06d}.png"),
                     (sem[i], f"{SD}/semantic_class/semantic_class_{i:06d}.png")):
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
with open(f"{SD}/traj.txt", "w") as f:     # TUM -> 16-float c2w (OpenCV frame; loader flips) — fork-proven
    for row in gt[:n]:
        t = [float(x) for x in row[1:4]]; q = [float(x) for x in row[4:8]]
        c2w = np.eye(4); c2w[:3, :3] = R.from_quat(q).as_matrix(); c2w[:3, 3] = t
        f.write(" ".join(f"{v:.10f}" for v in c2w.reshape(-1)) + "\n")
print(f"[bridge] {n} frames linked + traj.txt written")
PY
  touch "$SD/.BRIDGED"
}

# ---------------------------------------------- per-snippet config authoring
mk_sni_cfg(){ local NAME=$1 DD="/content/rect_staged/$NAME" SD="$SNI_REPO/data/CRCD/$NAME" CFG="$SNI_REPO/configs/CRCD/bench_${NAME}.yaml"
  PYTHONPATH= "$SNI_PY" - "$NAME" "$DD" "$SD" "$CFG" "$GT_SEM" <<'PY' || return 1
import glob, sys, yaml, cv2, os
NAME, DD, SD, CFG, GT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
b = yaml.safe_load(open(f"{DD}/bound.yaml"))["mapping"]          # SAME bounds as the DDS arm
img = cv2.imread(sorted(glob.glob(f"{SD}/rgb/rgb_*.png"))[0]); H, W = img.shape[:2]
intr = {}
for ln in open(f"{DD}/rectified_calib.txt"):
    p = ln.split()
    if len(p) >= 2 and p[0] in ("fx", "fy", "cx", "cy"):
        intr[p[0]] = float(p[1])
assert all(k in intr for k in ("fx", "fy", "cx", "cy")), intr
# CRCD metric depth is ~0.1m. The GT-pose ablation (SNI_GT_POSE=1) proved the SDF/geometry forms fine at
# this scale (sharp rendered depth, PSNR 9->11.2) -> the earlier BLACK renders were TRACKER over-travel
# (est path ~2.6x GT), NOT scale. SNI_SCALE is REFUTED as a fix: the tracker render-loss is scale-INVARIANT
# (scale=8 gave byte-identical Sim3 + worse PSNR + OOM). The REMAINING issue is a COLOUR wash (grey RGB,
# sharp depth): the RGB decoder cats the DINO semantic feature (decoders.py:117-119), so colour rides on the
# seg head. GT_SEM couples head+target -> GT_SEM=1 (Replica head + GT-mask target) == June's coloured config;
# GT_SEM=0 (loso 4-class head + DINO argmax) == the wash. Under investigation.
_scale = float(os.environ.get("SNI_SCALE", "1"))          # REFUTED as a fix (see above). Default off; kept for experiments.
_gtpose = os.environ.get("SNI_GT_POSE", "0") == "1"       # ablation: map+track at GT poses (render ceiling)
# T4 RESOURCE CAP (not a faithfulness knob): pixels is a memory/quality knob. The authors' faithful
# values (tracking 2000 / mapping 4000) can OOM a RAM-limited T4. Setting SNI_TRACK_PIXELS/SNI_MAP_PIXELS
# caps ONLY the per-iter ray budget while lr/const_speed/joint_opt/iters/keyframe cadence stay faithful.
# Default = unset -> authors' values (fully faithful). Document any cap as a T4 hardware limit.
_trk_px = os.environ.get("SNI_TRACK_PIXELS", "")
_map_px = os.environ.get("SNI_MAP_PIXELS", "")
cfg = {
  "inherit_from": "configs/CRCD/crcd_sni_base.yaml",
  "scale": _scale,
  # GT-pose ablation must be a TRUE GT-pose run: freeze SNI's joint pose BA. Mapper.py:469 has NO
  # use_gt_pose guard, so joint_opt otherwise drifts the GT poses during mapping (path-ratio ~2.6
  # even at GT poses). Normal runs keep joint_opt=True (authors' faithful default).
  "mapping": {"bound": b["bound"], "marching_cubes_bound": b["marching_cubes_bound"],
              **({"joint_opt": False} if _gtpose else {})},
  "data": {"input_folder": f"data/CRCD/{NAME}/", "output": f"output/CRCD/bench_{NAME}"},
  "cam": {"H": H, "W": W, "fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
          "png_depth_scale": 10000, "crop_edge": 0},
  "model": {"truncation": round(0.01 * _scale, 4),
            "cnn": {"n_classes": 4,
                    "pretrained_model_path": ("seg/dinov2_replica.pth" if GT else "seg/dinov2_crcd.pth")}},
  "func": {"use_gt_semantic": bool(GT), "use_gt_pose": _gtpose},
}
# SNI defaults keyframe_device/feature_device to "cpu" (system RAM) to spare VRAM. On a T4 with
# 12GB RAM but ~8GB free VRAM, that's inverted -> SNI_STORE_DEVICE=cuda:0 puts keyframe+feature
# storage on the GPU, freeing the scarce RAM (the no-High-RAM path; long snippets may then hit VRAM).
_sd = os.environ.get("SNI_STORE_DEVICE", "")
if _sd:
    cfg["keyframe_device"] = _sd
    cfg["feature_device"] = _sd
if _map_px: cfg["mapping"]["pixels"] = int(_map_px)          # T4 cap; else inherits authors' 4000
if _trk_px: cfg.setdefault("tracking", {})["pixels"] = int(_trk_px)   # T4 cap; else inherits authors' 2000
yaml.safe_dump(cfg, open(CFG, "w"), sort_keys=False)
print(f"[cfg] {NAME}: HxW={H}x{W} fx={intr['fx']:.1f} bound={b['bound']} scale={_scale} "
      f"trunc={round(0.01*_scale,4)} gt_pose={_gtpose} gt_sem={GT} store_device={_sd or 'cpu(default)'} "
      f"pixels=trk:{_trk_px or 'auth2000'}/map:{_map_px or 'auth4000'} -> {CFG}")
PY
}

# ------------------------------------------------------------- run one ----
run_one(){ local NAME=$1 SD="$SNI_REPO/data/CRCD/$NAME" OUTD="$SNI_REPO/output/CRCD/bench_${NAME}" DST="$DRIVE_OUT/$NAME"
  mkdir -p "$DST"
  [ -f "$DST/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { say "$NAME done -> skip"; return 0; }
  stage_bridge "$NAME" || { echo "FAILED stage" > "$DST/status.txt"; return 1; }
  if [ "$GT_SEM" != 1 ]; then   # stage THIS snippet's held-out head (sequential runs -> overwrite is safe)
    local HEAD; HEAD=$(seg_head_for "$NAME") || { echo "FAILED no seg head" > "$DST/status.txt"; return 1; }
    cp -f "$HEAD" "$SNI_REPO/seg/dinov2_crcd.pth"
    echo "$HEAD" > "$DST/seg_head_provenance.txt"; say "$NAME seg head: $HEAD"
  fi
  mk_sni_cfg "$NAME"   || { echo "FAILED cfg"   > "$DST/status.txt"; return 1; }
  # VRAM watcher (peak logging; SNI CRCD was only ever proven on A100)
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 30; done \
    > "$DST/vram_samples.txt" ) & local VPID=$!
  say "$NAME: SNI-SLAM run.py (frames=$(ls "$SD/rgb"/*.png|wc -l))"
  ( cd "$SNI_REPO" && PYTHONPATH= "$SNI_PY" -W ignore run.py "configs/CRCD/bench_${NAME}.yaml" ) 2>&1 | tee "$DST/run.log"
  local RC=${PIPESTATUS[0]}; kill $VPID 2>/dev/null
  sort -rn "$DST/vram_samples.txt" 2>/dev/null | head -1 | xargs -I{} echo "[peak VRAM] {} MiB" | tee -a "$DST/run.log"
  [ "$RC" -eq 0 ] || { echo "FAILED run.py rc=$RC" > "$DST/status.txt"; say "$NAME FAILED (isolated) -> next"; return 1; }
  # ---- export est trajectory from the newest ckpt (estimate_c2w_list -> 16-float c2w rows) ----
  local CKPT; CKPT=$(ls -t "$OUTD"/ckpts/*.tar 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { echo "FAILED no ckpt" > "$DST/status.txt"; return 1; }
  PYTHONPATH= "$SNI_PY" - "$CKPT" "$DST/est_c2w_data.txt" <<'PY' || { echo "FAILED export" > "$DST/status.txt"; return 1; }
import sys, torch
ck = torch.load(sys.argv[1], map_location='cpu')
est = ck['estimate_c2w_list'][:ck['idx'] + 1].numpy()
with open(sys.argv[2], 'w') as f:
    for M in est:
        f.write(' '.join(f'{v:.10f}' for v in M.reshape(-1)[:16]) + '\n')
print(f"[export] {est.shape[0]} poses")
PY
  # ---- post-hoc render every frame (RGB + depth) with the hardened fork renderer ----
  ( cd "$SNI_REPO" && PYTHONPATH= "$SNI_PY" Addons/viz/render_all_frames_sni.py "configs/CRCD/bench_${NAME}.yaml" \
      --skip 1 --ignore_scaled_config ) 2>&1 | tail -5 | tee -a "$DST/run.log" \
    || say "$NAME WARN renderer failed (render metrics will be skipped)"
  # renders -> harness naming (<idx>.jpg / depth/<idx>.png) beside the GT
  python3 - "$OUTD/rendered" "$DST" <<'PY' || say "WARN render rename"
import glob, os, re, shutil, sys
rd, out = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(out, 'render'), exist_ok=True); os.makedirs(os.path.join(out, 'render', 'depth'), exist_ok=True)
n = 0
for p in sorted(glob.glob(os.path.join(rd, '*.jpg'))):
    i = int(re.findall(r'\d+', os.path.basename(p))[-1]); shutil.copy(p, os.path.join(out, 'render', f'{i}.jpg')); n += 1
m = 0
for p in sorted(glob.glob(os.path.join(rd, 'depth', '*.png'))):
    i = int(re.findall(r'\d+', os.path.basename(p))[-1]); shutil.copy(p, os.path.join(out, 'render', 'depth', f'{i:04d}.png')); m += 1
print(f'[rename] {n} rgb + {m} depth renders')
PY
  # ---- THE BATTERY (DDS harness, system python) ----
  local DD="/content/rect_staged/$NAME"
  python3 "$REPO/Addons/eval/sim3_ate.py" --est "$DST/est_c2w_data.txt" --gt "$DD/groundtruth.txt" \
     --name "SNI-SLAM $NAME" --out "$DST/sim3_metrics.txt" || say "$NAME WARN sim3"
  python3 "$REPO/Addons/eval/eval_rendering.py" --gt_dir "$DD/video_frames" --render_dir "$DST/render" \
     --name "SNI-SLAM" --sequence "CRCD ($NAME)" --output_csv "$DST/render_eval.csv" \
     > "$DST/render_eval.txt" 2>&1 || say "$NAME WARN render eval"
  python3 "$REPO/Addons/eval/depth_l1.py" --render_depth_dir "$DST/render/depth" --render_scale 10000 \
     --input_depth_dir "$DD/depth" --input_scale 10000 --sc_factor 1.0 \
     --out "$DST/depth_l1.txt" || say "$NAME WARN depth_l1"
  python3 "$REPO/Addons/viz/generate_video.py" --rgb_input_dir "$DD/video_frames" --rgb_input_pattern '*l.png' \
     --rgb_output_dir "$DST/render" --rgb_output_pattern '[0-9]*.jpg' \
     --depth_input_dir "$DD/depth" --depth_output_dir "$DST/render/depth" --png_depth_scale 10000 \
     --seg_dir "$DD/semantic_class" --seg_classmap \
     --trajectory_est "$DST/est_c2w_data.txt" --trajectory_gt "$DD/groundtruth.txt" \
     --output "$DST/panels.mp4" --fps 15 || say "$NAME WARN video"
  echo "PASS" > "$DST/status.txt"; sync; touch "$DST/.DONE"; sync
  say "$NAME DONE -> $DST"
  grep -h "rmse/mean/median/max\|PSNR\|SSIM\|LPIPS\|Depth-L1" "$DST/sim3_metrics.txt" "$DST/render_eval.txt" "$DST/depth_l1.txt" 2>/dev/null | head -8
}

aggregate(){
  say "=== SNI-SLAM bench summary ($DRIVE_OUT) ==="
  for NAME in $SNIPPETS; do
    echo ""; echo "--- $NAME [$(cat "$DRIVE_OUT/$NAME/status.txt" 2>/dev/null || echo 'not run')] ---"
    grep -h "Sim3 ATE\|path ratio\|Pearson\|scale s\|PSNR\|SSIM\|LPIPS\|Depth-L1" \
      "$DRIVE_OUT/$NAME"/sim3_metrics.txt "$DRIVE_OUT/$NAME"/render_eval.txt "$DRIVE_OUT/$NAME"/depth_l1.txt 2>/dev/null
  done
}

case "$VERB" in
  env)  build_env ;;
  crcd) mkdir -p "$DRIVE_OUT"; exec > >(tee -a "$DRIVE_OUT/runbook.log") 2>&1
        say "=== SNI-SLAM CRCD bench  fork=$(cd "$SNI_REPO" 2>/dev/null && git rev-parse --short HEAD)  snippets=$SNIPPETS gt_sem=$GT_SEM ==="
        build_env || exit 1
        for NAME in $SNIPPETS; do run_one "$NAME" || true; done
        aggregate ;;
  eval) aggregate ;;
  all)  build_env && { for NAME in $SNIPPETS; do run_one "$NAME" || true; done; aggregate; } ;;
  *) echo "usage: run_snislam.sh env|crcd|eval|all"; exit 2 ;;
esac
