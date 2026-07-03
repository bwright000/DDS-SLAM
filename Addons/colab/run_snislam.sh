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
SEG_PTH=${SEG_PTH:-/content/drive/MyDrive/Outputs/seg/dinov2_crcd.pth}
SNI_GDRIVE_ID=1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h   # authors' folder (dinov2 backbone zip fallback)
DRIVE_OUT=${DRIVE_OUT:-/content/drive/MyDrive/Outputs/SNI-SLAM_bench_$DATE}
GT_SEM=${GT_SEM:-0}
SNIPPETS=${SNIPPETS:-"E3_005 C1_001 C2_001 C3_001 G3_001"}   # shortest-first (E3 = T4 smoke)
VERB=${1:-}

say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }

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
  if [ ! -d "$SNI_REPO/seg/facebookresearch_dinov2_main" ]; then
    say "fetching DINOv2 backbone code (authors' Drive folder)"
    pip install -q gdown 2>/dev/null
    gdown --folder "https://drive.google.com/drive/folders/$SNI_GDRIVE_ID" -O /tmp/snidl --remaining-ok 2>/dev/null || true
    Z=$(find /tmp/snidl -name 'facebookresearch_dinov2_main.zip' | head -1)
    [ -n "$Z" ] && unzip -qo "$Z" -d "$SNI_REPO/seg/" || { say "FATAL: dinov2 backbone zip not obtained (gdown quota? fetch manually to $SNI_REPO/seg/)"; return 1; }
  fi
  if [ "$GT_SEM" != 1 ]; then
    [ -f "$SEG_PTH" ] || { say "FATAL: in-domain seg head missing at $SEG_PTH (train_seg_head output). GT_SEM=1 to run the GT-mask ablation instead."; return 1; }
    cp -f "$SEG_PTH" "$SNI_REPO/seg/dinov2_crcd.pth"
    say "seg head: dinov2_crcd.pth (held-out on the benchmark 5) staged"
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
rgb  = sorted(glob.glob(f"{DD}/video_frames/*l.png"))
dep  = sorted(glob.glob(f"{DD}/depth/[0-9]*.png"))
sem  = sorted(glob.glob(f"{DD}/semantic_class/*.png"))
gt   = [l.split() for l in open(f"{DD}/groundtruth.txt") if l.strip() and not l.startswith('#')]
n = min(len(rgb), len(dep), len(sem), len(gt))
assert n > 0, f"empty staging: rgb={len(rgb)} dep={len(dep)} sem={len(sem)} gt={len(gt)}"
if not (len(rgb) == len(dep) == len(sem) == len(gt)):
    print(f"[bridge] WARN count mismatch rgb={len(rgb)} dep={len(dep)} sem={len(sem)} gt={len(gt)} -> truncating to {n}")
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
import glob, sys, yaml, cv2
NAME, DD, SD, CFG, GT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] == "1"
b = yaml.safe_load(open(f"{DD}/bound.yaml"))["mapping"]          # SAME bounds as the DDS arm
img = cv2.imread(sorted(glob.glob(f"{SD}/rgb/rgb_*.png"))[0]); H, W = img.shape[:2]
intr = {}
for ln in open(f"{DD}/rectified_calib.txt"):
    p = ln.split()
    if len(p) >= 2 and p[0] in ("fx", "fy", "cx", "cy"):
        intr[p[0]] = float(p[1])
assert all(k in intr for k in ("fx", "fy", "cx", "cy")), intr
cfg = {
  "inherit_from": "configs/CRCD/crcd_sni_base.yaml",
  "mapping": {"bound": b["bound"], "marching_cubes_bound": b["marching_cubes_bound"]},
  "data": {"input_folder": f"data/CRCD/{NAME}/", "output": f"output/CRCD/bench_{NAME}"},
  "cam": {"H": H, "W": W, "fx": intr["fx"], "fy": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
          "png_depth_scale": 10000, "crop_edge": 0},
  # depth is already METRIC (every-120 stereo anchor, sc=1) -> truncation at surgical scale,
  # NOT the 0.06 Replica anchor (fork note: set 0.01 manually when bypassing Phase 3.7).
  "model": {"truncation": 0.01,
            "cnn": {"n_classes": 4,
                    "pretrained_model_path": ("seg/dinov2_replica.pth" if GT else "seg/dinov2_crcd.pth")}},
  "func": {"use_gt_semantic": bool(GT), "use_gt_pose": False},
}
yaml.safe_dump(cfg, open(CFG, "w"), sort_keys=False)
print(f"[cfg] {NAME}: HxW={H}x{W} fx={intr['fx']:.1f} bound={b['bound']} gt_sem={GT} -> {CFG}")
PY
}

# ------------------------------------------------------------- run one ----
run_one(){ local NAME=$1 SD="$SNI_REPO/data/CRCD/$NAME" OUTD="$SNI_REPO/output/CRCD/bench_${NAME}" DST="$DRIVE_OUT/$NAME"
  mkdir -p "$DST"
  [ -f "$DST/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { say "$NAME done -> skip"; return 0; }
  stage_bridge "$NAME" || { echo "FAILED stage" > "$DST/status.txt"; return 1; }
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
