#!/bin/bash
# ============================================================================
# run_sni_fresh_20260712.sh — FRESH base-SNI-SLAM CRCD bench (clean reset).
#
# Built on PRISTINE IRMVLab/SNI-SLAM + exactly THREE clean additions from
# Addons/sni_bench/ (dataloader, config, bound helper). NO fork patches, no
# accreted state. ALL input adaptation lives in the CRCD dataloader; the config
# is reviewed vs DDS-SLAM + the other benches (see crcd.yaml header).
#
#   design decisions (user, 2026-07-12):
#     - metric units, mirror DDS (scale 1; planes/trunc scaled to the ~3cm scene)
#     - frame-0-relative poses (DDS identity-init / SGS relative_pose convention)
#     - GT masks (use_gt_semantic=True), n_classes=52 so dinov2_replica.pth loads
#       UNMODIFIED -> zero code patches to base SNI
#     - reuse the proven `sni` conda env-build; C1_001 SMOKE first
#
# Usage (fresh box, Drive mounted):
#   git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM /content/DDS-SLAM
#   nohup bash /content/DDS-SLAM/Addons/colab/run_sni_fresh_20260712.sh \
#       &> /content/sni_fresh.out & disown ; tail -f /content/sni_fresh.out
#
#   SNIPPETS="C1_001 C2_001 ..." to widen after the smoke passes.
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}
SNI_REPO=${SNI_REPO:-/content/sni-fresh}                         # SEPARATE from the fork (/content/sni-slam)
SNI_URL=${SNI_URL:-https://github.com/IRMVLab/SNI-SLAM}          # PRISTINE authors
CONDA_ROOT=${CONDA_ROOT:-/content/miniconda3}
SNI_ENV=${SNI_ENV:-sni}; SNI_PY="$CONDA_ROOT/envs/$SNI_ENV/bin/python"
ENV_CACHE=${ENV_CACHE:-/content/drive/MyDrive/dds_cache/sni_env.tar.gz}
SNI_GDRIVE_ID=${SNI_GDRIVE_ID:-1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h}  # authors' DINOv2 folder (from run_snislam.sh)
# CRCD inputs = the rect_staged v2 tars: the SAME rectified frames + metric stereo-anchored MoGe
# depth + 4-class semantic_class every other benchmarked method consumed (comparability requirement).
STAGE_CACHE_DIR=${STAGE_CACHE_DIR:-/content/drive/MyDrive/dds_cache/rect_staged}; STAGEVER=${STAGEVER:-v2}
SNIPPETS=${SNIPPETS:-"C1_001"}
DRIVE_OUT=${DRIVE_OUT:-/content/drive/MyDrive/Outputs/SNI_fresh_$(date +%Y%m%d)}
STORE_DEV=${SNI_STORE_DEVICE:-cuda:0}
mkdir -p "$DRIVE_OUT"
exec > >(tee -a "$DRIVE_OUT/runbook.log") 2>&1
say(){ echo ""; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }

# ---------------------------------------------------------------- env ----
build_env(){
  [ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; return 1; }
  [ -d "$SNI_REPO/.git" ] || git clone -q "$SNI_URL" "$SNI_REPO" || { say "FATAL clone $SNI_URL"; return 1; }
  # reuse the proven sni conda env (env-only -> still 'base SNI'); restore cache or build once
  # readiness = conda sentinels (torch/pytorch3d) AND pip-stage sentinels (cv2 etc.). A failed
  # `conda env create` dies at the PIP stage, leaving torch importable but cv2/skimage missing --
  # probing torch alone declared that half-built env "ready" (bit us 2026-07-13).
  env_ok(){ PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d,cv2,skimage,trimesh,open3d,wandb,timm;assert torch.__version__.startswith('1.11')" 2>/dev/null; }
  # env-only repair (proven: fork b725dd4). Pristine environment.yaml lists unpinned `timm`,
  # which resolves safetensors>=0.5 -> no py3.7 wheel -> Rust source build -> pip FATAL.
  # timm is needed by the downloaded DINOv2 backbone (facebookresearch/dinov2 pins 0.9.2),
  # so PIN it (don't drop); safetensors 0.3.3 = last series with cp37 wheels.
  grep -q 'timm==0.9.2' "$SNI_REPO/environment.yaml" \
    || sed -i 's/^\( *- \)timm$/\1timm==0.9.2\n\1safetensors==0.3.3/' "$SNI_REPO/environment.yaml"
  if [ "${REBUILD_ENV:-0}" != 1 ] && [ -x "$SNI_PY" ] && env_ok; then
    say "sni env ready"
  else
    [ -x "$CONDA_ROOT/bin/conda" ] || { wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash /tmp/mc.sh -b -p "$CONDA_ROOT"; }
    "$CONDA_ROOT/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
    # openexr==1.3.7 is an sdist -> needs the OpenEXR headers (old runbook did this too; without it
    # the pip stage dies at 'Failed building wheel for openexr')
    sudo apt-get -qq update 2>/dev/null; sudo apt-get -qq install -y libopenexr-dev 2>/dev/null || true
    # restore candidates: our own cache, then the PROVEN fork-bench env (sni_env_bench.tar.gz,
    # built+cached by run_snislam.sh on 2026-07-11 -- env-only, no method code, so reuse is clean)
    for EC in "$ENV_CACHE" /content/drive/MyDrive/dds_cache/sni_env_bench.tar.gz; do
      [ -f "$EC" ] || continue
      say "restoring sni env from Drive cache: $EC ($(du -h "$EC" 2>/dev/null | cut -f1))"
      rm -rf "$CONDA_ROOT/envs/$SNI_ENV"; mkdir -p "$CONDA_ROOT/envs/$SNI_ENV"
      tar -xzf "$EC" -C "$CONDA_ROOT/envs/$SNI_ENV" && env_ok && { say "cache OK"; break; }
      say "cache restore failed/incomplete -> next candidate"; rm -rf "$CONDA_ROOT/envs/$SNI_ENV"
    done
    # FAST PATH: conda stage intact (torch+pytorch3d import) but pip stage missing/partial ->
    # just complete the pip deps from the (pinned) yaml into the existing env (~5 min, not ~35).
    if ! env_ok && PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d" 2>/dev/null; then
      say "pip-completing half-built env (conda stage OK, pip stage missing)"
      python3 - "$SNI_REPO/environment.yaml" /tmp/sni_req.txt <<'PY' \
        && PYTHONPATH= "$SNI_PY" -m pip install -q -r /tmp/sni_req.txt || say "WARN pip-complete failed -> full rebuild"
import sys, yaml
env = yaml.safe_load(open(sys.argv[1]))
pips = next(d['pip'] for d in env['dependencies'] if isinstance(d, dict) and 'pip' in d)
open(sys.argv[2], 'w').write('\n'.join(pips) + '\n')
print(f"[pipfix] {len(pips)} pip deps -> {sys.argv[2]}")
PY
    fi
    if ! env_ok; then
      rm -rf "$CONDA_ROOT/envs/$SNI_ENV"   # clear any partial env (create errors on existing prefix)
      say "conda env create (~25-40 min; pytorch3d 0.7.1/cu113/py37 landmine; timm/safetensors pinned)"
      "$CONDA_ROOT/bin/conda" env create -n "$SNI_ENV" -f "$SNI_REPO/environment.yaml" || { say "FATAL env create"; return 1; }
    fi
    env_ok || { say "FATAL: env still incomplete after build (check pip output above)"; return 1; }
    [ -f "$ENV_CACHE" ] || { say "caching env to Drive (one-time; later runs restore in ~2 min)"
      tar -czf /tmp/sni_env.tar.gz -C "$CONDA_ROOT/envs/$SNI_ENV" . && mv -f /tmp/sni_env.tar.gz "$ENV_CACHE" || true; }
  fi
  PYTHONPATH= "$SNI_PY" -c "import torch,pytorch3d;print('[env] torch',torch.__version__,'pytorch3d',pytorch3d.__version__,'GPU',torch.cuda.get_device_name(0))" || { say "FATAL env smoke"; return 1; }
  # seg assets: DINOv2 backbone CODE (always) + Replica head weights (use_gt_semantic loads dinov2_replica.pth)
  if [ ! -d "$SNI_REPO/seg/facebookresearch_dinov2_main" ] || [ ! -f "$SNI_REPO/seg/dinov2_replica.pth" ]; then
    say "fetching DINOv2 backbone + Replica head"
    pip install -q gdown 2>/dev/null
    gdown --folder "https://drive.google.com/drive/folders/$SNI_GDRIVE_ID" -O /tmp/snidl --remaining-ok 2>/dev/null || true
    Z=$(find /tmp/snidl -name 'facebookresearch_dinov2_main.zip' | head -1); [ -n "$Z" ] && unzip -qo "$Z" -d "$SNI_REPO/seg/"
    for f in dinov2_replica.pth; do S=$(find /tmp/snidl -name "$f" | head -1); [ -n "$S" ] && cp -f "$S" "$SNI_REPO/seg/$f"; done
    [ -d "$SNI_REPO/seg/facebookresearch_dinov2_main" ] && [ -f "$SNI_REPO/seg/dinov2_replica.pth" ] \
      || { say "FATAL: DINOv2 backbone/head not obtained (gdown quota? place manually in $SNI_REPO/seg/)"; return 1; }
  fi
  python3 -c "import lpips" 2>/dev/null || pip install -q lpips
  python3 -c "import cv2,matplotlib,imageio" 2>/dev/null || pip install -q opencv-contrib-python matplotlib imageio imageio-ffmpeg
  say "env DONE"
}

# ------------------------------------------- inject the 3 clean additions ----
inject_crcd(){
  mkdir -p "$SNI_REPO/configs/CRCD"
  cp -f "$REPO/Addons/sni_bench/crcd.yaml" "$SNI_REPO/configs/CRCD/crcd.yaml"
  # append the CRCD dataloader into base datasets.py (BaseDataset in scope) + register 'crcd'. Idempotent.
  REPO="$REPO" SNI_REPO="$SNI_REPO" PYTHONPATH= "$SNI_PY" - <<'PY' || { say "FATAL inject"; return 1; }
import os
repo, sni = os.environ['REPO'], os.environ['SNI_REPO']
dsf = f"{sni}/src/utils/datasets.py"
txt = open(dsf).read()
if "class CRCD(BaseDataset)" in txt:
    print("[inject] CRCD already present -> skip"); raise SystemExit(0)
src = open(f"{repo}/Addons/sni_bench/crcd_dataset.py").read()
body = src[src.index("def _fid"):]                     # _fid + CRCD class (drop standalone import header)
txt += ("\n\n# ==== fresh SNI-CRCD bench (Addons/sni_bench/crcd_dataset.py) ====\n"
        "import glob, os, re\nimport cv2\nimport numpy as np\nimport torch\n"
        "from scipy.spatial.transform import Rotation\n" + body +
        "\ndataset_dict['crcd'] = CRCD\n")
open(dsf, 'w').write(txt)
print("[inject] CRCD dataloader appended + registered")
PY
  say "inject DONE"
}

# ---------------------------------- stage rect_bench v2 tar (per snippet) ----
# Restores the cached rectified staging (rectified left frames + METRIC stereo-anchored MoGe depth
# NNNNNN.png @1e4 + semantic_class/ RAW 4-class {0,1,2,3} + groundtruth + rectified_calib) and links
# it into the pristine repo's data dir. NOTE: the staging's masks/ is a BINARY tool mask -- the
# dataloader reads semantic_class/ for semantics (and hard-asserts raw ids <= 3).
stage_rect(){ local NAME=$1 DD="/content/rect_staged/$NAME" SD="$SNI_REPO/data/CRCD/$NAME"
  if [ ! -f "$DD/.STAGED" ]; then
    local TGZ="$STAGE_CACHE_DIR/${NAME}_$STAGEVER.tar"
    [ -f "$TGZ" ] || { say "FATAL: no stage cache $TGZ (the rectified bench staging every other method used; run the rect_bench staging for $NAME first)"; return 1; }
    mkdir -p /content/rect_staged && tar -xf "$TGZ" -C /content/rect_staged || { say "FATAL untar"; return 1; }
    [ -f "$DD/.STAGED" ] || { say "FATAL: restored tar lacks .STAGED"; return 1; }
  fi
  mkdir -p "$SD"
  for sub in video_frames depth semantic_class masks; do
    [ -d "$DD/$sub" ] && ln -sfn "$DD/$sub" "$SD/$sub"
  done
  for f in groundtruth.txt rectified_calib.txt; do
    [ -f "$DD/$f" ] || { say "FATAL: $DD missing $f"; return 1; }
    ln -sf "$DD/$f" "$SD/$f"
  done
  local NC ND NS
  NC=$(ls "$SD/video_frames"/*l.png 2>/dev/null | wc -l)
  ND=$(ls "$SD/depth"/[0-9]*.png 2>/dev/null | wc -l)
  NS=$(ls "$SD/semantic_class"/*.png 2>/dev/null | wc -l)
  say "  $NAME staged (rect $STAGEVER): rgb=$NC depth=$ND semantic_class=$NS"
  [ "$NC" -gt 0 ] && [ "$ND" -gt 0 ] && [ "$NS" -gt 0 ] || { say "FATAL: empty modality"; return 1; }
}

# ---------------------------------------------------------- run one snippet --
# run_one NAME [VARIANT] -- VARIANT defaults to 'faithful' (the benchmark cell).
# Diagnostic variants (NEVER benchmark rows; they exist to attribute a faithful failure):
#   oracle   : func.use_gt_pose=True + mapping.joint_opt=False -> isolates metric-scale
#              mapping/render quality from tracking entirely (sim3 ATE ~0 = pipeline sanity)
#   depthpin : tracking.w_depth 1->5  -> tests the color-dominant-loss runaway hypothesis
#   noconst  : tracking.const_speed_assumption=False -> tests the const-vel compounding hypothesis
run_one(){ local NAME=$1 VARIANT=${2:-faithful} TAG="${1}_${2:-faithful}"
  local SD="$SNI_REPO/data/CRCD/$NAME" CFG="configs/CRCD/${TAG}.yaml" DST="$DRIVE_OUT/$TAG"
  mkdir -p "$DST"
  [ -f "$DST/.DONE" ] && [ "${FORCE:-0}" != 1 ] && { say "$TAG done -> skip"; return 0; }
  stage_rect "$NAME" || { echo FAILED > "$DST/status.txt"; return 1; }
  # per-snippet config (relative-frame bound) inheriting crcd.yaml
  python3 "$REPO/Addons/sni_bench/compute_bound_relative.py" --data_dir "$SD" \
     --out "$SNI_REPO/$CFG" --name "$TAG" --input_folder "data/CRCD/$NAME" \
     || { say "FATAL bound"; echo FAILED > "$DST/status.txt"; return 1; }
  # variant overrides: DEEP-MERGE into the generated yaml (an appended duplicate top-level
  # 'mapping:' block would silently CLOBBER the bound under PyYAML last-wins)
  python3 - "$SNI_REPO/$CFG" "$VARIANT" <<'PY' || { say "FATAL variant merge"; echo FAILED > "$DST/status.txt"; return 1; }
import sys, yaml
p, v = sys.argv[1], sys.argv[2]
OV = {
  'faithful': {},
  'oracle':   {'func': {'use_gt_pose': True}, 'mapping': {'joint_opt': False}},
  'depthpin': {'tracking': {'w_depth': 5}},
  'noconst':  {'tracking': {'const_speed_assumption': False}},
}[v]
cfg = yaml.safe_load(open(p)) or {}
def merge(dst, src):
    for k, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(k), dict): merge(dst[k], val)
        else: dst[k] = val
merge(cfg, OV)
yaml.safe_dump(cfg, open(p, 'w'), default_flow_style=None, sort_keys=False)
print(f"[variant] {v}: {OV if OV else 'faithful (no overrides)'}")
PY
  cp -f "$SNI_REPO/$CFG" "$DST/config_used.yaml"
  # ---- train (VRAM watcher for the T4-capacity question; wall time logged) ----
  say "$TAG: SNI-SLAM run.py"
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null; sleep 30; done \
    > "$DST/vram_samples.txt" ) & local VPID=$!
  local T0=$SECONDS
  ( cd "$SNI_REPO" && SNI_STORE_DEVICE="$STORE_DEV" PYTHONPATH=. "$SNI_PY" run.py "$CFG" ) 2>&1 | tee "$DST/run.log"
  local RC=${PIPESTATUS[0]}
  kill $VPID 2>/dev/null
  echo "[wall] train $(( (SECONDS-T0)/60 )) min | peak VRAM $(sort -rn "$DST/vram_samples.txt" 2>/dev/null | head -1) MiB" | tee -a "$DST/run.log"
  [ "$RC" -eq 0 ] || { echo "FAILED run.py rc=$RC" > "$DST/status.txt"; say "$TAG FAILED (isolated) -> next"; return 1; }
  local OUTD="$SNI_REPO/output/CRCD/$TAG"
  # ---- export est trajectory ----
  local CKPT; CKPT=$(ls -t "$OUTD"/ckpt/*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { say "no ckpt"; echo FAILED > "$DST/status.txt"; return 1; }
  PYTHONPATH= "$SNI_PY" - "$CKPT" "$DST/est_c2w_data.txt" <<'PY' || { say "export fail"; return 1; }
import sys, torch, numpy as np
ck = torch.load(sys.argv[1], map_location='cpu')
est = ck['estimate_c2w_list'][:ck['idx']+1].numpy()
with open(sys.argv[2],'w') as f:
    for M in est: f.write(" ".join(f"{v:.10f}" for v in M.reshape(-1))+"\n")
print(f"[export] {est.shape[0]} poses")
PY
  # ---- per-frame render (drives SNI's own Renderer) ----
  ( cd "$SNI_REPO" && PYTHONPATH=. "$SNI_PY" "$REPO/Addons/sni_bench/render_all_frames_sni.py" "$CFG" --skip 1 --ignore_scaled_config ) 2>&1 | tail -5 | tee -a "$DST/run.log" || say "$TAG WARN render"
  python3 - "$OUTD/rendered" "$DST" <<'PY' || say "WARN rename"
import sys, glob, os, re, shutil
rd, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/render/depth", exist_ok=True); n=m=0
for p in sorted(glob.glob(f"{rd}/*.jpg")):
    i=int(re.findall(r'\d+', os.path.basename(p))[-1]); shutil.copy(p, f"{out}/render/{i}.jpg"); n+=1
for p in sorted(glob.glob(f"{rd}/depth/*.png")):
    i=int(re.findall(r'\d+', os.path.basename(p))[-1]); shutil.copy(p, f"{out}/render/depth/{i:04d}.png"); m+=1
print(f"[rename] {n} rgb + {m} depth")
PY
  # ---- eval battery (OUR dataset-agnostic metric tools) ----
  python3 "$REPO/Addons/eval/sim3_ate.py" --est "$DST/est_c2w_data.txt" --gt "$SD/groundtruth.txt" \
     --name "SNI-fresh $TAG" --out "$DST/sim3_metrics.txt" || say "WARN sim3"
  python3 "$REPO/Addons/eval/eval_rendering.py" --gt_dir "$SD/video_frames" --render_dir "$DST/render" \
     --name "SNI-fresh" --sequence "CRCD ($TAG)" --output_csv "$DST/render_eval.csv" > "$DST/render_eval.txt" 2>&1 || say "WARN render eval"
  python3 "$REPO/Addons/eval/depth_l1.py" --render_depth_dir "$DST/render/depth" --render_scale 10000 \
     --gt_depth_dir "$SD/depth" --png_depth_scale 10000 --out "$DST/depth_l1.txt" || say "WARN depth_l1"
  python3 "$REPO/Addons/viz/generate_video.py" --rgb_input_dir "$SD/video_frames" --rgb_input_pattern '*l.png' \
     --rgb_output_dir "$DST/render" --rgb_output_pattern '[0-9]*.jpg' \
     --depth_input_dir "$SD/depth" --depth_output_dir "$DST/render/depth" --png_depth_scale 10000 \
     --trajectory_est "$DST/est_c2w_data.txt" --trajectory_gt "$SD/groundtruth.txt" \
     --output "$DST/panels.mp4" --fps 15 || say "WARN video"
  # ---- SMOKE DIAGNOSTICS (does the fresh setup actually work?) ----
  say "$TAG SMOKE DIAGNOSTICS"
  echo "  marching_cubes errors: $(grep -ic 'marching_cubes error' "$DST/run.log")   (want 0)"
  python3 - "$DST/est_c2w_data.txt" "$DST/render" <<'PY'
import sys, glob, os, numpy as np, cv2
M=np.loadtxt(sys.argv[1]); t=M[:,[3,7,11]]
print(f"  est translation std x/y/z: {t.std(0).round(5)}   (frozen if ~0, runaway if >>scene)")
fs=sorted(glob.glob(sys.argv[2]+"/*.jpg"), key=lambda p:int(''.join(filter(str.isdigit,os.path.basename(p))) or 0))
if fs:
    mid=cv2.imread(fs[len(fs)//2]); print(f"  mid render spatial_std: {mid.std():.2f}   (empty if <10)")
PY
  echo DONE > "$DST/status.txt"; touch "$DST/.DONE"
  say "$TAG DONE -> $DST"
  grep -E "recovered scale|path ratio|Pearson|ATE  rmse" "$DST/sim3_metrics.txt" 2>/dev/null
  grep -iE "PSNR|SSIM|LPIPS" "$DST/render_eval.txt" 2>/dev/null | head -3
}

# ---------------------------------------------------------- overnight summary
overnight_summary(){
  say "OVERNIGHT SUMMARY ($DRIVE_OUT)"
  for D in "$DRIVE_OUT"/*/; do
    local TAG; TAG=$(basename "$D")
    echo ""; echo "--- $TAG [$(cat "$D/status.txt" 2>/dev/null || echo NO-STATUS)] ---"
    grep -E "\[wall\]" "$D/run.log" 2>/dev/null | tail -1
    echo "  marching_cubes errors: $(grep -ic 'marching_cubes error' "$D/run.log" 2>/dev/null)"
    [ -f "$D/est_c2w_data.txt" ] && python3 - "$D/est_c2w_data.txt" <<'PY'
import sys, numpy as np
M=np.loadtxt(sys.argv[1]); t=M[:,[3,7,11]]
print(f"  est t-std x/y/z: {t.std(0).round(5)}  span: {(t.max(0)-t.min(0)).round(4)}")
PY
    grep -E "recovered scale|path ratio|ATE  rmse|Pearson\| dom" "$D/sim3_metrics.txt" 2>/dev/null | sed 's/^/  /'
    grep -iE "^PSNR|^SSIM|^LPIPS" "$D/render_eval.txt" 2>/dev/null | sed 's/^/  /'
    grep -E "Depth-L1" "$D/depth_l1.txt" 2>/dev/null | sed 's/^/  /'
  done
  echo ""
  echo "READ: faithful = the benchmark candidate. oracle good + faithful bad => tracker is the"
  echo "problem (scale regime OK). oracle bad => metric-scale hyperparams wrong. depthpin/noconst"
  echo "= which mechanism drives a faithful runaway (color-dominant loss vs const-vel compounding)."
}

# ------------------------------------------------------------------- main ----
say "FRESH SNI-CRCD bench  snippets='$SNIPPETS'  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1)"
case "${1:-all}" in
  env)  build_env ;;
  overnight)  # C1 4-cell diagnostic overnight: benchmark candidate + the three attribution arms
    build_env && inject_crcd && { for V in faithful oracle depthpin noconst; do
      run_one "${2:-C1_001}" "$V" || true; done; }
    overnight_summary ;;
  all)  build_env && inject_crcd && { for NAME in $SNIPPETS; do run_one "$NAME" faithful || true; done; } ;;
  *)    build_env && inject_crcd && run_one "$1" "${2:-faithful}" ;;
esac
say "fresh bench complete -> $DRIVE_OUT"
