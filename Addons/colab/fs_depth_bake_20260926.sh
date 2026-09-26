#!/usr/bin/env bash
# FoundationStereo (FS) depth for the CRCD bench-5 + inspection vs the current MoGe-2+stereo120 depth.
# Plan: benchmarking/FS_BENCH_PLAN_20260926.md (D1). Colab T4 (smoke tier) via the VS Code tunnel.
#
#   cd /content/DDS-SLAM && git pull
#   bash Addons/colab/fs_depth_bake_20260926.sh env       # clone FS @pin, deps, weights (Drive-cached)
#   bash Addons/colab/fs_depth_bake_20260926.sh smoke     # C1_001 first 5 frames, hiera 0 vs 1: VRAM + s/frame
#   bash Addons/colab/fs_depth_bake_20260926.sh bake      # all 5 snippets -> Drive CRCD-Published-FS
#   bash Addons/colab/fs_depth_bake_20260926.sh compare   # vs MoGe stage cache -> Drive Outputs/fs_depth_inspect_<TAG>
#   bash Addons/colab/fs_depth_bake_20260926.sh all       # env + bake + compare
# knobs: SNIPPETS="C1_001"  HIERA=0|1  VALID_ITERS=32  TAG=<yyyymmdd>
#
# Per snippet: raw CRCD-Published (Drive) copied LOCAL -> preprocess_crcd_published (rectify L+R, semantic_class,
# rectified_calib) -> generate_depth_foundationstereo.py -> depth_fs/ (uint16 x1e4 metres, rectified-left, 1280x720,
# 0 = invalid) -> tar to Drive. compare = extract depth/*.png from the rect_staged v2 tar (the MoGe-2+stereo120 depth
# the bench trained on) and run compare_depth_fs_moge.py. Every step idempotent (.DONE markers), fail-loud.
set -uo pipefail
PHASE="${1:-all}"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
SNIPPETS="${SNIPPETS:-C1_001 C2_001 E3_005 C3_001 G3_001}"
HIERA="${HIERA:-0}"; VALID_ITERS="${VALID_ITERS:-32}"; TAG="${TAG:-$(date +%Y%m%d)}"
FS_PIN=6e88068                                   # NVlabs/FoundationStereo HEAD 2025-12-18
FS_ROOT=/content/FoundationStereo
FS_W_URL=https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf   # README "23-51-11" (ViT-L)
FS_W_CACHE=/content/drive/MyDrive/dds_cache/foundation_stereo/23-51-11
CKPT=$FS_ROOT/pretrained_models/23-51-11/model_best_bp2.pth
DPUB=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB=$DPUB/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
DFS=/content/drive/MyDrive/Datasets/CRCD-Published-FS                       # FS corpus (output)
MOGE_CACHE=/content/drive/MyDrive/dds_cache/rect_staged                     # <NAME>_v2.tar (MoGe-2+stereo120)
STAGE=/content/fs_stage; INSPECT=/content/fs_inspect
OUT=/content/drive/MyDrive/Outputs/fs_depth_inspect_${TAG}
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
die(){ say "FATAL: $*"; exit 1; }
[ -d /content/drive/MyDrive ] || die "Drive not mounted"
mkdir -p "$OUT" "$STAGE" "$INSPECT"
exec > >(tee -a "$OUT/run_${PHASE}.log") 2>&1
say "=== FS depth [$PHASE]  REPO HEAD=$(git rev-parse --short HEAD)  snippets=$SNIPPETS  hiera=$HIERA iters=$VALID_ITERS ==="

snip_src(){ local n=$1 ep=${1%_*} sn=${1#*_}; echo "$DPUB/${ep:0:1}_${ep:1}/snippet_${sn}"; }
snip_dfs(){ local n=$1 ep=${1%_*} sn=${1#*_}; echo "$DFS/${ep:0:1}_${ep:1}/snippet_${sn}"; }

do_env(){
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "no GPU"
  if [ ! -d "$FS_ROOT/.git" ]; then git clone -q https://github.com/NVlabs/FoundationStereo.git "$FS_ROOT" || die "clone"; fi
  git -C "$FS_ROOT" checkout -q "$FS_PIN" || die "checkout $FS_PIN"
  # FS pins torch 2.4.1; we use Colab's system torch 2.x (core needs no flash-attn; xformers is an optional try-import).
  pip install -q omegaconf timm einops open3d trimesh gdown imageio scikit-image || die "pip"
  if [ ! -f "$FS_W_CACHE/model_best_bp2.pth" ] || [ ! -f "$FS_W_CACHE/cfg.yaml" ]; then
    say "weights not cached -> gdown $FS_W_URL"
    rm -rf /tmp/fsw && gdown -q --folder "$FS_W_URL" -O /tmp/fsw \
      || die "gdown failed (Drive quota?). Manually copy the README '23-51-11' folder to $FS_W_CACHE and re-run env"
    src=$(dirname "$(find /tmp/fsw -path '*23-51-11*' -name model_best_bp2.pth | head -1)")
    [ -f "$src/model_best_bp2.pth" ] && [ -f "$src/cfg.yaml" ] || die "23-51-11 not in download"
    mkdir -p "$FS_W_CACHE" && cp "$src"/* "$FS_W_CACHE/" || die "cache weights"
  fi
  mkdir -p "$FS_ROOT/pretrained_models" && ln -sfn "$FS_W_CACHE" "$FS_ROOT/pretrained_models/23-51-11"
  python3 - "$FS_ROOT" "$CKPT" <<'PY' || die "FS model build/load"
import sys, torch; sys.path.insert(0, "Addons/depth")
from generate_depth_foundationstereo import load_model
m, a = load_model(sys.argv[1], sys.argv[2], 32, 0)
print(f"FS OK: vit={a.get('vit_size')} max_disp={a.get('max_disp')} mixed_precision={a.get('mixed_precision')} "
      f"params={sum(p.numel() for p in m.parameters())/1e6:.1f}M torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}")
PY
  touch /content/.fs_env_ok
}

stage(){ local NAME=$1 SRC DD L; SRC=$(snip_src "$NAME"); DD="$STAGE/$NAME"; L="$STAGE/_raw/$NAME"
  [ -f "$DD/.STAGED_FS" ] && { say "  $NAME already staged ($(ls "$DD/video_frames"/*l.png | wc -l) pairs)"; return 0; }
  [ -d "$SRC/rgb" ] && [ -d "$SRC/rgbright" ] || { say "  FATAL: raw rgb/rgbright missing at $SRC"; return 1; }
  say "  $NAME copy raw local (Drive FUSE drops on long reads)"
  mkdir -p "$L" && cp -rn "$SRC/rgb" "$SRC/rgbright" "$SRC/semantic_instance" "$SRC/intrinsics.yaml" "$SRC/groundtruth.txt" "$L/" \
    || { say "  FATAL: raw copy"; return 1; }
  local nl nr; nl=$(ls "$L/rgb"/*.png | wc -l); nr=$(ls "$L/rgbright"/*.png | wc -l)
  [ "$nl" -gt 0 ] && [ "$nl" -eq "$nr" ] || { say "  FATAL: raw left $nl != right $nr (truncated copy? rm -rf $L)"; return 1; }
  python3 Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$L" --calib_pkl "$CALIB" --output_dir "$DD" \
    || { say "  FATAL: rectify"; return 1; }
  local vl vr; vl=$(ls "$DD/video_frames"/*l.png | wc -l); vr=$(ls "$DD/video_frames"/*r.png | wc -l)
  [ "$vl" -eq "$nl" ] && [ "$vr" -eq "$nl" ] || { say "  FATAL: rectified L $vl R $vr != raw $nl"; return 1; }
  touch "$DD/.STAGED_FS"; say "  $NAME staged: $vl rectified pairs"
}

bake_one(){ local NAME=$1; shift; local DD="$STAGE/$NAME" D; D=$(snip_dfs "$NAME")
  if [ -f "$D/.DONE" ] && [ -z "${MAX_FRAMES:-}" ]; then say "  $NAME FS depth already on Drive -> skip"; return 0; fi
  stage "$NAME" || return 1
  python3 Addons/depth/generate_depth_foundationstereo.py --staged "$DD" --calib_pkl "$CALIB" --fs_root "$FS_ROOT" \
    --ckpt "$CKPT" --valid_iters "$VALID_ITERS" --hiera "$HIERA" "$@" || { say "  FATAL: FS inference $NAME"; return 1; }
  [ -n "${MAX_FRAMES:-}" ] && return 0                                  # smoke: never ship partial corpora
  [ -f "$DD/depth_fs/.DONE" ] || { say "  FATAL: $NAME depth_fs incomplete"; return 1; }
  mkdir -p "$D" && tar -cf "/tmp/${NAME}_depth_fs.tar" -C "$DD" depth_fs && mv -f "/tmp/${NAME}_depth_fs.tar" "$D/depth_fs.tar" \
    && cp -f "$DD/depth_fs/depth_stats.json" "$DD/rectified_calib.txt" "$D/" && echo "$(git rev-parse --short HEAD) hiera=$HIERA iters=$VALID_ITERS" > "$D/.DONE" \
    || { say "  FATAL: upload $NAME"; return 1; }
  say "  $NAME -> $D ($(du -h "$D/depth_fs.tar" | cut -f1))"
}

do_smoke(){
  [ -f /content/.fs_env_ok ] || do_env
  for h in 0 1; do
    say "--- smoke C1_001 5 frames hiera=$h ---"
    HIERA=$h MAX_FRAMES=5 bake_one C1_001 --max_frames 5 --out_subdir "depth_fs_smoke_h$h" || die "smoke hiera=$h (OOM? see above)"
    python3 -c "import json;s=json.load(open('$STAGE/C1_001/depth_fs_smoke_h$h/depth_stats.json'));print('hiera=$h', {k:s[k] for k in ['sec_per_frame','peak_vram_gib','valid_pct']}, s['depth_m_percentiles'])"
  done
  mkdir -p "$OUT/smoke" && cp -r "$STAGE/C1_001"/depth_fs_smoke_h* "$OUT/smoke/"
  say "smoke done -> $OUT/smoke  (projected bake time = sec_per_frame x 4869 frames)"
}

do_bake(){
  [ -f /content/.fs_env_ok ] || do_env
  local fail=0; for n in $SNIPPETS; do say "=== bake $n ==="; bake_one "$n" || fail=1; done
  [ $fail -eq 0 ] || die "one or more snippets failed (re-run: completed ones are skipped)"
}

compare_one(){ local NAME=$1 DD="$STAGE/$NAME" D T MR="$INSPECT/_moge/$NAME"; D=$(snip_dfs "$NAME"); T="$MOGE_CACHE/${NAME}_v2.tar"
  [ -f "$D/.DONE" ] || { say "  $NAME: no FS corpus on Drive (run bake)"; return 1; }
  [ -f "$T" ] || { say "  WARN $NAME: MoGe reference $T missing -> compare skipped"; return 0; }
  stage "$NAME" || return 1
  [ -f "$DD/depth_fs/.DONE" ] || tar -xf "$D/depth_fs.tar" -C "$DD" || { say "  FATAL: restore FS tar"; return 1; }
  if [ ! -d "$MR/depth" ]; then
    mkdir -p "$MR" && tar -xf "$T" -C "$MR" --strip-components=1 --wildcards "$NAME/depth/[0-9]*.png" \
      || { say "  FATAL: extract MoGe depth from $T"; return 1; }
  fi
  python3 Addons/depth/compare_depth_fs_moge.py --staged "$DD" --fs_dir "$DD/depth_fs" --moge_dir "$MR/depth" \
    --out "$INSPECT/$NAME" || { say "  FATAL: compare $NAME"; return 1; }
  mkdir -p "$OUT/$NAME" && cp -f "$INSPECT/$NAME"/* "$OUT/$NAME/" && cp -f "$DD/depth_fs/depth_stats.json" "$OUT/$NAME/fs_depth_stats.json"
}

do_compare(){
  for n in $SNIPPETS; do say "=== compare $n ==="; compare_one "$n"; done
  python3 - "$OUT" $SNIPPETS <<'PY'
import json, os, sys
out, names = sys.argv[1], sys.argv[2:]
cls = {"0": "bg", "1": "Liver", "2": "GB", "3": "Tool"}
L = ["| snippet | valid FS/MoGe % | median depth FS / MoGe (m) | p1 FS (m) | ratio FS/MoGe (cv) | AbsRel raw / aligned | "
     "stereo photo L1 FS / MoGe / plane | jitter FS / MoGe | Tool ratio / AbsRel-al |", "|" + "---|" * 9]
for n in names:
    p = os.path.join(out, n, "summary.json")
    if not os.path.exists(p):
        L.append(f"| {n} | (missing) |" + " |" * 7); continue
    s = json.load(open(p)); r = s["scale_ratio_fs_over_moge"]; ph = s["stereo_photometric_L1"]; pc = s["depth_m_percentiles"]
    t = s["per_class"].get("3", {})
    L.append(f"| {n} | {s['valid_pct']['fs']:.1f} / {s['valid_pct']['moge']:.1f} | {pc['fs']['p50']:.4f} / {pc['moge']['p50']:.4f} | "
             f"{s['unit_gate']['fs_p1_m']:.4f} | {r['median']:.3f} ({r['cv']:.3f}) | "
             f"{s['agreement_raw']['absrel']:.3f} / {s['agreement_scale_aligned']['absrel']:.3f} | "
             f"{ph['fs']:.2f} / {ph['moge']:.2f} / {ph['plane_floor']:.2f} | "
             f"{s['temporal_median_jitter']['fs']:.4f} / {s['temporal_median_jitter']['moge']:.4f} | "
             f"{t.get('ratio_median', float('nan')):.3f} / {t.get('absrel_aligned', float('nan')):.3f} |")
txt = "\n".join(L); open(os.path.join(out, "SUMMARY.md"), "w").write(txt + "\n"); print(txt)
PY
  say "inspection -> $OUT  (per snippet: montage.png timeseries.png hist.png compare.mp4 summary.json per_frame.csv)"
}

case "$PHASE" in
  env) do_env ;;
  smoke) do_smoke ;;
  bake) do_bake ;;
  compare) do_compare ;;
  all) do_env; do_bake; do_compare ;;
  *) die "phase must be env|smoke|bake|compare|all" ;;
esac
say "=== [$PHASE] finished ==="
