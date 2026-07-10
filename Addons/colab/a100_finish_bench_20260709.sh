#!/bin/bash
# ============================================================================
# a100_finish_bench_20260709.sh — finish the CRCD Arm-4 benchmark on ONE fresh
# A100-80GB high-RAM box. Three INDEPENDENT phases, run SEQUENTIALLY (the GPU is
# shared), each fail-isolated + logged to Drive. Total ~8-12 h; safe to walk away.
#
#   A) SemGauss LPIPS rescore  — fills the C2/C3/E3 render_eval LPIPS cells that
#      were "--" (lpips lib was absent at run time). No SLAM: reads the renders
#      already on Drive, recomputes, patches metrics.json. ~5 min.
#   B) SemGauss G3             — the two missing G3 rows: (B1) faithful capped
#      750-frame main-table row, (B2) tightprune full-length ablation. Fresh box
#      -> /content empty -> no corrupted local G3 frame, clean stage. C1/C2/C3/E3
#      are .DONE on Drive -> auto-skipped. sem_gauss conda env (py3.10/torch1.12/
#      cu116, rasterizer sm_80).
#   C) SNI-SLAM 5-snippet bench — the whole SNI row (currently 0/5). FAITHFUL
#      config incl. the metric-proven lr_T/lr_R=1e-4 (fork @b87d61d). A100-80GB
#      -> full authors' pixels (no T4 caps); keyframe store on cuda:0. Render is
#      architecturally weak (~11 PSNR, expected) -> the TRACKING metrics are the
#      point (path-ratio should be ~11x, not the 23x of the reverted 1e-3).
#
# Usage (fresh A100, Drive mounted in a notebook cell first):
#   git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM /content/DDS-SLAM
#   nohup bash /content/DDS-SLAM/Addons/colab/a100_finish_bench_20260709.sh \
#       &> /content/finish_bench.out & disown
#   tail -f /content/finish_bench.out
#
# Skip a phase:  PHASES="B C" bash .../a100_finish_bench_20260709.sh   (default "A B C")
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}
DATE=$(date +%Y%m%d)
PHASES=${PHASES:-"A B C"}
SG_OUT=${SG_OUT:-/content/drive/MyDrive/Outputs/SemGauss-SLAM_bench_20260704}   # existing 4/5 dir (skip C1-E3)
SNI_OUT=${SNI_OUT:-/content/drive/MyDrive/Outputs/SNI_bench_$DATE}
SG_PY=/content/miniconda3/envs/sem_gauss/bin/python
LOG=/content/drive/MyDrive/Outputs/a100_finish_bench_$DATE.log
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "==================== [$(date +%H:%M:%S)] $* ===================="; }
has(){ case " $PHASES " in *" $1 "*) return 0;; *) return 1;; esac; }

say "A100 FINISH-BENCH start  phases='$PHASES'  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ -d "$REPO/.git" ] || { echo "FATAL: $REPO not a git repo (clone -b diagnosis-live first)"; exit 1; }

# ---------------------------------------------------------------- Phase A ----
if has A; then
  say "PHASE A: SemGauss LPIPS rescore (C2/C3/E3; renders already on Drive)"
  python3 -m pip install -q lpips 2>/dev/null || echo "[A] WARN lpips pip (may already be present)"
  for S in C2_001 C3_001 E3_005; do
    D="$SG_OUT/$S"
    [ -d "$D" ] || { echo "[A] $S dir missing on Drive -> skip"; continue; }
    [ "$(ls "$D"/*.jpg 2>/dev/null | wc -l)" -gt 0 ] || { echo "[A] $S has no renders on Drive -> skip"; continue; }
    python3 "$REPO/Addons/eval/eval_rendering.py" --gt_dir "$D" --render_dir "$D" \
       --sequence "CRCD ($S)" --output_csv "$D/render_eval.csv" --summary_csv "$D/render_eval.txt" \
       && python3 - "$D" <<'PY' || echo "[A] $S rescore failed"
import sys, os, csv, json
d = sys.argv[1]; rows = list(csv.DictReader(open(os.path.join(d, 'render_eval.txt'))))
mp = os.path.join(d, 'metrics.json'); m = json.load(open(mp)) if os.path.isfile(mp) else {}
r = rows[-1] if rows else {}
for kj, kc in (('psnr','psnr_mean'), ('ssim','ssim_mean'), ('lpips','lpips_mean')):
    if r.get(kc): m[kj] = float(r[kc])
json.dump(m, open(mp, 'w'), indent=2)
print(f"[A-rescore] {os.path.basename(d)}: psnr={m.get('psnr')} ssim={m.get('ssim')} lpips={m.get('lpips')}")
PY
  done
fi

# ---------------------------------------------------------------- Phase B ----
if has B; then
  say "PHASE B: SemGauss env build (sem_gauss py3.10/torch1.12/cu116, rasterizer sm_80)"
  if bash "$REPO/Addons/colab/run_semgauss.sh" env; then
    say "PHASE B1: G3 capped 750f (faithful; main-table row)"
    NUM_FRAMES=750 DRIVE_OUT="$SG_OUT" bash "$REPO/Addons/colab/run_semgauss.sh" crcd g3_001 \
      || echo "[B1] G3 capped FAILED (isolated)"
    say "PHASE B2: G3 tightprune full-length (NON-FAITHFUL ablation; side-note)"
    SEMGAUSS_VARIANT=tightprune DRIVE_OUT="$SG_OUT" bash "$REPO/Addons/colab/run_semgauss.sh" crcd g3_001 \
      || echo "[B2] G3 tightprune FAILED (isolated)"
    say "PHASE B: final Gaussian counts (receipts)"
    for R in G3_001_2027 G3_001_tightprune_2027; do
      N=$(ls -t /content/SemGauss-SLAM/experiments/CRCD/$R/params*.npz 2>/dev/null | head -1)
      [ -n "$N" ] && PYTHONPATH= "$SG_PY" -c "import numpy as np; p=np.load('$N'); print('  $R:', f'{p[\"means3D\"].shape[0]:,} gaussians')" 2>/dev/null
    done
  else
    echo "[B] SemGauss env build FAILED (exit $?) -> skipping G3"
  fi
fi

# ---------------------------------------------------------------- Phase C ----
if has C; then
  say "PHASE C: SNI env build (sni py3.7/torch1.11/pytorch3d0.7.1; restores dds_cache if present)"
  if bash "$REPO/Addons/colab/run_snislam.sh" env; then
    say "PHASE C-run: SNI 5-snippet bench (FAITHFUL lr-1e-4; A100 full pixels; store=cuda:0) -> $SNI_OUT"
    # NO pixel caps (A100-80GB has room -> authors' 2000/4000). cuda:0 store leverages the 80GB VRAM.
    SNI_STORE_DEVICE=cuda:0 DRIVE_OUT="$SNI_OUT" bash "$REPO/Addons/colab/run_snislam.sh" crcd \
      || echo "[C] SNI bench had per-snippet failures (isolated)"
  else
    echo "[C] SNI env build FAILED (exit $?) -> skipping SNI. Check the pytorch3d 0.7.1/cu113 build + cache OS stamp."
  fi
fi

# ---------------------------------------------------------------- summary ----
say "ALL REQUESTED PHASES DONE -> aggregates"
has B && bash "$REPO/Addons/colab/run_semgauss.sh" eval 2>/dev/null || true
has C && DRIVE_OUT="$SNI_OUT" bash "$REPO/Addons/colab/run_snislam.sh" eval 2>/dev/null || true
say "finish-bench complete. master log: $LOG"
