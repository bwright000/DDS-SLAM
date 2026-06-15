#!/bin/bash
# ============================================================================
# OVERNIGHT — Inc-1/Inc-2 tracking hedge A/B + field-finding confirm (T4). 2026-06-15.
# Two-pronged (CRCD + SemSup; StereoMIS dropped):
#   GATE  Inc-0 regression: flags-OFF build == pre-Inc-1 base (bit-identical) -> Inc-1/2 safe to trust.
#   B(1)  CRCD c1_001 (reference snippet) A/B: base vs +uncertainty -> tracking. base tracks c1_001 WELL
#         overall but SPIKES error when deformation happens -> the metric is per-frame ATE max/p90 (does the
#         uncertainty down-weight CUT the deformation-moment spikes), not mean ATE.
#   B(2)  SemSup trail3_moge2 A/B: base vs +uncertainty -> render (ship ckpts+logs; metric post-hoc).
#   A     Confirm "field gets ~0 gradient" on the GOOD model: verify_grad + grad_attrib + field3d
#         on the freshly-trained SemSup base ckpt (good model, not the pb7 stabiliser).
# Resume-safe (.DONE per cell). set -uo. Order = GATE -> CRCD A/B (priority) -> SemSup A/B + diagnose.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
REPO=/content/DDS-SLAM
PRE_INC1=8579f4c                                   # commit just before Inc-1 (for the rigorous Inc-0 golden)
DRIVE=/content/drive/MyDrive/Outputs/dds_inc1inc2_${DATE}
LWORK=/content/inc1inc2; mkdir -p "$DRIVE" "$LWORK"
LOG="$DRIVE/runbook.log"; exec > >(tee -a "$LOG") 2>&1
say(){ echo ""; echo "[$(date +%H:%M:%S)] $*"; }
done_marker(){ [ -f "$1/.DONE" ]; }
say "=== inc1/inc2 overnight start $(date -Iseconds)  DRIVE=$DRIVE  HEAD=$(cd $REPO && git rev-parse --short HEAD) ==="
[ -d /content/drive/MyDrive ] || { say "FATAL: Drive not mounted"; exit 1; }
activate_dds_env(){
  if ! python -c "import torch, tinycudann, marching_cubes" 2>/dev/null; then
    say "env rebuild (~15 min)"; bash "$REPO/Addons/env/colab_setup.sh" --skip-data --skip-tunnel; fi
  python -c "import torch, tinycudann; assert torch.cuda.is_available()" || { say "env FAIL"; exit 1; }
  python -c "import lpips" 2>/dev/null || pip install -q lpips || true
  export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
}
stage_semsup(){
  local SRC=/content/drive/MyDrive/Datasets/SemSup/v2_data/trial_3
  [ -d "$REPO/data/Super/trail_3/rgb" ] && return 0
  [ -d "$SRC/rgb" ] || { say "FATAL: SemSup source missing"; return 1; }
  mkdir -p "$REPO/data/Super"; cp -r "$SRC" "$REPO/data/Super/trail_3"
}
ship(){ local OUTB=$1 LW=$2 DST=$3   # copy ckpt+est+output, tar to Drive, mark DONE
  local DEMO="$OUTB/demo"; local CKPT=$(ls -t "$DEMO"/checkpoint*.pt 2>/dev/null | head -1)
  [ -n "$CKPT" ] && cp "$CKPT" "$LW/checkpoint.pt" 2>/dev/null
  cp "$DEMO"/est_c2w_data.txt "$DEMO"/output.txt "$LW/" 2>/dev/null || true
  tar czf "$DST/payload.tgz.partial" -C "$LW" . && mv "$DST/payload.tgz.partial" "$DST/payload.tgz"; sync; touch "$DST/.DONE"; }
train(){ local CFG=$1; say "  train $CFG"; local T0=$(date +%s)
  python -W ignore ddsslam.py --config "$CFG" 2>&1 | tee -a "$LOG" || say "  WARN $CFG nonzero exit"
  say "  $(basename $CFG) train $(( ($(date +%s)-T0)/60 )) min"; }

nvidia-smi -L || true
cd "$REPO"; activate_dds_env; stage_semsup || exit 1

# ---- GATE: Inc-0 regression (rigorous bit-identity vs PRE-Inc-1, via a git WORKTREE so the
# running script + working tree are NEVER disturbed; falls back to a HEAD self-check if it fails) ----
say "########## GATE: Inc-0 regression ##########"
G="$DRIVE/inc0"; mkdir -p "$G"
if ! done_marker "$G"; then
  CFG="$REPO/configs/Super/trail3_paper_faithful.yaml"; GLD=/content/golden_preinc1.json
  if git worktree add -q /content/preinc1 "$PRE_INC1" 2>/dev/null; then
    ( cd /content/preinc1 && python Addons/regression/test_inc0_bitidentical.py --config "$CFG" --golden "$GLD" --write-golden ) 2>&1 | tee -a "$LOG"
    # check HEAD (Inc-1 code, flags OFF) against the PRE-Inc-1 golden -> proves bit-identity
    python Addons/regression/test_inc0_bitidentical.py --config "$CFG" --golden "$GLD" 2>&1 | tee "$G/inc0_check.txt"
    cp "$GLD" "$G/" 2>/dev/null; git worktree remove --force /content/preinc1 2>/dev/null
  else
    say "  WARN: worktree add failed -> HEAD self-check only (static harden already confirmed off==base)"
    python Addons/regression/test_inc0_bitidentical.py --config "$CFG" --write-golden 2>&1 | tee "$G/inc0_check.txt"
    python Addons/regression/test_inc0_bitidentical.py --config "$CFG" 2>&1 | tee -a "$G/inc0_check.txt"
  fi
  sync; touch "$G/.DONE"
fi

# ---- PRIORITY: CRCD c1_001 tracking A/B (base vs +uncertainty) ----
say "########## B(1): CRCD c1_001 tracking A/B (reference; spike-at-deformation) ##########"
crcd_stage(){ local STAGED="$REPO/data/CRCD/C1_001"
  [ -d "$STAGED/video_frames" ] && return 0
  local SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
  local CALIB=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
  local MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001/depth
  [ -d "$SNIP" ] && [ -f "$CALIB" ] && [ -d "$MOGE" ] || { say "  CRCD c1 prereqs MISSING -> SKIP CRCD A/B"; return 1; }
  python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$SNIP" --calib_pkl "$CALIB" --output_dir "${STAGED}.tmp" 2>&1 | tee -a "$LOG" && mv "${STAGED}.tmp" "$STAGED" || return 1
  mkdir -p "$STAGED/depth.tmp"; python3 - "$MOGE" "$STAGED/depth.tmp" <<'PY'
import os,sys,shutil
src,dst=sys.argv[1],sys.argv[2]
for i,f in enumerate(sorted(x for x in os.listdir(src) if x.endswith('.png'))): shutil.copy2(os.path.join(src,f),os.path.join(dst,f'{i:06d}.png'))
print('copied',len(os.listdir(dst)),'MoGe depth')
PY
  rm -rf "$STAGED/depth" && mv "$STAGED/depth.tmp" "$STAGED/depth"; }
if crcd_stage; then
  for V in c1_001_uncert_base c1_001_uncert; do
    DST="$DRIVE/$V"; LW="$LWORK/$V"; done_marker "$DST" && { say "  $V done"; continue; }
    mkdir -p "$LW" "$DST"; OUTB="output/CRCD/C1_001_${V#c1_001_}"
    train "configs/CRCD/$V.yaml"; ship "$OUTB" "$LW" "$DST"; say "  $V shipped"
  done
fi

# ---- SemSup trail3_moge2 render A/B + field-finding diagnose on the base ckpt ----
say "########## B(2)+A: SemSup trail3_moge2 A/B + diagnose ##########"
if [ ! -d "$REPO/data/Super/trail_3/depth/moge2" ]; then
  say "  WARN: SemSup moge2 depth MISSING (data/Super/trail_3/depth/moge2) -> SKIP SemSup A/B. Stage moge2 depth or switch configs to paper_faithful."
else
  for V in trail3_moge2_uncert_base trail3_moge2_uncert; do
    DST="$DRIVE/$V"; LW="$LWORK/$V"; done_marker "$DST" && { say "  $V done"; continue; }
    mkdir -p "$LW" "$DST"; OUTB="output/$V"
    train "configs/Super/$V.yaml"
    CKPT=$(ls -t "$OUTB"/demo/checkpoint*.pt 2>/dev/null | head -1)
    if [ "$V" = "trail3_moge2_uncert_base" ] && [ -n "$CKPT" ]; then
      say "  --- PRONG A: confirm field finding on the GOOD SemSup model ---"
      python diagnosis/infra/verify_grad_probe.py --config configs/Super/trail3_moge2.yaml --checkpoint "$CKPT" --json "$LW/verify_grad.json" 2>&1 | tee -a "$LOG" || say "  WARN verify_grad"
      python diagnosis/infra/grad_attribution_probe.py --config configs/Super/trail3_moge2.yaml --checkpoint "$CKPT" --json "$LW/grad_attrib.json" --max_frames 12 2>&1 | tee -a "$LOG" || say "  WARN grad_attrib"
      python diagnosis/infra/dx_hook.py --config configs/Super/trail3_moge2.yaml --checkpoint "$CKPT" --output_dir "$LW/dx" 2>&1 | tee -a "$LOG" || say "  WARN dx_hook"
      python diagnosis/infra/field_viz3d.py --dx_dir "$LW/dx" --out "$LW/field3d.png" --bound "[[-0.7,0.7],[-0.7,0.7],[0.7,1.2]]" --max_frames 12 2>&1 | tee -a "$LOG" || say "  WARN field3d"
    fi
    ship "$OUTB" "$LW" "$DST"; say "  $V shipped"
  done
fi

# ---- SUMMARY: CRCD tracking A/B (Sim3 ATE + path ratio) ----
say "########## SUMMARY ##########"
python3 - "$LWORK" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys, numpy as np
LW=sys.argv[1]
def est(p):
    P=[]
    if not os.path.isfile(p): return np.zeros((0,3))
    for l in open(p):
        v=l.split()
        if len(v)>=12 and not v[0].startswith('#'): P.append(np.array(list(map(float,v[:12]))).reshape(3,4)[:3,3])
    return np.array(P)
def tum(p):
    P=[]
    if not os.path.isfile(p): return np.zeros((0,3))
    for l in open(p):
        v=l.split()
        if len(v)>=8 and not v[0].startswith('#'): P.append([float(v[1]),float(v[2]),float(v[3])])
    return np.array(P)
def horn(m,d):
    mc,dc=m.mean(0),d.mean(0); mm,dd=m-mc,d-dc; H=mm.T@dd; U,S,Vt=np.linalg.svd(H)
    s=np.sign(np.linalg.det(Vt.T@U.T)); R=Vt.T@np.diag([1,1,s])@U.T
    sc=(S*np.array([1,1,s])).sum()/(mm*mm).sum(); return (sc*(R@m.T)).T+(dc-sc*R@mc), sc
GT='/content/DDS-SLAM/data/CRCD/C1_001/groundtruth.txt'; g=tum(GT)
print("CRCD c1_001 tracking A/B (Sim3-aligned per-frame ATE, mm). base tracks well overall -> watch the")
print("SPIKES (ATE_max / ATE_p90 = the deformation moments), not the mean:")
for V in ['c1_001_uncert_base','c1_001_uncert']:
    e=est(f'{LW}/{V}/est_c2w_data.txt')
    if len(e)<10 or len(g)<10: print(f"  {V:<22} (no est)"); continue
    n=min(len(e),len(g)); a,sc=horn(e[:n],g[:n]); pf=np.linalg.norm(a-g[:n],axis=1)*1000   # per-frame ATE mm
    pr=(np.linalg.norm(np.diff(e[:n],axis=0),axis=1).sum())/(np.linalg.norm(np.diff(g[:n],axis=0),axis=1).sum()+1e-9)
    print(f"  {V:<22} ATE_mean={pf.mean():6.2f}  ATE_p90={np.percentile(pf,90):6.2f}  ATE_max={pf.max():6.2f}  path_ratio={pr:5.2f}")
print("\nHEDGE WORKS if uncert ATE_max/ATE_p90 < base (it CUTS the deformation-moment spikes), even if the")
print("  means are similar. SemSup render A/B + field-finding: see per-cell jsons.")
print("PRONG A (field on good model): trail3_moge2_uncert_base/{verify_grad,grad_attrib,field3d}.json")
PY
say "=== overnight DONE $(date -Iseconds) ==="
python3 -c "from google.colab import runtime; runtime.unassign()" 2>/dev/null || say "(not Colab/already free)"
