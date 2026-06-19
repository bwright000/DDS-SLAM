#!/bin/bash
# ============================================================================
# train_seg_head_crcd_loso_20260619.sh  -  Arm 4 seg-head LOSO + recipe/backbone sweep.
#
#   Leave-one-SNIPPET-out (LOSO) over the 5 benchmark snippets: for each, train on the
#   OTHER 19 snippets, hold out 2 TRAIN snippets as inner-val (honest model selection, NO
#   leaky random-frame split), test on the 1 held-out benchmark snippet.
#
#   For the 4 single-snippet episodes (C1/C2/C3/G3) LOSO == leave-one-EPISODE-out -> a clean
#   cross-patient number. E3_005 shares episode E_3 with the train E3_001-004 -> under LOSO it
#   is IN-DOMAIN: it is reported SEPARATELY and EXCLUDED from the cross-episode headline mean.
#
#   ONE config per run (env-driven), so it doubles as the freeze-granularity x backbone SWEEP:
#     # baseline (the papers' recipe) reference:
#     TAG=ref_blocks8 TRAIN_BLOCKS=8 AUG=0 LOSS=ce            bash ...loso....sh
#     # improved recipe, frozen + aug + balanced loss, generic DINOv2:
#     TAG=v2_freeze   TRAIN_BLOCKS=0 AUG=1 LOSS=wce_dice      bash ...loso....sh
#     # same recipe, linear head / +LoRA-substitute / surgical / dinov3 backbones:
#     TAG=v2_linear   TRAIN_BLOCKS=0 AUG=1 LINEAR_HEAD=1                         bash ...
#     TAG=surgenet    BACKBONE=surgenet BACKBONE_WEIGHTS=$REPO/seg/DINOv2_ViTb14_size336_SurgeNetXL.pth bash ...
#     TAG=dinov3      BACKBONE=dinov3   BACKBONE_WEIGHTS=$REPO/seg/dinov3_vitb16.pth                     bash ...
#
#   RUNS ON COLAB (A100/T4). Frozen configs are fast (no backbone grads).
#   Usage:  SKIP_ENV=1 TAG=v2_freeze bash Addons/colab/train_seg_head_crcd_loso_20260619.sh
# ============================================================================
set -uo pipefail
REPO=${REPO:-/content/DDS-SLAM}; cd "$REPO"

TRAIN15="B2_001 E1_001 E3_001 E3_002 E3_003 E3_004 F1_002 F3_001 F3_002 F3_003 F3_004 F3_005 F3_006 F3_007 G2_003"
BENCH5=${BENCH5:-"C1_001 C2_001 E3_005 C3_001 G3_001"}
CLEAN4="C1_001 C2_001 C3_001 G3_001"     # E3_005 excluded from the cross-episode headline (in-domain under LOSO)
ALL20="$TRAIN15 $BENCH5"

# ---- config (env-driven; defaults = the improved recipe on generic DINOv2) ------------------
TAG=${TAG:-v2_freeze}
BACKBONE=${BACKBONE:-dinov2}                      # dinov2 | surgenet | dinov3
BACKBONE_WEIGHTS=${BACKBONE_WEIGHTS:-url}         # url | local .pth (surgenet/dinov3 need a path)
TRAIN_BLOCKS=${TRAIN_BLOCKS:-0}                   # 0=freeze (head only); 8=baselines' blocks 4-11
LOSS=${LOSS:-wce_dice}                            # ce|wce|focal|dice|wce_dice
AUG=${AUG:-1}; GIN=${GIN:-0}; LINEAR_HEAD=${LINEAR_HEAD:-0}; TUNE_NORMS=${TUNE_NORMS:-0}
VAL_SNIPS=${VAL_SNIPS:-"G2_003 F3_007"}          # 2 TRAIN snippets held out as inner-val (never benchmark)
FRAME_STRIDE=${FRAME_STRIDE:-2}                   # keep every Nth train/val frame (cost cut; test always full)
PATIENCE=${PATIENCE:-6}                           # early-stop after N epochs w/o inner-val gain (0=off)
IMG_H=${IMG_H:-504}; IMG_W=${IMG_W:-896}; EPOCHS=${EPOCHS:-25}; LR=${LR:-1e-4}
SEGDATA=${SEGDATA:-/content/crcd_seg_data}
DRIVE_CRCD=${DRIVE_CRCD:-/content/drive/MyDrive/Datasets/CRCD-Published}
OUT_DRIVE=${OUT_DRIVE:-/content/drive/MyDrive/Outputs/seg/loso_$TAG}
SURGENET_URL="https://huggingface.co/rlpddejong/SurgeNetXL_DINOv1-v3/resolve/main/DINOv2_ViTb14_size336_SurgeNetXL.pth?download=true"

echo "=== seg-head LOSO  TAG=$TAG  $(date -Iseconds) ==="
echo "  backbone=$BACKBONE weights=$BACKBONE_WEIGHTS train_blocks=$TRAIN_BLOCKS loss=$LOSS aug=$AUG gin=$GIN linear=$LINEAR_HEAD norms=$TUNE_NORMS"
echo "  val(inner)=$VAL_SNIPS  res=${IMG_H}x${IMG_W} epochs=$EPOCHS"
mkdir -p "$OUT_DRIVE"

if [ "${SKIP_ENV:-0}" = 1 ]; then
  python -c "import torch,cv2,numpy" 2>/dev/null || pip install -q opencv-contrib-python numpy
else
  bash Addons/env/colab_setup.sh --skip-data --skip-tunnel >/dev/null 2>&1 || true
fi
export XFORMERS_DISABLED=${XFORMERS_DISABLED:-1}

# vendored dinov2 main (needed for /14 backbones; dinov3 uses torch.hub)
if [ -z "${DINOV2_MAIN:-}" ]; then
  for c in /content/SemGauss-SLAM/segmentation/facebookresearch_dinov2_main \
           /content/sni-slam/seg/facebookresearch_dinov2_main; do
    [ -f "$c/dinov2/models/vision_transformer.py" ] && DINOV2_MAIN="$c" && break
  done
fi
[ -n "${DINOV2_MAIN:-}" ] || { echo "FATAL: no vendored dinov2 main"; exit 30; }

# backbone weights
if [ "$BACKBONE" = surgenet ]; then
  [ "$BACKBONE_WEIGHTS" = url ] && BACKBONE_WEIGHTS="$REPO/seg/DINOv2_ViTb14_size336_SurgeNetXL.pth"
  [ -f "$BACKBONE_WEIGHTS" ] || { echo "[surgenet] downloading SurgeNetXL DINOv2_ViTb14..."; \
     mkdir -p "$(dirname "$BACKBONE_WEIGHTS")"; curl -fL -o "$BACKBONE_WEIGHTS" "$SURGENET_URL" || { echo "FATAL: SurgeNetXL download failed"; exit 1; }; }
elif [ "$BACKBONE" = dinov3 ]; then
  [ -f "$BACKBONE_WEIGHTS" ] || { echo "FATAL: dinov3 needs gated weights at BACKBONE_WEIGHTS=<.pth> (accept terms on HF, download with token); see DINOV3_HUB/DINOV3_ENTRY"; exit 1; }
fi

# snippet NAME -> "EP SID" (C1_001 -> C_1 001)
ep_sid(){ local n=$1; [[ "$n" =~ ^[A-Za-z][0-9]_[0-9]{3}$ ]] || { echo ""; return; }; echo "${n:0:1}_${n:1:1} ${n:3}"; }
stage_raw(){ local NAME=$1 dst="$SEGDATA/$NAME"
  [ "$(ls "$dst/rgb"/*.png 2>/dev/null | wc -l)" -gt 0 ] && [ "$(ls "$dst/semantic_instance"/*.png 2>/dev/null | wc -l)" -gt 0 ] && { echo "[$NAME] staged"; return 0; }
  local EP SID; read -r EP SID <<< "$(ep_sid "$NAME")"; [ -n "$EP" ] || { echo "[$NAME] bad name"; return 1; }
  local SRC="$DRIVE_CRCD/$EP/snippet_$SID"
  [ -d "$SRC/rgb" ] && [ -d "$SRC/semantic_instance" ] || { echo "[$NAME] raw not on Drive ($EP/snippet_$SID)"; return 1; }
  rm -rf "$dst"; mkdir -p "$dst/rgb" "$dst/semantic_instance"   # clean restage; cp CONTENTS (avoid nested rgb/rgb on partial)
  cp -rn "$SRC/rgb/." "$dst/rgb/"; cp -rn "$SRC/semantic_instance/." "$dst/semantic_instance/"
  echo "[$NAME] staged $(ls "$dst/rgb"/*.png 2>/dev/null | wc -l) frames"; }
for NAME in $ALL20; do stage_raw "$NAME" || true; done

flags="--backbone $BACKBONE --backbone_weights $BACKBONE_WEIGHTS --train_blocks $TRAIN_BLOCKS --loss $LOSS"
[ "$AUG" = 1 ]         && flags="$flags --aug"
[ "$GIN" = 1 ]         && flags="$flags --gin"
[ "$LINEAR_HEAD" = 1 ] && flags="$flags --linear_head"
[ "$TUNE_NORMS" = 1 ]  && flags="$flags --tune_norms"
flags="$flags --frame_stride $FRAME_STRIDE --patience $PATIENCE"
# sweep-only configs (non-deployable head/arch) get a _SWEEPONLY .pth so they can't be mistaken for a deploy ckpt
SUF=""; { [ "$LINEAR_HEAD" = 1 ] || [ "$BACKBONE" = dinov3 ]; } && SUF="_SWEEPONLY"

# ---- LOSO folds: for each benchmark snippet, train on the other 17 (=19 non-held-out minus 2 inner-val) ----
for S in $BENCH5; do
  TR=""; for n in $ALL20; do [ "$n" = "$S" ] && continue; case " $VAL_SNIPS " in *" $n "*) continue;; esac; TR="$TR $n"; done
  echo ""; echo "### FOLD test=$S  (train on the other 17; 19 non-held-out minus 2 inner-val) ###"
  case " $CLEAN4 " in *" $S "*) echo "    [fold] $S = TRUE cross-episode";; *) echo "    [fold] $S = IN-DOMAIN under LOSO (E_3 in train) - reported separately";; esac
  log="$OUT_DRIVE/fold_${S}.log"; rm -f "$OUT_DRIVE/fold_${S}.FAILED"
  python Addons/seg/train_dinov2_crcd.py \
    --crcd_root "$SEGDATA" --snippets $TR --val_snippets $VAL_SNIPS --test_snippets "$S" \
    --dinov2_main "$DINOV2_MAIN" --out "$OUT_DRIVE/dinov2_crcd_${S}${SUF}.pth" \
    --rgb_subdir rgb --label_subdir semantic_instance \
    --n_classes 4 --dim 16 --img_h "$IMG_H" --img_w "$IMG_W" --crop_edge 0 \
    --epochs "$EPOCHS" --lr "$LR" $flags 2>&1 | tee "$log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[$S] FAILED (see $log)"; touch "$OUT_DRIVE/fold_${S}.FAILED"; }
done

# ---- summarize: mean over the 4 TRUE cross-episode folds; E3_005 separate ---------------------
echo ""; echo "=== LOSO SUMMARY  TAG=$TAG  backbone=$BACKBONE ==="
python - "$OUT_DRIVE" "$CLEAN4" <<'PY'
import sys, glob, os, re
out=sys.argv[1]; clean=set(sys.argv[2].split())
def heldout(log):
    t=open(log,encoding='utf-8',errors='ignore').read()
    m=re.search(r'\[HELD-OUT TEST\][^=]*=\s*([0-9.]+)',t); return float(m.group(1)) if m else None
rows={}
for lg in sorted(glob.glob(os.path.join(out,'fold_*.log'))):
    s=os.path.basename(lg)[5:-4]; rows[s]=heldout(lg)
import statistics as st
failed={os.path.basename(p)[5:-7] for p in glob.glob(os.path.join(out,'fold_*.FAILED'))}
cl=[v for s,v in rows.items() if s in clean and v is not None]
for s,v in rows.items():
    tag='cross-episode' if s in clean else 'IN-DOMAIN(sep)'
    mark=' FAILED' if s in failed else ''
    print(f"  {s:<8} mIoU={v if v is None else round(v,4)}  [{tag}]{mark}")
miss=[s for s in clean if rows.get(s) is None]
if miss or failed:
    print(f"  !! INCOMPLETE: missing/failed clean folds = {sorted(set(miss)|(failed&clean))} "
          f"-> headline is n={len(cl)}/{len(clean)}; DO NOT compare to a full-{len(clean)} run")
if cl:
    print(f"  --> CROSS-EPISODE HEADLINE (n={len(cl)}/{len(clean)} clean folds): mean={st.mean(cl):.4f}"
          + (f" std={st.pstdev(cl):.4f}" if len(cl)>1 else ""))
print("  (E3_005 excluded from headline: E_3 is in train under LOSO -> in-domain)")
PY
echo "DONE -> $OUT_DRIVE (per-fold .pth + logs + summary)"
