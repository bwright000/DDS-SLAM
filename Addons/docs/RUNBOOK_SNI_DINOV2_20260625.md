# RUNBOOK 1 — SNI-SLAM with a CRCD-trained DINOv2 head predicting seg LIVE

**Goal.** Run SNI-SLAM with `use_gt_semantic=False` so its `DINO2SEG` predicts the 4-class CRCD seg
**live** from *our* per-snippet leave-one-out (LOSO) DINOv2 ViT-B/14 head — the in-domain, **no-bake**
counterpart of the DINOv3 run-B that degraded. A/B = **GT-seg baseline (`SEG=gt`)** vs **predicted
DINOv2 seg (`SEG=dinov2`)**.

**Why it works (no SNI source change needed).** DINOv2/14 runs natively in SNI's py3.7/torch1.11 env
(no torch2, no offline bake). When `use_gt_semantic=False`, `Mapper.py` calls `DINO2SEG.forward(mode='train')`
→ predicted class map. Our head (`train_dinov2_crcd.py --backbone dinov2`: 2-conv 768→16→4, `dim=16`,
`crop_edge=0`) is param-name-identical to SNI's `DINO2SEG`; `model_manager.py` loads it `strict=False`
with **no dropped keys** (unlike the 52-class `dinov2_replica.pth`).

**Metric = TRACKING** (Sim3-ATE / path-ratio / Pearson in `extended_metrics.txt`) + render PSNR/SSIM.
Render stays render-weak (~12–13, architectural — see the SGS-vs-SNI lessons); the seg only moves the
semantic field + (via `w_semantic`/`w_feature`) the tracking.

---

## Prerequisites
- **Data (Drive):** `Datasets/CRCD-Published/<EP>/snippet_<SID>/{rgb,rgbright,semantic_instance,groundtruth.txt,intrinsics.yaml}` + `Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/depth/`. Use the **360-row** Published GT, never the stale 271-row repo copy.
- **Env A (training):** DDS-SLAM modern torch2 stack (`Addons/env/colab_setup.sh`). DINOv2 ViT-B/14 weights download freely (`dl.fbaipublicfiles.com/dinov2/...`, 346 MB) — **not gated** (unlike DINOv3).
- **Env C (SNI):** conda `sni` (py3.7/torch1.11) via `sni-slam/scripts/colab_setup_sni.sh`.

---

## Phase A — train the 5 LOSO DINOv2 heads (DDS-SLAM, torch2 env)
```bash
cd /content/DDS-SLAM && git pull
TAG=v2_freeze BACKBONE=dinov2 TRAIN_BLOCKS=0 LOSS=wce_dice AUG=1 \
  bash Addons/colab/train_seg_head_crcd_loso_20260619.sh
```
Trains 5 folds over **BENCH5 = C1_001 C2_001 E3_005 C3_001 G3_001** (each fold trains on the other ~17
snippets, holds out the test snippet; inner-val = G2_003 F3_007). Frozen backbone + aug + wce_dice,
504×896, 25 ep. **Check each `fold_<S>.log` for a sane held-out mIoU.** Single fold: `FOLDS="C1_001"`.
Output: `/content/drive/MyDrive/Outputs/seg/loso_v2_freeze/dinov2_crcd_<S>.pth`.

## Phase B — stage the heads where SNI expects them
```bash
mkdir -p /content/drive/MyDrive/Datasets/seg/CRCD_DINOv2_heads
for S in C1_001 C2_001 E3_005 C3_001 G3_001; do
  cp /content/drive/MyDrive/Outputs/seg/loso_v2_freeze/dinov2_crcd_${S}.pth \
     /content/drive/MyDrive/Datasets/seg/CRCD_DINOv2_heads/dinov2_crcd_${S}.pth
done
```
(The runbook's default `CRCD_HEAD_PTH` points here; override with `CRCD_HEAD_PTH=...` if you keep them elsewhere.)

## Phase C — run the A/B (SNI, `sni` env)
```bash
cd /content/sni-slam && git fetch && git checkout sni-pixel-density && git pull
conda activate sni
# GT-seg baseline (current best config: low-iters + frozen joint BA)
ONLY="c1_001 c2_001" SEG=gt     nohup bash scripts/run_crcd_4snippets.sh > /content/sni_gt.log 2>&1 &
# wait for it, then the live-DINOv2-seg arm:
ONLY="c1_001 c2_001" SEG=dinov2 nohup bash scripts/run_crcd_4snippets.sh > /content/sni_dv2.log 2>&1 &
tail -f /content/sni_dv2.log | grep -a --line-buffered -E "PHASE|SEG=dinov2|DINOv2 predicted|FATAL|Traceback|PSNR|SSIM|pearson|done"
```
Phase 2.8 prints `LIVE DINOv2 predicted seg: head=…` and Phase 3.7 `SEG=dinov2: use_gt_semantic=False + …`.
Results: `MyDrive/Outputs/sni_crcd_4snippets_<date>/{c1_001 (gt) , c1_001_dinov2}/` — compare
`extended_metrics.txt` (tracking) between the two TAGs.

## Phase D (recommended) — sanity-check the head before trusting the SLAM A/B
In the `sni` env, load `CRCD_HEAD_PTH` into `DINO2SEG(720,1280,num_cls=4,edge=0,dim=16)`, `mode='train'`,
argmax one staged `rgb_*.png`, save the mask. Confirm classes 0–3 (bg/Liver/Gallbladder/Tool) — the head
trained at 504×896 but SNI infers at 720×1280 (DINOv2 is patch-grid-agnostic, but verify it's not garbage).

---

## OPEN ITEMS (resolve before scaling past c1/c2)
1. **SNIPPETS coverage.** `run_crcd_4snippets.sh` `SNIPPETS` lists only f3_007/c1_001/c2_001/f1_002. **c1_001 + c2_001 overlap BENCH5 and run today.** To do the full BENCH5 A/B you must add `e3_005`/`c3_001`/`g3_001` rows + their per-snippet configs (`configs/CRCD/{e3_005,c3_001,g3_001}.yaml` with `mapping.bound`) + their `sc_factor` (the Phase-3.6 hardcoded fallback only has f3/c1/c2/f1) — same gap as the earlier "SNI on best config" run.
2. **Head filename.** Trainer writes `dinov2_crcd_<S>.pth` (no suffix); Phase B + the runbook default both use that. If you keep a different name, set `CRCD_HEAD_PTH`.
3. **Train/infer res** (504×896 → 720×1280): Phase D verifies it's fine.
