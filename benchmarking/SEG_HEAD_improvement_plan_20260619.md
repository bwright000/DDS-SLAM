# CRCD seg-head improvement plan — research synthesis + ladder (2026-06-19)

> Goal: raise the CRCD 4-class seg head's **held-out (unseen-episode) mIoU** (currently ~0.34; in-domain val ~0.93) on a FIXED ~15-episode train set, ideally keeping the shared DINOv2-ViT-B/14 `DINO2SEG` checkpoint that SNI-SLAM + SemGauss-SLAM both load.
> Source: deep-research `wd3lbqadr` (24 sources, 25 claims adversarially verified, 11 confirmed) + targeted follow-up fetches. ⚠️ The run hit a session limit: surgical-model specifics were partly unverified (abstained, not refuted) and re-fetched here; the auto-synthesis was done by hand below.

## 0. TL;DR recommendation — UPDATED 2026-06-19 (research wf w2t409q34, 8-agent, cited)
**The held-out win is the TRAINING RECIPE, not a new head or backbone.** One metric-first plan, cheapest-first, every step gated on the SLAM tracking/render metric (seg mIoU is only a proxy; head is 2nd-order at semantic weight 0.05-0.1):
- **Step 0 — split + bug fix (do FIRST):** the current `val_frac 0.1` random-FRAME split (`train_dinov2_crcd.py:265-267`) leaks temporal neighbours → **the 0.93 val is leakage-inflated and an INVALID checkpoint-selection signal**. Replace with an **episode-grouped inner val** (hold out 1-2 whole TRAIN episodes; reuse the `ep_sid` parser). Adopt **leave-one-EPISODE-out (LOEO), NOT leave-one-snippet-out** — our 20 snippets = 10 episodes; C1/C2/C3/G3 are single-snippet episodes (true cross-patient) but **E3_005 shares episode E_3 with E3_001-004 (train) → in-domain, report SEPARATELY**. Headline = mean±std over the **4 true folds (C1/C2/C3/G3)**.
- **Step 1 — recipe (the actual win, keeps /14 drop-in):** **FREEZE the backbone** (Kumar ICLR'22: full-FT = −7% OOD, LP-FT = +10% OOD vs full-FT; Rein/Rein++ CVPR'24: frozen-VFM beats full-FT cross-domain) + **add domain-randomization augmentation** (GIN intensity-randomization #1 + heavy photometric/blur/elastic/flip + CLAHE; currently ZERO; +9-23 mIoU cross-center in surgical seg) + **class-balanced/Focal loss** (rescue Tool 0.20 / GB 0.39). Expect **0.34 → ~0.45-0.50**. Add **LoRA r=16-32** only if frozen-head underfits in-domain (merge before export to keep keys). Decoder head = KEEP our 2-conv (NOT the bug; frozen+linear generalizes as well as heavy decoders).
- **Step 2 — backbone A/B** on the Step-1 recipe: **ARM-A SurgeNetXL** `DINOv2_ViTb14_size336` (zero geometry change, in-domain prior, drop-in /14, CC-BY-NC-4.0) FIRST (free); **ARM-B DINOv3-ViT-B/16** SECOND (~10-20 LOC patch-14→16 across both `dinov2_seg.py` + cache re-bake). SLAM metric picks — neither dominates a priori.
- **Step 3 — acceptance:** confirm winning recipe×backbone end-to-end on SNI/SemGauss SLAM (sim3_ate ATE + PSNR/SSIM/LPIPS). If indifferent at weight 0.05-0.1, the **in-domain deploy head stands** and the cross-episode work ships as an honest side-finding.

**Realistic held-out ceiling ~0.45-0.55** (SOTA surgical cross-center 0.49-0.50); 0.60+ likely = E3_005 leak.

### DINOv3 reassessment (verdict FLIPPED: AVOID → CONDITIONAL)
Two prior assumptions were WRONG: (1) **DINOv3 license PERMITS commercial use** (custom DINOv3 License, 14 Aug 2025; *more* permissive than SurgeNetXL CC-BY-NC; gated on HF, cite-only obligation); (2) **direct RGB-endoscopy evidence EXISTS** (arXiv:2509.06467: frozen DINOv3+decoder = SOTA EndoVis18 IoU ~89 ViT-B / AutoLaparo / EDD2020 polyp; CRCD = the favorable RGB-colorectal bucket; DINOv3 *fails* on non-RGB). BUT all DINOv3 = **patch-16 (no /14)** → breaks the SNI/SemGauss geometry (bounded ~10-20 LOC + cache re-bake), and DINOv3-generic is **NOT demonstrated to beat SurgeNetXL-surgical** on surgery. → run as the SECOND backbone arm, metric-gated. ⚠️ verify the exact SurgeNetXL `DINOv2_ViTb14` LICENSE (research found CC-BY-NC-4.0, our earlier note said NC-SA).

---
*(Sections 1-6 below = the earlier wd3lbqadr synthesis + ladder; superseded for the headline by §0 above but kept for the per-option detail.)*

## 1. What the research CONFIRMED (3-0 verified)
- **DINOv2 family / our backbone:** ViT-S/14 21M, **ViT-B/14 86M (ours)**, L/14 300M, g/14 1.1B. [github.com/facebookresearch/dinov2]
- **Official DINOv2 seg recipe = lightweight head on a FROZEN backbone** (linear 1/4-layer, multi-scale, Mask2Former for g) — i.e. the *standard/recommended* approach is NOT full backbone fine-tuning. [dinov2 repo]
- **DINOv2-with-registers-base: Apache-2.0** (commercial-OK), cleaner dense/attention features (removes ViT register artifacts). [hf.co/facebook/dinov2-with-registers-base]
- **SegDINO**: frozen DINOv3 backbone + lightweight MLP decoder, train *only* the decoder — explicitly the frozen-features-suffice design. [arXiv 2509.00833]
- **DINOv3** (arXiv 2508.10104, released 2025-08-14): better **dense** features via "Gram anchoring" (fixes dense-feature degradation), strong **without fine-tuning**, distilled ViT-B/L + ConvNeXt T/S/B/L for small compute — **but COMMERCIAL license** (Meta), more restrictive than DINOv2's Apache. [arXiv + ai.meta.com blog]
- **SurgeNetXL** (surgical foundation model, DINO-based, **4.7M+ surgical frames, multi-procedure**): ships **DINOv1/v2/v3 ViT-s/b/l checkpoints** incl. a **DINOv2 ViT-B trained on surgical data**; reports **+14.4% segmentation over ImageNet-pretrained** and +2.4% over the best prior surgical FMs; weights public on HuggingFace; **license CC-BY-NC-SA (non-commercial)** = fine for a thesis, not commercial. [github.com/timjaspers0801/surgenet]
- **EndoDINO** (arXiv 2501.05488): DINOv2-method, pretrained on the largest GI-endoscopy video dataset; ViT-B/L/g (86M/307M/1B); **frozen encoder + simple decoder heads**; page lists CC-BY-4.0. ⚠️ its segmentation gain over natural-image DINOv2 is reportedly **small** (~+0.03 mIoU on polyp, *unverified* — session limit).

## 2. What the verifier KILLED (stay honest)
- "Frozen backbone is *strictly* better / DINOv2 needs no fine-tuning / robust across domains with a linear probe" — **refuted (0-3)**. On *large in-domain* data (ADE20K) full fine-tuning (62.9) actually beats frozen+adapter (60.2). So "freeze = better" is **not universal** — it's the right bet for our **small-data, cross-domain** regime, but it's a **hypothesis to test on the held-out 5**, not a guarantee.
- "DINOv2-registers is a strictly-better drop-in" — **refuted**. Try it, don't assume; and it changes the arch (breaks the shared `.pth`).
- "DINOv3 transfers to medical/endoscopy" — **refuted (0-3)**: no evidence DINOv3 specifically helps surgical seg.
- A specific SegDINO-beats-supervised Kvasir number — refuted (the *frozen-foundation > from-scratch* direction is still plausible, just that exact claim didn't hold).

## 3. Options, weighed
| Option | Keeps shared `.pth`? | Expected held-out gain | License | Effort / Colab |
|---|---|---|---|---|
| **A. Training fixes** (full-freeze + aug + episode-val + Dice) on natural DINOv2-ViT-B | ✅ yes | medium (0.34→~0.45-0.55, *to test*) | Apache-2.0 | low; T4-fine |
| **B. SurgeNetXL DINOv2-ViT-B backbone init** + the A recipe | ✅ **yes if keys+patch14 match** | likely **largest** (surgical domain; +14.4% vs ImageNet reported) | CC-BY-NC-SA (thesis-OK) | low-med (download+key check); T4-fine |
| C. DINOv2-with-registers-B + decoder head | ❌ (arch change) | small-med (cleaner dense feats) | Apache-2.0 | med |
| D. DINOv3 ViT-B/L (distilled) + decoder | ❌ | med dense, **no surgical evidence** | commercial | med; A100 for L |
| E. EndoDINO ViT-B (frozen+head) | ❌ unless ViT-B keys match | small (per its own polyp numbers) | CC-BY-4.0? weights TBC | med |
| F. DeepLabV3+/SegFormer from scratch | n/a (not a DINO head) | likely **worse** cross-episode at 15 eps | — | med — contrarian baseline only |

**Key insight:** Option **B** is the sleeper — SurgeNetXL provides a DINOv2-ViT-B, so swapping our backbone *init* (natural→surgical) leaves the `DINO2SEG` architecture and checkpoint keys unchanged → **still loads into SNI/SemGauss**, just with surgical-pretrained weights. That's the only way to get the domain-pretraining benefit *without* breaking the one-shared-`.pth` design. **Verify first** that SurgeNetXL's DINOv2-ViT-B is patch-14 and its state_dict loads into our `vit_base` (strict=False, ~0 unexpected); if it's ViT-B/**16**, it breaks compat (and SNI/SemGauss expect /14).

## 4. Recommended experiment ladder (each measured on the held-out 5)
**Phase A — compat-preserving, on natural DINOv2-ViT-B (the 4 in-pocket levers):**
1. **Full-freeze backbone, train only `segmentation_conv`** (faithfulness fix + #1 lever; 1-line `requires_grad=False`). 
2. **+ Heavy photometric/style aug** (brightness/contrast/saturation/**hue**/gamma/gain/noise + h-flip/scale/crop) — attacks the appearance gap directly.
3. **+ Episode-level val** (leave 2 *training* episodes out) so model-selection tracks generalisation, not memorisation.
4. **+ Dice/Lovász + inverse-freq class weights** (rescue Tool @0.20).

**Phase B — compat-preserving backbone upgrade:**
5. **Re-init backbone from SurgeNetXL DINOv2-ViT-B** (after the patch-14/key-load check), keep the Phase-A recipe. Expected biggest jump if surgical features transfer to colorectal CRCD.

**Phase C — only if A+B fall short (accept per-method heads / arch change):**
6. Better head (multi-scale / DPT / Mask2Former decoder) on frozen features.
7. DINOv3-ViT-B/L distilled (+decoder) — commercial license, no surgical evidence.
8. DeepLabV3+/SegFormer from scratch — run once as the contrarian baseline.

## 5. Honest ceiling
With **15 correlated episodes** and a hard appearance gap across surgical episodes, the realistic held-out ceiling is **~0.5-0.65 mIoU** even with the full stack — SurgeNetXL's "+14.4% vs ImageNet" is a *relative* gain over a weak baseline, not a promise of high absolute cross-episode mIoU on a 4-class colorectal task, and no training trick manufactures unseen-anatomy coverage. The biggest single lever is the **surgical backbone (B)**; beyond that it's a **data-diversity limit**. If a strong generalisation *result* is required for the thesis, that points to either more episodes or framing 0.34→~0.55 as "domain-pretraining + frozen-features substantially close, but don't eliminate, the cross-episode gap" (an honest, citable finding).

## 6. Reminder of scope
The seg head only matters for **SNI/SemGauss** (SGS uses GT masks). Its quality is a **second-order input** to the SLAM metric (semantic loss weight 0.05-0.1), so the SLAM benchmark may be nearly indifferent — run Phase A (cheap) first, wire the head in, and let the SLAM metric say whether to invest in Phase B. Deployment for the benchmark can still be the **in-domain** head (trained incl. the eval scenes, baseline convention); this whole plan is about whether a *generalising* head is worth pursuing as a side result.
