# CRCD seg-head improvement plan — research synthesis + ladder (2026-06-19)

> Goal: raise the CRCD 4-class seg head's **held-out (unseen-episode) mIoU** (currently ~0.34; in-domain val ~0.93) on a FIXED ~15-episode train set, ideally keeping the shared DINOv2-ViT-B/14 `DINO2SEG` checkpoint that SNI-SLAM + SemGauss-SLAM both load.
> Source: deep-research `wd3lbqadr` (24 sources, 25 claims adversarially verified, 11 confirmed) + targeted follow-up fetches. ⚠️ The run hit a session limit: surgical-model specifics were partly unverified (abstained, not refuted) and re-fetched here; the auto-synthesis was done by hand below.

## 0. TL;DR recommendation
Do the **compat-preserving** work first (keep DINOv2-ViT-B/14, shared `.pth`): **full-freeze the backbone + heavy photometric/style augmentation + episode-level val + Dice/Lovász loss** (the 4 in-pocket levers). Then the single highest-leverage move that *still keeps compat*: **swap the backbone init from natural-image DINOv2-ViT-B to SurgeNetXL's DINOv2-ViT-B** (surgical-pretrained, same architecture → same checkpoint keys). Treat DINOv3 / registers / DeepLabV3+ as compat-breaking fallbacks only if that's not enough.

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
