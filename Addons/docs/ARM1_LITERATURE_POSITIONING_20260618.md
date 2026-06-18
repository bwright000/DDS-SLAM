# ARM-1 LITERATURE POSITIONING (2026-06-18, workflow wkf02pcz6)

Uncertainty hedge (Inc-1 sigma^2 + Inc-2 pose down-weight, geo/dino/dino_reg) vs SoTA. 4-agent lit scan + synthesis.

---

This is a synthesis task, not a research-from-scratch task. The four primary-source readings are already comprehensive and cross-verified. My job is to consolidate them into the five deliverables the user asked for. I have everything I need — no tool calls required.

---

# Arm-1 Synthesis: Positioning, Improvements, Reading List, Analyses, Citations

## 1. POSITIONING — where Arm-1 honestly stands

**What is STANDARD (do not over-claim).** Your machinery is the textbook heteroscedastic-aleatoric template:
- The loss `0.5·(rgb_err²/σ² + log σ²)` is **literally Kendall & Gal (NeurIPS 2017)**; NeRF-W used the same NLL.
- Per-pixel σ² from a shallow MLP, inverse-variance down-weighting of **both mapping (Inc-1) and tracking (Inc-2)**, `clamp(1/σ²)` to bound the pose weight — this is the **WildGS-SLAM / NeRF On-the-go / UncLe-SLAM template**, point-for-point.
- Your v2 'dino' backend (compact grid + on-demand bilinear DINOv2 sample) is a **faithful WildGS-SLAM reimplementation**; `dino_reg` (registers) is the correct, current backbone hygiene step (Darcet ICLR 2024).

So the **architecture and use are standard**. Claiming novelty *there* is indefensible.

**The ONE deliberate divergence — and it is the source of your session finding.** You train σ² against the **raw photometric L2 residual**. Every strong method in this exact lineage deliberately does NOT: On-the-go uses **SSIM**, WildGS uses **SSIM + depth-consistency (Metric3D-v2)**, UncLe-SLAM uses the **depth residual**. The reason is the closed-form degeneracy On-the-go spells out: the NLL optimum is **σ ∝ ‖C−Ĉ‖**, so σ² is *definitionally* a readout of the photometric residual map → a contrast/edge/specular/motion detector. **Your observation (σ² on pins/tools/specular, not clean tissue) is the predicted artifact of the raw-residual teacher, not a quirk of your data.** This is good: it means your diagnosis is grounded in known theory, and the fix (change the teacher) is well-precedented.

**What is GENUINELY NOVEL (defensible white space).** Cross-checking all four readings, the unoccupied territory is consistent:
1. **No surgical method does explicit Kendall-Gal aleatoric σ² by photometric NLL on a neural-SDF SLAM substrate.** EndoSurf is the SDF cousin and has *zero* uncertainty; UW-DNeRF (your own lab, IRMVLab) has uncertainty but only on the **depth prior**, is a NeRF not SLAM, and does not touch pose.
2. **No surgical method couples one σ² to BOTH mapping AND tracking pose.** Hayoz down-weights pose but with *implicit declarative-network weights* (no NLL, no log-σ²); WildGS does both but in rigid RGB-GS where the goal is to *delete* dynamics.
3. **Nobody separates deformation-uncertainty from appearance/tool uncertainty.** NRGS-SLAM (the single closest threat) attempts camera-vs-deformation separation but stays **residual-self-supervised → cause-blind** (same conflation you diagnosed). The COMBINE (exogenous deformation teacher × seg-prior → 3-way TriGauge route) is the missing piece.

**Strongest 1-line defensible novelty claim:**
> *The first learned aleatoric uncertainty hedge (Kendall-Gal NLL) on a neural-SDF surgical SLAM that down-weights both mapping and tracking, with an honest diagnosis that a photometric-residual teacher yields a cause-blind robustness hedge — motivating a deformation-teacher-supervised, seg-prior-gated uncertainty that attributes variance to deforming tissue (TriGauge).*

Two honesty caveats to bake into the writeup: (a) **render is unchanged** — Inc-1 helps robustness/tracking, NOT appearance; do not claim PSNR gains. (b) Surgical motion is **sub-SNR** — always report Sim3 ATE + path-ratio + Pearson, never a lone ATE.

---

## 2. BEST MISSING IDEAS, ranked by expected value

### Arm-1 ALONE (no Arm-2 teacher needed — ship-able now)

**#1 — Swap the σ² teacher from RGB-L2 to SSIM (+ MoGe-2 depth-consistency). [HIGHEST EV, LOW EFFORT]**
- *What:* Replace `target = rgb_err²` with an **SSIM structural residual** (On-the-go Eq.) and/or a **depth-consistency residual vs your already-baked MoGe-2 depth** (WildGS uses Metric3D-v2 in this exact slot — for you it's basically free).
- *Source:* NeRF On-the-go (CVPR 2024); WildGS-SLAM (CVPR 2025); UncLe-SLAM (ICCVW 2023, depth-residual precedent).
- *Fit:* Drop-in, flag-gated default-off → base bit-identical. Directly addresses the contrast-detector failure with a *citable* fix and is the architectural bridge to Arm-2 (Δx* slots where WildGS puts depth-consistency).
- *Effort:* ~half a day (SSIM is in your eval already; depth residual reuses baked npy). n=3 A/B.

**#2 — Stop-gradient / decouple the σ² head from the loss it weights. [HIGH EV, LOW EFFORT]**
- *What:* Detach gradients between σ² and the SDF/map (Inc-1) and between σ² and the pose (Inc-2), so the map can't "buy down" its loss by inflating σ², and the tracker can't drive σ². β-NLL's stop-grad / On-the-go's detach / WildGS's explicit two-way detach.
- *Source:* Seitzer et al. β-NLL (ICLR 2022); Stirn et al. Faithful Heteroscedastic Regression (AISTATS 2023); Wong-Toi (2024); WildGS-SLAM detach.
- *Fit:* Prevents the explain-away collapse where σ² inflation on deforming tissue lets the map *stop reconstructing* it — directly protects the signal Arm-2 will need. Flag-gated.
- *Effort:* A few `.detach()` calls + an n=3 A/B to confirm no regression. Cheap insurance, well-precedented.

**#3 — Feature-space σ²-consistency regulariser. [MEDIUM EV, LOW-MEDIUM EFFORT]**
- *What:* Add `L_reg = mean over DINO-feature-neighbours of (β̄ − β(r'))²` — smooth σ² across DINOv2-similar pixels so it is spatio-temporally consistent, not per-frame flicker.
- *Source:* On-the-go Eq.3; WildGS `L_reg_V`.
- *Fit:* You already bake the DINO grid (v2/dino_reg). This is the property DINO was adopted *for* (multi-view consistency), and it's the most likely place v2 *actually* beats v1 — so pair it with a **σ²-temporal-consistency metric** (variance of σ² at corresponded points across frames) to measure the win the literature predicts.
- *Effort:* ~1 day. Only meaningful on the DINO arms.

**#4 — σ² inside the BA / per-correspondence, not a scalar pose loss. [MEDIUM EV, MEDIUM EFFORT]**
- *What:* WildGS weights the per-pixel reprojection term *inside the DBA* by 1/β²; DROID-SLAM learns per-edge confidence. Your Inc-2 is a coarser scalar down-weight.
- *Source:* WildGS-SLAM (CVPR 2025); DROID-SLAM (NeurIPS 2021).
- *Fit:* If your tracker exposes per-correspondence residuals, weighting *those* is the more faithful, likely-stronger form. Gated default-off.
- *Effort:* Medium — depends on tracker internals; worth a scoping read before committing.

### NEEDS the Arm-2 teacher (the COMBINE)

**#5 — Replace the deformation teacher: rigid-vs-deformable residual CONTRAST, NOT |Δx*| alone. [HIGHEST EV of the combine, the decisive design call]**
- *What:* Supervise σ² (or a deformation-confidence `w_d`) toward `Δ = Ē^rigid − Ē^deform` — render a second **deformation-off** pass (you already have `deformation_off` plumbing), and let the head fire where the *deformable* model lowers residual.
- *Source:* **NRGS-SLAM (arXiv:2602.17182, 2026)** dual-hypothesis Bayesian posterior + BCE; D²NeRF (NeurIPS 2022) static/dynamic routing.
- *Why it beats your planned |Δx*|:* Magnitude alone is **gauge-confounded** — large |Δx*| also comes from camera parallax leakage, depth noise, and tool motion. The rigid-vs-deformable contrast is **gauge-correct**: it lights up *only* where deformation is load-bearing for the data (deforming tissue), automatically rejecting (i) static tissue the camera explains and (ii) specular/tool pixels neither hypothesis explains. It is also **scale-robust** (compares two renders in image space) — critical given MoGe-2 is up-to-scale. Use |Δx*| at most as a *gate inside* the dual-hypothesis, never the raw regression target.
- *Fit:* Self-supervised → no STIR GT needed to *train*; STIR EPE stays your *evaluation* gate. Must be stop-gradded (see #2). Flag-gated default-off.
- *Effort:* Medium-high (second render pass + posterior/BCE). This is the centrepiece of the COMBINE — get the teacher right here and the TriGauge story follows.

**🚨 BLOCKING ACTION before any combine novelty claim:** the NRGS-SLAM PDF (arXiv:2602.17182) was read only from the abstract in two of the readings (fetch exceeded size limit). **Read its full deformation-probability loss before finalising novelty** — it is the single closest prior art to your COMBINE.

---

## 3. RANKED READING LIST (top first)

1. **NeRF On-the-go** (Ren et al., CVPR 2024, arXiv:2405.18715) — the closest prior art to Inc-1; contains the explicit `σ ∝ ‖error‖` degeneracy derivation that *explains your session finding*, and the SSIM-teacher + feature-consistency fixes (#1, #3).
2. **NRGS-SLAM** (arXiv:2602.17182, 2026) — your closest surgical threat AND template: dual-hypothesis deformation probability + tracking routing. **Read the full deformation-probability loss** (#5, blocking).
3. **WildGS-SLAM** (Zheng et al., CVPR 2025, arXiv:2504.03886) — your stated inspiration; the exact decoupled-teacher (SSIM+depth) + two-way detach template, σ² in both DBA and render.
4. **Hayoz et al.** (IJCARS 2023, arXiv:2304.08023) — the direct surgical pose-downweight precedent for Inc-2; note they use *implicit declarative weights, not an NLL* — that's your methodological difference.
5. **Seitzer et al. β-NLL** (ICLR 2022, arXiv:2203.09168) — formalises the explain-away collapse and the stop-gradient fix (#2); the theory backbone of your "why it detects appearance" claim.
6. **Darcet et al. Vision Transformers Need Registers** (ICLR 2024, arXiv:2309.16588) — the `dino_reg` basis; the *measurable* claim is 2.37%→~0% artifact tokens (NOT "halve"), norm>150 threshold.
7. **Kendall & Gal** (NeurIPS 2017) — cite for the NLL form + loss-attenuation = your robustness-hedge behaviour.
8. **UncLe-SLAM** (Sandström et al., ICCVW 2023, arXiv:2306.11048) — the dense-neural-SLAM NLL precedent (on depth); cite as the SLAM ancestor.
9. **UW-DNeRF** (IEEE TMI 2025, IRMVLab) — your own lab's surgical uncertainty-on-depth-prior; shows the lineage already buys "uncertainty on a prior helps," which you extend to the photometric residual + tracking.
10. **Jiang/Dravid et al. ViTs Don't Need Trained Registers** (NeurIPS 2025, arXiv:2506.08010) — test-time registers (zero retrain) = cheaper `dino_reg`; tempers expectations (+0.9 mIoU dense gain = modest).

(Secondary, as needed: EndoFlow-SLAM MICCAI 2025 — flow-as-motion-teacher, closest surgical analog to Arm-2; Surgical-DINO arXiv:2401.06013 + EndoDINO arXiv:2501.05488 — surgical-backbone evidence for a v3 ablation; D²NeRF NeurIPS 2022 — static/dynamic routing.)

---

## 4. ANALYSES TO RUN on current results (σ² maps + renders on Drive) — concrete + cheap

These test the readings' core claims against *your* data before you spend any A/B compute.

**A. σ²-vs-gradient-vs-motion correlation (tests the "appearance not deformation" diagnosis quantitatively). [cheapest, do first]**
- Per frame, compute: image gradient magnitude |∇I|, optical-flow / inter-frame motion magnitude, and (if available) a specular mask (top-intensity threshold). Correlate each against σ² (Pearson + point-biserial).
- *Predicted:* σ² correlates strongly with |∇I| and motion, weakly/nil with a deformation proxy. This is the figure that *proves* the session finding and motivates the whole COMBINE.

**B. Segment-stratified mean σ² {tool, tissue, bg}. [cheap]**
- Using your seg masks, report mean+var σ² per class for geo vs dino vs dino_reg.
- *Predicted:* high σ² on tool/specular (real residual), and dino_reg *lowers σ² speckle on flat tissue/bg without lowering tool σ²* = "registers remove false uncertainty, keep true uncertainty."

**C. Register artifact-token analysis on YOUR vits14 (tests Q1 — and may KILL the dino_reg arm). [cheap, do before any dino_reg A/B]**
- For each frame, take pre-head DINO patch-token grid (no-reg vs `_reg`). Compute per-token L2 norm; threshold at **150** AND at the data-driven histogram knee (vits14 absolute scale differs from ViT-g). Report **artifact fraction** per backbone.
- *Predicted by literature:* no-reg ≈ 2.37% → reg ≈ 0%. **But vits14 is small/less-trained → artifacts may already be ~0%, making dino_reg a no-op.** Measure before claiming; if ~0%, kill dino_reg honestly and save the A/B.
- Then **σ²-on-artifact correlation**: mean σ² inside vs outside the artifact mask (point-biserial) — direct evidence artifacts were polluting σ².

**D. σ² temporal-consistency metric (tests where v2/dino actually beats v1). [medium]**
- Variance of σ² at corresponded points (re-projected via est pose) across N consecutive frames. geo (live SDF) should flicker more; DINO (frozen, consistent features) should be steadier.
- *Why:* the literature says DINO buys *consistency*, not ATE. If v2 ties v1 on ATE but wins here, that's a real, citable contribution — measure the property you paid for.

**E. Per-region / rigid-vs-deformable render diff (dry-run of the #5 teacher). [medium]**
- Render a deformation-off pass on existing checkpoints; compute `Ē^rigid − Ē^deform` per pixel and overlay on seg + σ². Does the contrast localise on tissue and avoid tools/specular?
- *Why:* validates the proposed COMBINE teacher *before* building it. If the contrast is sub-noise on your sub-SNR motion, that's the STIR-gate signal — report the negative.

---

## 5. CITATIONS for the write-up (grouped by claim)

**The NLL form / aleatoric uncertainty (Inc-1/2 machinery is standard):**
- Kendall & Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 2017.
- Martin-Brualla et al. *NeRF in the Wild (NeRF-W).* CVPR 2021 (arXiv:2008.02268).

**Architecture & use template (per-pixel σ², down-weight tracking+mapping):**
- Zheng et al. *WildGS-SLAM: Monocular Gaussian Splatting SLAM in Dynamic Environments.* CVPR 2025 (arXiv:2504.03886).
- Ren et al. *NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild.* CVPR 2024 (arXiv:2405.18715).
- Sandström et al. *UncLe-SLAM: Uncertainty Learning for Dense Neural SLAM.* ICCVW 2023 (arXiv:2306.11048).
- Wang et al. *Uni-SLAM: Uncertainty-Aware Neural Implicit SLAM.* WACV 2025 (arXiv:2412.00242).
- Hu et al. *CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field.* ECCV 2024.
- Teed & Deng. *DROID-SLAM.* NeurIPS 2021.

**Why raw-residual σ² = appearance/robustness detector (the degeneracy + the fix):**
- Ren et al. *NeRF On-the-go.* CVPR 2024 (the `σ ∝ ‖error‖` derivation + SSIM teacher).
- Seitzer et al. *On the Pitfalls of Heteroscedastic Uncertainty Estimation (β-NLL).* ICLR 2022 (arXiv:2203.09168).
- Stirn et al. *Faithful Heteroscedastic Regression with Neural Networks.* AISTATS 2023.
- Wong-Toi et al. *Understanding Pathologies of Deep Heteroskedastic Regression.* 2024.

**Surgical precedent / domain positioning (the white space):**
- Hayoz et al. *Learning How To Robustly Estimate Camera Pose in Endoscopic Videos.* IJCARS 2023 (arXiv:2304.08023) — Inc-2 ancestor (implicit declarative weights, not NLL).
- *NRGS-SLAM: Monocular Non-Rigid SLAM for Endoscopy via Deformation-Aware 3D Gaussian Splatting.* arXiv:2602.17182, 2026 — closest combine prior art.
- Zhang et al. *UW-DNeRF.* IEEE TMI 2025 (IRMVLab) — uncertainty on depth prior; same-lab lineage.
- Zha et al. *EndoSurf.* MICCAI 2023 (arXiv:2307.11307) — SDF surgical cousin, no uncertainty.
- *EndoFlow-SLAM.* MICCAI 2025 (arXiv:2506.21420) — flow-as-motion-teacher.
- Wang et al. *EndoNeRF.* MICCAI 2022 — heuristic mask-importance (not learned uncertainty).

**Deformation teacher = rigid-vs-deformable contrast (the COMBINE design call):**
- *NRGS-SLAM.* arXiv:2602.17182, 2026 — dual-hypothesis posterior + BCE + routing.
- Wu et al. *D²NeRF: Self-Supervised Decoupling of Dynamic and Static Objects.* NeurIPS 2022.

**DINO backbone / registers (v2, dino_reg, v3 ablation):**
- Darcet et al. *Vision Transformers Need Registers.* ICLR 2024 (arXiv:2309.16588) — 2.37%→~0% artifact tokens, norm>150.
- Jiang/Dravid et al. *Vision Transformers Don't Need Trained Registers.* NeurIPS 2025 (arXiv:2506.08010) — test-time registers; modest +0.9 mIoU.
- Cui et al. *Surgical-DINO.* IPCAI/IJCARS 2024 (arXiv:2401.06013) — frozen generic DINOv2 insufficient for surgery.
- Dermyer et al. *EndoDINO.* arXiv:2501.05488, 2025 — surgical backbone, v3 ablation.
- Siméoni et al. *DINOv3.* Meta 2025 (arXiv:2508.10104) — registers + Gram anchoring, v-next.

---

## Honest scorecard
- **Behind:** your teacher is the one the whole field abandoned; no stop-grad; no feature-consistency reg; Inc-2 is coarser than WildGS's in-BA weighting. All four are *known, cheap, citable* gaps (#1–#4).
- **At parity:** architecture, NLL form, both-loop down-weighting, DINO/register backbone hygiene.
- **Ahead / novel:** surgical + neural-SDF + both-loop aleatoric hedge with an honest cause-blindness diagnosis, and the deformation-teacher × seg-prior TriGauge combine that even NRGS-SLAM lacks.

**One open question I cannot resolve from the readings — confirm before the dino_reg writeup:** is your `dino_reg` arm loading Meta's register-trained `_reg` checkpoint, or applying test-time registers to stock vits14? It decides the primary citation (Darcet 2024 vs Jiang/Dravid 2025) and whether your baseline artifact fraction is even nonzero. Analysis **C** answers it empirically.