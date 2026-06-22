I have all four verification areas. The task is synthesis into one decisive report — no further investigation needed since the areas already contain the verified findings with sources. Let me integrate them.

# GS Migration — Verification Verdict

## 1. BOTTOM LINE

**GO-WITH-CONDITIONS** — the base (EndoGSLAM) is verified real, released, online-SLAM-with-unknown-pose, surgically validated code with full densification, and the uncertainty/flow/eval infrastructure ports cleanly, so Phase 0–1 is low-risk and has a defensible fallback (replicate the −23% ATE uncertainty win in GS); the single biggest reason this is *conditional* and not an unqualified GO is that **the headline thesis bet — online deformable GS — is RESEARCH, not a port: Deform3DGS is verified offline/batch with global-temporal [0,1] parameterization and known-pose, no released RGB-D online deformable GS-SLAM exists to clone, and the topology-change "GS beats SDF on tool ghosting" render win is mechanically plausible but EMPIRICALLY UNPROVEN on any surgical tool-occlusion benchmark.**

Proceed only if the team accepts a 4–8 week Phase-2 research milestone (not a feature flag) with Phase-1 uncertainty as the guaranteed-shippable result.

## 2. THE FOUR GATING QUESTIONS

**Q1 — EndoGSLAM + MoGe-2 viable as the base? → YES, with a known depth-scale tax (VERIFIED).**
EndoGSLAM is true online SLAM with `use_gt_poses=False` (config `c3vd_base.py`), an independent tracking phase (render-and-compare photometric+depth L1, Adam on `cam_unnorm_rots`/`cam_trans` only, Gaussians frozen) and a mapping phase (pose frozen) — *verified by code trace* in `scripts/main.py`. It is publication-grade (modular loaders, eval harness `calc_metrics.py`, configs, viz), MICCAI 2024, code released (`Loping151/EndoGSLAM`). It accepts arbitrary depth (no source validation). **The catch (VERIFIED):** the depth term is L1 absolute `|depth_gt − depth_pred|`, which is *not* scale-invariant — MoGe-2's up-to-scale `α·depth_true` confounds fidelity with scale, exactly as it does in our NeRF history. Mitigation is the same proven move: per-snippet scale calibration + Sim3 ATE (never rigid). This is a tax, not a wall. *(Area 1 b/c; Area 4 Q-EndoGSLAM-fitness)*

**Q2 — Is the NRGS niche open? → YES, but narrowed and partly opaque (VERIFIED mechanism; ASSUMED scope).**
NRGS-SLAM (arXiv 2602.17182) IS online and routes rigid-vs-deformable via a learnable per-Gaussian scalar `w_d,i ∈ [0,1]` under Bayesian self-supervision `w*_d,i = σ(log(π_d/π_r) + β(Ē^R − Ē^D))` — no external labels. Claims "up to −50% RMSE" on StereoMIS/Hamlyn/C3VDv2 (NOT CRCD; monocular, depth synthesized via a MoGe-style model). **Code is NOT released** ("upon acceptance"; none found as of June 2026). The genuinely open gaps vs NRGS: (i) explicit per-Gaussian **RBF-basis** deformation (NRGS is a scalar probability, no basis); (ii) **object-level tool SE(3) lifetimes** (tools enter/exit, each its own rigid transform — unclaimed by NRGS *and* by every other 2025–26 surgical GS paper surveyed: FeatureEndo-4DGS, BridgeSplat, EndoFlow-SLAM, SurgicalGaussian, SAGS, Instrument-Splatting all stop at masking/segmentation). The weakest claim is "learned routing" per se — NRGS *also* learns a gate, just via Bayes on photometric residuals rather than σ²+seg+flow; do not headline routing alone. *(Area 2 a/b/c/d/e; Area 4 Q-NRGS-confidence)*

**Q3 — One rasterizer for all channels (RGB+depth+σ²+semantic+deform)? → YES, feasible, with ~1–2 days CUDA work (VERIFIED-with-caveat).**
Consolidate on the **local, complete SemGauss rasterizer** (`SemGauss-SLAM/diff-gaussian-rasterization-w-depth_sem_gauss/`), which already emits RGB[3]+depth[1]+semantics[16] (forward returns color/radii/depth/semantics; backward covers all four). Extend it with one optional per-Gaussian scalar `w_d` (alpha-blended like opacity, the slothfulxtx 34-attribute design proves ~26 attributes fits well under limit). σ² stays **image-space** (DINO→MLP post-rasterize, no kernel change) — so σ² and deform-routing do not conflict at the kernel level. WildGS's `dyn_uncertainty/` is self-contained and liftable; the WildGS w-pose rasterizer submodule is uninitialized and should be **skipped** (use SemGauss + WildGS uncertainty *logic*). sm_80: WildGS `setup.py` already carries `-gencode=arch=compute_80,code=sm_80`; SemGauss is silent and will likely need that line added + a ~10–15 min Colab compile. *(Area 3 — all findings)*

**Q4 — Online-deformable integration: port or research? → RESEARCH (VERIFIED, decisive).**
Deform3DGS is **definitively offline/batch**: it normalizes the whole video to a global `[0,1]` temporal basis (all frames upfront), assumes pre-computed intrinsics+extrinsics, ~60 s train per clip — *verified via paper fetch*. It cannot stream and cannot drop into EndoGSLAM's online loop without (a) windowed/local temporal basis, (b) joint pose+deformation backprop re-derivation, (c) streaming init. EndoGSLAM itself lists deformation as future work. The migration plan's "port / ADAPT" label for Phase 2 is **misleading** — realistic cost 4–8 weeks (experienced) to 2–3 months (otherwise). NRGS routing logic is portable *as reference* but its code is unavailable. *(Area 4 Q-Deform3DGS, Q-Phase2-effort, Q-online-deformable-exists)*

**Q5 — Does GS actually fix the topology/tool-ghost? → PLAUSIBLE but UNPROVEN on surgical data (the load-bearing uncertainty).**
Densification (clone/split/prune) is fully implemented in EndoGSLAM (`slam_external.py:densify()`) and *can* spawn/prune Gaussians — the structural mechanism is real. TagSplat shows topology-aware Gaussian lifecycle (unbind on topology boundaries). BUT: (i) the literature documents real GS failure modes — floaters from split/clone randomness, overfit to sparse views, gradient densification failing in high-texture regions (Revising Densification); (ii) surgical GS papers (Diff2DGS, GauSTAR) resort to **inpainting/detection preprocessing** rather than native online topology handling; (iii) **NO published head-to-head shows GS beating SDF on a mid-sequence tool-entry/occlusion benchmark.** GS-vs-NeRF gains (EndoGaussians +5.7 PSNR over EndoNeRF) are on generic deformation data that does not stress tool entry/exit. The SDF "one slot per voxel → ghost" claim is mechanically sound but never measured in Co-SLAM/DDS-SLAM context. *(Area 4 Q-densification, Q-render-win)*

## 3. BLOCKERS & CAUTIONS

### BLOCKERS (would kill it if unmitigated)

| # | Blocker | Mitigation |
|---|---|---|
| B1 | **MoGe-2 up-to-scale depth vs EndoGSLAM's scale-sensitive L1 depth loss + Gaussian scale init.** If scale is uncalibrated, depth loss collapses to scale error and corrupts both tracking and Gaussian geometry. (Area 1 c) | Per-snippet scale calibration (empirically fit α minimizing loss) — proven in our NeRF history. Headline Sim3 ATE only, never rigid Horn. **Gate Phase 0 on this working before any deformation work.** |
| B2 | **Phase-2 online-deformable is research, not a port** (Q4). Mislabeled "ADAPT" hides a 4–8 wk milestone with real chance of failure (windowed basis + joint pose/deform optimization). (Area 4 Q-Deform3DGS/Q-Phase2) | Re-budget Phase 2 explicitly as research with a kill-criterion. Make Phase-1 uncertainty (−23% ATE replication in GS) the contracted deliverable so a Phase-2 stall is not fatal to the thesis. |

### CAUTIONS (need a fallback)

| # | Caution | Fallback / Mitigation |
|---|---|---|
| C1 | **EndoGSLAM LICENSE unconfirmed** (no LICENSE file; `license=null`; MIT/Apache only assumed; depends on Inria non-commercial research-license rasterizer). (Area 1 a) | Contact authors for explicit written grant; if silent, build directly on `diff-gaussian-rasterization-w-depth` as the licensed submodule. Resolve before any public release/thesis submission. |
| C2 | **Topology-change render win unproven on surgical tool data** (Q5). Risk of building Phase-3 on an unvalidated premise. | Run an early, cheap tool-entry A/B (GS-densify vs SDF) on CRCD/STIR-with-tool-masks as a *go/no-go probe before* committing to Phase-3 object-SE(3). If GS doesn't visibly win on tool sharpness, descope to render+uncertainty contributions. |
| C3 | **NRGS code unreleased + scope opaque** (Q2). Novelty could shrink if post-acceptance code already includes basis/object tracking. | Monitor for code release through Phase-0; if unavailable, cite as "concurrent work, no code for comparison." Headline the genuinely-open angles (RBF basis + object-lifetime tool SE(3)), not "learned routing." |
| C4 | **Deformation-teacher refactor understated** (Area 4 last finding). NeRF teacher emits Δx* per volumetric query; GS needs nearest-Gaussian selection + different backprop (~1–2 wks, not "unchanged"). | Budget 1–2 weeks. Replay buffer + gauge regauge are pose-space and carry over exactly; only the query→Gaussian selection logic changes. |
| C5 | **sm_80 rasterizer rebuild** — SemGauss CMake silent on arch; test-build only on Colab/A100 (local GTX 970 = sm_52, incompatible). (Area 3 sm_80) | Add `-gencode=arch=compute_80,code=sm_80` (pattern from WildGS `setup.py`); validate compile on Colab *before* the full port, allow ~15 min. |

## 4. DE-RISKED FIRST STEP (the smallest thing that proves the base on our data)

**Run unmodified EndoGSLAM on CRCD `c1_001` with MoGe-2 depth — Phase 0, no GS additions, no deformation.** This isolates B1 (depth scale) and Q1 (base viability) before any research commitment.

Concretely:
1. **Build** the SemGauss/EndoGSLAM rasterizer for **sm_80 on Colab** (verify compile + a single forward render) — clears C5 cheaply up front.
2. **Stage** CRCD-Published `c1_001` (the **360-frame** GT — *not* the stale 271-row local copy) + MoGe-2 depth, raw-left.
3. **Calibrate** the per-snippet depth scale α; confirm the L1 depth loss is not scale-dominated.
4. **Run** EndoGSLAM online (unknown pose) end-to-end.
5. **Measure (the arbiter):** Sim3 ATE (`sim3_ate.py`) + est/GT path-ratio + dominant-axis |Pearson| for tracking; render PSNR/SSIM/LPIPS; ship the 6-panel inline video. Compare directly against the live NeRF-SDF canon (base 3.15 mm / +uncert 2.43 mm; render ~28.5 PSNR).

**Pass criterion:** EndoGSLAM converges, produces a non-degenerate trajectory and renders, and Sim3 ATE lands in the same order of magnitude as DDS-SLAM-Base — i.e. the GS base is competitive on *our* data before we spend a week on anything else. **Fail → stop and reconsider the base; do not proceed to Phase 1.**

## 5. REVISIONS THE VERIFICATION FORCES

1. **Re-label Phase 2 "port" → "research milestone" (4–8 wk) with a kill-criterion.** This is the biggest correction. Deform3DGS is offline/global-temporal/known-pose; the online conversion is the contribution, so treat it as such in the timeline and risk register. *(forced by Q4)*
2. **Do NOT ship Deform3DGS's rasterizer or its batch loop.** Keep only the **FDM/RBF basis insight**; implement deformation inside the consolidated SemGauss rasterizer + EndoGSLAM online loop. Deform3DGS stays a *reference*, not a code dependency. *(Area 3 / Q4)*
3. **Standardize on the SemGauss rasterizer; drop the WildGS w-pose rasterizer.** Lift only WildGS's image-space `dyn_uncertainty/` MLP logic (σ² stays image-space, off the per-Gaussian path). *(Area 3)*
4. **Reframe the novelty statement away from "learned routing."** Lead with **RBF-basis deformation + object-level tool SE(3) lifetimes** (both unclaimed across NRGS and all 2025–26 surgical GS work); position NRGS as concurrent prior on scalar deformation-probability routing. *(Area 2 d; Area 4 Q-object-tracking)*
5. **Insert an early empirical tool-entry probe** (C2) as an explicit gate between Phase 1 and Phase 3, because the topology-change render win — the structural rationale for the whole migration — is currently unproven on surgical data. If it fails, the defensible thesis contracts to **GS + uncertainty (Phase 1) + honest negative on topology**, which is still publishable.
6. **No base change.** EndoGSLAM stands as the Phase-0 base (verified online, RGB-D, densification, surgical) — the only forced caveats are the license confirmation (C1) and the monocular-depth constraint (B1), both with known mitigations.

**Verified vs assumed, in one line:** *Verified* — EndoGSLAM online/unknown-pose/densification/code-released; Deform3DGS offline/batch; NRGS online Bayesian scalar routing + no code; SemGauss rasterizer channels; depth-L1 scale-sensitivity. *Assumed* — EndoGSLAM permissive license; NRGS exact datasets and full post-release scope; that GS densification cleanly beats SDF on surgical tool entry (the one bet the literature does not yet settle).