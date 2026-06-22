# GS-base Contribution Prep — C1 / C2 / C3 onto EndoGSLAM (2026-06-22)

Self-contained build-prep for porting the three DDS-SLAM (NeRF) contributions onto the
Gaussian-Splatting base (EndoGSLAM). Metric-first, every addition flag-gated default-off
(base stays bit-identical), CRCD rectified + MoGe-2 up-to-scale, never headline bare ATE.
All code claims below are verified against the actual files (file:line cited).

---

## 1. Framing — the three contributions vs the live jitter finding

Phase-0 (c1_001, rigid EndoGSLAM): the GS base has **no deformation DOF**, so it absorbs
deforming tissue into **spurious camera motion** → camera **jitter** (Sim3-ATE 4.21 mm but
**path-ratio 93×**, ~8× of which is excess arc-length = jitter, not real travel). The three
contributions attack this from two sides and one map-side:

- **C1** (geo σ² → pose down-weight) and **C2** (camera-vs-scene flow gate) both **stop deforming
  pixels from dragging `cam_trans`** → cut excess-arc / path-ratio / ATE. C2 measures **motion**
  (the right signal for deformation); C1 measures **uncertainty** (a hedge/backbone).
- **C3** (feature-flow map consistency) is a **mapping-side** sharpener of the deform region.

The staged plan (build-order §3) is: **C2 first (clean image-space port, highest-EV, measures
motion directly) → C1 as the uncertainty backbone C2 modulates → C3 last (largely net-new /
research, earns its keep mostly in combine)**. Each is a single-variable A/B vs the **seed-0
pristine-EndoGSLAM baseline**, default-off, judged by whether **path-ratio collapses toward ~1×
with |Pearson|dom UP and render not regressed**.

**Verified shared hooks** (`EndoGSLAM/scripts/main.py`):
- `get_loss` def `198-200`; tracking transform `207-209` (`camera_grad=True, gaussians_grad=False`
  → tracking optimizes ONLY `cam_unnorm_rots`/`cam_trans`).
- Existing detached per-pixel depth-variance `uncertainty = depth_sq - depth**2` at `247-248`.
- Tracking residuals: depth `268` (`[mask].sum()`), rgb `275`/`277`; mapping depth `270`
  (`.mean()`), mapping rgb `279` (`0.8*L1 + 0.2*(1-SSIM)`).
- Loss aggregation `281-282` (`weighted_losses = {k: v*loss_weights[k]}`); `loss_weights` is a
  per-mode dict with NO `nll`/`featflow` key → adding a mapping-only key is parity-safe.
- **Tracking candidate-selection (CRITICAL, see C1 risk):** best-candidate kept by `loss <
  current_min_loss` (`677`), early-stop by `losses['depth'] < depth_loss_thres` (`693`). Both
  consume loss **magnitude**.
- `initialize_camera_pose` (const-velocity init) at `644` — the FIX "copy prev pose" is FREE.
- `initialize_new_params` `292-309` — Gaussians are explicit (means/rgb/rot/opacity/scale,
  optional SH `feature_rest`); **no per-point learned geometry latent** (the geo_feat problem).
- `get_loss` is **only ever called with `mapping=True`, never `do_ba=True`** (call sites `828`,
  `844`, `864`) → no BA stage re-optimizes a fixed past pose → **freeze_ba hook is a no-op**.

---

## 2. Per-contribution

### C1 — Geo σ² → pose down-weight (Inc-1 geo + Inc-2)

**Intended mechanism.** A learned per-pixel aleatoric σ² head; `w = clip(1/σ², w_min, w_max)
.detach()` multiplies ONLY the camera-pose tracking loss; σ² trained by a Kendall-Gal Gaussian
NLL `0.5*(err²/σ² + log σ²)` on the **mapping** residual (never co-optimized with pose — VarSplat
freeze-variance lesson). Targets the jitter directly: deforming pixels → high σ² → low weight →
they stop dragging `cam_trans`.

**GS hooks** (all gated `if unc_net is not None`, default-off keeps lines byte-identical):
- `main.py:get_loss` `198-200` — new default-None kwargs `unc_net, unc_features, nll_weight=0.0,
  detach_residual=False, w_min, w_max`.
- `main.py:268` / `:275,277` — multiply `w` **elementwise on the full `[1,H,W]`/`[3,H,W]` residual
  BEFORE `[mask]`** (the `[mask].sum()` flattens; multiplying after `.sum()` is a scalar no-op bug).
- `main.py:281` — `losses['nll']` added in the **mapping branch only** + `loss_weights['nll']=nll_weight`.
- `main.py:826`-region mapping call passes head + features so NLL trains; tracking call `665`
  passes head/weight. Head gets a **dedicated `torch.optim.Adam`** stepped beside the Gaussian
  `optimizer.step()` at `673` (mapping loop) — `initialize_optimizer` only builds from the param
  dict, so the head optimizer must be added explicitly.

**Lift vs build.** LIFT verbatim (image-space): WildGS `MLPNetwork`
(`WildGS-SLAM/src/utils/dyn_uncertainty/uncertainty_model.py:5-71`), the tracking down-weight
recipe `weights=0.5/u**2; where(<0.1,0); l1*=weights` (`WildGS-SLAM/src/utils/slam_utils.py:84-86`
= Inc-2's clip(1/σ²) in WildGS clothing), DINOv2 extractor (`.../mono_priors/img_feature_extractors.py:67-140`).
PORT-as-form (representation-agnostic): NLL + beta-NLL detach guard + softplus/floor from our
`DDS-SLAM/model/scene_rep.py:140,145,654-655,658`. BUILD net-new: the GS feature-source plumbing,
kwargs, head Adam, parity flag.

**Signal source (the GS analog of geo_feat) — THIS IS WHERE C1 IS HAND-WAVY; corrected design.**
There is **no SDF geo_feat in GS** (`initialize_new_params:292-309`). The fallbacks are NOT
interchangeable, and the verdicts (mechanistic + novelty) found the naive plan flawed:
- **v1 = rendered depth-variance `main.py:247`** is a **fixed, deterministic, detached**
  E[d²]−E[d]² along the ray (from `slam_helpers.transformed_params2depthplussilhouette`). It is an
  **occlusion/depth-EDGE detector, NOT a deformation detector** — smooth deforming tissue gets LOW
  variance → HIGH weight → keeps dragging the camera. It is also CG-SLAM-style **closed-form**
  variance (not learned, not aleatoric), so it both **mis-aims** at the jitter target and
  **collapses C1's novelty**. → **DROP as headline; keep only as a zero-param ablation baseline
  (a useful negative result).**
- **v1b = MLP-on-depth-variance** does NOT rescue v1: a per-pixel MLP whose only input is that
  scalar map is a monotone reparameterisation — no new information enters. Vacuous as written.
- **CORRECTED v1 (recommended): σ² head trained by the Gaussian-NLL on the photometric+depth
  RESIDUAL** — let σ² be learned from *where the rigid map fails to explain the frame*, which IS
  where deforming tissue lives. This is **VarSplat-style** (per-pixel σ² from photometric NLL),
  honestly framed, routed specifically to the **pose** loss in a deformable surgical setting.
- **v2 = image-space DINO→MLP** is the WildGS path → a **WildGS reproduction**; keep as
  ablation/fallback only.
- **Faithful geo analog (v3, DEFERRED):** a per-Gaussian geometric descriptor rendered via a
  Feature-3DGS / SemGauss N-D rasterizer (CUDA recompile). The ONLY version that supports the
  "σ² from geometry" claim; not buildable now.

**Default-off / parity.** `uncertainty.enable=false` → head never built, kwarg None, `nll_weight=0`
(belt-and-braces), every block `if ...is not None` → tracking keeps exact `[mask].sum()` and no
`nll` key → bit-identical. Head built LAST (separate RNG generator) for a clean single-variable A/B.
Validate with an Inc-0 parity gate (port `Addons/regression/test_inc0_bitidentical.py`):
RNG-after-build + param count + first-N-frame est-pose identical.

**Metric vs seed-0 baseline.** PRIMARY: **path-ratio + excess-arc + |Pearson|dom** (`Addons/eval/
sim3_ate.py`; NEVER bare rigid `output.txt` — it inverts on up-to-scale MoGe). SECONDARY (must not
regress): SemSup render PSNR/SSIM/LPIPS/L1-Depth ("hedge doesn't cost render"). n=3 seeds, clear
base seed-std. **Win-gate (hardened):** path-ratio↓ **AND |Pearson|dom non-decreasing** (the one
quantity a freeze-everything degenerate cannot fake) **AND ATE flat-or-down AND render not
regressed**. Reject any arm where path-ratio drops but Pearson is flat/down (= under-travel, not
de-jitter). Implement an arc/chord (= pathlen(est)/‖e[-1]−e[0]‖) jitter-specific metric (~3 lines
into `sim3_ate.py`), since "excess arc" currently has no estimator.

**Novelty position (honest).** The down-weight-by-σ² half is **fully SOTA-occupied**: WildGS-SLAM
(arXiv 2504.03886, CVPR'25, code released; DINO→MLP→σ² weighting tracking AND mapping), VarSplat
(2603.09673, per-splat σ²), UP-SLAM (2505.22335), CG-SLAM (2403.16095, ECCV'24, closed-form depth
variance → primitive selection, NOT pose-weighting). On GS the "σ² from geometry feature" axis
**dissolves** (no geo_feat): v1=CG-SLAM, v2=WildGS, v3 needs a deferred rasterizer. **Verdict
(adversarial-novelty): the standalone headline is REFUTED — a bare port is a WildGS reproduction
with a CG-SLAM signal on an NRGS-occupied surgical problem.** **DECISION: demote C1 to default-off
INFRASTRUCTURE — the σ²-weight backbone that C2's flow gate modulates. Headline C2.** Read
**NRGS-SLAM (arXiv 2602.17182)** in full before any novelty sentence (contemporaneous, surgical, GS,
cites DDS-SLAM + EndoGSLAM, solves the identical jitter via per-Gaussian deformation-probability;
C1's only surviving differentiator is continuous-aleatoric-σ²-on-pose, NOT "from geometry").

**Risks (with fixes folded in).**
- **Loss-magnitude corruption (NEW, verified, decisive):** multiplying `w` into the tracking
  residual rescales `loss` → silently corrupts `loss < current_min_loss` candidate pick (`677`) and
  the `depth_loss_thres` early-stop (`693`). **FIX:** use a **mean-preserving weight**
  (`w_norm = w / w[mask].mean()`) before the multiply, OR gate candidate-selection/early-stop on the
  UNWEIGHTED loss. Mandatory.
- VarSplat instability (live σ²+pose) → train σ² in mapping only, `.detach()` weight in tracking.
- Explain-away degeneracy (Seitzer ICLR22) → beta-NLL `detach_residual` guard.
- Depth-variance v1 mis-aims at edges not deformation → corrected v1 = residual-NLL.
- CRCD sub-SNR → lead with path-ratio/Pearson, n=3, ≥2 snippets.

**Effort:** medium (the head + NLL + kwargs are small; the signal-source decision is the research bit).

---

### C2 — Optical-flow-on-DINO camera-vs-scene pose gate (HEADLINE)

**Mechanism.** Per-frame RAFT dense flow vs a causal past frame → fit ONE camera-consensus model
(fundamental + Sampson, or homography for zoom-robust per-pixel residual) → pool residual into
DINOv2 patch regions → **region-level rigid-motion AGREEMENT** test producing two consumables:
(i) a HARD **FIX** (skip the pose optimizer, keep the const-velocity pose, still map) when camera
still OR scene deforming; (ii) a SOFT `[H,W]` weight down-weighting deforming pixels in the
tracking loss. The agreement decision (the WON DDS config) is the AND
`do_track = (cam_mag > cam_thresh) AND (frac_disagree_regions <= disagree_thresh)`.

**GS hooks** (signal half ports **byte-for-byte**; consumption is THINNER than NeRF — whole-image
render, no ray-gather):
- `main.py:get_loss` `198-200` — new default-None `track_weight`; at `268` use
  `(torch.abs(curr_data['depth']-depth) * w)[mask].sum()`, at `275` broadcast `w→[3,H,W]`; guard
  `if tracking and track_weight is not None`. `w = residual_to_weight(...).detach()`.
- **HARD FIX = simply wrap the optimizer block `650-706` in `if do_track:`** — do NOT add pose-copy
  code; the existing `initialize_camera_pose` (`644`) IS the copy.
- Gate runs once per frame just before the tracking loop (after `644`).
- `GSFlowGate` constructed in `rgbd_slam` init only if `cfg['flow_track']['enable']`.
- **DELETE the freeze_ba hook** — verified no-op (`get_loss` never called `do_ba=True`).

**Lift vs build.** LIFT whole, unchanged: `DDS-SLAM/Addons/motion/flow_track.py` (`_raft_flow`,
`_sampson`, `flow_residual`, `camera_motion`, `region_route` `128-190` [verified: region/pixel modes,
soft_scale ramp, all-zeros-on-fit-fail = neutral], `residual_to_weight` `193-200` [verified deadband
NOP]) — pure numpy/opencv/torchvision, zero coupling. BUILD net-new (small): the `GSFlowGate`
adapter (frame capture + causal `_flow_buf` keyed on `time_idx`), the `track_weight` kwarg plumbing.

**Signal source.** **No geo_feat needed — C2's key advantage.** Entirely image-space: 2-frame RGB →
RAFT → opencv F/homography → Sampson/reproj residual = camera-vs-scene split; DINOv2 tokens only
**pool** the residual. Survives the SDF→Gaussian migration untouched. Optional v2 enrichment
(deferred): analytic GaussianFlow from the Gaussians (cf. GaussianFlow-SLAM 2604.15612).

**Frame-source gotcha (verified, must get right):** EndoGSLAM yields color as a permuted/normalized
tensor. The adapter MUST capture color **pre-normalization** ([H,W,3] 0-255) and cv2-convert
**RGB→BGR**; wrong order silently corrupts the F-fit.

**Default-off / parity.** `flow_track.enable=false` → `flow_track.py` never imported (no RAFT/DINO
load), `GSFlowGate` never built, `track_weight` None → exact `.sum()` path, tracking loop unguarded.
Byte-identical; Inc-0 gate. Config: `gate` (false=soft-weight, true=FIX), `agreement`
(true=region DINO gate), `ref_stride 8`, `cam_thresh 2.0`, `disagree_thresh 0.2`, `n_groups 12`,
`deadband 3.0`, `alpha 0.5`, `w_min/w_max 0.1/1.0`. **Pin KMeans `random_state`.**

**Metric vs seed-0 baseline (hardened against the freeze-everything degenerate).** PRIMARY:
path-ratio toward ~1× **AND |Pearson|dom UP** (mandatory falsifier — a frozen trajectory has ~0
shape-correlation with GT, so Pearson rules out "suppressed all motion") **AND ATE flat-or-down**.
SECONDARY: render not regressed. **Add a const-velocity-copy (FIX-every-frame) NEGATIVE CONTROL —
C2 must BEAT it on Pearson + render**, proving discrimination not freezing. Reject any arm with
path-ratio < ~1× or FIX-rate > ~60% (under-tracking). A/B order:
**base → +soft-weight(A) → +agreement-FIX(B) → freeze-all control.** Tune thresholds on one
snippet, **validate on a held-out snippet** (single-snippet n=3 is train==test for the gate).

**Novelty position (the headline, but THIN — guard it).** Halves occupied: ego-motion-residual →
mask → pose is Flow4DGS-SLAM (2604.22339, RAFT+twist+MAD+pose-init, but global fit, per-pixel,
YOLOv9 not DINO, hard mask); mask-zeroes-BA is DG-SLAM (2411.08373, NeurIPS'24) + WildGS;
DINO→region→pose-weight is WildGS (appearance, not motion). **Surviving gap = REGION-LEVEL
RIGID-MOTION CONSENSUS over DINO-segmented regions → soft per-region pose weight + AND-gated FIX**
— neither Flow4DGS (geometric global fit, no learned regions) nor WildGS (appearance, no flow).
**Dominant threat = NRGS-SLAM (2602.17182):** contemporaneous surgical GS, same jitter problem,
cites DDS+EndoGSLAM, but per-Gaussian deformation-prob + sparse SpatialTrackerV2 + Bayesian energy —
mechanistically distinct from dense-flow × DINO-region consensus. EndoFlow-SLAM (MICCAI'25,
2506.21420) uses a RIGID flow loss that does NOT separate deformation from camera — the weakness C2
exploits. **Verdict (adversarial-novelty): sound-with-fix — only the agreement arm (Mode-3,
region) is the contribution; the soft-weight/magnitude arms and the DINO-FREE homography 'pixel'
fallback are NOT novel (Flow4DGS-occupied / DINO evaporates).**

**Risks (fixes folded).**
- **Attribution precondition (do FIRST):** confirm the Phase-0 path-ratio excess **correlates with
  deforming frames** (Sampson residual ↔ per-frame excess-arc), not a MoGe depth-scale artifact —
  else C2 is mis-aimed. Soundness precondition, not a secondary risk.
- **Novelty collapse if the headline arm is soft-weight or DINO-free pixel-mode** → headline ONLY
  Mode-3 region-consensus; add a **faithful Flow4DGS (global-fit, no-DINO) ablation** — region
  consensus must BEAT it or there is no contribution.
- Over-gating (CRCD ~55% GT held, deformation pervasive) → most frames FIX → ATE explodes; prefer
  soft-weight where possible, tune `disagree_thresh` on held-out.
- F degenerate for forward/zoom endoscopic motion → homography 'pixel' mode fallback (report as
  ablation, not headline — it is DINO-free).
- Drift from FIX-then-map → const-velocity copy (not pure freeze) + `cam_mag` clause; check
  path-ratio does not collapse BELOW 1×.

**Effort:** medium. **This is a clean image-space port + a thin consumption hook.**

---

### C3 — Feature-flow map consistency (largely NET-NEW / research)

**Mechanism (corrected).** A MAPPING-side regulariser: render a per-Gaussian DINO-feature map,
warp it by flow between a keyframe pair, penalise the warped-feature residual against the
destination DINOv2 target, **gated to fire only in the camera-rigid-DISAGREE (tissue) region**.
Makes the rendered feature field temporally consistent on specular/non-Lambertian tissue, where
EndoGSLAM's RGB loss (`main.py:279`) is weakest. The mapping-side **dual of C2** (reuses
`region_route`).

**WRONG-WARP FIX (decisive — folded from the novelty verdict).** Do **NOT** warp the deform-region
features by the **camera-rigid** flow (invalid exactly where C3 fires — the residual there is
dominated by un-modelled deformation). Instead: **warp by the FULL dense RAFT flow** (which already
contains the deformation) and use the camera-rigid residual ONLY as the **GATE/route_w** (where to
apply the term), not as the warp. This preserves the "deform-region map boost" story and avoids the
NeRF-C3 un-co-adapted-warp failure (E0: PSNR 25.83 < 27.70 base).

**GS hooks** (all default-off byte-identical):
- `main.py:get_loss` `198-289` — new default-None kwargs (`feat_target, flow_warp, route_w,
  featflow_weight`); `losses['featflow']` in the mapping branch before `281`.
- `main.py:initialize_new_params` `292-309` — per-Gaussian `feat` param (default-off: not added →
  param dict + RNG + state-dict keys unchanged) + add to mapping optimizer groups.
- `slam_helpers.transformed_params2rendervar` — sibling `..2featurevar` swapping color for `feat`
  (built only when on).
- Mapping call site (`826`-region) passes baked DINO target + precomputed flow/route_w.

**Lift vs build.** LIFT: C2's `flow_track.py` (region_route gate), DINOv2 extractor + the existing
baked corpus `dino/<stem>_dino.npy` (C=384), `Addons/dino/generate_dino_features.py`. **Start
N=3 PCA-DINO via the EXISTING SH rasterizer — NO CUDA recompile** (make this the default, not a
fallback); escalate to a Feature-3DGS N-D rasterizer only if N=3 is insufficient. BUILD net-new:
the warp-by-flow feature-residual loss + its rigid-disagree gating — **does not exist in DDS-SLAM
(only a flow+DINO ROUTER/diagnostic exists — `feature_flow_probe.py` assembles NO loss) nor in any
surveyed GS paper.**

**Signal source.** Rendered per-Gaussian DINO-feature map (GS analog of a learned geometry latent),
supervised against the **baked image-space DINOv2 target**. Motion signal = RAFT + camera-rigid
residual from `region_route` (same operand as C2, consumed on the mapping feature residual). NOT
the depth-variance (that is C1's).

**Default-off / parity.** `featflow.enable=false` → `feat` param not added, no feature rendervar,
`featflow_weight` None → exact `:270/:279/:281` lines, RAFT/DINO never imported. `featflow_weight=0`
second gate. New params appended LAST. Inc-0 parity gate.

**Metric vs seed-0 baseline (REORDERED — the global-render headline cannot isolate a region-gated
term).** PRIMARY: **region-restricted render PSNR/LPIPS/L1-Depth on the DEFORMING-TISSUE region**,
n=3, win clears the **region** seed-std (whole-frame averaging dilutes a real +1–2 PSNR on tissue
to <0.1 globally). **Define the eval region from an INDEPENDENT source (CRCD GT seg tissue/tool
mask, which the workspace has, or a held-out RAFT residual NOT consumed by the loss) — NEVER the
trained `route_w` (circular).** Report the rigid (control) region too: proof = gain-on-tissue AND
no-loss-on-rigid. Global render PSNR/SSIM = a HARD no-regression guard (NeRF-C3 precedent).
Sim3-ATE = no-HARM check only (never headline a mapping term on sub-SNR CRCD tracking). **Build
net-new eval tooling: `Addons/eval/eval_rendering.py` currently has NO region/held-out masking.**
Run the **RGB-flow vs feature-flow ablation** (same warp/gate) — this ablation IS the contribution
(defends "why feature not pixel"); without it the DINO half is unfalsifiable.

**Novelty position.** Genuine combination gap but **weakest of the three and heavily flanked.**
Camp A (flow regularises the GS map: GaussianFlow-SLAM 2604.15612, Flow4DGS 2604.22339, EndoFlow
MICCAI'25, EndoWave 2510.23087) — geometric flow, no DINO. Camp B (DINO→GS: Feature-3DGS 2312.03203,
FMGS 2401.01970, GSFF-SLAM 2504.19409 [explicitly decouples features from geometry], SemGauss
2403.07494, GauSSmart 2510.14270) — semantics, no flow. **The triple (flow residual)×(supervised by
DINO)×(boost the MAP) is UNOCCUPIED** (closest: GaussianFlow-SLAM has flow-regularises-map without
DINO; GauSSmart has DINO-supervises-map-quality without flow/per-Gaussian/warp; FMGS's DINO only
sharpens CLIP boundaries, own ablation 93.2→90.4). **Verdict (adversarial-novelty): sound-with-fix
— survives literally but thin; position explicitly vs NRGS-SLAM (2602.17182), not just
GaussianFlow/FMGS.**

**Risks (fixes folded).**
- Co-adaptation failure / wrong-warp (NeRF E0 precedent) → full-flow warp + soft route_w gate +
  rigid Gaussians untouched + HARD global-no-regression gate.
- Custom N-D rasterizer cost → start N=3 PCA-DINO via existing SH rasterizer.
- "Just GaussianFlow + a DINO loss" → the RGB-flow-vs-feature-flow ablation is mandatory.
- May not clear seed-std alone → **gate the C3 BUILD on C2 first winning**; declare the A/B as
  **base vs C2 vs C2+C3 with the marginal-over-C2 delta as the arbiter** up front (not retrofitted).
- F degenerate for forward/zoom → homography 'pixel' mode.

**Effort:** large / research.

---

## 3. Build ORDER (cheapest / highest-EV first, each metric-gated default-off)

| # | Contribution | Type | Why this slot |
|---|---|---|---|
| **0** | **Inc-0 parity harness + attribution check** | infra | Port `test_inc0_bitidentical.py`; add arc/chord + region-restricted eval tooling. **Confirm Phase-0 path-ratio excess correlates with deforming frames (Sampson↔excess-arc) — soundness precondition for C1+C2.** |
| **1** | **C2** | **clean image-space PORT** | Highest-EV, measures **motion** (the right signal for deformation→jitter). `flow_track.py` lifts byte-for-byte; consumption is one `track_weight` kwarg + wrapping the optimizer block in `if do_track:`. Headline contribution. Build soft-weight(A) → agreement-FIX(B) → freeze-all control. |
| **2** | **C1** | **RE-ROOT (signal source)** | The σ²-weight **backbone C2 modulates** (infra, not headline). Build CORRECTED v1 (residual-NLL head) + the loss-magnitude-preserving fix. Keep depth-variance v1 as a negative-result ablation, DINO v2 as a WildGS-reproduction ablation. |
| **3** | **C3** | **largely NET-NEW / research** | Gate the build on C2 winning. Start N=3 PCA-DINO via existing SH rasterizer; full-flow warp + rigid gate; region-restricted PSNR vs an independent mask; RGB-vs-feature-flow ablation. Earns its keep mostly in combine. |
| **4** | **COMBINE** | — | C1+C2 (backbone+gate), then +C3. Marginal-over-prior-arm deltas. |

Each step: one change → run FROM seed-0 base in isolation → did path-ratio↓ / |Pearson|dom↑ /
render-not-regressed clear base seed-std (n=3, ≥2 CRCD snippets, held-out threshold validation)? →
yes → keep. Ship the canonical 6-panel video + σ²/decision panels (two-diagnostic-sets standing rule).

---

## 4. Open decisions

1. **C1 signal source (the pivotal one):** corrected v1 = **residual-NLL head** (VarSplat-style,
   honest) vs deferring to the v3 per-Gaussian geometric descriptor (faithful "geo" novelty, needs a
   rasterizer). Recommendation: residual-NLL now, frame C1 as infra; v3 only if C1 must stand alone.
2. **Read NRGS-SLAM (2602.17182) in full BEFORE writing any novelty sentence** — it is the
   contemporaneous surgical-GS competitor for all three; current differentiation is abstract-level.
3. **C2 headline arm:** lock Mode-3 (region-consensus, `mode='region'` with DINO) as THE
   contribution; is the DINO-free homography 'pixel' fallback acceptable for CRCD forward/zoom motion
   while keeping the region claim alive, or does pixel-mode use forfeit the headline?
4. **C3 go/no-go:** build only if C2 wins. Accept N=3 PCA-DINO as sufficient, or budget the
   Feature-3DGS N-D rasterizer recompile up front?
5. **Attribution:** is the Phase-0 93× path-ratio deformation-absorption or partly a MoGe
   up-to-scale / scale-drift artifact? If the latter dominates, C1/C2 are mis-aimed and the fix is
   depth-scale, not these contributions.
6. **Threshold generalisation:** which held-out CRCD snippet is the validation set for C2's
   `cam_thresh`/`disagree_thresh` (single-snippet tuning is train==test)?

---

### Citations
WildGS-SLAM arXiv 2504.03886 (CVPR'25) https://arxiv.org/abs/2504.03886 · code
https://github.com/GradientSpaces/WildGS-SLAM ; VarSplat 2603.09673 https://arxiv.org/abs/2603.09673 ;
UP-SLAM 2505.22335 https://arxiv.org/abs/2505.22335 ; CG-SLAM 2403.16095 (ECCV'24)
https://arxiv.org/abs/2403.16095 code https://github.com/hjr37/CG-SLAM ; NRGS-SLAM 2602.17182
https://arxiv.org/abs/2602.17182 ; Flow4DGS-SLAM 2604.22339 https://arxiv.org/abs/2604.22339 ;
GaussianFlow-SLAM 2604.15612 https://arxiv.org/abs/2604.15612 code
https://github.com/url-kaist/gaussianflow-slam ; DG-SLAM 2411.08373 (NeurIPS'24)
https://arxiv.org/abs/2411.08373 ; EndoFlow-SLAM 2506.21420 (MICCAI'25)
https://arxiv.org/abs/2506.21420 ; EndoWave 2510.23087 https://arxiv.org/abs/2510.23087 ;
Feature-3DGS 2312.03203 https://arxiv.org/abs/2312.03203 ; FMGS 2401.01970
https://arxiv.org/abs/2401.01970 ; GSFF-SLAM 2504.19409 https://arxiv.org/abs/2504.19409 ;
SemGauss-SLAM 2403.07494 https://arxiv.org/abs/2403.07494 ; GauSSmart 2510.14270
https://arxiv.org/abs/2510.14270 ; NeRF On-the-go 2405.18715 https://arxiv.org/abs/2405.18715 .
