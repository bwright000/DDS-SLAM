# SoTA: Articulated Tools + Uncertainty Mapping + Scene Decomposition in Surgical GS-SLAM — and Our Open Niche

*Survey synthesis, 2026-06-23. Adversarial novelty audit across four cross-prong literature scans. Personality note: claims below are stated plainly — where something is already done, it says so.*

---

## 1. BOTTOM LINE

**Is the thesis direction novel?** YES — but only as a *specific intersection*, not as any single component. The defensible novelty is one sentence:

> *Online surgical Gaussian-Splatting SLAM that isolates instruments as **separate articulated objects** (per-part SE(3) + learnt joint angles), learnt **self-supervised from monocular endoscopic video** (no CAD, no synthetic joint labels), with **per-pixel σ²-weighted mapping** driving a **static-tissue / deforming-tissue / tool** decomposition.*

No published method (as of mid-2026) combines all of: (a) online/SLAM operation, (b) tool-as-separate-articulated-object, (c) CAD-free self-supervised part+joint learning, (d) from continuous monocular surgical video, (e) σ²-driven decomposition. Every survey's verdict converges on **NOVEL by combination**, **NOT novel by component**.

**The single biggest novelty risk** is the collision with two papers that each own a *large fraction* of the claim:

- **Instrument-Splatting / Instrument-Splatting++** (arXiv:2503.04082, 2603.22792) already does *surgical articulated instrument reconstruction with forward kinematics, per-part SE(3), and joint-angle estimation*. Our ONLY differentiators are: **(i) CAD-free / self-supervised** (they need CAD geometry + predefined kinematic chain + synthetic gripper-tip semantic supervision), and **(ii) online/SLAM-integrated, decomposing tools-from-tissue in a unified map** (they are offline, single-instrument, on pre-segmented video). If a reviewer reads "articulated surgical tool in Gaussians" as the headline, we lose — the differentiation MUST be CAD-free + online + unified-map, stated up front.
- **ArticulatedGS / SplArt / GaussianArt** (arXiv:2503.08135, 2506.03594, 2508.14891) already do *self-supervised CAD-free part-segmentation + joint-angle learning* — the exact "learning mechanism" we'd claim — but **offline, on posed multi-view, non-surgical, two static articulation states**. If "self-supervised articulation learning" is claimed as the *mechanism* novelty, we lose; the surgical/online/streaming-video setting is the only safe ground.

**Net:** the contribution is real but *thin per-axis*. It lives entirely in the **conjunction** and in **"continuous monocular surgical video, online"**. The thesis must be defended as a systems-integration + domain-transfer contribution under a genuinely harder data regime (continuous deforming-tissue context, no two-state captures, no CAD), NOT as a new learning mechanism or a new uncertainty formulation.

---

## 2. SoTA LANDSCAPE

| Method | Venue / Year | What it does | What it lacks vs us | Source |
|---|---|---|---|---|
| EndoNeRF | 2021–22 (benchmark) | Dual-field NeRF, tool-mask ray exclusion; pulling/cutting datasets | Masks tools entirely; no tool object/pose/articulation; offline, slow | benchmark; rev. arXiv:2502.14886 |
| SurgicalGaussian | MICCAI 2024 | Deformable 3DGS tissue, tool-mask-guided training | Hard-masks tools; no tool object, no σ² weighting | arXiv:2407.05023 |
| Deform3DGS | MICCAI 2024 | FDM basis-function tissue deformation; spatio-temporal tool-occlusion masks | Tool masking not reconstruction; no uncertainty; no articulation | arXiv:2405.17835 |
| EndoGaussian(s) | 2024 | Real-time deformable-tissue GS (HGI + SGT) | Tools = background occlusion; no tool object/pose/joints | arXiv:2401.12561 / .13352 |
| Endo-4DGS / FeatureEndo-4DGS | MICCAI'24 / IJCAI'25 | 4D tissue + depth priors; semantic feature distillation | Semantics on tissue not tools; no tool recon; no σ² map weighting | arXiv:2401.16416 / 2503.06161 |
| EndoGSLAM | MICCAI 2024 | Real-time dense endoscopic GS-SLAM, >100 fps | Tissue-only; tools masked; no decomposition/uncertainty | arXiv:2403.15124 |
| **Instrument-Splatting (++)** | 2025 / 2026 | **Articulated** instrument twin: shaft+wrist+gripper, FK, per-part SE(3), joint angles via render-and-compare; SAPET + Robust Texture Learning | **CAD-dependent + predefined kinematic chain + synthetic semantic labels; OFFLINE; single tool; pre-segmented; not unified map** | arXiv:2503.04082 / 2603.22792 |
| T²GS | MICCAI 2025 | Scene decomposition: tissue (deform) vs **tool** (global rigid traj + local shape) | Tool motion = rigid+local shape, **NO articulation / joints / per-part SE(3)** | MICCAI 2025 |
| EndoLRMGS | Mar 2025 | Tissue via GS + tools via LRM; tool pos/scale via OPjPO | No tool pose/kinematics/articulation; modules not integrated; no online decomposition | arXiv:2503.22437 |
| SurgPose | 2025 | ~120k articulated-tool instances, 7 UV-keypoints/instance | Dataset/keypoints only; no dense recon/appearance | arXiv:2502.11534 |
| SurgiPose | IROS 2025 | Tool **kinematics from monocular video** via diff. rendering, no GT kinematics | Needs tool GEOMETRY model; no GS; no learnt topology; single robot | arXiv:2512.18068 |
| NRGS-SLAM | Feb 2026 | Monocular non-rigid endoscopic SLAM; per-Gaussian deformation prob. via Bayesian self-supervision | **Per-primitive, tissue-only; NO tool object, no SE(3), no joints; binary routing ≠ graded σ²** | arXiv:2602.17182 |
| WildGS-SLAM | CVPR 2025 | Per-pixel uncertainty (MLP+DINOv2) weights BA tracking + map loss for dynamic removal | Non-surgical; predicted (not learnt-per-Gaussian) σ²; no tools; no decomposition | arXiv:2504.03886 |
| VarSplat | CVPR 2026 | Learns **per-splat σ²** via law-of-total-variance; renders per-pixel uncertainty for tracking/loop | Appearance variance only; not in map photometric weighting per se; no tool tracking; non-surgical | arXiv:2603.09673 |
| RU4D-SLAM | CVPR 2026 | Semantic-guided per-pixel uncertainty reweighting (track+map) for 4D dynamic scenes | Dynamic-region focus; no tool/articulation; mechanism under-specified | arXiv:2602.20807 |
| CG-SLAM | ECCV 2024 | Depth-uncertainty to **select** Gaussians (filter, not weight) | Excludes uncertain regions; not principled σ² loss weighting; no tools | arXiv:2403.16095 |
| NeRF On-the-go | CVPR 2024 | DINOv2 per-pixel uncertainty for distractor removal | NeRF not SLAM; masking not graded weighting; no deformation/tools | arXiv:2405.18715 |
| DeGauss / CAD-SLAM | 2025 | Dynamic-static decomposition via **rendering consistency** comparison | Binary dyn/static, not uncertainty-routed; not tissue/tool/static; non-surgical | arXiv:2503.13176 / 2505.19420 |
| DynaGSLAM | Mar 2025 | First online GS-SLAM with per-object dynamic tracking + motion flow (Kalman) | **Rigid** objects only; no joints/articulation; no tools; no self-sup part learning | arXiv:2503.11979 |
| SpeeDe3DGS | 2026 | GroupFlow: cluster Gaussians by motion → J shared SE(3) | Groupwise ≠ per-object; clusters mix motions; **no joint angles** | arXiv:2506.07917 |
| ArticulatedGS | CVPR 2025 | **Self-sup** part-seg + per-part SE(3) + joints from 2 multi-view RGB sets, no CAD/labels | **OFFLINE (9–12 min); two static states; posed multi-view; non-surgical; no SLAM/video** | arXiv:2503.08135 |
| SplArt | 2025 | Self-sup category-agnostic; per-Gaussian mobility ∈[0,1] for part-seg + joints | Offline; two-state posed RGB; non-surgical; no streaming/SLAM | arXiv:2506.03594 |
| GaussianArt / Part2GS / DeGSS / REArtGS(++) / FreeArtGS | 2025–26 | General articulated-object GS: per-part SE(3), joints, soft part affiliation (≤20 parts), motion-clustering, FK, monocular-RGBD (FreeArtGS) | All **OFFLINE batch**; multi-view or two-state; non-surgical; not SLAM; no real-time joint prediction | arXiv:2508.14891 / 2506.17212 / 2506.09663 / 2503.06677 / 2603.22102 |
| FreeGaussian | AAAI 2026 Oral | Annotation-free articulated control via flow derivatives; disentangles ego vs articulated motion | Offline; handheld RGB, non-surgical; needs optical-flow priors; no SLAM | arXiv:2410.22070 |

---

## 3. PER-CLAIM VERDICT (our three pillars)

### (i) Articulated, self-supervised, learnt tool-object — **NOVEL (by domain + setting), NOT by mechanism**
- **Already done — surgical + articulated:** Instrument-Splatting++ (2603.22792) models tools as articulated multi-part Gaussians with FK and per-frame joint-angle estimation. So "articulated surgical tool in Gaussians" is **DONE**. Our wedge is *not* articulation per se.
- **Already done — self-supervised CAD-free articulation:** ArticulatedGS / SplArt / DeGSS / GaussianArt learn part-seg + joints with no CAD, no labels. So "self-supervised joint learning" as a *mechanism* is **DONE** (offline, general objects).
- **Genuinely open:** the *conjunction* — CAD-free self-supervised articulated tool, learnt **online from continuous monocular surgical video**, where the prior offline methods' enabling assumption (two clean static articulation states, posed multi-view) is **unavailable**. No method does this. Verdict: **NOVEL**, defensible only as "CAD-free + online + continuous-video + surgical." Risk: high overlap; the claim must never be "we learn articulation self-supervised" (that's ArticulatedGS) but "we do it **without two-state captures, in a streaming SLAM loop, amid deforming tissue**."

### (ii) σ²-weighted mapping — **CONTRIBUTION, but borderline / soft — NOT a headline; closer to table-stakes-in-progress**
- **Per-pixel uncertainty-weighted map/track loss is EMERGING-STANDARD**, not novel: WildGS-SLAM, RU4D-SLAM, NeRF On-the-go all do predicted per-pixel uncertainty weighting; VarSplat learns per-splat σ². The *weighting trick itself is standard ML* (Kendall-Gal heteroscedastic NLL).
- **What is NOT yet standard:** in **surgical** GS-SLAM the established practice is **hard tool masking** (SurgicalGaussian, EndoGSLAM, Deform3DGS, Endo-4DGS) — so replacing masking with a learnt graded σ² *is* a delta vs surgical baselines. And **uncertainty-DRIVEN decomposition** (σ² routes uncertain→dynamic/tool) is not in the literature (DeGauss/CAD-SLAM decompose by rendering-consistency, NRGS by Bayesian deformation prob — neither is σ²-routing).
- **Verdict:** σ²-weighted mapping alone is **NOT a defensible standalone contribution** — too many 2025–26 papers do per-pixel/per-splat σ². Claim it only as **(a) the surgical replacement for hard masking, and (b) the routing signal for decomposition.** Do not headline it; frame as enabling machinery. **Adversarial flag:** VarSplat (CVPR 2026) + WildGS-SLAM make a pure "we add uncertainty to the map loss" claim indefensible by the time of submission.

### (iii) static / tool / tissue decomposition — **NOVEL vs NRGS, but EndoLRMGS + T²GS narrow it**
- **NRGS-SLAM** decomposes at the **per-Gaussian primitive** level (rigid vs deform, Bayesian gating) — **NOT object-level, NOT tool-vs-tissue, NO tool SE(3)/joints**. So vs NRGS our object-level tool decomposition is clearly distinct.
- **BUT two surgical papers already do tissue-vs-tool decomposition:** **T²GS** (MICCAI 2025) explicitly separates deformable tissue from interacting tools (rigid traj + local shape) — *minus articulation*. **EndoLRMGS** separates tissue (GS) + tools (LRM) — *minus pose/kinematics*. So "decompose surgical scene into tissue and tool" is **partially DONE**.
- **Our delta:** a **three-way** static-tissue / deforming-tissue / tool split where the tool branch is **articulated (per-part SE(3)+joints)** AND the split is **driven by σ² uncertainty** rather than seg masks or rigid-trajectory heuristics, online. Verdict: **NOVEL as a three-way articulated σ²-routed decomposition**, but T²GS is the citation that forces us to *not* claim "first to decompose tissue/tool" — only "first articulated, uncertainty-routed, online."

---

## 4. THE OPEN NICHE — precise contribution statement

**CLAIM THIS (defensible):**
> *We present the first **online surgical Gaussian-Splatting SLAM** that reconstructs instruments as **articulated separate objects** — learning part segmentation, per-part SE(3), and joint angles **self-supervised from continuous monocular endoscopic video**, with **no CAD geometry, no predefined kinematic chain, and no synthetic joint labels** — and that uses a **learnt per-pixel σ² uncertainty** to **route** the scene into static-tissue / deforming-tissue / tool layers within a single online map.*

The four pillars to lean on, in order of defensibility:
1. **CAD-free, label-free articulated tool learning** (vs Instrument-Splatting++ which needs CAD + FK chain + synthetic semantics). *Strongest single differentiator.*
2. **Online / streaming-video / SLAM-integrated** (vs all offline articulated-GS: ArticulatedGS, SplArt, GaussianArt, FreeArtGS, Instrument-Splatting). *Strong; the whole offline field's two-state assumption breaks here.*
3. **Continuous monocular surgical video amid deforming tissue** (vs two-state posed multi-view; vs general non-surgical objects). *Strong domain-transfer argument.*
4. **σ²-driven three-way articulated decomposition** (vs NRGS per-primitive, T²GS non-articulated, EndoLRMGS no-pose, DeGauss/CAD-SLAM consistency-not-uncertainty). *Moderate; depends on showing routing beats masking.*

Optional 5th if it survives implementation: **topology changes** — tools entering/exiting, jaw open/close triggering Gaussian spawn/prune. **No prior surgical or articulated-GS work handles this.** If demonstrated, it is the cleanest unclaimed sub-niche; if not, drop it silently.

**DO NOT CLAIM (will be refuted by a reviewer):**
- "First articulated surgical tool in Gaussians" → Instrument-Splatting++ owns it.
- "Novel self-supervised articulation-learning mechanism" → ArticulatedGS / SplArt / DeGSS own the mechanism.
- "First uncertainty-weighted GS-SLAM mapping" → WildGS-SLAM, VarSplat, RU4D-SLAM own it.
- "First per-Gaussian rigid/deform routing" → NRGS-SLAM owns it.
- "First tissue/tool scene decomposition in surgery" → T²GS / EndoLRMGS own it (non-articulated / no-pose, but they own the headline phrase).
- "Novel σ² formulation" → it's Kendall-Gal heteroscedastic NLL; standard.

**The defense in one line:** *novelty = the conjunction under the hardest data regime (CAD-free, online, continuous monocular surgical video) — not any isolated mechanism.* Measurability must be a direct articulation/decomposition metric (per-part SE(3) error, joint-angle error, tool-pose ATE, held-out tool/tissue PSNR split), because render-PSNR alone will not isolate the contribution.

---

## 5. ADOPT vs BUILD

**ADOPT (portable machinery — solved elsewhere, do not reinvent; cite and reuse):**
- **Forward kinematics + per-part SE(3) + render-and-compare joint-angle refinement** — standard in Instrument-Splatting++, Part2GS, REArtGS. Portable as-is.
- **Self-supervised part segmentation via motion/trajectory clustering** — DeGSS (trajectory descriptors), ArticulatedGS/SplArt (mobility param ∈[0,1], consistency). Adopt the *clustering + per-Gaussian mobility* idea; adapt to streaming.
- **Per-object SE(3) online tracking** — DynaGSLAM (Kalman per-object, joint ego+object motion). Adopt the online per-object tracking scaffold; add joints on top.
- **Per-pixel / per-splat σ² head** — WildGS-SLAM (MLP+DINOv2 predicted), VarSplat (learnt per-splat via law-of-total-variance). Adopt one; do not invent a new σ² formulation (it's Kendall-Gal).
- **Groupwise rigid-motion distillation** — SpeeDe3DGS GroupFlow, as an efficiency layer for the tool's rigid parts.
- **Deformable-tissue backbone** — existing surgical GS (Deform3DGS basis functions, EndoGaussian SGT) for the tissue branch; do not rebuild tissue deformation.

**BUILD (genuinely net-new — this is the thesis):**
- **Online streaming adaptation of self-supervised articulation** — turning two-state offline part/joint learning (ArticulatedGS) into a continuous monocular-video, single-pass SLAM estimator. *This is the core engineering+research contribution.*
- **CAD-free tool topology induction from surgical video** — discovering shaft/jaw part structure without CAD or synthetic semantic supervision (vs Instrument-Splatting++'s gripper-tip net).
- **σ²-routed three-way decomposition** — using learnt uncertainty (not seg masks, not rendering-consistency) to assign Gaussians/pixels to static / deforming / tool branches, online.
- **Joint articulation-in-deforming-context disentanglement** — separating tool joint motion from surrounding tissue deformation (a confound absent in all non-surgical articulated-GS, and avoided by EndoLRMGS/T²GS which don't do joints). FreeGaussian's flow-based ego-vs-articulation disentanglement is the nearest prior idea to *adapt*, not adopt.
- **Topology-change handling (spawn/prune on tool enter/exit, jaw open/close)** — unclaimed anywhere; build only if time permits.

---

## 6. READING LIST (must-read, 5–8)

1. **Instrument-Splatting++** — arXiv:**2603.22792** — *the direct surgical competitor*; read in full to nail the CAD-free + online + unified-map differentiation. Highest priority.
2. **ArticulatedGS** — arXiv:**2503.08135** (CVPR 2025) — the self-supervised CAD-free part+joint mechanism we must out-position (offline, two-state). Defines our "online streaming" wedge.
3. **NRGS-SLAM** — arXiv:**2602.17182** — closest surgical SLAM; per-Gaussian Bayesian deformation routing. Establishes the "object-level vs primitive-level" boundary.
4. **WildGS-SLAM** — arXiv:**2504.03886** (CVPR 2025) — the uncertainty-weighting template (DINOv2 + per-pixel σ²); shows σ²-weighting is not novel. Our Inc-1/Inc-2 lineage already mirrors this.
5. **T²GS** — MICCAI 2025 (arXiv pending) — surgical tissue/tool decomposition *without* articulation; the citation that bounds our decomposition claim.
6. **VarSplat** — arXiv:**2603.09673** (CVPR 2026) — learnt per-splat σ²; confirms uncertainty mapping is becoming table-stakes — do NOT headline σ².
7. **SplArt** — arXiv:**2506.03594** — second self-supervised articulation method; per-Gaussian mobility ∈[0,1], useful machinery to adopt.
8. **DynaGSLAM** — arXiv:**2503.11979** — the online per-object GS-SLAM scaffold to build joints onto (rigid-only today).

*Honorable mentions if space:* EndoLRMGS (2503.22437, tissue+tool minus pose), FreeGaussian (2410.22070, flow-based ego-vs-articulation disentanglement), SurgiPose (2512.18068, monocular tool kinematics, no GS).

---

*Adversarial summary: the thesis survives novelty scrutiny ONLY as a conjunction-under-hard-data claim. Each of the three pillars individually has a 2025–26 owner (Instrument-Splatting++ / ArticulatedGS for articulation; WildGS-SLAM+VarSplat for σ²; T²GS+NRGS for decomposition). Defend the intersection, lead with CAD-free + online + continuous-monocular-surgical-video, and measure with a direct articulation/decomposition metric — never render-PSNR alone.*