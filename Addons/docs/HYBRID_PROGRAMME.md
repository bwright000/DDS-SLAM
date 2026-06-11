# Hybrid Programme — Segmentation-Supervised Gauge Attribution for Deformable SLAM

Reconciles our seg-first build agenda with the best instruments from the "Unconstrained Research Programme"
document, scoped to a thesis (one researcher, RTX2080/Colab), grounded in the live literature (2026-06-10).

- Constructive spine: [[SEG_CONTRIBUTION_BUILD_LADDER]] (CP-Rig / TriGauge ladder).
- Research agenda: [[project_dds_seg_research_agenda_20260610]].
- StereoMIS reproduction reality: [[project_stereomis_dds_audit_20260610]].

---

## 0. Thesis — the OPEN slice (why this is novel after the literature check)

The problem is the **gauge non-identifiability** of pose vs deformation (DefSLAM's "floating map ambiguity"):
image motion is jointly explained by camera pose and scene deformation, non-separable along a per-frame rigid
orbit `(T_t, D) → (T_t·G_t⁻¹, G_t∘D)`. Our dead deformation field is this gauge collapsed (pose absorbs all
motion). Every adjacent line resolves the gauge with *some* anchor:

| Lineage | Gauge anchor | Why it doesn't cover us |
|---|---|---|
| DefSLAM / Lamarca (TRO'20) | isometric SfT + NRSfM, sparse | sparse features, not a dense field |
| DefVINS (arXiv 2601.00702, Jan'26) | **IMU** + conditioning-based DoF activation | endoscopy has **no IMU** |
| NRGS-SLAM (2602.17182) | photometric **self-supervision** (Bayesian, no labels) | unsupervised; conflates moving-rigid tool |
| WildGS-SLAM (CVPR'25) | DINO-uncertainty → **deletes** distractors | discards what we must model |
| KillingFusion (CVPR'17) | isometry (Killing-field) regulariser | a prior, not a learned attribution |

**The unclaimed slice: dense neural-field, monocular (no IMU), tissue-SEGMENTATION-SUPERVISED gauge-fixing.**
Supervised semantics is the one anchor nobody has used — and the anchor NRGS explicitly calls "unavailable."
That is our contribution axis. (The "Unconstrained" doc demoted segmentation to one prior among many; the
literature says that is backwards — theory/benchmark ground is contested, the seg slice is open.)

**Contribution = three legs:**
1. **Diagnosis** — the gauge collapse, direction-reversed (deformation *under*-explains), verified + byte-identical to release.
2. **Construction** — the seg-supervised gauge fix: **CP-Rig** (supervise the rigidity NRGS can't) + **TriGauge**
   (model the moving-rigid tool NRGS structurally can't).
3. **Measurement** — a controlled testbed with GT decomposition + gauge-aware metrics that make the win *provable*
   (not an ATE number that moves while the field stays dead).

---

## 1. Positioning / framing (theory at the right altitude — folded, not a separate paper)

Frame the method against the classical lineage; state the gauge result as an **empirically-validated condition**,
NOT a standalone TRO theorem (DefVINS already published the IMU version of that observability/conditioning analysis):
- **TriGauge ≈ the dense, no-IMU, semantically-supervised analogue of OC-EKF** (Hesch/Huang/Mourikis/Roumeliotis):
  information-restricted pose updates along the unobservable (deformable/tool) directions, but the restriction is
  driven by a *supervised semantic mask* instead of an inertial observability test.
- **CP-Rig's gauge projection = an inner-constraint** (Triggs et al., *BA: A Modern Synthesis*) and, equivalently,
  **removal of the global Killing (rigid-isometry) component** (KillingFusion) — Killing fields of ℝ³ *are* se(3),
  so projecting the rigid part out of `D` is exactly removing its global Killing component.
- Cite DefSLAM/floating-map-ambiguity + DefVINS as the lineage we extend to the no-IMU, supervised, dense regime.
This lives in the system paper's intro/method. We do NOT chase a standalone identifiability theorem.

---

## 2. Three layers

### Layer A — CONSTRUCTIVE SPINE (unchanged: the build ladder)
The 18-rung ground-up ladder in [[SEG_CONTRIBUTION_BUILD_LADDER]] is the build. Discipline unchanged:
plumbing (regression-tested) → elements (isolated-tested) → combines (Δx-tested). Spine:
`enabling plumbing → CPPR (smoke) → CP-Rig(a/b/c) → TriGauge → {Background-Canonical | Mixture} contingency`.

### Layer B — MEASUREMENT (folded from the document — its highest-value contribution)

**B1. Self-consistent synthetic testbed, TIER-1 ONLY, built EARLY as a TOOL (~1–2 wk), not a benchmark paper.**
- Warp a converged static SemSup reconstruction with KNOWN parametric deformation fields `D_gt(x,t)`:
  controllable axes — amplitude sweep, temporal-frequency sweep, spatial support (local tool-prod vs global
  breathing), and the adversarial **rigid-injection** case (`D_gt` carrying an exact SE(3) component).
- Render through the model's own renderer → synthetic frames with a **known (pose, D) decomposition**.
- This is the GT our Δx-revival metric currently lacks: it upgrades the metric from "is Δx ≠ 0?" to
  "**is Δx CORRECT?**" (attribution-accuracy = fraction of injected rigid content landing in pose vs field).
- **Honesty constraint (the document's own caveat, enforced):** tier-1 is *self-consistent* (data from the
  model's forward model) → **optimization-science only, never an external claim.** External validity (Blender/
  VisionBlender/SOFA tier-2) is STRETCH, not core — and note VisionBlender already emits deformation-field GT, so
  tier-2 is "adopt existing," not "build a benchmark."

**B2. Gauge-aware metric set (replaces ad-hoc ATE reporting).**
| Metric | What it measures | Where valid |
|---|---|---|
| **Attribution-accuracy** | rigid-in-pose vs rigid-in-field (needs GT decomp) | testbed only |
| **Δx-revival (spatially-structured)** | field alive, tissue≫bg/tool, temporally coherent | SemSup, CRCD |
| **Gauge-fixed ATE** | project est into GT gauge via weighted-Kabsch over est field, then compare | CRCD, StereoMIS |
| **Gauge-invariant 3D-track err** | point-trajectory error vs stereo/kinematic GT tracks | StereoMIS, CRCD |
| **Tracker-health Pearson** | per-frame-motion correlation, *published definition* | all (report w/ path-ratio) |
| Render PSNR/SSIM/LPIPS + per-tissue SDF sharpness | mapping quality / anti-mush | SemSup, CRCD |

Per-dataset validity is explicit: **SemSup** (no real pose → render + Δx + attribution lead), **CRCD**
(seg + poses → gauge-fixed ATE + Δx + tool tests), **StereoMIS** (no seg → seg-path no-op *control*: gauge-fixed
ATE only, must show no regression; this is where the [[project_stereomis_dds_audit_20260610]] ~40 mm floor is the
honest baseline, not a target).

### Layer C — THEORY FRAMING
As §1. Folded into the system paper; not a separate output.

---

## 3. Decision gates (the document's oracle-headroom + our two falsifiers, reconciled)

- **Gate 0 — Oracle-headroom (NEW front gate, on the testbed, before any learned attribution).** Set the
  attribution weight `w` from `D_gt` (oracle). Run. This upper-bounds the value of *perfect* attribution.
  → If oracle headroom is small on realistic axes, attribution is NOT the binding constraint → pivot early/cheap.
  Every later learned-`w` result is reported as a fraction of oracle headroom. (Strictly dominates our old
  Rung-9.5 go/no-go: it bounds the ceiling, not just the sign.)
- **Gate 1 — Rung 5 (CPPR ATE on CRCD).** Non-trivial τ_c moves no gauge-fixed ATE ⇒ deadness not per-ray-trust.
- **Gate 2 — Rung 9.5 (forced-m + frozen-pose Δx).** Δx≈0 ⇒ structural/frame-0 anchor ⇒ Background-Canonical.
- If Gate 0 says "no headroom" OR (Gate 1 ∧ Gate 2) fire ⇒ the soft seg-gate line is falsified ⇒ pivot to the
  structural/frame-0 attack (learned canonical frame), and the *diagnosis + the gauge-aware testbed/metrics*
  remain a defensible thesis core regardless.

---

## 4. Order — seg-first, tools-early (mapped to the forward plan)

Phase order is OUR inversion of the document: **tools + spine first, theory folded, clinical stretch.**

| Slot | Work |
|---|---|
| Day+3 (DDS upgrades) | Enabling plumbing (ladder Rungs 0–3) **+** testbed tier-1 (B1) **+** gauge-aware metrics (B2). These are the shared tools every later rung consumes. |
| Day+5 | **Gate 0 oracle-headroom** on testbed → CPPR (Rungs 4–5, Gate 1) → CP-Rig elements (6–10). |
| Day+6 | CP-Rig(a/b/c) combine (Rungs 11–13) — the thesis-spine revival result + the (a)>(b)>(c) defence. |
| Day+7 | TriGauge (Rungs 14–15) — the standout novelty, on CRCD's tool class. |
| (parked) | Background-Canonical / Mixture (16–17) — contingency-gated by Gates. |
| STRETCH (not critical path) | clinical surrogate task (WEISS collab), tier-2 testbed (adopt VisionBlender), 3DGS instantiation, formal identifiability condition. |

Cross-cutting (separate track): the StereoMIS reproduction ladder ([[project_stereomis_dds_audit_20260610]]) —
note its result re-uses Layer-B gauge-fixed ATE; StereoMIS is the seg-path control, not a seg target.

---

## 5. Scope honesty

- **Core thesis (achievable on our hardware):** diagnosis + seg-supervised CP-Rig/TriGauge + testbed-tier-1 +
  gauge-aware evaluation, SemSup→CRCD, StereoMIS as control. One system paper + the diagnosis. Self-contained.
- **Stretch (only if time/collaboration):** clinical task, tier-2 (Blender/VisionBlender) external validity,
  3DGS port, the identifiability *condition* as a formal result.
- **Explicitly NOT doing** (rejected from the document): the standalone theory paper (DefVINS/DefSLAM own it),
  the full benchmark paper (VisionBlender/MultiViPerFrOG exist), the 3-paper TRO+CVPR+MICCAI programme, and the
  priority inversion to theory-first. We take the instruments, not the order.

## 6. What we adopt vs reject (for the record)

ADOPT: gauge-aware metrics (B2); oracle-headroom as Gate 0; self-consistent testbed as a *tool* (B1, tier-1);
classical-lineage framing (Triggs/OC-EKF/DefSLAM/KillingFusion); "fail gracefully when unobservable" for
whole-field deformation; restore the clinical endpoint as stretch.
REJECT: theory-first ordering; standalone theory + benchmark papers as moats (contested ground); segmentation
demotion (it's our open slice); the WildGS *delete-the-distractor* objective (we model, not delete); Phase-0's
stale audit items (re-verify against code — cf. live `smooth_weight`, configurable `edge_semantic_weight`);
treating "reproduce published numbers" as a clean gate (on StereoMIS it's a finding — ~40 mm floor).
