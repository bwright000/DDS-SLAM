# Segmentation-Driven DDS-SLAM — Ground-Up Build Ladder

Source: workflow `winujr7wl` (2 decompositions → adversarial verify → synthesis), 2026-06-10.
Contribution agenda: `[[project_dds_seg_research_agenda_20260610]]` (CP-Rig + TriGauge, supervised tissue-class).

**Discipline:** build ONE atomic element → test in ISOLATION (regression test = knob-at-identity reproduces
baseline; + targeted test of the new behaviour) → only then COMBINE → test the combination. No big-bang.

**Regression oracle:** `baseline_fp.json` (Rung 0) = SHA256 over {est_c2w bytes, per-frame loss scalars, Δx array}.
Plumbing/identity rungs that touch no float math → **byte-identical**. Rung 3 (re-normalises a reduction) →
**algebraic-equivalence + `allclose(atol=1e-6)`** (the ONE relaxed oracle). Combines inherit the weakest oracle.

**Class LUT (verified frame_select.py:15):** `0=bg, 1=Liver, 2=GB, 3=Tool`. Rigidity BCE prior collapse:
`{Liver,GB}→deformable(0)`, `{bg,Tool}→rigid(1)`.

**Metric rule:** revival/gauge claims → **Δx-revival (dx_hook, OFFLINE)**; tracking → Sim3+SE3 ATE + est/GT
path-ratio + dominant-axis Pearson; plumbing → regression equality. SemSup ATE is fictional — render+Δx lead;
the ATE falsifier runs on **CRCD** (seg + real-ish poses).

## Three bands
- **Plumbing (Rungs 1–3, +0 harness):** pure regression rungs — "disabled ⇒ identical to baseline".
- **Elements (4,6,7,8,9,10,14,16):** independently testable (does `m` learn the prior? does the gate kill motion?).
- **Combines (5,9.5,11,12,13,15,17):** only verifiable together, via Δx-revival / ablation deltas.

---

## Rungs

**0 — Harness + frozen baseline fingerprint** (element, 0.5 d). `diagnosis/infra/ladder_harness.py`: deterministic
mini-run (seeded) on a tiny fixed SemSup slice → fingerprint; freeze `baseline_fp.json`. **dx_hook is OFFLINE**
(loads ckpt + runs its own forward — NOT a live callback; harness saves a mini-ckpt + est_c2w then invokes
dx_hook's forward). Gate: same-seed identical, diff-seed differ.

**1 — PLUMBING-A: un-collapse seg → 4-class int map** (element, 1 d). `dataset.py`: `compute_class_map()` (the info
discarded at line 45), new key `seg_class` in BOTH StereoMIS + Super getitem, behind `seg.per_class:false`. **Never
touch `compute_edge_semantic`/line 45.** StereoMIS/seg-absent → sentinel `-1`. Regression: flag-off byte-identical
on both datasets. Targeted: per-class fractions match frame_select within 2%.

**2 — PLUMBING-B: thread seg_class to per-ray** (element, 0.5 d). `ddsslam.py` 3 slice sites (:239,:322,:552) +
`scene_rep.forward(:381)` kwarg `seg_class=None`, stashed unused. Use the EXACT slice expression `target_d` uses.
Regression: flag-off byte-identical. Targeted: per-ray class histogram matches frame-level.

**2.5 — TRACKING-SCOPE kwarg** (element, 0.25 d; fixes FP-2). `forward` has no tracking/mapping discriminator.
Add `apply_pose_reweight=False`, true ONLY from `tracking_render`(:563), false from both mapping calls. Unread →
byte-identical. Ensures later pose-weighting touches tracking only, never the map.

**3 — PLUMBING-C: boolean-mask → per-ray multiplicative pose-loss weight** (element, 1 d; **load-bearing, FP-1**).
Today `get_loss_from_ret`(:191-209) reduces to scalars — per-ray pose weighting does NOT exist. Generalise the
per-ray `rgb_weight` precedent (scene_rep.py:402-404) to `w_pose [N,1]` (default ones), `loss=((res²)*w).sum()/w.sum()`,
applied only when `apply_pose_reweight`. Flag `pose_reweight.enable:false`. **Regression = allclose(1e-6) NOT
byte-identical** (masked-mean→weighted-mean changes summation order; divide by `w.sum()`). Targeted: toy ret, zero-
weight rays → zero pose grad. **Must precede every consumer (CPPR Rung 5, detach Rung 9) — folding it into CPPR was
Ladder B's fatal big-bang.**

**4 — CPPR-A: fixed class→trust table τ_c** (element, 0.25 d). `cppr.tau:{0:1,1:1,2:1,3:1}`, `tau_of_class()` gather,
sentinel→1.0. Default identity, no consumer. Targeted: lookup correctness.

**5 — CPPR-FULL: τ_c drives tracking-only pose down-weight** (combine; builds 2.5,3,4; 1 d; idea #5 live). Wire
`w_pose=tau_of_class(seg_class)` into Rung-3, tracking only. Regression: all τ=1 → allclose(1e-6) baseline.
Targeted: τ_tool=0 → tool-ray pose grad 0. **⚑ FALSIFIER #1: ATE check on CRCD** (not SemSup) — non-trivial τ moves
no ATE ⇒ deadness not per-ray-trust (CPPR stays the published baseline; bias effort to structural/frame-0).

**6 — CP-Rig-A: RigidityNet head m=σ(MLP)** (element, 1 d). `decoder.py`: clone EdgeNet_Semantic(:256-304),
1-dim out, inputs `[embed_time,embed_pos]`. Construct ONLY when `rigidity.enable` (no params/RNG when off — joins
optimizer groups only when enabled). Regression: off → byte-identical param count. Targeted: m∈(0,1), grads flow.

**7 — CP-Rig-B: BCE supervision of m to class prior** (element, 1 d; weight 0 default). Volume-integrate m→`M_def`
(weights :120); BCE(M_def, prior). `if rigidity_weight>0: loss += ...` (wrap in `if`, NOT `0*term`, so weight=0 =
no autograd node = byte-identical). Targeted: overfit one mixed frame (pin t — FP-6 integer-time) → M_def AUC vs
prior > 0.8.

**8 — CP-Rig-C: deformation gating vox_motion←(1−m)·vox_motion** (element, 0.5 d; **SUPPLIED m**, isolated from head
quality). scene_rep.py:178-180 before the frame-0 where/add. Targeted brackets BOTH extremes: m≡1 → Δx==0
(==deformation_off); m≡0 → `1.0*x==x` exact == baseline. Regression: flag-off byte-identical.

**9 — CP-Rig-D: detached per-ray pose down-weight (1−M_def).detach()** (element, 1 d; **SUPPLIED/constant M_def**,
isolates the detach from head quality; builds 3,2.5 — NOT 6). Tracking pose loss only. The detach is load-bearing:
pose down-weighted on deformable rays WITHOUT training m via pose. Targeted (autograd): `grad_fn is None`; bg
(M_def=0) grads survive, deformable (M_def=1) grads→0. Regression: off → allclose(1e-6).

**9.5 — Falsifier-constructive probe: forced-supervised m + frozen pose → Δx verdict** (combine; builds 7,8,9; 0.5 d;
NEW). Force m to the class prior, freeze pose, run map+deformation on a SemSup frame. **⚑ FALSIFIER #2:** Δx>0 on
tissue ⇒ deadness is optimization-race (warm-start fixes it → proceed). Δx≈0 ⇒ structural/frame-0 anchor ⇒
**pivot to Rung 16** (skip 10–13). Answers the agenda falsifier BEFORE the expensive combine.

**10 — CP-Rig-E: warm-start anneal** (element, 1 d). Freeze `pose_optimizer.step()`(:589) for `warmup_frames`, then
release. Default 0 (byte-identical). Targeted: pose unchanged during freeze, moves after.

**11 — CP-Rig(a): full supervised config** (combine; builds 7,8,9,10; 1 d; **THESIS-SPINE**). `cp_rig_a.yaml` wires
the LEARNED M_def into Rung-9. Primary gate: **Δx spatially-structured revival** (tissue≫bg/tool≈0 — NOT bare mean>0)
while tracking loss still falls; render ≥ baseline. Identity-config → allclose(1e-6) baseline. On-fail: re-enable one
flag at a time (combination interaction = suspect); escalate warmup / add concentration prior; if still dead, 9.5
already said structural → Rung 16.

**12 — CP-Rig(b): unsupervised-gate control (NRGS-equiv)** (combine; 0.5 d). Same as (a) but `rigidity_weight=0`.
Gate: **(a) > (b)** on revival/render — else supervised-class novelty undemonstrated → pivot to TriGauge.

**13 — CP-Rig(c): shuffled-label control** (combine; 0.5 d). `seg.shuffle_labels:true`. Gate: **(a) ≫ (c)** — proves
SEMANTIC content carries the gain, not "any per-ray modulation". Else reframe / finer granularity.

**14 — TriGauge-A: per-frame tool SE(3) block** (element, 1.5 d). `T_tool[frame_id]` 6-dof, own optimizer group,
behind `trigauge.enable`. Tool-class samples transformed before SDF query. Targeted: identity T_tool → baseline
render; known translation → tool depth shifts exactly, tissue/bg unaffected. Off → byte-identical.

**15 — TriGauge-FULL: 3-state solve** (combine; builds 5,9,14; 1.5 d). bg→full pose weight (anchor); tissue→CP-Rig
down-weight; tool→EXCLUDED from camera pose grad (`w_pose=0`) but tracked by its own `T_tool`. Gate: camera ATE not
corrupted by tool motion + tool-region render improves. **Demonstrates what NRGS structurally cannot.** On-fail
(tool too small/sub-SNR): fall back to tool-EXCLUSION-only (still valid).

**16 — STRUCTURAL: Background-Canonical** (element, 1 d; contingency-gated by 9.5/11). Force `vox_motion=0` on
bg-labelled samples STRUCTURALLY (from the label, not learned m) → bg rigid-by-construction anchors the gauge.
Targeted: Δx>0 tissue, ==0 bg. On-fail: if still dead, the gauge IS the frame-0 anchor (`torch.where` :179) → anchor
to a LEARNED canonical frame, not frame 0.

**17 — STRUCTURAL: Mixture-of-Canonicals** (element, 2 d; biggest blast radius, built last). K class-indexed
deformable SDF sub-fields blended by supervised posterior at `query_sdf_at_time`(:217-253) — the first composition
hook (none exist today). Targeted: K=1 == single field (render-equal); K>1 specialises per class. Off → byte-identical.
On-fail (memory/overfit): shared-trunk + K small heads, or restrict to {bg-static, tissue-deformable}.

---

## Critical path & falsifiers
Spine: `0 → 1 → 2 → 2.5 → 3 → {6→7} → 8 → 9 → 9.5 → 10 → 11 (CP-Rig(a)) → 12/13 (a/b/c)`. CPPR (4→5) and TriGauge
(14→15) are parallel branches off the plumbing; structural (16,17) are gated by 9.5/11.

**Two falsification points** (null ⇒ pivot off the soft seg-A line):
1. **Rung 5** (CPPR ATE on CRCD) — no ATE move ⇒ deadness not per-ray-trust.
2. **Rung 9.5** (forced-m + frozen-pose Δx) — Δx≈0 ⇒ structural/frame-0, jump to Rung 16.
If BOTH fire: abandon the soft gate, go structural/frame-0 (Rung 16 → learned canonical frame).

## First 3 rungs to execute (all local/CPU-toy-able)
1. **Rung 0** — harness + `baseline_fp.json`. Adapt dx_hook OFFLINE (FP-3, the #1 trap).
2. **Rung 1** — 4-class `seg_class` key (verified LUT, NOT 3-state — FP-4). Never touch line 45; flag-off byte-identical.
3. **Rung 3** — per-ray pose-loss weight (after 2/2.5 transport, ~0.75 d combined). The architectural correction the
   whole ladder rests on; oracle = allclose(1e-6), NOT byte-identical.

Element isolated-tests (1,3,4,6,7,8) are CPU-toy-able locally (GTX970/T4); full combines (5,11–17) need Colab A100.
Commit after each rung. StereoMIS seg-path no-ops throughout (Rung-1 regression verifies flag-true == flag-false == baseline).
