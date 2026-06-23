# Critical Review — C2 (flow-as-sensor gate) on EndoGSLAM as base model + first contribution (2026-06-23)

> 7-agent adversarial review (workflow wy5fws7t4), claims verified against live source. Reviews
> `GS_C2_IMPL_SPEC_20260623.md` + `GS_CONTRIB_PREP_20260622.md`. Verdict: **C2-on-EndoGSLAM is NOT a sound
> first contribution to build now — re-scope.** Three blocking problems, two weakening, grounded below.

## DISAGREE — change the plan

### 1. The hard FIX games the win-metric (frozen-camera → low path-ratio). [BLOCKS the headline]
Verified `initialize_camera_pose` (main.py:375-378): under the `nofwd_lr` baseline (`forward_prop=False`, the
operating reference) a FIX frame copies `cam_trans[t-1]` verbatim → **frozen camera, zero arc**. The win-gate's
primary criterion rewards path-ratio *down* (`sim3_ate.py:86` `path_ratio = pathlen(est)·s / pathlen(gt)`). Every
frozen FIX frame mechanically lowers path-ratio at ~zero true-error cost on a sub-SNR snippet → **"FIX more frames"
scores as a "win" regardless of reconstruction quality.** The guards (`max_fix_rate=0.30`, freeze control, band,
Pearson) *bound* the artifact but never *attribute* it: at 20-30% FIX (below cap, above degeneracy) the path-ratio
drop is a mixture of real de-jitter + frozen-arc suppression, and nothing separates them. This is exactly the
degeneracy `feedback_sim3_ate_misleading_subsnr` warns about, re-introduced as the FIX *mechanism*.
**Fix:** recompute path-ratio + Pearson over **non-fixed frames only**; add a **FIX-on-moving-GT rate** (FIXing a
truly-moving frame is a tracking miss the metric rewards).

### 2. The attribution precondition was never run — and the one experiment we ran argues AGAINST deformation. [BLOCKS]
The deformation framing ("the GS base absorbs deforming tissue into spurious camera motion → jitter") is stated as
established, but the **build spec omits the attribution check entirely** (absent from the §9 checklist and §10
risks). Worse: the `nofwd_lr` sweep **falsifies** it. A config-only change (forward_prop OFF + LR×0.2) — touching
nothing about tissue — cut path-ratio 93→31.8 and ATE 5.2→3.24. **If the excess were deforming tissue dragging
`cam_trans`, an optimizer knob could not touch it** (deformation lives in the data, not the integrator). So the
dominant component was **optimizer dynamics on sub-SNR motion**, not deformation — `project_dds_slam_flaws_catalog`'s
"tracker optimizer dynamics dominate." The residual ~32× is mostly {scale floor ~8-11× + solver noise + specular},
with deformation a residual-of-a-residual never isolated. **My earlier "attribution answered" was backwards.**
**Fix (cheap, decisive, hours of numpy on existing trajectories, pre-registered vs a shuffled-label null):**
(a) Sampson residual ↔ per-frame excess-arc Pearson (must beat shuffled, |r|>0.5); (b) Sampson ↔ CRCD GT seg
(elevated on tool/deform frames?); (c) nofwd_lr seed-std vs the 8-11× scale floor — if 31.8±std straddles the floor,
**there is no isolable deformation budget for C2 to attack.**

### 3. Validating a deformation claim on sub-SNR CRCD while STIR sits shelved is a methodological error. [BLOCKS]
§8 validates entirely on CRCD (sub-SNR: GT ~18mm/360f ≈ 0.05mm/frame; up-to-scale 8-11× floor; "deformation⊥awake-GT"
confound). The project **already declared measurability SOLVED via STIR** (real, metric, unbiased deformation
endpoint-EPE, above-SNR — `project_stir_dataset_measurability_20260614`) and shelved it. Demonstrate an effect where
signal exceeds noise *first*, then show transfer. CRCD is the legitimate transfer check; it **cannot be the existence
proof** (and by the LOCKED rule, CRCD-only can't headline). The 550 lines of gate-hardening is rigor on the wrong
instrument. **Fix:** STIR endpoint-EPE ON-vs-OFF (n=3) as the existence gate; CRCD = transfer.

### 4. The agreement gate conflates four causes → "deformation gate" is an over-claim. [WEAKENS]
Verified the label is a single threshold on the median Sampson residual (flow_track.py:123). A region disagreeing
with the rigid-camera consensus F is *not* necessarily deforming — it is also **specular glare** (highlights slide
with the camera; RAFT tracks the highlight, not the surface — the endemic endoscopic failure WildGS cites),
**low-texture flow failure**, or **occlusion** at the tool boundary. Pooling into ViT patches changes the granularity
of the conflation, not its nature. So C2 is mechanically "down-weight pixels inconsistent with the dominant rigid
flow" = the WildGS/On-the-go robustness recipe with a flow residual swapped in; NRGS-SLAM does the actual
rigid-vs-deform disambiguation with a Bayesian per-Gaussian energy (the strictly stronger instrument). **Thesis
sting:** on a tool sweep (specular, occluding, fast) the gate FIXes those frames as "deforming" — but the tool is
exactly what the locked end-goal wants **rendered as a composited object, not suppressed**. **Fix:** drop
"deformation" from the headline (validate as a robustness gate), or add a deformation discriminant (temporal
persistence: glare transient, deformation persists; tools segmentable) before any NRGS-novelty claim.

### 5. Demoting C1 (the only PROVEN metric) while headlining C2 violates "metrics are the only arbiter." [WEAKENS]
C1 was demoted on a **novelty** argument ("bare port = WildGS reproduction"). But the LOCKED arbiter is metrics, and
C1 has the only proven metric in the program (−23% Sim3-ATE, n=3 non-overlapping, NeRF DDS-SLAM). EndoGSLAM tracking
is photometric render-and-compare; a learned appearance-grounded `1/σ²` can suppress specular/low-texture pixels —
i.e. C1 attacks the base's *foundational* defect, which C2's model-free flow agreement is **blind to** (speculars are
rigid-consistent). We're headlining the contribution that doesn't address the base's worst failure and demoting the
one that does, on a rule-violating novelty argument. **Fix:** keep C1 as the empirically-grounded floor; if anything,
headline C1 (proven) and treat C2 as a research arm pending its attribution check.

## AGREE-BUT-RISKY — keep, eyes open
- **Mean-preserving weight:** keep, but re-justify as **LR-invariance** (an un-normalized `w∈[w_min,1]` shrinks the
  tracking gradient frame-dependently → confounds the A/B), NOT the spec's dead-early-stop/argmin reasons. AND the
  optimizer is **Adam** (main.py:149), so the spec's SGD "total-mass = |M| → same step" identity (§2d) is unclean —
  Adam's second-moment normalization partly cancels a global rescale. Verify the invariance survives Adam empirically.
- **Two RAFT passes/frame (OPEN-G):** false dilemma. Call `_raft_flow` **once** in the net-new `gs_flow_gate.py` and
  feed both pure helpers — flow_track.py stays byte-for-byte, halves the dominant gate cost across the whole matrix.
- **Default-off parity (§7):** strong; the import-isolation + execution-trace approach is the correct GPU-safe analogue.

## BOTTOM LINE
**C2-on-EndoGSLAM is not a sound first build, and "this can be the base model" is the trap.** A rigid GS base + a
camera-de-jitter gate is a *tracking* base; the locked thesis needs a *deformation/tool* base. C2 reconstructs no
deformation, renders no tool, routes into no deform DOF (the base has none) — it detects "moving" pixels and
**discards** them, anti-aligned with a thesis whose contribution is *reconstructing* deformation. The spec's own
honesty is its indictment: it wrote down every reason C2 is thin and headlined it anyway.

**Recommended re-order (re-scope, don't delete):**
1. **Attribution panel first** (numpy on existing trajectories, hours) — if deformation isn't isolable above the
   scale/optimizer/specular floors, C2 is mis-aimed and the fix is depth-scale/LR, not a flow gate.
2. **Run the tool-entry GO/NO-GO probe** — the migration premise ("GS beats NeRF on tool-ghosting") is unverified;
   every line of C2 is sunk cost if SH-0 isotropic + densify-off can't render a tool entry without ghosting.
3. **Stand up STIR endpoint-EPE** as the existence gate; CRCD = transfer.
4. **Keep C1 as the empirically-grounded floor** (only proven metric; attacks the photometric defect C2 is blind to).
5. **If C2 survives 1-3, re-scope it** from "FIX the camera" to "the motion *sensor* whose `moving` output routes
   INTO a deform/tool DOF" — the difference between a detour and the contribution. That needs a deform DOF to exist first.
