# COMBINE + TriGauge — THE DESTINATION DESIGN (2026-06-19, workflow wv95xikqb)

How Arm-2 motion teaches Arm-1 sigma^2 (the COMBINE) + the 3-way seg-conditioned router (TriGauge). Code-grounded, NRGS-SLAM read in full.

---

All four readings are confirmed against live code. The interfaces are exactly as described: `forward()` signature at `:481`, the NLL teacher switch at `:589`, the `oracle_w` gauge slot at `:223-225`, the per-ray `_w_ray` Inc-2 site at `:541`, and `deformation_off` as a whole-run flag at `:202`. I have what I need to architect.

---

# DDS-TriGauge — THE DESTINATION DESIGN + BUILD PLAN

Chief Architect verdict. Caveman-direct, code-grounded, metric-first. Every claim below is anchored to a verified `file:line`.

---

## 1. THE COMBINE — Arm-2 motion teaches Arm-1 σ²

### Teacher signal — DECIDED: **rigid-vs-deformable residual contrast `E_R − E_D`**, NOT raw |Δx*|, NOT a binary BCE posterior.

Three candidates, scored on the only axis that matters (does it make σ² land on deforming *tissue* without re-introducing a gauge that self-confirms):

| Teacher | What σ² learns | Gauge-clean? | Verdict |
|---|---|---|---|
| **raw \|Δx*\|** (baked target) | "where the field *output* is big" | **NO** — `Δx*` is depth+DINO-correspondence pull-back; it is gauge-confounded (memory: `residual-Pearson self-confirms`, battery-7 `dx≈0 + pearson≈0` = field moves empty space). Teaching σ² from \|Δx*\| just copies the field's own bias. | **REJECT** |
| **dual-hypothesis BCE posterior** (NRGS Eq. 25–28) | binary `P(z=D)` per primitive | clean, but it is a *different object* — a 0/1 rigid/deform classifier regressed by BCE. Discards σ²'s aleatoric-NLL meaning. | **REJECT for the σ² head** (keep as the *routing gauge* — see §2) |
| **contrast `E_R − E_D`** (NRGS Eq. 28 signal, our NLL form) | "where deformation *explains residual the rigid render cannot*" | **YES by construction** — both renders share the SAME pose, SAME SDF/map, SAME rays; only the Δx toggle differs. This is exactly the invariant NRGS uses (`:202` zeroes `vox_motion`, nothing else moves). | **ADOPT** |

**Reasoning.** The whole COMBINE exists to fix our own diagnosed failure: σ² trained on the raw photometric residual is a contrast/motion detector (NeRF-On-the-go `σ∝‖C−Ĉ‖` degeneracy, confirmed in our 06-18 verdict). The contrast `E_R − E_D` is the *one* signal that is high specifically where the deformation field pays its way — and it is gauge-clean because the rigid hypothesis `E_R` is rendered at the **identical** pose/map state (the trap memory flags is sidestepped exactly as NRGS sidesteps it). Raw |Δx*| fails the gauge test we already documented; the binary BCE destroys the aleatoric semantics we want to keep.

### How it makes σ² deformation-aware

We do NOT replace the NLL. We **inflate σ²'s target where deformation explains residual**, keeping the NLL meaning. Concrete: at the teacher site (`scene_rep.py:589`), add a branch that sets the σ² residual target to a contrast-modulated term. The cleanest formulation that preserves NLL semantics:

```
_contrast = relu(E_R - E_D)          # per-ray, >0 only where deform helps; detached
_err      = _rgb_e2 * (1 + lambda_c * _contrast_normed)   # up-weights residual where deform explains it
_nll      = 0.5 * (_err / _s2 + log(_s2))                  # unchanged NLL form
```

Effect: where deformation genuinely explains the scene (`E_D ≪ E_R`), the effective residual the head must "cover" is larger → σ² rises → high uncertainty lands on **deforming tissue**, not on speculars/pins (which have `E_R ≈ E_D`, no contrast). This is the principled inversion: σ² now encodes "this pixel's residual is deformation-driven, trust its photometric term less for pose" — which feeds Inc-2's pose down-weight (`:541`) so the tracker stops fighting tissue motion. That is the COMBINE's whole point.

### Plug-in point (from the interfaces reading — exact)

- **NLL branch:** `scene_rep.py:589`. Add `elif _teacher == 'contrast':`. `_s2` (`:581`), `valid_depth_mask` (`:523`), `target_rgb`, `rend_dict['rgb']` (= E_D side) are ALL in scope. Backend-agnostic — supervises geo (`:580`) and dino (`:509-517`) σ² unchanged.
- **The missing piece — `E_R` in-graph:** this is the single biggest build. `deformation_off` is a *whole-run* flag (`:202`), offline-only (`deform_off_render.py:58`). For an in-graph teacher we need a **second `render_rays` pass with Δx suppressed locally**. Thread a `deform_off=False` kwarg through `render_rays → run_network` that locally overrides `:202` (do NOT touch the global cfg). Call it once more inside `forward`, **`.detach()` it**, form `E_R = (rgb_R − target_rgb)²`. Cost = one extra mapping forward (mapping only, never tracking).

Parity: the branch is gated on `uncertainty.teacher == 'contrast'` (default `'l2'`) AND the twin render only fires when that teacher is selected. Flags-off = byte-identical (the `elif` is unreached, the second render is never called). Parity gate `test_inc0_bitidentical.py` covers it unchanged.

### Cheapest dry-run BEFORE committing the twin-render plumbing

**Do not build the in-graph twin render first.** De-risk offline, zero new training code:

1. Take an existing trained geo-uncert checkpoint (we have these — canon n=3).
2. Run `deform_off_render.py` (`:58`) → get `E_R` images. Run normal render → `E_D` images.
3. Offline numpy: compute `relu(E_R − E_D)` per pixel. Correlate against (a) the trained σ² map, (b) the seg tissue mask, (c) where available, STIR EPE.
4. **GATE:** does `E_R − E_D` concentrate on tissue (not pins/speculars/tool edges)? Is it non-trivial (not ≈0 everywhere — the field is diagnosed dead, so this is a REAL risk)?

If `E_R − E_D ≈ 0` everywhere (dead field → no contrast), the COMBINE has no signal and is **blocked on Arm-2 field revival** — you learn this for the cost of two render passes on an existing ckpt, before writing a line of twin-render code. This dry-run is the most important gate in the whole plan.

---

## 2. TRIGAUGE — the 3-way seg-conditioned router

### Architecture: two non-redundant factors

```
gauge  g(x,t) = sigmoid( DINO-query cross-attn fused feature )   # HOW MUCH motion (rigid vs moving)
seg    s(x)   = softmax class prior {bg, tissue, tool}           # WHAT KIND (the missing axis)

route (per ray):
  w_static = (1-g) · s_bg      -> camera ego-motion  (anchors gauge; the only route with a hard metric: Sim3 ATE)
  w_tissue =    g  · s_tissue   -> deformation field Δx   [fills the existing oracle_w slot, :223-225]
  w_tool   =    g  · s_tool     -> tool SE(3) head T_tool(t)   [NEW destination]
  KEEP     =    everything not confidently routed stays in the static/map solve (no hard assignment)
```

**Why 2 factors, not NRGS's 1:** `g` alone cannot split deforming tissue from a moving rigid tool (both high-motion, high-σ); `s` alone cannot split a parked tool from a moving one. The 2×2 is the formal justification for TriGauge over NRGS's scalar `w_d`. NRGS *structurally folds a moving tool into the deform bucket and discards it*; the seg axis is precisely what it lacks.

### The DINO-as-query cross-attention gauge (per-ray-sampling compatible)

Verified: today's only fusion is plain CONCAT `[DINO; rgb.detach; depth.detach]` → `dino_unc_net` (`:511-515`). Upgrade to **asymmetric cross-attention over the along-ray sample axis** — the SNI pattern, which already runs on `(ray, points, dim)` tensors, so it is 1:1 with our `(N_rays, N_samples, ·)` layout, no full-image requirement:

```
q = W_q · f_dino                                  # DINO = QUERY (deformation-reliable cue), broadcast along samples
k,v = W_{k,v} · [f_geo ; f_app]                   # geometry(geo_feat) + appearance, genuinely per-point
Aff = softmax_over_samples( k @ qᵀ / sqrt(d) )    # (N_rays, N_samp, N_samp), attention over the along-ray axis
f_fused = Aff @ v
g  = sigmoid(gauge_head(f_fused))                 # then volume-render g along the ray exactly like Inc-1's σ²
```

**Why DINO is the query (justified, with the honest caveat):** the query steers the fusion's focus (cross-attn consensus); we want the gauge steered by the deformation-reliable modality. Our own A/B says DINO-σ² is *more deformation-reliable but jumpier*; making DINO the **query** (not the rendered value) keeps its localisation while **anchoring the jumpiness to stable rendered geometry/appearance values**. This is the principled fix for the exact failure we measured. SNI does the *opposite* (appearance+geo correct semantic) because in SNI semantic is the *noisy* target; our role-swap is correct because here DINO is the *trusted steerer*. Cite SNI for the mechanism, justify the swap by the task.

**Caveat (do not oversell):** cross-attention sharpens *where* g lands; it does NOT cure photometric-residual deformation-blindness. That is the COMBINE's job (§1). Sell as "better fusion of the cue," not "solves deformation."

### The tool SE(3) head

`T_tool(t) ∈ SE(3)` — 6 Lie-algebra params per frame, an `nn.Parameter` table indexed by frame, optimised alongside camera pose in `get_pose_param_optim` but **excluded from the camera-pose residual**. Rays with high `w_tool` have their sample points transformed by `T_tool(t)` instead of the deformation field. **Minimum viable = rigid per-frame SE(3)**; cite articulation (SurgPose/SurgRIPE) as future work.

**Critical guard:** a free per-frame `T_tool(t)` is a 6-DoF nuisance that WILL absorb camera + tissue motion if unconstrained (same over-explain trap as D²NeRF's dynamic field). Two guards: (a) applied ONLY where `w_tool` high (seg-gated), (b) temporal-smoothness regularised. **The seg-prior is what makes the tool head safe — that safety is the contribution, not the SE(3) head itself** (tool pose estimation is mature).

### What renders / what's supervised

- **Renders:** rgb, depth, edge_semantic (existing). `g` volume-rendered for viz (like σ²). Tissue rays warped by Δx; tool rays warped by `T_tool(t)`.
- **Supervised:**
  - `g` (gauge): by the **contrast `E_R − E_D`** distilled (NRGS Eq. 28 signal — this is where the binary-posterior idea legitimately lives, as a *soft target for the routing gauge*, distinct from the σ² NLL).
  - `Δx` (tissue): by the Arm-2 `deform_teacher_loss` (`:249`, already wired), gated by `w_tissue`.
  - `T_tool`: by photometric residual on tool rays + temporal smoothness.
  - σ² (the COMBINE): by §1's contrast-modulated NLL.

### The load-bearing enabling change (verified live)

`dataset.py:45` collapses the seg map to a Canny EDGE field; raw {bg,tissue,tool} class id is destroyed at load (`semantic_data` read at `:211-212`, consumed only by `compute_edge_semantic`, then DROPPED — never in `ret`). **Fix:** add `ret["seg"] = <canonicalised {0,1,2} label>` at `dataset.py:243-251`, gather per-ray next to `ddsslam.py:361/:618` (same `indice_h,indice_w`), thread `target_seg=` into `forward` (`:481`). The "seg-prior × per-ray quantity" multiply pattern ALREADY exists (`:499`, `:223-225`, and the per-ray `_w_ray [N,1]` at `:541`) — TriGauge fills those slots with `class label`, not edge field.

---

## 3. STAGED BUILD ORDER — each metric-gated

Mapped onto contribution staging (machinery behind the gate, not separate contributions). **One change → run FROM pristine base → did OUR metric (PSNR/SSIM/LPIPS + Sim3 ATE) clear the n=3 seed noise floor? → keep.**

| Stage | Build | Gated on | Metric gate | Needs STIR? |
|---|---|---|---|---|
| **S0 (DRY-RUN, do first)** | Offline `E_R − E_D` correlation on existing ckpt (§1 dry-run) | nothing — existing ckpts | does contrast concentrate on tissue AND is it non-zero? PASS→S2 has signal; FAIL→COMBINE blocked on Arm-2 | No (STIR is the *confirm* later) |
| **S1** | Surface raw seg `ret["seg"]` + thread `target_seg` (the enabling change). NO router yet — just plumb + **oracle Gate-0**: set `w` from GT seg, upper-bound the win | in-flight Arm-1 (geo backend already won) | oracle routing improves Sim3 ATE / held-out render vs everything-rigid? If oracle can't, learned can't → STOP | No |
| **S2** | THE COMBINE: `teacher=='contrast'` branch (`:589`) + in-graph twin render (`deform_off` kwarg through `render_rays`) | **S0 PASS** + Arm-1 teacher A/B (depth/rgb_depth) settled | σ² lands on tissue (qual) AND Inc-2 pose down-weight improves Sim3 ATE n=3 vs geo-l2 | No to ship; **STIR to PROVE** σ² is deformation-aware not contrast |
| **S3** | DINO-query cross-attn gauge (replace concat `:511-515`) | Arm-1 fusion result (the in-flight `[DINO;rgb;depth]` concat A/B) + **dino_reg** (artifact-token fix) | render + Sim3 ATE: cross-attn > concat > geo? If concat already ≥ attn, SHIP concat (Occam) | No |
| **S4** | TriGauge tissue+static routing (2-way, no tool yet): `g × s_tissue` → Δx, `(1-g) × s_bg` → camera | S1 oracle PASS + S2 + S3 | Sim3 ATE (static purity) + held-out-beats-rigid (tissue). (a)seg-sup > (b)unsup-gate > (c)shuffled-label ablation | held-out-beats-rigid now; STIR-EPE = un-gameable confirm |
| **S5** | Tool SE(3) head + 3-way route | S4 + tool-contact frames identified | camera ATE + held-out render in tool-contact window; ON vs DynaSLAM-delete vs tool-into-field | STIR helps but tool is rigid → ATE suffices |
| **S6 (COMBINE-LAST)** | Merge with FIX-branch field revival; full TriGauge live | everything above + Arm-2 field actually alive | did fixing add value ON TOP of improving | **STIR EPE decisive here** |

**What's gated on running experiments (be honest):**
- S2 (COMBINE) is **gated on S0** — if the dead field produces `E_R−E_D≈0`, there is no teacher signal and S2 waits on FIX-branch revival.
- S3 (cross-attn) is **gated on the in-flight concat-fusion A/B** — if concat already wins, do not build attention (metric-first Occam).
- S4+ are **gated on S1 oracle Gate-0** — oracle headroom must exist before any learned router.

---

## 4. NOVELTY vs NRGS-SLAM — the honest one-liner

> **NRGS-SLAM is GS, binary, seg-free, photometric-only: one per-Gaussian rigid/deformable probability (BCE on a 2-hypothesis posterior) that down-weights tracking. DDS-TriGauge is SDF, aleatoric, 3-way, seg-conditioned, semantic-fused: a continuous NLL-trained σ² *taught to be deformation-aware by NRGS's own `E_R−E_D` contrast*, routed by `seg-prior × DINO-query-gauge` to three destinations — static→camera, tissue→deformation field, and a tool→own-SE(3) branch NRGS structurally cannot express (it folds a moving tool into the deform bucket and discards it).**

NRGS *validates our signal* (independent confirmation that `E_R−E_D`, not raw |Δx|, is the right teacher and can be self-supervised — de-risks S2, gives us Eq. 30–33 to cite-and-reuse) while our **defensible delta** is exactly the four axes it lacks: SDF substrate, aleatoric-NLL σ² (vs binary BCE), 3-way seg-prior routing with a tool branch, and a semantic/DINO-conditioned gauge. NRGS lists DDS-SLAM as a baseline it beats — which directly motivates a like-for-like SDF answer, benchmarkable head-to-head on StereoMIS/SCARED where NRGS already reports DDS numbers.

---

## 5. OPEN DECISIONS — your call

1. **Twin-render cost vs the lighter |Δx*| path for the COMBINE teacher.** The contrast `E_R−E_D` is gauge-clean but **doubles the mapping forward**. The |Δx*| path (thread baked `deform_dx` into `forward`) is one cheap gather, no twin render — but gauge-confounded (memory-flagged). **My recommendation: contrast, because gauge-cleanliness is the entire reason the COMBINE exists** — but you eat 2× mapping cost. Accept the cost, or accept a confounded-but-cheap teacher as a stepping stone? (I'd run S0 first to decide — if contrast signal is strong, the cost is justified.)

2. **Does the tool SE(3) branch ship in THIS thesis, or is it future-work?** It is the cleanest NRGS differentiator but also the riskiest (6-DoF nuisance, needs careful regularisation, narrow win-window = tool-contact frames only). **My lean: build the 2-way (static/tissue) router to a metric win first (S4); stage tool (S5) as "ships only if S4 lands AND tool-contact frames show a measurable ATE win."** Tissue routing is the load-bearing claim; tool is the flourish.

3. **Cross-attention gauge vs keep-the-concat (S3).** If the in-flight `[DINO;rgb;depth]` concat fusion already beats geo on render+ATE, cross-attention is extra surface for no metric need. **My lean: build cross-attn ONLY if concat fails to beat geo OR if S0 shows DINO localisation is the bottleneck.** Metric-first says don't build attention speculatively.

---

## 6. THE SINGLE BIGGEST RISK + cheapest de-risk

**RISK:** The deformation field is DIAGNOSED DEAD (trains to |dx|=0, starved). If it is genuinely inert, then `E_R − E_D ≈ 0` everywhere — **the COMBINE has no teacher signal, TriGauge's tissue route has nothing to route, and the whole Arm-1↔Arm-2 fusion collapses to "σ² stays a contrast detector."** This single fact can null the entire destination.

**CHEAPEST DE-RISK = S0, the offline contrast dry-run.** Two render passes on an *existing* canon checkpoint (`deform_off_render.py:58` for `E_R`, normal render for `E_D`), then numpy `relu(E_R−E_D)` correlated against the tissue seg mask. Zero new training code, ~minutes. It tells you BEFORE you build any twin-render plumbing whether:
- the contrast is non-trivial (field does *something*) → COMBINE has signal, proceed to S2; or
- contrast ≈ 0 (field truly dead) → COMBINE is **blocked on FIX-branch field revival**, and you've learned it for the price of two forward passes instead of weeks of twin-render + router code.

This is the load-bearing gate. **Run S0 before committing to anything downstream.** It also doubles as the first STIR-confirmable measurement when STIR EPE comes online (correlate `E_R−E_D` against tattoo-endpoint EPE = the un-gameable proof that the contrast is real deformation, not residual noise).

---

**Code anchors (all verified this session):** NLL teacher switch `scene_rep.py:589`; σ² in scope `:581`; `valid_depth_mask` `:523`; gauge slot `oracle_w` `:223-225`; per-ray `_w_ray [N,1]` Inc-2 site `:541`; dino σ² concat fusion `:511-515`; `deformation_off` whole-run flag `:202`; `deform_teacher_loss` (Arm-2, wired) `:249-267`; `forward` signature `:481`; seg-Canny-collapse `dataset.py:45`; offline E_R render `deform_off_render.py:58`.