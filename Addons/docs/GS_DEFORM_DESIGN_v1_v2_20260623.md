# GS deformation contribution — v1 (build now) + v2 (sitting by) — co-designed 2026-06-23

> Supersedes the spec's standalone C2. After the [critical review](GS_C2_CRITICAL_REVIEW_20260623.md) re-scoped
> C2, the design became **one sensor, two+ consumers**. v1 is the immediate fast build; v2 is the headline
> contribution, parked until v1 proves the signal is real. STIR is CUT (no time) — render is the arbiter.

## The shared sensor (one measurement, reused everywhere)
**Consensus flow residual** = the per-pixel image motion left over after the single best camera rigid-motion is
removed (RAFT flow → fit one camera model → residual = the part the camera can't explain). Gated by a
**flow-AND-depth discriminant**: a residual counts as "scene moved" only if the flow residual AND a depth/geometry
residual both fire. Filters **specular glare** (slides in 2D, surface depth unchanged → flow-residual high,
depth-residual ~0 → reject) before it can corrupt anything.

## Consumer A — tracking gate (KEEP)
Per-frame DINO-region rigid-motion **agreement** test: do the regions agree on one camera motion?
- agree + motion → camera DOF (trust pose)  ·  disagree / still → don't trust pose there (down-weight; optional FIX).
- This is an **ego-motion reliability sensor** (camera/scene split), = the front-end of the locked TriGauge router,
  NOT a "deformation gate." Validate on **non-fixed frames** (the hard FIX freezes the camera = zero arc, and
  path-ratio rewards that — frozen and working look identical otherwise). Prefer soft down-weight over hard FIX.

## v1 — mapping catch-up (BUILD NOW, fast)
Same residual, **opposite sign**: where the scene moved, **up-weight the mapping loss** so the static Gaussian map
re-fits the deformed tissue fast — "the scene deformed, the map must catch up." Localized (only where the residual
fires) so the rest of the map is protected.
- **What it is, honestly:** forget-and-refit. The static map keeps up; it does NOT store the deformation. Right MVP.
- **Mechanism (EndoGSLAM) — GEOMETRY path, not appearance (NeRF-agent #1, 2026-06-23):** the NeRF side
  proved an RGB-loss up-weight is a **total no-op** for deformation — the appearance loss is a weak lever;
  what moves the rendered surface is the **geometry** supervision. So `w_map = f(residual)` must up-weight
  the **DEPTH term** (which drives the Gaussian positions `means3D` via the rasterized depth), not just the
  colour term. GS is *more* favourable than NeRF here: EndoGSLAM mapping `loss_weights = {im:1.0, depth:1.0}`
  ([main.py](../../EndoGSLAM/scripts/main.py)) — depth is already first-class (not the 0.1-vs-1000 NeRF
  imbalance), so a per-pixel depth up-weight on the moved region genuinely re-fits the geometry there.
  Apply `w_map` to the **depth** residual in the mapping branch of `get_loss`
  ([main.py:278](../../EndoGSLAM/scripts/main.py#L278)) (+ colour, secondary); optional residual-biased
  densification. **Drive `w_map` from the PER-PIXEL residual, never a region/patch median** (NeRF-agent #3:
  region-median logged `moving=0` on a frame where 7.9% of px moved >3px → it silently ignores the tool).
- **Metric (the arbiter):** **held-out render** PSNR/SSIM/LPIPS — NOT the just-fit frame (up-weighting trivially
  raises the training-frame PSNR = overfit, not reconstruction). The claim "map keeps up" only shows on frames it
  didn't fit. **Guard:** render on the *non-deforming* regions must NOT drop (else catch-up is bleeding into the
  whole map = the known co-adapt blur, `project_deform_gauge_bug`).
- v1 = the **existence proof** the residual carries usable deformation signal. If it can't beat OFF on held-out
  render, v2 has nothing real to learn → stop.

## v2 — the deformation field (HEADLINE, sitting by)
A map that **remembers how things move**, not one that forgets. Two parts:
1. **Canonical Gaussian map** — the tissue at rest (the template, stable).
2. **Per-blob position-only motion basis** — each blob carries a few coefficients over a fixed playbook of smooth
   time-shapes (Deform3DGS-style RBF/polynomial). `pos_t = pos_canonical + Σ coeff · basis(t)`.
   - **Basis over MLP (locked):** can't forget by construction (trajectory baked per-blob → deletes the NeRF replay
     machinery), cheap (no net query → real-time), local (no cross-talk), surgical-proven. Trade: only smooth
     motions — which is exactly tissue (breathing/heartbeat/tool-push). Position-only first (matches the NeRF Δx* teacher).
3. **Taught by the flow residual** — the residual is a *measured target* for what the field must produce (the GS port
   of the NeRF **teacher** that revived the dead field, +67% pin-EPE; render-only loss leaves the field lazy/inert).
4. **DINO-grouped coherence regularizer** — soft loss: **DINO-similar spatial neighbors deform together** (feature
   similarity *among spatial neighbors*, not pure-feature — else disconnected same-type folds wrongly couple; not
   pure-spatial — else a touching tool couples to tissue). DINO cuts the boundary. **Bonus = the thesis hook:** the
   same cut **peels the tool off the tissue field** → tool free to have its own motion → **tool-as-object** falls out
   for free. Lift 2D DINO → 3D blobs by project-and-pool (Feature-3DGS move).
5. **Gauge** resolved by the tracking gate (camera/scene split → the field only ever sees the scene residual).

**DINO = one semantic backbone for the whole system:** gate regions (tracking) + C1 uncertainty + deform grouping
(+ tool peel). One feature, multiple jobs = a coherent thesis, not bolted-on tricks.

### v2 NeRF scars to reuse (we already paid for these)
- **Gauge bug** (`project_deform_gauge_bug`): camera↔field coupling → wrong-frame field. Fix = the gate's camera/scene
  split feeds the field only the scene part.
- **Forgetting** (`project_field_warped_pin_epe`): basis representation sidesteps it (no replay needed).
- **Co-adapt blur:** sharpen MAP-only on the current frame, field on its own targets; localization is the guard.
- **Render cost** (NeRF field cost ~4-6 PSNR un-routed): the flow-AND-depth discriminant + DINO localization route it.

### Measurability without STIR (the honest cost of building fast)
No direct deformation EPE. v2's "we reconstruct deformation" leans on **held-out render with a view/time gap** (v2's
canonical+field generalizes; v1's chase-the-latest map can't — that gap is the discriminator) + field visualization.
Weaker than STIR-EPE, but real. Accepted trade for speed.

## Build order
1. **v1 now** (fast): residual-up-weighted mapping + discriminant, held-out-render judged. Existence proof.
2. **v2 if v1 beats OFF**: basis deform field + residual teacher + DINO coherence + tool peel. The headline.

Status (2026-06-23): design parked ("sitting by"); awaiting the render-boost config sweep (`render_boost_crcd.sh`,
→ `MyDrive/Outputs/GS_render/`) before resuming the v1 build.
