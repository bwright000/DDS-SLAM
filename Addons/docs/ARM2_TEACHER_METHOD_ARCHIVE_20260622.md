# ARM-2 Teacher Method — ARCHIVE (paused 2026-06-22, resumable)

> **Status: PARKED, fully preserved, flag-gated default-off.** The Δx* correspondence-teacher REVIVED
> the dead DDS-SLAM deformation field (pin-EPE +52.5→+62.8%, cos(D,baked) +0.79→+0.85) but did NOT win
> the LIVE render on SemSup (best campaign arm 26.0 < dead-field base 27.70). Root cause is now known and
> is NOT a flaw in the teacher: it is a **depth-scale render-gate mismatch** upstream of the field
> (`DEPTH_SCALE_DEFORMATION_ROOTCAUSE_20260622.md`). We are pausing the teacher to pursue the
> flow-as-sensor combine on the pristine base field. This doc is the single point of return.

---

## 1. What the teacher method IS (one paragraph)
Base DDS-SLAM's deformation field (`time_net`, `vox_motion = time_net(h); pts = pts + vox_motion`,
`scene_rep.py:163-165`) is supervised **only through the render** (rgb/depth/sdf/fs/edge/smooth) and is
therefore **inert by design** (|Δx|≈0 — starved, faithful to the paper; see
`project_paper_vs_code_deformation_20260618`). The teacher method gives the field the **external
displacement target the paper deliberately forgoes**: bake `Δx* = X_ref − X_k` from DINO correspondence +
depth back-projection, then regress `time_net` toward it. This is OUR contribution, not a port-fix — base
has no such target (no `Addons/deform/`, no teacher loss). It is the ONE wire that makes the field alive.

## 2. The methodology — the flag chain (each config = one single-variable rung)
Inheritance: `trail3_teacher_off → on → replay → replay_sharp → replay_sharp_route(+12 fix variants)`.

| config | the one delta | why |
|---|---|---|
| `trail3_teacher_off` | `deformation_sup_weight: 0` | == pristine base (parity anchor) |
| `trail3_teacher_on` | `deformation_sup_weight: 10000` + `deform_field_teacher_only: true` + `deform_teacher_only: true` | regress `time_net`→Δx*. **THE fix**: keep `time_net` OUT of `global_BA`'s optimizer (render can't collapse it) and make `current_frame_mapping` loss = def_sup ALONE. `deform_subdir: deform` (baked @ `--grid_scale 3`, validated +86.2%/cos0.935 at held-out pins) |
| `trail3_teacher_replay` | `deform_replay_iters: 100`, `deform_replay_frames: 5`, `cur_frame_iters: 0` | **forgetting fix**: train the field each step on a CAUSAL REPLAY of past frames (time-specific) instead of frame-k-only (recency bias). Discriminator = judge's shuffled-time control: real-time must BEAT shuffled |
| `trail3_teacher_replay_sharp` | `cur_frame_iters: 100` + `cur_frame_map_only: true` | **render recovery**: turn the sharpening pass back ON but MAP-ONLY (field excluded from `cur_map_optimizer`) → render sharpens the MAP, field stays alive via replay. Co-adapt MAP→field (right direction; naive coadapt collapsed the field). Render 22→24.1 |
| `trail3_teacher_replay_sharp_route` | `map_route: {enable, mode:pixel, soft_scale:2.0, ...}` | E0: route the field to MOVING regions only (flow-as-sensor inverse of the tracking gate) |
| `recency` variant | `deform_replay_recency: true` | recency-weighted replay — **the one campaign win on the field**: pin-EPE +52.5→+62.8%, cos +0.79→+0.85 |

## 3. Components (all preserved on `diagnosis-live`)
- **Baker** `Addons/deform/generate_deform_targets.py` — DINO correspondence + depth back-proj → `Δx* = X_ref − X_k`, stored as a grid (`--grid_scale 3`; scale-1 storage too coarse = FAIL). ~50× sped up (upsampled-grid matmul + parabola).
- **Regauge** `Addons/deform/regauge_deform_targets.py` — rotate Δx* by the SLAM pose gauge R (bake=identity vs SLAM=const diag(1,−1,−1)); idempotent. Fixed the "wrong-frame field" (cos −0.64 → +0.78). See `project_deform_gauge_bug_20260620`.
- **Validate** `Addons/deform/validate_deform_targets.py` — target sanity (reduction/cos at held-out pins).
- **Judge** `Addons/eval/field_warped_pin_epe.py` — the ONLY field-sensitive metric (render/ATE are field-blind). Field-warped pin EPE + shuffled-time control + |Δx| + cos(D,baked) + anchor check; ships a 6-panel PNG. ⚠️ measures in the depth scale → its % is **scale-relative, not metric** until depth is anchored.
- **Model hooks** (`ddsslam.py`, `model/scene_rep.py`, all flag-gated): `deform_teacher_loss`, `deform_field_teacher_only`, `deform_teacher_only`, `_deform_replay_step` + `field_optimizer` (separate field optimizer), `deform_replay_recency`, `cur_frame_map_only`, `deform_field_teacher_only` (run_network detach).
- **Runbook** `Addons/colab/teacher_ab_t4_20260619.sh` — Drive-persisted bake → parity → SMOKE → n=3.
- **Configs** the 7 chain + 14 route/fix variants under `configs/Super/`.

## 4. Results
- **Field REVIVED + generalises** (not point-reproduction): held-out pin-EPE +52.5%, recency +62.8%, cos(D,baked) +0.79→+0.85, |Δx| 0.0077→0.0090. shuffled-time control beaten (+35% < +52.5%) → time-specific.
- **Render trade-off** (the un-routed cost): dead-field 27.70 → un-routed 24.1 → sharp (map-only) 24.1→ campaign best (sb2_edge) 26.0. NEVER reached do-no-harm (≥27.70).
- **E0 routing campaign (12 arms, seed0)**: marginal. edge-mask top (+0.15), B2 (+0.07), surf-bind sb1/2/4 **INERT** (σ=k·trunc·sc ≥ render band → no gating; need k=0.25/0.5), hardbound OUT (clips deform), recency = the field win. Full ranking in `project_combine_routing_model_20260619`.

## 5. WHY it under-delivered (NOT a teacher flaw — read this before resuming)
`DEPTH_SCALE_DEFORMATION_ROOTCAUSE_20260622.md` (wf w105m3lde): MoGe depth is **consistent global-scale**
(CV 2.91%, not per-frame jitter), so the field learns `s × real` **faithfully** — pin-EPE +52.5% is
**correct in the WRONG units**. The damage is a **render-gate mismatch**: the `s×`-too-large `vox_motion`
(max|Δx*|=0.084 = 84% of the ±0.1 band, implausible for sub-mm tissue) is pushed through METRIC-calibrated
gates (`trunc 0.1`, `range_d 0.1`, `surf_w`, `depth_trunc 5`) → over-warp + mis-gated surf_w + (if
target_d>5) silenced depth-loss. **The teacher is the wire that carries the mis-scaled depth into the
field** — base avoids it only by having NO external target (and paying with a dead field). Every routing
fix tuned WHERE the field applies, never its MAGNITUDE (set upstream by the depth). So the campaign's
ceiling was structural.

## 6. Lessons learnt (durable)
1. **The field can be revived** — render-only starvation is reversible with an external Δx* target; the field GENERALISES (held-out pins), it is not memorising sample points.
2. **Isolate the field from the render optimizer** (`deform_field_teacher_only`) or the render collapses it to 0; co-adapt **MAP→field**, never field→render.
3. **Replay beats frame-k-only** for time-specificity; **recency-weight** it (the one clean field win).
4. **pin-EPE is the only field-sensitive judge** — render & ATE are field-blind; but pin-EPE % is scale-relative until depth is metric.
5. **Routing fixes placement, not magnitude.** A mis-scaled field cannot be fixed downstream of the bake.
6. **surf_w gating needs σ < range_d** (k=0.25/0.5, not 1/2/4 — those are inert).
7. **A depth-scale mismatch corrupts the field via the teacher, the static render only mildly** — the field is the scale-sensitive consumer.

## 7. How to RESUME (when we return)
1. **Pre-req fix** (the open blocker): make the field's targets band-consistent — EITHER
   **(A) anchor MoGe depth to metric before baking** (`generate_depth_moge --ref <stereo metric>`, re-bake Δx*; CRCD has a clean stereo anchor) OR **(B) scale-free contrast teacher** (supervise with a render-space rigid-vs-deform contrast instead of an absolute Δx* — base-faithful, immune to scale). Decision rule in the rootcause doc §5.
2. Re-enable the chain: `trail3_teacher_replay_sharp_route_recency` (recency = the field win) on the anchored bake; keep hardbound OFF, surf-bind at k=0.5 (or off).
3. Judge: `field_warped_pin_epe.py` (now in metric) + LIVE render. Pass = pin-EPE alive AND render ≥ base.
4. The proof testbed is **STIR** (real deformation GT, endpoint-EPE) — SemSup is only the do-no-harm guard.

**Nothing is deleted. Every flag defaults off → base run is byte-identical (parity gate
`test_inc0_bitidentical.py`). To return: flip the flags in the configs above.**
