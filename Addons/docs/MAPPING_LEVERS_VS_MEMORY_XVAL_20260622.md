All verdicts confirmed against memory line 41: voxel_sdf 0.001@hash16 (788mm), depth_weight↑ (SDF 3000:1), smooth_weight↑ (doesn't regularise field), Recipe A demoted. The cross-validation results are accurate. I'll now synthesize the integrated doc.

# Mapping Levers x Project Memory: Cross-Validation

> Integration of 4 cross-validation clusters (CLUSTER 1 scheduling/starvation, CLUSTER 2 capacity/config, CLUSTER 3 loss-weighting/lr, CLUSTER 4 why-map-wins mechanism) against project memory. Source docs: `DDS_MAPPING_MECHANISM_20260622.md`, `project_mapping_free_wins_audit_20260621`. Verdicts spot-checked against `project_graveyard_20260613.md` (lines 20, 33, 41, 42, 61).

## Master table

| Lever / Claim | Memory verdict | Evidence (memory file) |
|---|---|---|
| **C1a** `cur_frame_iters` 0→100 (curmap100) on CRCD | **NOVEL** (no run report) | `project_mapping_free_wins_audit_20260621` (Tier-2, DEFERRED; CRCD 0 = StereoMIS copy artifact @096d253) |
| **C1b** `map_every`/`keyframe_every`/`iters` tuning | **CONFIRMED** (quantified, deferred) | `project_mapping_free_wins_audit_20260621` (CRCD ~8k rays/frame = 75× under SemSup); `project_dds_slam_config_knob_deepdive_20260604` (kf_every=1 short-seq only) |
| **C1c** Dead co-vis windowing == FM3 dead code | **ALREADY-WON** (already diagnosed) | `project_dds_slam_failure_mode_audit_20260608` (FM3: `sample_overlap/global_keyframe` never called) |
| **C1d** Scheduling levers in graveyard? | **NOVEL** (not in graveyard; deferred not dead) | `project_graveyard_20260613` (map-throttle absent; only "STARVATION=root, teacher=fix") |
| **C1** STARVATION root cause | **CONFIRMED** (independent) | `project_graveyard_20260613` L61; `project_field_warped_pin_epe_verdict_20260618` |
| **C1** Cause-1 time_net starves field (shared map_optimizer) | **CONFIRMED** | `project_field_warped_pin_epe_verdict_20260618` (global_BA collapses field to 0 before teacher) |
| **C2a** `hash_size` 16→19 | **ALREADY-WON** | `project_paper_recreation_breakthrough_20260602` (30.148 PSNR); `project_graveyard_20260613` L33 |
| **C2b** MLP width/depth (`hidden_dim` 32→64, `num_layers` 2→3) | **NOVEL** (decoder capacity never swept) | `project_dds_slam_config_knob_deepdive_20260604` (14 knobs, no hidden_dim/num_layers); capacity proof was TimeNet-only |
| **C2c** `n_range_d` 11→16 (paper typo) | **ALREADY-WON** (+typo CONFIRMED) | `project_paper_recreation_breakthrough_20260602`; `project_dds_slam_config_knob_deepdive_20260604` (Rank-3 ship-now) |
| **C2d** `voxel_sdf`=0.001 (needs hash19) | **CONFIRMED** (locked pairing, already in force) | `project_dds_slam_config_knob_deepdive_20260604` (0.001@hash16 = 788mm; only hash19+) |
| **C2e** Scale `trunc`/`range_d`/`far` to confine warp | **CONTRADICTS** (UNRESOLVED) | `project_nerf_slam_tuning_rules`; `project_graveyard_20260613` L42 (scaled-all-4 → ATE 2.20→5.80mm) |
| **C3a** `depth_weight` UP (sweep [0.01,1]) | **GRAVEYARD** | `project_graveyard_20260613` L41 (0.1→0.5/5.0, SDF dominates 3000:1, zero effect) |
| **C3b** `smooth_weight` UP (1e-6→1e-5) | **GRAVEYARD** | `project_graveyard_20260613` L41 (spatial-TV does NOT regularise time_net; wiring on SDF only) |
| **C3c** `lr_trans` 1e-3→1e-4 (both mapping+tracking) | **ALREADY-WON** | `project_lr_fix_breakthrough_20260429` (BC0: −39% ATE 70.6→43.1mm); backlog **#0** |
| **C3d** `sdf_weight`:`rgb` ratio sweep (1000:5) | **NOVEL** (never swept) | no memory entry; frozen across all runs |
| **C3 lever-4** "loss reweighting: sweep depth_weight" | **GRAVEYARD** (depth-specific) | `project_mapping_free_wins_audit_20260621` ("DON'T CHASE: depth_weight↑") |
| **C3c'** `lr_rot` also 1e-3→1e-4? | **CONFIRMED (partial)** | `project_dds_slam_config_knob_deepdive_20260604` (Rank-4 "all 4 LRs together", sweep-candidate not ship-now) |
| **C4** Dangling knobs (`time_smoothness`/`plane_tv`/`l1_time_planes`) never read | **CONFIRMED** | `project_upgrade_brief_review_20260612`; `project_dds_slam_failure_mode_audit_20260608` (6 dead knobs) |
| **C4** `def_reg`=mean(Δx²) function-space, weight=0 default | **CONFIRMED** (mechanism correct, keep) | `project_upgrade_brief_review_20260612` (scene_rep.py:185) |
| **C4** Map absorbs deformation (capacity asymmetry) | **ALREADY-LOST** (contributing, NOT sole) | `project_graveyard_20260613` L20 (gridslow still dead) |
| **C4** Field revived via `deform_field_teacher_only` + teacher | **ALREADY-WON** | `project_field_warped_pin_epe_verdict_20260618`; `project_deform_gauge_bug_20260620` (+67% pin-EPE) |
| **C4** Indirect field grad decays w/ SDF slope (Cause-4) | **CONFIRMED** | `project_upgrade_brief_review_20260612`; `project_paper_vs_code_deformation_20260618` |
| **C4** Un-routed warp smears bg/tool/tissue (Cause-3) | **CONFIRMED** | `project_deform_gauge_bug_20260620` ("render recovery = ROUTING") |
| **C4 Lever-1/B2** thread `route_w_ba` into global_BA | **NOVEL** (not built) | `project_combine_routing_model_20260619`; `project_deform_gauge_bug_20260620` (co-adapt map→field) |
| **C4 Lever-2** `deform_surface_bind` Gaussian falloff | **ALREADY-WON** (built, in `replay_sharp`) | `project_deform_gauge_bug_20260620` (L29 cur_frame_map_only) |
| **C4 Lever-5** teacher_only + sup_weight>0 + replay | **ALREADY-WON** | `project_field_warped_pin_epe_verdict_20260618` (b0ef28e); `project_deform_gauge_bug_20260620` |
| **C4 Lever-6** keyframe windowing / recency (FM3) | **NOVEL** (dead code, not yet wired) | `project_mapping_free_wins_audit_20260621` (Tier-2); `project_dds_slam_failure_mode_audit_20260608` |

---

## (A) GRAVEYARD HITS — DO NOT RETRY (most important, listed first)

These were proposed by the mapping doc but memory has already killed them. Any EV the doc assigns here is **phantom EV**.

1. **`depth_weight` UP (doc §4 lever-4 "sweep depth_weight ∈ [0.01,1]")** — **DEAD.** Tested 0.1→0.5 **and** 0.1→5.0; **zero effect** because the SDF loss dominates the depth loss ~**3000:1**. The entire "loss reweighting" Tier-2 lever collapses to a no-op in its depth-specific form. `project_graveyard_20260613` L41; `project_mapping_free_wins_audit_20260621` ("DON'T CHASE: depth_weight↑").

2. **`smooth_weight` UP (1e-6→1e-5)** — **DEAD + WIRING BUG.** Spatial total-variation is applied only to the **static SDF grid** at global_BA, never to `vox_motion`/`time_net`. Raising it cannot regularise the deformation field by construction. (This is *why* function-space `def_reg` exists.) `project_graveyard_20260613` L41; `DDS_MAPPING_MECHANISM_20260622.md` §2 confirms the wiring.

3. **`voxel_sdf`=0.001 at hash16** — **CATASTROPHIC.** ATE 2.20→**788 mm** (358× worse). 0.001 is *only* viable when paired with **hash19** (where it is already in force on CRCD). Do not propose 0.001 as a standalone capacity lever. `project_dds_slam_config_knob_deepdive_20260604`; `project_graveyard_20260613` L41.

4. **Recipe A (`first_iters` 200→1000, `iters` 10→20 as a *tracking* win)** — **DEMOTED/DROPPED.** Tracker regression; worth ~5% vs the depth-source ~90%. (Note: raising `iters` as a *mapping/render* budget lever is distinct and still live — see C1b — but do not resurrect Recipe A's tracker framing.) `project_graveyard_20260613` L41.

5. **Map-absorption as the *sole* cause (doc Cause-5 capacity asymmetry, if read as dominant)** — **ALREADY-LOST as sole lever.** `gridslow` (throttled map) left the field dead. Map absorbing temporal variation is a **contributing structural factor, not the root**. The doc is safe *only* if Cause-5 stays secondary/contextual (which it does). `project_graveyard_20260613` L20.

6. **GAUGE-RACE (adjacent, if it resurfaces)** — refuted by pose-freeze no-revive; the gauge issue was a separate bake-vs-SLAM frame mismatch, already fixed via `regauge_deform_targets.py`. Distinct from scheduling. `project_deform_gauge_bug_20260620`.

---

## (B) ALREADY-WON — banked, so the doc's EV is double-counting

The doc lists these as opportunities; memory shows they are **already in the shipped config or already validated**. Do not re-budget runs for them.

- **`hash_size` 16→19** — banked (30.148 PSNR). On CRCD this is **already correct** in `crcd.yaml`. `project_paper_recreation_breakthrough_20260602`; `project_mapping_free_wins_audit_20260621`.
- **`n_range_d` 11→16** — banked, and the "11" is a confirmed paper typo. `project_paper_recreation_breakthrough_20260602`; knob_deepdive Rank-3.
- **`lr_trans` 1e-3→1e-4 (both sides)** — banked as backlog **#0** (−39% ATE). Highest-EV lever already known. `project_lr_fix_breakthrough_20260429`.
- **Field revival via teacher (`deform_field_teacher_only` / Lever-5)** — banked: field **ALIVE, +67% pin-EPE**. `project_field_warped_pin_epe_verdict_20260618`; `project_deform_gauge_bug_20260620`.
- **`deform_surface_bind` (Lever-2)** — built, default-off, already used in `trail3_teacher_replay_sharp`. `project_deform_gauge_bug_20260620`.
- **Dead co-vis windowing identified (FM3)** — the *diagnosis* is banked (06-08); only the *wiring* remains (see C below). `project_dds_slam_failure_mode_audit_20260608`.

---

## (C) GENUINELY NOVEL / UNTRIED — the real opportunities

These survive the graveyard and are **not yet banked**. This is where 48h of mapping EV actually lives.

1. **`cur_frame_iters` 0→100 on CRCD (curmap100)** — **NOVEL, principled, paper-faithful, never run.** CRCD inherited `0` purely as a StereoMIS copy artifact (@096d253); raising to the authors' surgical value (100) applies their surgical schedule to our surgical data. WildGS pair already went to 10. Tier-2 free-win, A/B-gated. `project_mapping_free_wins_audit_20260621`.
2. **Keyframe windowing / recency wiring (Lever-6, FM3)** — dead code (`sample_global_keyframe`/`sample_overlap_keyframe`) exists but is **never called**; live loop uses uniform `sample_global_rays`. Wiring it in is the lowest-effort render-tail lever. `project_mapping_free_wins_audit_20260621`; `project_dds_slam_failure_mode_audit_20260608`.
3. **`route_w_ba` into global_BA (Lever-1/B2)** — **not built** (~15 lines, parity-safe). Co-adapts the **map to the field** (the correct direction per gauge-bug finding), targets the un-routed-smear render cost (Cause-3). This is the combine bridge, not a tuning knob. `project_combine_routing_model_20260619`; `project_deform_gauge_bug_20260620`.
4. **Decoder MLP capacity (`hidden_dim` 32→64, `num_layers` 2→3)** — **NOVEL.** Prior capacity work was TimeNet-only; the SDF/color **decoder** width/depth was never swept (absent from the 14-knob audit). `project_dds_slam_config_knob_deepdive_20260604`.
5. **`sdf_weight`:`rgb` ratio sweep (1000:5)** — **NOVEL.** Frozen at 200:1 across every run; never A/B'd. Note: lower-confidence — adjacent to graveyard depth/smooth-weight nulls, so SDF dominance (3000:1 over depth) suggests the SDF arm may swamp any rgb move; validate small before budgeting n=3.

---

## (D) CONTRADICTIONS — doc vs memory, and which to trust

**Only one true contradiction (C2e).** The doc proposes scaling `trunc`/`range_d`/`far` (e.g. `deform_surface_bind`-style confinement) to lock the warp to a narrow surface band for render sharpness. Memory's `project_nerf_slam_tuning_rules` says **DO NOT scale these four to scene** — they are **pose regularisers**, and Run-3A scaling all four together **regressed ATE 2.20→5.80 mm**. `project_graveyard_20260613` L42 flags this as an explicit **OPEN CONTRADICTION — do not auto-resolve**.

**Resolution:** these are *orthogonal* objectives — render wants the band **narrow**, the tracker wants it **wide**. **Trust memory (KEEP `nerf_slam_tuning_rules`)** as the default: do **not** blanket-scale `trunc`/`range_d`/`far`. The doc's surface-binding is acceptable **only** if it confines the *deformation warp* via a separate Gaussian falloff (`deform_surface_bind`) **without** touching the global `trunc`/`range_d`/`far` sampling that the tracker depends on. Resolving the conflict requires the **decoupled-threshold experiment** (separate render-band vs pose-band thresholds) — until then, do not couple them. No other doc claim contradicts memory; CLUSTER 4 is a faithful synthesis of pre-06-20 state, not a new discovery.

---

## REVISED lever ranking for the 48h mapping win (graveyard-respecting)

Spend the 48h on the **NOVEL, untried, parity-safe** levers and skip everything memory has banked or buried. **Rank 1: `cur_frame_iters` 0→100 on CRCD (curmap100)** — the single highest-EV untried mapping move, principled (authors' own surgical schedule), config-only, A/B n=3 against the current CRCD canon. **Rank 2: wire FM3 keyframe windowing/recency** (`sample_overlap/global_keyframe`) — lowest-effort render-tail lever, dead code already written. **Rank 3: `route_w_ba`/B2 into global_BA** (~15 lines, parity-safe) — attacks the un-routed-smear render cost and doubles as the combine bridge. **Rank 4 (cheap probe, not a headline): decoder MLP `hidden_dim` 32→64** — genuinely untried capacity, but unproven EV. **Explicitly OUT of the 48h:** any `depth_weight`/`smooth_weight` sweep (GRAVEYARD, zero/negative), `voxel_sdf` 0.001 standalone (788 mm), Recipe A's tracker framing, and any blanket `trunc`/`range_d`/`far` scaling (open contradiction, regressed ATE) — and **do not re-budget** `hash19`, `n_range16`, `lr_trans 1e-4`, or the teacher revival, which are already banked. Keep `lr_trans 1e-4` (backlog #0) as the standing tracking baseline these mapping A/Bs run *on top of*, not as a new spend.