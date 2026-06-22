Both confirmed. Synthesizing the doc now.

# DDS-SLAM Render Maximisation (field-dead): Ranked Lever Plan

> Scope: 48h, deformation field LOCKED DEAD (per constraint). Arbiter = live render PSNR/SSIM/LPIPS (+ depth-L1 self-consistency) and CRCD Sim3 ATE. Pristine `DDS-SLAM-Base` is the eternal reference; every arm runs FROM base in isolation. Graveyard is absolute.

---

## 1. TL;DR — top 5 levers for the 48h, in order

| # | Lever | Type | Metric(s) | EV |
|---|---|---|---|---|
| 1 | **curmap100** — `cur_frame_iters 0→100` on CRCD | CONFIG (fair-baseline) | PSNR/SSIM/LPIPS (live) | **+1.0–2.5 PSNR** — single biggest untried mapping move; closes a 75× current-frame budget deficit by applying the authors' own surgical schedule (Super/SemSup use 100) to surgical CRCD data. Config exists: `c1_001_canon_curmap100.yaml`. |
| 2 | **decoder64/3** — `hidden_dim 32→64`, `num_layers 2→3` on the static SDF+color MLP | CONFIG | PSNR/SSIM/LPIPS | **+0.5–1.0 PSNR** (CRCD). Only untested capacity knob on the static map; prior audits swept only TimeNet. Monotonic, parity-safe, no RNG impact. Orthogonal to curmap100 → run as a crossed factor in the SAME arm. |
| 3 | **flow_agree_baf n=3 lock** (already-won contribution) | CONTRIBUTION | **Sim3 ATE** (+ Pearson), render preserved | **−70% ATE** (2.73→0.82mm, n=1), Pearson 0.866→0.990, render preserved (22.07→22.2). The publication tracking win — only thing missing is the n=3 seed lock to clear the seed-std floor. |
| 4 | **FM3 keyframe windowing** — wire dead `sample_global_keyframe`/`sample_overlap_keyframe` | CONFIG-wire (≈3–20 lines) | PSNR/SSIM (tail frames) | **+0.2–0.5 PSNR**. Lowest-effort render-tail polish; dead code exists (`keyframe.py:100–192`), only the call-site swap at `ddsslam.py:599` remains. Run AFTER the curmap100 baseline is set. |
| 5 | **edge_semantic on/off ablation** + **deformation_off ablation** (two cheap n=1 probes) | CONFIG | PSNR/SSIM/LPIPS | **±0.2 PSNR each**, but decisive diagnostics: edge_semantic may be a capacity sink (`ddsslam.py:273–274`); `deformation_off=True` (`scene_rep.py:202`, the SANCTIONED kill — NOT broken `dynamic:False`) confirms the dead warp costs nothing. One trial each, no retrain-of-suite. |

Levers 1+2 are the headline render spend (crossed in one A/B). Lever 3 is the headline ATE spend (a lock, not a build). Levers 4–5 are cheap polish/diagnostics.

---

## 2. MASTER TABLE — every lever

| Lever | Type | Metric(s) | EV | Memory | Evidence (file:line) |
|---|---|---|---|---|---|
| **1a curmap100** `cur_frame_iters 0→100` | CONFIG | PSNR/SSIM/LPIPS; ATE neutral+ | +1.0–2.5 PSNR | **NOVEL** | `crcd.yaml:25`; `ddsslam.py:390–404` (early-return `<=0`); cfg `c1_001_canon_curmap100.yaml`; `mapping_free_wins_audit_20260621` Tier-2 #3 |
| 1b schedule density `map_every 5→1`, `iters 20→?`, `keyframe_every 5→1` | CONFIG | PSNR/SSIM/LPIPS; ATE 2° | +0.5–1.5 PSNR (risky) | CONFIRMED-deferred | `ddsslam.py:508–643,521,555`; `keyframe.py:85–98`; `crcd.yaml:24,30,31`. **iters LOCKED 20** (Recipe-A regressed pose) |
| 1c FM3 keyframe windowing (wire recency/co-vis) | CONFIG-wire / ARCH | PSNR/SSIM (tail); ATE minor | +0.2–0.5 PSNR | **NOVEL (dead code)** | `keyframe.py:100–192`; `ddsslam.py:599,562–564` (uniform call site); `failure_mode_audit_20260608` FM3 |
| 1d `min_pixels_cur` current-frame injection | CONFIG | PSNR temporal; ATE minor | +0.1–0.3 PSNR (low conf) | NOVEL | `ddsslam.py:567`; `crcd.yaml:35` |
| **2a decoder64/3** `hidden_dim 32→64`,`num_layers 2→3` | CONFIG | PSNR/SSIM/LPIPS | +0.5–1.0 PSNR | **NOVEL** | `decoder.py:55–103,203–254`; `crcd.yaml:84–85`; `config_knob_deepdive_20260604` (no decoder in 14-knob audit) |
| 2b hash `n_levels`/`level_dim` | CONFIG | PSNR/SSIM/LPIPS | ~0 (exploratory) | NOVEL low-pri | `encodings.py:31–46`; `scene_rep.py:50`; `crcd.yaml:66` (only `hash_size` was tuned) |
| 2c **importance sampling** `n_importance>0` | ARCHITECTURAL | PSNR/SSIM | +0.3–0.7 (speculative) | **GRAVEYARD** | `scene_rep.py:452–468` (wrong arity, `edge_map` undefined); `crcd.yaml:122`; `graveyard_20260613:41` |
| 2d-i `trunc`/`sc_factor` SDF→weight width | CONFIG | PSNR/SSIM/LPIPS | — (locked) | **ALREADY-WON + CONTRADICTS** | `scene_rep.py:90–110`; `crcd.yaml:125`; `graveyard:42`; `nerf_slam_tuning_rules` (do NOT scale to scene; Run-3A regressed 2.20→5.80mm) |
| 2d-ii `white_bkgd` | CONFIG | PSNR/SSIM/LPIPS | −0.1–0.3 (likely neg) | NOVEL low-conf | `scene_rep.py:153–154`; `crcd.yaml:124` |
| 2d-iii **view-dir input to ColorNet** | ARCHITECTURAL | LPIPS/SSIM | +0.2–0.5 LPIPS (spec.) | NOVEL | `decoder.py:55–103` (no view_dir); `scene_rep.py:178` run_network; `ddsslam.py:550` |
| 2e OneBlob `n_bins` | CONFIG | PSNR/SSIM | 0–0.2 (marginal) | NOVEL low-pri | `encodings.py:61–71`; `crcd.yaml:80` |
| **3a flow_agree_baf** (cam/scene split + agreement gate + freeze_ba) | CONTRIBUTION | **Sim3 ATE**; Pearson; render preserved | −70% ATE (2.73→0.82mm n=1) | **ALREADY-WON** | `Addons/motion/flow_track.py`; `ddsslam.py:770–805,677–681`; commits 0ef03d5,b8d0e9b; `combine_routing_model_20260619` |
| 3b `lr_trans 1e-3→1e-4` (all-4 together) | CONFIG | ATE; PSNR 2° | −39% ATE (Co-SLAM StereoMIS); CRCD transfer open | **ALREADY-WON (StereoMIS)** | `ddsslam.py:503–504`; `crcd.yaml:28–29,45–46`; `lr_fix_breakthrough_20260429`; `knob_deepdive` Rank-4. **NEVER single-side** |
| 3c flow-routed mapping (E0 route) | CONTRIBUTION | PSNR/SSIM/LPIPS; pin-EPE; ATE | dead-field +0–0.5; alive +1.5–2.5 (needs B2) | **NOVEL (GATED on field alive)** | `flow_track.region_route`; `DDS_MAPPING_MECHANISM_20260622 §4`; E0 v0.4 render INVERTED (25.83<27.70) → un-co-adapted warp HURTS |
| 3d tracking iters/schedule (`iter=10`,`pose_accum_step=5`,`wait_iters=100`) | CONFIG | ATE; PSNR; Pearson | unknown, sub-noise on CRCD | NOVEL | `crcd.yaml:42`; `ddsslam.py:823–882` |
| 3e tracking_render sub-pixel (`ignore_edge`,`UseBorder`,`track_ray_w`) | CONTRIBUTION (legacy) | ATE; PSNR edges | <0.1mm ATE (sub-noise) | NOVEL (superseded by gate) | `ddsslam.py:761,829–837,855–856`; `flow_track.py:193–199`; geo+flow does NOT stack |
| **B2 route_w_ba** thread into `global_BA.forward` (co-adapt map to field) | ARCHITECTURAL | PSNR/SSIM/LPIPS | +1.5–2.5 (alive); **0 on dead field** | **ALREADY-WON (built, inert)** | `ddsslam.py:560–578,634`; `keyframe.py:16–18,80–83`; `scene_rep.py:520–526`; `DDS_MAPPING_MECHANISM §5` (~15 lines, default-off). **route_ba:False** |
| B2b `deform_surface_bind>0` (Gaussian falloff) | CONFIG | PSNR/SSIM/LPIPS | +0.5–1.0 (alive) | **ALREADY-WON (in replay_sharp)** | `scene_rep.py:237–238,430–436`; `deform_gauge_bug_20260620` |
| C1a `sdf_weight:rgb` ratio sweep (1000:5) | CONFIG | PSNR/SSIM/LPIPS; depth-L1 | +0.2–0.5 (low conf) | NOVEL (graveyard-adjacent) | `ddsslam.py:264,268`; `graveyard:41` (depth_weight↑ = 0 effect, SDF dominates 3000:1) |
| C1b SSIM/LPIPS term in mapping loss | ARCHITECTURAL | SSIM/LPIPS | +0.02–0.05 SSIM (unstable) | NOVEL | `eval_rendering.py:17–43` (post-hoc only); `ddsslam.py:264`; `scene_rep.py:602` (MSE only) |
| C2 depth-L1 (consequence, not lever) | — (metric) | depth-L1 | ~5–20mm ↓ per +1dB PSNR | CONFIRMED | `Addons/eval/depth_l1.py:1–31,78–147`; self-consistency, no GT |
| D1 `edge_semantic_loss` on/off ablation | CONFIG | PSNR/SSIM/LPIPS | ±0.2 (diagnostic) | GRAVEYARD-adjacent | `ddsslam.py:273–274`; `scene_rep.py:607–613`; `project_edge_semantic_weight.md` |
| D2 `deformation_off=True` ablation | CONFIG | PSNR/SSIM/LPIPS; ATE | unknown direction (diagnostic) | NOVEL | `scene_rep.py:202` (sanctioned kill); `feedback_colab_shared_box_isolation_20260618` (`dynamic:False` BROKEN) |
| **Input-depth denoise + per-snippet sc_factor** | CONTRIBUTION | PSNR/SSIM/depth-L1 (joint) | HIGH (depth-scale ~90% of ATE win) | **NOVEL** | `crcd_c1_001_SM_chain_results_20260604:30–47`; `depth_multiplier_math_audit_20260604`; `feedback_depth_scale_verify`; `depth_l1.py:120` |
| hash19 + n_range_d16 | CONFIG | PSNR/SSIM/LPIPS | +0.9 PSNR, −0.034 LPIPS | **ALREADY-WON** | `paper_recreation_breakthrough_20260602`; `crcd.yaml:66,121` (shipped) |
| Sim3 ATE (scale-aware) vs rigid output.txt | CONTRIBUTION | ATE | HIGH (rigid INVERTS A/Bs) | **ALREADY-WON** | `Addons/eval/sim3_ate.py:52–91`; `feedback_sim3_ate_misleading_subsnr:112–129` |
| Render masking / per-region PSNR | ARCHITECTURAL | PSNR/SSIM/LPIPS | diagnostic (CRCD LPIPS 0.52 = structural ceiling) | CONFIRMED | `eval_rendering.py:162–194`; `mapping_free_wins_audit:36` |
| Frame mis-pairing guard (271-vs-360 GT) | ARCHITECTURAL | ATE | HIGH (silent ATE break) | CONFIRMED | `feedback_sim3_ate_misleading_subsnr:28,127–128`; `eval_rendering.py:126–134`; `sim3_ate.py:72–77` |

---

## 3. The two-column split (attributability)

The win MUST be attributable: a CONFIG-tuning win is a fair-baseline correction (not a thesis claim); a CONTRIBUTION/ARCHITECTURAL win is ours. Keep them in separate arms so the headline number is cleanly assigned.

### Column A — CONFIG-tuning levers (fair baseline; NOT our contribution)
- **1a curmap100** `cur_frame_iters 0→100` — +1.0–2.5 PSNR (authors' own schedule). **← the 48h headline render move**
- **2a decoder64/3** `hidden_dim 32→64`,`num_layers 2→3` — +0.5–1.0 PSNR
- 1c FM3 windowing wiring — +0.2–0.5 PSNR (dead code, config-gated activation)
- 3b `lr_trans 1e-4` (all-4) — already-won on StereoMIS; CRCD transfer open
- hash19+n_range16 — already shipped baseline
- 1b schedule density (DEFERRED — iters LOCKED, map_every risky)
- C1a sdf:rgb ratio (graveyard-adjacent, cheap n=1 probe only)
- D1 edge_semantic, D2 deformation_off — cheap n=1 ablations/diagnostics
- 2b/2e/2d-ii hash dims/OneBlob/white_bkgd — low-priority micro-tuning

### Column B — OUR-CONTRIBUTION / ARCHITECTURAL levers (attributable wins)
- **3a flow_agree_baf** — −70% ATE, Pearson→0.99. **← the 48h headline tracking move (LOCK n=3)**
- sim3_ate.py (scale-aware metric) — already-won, the correct ATE
- **Input-depth denoise + per-snippet sc_factor** — joint PSNR+depth-L1 (NOVEL, future spend)
- B2 route_w_ba co-adapt — built, INERT (route_ba:False); +0 on dead field
- B2b deform_surface_bind — already-won in replay_sharp; alive-field only
- 3c flow-routed mapping (E0) — NOVEL, GATED on field alive
- C1b SSIM/LPIPS mapping term, 2d-iii view-dir, 2c importance — speculative/architectural, defer

**Attribution rule:** run curmap100 (Column A) FIRST to set the fair baseline, THEN any Column-B render contribution is measured as a delta on top of curmap100 — otherwise a config deficit gets miscredited to our method.

---

## 4. PER-METRIC map — single best lever each

| Metric | Best lever (48h) | Why / numbers |
|---|---|---|
| **PSNR** | **1a curmap100** (CONFIG) | +1.0–2.5 PSNR; 75× current-frame budget deficit is the dominant under-training cause. Backup: 2a decoder64/3 (+0.5–1.0). |
| **SSIM** | **1a curmap100** (then 2a decoder64/3) | SSIM tracks PSNR via map sharpness; same starvation bottleneck. |
| **LPIPS** | **hash19 (ALREADY-WON)**; 48h-new = 2a decoder64/3 | LPIPS is hash-collision sensitive — hash16→19 closed 28% of the gap (−0.034). CRCD LPIPS 0.52 is a structural ceiling (architecture, not a knob). Speculative: 2d-iii view-dir. |
| **depth-L1** | **Input-depth denoise + correct sc_factor** (CONTRIBUTION) | depth-L1 is input-vs-output self-consistency; it follows render sharpness (~5–20mm ↓ per +1dB PSNR) and input-depth quality. NO separate loss term (depth_weight↑ = graveyard null). 48h-cheap proxy: curmap100. |
| **ATE (Sim3)** | **3a flow_agree_baf (ALREADY-WON, lock n=3)** | −70% (2.73→0.82mm), Pearson 0.866→0.990. Backup CONFIG: 3b lr_trans 1e-4. Always report with seed-std + path-ratio + Pearson (sub-SNR). |

---

## 5. GRAVEYARD guardrail + ALREADY-WON

### DO NOT TOUCH (graveyard — absolute)
- **`n_importance` — keep 0.** Dead/broken code (`scene_rep.py:463–464` wrong arity, `edge_map` undefined). Re-wiring = 48h+ of testing for a speculative SDF-system gain. `graveyard:41`.
- **Do NOT scale `trunc`/`range_d`/`far`/`sc_factor` to scene.** Run-3A scaling all four regressed ATE 2.20→5.80mm. These are pose regularisers, not scene knobs. OPEN CONTRADICTION (`graveyard:42`) — do not auto-resolve. Lock `trunc=0.1`.
- **`depth_weight↑` / `smooth_weight↑` — null.** SDF dominates 3000:1; these are dead knobs (`graveyard:41`). C1a sdf:rgb ratio is adjacent-risky → n=1 probe ONLY.
- **`dynamic:False` is BROKEN** — use `deformation_off=True` (`scene_rep.py:202`) for any zero-warp render.
- **`mapping.iters` LOCKED at 20** — Recipe-A (iters bump as a tracking win) regressed pose; preserves paper-faithfulness.
- **Do NOT revive the deformation field** — DEAD by constraint. The whole stabilise-first/battery/dead-field redesign lineage is buried.
- **Never headline rigid `output.txt` ATE** — it INVERTS A/Bs on up-to-scale MoGe depth. Sim3 only.
- **Never drop a single `lr_trans` side** — all-4-together only; single-side regressed.

### ALREADY-WON (don't re-spend)
- **hash_size 16→19 + n_range_d 11→16** — +0.9 PSNR, −0.034 LPIPS; shipped in `crcd.yaml`.
- **sim3_ate.py (scale-aware)** — canonical CRCD metric, wired into runbooks (commit 77cb4a1).
- **`lr_trans 1e-4`** — −39% on Co-SLAM StereoMIS (CRCD transfer is the one open re-check, low priority).
- **flow_agree_baf** — built and validated n=1; only the n=3 lock remains (that IS lever 3).
- **B2 route_w_ba + B2b deform_surface_bind** — scaffolding BUILT, default-off, INERT. **Do NOT activate in 48h** (field is dead → +0). Keep as infra; defer to post-field-revival (gated behind teacher + depth-anchor, out of 48h scope).
- **GT-source guard**: eval CRCD against CRCD-Published 360-row GT, never the stale repo 271-row subset (silent ATE break).

---

## 6. Concrete RUN PLAN

**Reuse** `Addons/colab/dino_ab_rawleft_isolated_20260618.sh` (raw-left is the locked working pipeline; the runbook — not the config — decides raw-left). Stage CRCD-Published (360 GT). Every arm runs FROM pristine base in isolation, flags default-off elsewhere. Both diagnostic sets ship by default (metrics table + 6-panel inline video).

### Phase 0 — cheap n=1 diagnostics (parallel, ~minutes each, no suite retrain)
Run on CRCD c1_001 raw-left, n=1:
- `deformation_off=True` vs base — confirm dead warp costs ~0 PSNR (sanity: validates "field dead" assumption).
- `edge_semantic` on vs off — is the head a capacity sink?
- (optional) C1a `sdf_weight ∈ {500,1000,2000}` n=1 — kill-or-keep before spending n=3.

These gate whether Phase 1 wastes budget; none ship a claim.

### Phase 1 — n=1 DECOMPOSE the render headline (crossed 2×2, CRCD c1_001 raw-left)
Two orthogonal CONFIG factors, one crossed sweep, judged on **live** PSNR/SSIM/LPIPS:
| Arm | cur_frame_iters | decoder |
|---|---|---|
| base | 0 | 32×2 |
| curmap100 | 100 | 32×2 |
| decoder64 | 0 | 64×3 |
| curmap100+decoder64 | 100 | 64×3 |

Config `c1_001_canon_curmap100.yaml` exists. n=1 decomposes which factor carries the win and whether they add. **Stop here if curmap100 alone is the whole win** (cheaper to lock).

### Phase 2 — LOCK n=3 (seed coin-flip is real; a win must clear base seed-std)
- **Render:** the Phase-1 winner (expected curmap100, or curmap100+decoder64) at **n=3** on CRCD c1_001 raw-left AND **SemSup trail3** (render-only; SemSup ATE is fictional). Report {PSNR, SSIM, LPIPS, depth-L1} table + 6-panel video.
- **Tracking:** **flow_agree_baf at n=3** on CRCD c1_001 raw-left (this is the lock, the build is done). Report {Sim3 ATE_mean/max, seed-std, est/GT path-ratio, |Pearson|dom} + video. Confirm render preserved (~22.2, no regression).

### Phase 3 — n=1 polish IF render plateaus (low effort)
- **FM3 windowing** (lever 1c): swap `ddsslam.py:599` to `sample_overlap_keyframe` (conservative `n_fixed`), n=1 on the curmap100 baseline. Ship only if it clears the tail-frame render bar.

### Datasets / metrics
- **CRCD c1_001 (raw-left, 360-row Published GT)** = render + ATE (the only dataset giving both).
- **SemSup trail3** = render-only (PSNR/SSIM/LPIPS; ATE fictional/identity).
- Every result = table {Sim3 ATE, PSNR, SSIM, LPIPS (+ depth-L1)} + canonical 6-panel inline video, by default.

### Explicitly NOT in 48h
B2/B2b activation, flow-routed mapping (E0), input-depth denoising, teacher/field revival, importance sampling, view-dir, SSIM/LPIPS mapping loss — all gated on field-alive or are architectural builds beyond budget. Keep their scaffolding inert.

---

**Bottom line:** the dominant 48h render move is **curmap100** (CONFIG, fair-baseline, ~+1.5 PSNR) crossed with **decoder64/3** (CONFIG, novel-but-not-a-claim); the dominant tracking move is **locking flow_agree_baf to n=3** (CONTRIBUTION, already built, −70% ATE). The static-map problem is SCHEDULE STARVATION (80% of CRCD frames get zero mapping), not field routing. Field stays dead; our contribution render levers (B2, E0, depth-denoise) are correctly parked behind field revival.