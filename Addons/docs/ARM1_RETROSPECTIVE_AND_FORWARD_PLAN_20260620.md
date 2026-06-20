# ARM-1 (σ² UNCERTAINTY HEDGE) — RETROSPECTIVE & FORWARD PLAN

**Date:** 2026-06-20 · **Branch:** `diagnosis-live` @ `4cf5d2b` · **Status:** Arm-1-ALONE EXHAUSTED; the next leap is GATED on Arm-2 revival.

This is the single source for *where Arm-1 is and what's next*. Metric-first. Honest about what is gated.

**Arm-1 in one sentence:** a learned aleatoric σ² head (Inc-1, NLL-trained, mapping-only) that down-weights pose in tracking (Inc-2, `clamp(1/σ²).detach()`). It WORKS as a tracking hedge, it is now well-understood, and it has hit its ceiling alone. The remaining leap — making σ² *deformation-aware* — has exactly one lever, the COMBINE with Arm-2, which is gated only on Arm-2's GPU-blocked revival verdict.

---

## OPERATING CONSTRAINT (read first)

From 2026-06-18, **n=1**. No seed replication. Every claim is judged by THREE substitutes, not significance:
1. **Effect-size vs the known noise floor.** The prior n=3 canon (−23%, all non-overlapping) pins seed-std ≈ **0.2–0.3 mm ATE**. A delta bigger than that clears noise (credible); smaller does not headline.
2. **Coherence across independent metrics** — ATE + ATE_max + path-ratio + Pearson + render + the σ²-quality judge all agreeing = signal; one moving alone = noise.
3. **The σ²-quality judge as a 2nd independent measurement** of the same model (the seed-replication substitute). It already predicts tracking, so it is the cross-check re-seeding cannot give.

Writeup rule: n=1 = "indicative"; credibility = effect-size + coherence + cross-check.

**Metric guard (LOCKED):** CRCD tracking = the sub-SNR triple from `Addons/eval/sim3_ate.py` — **Sim3 ATE (Umeyama with scale) + est/GT path-ratio + dominant-axis |Pearson|**. NEVER the rigid `output.txt` (it inverts the A/B on up-to-scale depth). SemSup = render PSNR/SSIM/LPIPS only (pose GT fictional). CRCD is LOCKED **raw-left**; rectified numbers are a SEPARATE reference, never cross-compared.

---

# PART 1 — RETROSPECTIVE

## 1. THE JOURNEY (chronological, with numbers)

Five stages: **canon win → backend A/B → teacher A/B → the judge → fusion-fail.** Each stage's verdict drove the next decision.

### STAGE 0 — Canon build + first win (rectified, n=3) [06-16]
**Tested:** Inc-1 (per-ray σ² head, geo_feat input, NLL `0.5(err²/σ²+log σ²)`, mapping-only) + Inc-2 (pose down-weight `clamp(1/σ²).detach()`). base vs +uncert(geo), CRCD c1_001, pristine base, lr 1e-4, **rectified, n=3**.

| metric | base | +uncert(geo) | Δ | n=3 separated? |
|---|---|---|---|---|
| Sim3 ATE_mean (mm) | 3.15±0.14 | **2.43±0.07** | −23% | yes |
| Sim3 ATE_max (mm) | 14.79±0.00 | **9.43±0.75** | −36% | yes |
| \|Pearson\|dom | 0.81±0.02 | **0.97±0.01** | +20% | yes |
| path-ratio | ~7.0 | ~4.1 | closer to GT | — |

SemSup n=3 render ~28.5 PSNR, uncert ≈ base (hedge is render-neutral).

**Verdict:** the tracking hedge works — Inc-2 cuts the deformation-moment pose spikes (the ATE_max −36% is the signature). Robust n=3 win.
**Decision:** geo(v1) is the canon Arm-1 result. Build v2 DINO (WildGS-faithful) to test whether richer features beat geo. Also LOCKED here: the rigid-ATE trap (`output.txt` inverts; `sim3_ate.py` is the only valid CRCD metric).

### STAGE 1 — Backend A/B: geo vs dino vs dino_reg (raw-left, L2 teacher, n=1) [06-18→06-19]
**Tested:** three σ² backends, all with the **raw photometric-L2 teacher**: geo (geo_feat 15-dim), dino (DINOv2 vits14 384-dim), dino_reg (DINOv2-with-registers). CRCD now raw-left → NOT comparable to the rectified canon.

| cell | ATE mean/max (mm) | path-ratio | \|Pearson\| | PSNR/SSIM/LPIPS |
|---|---|---|---|---|
| geo | 2.29 / 8.74 | **4.20** | **0.975** | 22.61 / 0.723 / 0.493 |
| dino | 2.31 / 7.75 | 6.48 | 0.968 | 22.27 / 0.687 / 0.555 |
| dino_reg | 2.32 / 8.71 | 6.64 | 0.962 | 22.33 / 0.697 / 0.537 |

SemSup render: base 28.51 / geo 28.25 / dino 28.22 / dino_reg 28.19 PSNR → hedge render-neutral (≤0.3 dB cost, small SSIM gain). (`base_crcd_s0` interrupted at 107 frames — no raw-left CRCD base baseline.)

**Verdict:** ATE_mean is a 3-WAY TIE (within n=1 ~0.26 mm noise). **geo wins** — separates ONLY on path-ratio (4.20 vs 6.5–6.6) + Pearson (0.975) + render. **dino_reg ≈ dino → register blotch-cleaning is COSMETIC at the metric level (with the L2 teacher).**
**Decision:** geo is the live raw-left tracker. dino_reg is not justified by features alone — its value would have to ride on the teacher. → motivates Stage 2.

### STAGE 2 — Depth-consistency teacher A/B: geo_rd vs dino_reg_rd (raw-left, n=1) [06-19]
**Tested:** top-ranked Arm-1 fix — swap σ² teacher RGB-L2 → RGB+MoGe-depth-consistency (`uncertainty.teacher`, scene_rep.py:574).

| cell | ATE mean/max (mm) | path-ratio | \|Pearson\| | PSNR/SSIM |
|---|---|---|---|---|
| geo_rd | 2.41 / 8.05 | 5.12 | 0.979 | 22.58 / 0.712 |
| dino_reg_rd | **2.03 / 5.81** | 6.41 | 0.976 | 22.58 / 0.706 |

**Verdict — the teacher is BACKEND-DEPENDENT:**
- **geo_rd ~neutral/slightly worse** than geo-L2 (ATE 2.29→2.41, ratio 4.20→5.12).
- **dino_reg_rd UNLOCKED** vs dino_reg-L2: ATE 2.32→2.03, ATE_max **8.71→5.81 (−33%)**, render +0.25 PSNR, Pearson +0.014. **dino_reg_rd = best raw-left tracker of the whole set.** The max −33% (~3 mm) clears the seed floor → credible; ATE_mean 2.03 vs 2.29 (~0.26 mm) is at-floor → weak alone, carried by coherence with max + the judge.

**Decision:** OVERTURNS "dino_reg cosmetic" — *with* the depth teacher dino_reg becomes a real candidate. The two JOBS want DIFFERENT σ²: mapping-calibration → geo; tracking (the arbiter) → dino_reg+depth. Both kept as separate cells. → triggers building the σ²-quality judge to read WHY.

### STAGE 3 — σ²-quality judge built + validated [06-19]
**Built:** `Addons/eval/sigma2_quality.py` (AUSE / frac_oracle calibration + σ²-vs-inter-frame-motion Pearson = deformation-awareness proxy) + `sigma2_diagnostics.py` (contrast-corr + spatial speckle). The n=1 seed substitute = a 2nd independent measurement of the same model.

| cell | frac_oracle | motion-corr | contrast-corr | speckle |
|---|---|---|---|---|
| geo_rd | **+0.773** | 0.122 | 0.100 | 0.496 |
| dino_reg_rd | +0.476 | **0.249** | 0.283 | **0.018** |

**Verdict — the judge VALIDATED PREDICTIVELY:**
- **frac_oracle predicted MAPPING** (geo_rd 0.773 = well-calibrated render-error detector — but did NOT translate to tracking).
- **motion-corr predicted TRACKING** (dino_reg_rd 0.249 > geo_rd 0.122 → better ATE — the prediction held).
- speckle: dino_reg_rd 0.018 is **27× smoother** than geo_rd 0.496 → register feature-smoothing reached σ².
- **THESIS-LEVEL PROOF:** the depth teacher made σ² an honest reconstruction-ERROR predictor (calibration 0.773; contrast pulled OFF edges, 0.100 vs the L2 ~0.45 signature) **BUT error ≠ deformation** — both backends stay deformation-blind at the INPUT (motion-corr 0.12 / 0.25, both low). σ² on moving tissue stays LOW because deformation is appearance-preserving.

**Decision:** Arm-1 ALONE cannot make σ² deformation-aware (the depth teacher fixes the teacher-half; the input-half needs Arm-2 motion → COMBINE). The judge is a working predictive proxy → adopt it as the standing n=1 cross-check.

### STAGE 4 — geo_feat fusion FALSIFIED [06-20]
**Tested:** the fusion idea — volume-render the internal geo_feat and concat (detached) into the dino_reg head: `[DINO_reg ; geo_feat]` (`uncertainty.fuse:geo`). Hypothesis: geo's path-ratio smoothness AND dino_reg's spike-cut in ONE σ².

| metric | geofuse | parent geo | parent dino_reg_rd |
|---|---|---|---|
| ATE mean/max (mm) | 2.14 / 6.89 | 2.29 / 8.74 | **2.03 / 5.81** |
| path-ratio | 6.60 | **4.20** | 6.41 |
| \|Pearson\| | 0.970 | 0.975 | 0.976 |
| frac_oracle | 0.393 | — | **0.476** |
| motion-corr | 0.171 | — | **0.249** |
| speckle | 0.057 | — | 0.018 |
| PSNR/SSIM | 22.55 / 0.711 | 22.61/0.723 | 22.58/0.706 |

(Stats over 27.6 M px, not seed noise. Sibling `georgbd` = `[DINO;geo_feat;rgb;depth]` reached only 261 frames, interrupted → no valid sim3.)

**Verdict — FALSIFIED, the muddy-average the spec predicted.** geofuse beat NEITHER parent's strength: no path-ratio smoothness (6.60 vs geo 4.20), no spike-cut (max 6.89 vs dino_reg_rd 5.81), AND σ²-quality DROPPED on both halves (frac_oracle 0.393 < 0.476, motion-corr 0.171 < 0.249).
**Decision/LESSON:** **σ² behaviour is bound to the HEAD ARCHITECTURE/pathway, NOT its inputs** — concatenating detached geo_feat just hands the σ² MLP a wider, noisier, uninformative-for-uncertainty input → fits worse. **KILLS** georgbd (more of the same) + the v2 DINO-query cross-attn-of-geo_feat (a fancier feed of the same signal). **Survivors = geo (path-ratio 4.20) + dino_reg_rd (max 5.81) as SEPARATE cells.** Arm-1-ALONE is now exhausted; the real "smooth AND bounded" σ² is the COMBINE, not feature-glue.

### Reference (NOT on the live path)
- **STAGE 5 — WildGS v2 (rectified):** dino_wgs Sim3 ATE 2.42 mm > base_wgs; base box-faithful check reproduced 3.49/15.00 mm (≈ A100 canon 3.15). RECTIFIED → kept ONLY as the reference for the one final raw-vs-rectified test.

---

## 2. WHAT WE KNOW NOW (the settled findings)

1. **The hedge works on tracking.** Canon rectified n=3: +uncert(geo) cut Sim3 ATE_mean 3.15→2.43 (−23%), ATE_max 14.79→9.43 (−36%), Pearson 0.81→0.97 — all non-overlapping. Inc-2 cuts the deformation-moment pose spikes as designed (the ATE_max drop is the signature). Render-neutral (≤0.3 dB cost).

2. **σ² is a LEARNED PREDICTOR of the model's own reconstruction error.** The NLL optimum is σ ∝ ‖C−Ĉ‖, so raw-residual σ² is a contrast/edge/specular readout. This is theory-grounded (NeRF-on-the-go's σ∝‖error‖ degeneracy), not a data quirk — a predicted, citable artifact of the raw-residual teacher.

3. **σ² is STRUCTURALLY DEFORMATION-BLIND.** Deformation is appearance-preserving (smooth tissue moves, colour ~unchanged → low photometric residual → LOW σ² on moving tissue = "σ² in the wrong place"). Both geo AND dino backends inherit this — motion-corr stays ~0.12–0.25 regardless of feature. The depth teacher fixes the teacher-half (depth residual IS high on deformation) but **error ≠ deformation**, so it cannot reach full deformation-awareness alone. This is the "two halves" finding, now MEASURED: teacher-side = depth; input-side = motion (the combine).

4. **The TEACHER is the dominant lever, and it is BACKEND-DEPENDENT.** RGB-L2 → RGB+depth-consistency is ~neutral on geo (2.29→2.41) but UNLOCKS dino_reg (2.32→2.03, max −33%). dino_reg_rd is the best raw-left tracker.

5. **The two JOBS want DIFFERENT σ², and the judge separates them.** Mapping/calibration → geo (frac_oracle 0.773); tracking → dino_reg+depth (motion-corr 0.249, speckle 0.018). frac_oracle predicts mapping, motion-corr predicts tracking — VALIDATED as a predictive proxy.

6. **"register blotch-cleaning is cosmetic" was OVERTURNED, then bounded.** Registers are metric-cosmetic with the L2 teacher (dino_reg ≈ dino) but matter WITH the depth teacher (smoother σ² 0.018, better tracking) — yet still don't fix blindness. Use-dependent: HURTS Arm-1 calibration (frac_oracle 0.476 < geo 0.773), HELPS tracking + (later) Arm-2 correspondence.

7. **FUSION DOES NOT COMPOSE — σ² behaviour is bound to the head architecture, not its inputs.** geofuse was strictly worse than both parents on every axis. This KILLS georgbd + v2 cross-attn and EXHAUSTS Arm-1-alone.

8. **The ceiling is quantified.** A ~0.77-calibrated render-error detector that is deformation-blind (motion-corr ~0.12). Remaining Arm-1-alone fixes can only nudge CALIBRATION, never deformation-awareness. The ONLY lever for that is the COMBINE.

**Caveats (honesty):** live CRCD numbers are raw-left n=1. The rectified canon (geo 2.43, n=3) and WildGS (dino_wgs 2.42) are SEPARATE references for the one final raw-vs-rectified test — never cross-compared with raw-left. No L2-teacher σ²-quality baseline exists (the judge reads are absolute, not deltas). analysis-B (segment-stratified σ²) broke on CRCD masks (only `bg` printed — seg-label mapping fix pending; see Trap 6).

---

## 3. PROCESS — methodology to KEEP + traps that bit us

### 3A. KEEP (the methodology that worked)
- **Build the DIRECT judge, not argue through a noisy proxy.** `sigma2_quality.py` scores σ² on the two things it's USED for (mapping AUSE/frac_oracle; tracking σ²-motion Pearson) — exactly as Arm-2's `field_warped_pin_epe.py` scores the field directly. This is the single highest-leverage move: it made n=1 defensible AND predicted results (motion-corr → tracking, frac_oracle → mapping).
- **Self-testing tools.** `sigma2_quality.py --selftest` validates the AUSE/Pearson math on synthetic data (perfect<good<random≤anti) before trusting it on real data; the parity gate asserts an RNG invariant. Tools that verify their own math before being trusted.
- **Metric-first, the sub-SNR triple is the arbiter.** Sim3 ATE + path-ratio + |Pearson|dom from `sim3_ate.py`, never the rigid `output.txt`. Field-liveness probes are demoted to "what's it doing" — they explain, they don't ship.
- **Parity-gated, default-off → base bit-identical.** `test_inc0_bitidentical.py` checks the torch+cuda RNG state immediately AFTER `JointEncoding` construction (any leaked module draws init RNG and diverges the hash) + param count + state-dict keys. It deliberately does NOT byte-compare est trajectories (tinycudann atomic-add kernels are non-bit-reproducible on GPU). Runs before every A/B.
- **Two diagnostic sets, by default.** Every run ships NUMERICAL (the arbiter) + VISUAL (6-panel video + σ²/depth panels), both built inside `run_one`, auto-shipped to Drive co-located with each payload. The σ² judges run on the EPHEMERAL `uncert/` dir immediately, before the runtime can die.
- **The isolated loop.** ONE change → run FROM the base, in isolation (own clone, self-locating `REPO`) → did OUR metric improve? → keep. Two arms kept SEPARATE until COMBINE (a prior `time_normalize` "fix" introduced a bug — that burn is why arms don't stack).
- **Negatives recorded as cleanly as positives.** geofuse FALSIFIED, FIX time A/B/C clean NEGATIVE, "dino_reg cosmetic" overturned-then-bounded. Theory-grounding empirical findings (σ∝‖error‖) turns quirks into citable artifacts.

### 3B. TRAPS THAT BIT US (the rules learned)
- **TRAP 1 — Rigid-vs-Sim3 ATE INVERSION.** `output.txt` (rigid Horn, no scale) said "+uncert +27% WORSE"; truth was −23% BETTER. On up-to-scale MoGe depth (est ~7–8× GT scale) the rigid ATE is dominated by scale, not tracking, and inverts the A/B. **Rule: never headline `output.txt`; use `sim3_ate.py` + the triple.**
- **TRAP 1b — Sim3 ATE alone misleads on sub-SNR.** All CRCD is sub-SNR (GT motion below the tracker floor); Sim3 places the cluster centroid on GT → residual = cluster spread, not tracking. A "sit still" tracker scores better. **Rule: always quote path-ratio + Pearson; eyeball the trajectory before claiming a win.**
- **TRAP 2 — Raw-vs-rectified flip-flop.** CRCD configs are pipeline-AGNOSTIC (`datadir` holds whatever the last runbook staged) → "which pipeline" is invisible from the config. **Rule: the RUNBOOK decides; rectified runbooks guarded behind `ALLOW_RECTIFIED=1`; NEVER cross-compare pipelines (the −23% canon is rectified, the live cells are raw-left).**
- **TRAP 3 — Parallel-OOM + CPU oversubscription.** Two arms on one Colab box/tree; PARALLEL too high; threads uncapped. **Rule: ISOLATE FIRST (own clone/data); cap `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS ≈ nproc/PARALLEL`; `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb`; co-scheduled runbooks keep PARALLEL conservative; throttle with `wait -n`.**
- **TRAP 4 — Ephemeral render/σ² loss.** An A/B once shipped only ckpt+est → no frames for a post-hoc video; σ² maps die with the runtime. **Rule: ship EVERYTHING for assessment to Drive inside `run_one` before the runtime can die; commit small eval GT to the repo (`data/` is gitignored).**
- **TRAP 5 — The MoGe up-to-scale ~7× (NOT a hardcoded ×10).** MoGe-2 is genuinely up-to-scale per-snippet; the only place a real ~×10 could bake is the `generate_depth_moge.py --ref` median scale-match if the reference corpus median is off. **Rule: Sim3 absorbs the factor at EVAL, but neural SLAM is NOT scale-invariant — metric knobs (trunc/range_d/bound/near-far/lr) shape the loss, so scale must be right at INPUT, not just corrected at eval (×1.0 vs ×0.30 = 32% ATE delta Sim3 can't synthesise). Unique-value-check uint16 depth PNGs after gen.**
- **TRAP 6 — Seg analysis-B bug (CRCD masks → only `bg`).** Segment-stratified σ² printed only `bg` — the CRCD seg legend didn't match the `--tool_labels 3 --bg_labels 0` invoked, and the CRCD loader historically collapses masks to a Canny edge channel. **Rule: fix the seg-label mapping before leaning on per-segment σ²; analyse on the dataset where the phenomenon was OBSERVED (the register blotches were on CRCD, not SemSup which has no tool legend).** Until fixed, analysis-A (σ² vs |∇I| vs motion, no seg needed) carries the finding.
- **Secondary radar:** render-eval OOM → silent EMPTY metrics (fix: `CUDA_VISIBLE_DEVICES="" python -u`); staging skips on EXISTENCE not COMPLETENESS (verify depth-count == frame-count); `dynamic:False` is BROKEN for field-off (use `deformation_off:True`); render PSNR ≈ paper only IN-TRAINING (post-hoc drops ~7 dB; `render_all_frames.py` deleted, inline only); eval the 360-row CRCD-Published GT, never the stale 271-row local copy.

---

# PART 2 — FORWARD PLAN (prioritized, gated)

The fork in the road: **Arm-1-ALONE is exhausted** (finding #7/#8). The real leap (deformation-aware σ²) is GATED on Arm-2 revival. Everything below is split by that gate.

## 4. GATED ON ARM-2 REVIVAL — the real leap

**The blocking dependency:** Arm-2's `deform_field_teacher_only` fix (commit `b0ef28e`, code-verified `ddsslam.py:766-769`) drove frame-1 def_sup 6.4e-4 → 2.7e-6 (237× in 100 iters) ⇒ the field LEARNS Δx*. But the full `teacher_on_v1` run + `field_warped_pin_epe` judge has NOT landed (GPU-blocked: OOM + Arm-4 squatting a shared T4 + a Drive FUSE drop). **The entire combine hangs off this one verdict.** RESUME on any free ~14 GB GPU: `bash Addons/colab/run_cell.sh configs/Super/trail3_teacher_on.yaml teacher_on_v1` → judge wants |Δx|>0, reduction >> shuffled, cos>0, render≈28.

### 4.1 THE COMBINE — compute-once-feed-both (the cleanest part of the program)
Both arms independently converged on ONE signal: **the rigid-vs-deformable residual contrast `Δ = Ē_rigid − Ē_deform`**, rendered via the existing `deformation_off` pass (gauge-correct, scale-robust, image-space → immune to the chronic MoGe up-to-scale Sim3 trap).
- **Arm-1 consumes it as the σ² TEACHER** → makes σ² deformation-AWARE (the missing input-side fix; the one lever finding #3/#8 names).
- **Arm-2 consumes it as the per-Δx* TRUST/validity gate** (replaces the gauge-confounded `|Δx*|` magnitude as the trust weight; `|Δx*|` STAYS the field's vector target).

⇒ **COMBINE = compute `E_R − E_D` ONCE, feed both heads.** Not bolt-two-together — one signal, two consumers. Full design: `Addons/docs/COMBINE_TRIGAUGE_DESIGN_20260619.md`.

**The routing is HALF-WIRED already — combine builds NO new plumbing.** DDS-SLAM already runs tracking + mapping threads, and Inc-2 already down-weights high-σ² pixels in tracking (`clamp(1.0/sigma2, _wmin, _wmax).detach()`, scene_rep.py:560). Once σ² goes high on moving tissue, the EXISTING Inc-2 weighting routes for free: certain+still → tracking anchor; uncertain+moving → mapping/field (kept OUT of pose so it can't corrupt the camera). Soft weight, not a hard one-thread send = textbook dynamic-SLAM. **The combine just makes σ² CORRECT.**

### 4.2 THE S0 DRY-RUN — the FIRST gate (build the gate or report a negative)
**`deformation_off` render on a teacher_on ckpt → per-pixel `Ē_rigid − Ē_deform` → overlay on seg + pins (Analysis E).**
- Concentrates on deforming TISSUE (not tools/specular) AND non-zero ⇒ **build the contrast gate.**
- Sub-noise on our sub-SNR motion ⇒ report the NEGATIVE / fall to STIR; do NOT gate on a dead signal.
- **🚨 GATED on a LIVE field:** a dead field gives `E_R − E_D = 0` → S0 correctly blocks. So the whole combine is transitively gated on §4's revival verdict.
- **Prerequisite before the novelty writeup:** read **NRGS-SLAM (arXiv:2602.17182) IN FULL** — closest prior art (dual-hypothesis rigid-vs-deformable BCE + tracking routing ≈ this contrast gate). Only the abstract has been read. (This read is buildable NOW — see §5.)

### 4.3 TRIGAUGE — the destination (after S0 passes)
3-way soft routing (seg-prior × gauge): static-rigid bg → camera ego-motion (anchors gauge); deformable tissue → field D(x,t); moving-rigid tool → its own per-frame SE(3) (KEEP+model, not mask). σ² splits rigid/moving; DINO/seg splits tool/tissue.
**Learned-not-hard-rule (critical for the writeup):** the 'what-kind' axis is the **continuous DINO feature map** (learned, foundation, dataset-agnostic), NOT discrete `{bg,tissue,tool}` labels. Router = learned function of TWO scales: individual pixel motion (the §4.1 contrast) × feature-group regime (DINO-neighbours via cross-attention, DINO-as-QUERY). **DINO = what-kind (have it now); motion = how-much (gated on Arm-2).** GT seg = an AUXILIARY tested prior (ablation: seg-supervised vs pure-DINO vs shuffled-label), NOT the backbone, NOT an inference rule.

## 5. BUILDABLE NOW (while waiting on the GPU), ranked

| # | item | value | cost | parity | metric gate |
|---|---|---|---|---|---|
| 1 | **#3 feature-group σ²-consistency — reg version** | the spatial-STABILITY axis + the TriGauge "things-that-look-like-me" PRIMITIVE; Arm-1-side, NOT gated on Arm-2 | LOW (a regulariser term, spec ready `ARM1_3_FEATURE_SIGMA_CONSISTENCY_SPEC_20260619.md`, commit `0272896`) | flag-gated default-off, parity-safe | speckle ↓ + frac_oracle non-worse + ATE non-worse |
| 2 | **Read NRGS-SLAM in full** | prerequisite for the combine/novelty writeup (closest prior art) | LOW (reading) | n/a | n/a |
| 3 | **geo_feat viz** (why fusion failed) | turns the geofuse negative into a clean figure (σ² muddy-average) | LOW | read-only | n/a (explanatory) |
| 4 | **Seg-B bug fix** (CRCD label mapping) | unlocks segment-stratified σ² → the direct "σ² on tools not tissue" confirmation | LOW–MED | tooling-only | per-segment σ² prints tool/tissue/bg |
| 5 | **Final raw-vs-rectified A/B** | closes the one deferred pipeline question (geo raw-left vs the rectified canon 2.43) | MED (one GPU run-pair) | runbook `ALLOW_RECTIFIED=1` | the triple, matched config |

**On #3, the key design choice (reg vs architectural):**
- **reg version** = a feature-similarity regulariser (NeRF-on-the-go: similar DINO features → similar σ²). Cheap, parity-safe, ships the feature-group PRIMITIVE the router needs. **Recommended first.**
- **architectural version** = an aggregating head (cross-attention, DINO-as-query) that POOLS σ² over feature-neighbours. Heavier; it is the TriGauge router's actual machinery. ⚠️ finding #7 warns: σ² behaviour is bound to the HEAD ARCHITECTURE — an aggregating head is a genuine architecture change (unlike geo_feat concat which only fed the same head), so it is the *right* lever IF #3-reg shows the feature-group signal is real. **Gate the architectural build behind a positive #3-reg read.**

**Note:** #3 reaches only the SPATIAL/stability axis. The TEMPORAL/deformation axis (motion → σ²) is the COMBINE and stays gated on Arm-2.

## 6. DECISION POINTS FOR THE USER

**D1 — Is Arm-1-ALONE done? (just wait for Arm-2, or build #3 now)**
Recommendation: **Arm-1-alone is metric-EXHAUSTED for tracking** (teacher = the win; fusion = falsified; #7/#8 quantify the ceiling). But **build #3-reg now** — it is NOT an Arm-1-alone tracking fix, it is the **TriGauge router primitive** (the "things-that-look-like-me" feature grouping) that the combine needs, and it's buildable while the GPU is blocked. So: stop chasing Arm-1-alone tracking gains; DO build the #3-reg scaffold + read NRGS in parallel. *User: confirm we invest the wait in #3-reg scaffolding rather than idling.*

**D2 — #3: reg vs architectural?**
Recommendation: **reg FIRST** (cheap, parity-safe, proves the feature-group signal), then promote to the architectural aggregating-head ONLY on a positive reg read (finding #7 says the head architecture is the real lever — spend it deliberately, not speculatively). *User: approve the reg-first gate-then-architectural sequence.*

**D3 — Close-out items?**
The seg-B fix (#4) and the final raw-vs-rectified A/B (#5) are loose ends. *User: decide whether the seg-B fix is worth doing now (it gives the direct per-segment "σ² on tools not tissue" confirmation that strengthens the writeup) and whether the final raw-vs-rectified test runs now or stays deferred to the very end as planned.*

---

## CRITICAL-PATH SUMMARY

| item | status |
|---|---|
| Arm-1 tracking hedge (geo, dino_reg_rd) | **DONE — WIN.** geo (path-ratio 4.20) + dino_reg_rd (ATE 2.03 / max 5.81) survive as separate cells |
| Arm-1-alone σ² fixes (teacher swap, geo_feat fusion) | **DONE — EXHAUSTED.** dino_reg_rd won; geofuse falsified; georgbd + cross-attn-of-geo_feat KILLED |
| σ²-quality judge (the n=1 cross-check) | **BUILT + VALIDATED predictively** |
| #3 feature-group σ²-consistency (reg) — router primitive | **BUILDABLE NOW** (parity-safe; spec `0272896`) |
| Read NRGS-SLAM in full | **BUILDABLE NOW** (combine/novelty prerequisite) |
| seg-B fix / geo_feat viz / final raw-vs-rectified | **BUILDABLE NOW** (close-out) |
| Arm-2 `deform_field_teacher_only` revival (237× drop validated) | **BUILT — full run + judge PENDING on a free ~14 GB GPU** |
| S0 dry-run (Analysis E: does `E_R−E_D` concentrate on tissue + non-zero) | **GATED on Arm-2 revival** (dead field → contrast 0 → correctly blocks) |
| COMBINE (`E_R−E_D` → σ² teacher AND Δx* trust gate) | **GATED on S0 passing** ⇒ transitively gated on Arm-2 revival |
| Motion→σ² input (deformation axis) + TriGauge how-much half | **GATED on Arm-2 revival** |

**The single critical-path dependency:** the deformation-aware half of σ² (and of TriGauge) hangs off **Arm-2's revival verdict** — itself blocked only on a free ~14 GB GPU, not on code. Everything Arm-1-side (#3-reg primitive, the what-kind DINO half, NRGS read, close-outs) is buildable in parallel right now.
