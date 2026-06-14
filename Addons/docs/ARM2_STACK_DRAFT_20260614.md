Both load-bearing facts confirmed from the live code:

1. **`compute_loss` `weights=` does `loss * weights` elementwise** (utils.py:127/130) — multiply-args-trick would give `w²` weighting. Vet R1 is correct.
2. **`def_reg = (vox_motion ** 2).mean()` is computed on the POST-product `vox_motion`** (scene_rep.py:206), after both `oracle_w` (`* ow`, :200) and `surf_w` (`* surf_w`, :204) have multiplied it. So `def_reg`/NRGS already read the doubly-gated field — vet C1/C4 are confirmed and the leash acts on the shrunk product exactly as the vet warned.

Here is the final integrated draft.

---

# DDS-SLAM Arm-2 Stack — FINAL Integrated Draft (WildGS + SNI + NRGS)

**Status: DRAFT for iteration, vet folded in.** Code-grounded against `scene_rep.py`, `decoder.py`, `utils.py`, `ddsslam.py`, `dataset.py` (working copy; `DDS-SLAM-Base/model/` is empty, so "== Base" means the working copy's default-off paths). [W]=WildGS, [N]=SNI, [G]=NRGS.

**Two hard correctness facts the whole stack respects (both re-verified against live code):**
- The `n_importance>0` branch in `render_rays` (scene_rep.py:374-387) is **broken** (undefined `edge_map` :377, wrong `raw2outputs` arity :387, `pts` missing time column :384). All configs run `n_importance:0`. Every increment threads new tensors through the **primary** `run_network`/`raw2outputs` (scene_rep.py:371-372). **Do not depend on, and do not fix, the importance branch** — keep `n_importance:0` across the sweep.
- `def_reg = (vox_motion**2).mean()` (scene_rep.py:206) is computed on the **post-gating** `vox_motion`, after `* oracle_w` (:200) and `* surf_w` (:204). The leash already acts on the doubly-gated product. This drives C1/C4 below.

---

## 0. WHAT CHANGED FROM THE VET (read this first)

The vet found one fatal measurement gap, one wiring bug, and ~2× scope. All folded in:

1. **NEW Increment 0.5 — bootstrap-viability probe, zero new code, on existing checkpoint.** Gates the entire NRGS arm. If `|E_rigid−E_deform|` is sub-noise on SemSup (the best-SNR case), **NRGS is CUT** and replaced by a correspondence teacher (STIR/flow) that exists at Δx=0. (vet top-fix #1)
2. **w² weighting bug fixed.** Pose pathway MUST call `compute_loss(pred, target, weights=(1/u).detach())` (utils.py:127 does `loss*weights`), NOT the rgb-multiply-trick at scene_rep.py:448 which squares the weight → `1/σ⁴`. The `weights=` kwarg is currently dead in all call sites; using it is genuinely NEW wiring, regression-safe (defaults to None). (vet top-fix #2)
3. **`def_reg` masked by (1−w*)** so the leash only pulls rays the teacher calls rigid — stops def_reg and NRGS fighting on the same rays. (vet top-fix #3)
4. **`oracle_w × surf_w` product made explicit and intended.** surf_w = spatial support (hard mask); attribution gate = routing within support. `effective_gate = surf_w · max(a_ray, λ·seg_prior)`. Log `mean(effective_gate)`. (vet top-fix #4)
5. **Build order INVERTED: WildGS-on-`geo` first, SNI deferred to last/optional, NRGS strictly gated on 0.5.** Cheapest signal-bearing increment first. (vet CUT recommendation)
6. **Test plan corrected:** SemSup `load_poses()` returns identity → **pose pathway validates on StereoMIS real GT / Arm-1 STIR ONLY, never SemSup.** SemSup-first applies to render/field metrics only. (vet T3)
7. **NRGS pass condition calibrated against the shuffled-label null (>2σ), not hardcoded +0.3 dB**, plus a `corr(σ², dE)` decorrelation check so WildGS-σ² and NRGS-dE aren't counted as synergy when they're one signal. (vet T1/T2)
8. **SNI renamed** "SNI-inspired one-way conditioning" (FiLM ≠ SNI cross-attention); the `film` vs `concat` ablation validates one-way conditioning generically, not "the SNI operator." (vet P1)

---

## 0b. MAP-vs-FIELD COMPETITION — σ² gets a THIRD use (Battery-7 architecture read, 2026-06-14)

Code-verified (scene_rep.py:177-214, decoder.py:423-442, ddsslam.py:677-689): the map
(hash-grid + SDF + color) has **NO time input** — time enters ONLY via the TimeNet warp
(`vox_motion`, :191; `pts+vox_motion`, :213). So the map **cannot store per-frame motion**; it
fits a blurry **time-average** and **wins the gradient race** (hash/decoder lr 0.01 + huge
capacity vs TimeNet lr 0.001 @ `lr_mult 0.1`). It absorbs the *gradient*, not the motion — claims
the explanation budget first as blur and starves the field. (Battery-7: field surface-dead across
3 seeds even with pose frozen → not the pose race; this is the cause.)

**Consequence:** routing the field's capacity (ROUTE (b), `oracle_w` on `vox_motion`) is necessary
but **NOT sufficient** — the faster, higher-capacity map keeps claiming moving pixels. So σ² needs
a **THIRD use**, alongside pose-↓ and route:

> **(c) THROTTLE THE MAP** — down-weight the *map's* photometric gradient at high-σ² (moving)
> regions so the field is the only one left to explain them. WildGS-inverted applied to the
> map-vs-field competition, not just pose.

Mechanism options (decide after the probe): (i) weight the mapping rgb/depth loss for the
**map params only** by `(1−a_ray).detach()` at high σ² (needs the map-grad path separated from the
field-grad path — e.g. a stop-grad split, since one `rgb_loss` currently feeds both); or (ii) a
**canonical/stationarity prior** (penalise the map for deviating from a reference frame, forcing
deviations onto the field). **GATE:** only build this once the map-absorption probe confirms a real
residual at the moving tissue (else there's nothing to reassign). See diagnosis/infra/map_absorption_probe.py.

## 1. THE INTEGRATED PICTURE (one diagram in words)

```
                 warped point p+Δx  (run_network, scene_rep.py:205)
                        │
        sdf_net ──► geo_feat[15] ──┐  [N] one-way conditioning (default-off, build LAST):
        color/edge head outputs ───┤  f_fuse = Linear(σ(Lg[f_g,f_s])·f_a + Lb[f_g,f_s])
                                    │  (tcnn: tap head OUTPUTS, no penult activation — decoder.py:442)
                                    ▼
                    [W] UncertaintyNet (2-layer, bias-free ReLU)   ◄── reads geo_feat (default) OR f_fuse
                          softplus(fp32)+eps -> per-sample σ²
                          volume-render w/ SAME sdf weights (raw2outputs)
                          u_ray = σ²_ray ;  a_ray = σ²/(σ²+c) ∈ (0,1)
                 ┌──────────────────┘                         └──────────────┐
                 ▼ POSE (a) — tracking only                                  ▼ ROUTE (b) — field
   compute_loss(pred,target, weights=(clip(1/u,w_min,w_max)).detach())   effective_gate = surf_w·max(a_ray, λ·seg)
   ** weights= path, NOT the *w multiply-trick (that squares -> 1/σ⁴) **  fed as oracle_w (scene_rep.py:198)
                 │                                                            │
                 │                              [G] NRGS dual render (mapping/BA only, gated on Inc-0.5):
                 │                              E_rigid (force_rigid kwarg) vs E_deform ; SHARE z_vals/pts
                 │                              w* = σ(α(2·seg−1) + β(E_rigid−E_deform))   (detached TARGET)
                 │                                  ├─ L_gate_bce : BCE(gate, w*)
                 │                                  ├─ L_field    : w*·E_deform     (the missing teacher)
                 │                                  └─ def_reg    : (1−w*)·mean(Δx²)  ◄── MASKED leash
                 ▼                                                            ▼
            better pose (down-weight deforming/specular)        Δx routed + supervised where rigid SDF fails
```

**One sentence:** one view-robust feature (SNI, optional) → one uncertainty/attribution scalar (WildGS) → simultaneously **down-weights pose** (weights= path, [W]-a) and **routes deformation** (effective_gate, [W]-b) → NRGS's dual posterior `w*` is the positive teacher that supervises both AND masks the def_reg leash (1−w*), **iff Inc-0.5 proves `E_rigid≠E_deform` exists**. Seg edge-prior (`compute_edge_semantic` exp(−d/10), dataset.py:36-53) is the common prior: WildGS floor, NRGS `α(2·seg−1)`, SNI feat-loss buckets.

Novelty = the **composition on an SDF/Co-SLAM substrate + seg-supervision + inverted use** (WildGS uncertainty drives pose AND routing; deformation_off renders E_rigid free), NOT the published machinery [W][N][G].

---

## 2. BUILD ORDER (revised, ground-up, viability-gated)

**Inc-0 — Plumbing & regression harness (no method).** `raw2outputs`/`run_network`/`render_rays` always carry optional `(beta, fused_feat)` returns behind enable flags; one return-signature, no churn between [W]/[N]. All 4 `forward` call sites (tracking :575, current-frame :344, first-frame :261, BA :460) pass new args **keyword-only with defaults**. Master flags gate `__init__` so disabled modules are **never constructed** (no RNG consumed — required for bit-identity, since tracking uses `select_samples`/`perturb` RNG). **Gate: est_c2w + PSNR bit-identical to Base at fixed `seed` (seed_everything :43).**

**Inc-0.5 — Bootstrap-viability probe (NEW; zero new code; existing ckpt).** Load a trained mogev2 model; render deform-on vs deform-off on held-out tissue via the **existing `render_eval_attrib.py`**; measure the `|E_rigid−E_deform|` distribution vs the eval noise floor on **SemSup (best SNR)**. **This is the build-decision gate for NRGS:** if the delta is sub-noise, NRGS cannot bootstrap → **CUT NRGS, substitute a correspondence teacher** (supervise Δx against STIR/optical-flow residual after rigid warp — a teacher that exists at Δx=0; Arm-1 has STIR EPE infra). Do this BEFORE writing any NRGS code.

**Inc-1 — WildGS uncertainty head on `geo_feat` (feat_source=geo), ENABLE-ONLY.** Standalone, no SNI dependency, cheapest signal-bearing. Build `UncertaintyNet` + aleatoric NLL (mirror the `def_reg` block, ddsslam.py:214). σ² in **fp32 + eps** (fp16 softplus hazard). **Test before consuming:** σ² correlates with rigid-render RGB error on held-out; σ histogram non-degenerate; agrees with seg on tool boundaries but **diverges on photometric-but-non-edge failures** (= the new signal). (Note the calibration circularity in §5-T2.)

**Inc-2 — WildGS pose pathway (a).** Thread `use_uncertainty_pose_weight` through `forward`, **True only from `tracking_render`** (:575); absent at the 3 mapping calls → mapping untouched. Use `compute_loss(pred,target, weights=clip(1/u, w_min, w_max).detach())`. **detach() mandatory** (no tracker gaming). **Test on StereoMIS real GT / STIR ONLY** (SemSup pose GT is fictional): path-ratio `est/GT`↓, small-axis Pearson↑, Sim3/SE3 ATE ≤ base.

**Inc-3 — WildGS routing pathway (b).** `effective_gate = surf_w · max(a_ray, λ·seg_prior)` fed as `oracle_w` (:198). Log `mean(effective_gate)`. Test on field-attribution ON/OFF, SemSup-first. Generalises existing `oracle_routing` (§4).

**Inc-4 — NRGS dual-hypothesis (LAST; only if Inc-0.5 passed).** `force_rigid=True` **kwarg** to `run_network`+`render_rays` (per-call, NOT mutating `self.config['deformation_off']`). Second render in `forward` **only when `nrgs.enable and self.training and not render_only`, and only on mapping/BA batches** (never per-iter tracking). **Share sampled `z_vals`/`pts`** between the two renders (only Δx=0 vs Δx differs). `def_reg` masked by (1−w*). v1 `learned_gate:false` reuses `edge_semantic_map` → zero new params, checkpoint-compatible (avoids strict=False trap). Pre-test: 1-iter assert `E_rigid≠E_deform` when `surface_bind>0` + all-weights-0 loss bit-identical to Base.

**Inc-5 — SNI one-way conditioning (LAST/OPTIONAL).** Build ONLY if the WildGS geo-vs-sni A/B shows geo features are the bottleneck. `SNIFuse` + `return_feat` taps. Under `tcnn_network:true` the FullyFusedMLP exposes no penultimate activation (decoder.py:442 returns `[rgb,sdf], edge_semantic`) → tap head **outputs** (rgb pre-sigmoid ⊕ edge logit) ⊕ geo_feat; detect `decoder.tcnn_network` at construction and warn. Test: `fused_feat` shape `[N,16]` finite var>0; `film` vs `concat` (validates one-way conditioning generically).

**Chain:** Inc-0 → **Inc-0.5 (gate)** → WildGS head(geo) → pose(a, StereoMIS) → route(b, SemSup) → [NRGS iff 0.5 passed] → [SNI iff geo is bottleneck].

---

## 3. CONFIG-GATING + ABLATION MATRIX

All flags via `config.get(...)`; every master default-off ⇒ module never constructed.

```yaml
# ---- [W] WildGS (block; sub-flags read only when enable:true) ----
uncertainty:
  enable: false
  pose_weight: false           # (a) per-ray 1/σ² on TRACKING loss via weights= (NOT *trick)
  route_deform: false          # (b) gate Δx by learned a_ray
  nll_weight: 0.1
  seg_prior_lambda: 0.0        # 0=pure learned; >0=seg floor; =match-oracle for A/B
  eps: 1.0e-3
  attrib_c: 1.0                # a_ray = σ²/(σ²+c)
  feat_source: geo             # geo (default) | sni  (sni requires sni_fusion:true else falls back+warn)
  warmup_iters: 0
  w_min: 0.1
  w_max: 10.0                  # clamp 1/σ² (pose conditioning bound)

# ---- [G] NRGS (block; only if Inc-0.5 passed) ----
nrgs:
  enable: false
  beta: 1.0
  alpha_prior: 0.0
  gate_bce_weight: 0.0
  field_weight: 0.0
  mask_def_reg: true           # def_reg *= (1-w*)  — leash only where teacher says rigid
  bootstrap_iters: 0
  route_with_gate: false       # NRGS gate -> oracle_w (successor to oracle_routing)
  learned_gate: false          # false=reuse edge_semantic_map (0 new params)

# ---- [N] SNI (master; build LAST/optional) ----
sni_fusion: false
decoder: { sni_fuse_dim: 16, sni_fuse_mode: film }   # film | concat
training: { sni_feat_weight: 0.0, sni_feat_margin: 0.5 }  # >0 = MAPPING-only feat loss
```

| Sweep cell | uncertainty | (pose/route) | nrgs | sni | notes |
|---|---|---|---|---|---|
| **base** (==mogev2) | false | – | false | false | regression anchor |
| **+WildGS** | true | off/off | false | false | `feat_source=geo`; calibration only |
| **+W-pose** | true | pose=T | false | false | (a) — StereoMIS/STIR only |
| **+W-route** | true | route=T | false | false | (b) — SemSup |
| **+W-both** | true | both=T | false | false | a+b |
| **+NRGS** | false | – | true | false | iff 0.5 passed; `field_weight>0`, prior bootstrap |
| **+SNI** | false | – | false | true | optional; fusion only / +featloss |
| **+all** | true | both=T | true | true | `feat_source=sni`, `route_with_gate=T` |

**Load-time asserts (load-bearing):**
1. **At most one owner of `oracle_w`** — precedence: NRGS(`route_with_gate`) > WildGS(`route_deform`) > seg(`oracle_routing`). Assert-not-more-than-one OR enforce+log winner.
2. **`oracle_w × surf_w` is intended** (C1): surf_w = spatial support, attribution = routing within. Log `mean(effective_gate)` per increment.
3. If `nrgs.enable`: require `alpha_prior>0` OR `deform_surface_bind>0` (else `w*→0.5`, `dE→0`).
4. **`deform_hardbound==0 OR nrgs.field_weight==0`** (C2: tanh saturation vs positive teacher) — or budget hardbound ≥ expected tissue Δx.
5. If `feat_source=='sni'` but `sni_fusion==false`: warn + fall back to geo.

---

## 4. SHARED-HEAD vs SEPARATE — resolved

WildGS uncertainty **layers on / generalises** the existing `oracle_routing` seg-gate; seg is demoted from sole signal to **prior/floor**. Current gate: `oracle_w = target_edge_semantic` (fixed, no learning, scene_rep.py:434). WildGS: `oracle_w = max(a_ray, seg_prior_lambda·target_edge_semantic)` (then `× surf_w`). One shared attribution scalar, three sources by precedence (NRGS-learned > WildGS-learned > seg-fixed). Seg is the Bayesian prior everywhere. The sweep runs **base / oracle-only / learned-only / learned+seg-prior**; the **shuffled-seg-label control** proves any gain is seg-driven not raw capacity. When NRGS owns the gate, WildGS σ² is repurposed as NRGS's **prior** (`logit_prior := f(u, seg)`) — high σ² seeds a ray non-rigid, attacking the Δx≈0 bootstrap. Coordination point, not conflict.

---

## 5. TEST PLAN (per increment) + the vet's confound fixes

**Universal:** baseline = OUR mogev2 (`*.enable:false`, same data/seed). **Whole-frame PSNR is FIELD-BLIND — guard-rail only**; headline = `render_eval_attrib.py` (held-out, tissue-masked, ON−OFF). **SemSup trial_3 first for render/field** (alive seg head ~7-8% loss share); CRCD/StereoMIS second. Report battery-3: `Var_t(Δx)`, masked holdout-PSNR, Kabsch rigid-fraction, dx_hook revival. Every increment ships the **shuffled-label control**.

| Inc | Pass vs mogev2 | Controls / vet fixes |
|---|---|---|
| **0** | bit-identical est_c2w+PSNR @ seed (module not constructed) | RNG-stream parity (R3) |
| **0.5** | `|E_rigid−E_deform|` distribution vs noise floor on SemSup | **GATE: sub-noise ⇒ CUT NRGS, use correspondence teacher** |
| **1 head** | σ² ~ rigid RGB error (held-out); non-degenerate; diverges on non-edge photometric fails | **T2 circularity:** rigid error on tissue IS partly deformation; expect overlap, that's wanted for routing — but see decorrelation below |
| **2 pose** | path `est/GT`↓, small-axis Pearson↑, Sim3/SE3 ≤ base | **StereoMIS real GT / STIR ONLY — NOT SemSup (fictional GT).** **Use weights= not ×-trick (R1).** |
| **3 route** | `FIELD_ATTRIB(on−off)` > 0 and > oracle-seg-gate | shuffled seg; geo-vs-sni A/B; SemSup-first |
| **4 NRGS** | `FIELD_ATTRIB(on−off)` exceeds **shuffled-label null by >2σ** (NOT hardcoded +0.3 dB); `nrgs_w_mean` tracks seg then **diverges** (dE≠0); above-noise gate only | shuffled seg; field-only vs gate-only vs +route; **`corr(σ², E_rigid−E_deform)`** — if high, W & N redundant ⇒ drop NRGS (T2) |

**Non-negotiable confound:** a gate can look alive (Δx>0 in high-attribution regions) while **gauge-absorbing camera motion**. Attribution eval + dx_hook + Kabsch rigid-fraction must show Δx tracks **tissue deformation, not pose**. This is exactly why whole-frame PSNR is excluded as headline.

---

## 6. RUNTIME / A100 SWEEP

- **NRGS = cost driver, ~1.7× mapping wall-clock** (second force_rigid render). Mitigations baked in: share `z_vals`/`pts` (only network eval differs), no importance pass (broken anyway), rigid render **only on mapping/BA**, never per-iter tracking. Budget NRGS/+all cells at 1.7× base.
- **WildGS/SNI cheap:** SNIFuse <10k params; UncertaintyNet 2 layers. Extra accumulation negligible vs the 15-d geo_feat already accumulated.
- **fp16/tcnn:** softplus in **fp32 + eps**; verify `batchify` carries extra outputs without shape drift.
- **Parallelism:** 8 cells = independent processes. NRGS doubles the *render-graph activation* peak (not the model). On 80GB A100 (151 frames is small) pin NRGS/+all to dedicated slots; co-schedule base/+WildGS/+SNI.
- **Determinism:** same `seed` across base-vs-disabled cells for the Inc-0 / NRGS-all-zero bit-identical asserts.

---

## 7. OPEN DESIGN DECISIONS (user calls these)

1. **NRGS vs correspondence teacher** — depends entirely on Inc-0.5. If `|E_rigid−E_deform|` is sub-noise, do you want the STIR/flow-residual teacher as the replacement, or drop the positive-teacher idea and ship WildGS+SNI alone? (vet: STIR teacher exists at Δx=0, lower risk.)
2. **Seg-gate replace vs layer** — drafted as **layer** (seg=prior/floor, learned on top). Do you want a pure-replace cell too (learned-only, λ=0) as a cleaner ablation, accepting it loses the bootstrap floor?
3. **NRGS bootstrap mechanism** — `surface_bind` vs `alpha_prior` vs `warmup_iters` as the symmetry-breaker. Which do you trust to make `E_rigid≠E_deform` cold? (surf_w only restricts *where*, not *whether* — open question.)
4. **Lightweight-fusion form** — FiLM (drafted, cheap) vs a single-head one-way cross-attention (closer to SNI, more params, overfit risk on 151 frames). Worth the param cost?
5. **Pose-weight clamp `[w_min,w_max]`** — values? Too tight = no effect; too loose = pose conditioning blows up on near-zero σ².
6. **`corr(σ², dE)` redundancy threshold** — what ρ makes you drop NRGS in favour of WildGS-only?

---

## 8. HONEST RISKS + WHAT'S CUT

**Risks:**
1. **Field may stay dead even with NRGS** — collapse is STRUCTURAL (map absorbs temporal variation + weight_decay + frame-0 anchor), not just the pose race (hypconfirm_semsup falsified the race). If `E_rigid≈E_deform` everywhere (the inertness finding), NRGS degenerates to oracle_routing and adds nothing. **Inc-0.5 catches this before any NRGS code is written.**
2. **Pose pathway may not move ATE on CRCD** (sub-SNR tracker jitter dominant); validate on StereoMIS real GT / STIR. **And it CANNOT validate on SemSup** (fictional pose GT) — a real gap, now stated.
3. **WildGS-σ² and NRGS-dE may be the same signal** → "+all" synergy is one signal counted twice. The `corr(σ², dE)` check catches it; if high, drop NRGS (cheaper to keep WildGS routing).
4. **`DDS-SLAM-Base/model/` is empty** — "regression-safe == Base" is asserted against the working copy's default-off paths (verified clean branch-on-flag). True upstream diff needs the base model files restored.
5. **Broken `n_importance>0`** is a latent landmine — keep `n_importance:0` across the sweep; don't fix inside an increment (confound).

**Cut / deferred:**
- **NRGS** unless Inc-0.5 passes (cost driver, most likely to degenerate to oracle_routing).
- **SNI** to last/optional — adds nothing downstream until WildGS proves geo_feat insufficient; geo is the cheaper substrate.
- **The `n_importance>0` fix** — out of scope, stays broken on purpose.
- **Full SNI cross-attention** — replaced by FiLM (overfits 151 frames); renamed "SNI-inspired one-way conditioning" so the claim matches the mechanism.

**Files (absolute):** `c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\model\scene_rep.py` (run_network gating :198-206, def_reg-on-post-product :206, surf_w :203-204, broken importance :374-387, forward+rgb-multiply :448, BA timestamp :455-458), `...\model\decoder.py` (TimeNet bias-free :49, ColorSDFNet_v2 `[rgb,sdf],edge_semantic` :442 — penult NOT exposed), `...\model\utils.py` (compute_loss `weights=` does `loss*weights` :127/130 — dead in all call sites), `...\ddsslam.py` (seed_everything :43, def_reg loss wiring :182-221, tracking loss :575, BA :460), `...\datasets\dataset.py` (compute_edge_semantic exp(−d/10) :36-53).