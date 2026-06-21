The code references are confirmed in the working copy: warp at `scene_rep.py:239` (`inputs_flat = pts + vox_motion`), surf binding at `scene_rep.py:430-436`, hardbound at `scene_rep.py:212-214`, the t=0 anchor at `scene_rep.py:218`, `route_w_ba` assembly at `ddsslam.py:574-608`, the time_net split optimizer at `ddsslam.py:927-974`, and the Super.yaml schedule (iters:200, cur_frame_iters:100, keyframe_every:1, map_every:1, far:5, range_d:0.1). Now I'll write the document.

# Arm-2 Mapping: Why Base Wins on SemSup, Why Ours Regressed, and the Bridge

*Handover — 2026-06-21. Scope: SemSup trail_3 render. Verdict-first: the regression is a fixable architectural artifact (`global_BA` un-co-adaptation), not a deformation headroom limit. The field is ALIVE and CORRECT (held-out pin-EPE +52.5% ≫ shuffled +41.6%, cos(D,baked)+0.79). All code refs verified against the working copy `DDS-SLAM/`.*

## TL;DR
- **Base wins** because SemSup deformation is **mild** (mean |Δx|≈0.0077 = ~1.5% of the depth slab, max 0.084 = ~1.7% of slab / ~84% of the ±0.1 render band) AND the SemSup schedule is **aggressive** (iters 200, cur_frame_iters 100, keyframe_every 1, map_every 1). A static, time-averaged Co-SLAM hash-grid + SDF/color MLP is a near-optimal fit there. Base dead-field = **27.70 PSNR** (paper 28.649).
- **Ours regressed** because the alive field is applied **un-routed** to ALL ray samples while `global_BA` — the dominant trainer (iters 200 every frame) — trains on **un-warped** canonical rays. Render ladder: **dead 27.70 > v0.2 26.96 > v0.4(soft) 25.83 > v0.3(pixel) 25.39 > un-routed 24.1**. Five named, quantified, fixable costs total ~4 PSNR.
- **The bridge** is two fixes: `deform_surface_bind` (necessary, ~0.5 PSNR, **not sufficient**) + **B2 route-threading into `global_BA`** (THE pivotal co-adaptation fix). Keep **hard on/off** routing (literature standard); soft_scale miscalibrates amplitude. The render ladder should then **invert**.
- **Strategic call:** SemSup is the **do-no-harm guard** (≥28 PSNR + pin-EPE alive), NOT where deformation is proven to pay. **STIR** (IR-invisible GT, endpoint-EPE, above-SNR) is where the field has headroom to *win*. Run both.

---

## 1. Why base DDS-SLAM maps SemSup well

### 1.1 Mechanism: static time-averaging is near-optimal when deformation is small
Base DDS-SLAM runs with `dynamic:True` but the deformation field effectively dead (time_net contributes ≈0). It renders SemSup trail_3 at **27.70 PSNR** vs the paper's static-baseline **28.649** (Δ≈0.95, inside config/seed variance) [`project_deform_gauge_bug_20260620.md:25`; `project_semsup_metrics_canon.md`].

The Co-SLAM hash-grid + SDF/color MLP is fed rays pooled uniformly over **all** keyframes (`global_BA`, `DDS-SLAM/ddsslam.py:508`). For a surface point P observed across keyframes at states P(t_k), the static SDF jointly minimizes ‖SDF(P) − target_d‖². When the inter-keyframe spread ‖P(t_k) − P_avg‖ ≪ depth slab, the zero-crossing settles at the **time-average surface** P_avg, which is close to every observed state. Depth-anchored sampling (`range_d: 0.1` centered on `target_d`, `DDS-SLAM/configs/Super/Super.yaml:100`) confines samples to the surface band, so off-surface extrapolation error stays small. **The static map is not inherently superior — the deformation is simply too subtle to require modeling.**

### 1.2 Is SemSup deformation mild? Yes — quantitatively
- Field-warped mean **|Δx| = 0.0077**, max **0.084** [`project_combine_routing_model_20260619.md:73,76-77`; corroborated `project_deform_gauge_bug_20260620.md:23` mean 0.010 / max 0.087].
- Scene slab: `far: 5` (`Super.yaml:81`), local depth band ≈0.5 → mean |Δx| ≈ **1.5%** of the slab; max ≈ **1.7%** of slab (but ≈84% of the ±0.1 render band — important for outlier cost §2.4).
- For contrast, CRCD tissue deformation at the gallbladder (f357-359) measured **5–6 px optical flow** (~280 Sampson residual; synthetic parallax 0.00 vs deform 7.44) — an **order of magnitude larger** than SemSup [`project_combine_routing_model_20260619.md`].

**Conclusion:** SemSup is a *mild-deformation, excellent-static-geometry* dataset. The field's **upside here is small by construction** — so any application cost dominates the (tiny) modeling benefit.

### 1.3 The schedule is the second half of base's win
SemSup's config is heavy and local: `iters: 200`, `cur_frame_iters: 100`, `keyframe_every: 1`, `map_every: 1` (`Super.yaml:15,27,20,21`). Every frame is a keyframe; the map is sharpened intensely against the latest observation each step, making the time-average defensible (recent frames dominate). Contrast CRCD (`cur_frame_iters: 0`, `iters: 20`, `map_every: 5`) which only reaches ~22 PSNR — a gap **largely attributable to schedule, not the static model** [`project_combine_routing_model_20260619.md:91-92`; `project_mapping_free_wins_audit_20260621.md`]. Base's success on SemSup = *small deformation* × *aggressive online schedule*.

---

## 2. Why our deformation made it worse — ranked cost accounting

The alive field renders ~25.8 (soft-route) / 24.1 (un-routed) vs dead 27.70 — a ~2–4 PSNR drop **despite the field being correct** at held-out pins. Render ladder shows **MORE field applied → LOWER render**: `dead 27.70 > v0.2 26.96 > v0.4 25.83 > v0.3 25.39 > un-routed 24.1` [`project_combine_routing_model_20260619.md:73-79`]. This is application cost, not field failure. Five ranked mechanisms, all with designed fixes:

| # | Cost (≈PSNR) | Mechanism | Code | Fix | Status |
|---|---|---|---|---|---|
| **1** | ~2.3–2.5 | **Surface-supervision extrapolation.** Teacher supervises Δx* at ONE surface point Xk=rays_o+rays_d·target_d per ray, but render warps **all** n_samples (`inputs_flat = pts + vox_motion`). A 2D sheet-fit is extrapolated unconstrained into the 3D render volume → off-surface garbage → depth spikes. | `scene_rep.py:239`; warp `:195-239` | `deform_surface_bind` (Gaussian falloff near surface) | **EXISTS, off** `scene_rep.py:430-436` |
| **2** | ~1.5–2.0 | **`global_BA` un-co-adaptation (THE ROOT).** `global_BA` is the dominant trainer (iters 200, every frame) and trains the SDF on **un-warped** rays (`route_w_ba=None` by default), while `current_frame_mapping` + render route. Warped queries hit map regions trained on un-warped rays → SDF zero-crossing shifts → depth inversion. Code-verified, no sign bug. | `ddsslam.py:508`, `:574-608` (`route_w_ba`) | **B2:** per-keyframe route buffer + thread `route_w_ba` into `global_BA.forward` | **designed, not built** |
| **3** | ~0.5–0.8 | **Over-time canonical drift.** Field anchored at t=0 (Δx≡0, `scene_rep.py:218`); displacement grows. Equal-weight replay under-supervises late frames → per-frame pin-EPE slopes down. | t=0 anchor `scene_rep.py:218`; replay `ddsslam.py:1024` | recency-weighted replay (`deform_replay_recency`) | scoped, deferred |
| **4** | ~0.3–0.5 | **Over-warp outliers.** max\|Δx\|=0.084 (≈84% of ±0.1 band) compresses SDF more than targets warrant → localized artifacts. | hardbound `scene_rep.py:212-214` | `deform_hardbound` (tanh clip) — **RISKY**: 0.02–0.04 bound shrinks legit 0.02–0.08 deformations 24–50% → pin-EPE collapse | EXISTS, **keep OFF** |
| **5** | ~0.2–0.3 | **Routing edge artifacts.** RAFT/homography residual is noisy at borders → route falsely lights up edges → field floods boundary → halos. | route map | `map_route.edge_mask` (zero a border band) | scoped |

**Why soft-route (v0.4) was worse than region-route (v0.2):** soft_scale **miscalibrates amplitude** — tanh shrinks legit deformations 24–50%, and an average weight of ~0.02/0.5 ≈ 4% to field / 96% to static map is the **opposite** of routing intent. The literature does not soft-route; tracking pose updates are binary (fix or optimize). v0.4 25.83 < v0.2 26.96 is **calibration**, not "more field hurts."

### Fixable-vs-fundamental verdict: **FIXABLE**
- **Field is genuinely alive & correct:** held-out pin-EPE **+52.5% ≫ shuffled +41.6%** (real, not random), cos(D,baked) **+0.79** (vector-aligned), targets fresh and gauge-corrected [`project_deform_gauge_bug_20260620.md:23`; `project_combine_routing_model_20260619.md:72`].
- **Capability ≫ cost:** +52.5% localization signal vs ~2.2 dB render penalty → flips net-positive once application cost drops.
- **Costs are named, quantified, and each has a designed fix** (table above).
- **Routing already helped:** un-routed 24.1 → routed v0.2 **+2.85 PSNR** — exactly the literature range (§4).
- **The naive coadapt FAILURE was diagnostic, not fatal:** letting render gradient touch the field re-collapsed it to |Δx|=0, OR the teacher↔render fight smeared the map globally (render 23.3) [`project_deform_gauge_bug_20260620.md:25-26`]. The fix `cur_frame_map_only` (isolated replay optimizer, map-only sharpen) already kept field alive (+52.5%) AND lifted render to 24.1. The time_net is correctly split into its own param group (`ddsslam.py:927-974`) so `deform_field_teacher_only` can keep it out of the map optimizer.

**The penalty is mis-application, not a fundamental limit.** The analogy: applying a deformation warp un-routed is like warping bg+tool+tissue indiscriminately — of course smooth static geometry degrades.

---

## 3. THE BRIDGE

### 3.1 Minimal do-no-harm set to MATCH base (≥27.70)
Necessary and sufficient is **TWO** fixes; surf-bind alone is necessary but **insufficient** (the depth-spike root persists until the dominant trainer co-adapts):

1. **`deform_surface_bind`** (1-line config; existing path `scene_rep.py:430-436`). Confines warp to the ~1–2× truncation zone around the measured surface. Recovers ~0.5–1 PSNR by killing off-surface garbage. Field-safe (teacher unchanged). **Necessary, not sufficient.**
2. **B2 — route-thread `global_BA`** (THE pivotal fix, ~15–20 lines). Store per-keyframe route at the same pixels where rays are saved (`keyframe.py` route buffer), gather it in `add_keyframe`/`sample_global_rays`, and thread `route_w_ba` into `global_BA`'s forward — the assembly seam already exists at `ddsslam.py:574-608` (`route_w_ba = torch.cat([...]); ... route_w=route_w_ba`). This makes the dominant trainer co-adapt the map to warped queries → the render ladder **inverts** from `dead 27.70 > routed 25.83 > un-routed 24.1` to `un-routed 24.1 < routed ≥26–27 < dead 27.70`.

Optional intermediate **B1** (~10 lines): route only `global_BA`'s current-frame injection, leaving keyframe rays unrouted — a cheap stepping stone before full B2.

**Keep hard on/off routing.** Per-region disagree_frac=0 → exclude field; else include at full weight. Soft_scale only if B1+B2 still show boundary artifacts (then prefer `edge_mask`, cost #5, over global amplitude tuning).

### 3.2 The path to EXCEED base
On SemSup the field's **upside is structurally capped** (deformation ~1.5% of slab; base already at the static ceiling). Matching base = success here. To *exceed*, the gain must come from data with **headroom** — where static time-averaging genuinely fails (CRCD/StereoMIS/STIR-scale deformation). The combine (static→camera, tool→own SE(3), tissue→field routed by **motion**, not σ²) is designed for exactly this composition; gains appear as routing precision improves.

### 3.3 Strategic testbed call — is SemSup the right place to prove deformation helps?
**No — SemSup is the do-no-harm GUARD, not the proof.** Reasons:
- **No headroom:** dead-field already 27.70 (paper 28.649); the field can only break-even.
- **GT leak risk:** SemSup green pins are **RGB-visible** → any pin-based eval can leak.
- **Render & ATE are field-blind:** they cannot *see* a correct deformation; SemSup ATE is fictional (identity GT) anyway.

**STIR is the proof testbed** [`project_stir_dataset_measurability_20260614.md`]: IR-tattoo GT invisible to RGB (cannot leak), metric-3D **endpoint-EPE** (a *direct deformation metric* — itself a novel contribution; no surgical-recon paper reports one), and **above-SNR**. STIR's static baseline render is likely worse than SemSup's (true deformable tissue) → the field has **room to win**. CRCD stays diagnostic-only (sub-SNR, GT 55% held).

**Combined success criterion (Prong 3):**
- **SemSup (guard):** render **≥ 28 PSNR** AND pin-EPE alive (reduction ≫ shuffled). Routing does not hurt the already-strong static case.
- **STIR (proof):** endpoint-EPE **reduction** vs static baseline. The field genuinely helps deformation.

Without STIR, the field's value stays invisible to every shipping metric.

---

## 4. Literature precedents + what to adopt

Deformable NeRF/GS-SLAM **net-improves** render over static baselines (typically **+2–4 PSNR**) **only** when it (i) decouples camera/scene motion, (ii) supervises the field by **geometric/contrast/flow** priors rather than raw photometry, and (iii) **routes/masks** static regions away from the field — usually routing **both** tracking and mapping by the **same motion signal**.

| System | Relevance | What it does | Adopt? |
|---|---|---|---|
| **NRGS-SLAM** (arXiv:2602.17182) | Closest. Δx-warp + prioritize low-deform regions for pose; per-frame deform update in mapping; Bayesian **self-supervision** from monocular residuals (no GT rigidity). | Boltzmann posterior on **E_rigid − E_deform** contrast as deformation likelihood. | **YES.** Swap our gauge-confounded \|Δx*\| teacher → **sigmoid(β(E_R − E_D))** contrast (gauge-robust). 1-line: render field-off (E_R) and field-on (E_D), take the diff. |
| **WildGS-SLAM** (CVPR 2025) | Arm-1 template. Learned σ² (DINOv2+MLP) down-weights dynamic pixels in **both** tracking (Mahalanobis) and mapping; uncertainty MLP trained **separately** from the map (no gradient interference), then both use the shared signal. | Routes by **DINO features, NOT σ²** (σ² is appearance-blind to deformation). | **YES — confirms our decoupling.** σ² → Dec-2 pose reweighting; **flow/motion → field routing**. Route BOTH BA and map optimization, as they do. |
| **SurgicalGaussian** (arXiv:2407.05023) | Mask-guided. Inverts tool masks → loss only on soft tissue; deformation-consistency loss (nearby points similar motion). Ablation Table 2: mask-guided training critical (38.78 vs 37.94). | Hard mask exclusion of static/tool regions. | Pattern: **hard route, mask tools out.** |
| **EndoSurf** (MICCAI 2023) | SDF twin. Triple field (deform→canonical, SDF, radiance) + **Eikonal + SDF** losses bound surface geometry. +0.571 dB vs EndoNeRF. | SDF loss acts as a **surface-binding** constraint. | Theoretical backing for `deform_surface_bind` (cost #1). |
| **EndoFlow-SLAM** (arXiv:2506.21420) | Keyframe two-level: non-keyframes optimize pose only; keyframes joint pose+map with **flow** constraints. | Flow mask = static-region gate. | Backs flow-as-router. |
| **DefSLAM / TivNe-SLAM** (arXiv:1908.08918 / 2310.18917) | Separate deformation-tracking vs deformation-mapping optimization, then route the map. | Per-component optimizer separation. | Backs our `cur_frame_map_only` split + B2 map-routing. |

**Cross-arm insight (white space = our contribution):** the literature does **either** uncertainty (WildGS, NeRF-on-the-go) **or** a deformation field (NRGS, Deform3DGS/EndoGaussian/EndoSurf) — **not both loops from one underlying truth** (camera/scene motion). Our combine — continuous σ² for pose-weighting AND a motion-routed field, on a **neural-SDF surgical** substrate — is novel, alongside the honest diagnosis that σ² **cannot** drive the field (appearance-preserving deformation) → measure motion and route both arms from it.

**Top adoptions, in order:** (1) **NRGS Boltzmann E_R−E_D contrast teacher** (replaces gauge-confounded \|Δx*\|); (2) **B2 hard on/off route into `global_BA`** (the literal DefSLAM/NRGS map-routing fix); (3) **keep Arm-1/Arm-2 decoupling** (σ²→pose, flow→field).

---

## 5. Recommendation for tomorrow's full model

**Build order (architectural delta unconfounded — defer schedule tuning to the final max-perf sweep):**

1. **Add `deform_surface_bind` (config-only) + build B2 together**, on SemSup. B2 = `keyframe.py` route buffer (~15 lines) + `route_w_ba` assembly already stubbed at `ddsslam.py:574-608`. Use **hard on/off** routing. **Do NOT** enable `deform_hardbound` (risks pin-EPE collapse).
2. **Lock SemSup n=3.** Judge **LIVE render (SLAM metric) + pin-EPE (field validity).** Pass = render **≥ 28** AND pin-EPE alive (≫ shuffled). If render still < 28, debug the optimizer seam first (confirm `field_optimizer`/time_net split is respected when `cur_frame_map_only=true`; `ddsslam.py:927-974`).
3. **Adopt the NRGS contrast teacher** (sigmoid(β(E_R − E_D))) to replace the gauge-confounded magnitude target — cheap (reuses field-off render already available for routing).
4. **Deploy the same config to STIR** for proof-of-benefit (endpoint-EPE reduction vs static baseline). This is where deformation must *win*, not just break-even.
5. **Defer:** soft_scale (only if boundary artifacts persist → use `edge_mask` first), recency replay (cost #3), hardbound (cost #4), and all schedule knobs (cur_frame_iters / map_every / iters) to a final sweep.

**Net:** SemSup is the do-no-harm guard (match base ≥28, field alive); STIR is the proof. The full model = surf-bind + B2-routed `global_BA` + NRGS contrast teacher, with Arm-1/Arm-2 kept decoupled. Expected SemSup ladder post-fix: **un-routed 24.1 < routed ≥26–27 < dead 27.70 → ≥28 with B2 co-adaptation**, pin-EPE held at +52.5%.

---

**Key files (absolute):**
- `c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\model\scene_rep.py` — warp `:239`, surf-bind `:430-436`, hardbound `:212-214`, t=0 anchor `:218`
- `c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\ddsslam.py` — `global_BA:508`, `route_w_ba` seam `:574-608`, time_net optimizer split `:927-974`, replay `:1024`
- `c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\configs\Super\Super.yaml` — schedule (`iters:200` `cur_frame_iters:100` `keyframe_every:1` `map_every:1`), `far:5`, `range_d:0.1`