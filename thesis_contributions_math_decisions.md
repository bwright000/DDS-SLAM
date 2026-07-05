# BDDS-SLAM — Contributions: Math & Decisions Reference

Working reference for the thesis writeup. For each contribution: the **statement**, the **math**
(LaTeX-ready), the **decisions** (what we chose, what we rejected, *why*), and the **evidence**
(metrics that settled it). Prose is the author's; this is the substance.

Conventions: poses `c2w` in OpenGL; camera intrinsics `f=f_x=f_y≈1096.7` px (rectified CRCD),
principal point `c=(c_x,c_y)`. Flow `f_k∈R^2` in pixels. Depth `Z` up-to-scale (MoGe-2 monocular,
per-snippet stereo-anchored). All tracking metrics are **Sim3** (scale-corrected) — see §9.

---

## 1. Uncertainty-aware down-weighting ported to neural-SDF surgical SLAM (Inc-1 / Inc-2)

**Statement.** Port the WildGS-SLAM/NeRF-on-the-go aleatoric-uncertainty template from Gaussian-splat
SLAM on casual video to a **neural-SDF SLAM for deformable surgery**, driving *both* the mapping
supervision and the tracking pose solve. Not a new uncertainty method — a substrate + domain transfer
with a surgical-specific teacher choice.

### 1.1 Math

Per-ray heteroscedastic aleatoric noise `σ²(r)` from a small head on the shared geometry feature
(`geo_feat`, v1) or a per-pixel DINO head (v2, WildGS-faithful). Gaussian negative log-likelihood
(Kendall & Gal 2017) on the photometric residual:

$$\mathcal{L}_{\mathrm{NLL}} = \frac{1}{N}\sum_{r} \tfrac{1}{2}\!\left( \frac{\lVert C(r)-\hat C(r)\rVert^2}{\sigma^2(r)} + \log \sigma^2(r) \right)$$

**Teacher choice** (what residual σ² is trained to explain) — the key surgical decision:

- `l2` (baseline, byte-identical to the original): residual `= ‖Ĉ−C‖²`. Trains σ² on raw photometric
  error → the `σ ∝ ‖C−Ĉ‖` degeneracy (NeRF-on-the-go): σ² becomes a **contrast/edge detector**.
- `depth` (ours): scale-invariant depth-consistency residual
  $$e_{\text{dep}}^2 = \left(\frac{\hat d - d}{|d|+\epsilon}\right)^2$$
  so σ² tracks **geometric/deformation** uncertainty, not contrast.
- `rgb_depth`: sum of the two.

**Stop-gradient (β-NLL, Seitzer et al. 2022).** The `e²/σ²` term back-props a `1/σ²`-weighted gradient
into the *prediction*, letting the model lower NLL by *inflating* σ² on hard pixels ("explain-away")
instead of fixing the prediction. We detach the prediction inside the residual:
`ĉ ← stopgrad(ĉ)` (and `d̂` likewise) → NLL trains **σ² only**; the prediction is shaped by its own
undistorted `rgb_loss`. Flag `detach_residual`.

**Inc-2 — tracking down-weight.** The learned σ² becomes a per-ray weight in the *pose* solve:

$$w(r) = \mathrm{clamp}\!\left(\frac{1}{\sigma^2(r)},\, w_{\min},\, w_{\max}\right),\quad \text{(detached)}$$

applied multiplicatively to the per-ray RGB and depth losses. **Empty-ray fix** (`track_w_fix`): where
accumulated opacity `acc(r) < 0.5` there is no confident surface → set `w(r)=1`; then renormalize
`w ← w / \overline{w}` so the down-weight *redistributes* trust at fixed total.

### 1.2 Decisions

- **Two consumers, one σ².** σ² is trained on the **mapping** residual only (`not tracking`), then
  *consumed* at tracking time as the down-weight. Rationale: the NLL needs the dense mapping supervision
  to be stable; tracking only reads it.
- **Depth teacher over l2** — motivated by the architecture read (§7): the l2 teacher aims the
  down-weight at *contrast*, but the disease is *deformation*. Depth-consistency is the doable structural
  signal (WildGS/on-the-go use SSIM+depth; we don't sample spatial patches, so depth-consistency stands in).
- **v1 (geo_feat) vs v2 (per-pixel DINO):** v1 = per-point head off the geometry feature; v2 = per-pixel
  DINOv2 head (WildGS-faithful). v1 is the canon result below; v2 is the open A/B.

### 1.3 Evidence & caveats

- **Canon: −23% Sim3 ATE (n=3)** on CRCD c1_001 rect: ATE_mean 3.15→2.43 mm, ATE_max 14.79→9.43 mm,
  |Pearson|_dom 0.81→0.97, all n=3 non-overlapping.
- **Confound (must state honestly):** the −23% bundles a σ² empty-ray inversion + an ≈×10 rgb/depth
  rebalance (the 4-agent review). `uncfix` (the corrected version) vs `unc` is the pending A/B that
  de-confounds the claim. **Do not headline −23% without the uncfix control.**
- **Architecture limit (§7):** the down-weight reaches ~100% of the tracking gradient (RGB-dominated),
  so it *can* bite — but the l2 teacher aims it wrong, mean-normalization redistributes rather than
  removes, and there is no deformation DOF. → partial, not complete, fix.

---

## 2. Label-free camera-vs-scene motion detection — the region-VOTE detector (C′)

**Statement.** Split camera egomotion from independent (tool/tissue) motion with **no trained
segmenter**: unsupervised DINOv2 grouping + a depth-signature egomotion fit + a geometric epipolar
agreement. Differs from WildGS (learned uncertainty MLP) and DynaSLAM-class (semantic masks); the value
is that surgical labels are scarce.

### 2.1 Math

**Districting.** DINOv2 (`vits14_reg`, registers absorb artifact tokens) patch grid, L2-normalized rows
`X`, clustered:
$$L = \mathrm{KMeans}_{k}(X),\quad k=12,\ n_{\text{init}}=4,\ \text{seed fixed}.$$
Districts are *semantic* (an instrument votes as one bloc regardless of screen position), not spatial.

**Per-district votes** (district `k` valid iff pixel count ≥ 50 and has depth):
$$f_k=\mathrm{med}\,\text{flow}_k,\quad Z_k=\mathrm{med}\,\text{depth}_k,\quad p_k=\text{centroid}_k.$$

**Depth-signature egomotion model** (the design equation — separates motion *type* by how flow scales
with depth):
$$f_k \approx u + v\,\zeta_k + d\,r_k\,\zeta_k,\qquad \zeta_k=\frac{Z_{\mathrm{med}}}{Z_k},\quad r_k=\frac{p_k-c}{r_{\text{norm}}}$$
- `u` = **turn** (camera rotation): depth-blind, every district shifts equally;
- `v` = **slide** (lateral translation): depth-scaled (near districts shift more, `∝1/Z`);
- `d` = **zoom / forward** (radial): depth-scaled and radial (`p_k−c`).
- `ζ_k` dimensionless ⇒ **MoGe's unknown scale cancels**; params live in px at the median-depth plane.
- De-rotation is **implicit** (depth-blind `u` vs depth-scaled `v,d`) — no const-velocity pose needed.

Solve `θ=[u_x,u_y,v_x,v_y,d]` by least squares on `[1,\zeta_k,r_k\zeta_k]`. **Robust refit:** residual
`ρ_k=‖f_k−\hat f_k‖`, `med=median(ρ)`, `scale=max(1.4826·MAD, 0.3px)`, inliers `ρ_k≤med+c·scale`
(`c=2.5`), refit on inliers. Independent movers (tool, deforming tissue) → large `ρ_k` → **excluded from
the vote** and down-weighted, label-free:
$$w_k = \frac{1}{1+\big(\max(0,\rho_k-\mathrm{med})/\mathrm{scale}\big)^2}.$$

**Decision rule C′** (the incumbent, held-out-validated). Freeze (camera still) iff
$$q_{10} < \tau_q\ \ (2.5\text{ px}) \quad\text{OR}\quad \mathrm{dis3} > \tau_d\ \ (0.5),$$
where
$$q_{10} = P_{10}\big(\{\lVert f_k\rVert : k\ \text{valid}\}\big),\qquad
\mathrm{dis3} = \frac{|\{k:\ \mathrm{Sampson}(F,k)>3\text{px}\}|}{|\{k\ \text{valid}\}|}.$$
`q10` = 10th-percentile district flow ("do the *quietest* districts move"); `F` = RANSAC fundamental
matrix on the dense flow; Sampson = per-district pooled epipolar distance.

### 2.2 Decisions

- **Why `q10`.** Stillness requires *unanimity of the quiet*; motion is proven by the quietest district
  moving. Catches the "big mover + quiet majority" still frame (e.g. a stapler firing at 15–77 px while
  the far liver stays 0.8–5 px → `q10≈1.0` → correctly still) that a mean/median statistic misses.
- **Why `dis3`.** "Does *any* single rigid-camera story explain the chamber?" A majority of dissenting
  districts ⇒ don't trust the tracker this frame.
- **Why OR, not AND.** The AND-form `q10<2.5 ∧ dis3≤0.15` hit E3 precision 81% but collapsed C1 recall to
  49% — the *conjunction failure*. OR keeps both. The asymmetry (a wrong freeze costs missed motion in mm)
  is priced in the damage metric; C′ is never net-harmful on any snippet.
- **Exclude-don't-veto.** Tools/deformation fall out as fit *outliers* (down-weighted), never as a hard
  veto — a hard tool-veto froze the wrong frames on E3 (the old gate's inversion).
- **`k=12`.** k-sweep: k=24 **died held-out** (C2 precision 86→84, agree 80→78), k=6 kills E3 recall
  (86→69, the quiet minority merges into moving districts). 12 confirmed.
- **Registers.** `dinov2_vits14_reg` for live grouping (register tokens absorb artifact tokens); plain
  `dinov2_vits14` for the baked WildGS-faithful features.

### 2.3 Evidence

- **Held-out 4/4** (freeze precision / still-recall): E3 64.3/85.7, C1 91.9/73.1, **C2 86.2/81.7,
  C3 87.9/80.5** (C2/C3 held out). The only rule that holds on all four.
- Synthetic gate `test_region_vote.py` V1–V6 PASS (still / turn / slide / zoom / tool / turn+tool).

---

## 3. One motion signal, two consumers (freeze + tracking down-weight)

**Statement.** The *same* region-vote motion signal feeds **both** halves of SLAM: a freeze decision
(tracking gate) *and* a per-ray tracking down-weight (`vote_trust`). Not seen in surgical SLAM; the
substrate for the deformation-adaptive map (§ future).

### 3.1 Math

Two consumers of the district trust `w_k` (§2.1) → dense trust map `W(x)=w_{L(x)}`:

- **Consumer A — freeze** (`vote_freeze`): C′ decision (§2.1) → copy previous pose, skip the solve.
- **Consumer B — down-weight** (`vote_trust`): feed `W` as the per-ray weight `w(r)` into the RGB pose
  solve (which owns ~100% of the tracking gradient, §7). Deforming/tool rays get low trust; tracking
  stays **continuous** (no freeze ⇒ no cold-start lurch, no BA keyframe jitter — §8).

Arms: `vote` (A only) · `vote_trust` (B only) · `vote_both` (A+B).

### 3.2 Decisions

- **Down-weight is the moving-frame lever.** The oracle decomposition (§6) proves freezing is a
  *still-frame* tool; the residual disease is *moving-frame* over-travel. Consumer B targets exactly that,
  label-free, and — unlike σ² (§1) — its trust is keyed to **deformation** (fit residual), not contrast.
- **Failure mode to watch.** If the *whole* field deforms (E3 busy scenes), the MAD scale normalizes and
  all districts get ≈equal trust ⇒ no effective down-weight — the same normalization limit as σ².

### 3.3 Evidence

Running (E3 bake-off): `vote_trust` vs `uncfix` vs `dpool` vs base noise floor vs oracle ceiling, judged
on the **moving-region path-ratio** (base 1.20 → oracle 2.04 in Sim3-metric; target → 1.0). *[pending]*

---

## 4. Depth-free rotation sensor + de-rotation (dis_rot)

**Statement.** A 3-dof rotational-flow fit that (a) recovers camera rotation **metrically from flow
alone, depth-free**, and (b) de-rotates the flow so the residual is translation-parallax + scene-motion
only — the regime where depth-normalization is valid. Yields the first **polarity-stable** dissent
feature.

### 4.1 Math

Pure camera rotation induces flow = a homography `K R K⁻¹` — depth never enters. Small-angle linearized
(`x'=x−c_x, y'=y−c_y`):
$$
u = \omega_x\frac{x'y'}{f} - \omega_y\Big(f+\frac{x'^2}{f}\Big) + \omega_z\,y',\qquad
v = \omega_x\Big(f+\frac{y'^2}{f}\Big) - \omega_y\frac{x'y'}{f} - \omega_z\,x'.
$$
Linear in `ω` ⇒ stack `Aω=b` over a stride-16 grid, solve by least squares, **MAD-IRLS** (3 iters,
tools drop as outliers). The `(x/f)^2` **edge terms are the model** — the ≈34% edge-flow boost at our
≈60° FOV is used to the extent it is measurable.

- **Rotation sensor:** `|ω̂|` (deg/window). Depth-free, no scale ambiguity.
- **De-rotation:** residual `ρ^{\text{derot}}_k = \lVert f_k - \hat f_k(\hat\omega)\rVert`. On this residual
  the `×Z` normalization ("same plane") is *valid* (the depth-blind rotation term is removed).
- **`dis_rot`** = fraction of districts with median `ρ^{\text{derot}} > 3` px.

### 4.2 Decisions

- **Why a constrained 3-dof story, not a generic 8-dof homography.** 8 params absorb half the scene
  motion; the constrained rotation story is too rigid to be hijacked. IRLS anchors on the quiet majority,
  so scene movers dissent as a *minority* while camera *translation* makes the *whole* chamber dissent
  (parallax) — dissent **size** finally separates camera from scene.
- **Depth normalization is rotation-unsafe — this is the fix.** Plain `×Z` (`depth_pool`) *creates*
  depth-proportional spread on rotation frames (measured: dispersion rises). De-rotate *first*, then `×Z`.

### 4.3 Evidence

- **Rotation sensor metrically validated:** est/GT gain **1.02 (E3), 0.99 (C2)** on real-motion windows;
  |gain−1| ≤ 2%. C3 0.45 = sub-SNR floor (GT rotation ≈ 0).
- **`dis_rot` never inverts:** AUC 0.82 / 0.73 / 0.82 / 0.48 across E3/C1/C2/C3, vs `dis3`'s 0.81 / 0.30 /
  0.52 / 0.25 (inverted on 2/4). Rigid-story-can't-be-hijacked confirmed.
- **But `quiet2` (`dis_rot<0.25 ∧ rot<0.30°`) died held-out** (C3 p75) → `dis_rot` ships as a *diagnostic
  / instrument*, not a decision rule. Honest.

---

## 5. [Supporting] The flow-plateau — a rigorous negative result

**Statement.** Flow-only statistics **cannot** decide camera-vs-scene under *simultaneous* camera+scene
motion. Established by an AUC ranking audit and confirmed 3× by independent feature families dying
held-out. Motivates the external (map) reference.

### 5.1 Analysis / math

- **AUC audit** (probability a random GT-moving frame outranks a random still frame): no single flow
  feature ≥ 0.85 anywhere, and the *best* feature differs by regime — E3 `dis3` 0.81 vs C1 `cam_mag` 0.84.
- **The `dis3` polarity flip** (the crux): AUC 0.81 (E3) vs 0.30 (C1). Mechanism: `F` (8-dof, RANSAC)
  crowns the biggest coherent bloc as "the story"; on C1 drift-stills it fits the *drag*, so the still
  background becomes the dissenter → the statistic **inverts**.
- **Structural ambiguity** (Saputra single-residual): C1 drift-stills need `incoherent→freeze`;
  E3 busy-moving need `incoherent≠freeze`. One residual cannot encode both.

### 5.2 Evidence (3× confirmation)

Three feature families each won the E3+C1 design set and **died held-out**: (i) threshold candidate-A
(`q10<2.5 ∨ dis3>0.30`, E3 p51 fails bar); (ii) `pgain` (E3 p100 but **inverts on C3**); (iii) rotation
`quiet2` (E3 p91 but **C3 p75**). → the route above C′ is an *external reference* (the map), not another
flow feature.

---

## 6. [Supporting] The oracle decomposition — freezing is a still-frame tool

**Statement.** With a GT-perfect freeze detector, freezing does **not** fix E3 (fundamental moving-frame
problem) and only *partly* fixes C1. Bounds the entire freeze-gate direction; localizes the disease to
moving frames.

### 6.1 Math

- **Oracle gate:** freeze iff GT translation step `‖t_i − t_{i-1}‖ ≤ 10^{-4}` mm (matches `flow_diag`
  `gstill`; the rotation guard is loosened to `10^{-2}°` because the GT quaternion has a constant
  `2.3×10^{-4}°` quantization floor on bit-identical rows).
- **First-order oracle simulation** (predict the freeze effect from a base trajectory, no GPU): re-integrate
  with GT-still relatives zeroed,
  $$C'_i = \begin{cases} C'_{i-1} & \text{still}_i\\ C'_{i-1}\,(C_{i-1}^{-1}C_i) & \text{else}\end{cases}$$
  then Sim3-align and re-measure.

### 6.2 Evidence

| | base ATE | oracle ATE | path-ratio | verdict |
|---|---|---|---|---|
| E3 | 4.94 | **4.67** (−5%) | 1.35→**1.04** | fundamental — freezing fixes path-ratio, not ATE; Sim3 scale stuck ≈3× (depth-scale error) |
| C1 | 2.03 | **1.16** (−43%) | 5.50→**2.32** | gate halves ATE, but **moving frames still over-travel 2.3×** |

Decomposition (Sim3-metric, GT-still vs GT-moving regions): E3 GT is still 61 / moving 145 / still 60
frames. Moving-region ratio base **1.20** → oracle **2.04**. **Two distinct still-over-travel sources:**
non-keyframe tracking jitter (freezable) + BA keyframe jitter (§8, *not* freezable by a tracking gate).
⇒ still frames = gate; moving frames = down-weight. Complementary, quantified.

---

## 7. [Supporting] DDS tracking is photometric — the architecture read

**Statement.** The per-frame pose solve is ~100% RGB-photometric; SDF geometry (weight 1000) built the
map but is inert in tracking. Explains the moving-frame over-travel and where the down-weight bites.

### 7.1 Math / measurement

Tracking loss:
$$\mathcal{L} = w_{\text{rgb}}\mathcal{L}_{\text{rgb}} + w_{d}\mathcal{L}_{d} + w_{\text{sdf}}\mathcal{L}_{\text{sdf}} + w_{\text{fs}}\mathcal{L}_{\text{fs}},\quad (w_{\text{rgb}},w_d,w_{\text{sdf}},w_{\text{fs}})=(5,\,0.1,\,1000,\,10).$$
**Measured budget** (weight × median raw loss, from `debug_log`, base + best): **RGB ≈ 100%**, depth
≈ 0.2%, SDF/fs ≈ 0%. The 1000× SDF weight dominates *mapping*; in tracking the map is fixed, so the
near-surface SDF residual sits flat at its minimum (fires 21/106 frames) — RGB texture carries the pose
gradient.

### 7.2 Consequence (the mechanism of over-travel)

RGB photometric alignment **cannot separate camera motion from tissue motion** — both shift the texture.
On a static map, moved tissue pulls the *camera* to chase it ⇒ moving-frame over-travel. σ² (§1) reaches
this (RGB owns the gradient) — hence it works at all — but (i) l2-teacher aims it at contrast; (ii)
`w/\overline{w}` normalization redistributes; (iii) no deformation DOF ⇒ the residual stays ambiguous.
Depth is near-inert in tracking (0.2%) despite up-to-scale MoGe supervising the *map*.

---

## 8. [Supporting] Freeze / BA machinery — cold-start, BA keyframe jitter, `freeze_ba`

**Statement.** Hard freezing interacts badly with (a) the const-velocity init and (b) global bundle
adjustment, injecting jitter. `freeze_ba` (and its cleaner fixed-node alternative) is the remedy;
`consistent_poses` handles the bookkeeping.

### 8.1 Math / mechanism

- **Const-velocity init:** `C_i^{\text{init}} = (C_{i-1}C_{i-2}^{-1})\,C_{i-1}`. After a freeze,
  `C_{i-1}=C_{i-2}` ⇒ `delta = I` ⇒ **velocity zeroed** ⇒ the next tracked frame **cold-starts**; DDS's
  lazy tracker (base under-shoots: 0.27 mm vs 0.82 mm GT/frame) then over-shoots from the cold start.
- **BA keyframe jitter (the dominant still-region source).** `global_BA` re-optimizes keyframe poses
  every keyframe cycle *even when the camera is dead still*. Measured: under the oracle, **100% of
  still-region motion is on keyframes** (period-5 = `keyframe_every`); the non-keyframes are perfectly
  frozen (0.000 mm). This is BA absorbing map noise/deformation into camera-pose change — the §7
  camera-vs-scene ambiguity, in the *mapping* half.
- **`freeze_ba`:** re-apply the gate-frozen pose to keyframes *after* BA:
  `est[f] ← gate_fixed[f]` for keyframes `f`. Flags: `flow_track.freeze_ba` (flow/vote) or
  `tracking.freeze_ba` (oracle).
- **`consistent_poses`:** re-anchor non-keyframes onto BA-moved keyframes,
  `est[f] ← rel[f] · est[\text{kf}(f)]` — kills the BA-jump extrapolation bookkeeping (does *not* freeze
  keyframes).

### 8.2 Decisions

- **`freeze_ba` safety scales with gate precision.** It locks in false-freezes on keyframes (which BA can
  no longer correct). Safe at oracle precision (100%); risky on E3 (64% → false-frozen keyframes anchor
  wrong non-keyframes); safer on C1 (92%).
- **Cleaner alternative:** hold frozen keyframes **fixed as constants inside** BA (fixed-node BA) — the
  map co-optimizes around them, no map–pose mismatch. `freeze_ba` (post-hoc override) is the quick test;
  fixed-node is the correct version if the mismatch bites.

---

## 9. [Methodology] The metric arbiter

**Statement.** Metrics are the only arbiter. The single biggest trap is rigid-vs-Sim3 ATE on up-to-scale
depth; the whole battery is built to be scale- and gauge-robust.

### 9.1 Math

- **Sim3 ATE (Umeyama, with scale).** `min_{s,R,t} Σ_i ‖s R\,\mathbf{p}_i + t - \mathbf{g}_i‖²`. 🚨 The
  pipeline's own rigid Horn (`tools/eval_ate.py`) has **no scale** → on up-to-scale MoGe depth it is
  dominated by the ≈8× scale mismatch and **inverts A/Bs**. Always Sim3.
- **Path-ratio** = `s · Σ‖ΔP‖ / Σ‖ΔG‖` (over/under-travel; Sim3-scale-corrected).
- **Aligned-Pearson (dominant axis)** — shape tracking, computed on **Sim3-aligned** trajectories
  (rotation-confounded if computed pre-alignment).
- **Rotation channel** (thesis-novel, gauge-free): relative rotation steps
  `Δθ_i = \arccos((\mathrm{tr}(R_{i}^\top R_{i+1})-1)/2)` need **no** Sim3 (gauge-free) and have **no
  scale ambiguity** ⇒ `rot_path_ratio = Σθ^{\text{est}}/Σθ^{\text{gt}}` is an *absolute* over/under-travel
  measure (unlike translation path-ratio, which rides the Sim3 scale).
- **Freeze-confusion:** `precision = P(\text{GT-still}\mid\text{frozen})`, `recall = ` coverage of GT-still
  frames. The metric that vindicated the C1 gate (93% precision) and exposed the E3 inversion (28%).

### 9.2 Decisions

- **n=3 seeds always.** Seed coin-flip is real; base seed-std = the noise floor a win must clear.
- **Sub-SNR quoting.** On tiny-motion snippets (E3, StereoMIS) quote **path-ratio + Pearson**, not ATE
  alone (ATE is misleading at sub-SNR).
- **Dataset-specific arbiter.** CRCD = Sim3 ATE (pose GT real); SemSup = render PSNR/SSIM/LPIPS only
  (pose GT fictional/identity).

---

## Appendix — claim ↔ status ledger (for honest positioning)

| # | Claim | Status | Evidence gate |
|---|---|---|---|
| 1 | Uncertainty transfer → SDF surgical SLAM | **strong, confounded** | −23% n=3; needs `uncfix` de-confound |
| 2 | Label-free camera-vs-scene voting (C′) | **validated** | held-out 4/4 |
| 3 | One signal, two consumers | **in test** | E3 bake-off (`vote_trust`) pending |
| 4 | Depth-free rotation sensor + de-rotation | **validated (instrument)** | est/GT 1.02/0.99; `dis_rot` no-invert |
| 5 | Flow-plateau (negative result) | **established** | 3× held-out death + AUC audit |
| 6 | Oracle decomposition | **established** | oracle run + first-order sim |
| 7 | Photometric-tracking architecture | **measured** | debug_log budget, base+best |
| 8 | Freeze/BA machinery + `freeze_ba` | **built, in test** | E3 fba arms pending |
| 9 | Metric arbiter | **methodology** | rigid-vs-Sim3 inversion demonstrated |
