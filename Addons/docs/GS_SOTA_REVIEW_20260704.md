# GS SoTA Review & Plan Revision (2026-07-04)

> 9-agent research sweep (6 web SoTA + 3 local code audits, ~776k tokens) reviewing
> `GS_PORT_BUILD_MAP_20260703.md` against the mid-2026 state of the art. Raw findings with all
> citations: `GS_SOTA_SWEEP_RAW_20260704.md`. This doc = the verdicts and the concrete plan changes.
> NOTE: the adversarial-verify pass was cut short (spend limit); verdicts below marked [SWEEP] are
> single-sourced from the sweep and should be spot-checked when load-bearing.

## 1. The four landscape shifts (what changed since 2026-06-22)

**S1 — NRGS-SLAM (arXiv 2602.17182, Feb 2026) occupies our P1+P2 square at paper level.**
Monocular ONLINE non-rigid endoscopic GS-SLAM: per-Gaussian temporal Gaussian-basis deformation
(Deform3DGS's FDM run online) + a learned per-Gaussian deformation probability that down-weights
deforming regions in tracking. Co-author: **Hesheng Wang — IRMVLab, the DDS-SLAM lab**. Code
"upon acceptance" (none as of 2026-07-03). Their streaming-time solution: **basis centers anchored
to keyframe timestamps, tau/sigma frozen as priors, per-frame residual updates on basis weights.**
Same lab also shipped Local-EndoGS (windowed offline 4D endo GS) and D2GSLAM — **the original
authors abandoned the neural-SDF line too** (one-sentence thesis validation; monitor
github.com/IRMVLab monthly for code drops).

**S2 — 4DTAM (CVPR 2025, Matsuki/Davison, CODE RELEASED) = first general online non-rigid GS-SLAM.**
2DGS surfels + global time-MLP warp + random-historical-keyframe replay + ARAP; tracks against the
map WARPED to the latest keyframe's timestamp (independent confirmation of our BUILD LAW); judged
on held-out views. "First online non-rigid GS-SLAM" is gone in general vision — our claim must be
surgical-specific. Its representation class (global time-MLP trained by render losses) is exactly
what died in DDS-SLAM; replay is what keeps theirs alive. Usable as a runnable reference/baseline.

**S3 — Free-DyGS (arXiv 2409.01003, Jin lab, no code ever released) already ran the Deform3DGS
basis ONLINE on surgical video** (pose-free, frame-by-frame): absolute-frame-index time, **partial
activation** (only near-current bases optimized), **Retrospective Deformation Recapitulation**
(trailing replay window w=100) against overwriting history. This + S1 means our P2 streaming-time
"research crux" has two published solutions — P2 drops from research-risk to engineering-with-
precedent, and our sliding-window-local-time first cut is superseded (renormalizing time as the
clip grows shifts every basis center = forgetting by re-parameterization).

**S4 — T2GS (MICCAI 2025, Speidel lab) models the tool, offline.** Pose-guided global rigid
trajectory + local shape change, composited with deforming tissue. The literal claim "all surgical
GS stops at masking the tool" is DEAD. The narrowed claim SURVIVES and was verified across
EndoGSLAM / Endo-2DTAM / LumenGSLAM / NRGS-SLAM: **no ONLINE surgical GS-SLAM models the tool as a
tracked object** — P3 repositions to "first online, unknown-pose, with enter/exit lifecycle, over
deforming tissue" and cites T2GS as the offline precedent (adopt its tool-region/tissue-region
split-metric protocol).

## 2. Niche & protocol verdicts

| Question | Verdict (as of 2026-07-03) |
|---|---|
| Released code for online deformable surgical GS-SLAM? | **Still NONE** (NRGS/Free-DyGS/EndoFlow/Local-EndoGS/D2GSLAM all unreleased or placeholder repos) → we can still be the **first released artifact**; paper-level "first" is gone. |
| Online tool-as-object in surgery? | **Open.** T2GS = offline; everything else masks/inpaints (EndoGS, EndoGaussian, SurgicalGaussian, Diff2DGS, G-SHARP). |
| Is our held-state protocol eccentric? | **No — it matches GS norms** (EndoNeRF 7:1 held-out from final model; SplaTAM-lineage evaluates post-hoc; 4DTAM held-out extrapolated views; SLAM&Render benchmark documents train-view overfit). The NOVEL part is the explicit recency-vs-held-state factorial showing protocol INVERSION (−0.24 vs +1.25 dB) — write that as a protocol-analysis contribution. Report BOTH protocols. |
| Is EndoGSLAM still the right base? | **Yes for continuity** (running on CRCD, adapter built), but it is 2023-era: no BA at all (do_ba dead), photometric-GD tracking, C3VD-only upstream. **Endo-2DTAM (ICRA 2025, code)** is the released stronger-geometry alternative (2DGS surfels, point-to-plane tracking) — treat as the upgrade path if jitter persists, not a base swap now. EndoFlow-SLAM beats EndoGSLAM on C3VD (+3 dB, ATE 0.34→0.23 mm) but repo is a README placeholder. |
| Uncertainty in endoscopic GS-SLAM? | **Slot is EMPTY** → P1 is first-in-domain. But cite/differentiate: WildGS-SLAM (image-space DINO→β, CVPR 25, Apache-2.0), VarSplat (per-splat variance, CVPR 26), NRGS (per-Gaussian deform probability), ConfidentSplat, Feature-EndoGaussian (offline DINO-in-surgical-GS — tightens our DINO wording). |

## 3. Plan changes (numbered, actionable)

### P2 — deformation (biggest redesign)
1. **Streaming time = absolute-time keyframe-anchored local RBF bases + partial activation +
   trailing replay** (the Free-DyGS/NRGS recipe), NOT sliding-window local time. Spawn basis
   centers at keyframe timestamps (EndoGSLAM already records per-Gaussian spawn `timestep` in
   params.npz — free anchor); freeze old bases; optimize only near-current bases + per-frame
   residuals. Replay is MANDATORY: every streaming paper surviving >300 frames has keyframe
   refit/replay; held-state eval is exactly the judge that exposes its absence. Fallback
   parameterization if RBF tails leak: SE(3) B-splines (CVPR 2026, code) — strictly local support.
2. **Deform3DGS lift corrections (local audit, file:line in raw doc):** coefs stored FLAT
   [N,510] (=10ch × {w,μ,σ} × 17); bump is QUARTIC exp(−(((t−μ)²/(σ²+1e-4))²)) — copy the exact
   form incl. ε=1e-4; deltas add to RAW params (log_scales, unnorm quats) BEFORE activations;
   opacity never deformed; weight=0 init ⇒ **free bit-parity test vs stock EndoGSLAM**; own
   optimizer group lr≈1.6e-3; `_deformation_table` is a no-op — skip it; 17 bases is a
   clip-length-tuned constant — re-derive from basis density (~1 per 9 frames); beware
   `.squeeze()` at N==1 and in-place `+=` on leaves. **Budget ~60–100 lines** (param-lifecycle
   plumbing at add/prune/save), not "~10 lines".
3. **🚨 BUILD-LAW hook trap (the audit's most consequential catch):** EndoGSLAM has TWO transform
   paths — `transform_to_frame` (tracking/mapping/spawn, slam_helpers.py:226-261) and
   `transform_to_frame_eval` (:264-292) which is what gs_eval/eval_save render with. A deform hook
   only in the online path is INVISIBLE at held-state eval = the exact ~0-gain gauge-mismatch cell
   of our factorial. **Hook BOTH** (or route eval through the online path with time index).
4. **Teacher stays our unique piece** — direct dx* displacement supervision vs everyone's
   render-only/photometric self-supervision (our NeRF autopsy is the argument for why that
   matters; Free-DyGS's basis under render-only survives via replay, but 4DTAM lists "2D priors
   like point tracking or optical flow" as their own missing piece). Teacher-bake corrections:
   baker lives at `Addons/deform/generate_deform_targets.py` (plan doc said Addons/dino — WRONG);
   CRCD bakes need `--est_c2w` + CRCD intrinsics + OpenCV ray convention (DDS default is OpenGL);
   the npz embeds a `gauge` key — keep that pattern. MoDGS warning: per-FRAME MoGe scale drift
   masquerades as deformation → rescale per-frame on rigid regions (vote-detector output) before
   baking dx*.
5. **Deform-vs-prune interaction:** the 0.1×scene_radius big-Gaussian prune can silently delete
   deform-scaled Gaussians — guard when RBF deforms scales.

### P1 — uncertainty + tracking
6. **Copy WildGS-SLAM's exact loss discipline:** (SSIM′ + λ·depth-L1)/β² + λ·log β +
   feature-similarity regularizer, and the TWO-WAY gradient detach (uncertainty loss never touches
   Gaussians; render loss never touches the MLP). Residual form (SSIM vs pure L2, VarSplat's
   argument) = one cheap ablation. Specularity caveat (WildGaussians): RGB-residual σ² conflates
   speculars with transients — fine for tracking down-weight, harmful in mapping; consider the
   DINO-feature-space residual term.
7. **EndoGSLAM's `energy_mask` is a NO-OP** (recon_helpers.py:104 returns ones; the paper's
   claimed brightness mask is disabled in code). The base has NO specular masking — possible
   jitter contributor and a cheap honest A/B lever (re-enable, flag-gated, metric-first).
8. **Free Inc-2 baseline already in the code:** per-pixel rendered depth-variance
   (E[z²]−E[z]², main.py:247) is computed and detached every iteration, currently only NaN-masked.
   Un-detach → fixed 1/σ² weight = the zero-cost first cell; the DINO head must then beat it.
   Injection points: the two masked SUM reductions (main.py:268 depth, :275 rgb) — sums, not
   means, so normalize weights per-mask-size. σ² needs no rasterizer change: the [z,1,z²]
   colors_precomp channel trick renders any per-pixel scalar.
9. **Jitter (path-ratio 93×) levers, evidence-ranked:** (a) GauS-SLAM visible-surface gating —
   track only against currently-visible local map (days); (b) Spectral GS-SLAM-style
   degeneracy guard — monitor the pose Hessian's weak eigendirections, shrink toward the
   zero-motion prior (the principled version of LEAN-CORE; IROS 2026, no code = we'd be early);
   (c) SmallGS-style feature-metric tracking on rendered DINO maps for small-baseline video;
   (d) G-ICP init only with scale care (up-to-scale MoGe). Also port the map-anchored still gate
   (`_still_gate_decide`) — renders zero-motion vs const-velocity hypotheses; translates to GS
   almost verbatim and GS renders are cheap. MASt3R-SLAM/VGGT = offline pose ORACLE for CRCD
   sanity checks only, not an integration.
10. **Port-table corrections (local audit):** there is no `Addons/tracking/` — zero-motion prior
    (flow_track.py:427-444) and region-vote detector (:319-424) live INSIDE the condemned
    flow_track.py → **extract functions, never import the module**. `uncfix` = the
    `uncertainty.track_w_fix` FLAG (A/B still pending) — port as a flag, not baked in; its
    empty-ray fix maps to the GS silhouette channel. The geo σ² head does NOT port (reads
    SDF-internal geo_feat); dino head + fuse modes ''/'rgbd' are representation-free. Vote
    detector should consume the baked *_dino.npy instead of live torch.hub DINO.

### Depth / metrics honesty
11. **🚨 Depth-L1 vs MoGe-2 is CIRCULAR** (eval prior == training prior) — cannot headline a
    reconstruction claim. Fix: (a) independent stereo depth on CRCD eval frames (we already
    stereo-anchor); (b) **add C3VD as a benchmark** — GT depth + GT pose, EndoGSLAM-preprocessed
    data exists, direct comparability with EndoGSLAM/Endo-2DTAM/EndoFlow published numbers;
    (c) STIR endpoint EPE for deformation GT; DynamicColon (per-timestep GT point clouds) as a
    candidate. Depth-L1-vs-MoGe stays as a consistency diagnostic only.
12. **One-line depth fix first:** normalize alpha-blended depth by accumulated alpha
    (both channels already rendered; Endo-2DTAM + GauS-SLAM both do this) — attacks the two
    documented alpha-depth biases at zero cost. Surfel-rasterizer swap is calibrated by
    Endo-2DTAM: −14% depth RMSE, ATE unchanged, PSNR −2 dB, +15% runtime → NOT default; late
    option via their repo as template. TSDF-fused mesh (Open3D, ~50 lines, offline) = the cheap
    reconstruction deliverable. Mip-Splatting: skip (fixed intrinsics, near-constant standoff).
    Pin the rasterizer commit (requirements.txt installs unpinned git).
13. **v1 holdout leak (audit):** held-out frames still enter the keyframe list
    (main.py:874-887 not gated) → HELD-OUT∩DYNAMIC contaminated for frames on the keyframe
    cadence; gate the keyframe-add on holdout or pick holdout_every coprime+offset.
    gs_eval is CRCD-layout-hardcoded → stage SemSup into the same assembled layout.

### P3 — tool-as-object (repositioned)
14. Claim = **first ONLINE surgical GS-SLAM with tool as tracked SE(3) object + enter/exit
    lifecycle, composited over deforming tissue**; T2GS = offline precedent; mask-and-inpaint zoo
    = motivation. Design template: driving scene-graphs (OmniRe/Street Gaussians; drivestudio
    code) — background node + rigid object node in canonical frame + per-frame SE(3), rasterize
    the union; tool SE(3) optimized with the SAME rasterizer machinery as camera pose. Pose
    ladder: masked-ICP → render-and-compare refine → FoundationPose only if needed (POGS = online
    grouped-Gaussian precedent). **Lifetime at the OBJECT level** (state machine: absent→entering→
    tracked→exiting→dormant, driven by mask coverage + pose-fit confidence) — per-Gaussian
    temporal opacity is prior art (Spacetime Gaussians et al.), cite to show deliberate choice.
    Identity: label Gaussians at spawn (spawn already flows through add_new_gaussians). Answer
    "why model vs inpaint": tool pose is a clinical output; compositing enables replay/removal;
    tool-region PSNR (T2GS protocol).

### Positioning / paper
15. Related-work scaffold: 3DGS-SLAM survey (2602.04251) + awesome-NeRF-and-3DGS-SLAM (recheck at
    submission). Key contrasts: WildGS rejects dynamics / we reconstruct them; NRGS
    self-supervises from photometric / we teacher-supervise displacement (and our autopsy shows
    why); 4DTAM global-MLP class / we use local bases; T2GS offline / we online. Wording: "online
    reconstruction OF deforming tissue during SLAM" (avoids the distractor-removal collision).
    Licenses: WildGS Apache-2.0 OK; RaDe-GS Inria non-commercial — avoid code lift; EndoGSLAM
    license still UNRESOLVED (C1 stands; verify agent was cut).

## 4. Revised risk register (delta)
- RETIRED: "streaming-time basis is unsolved research" (two published recipes; now engineering).
- RETIRED: "held-state protocol needs defending" (GS norms align; cite SLAM&Render/4DTAM).
- NEW: **scoop risk** — NRGS-SLAM/Local-EndoGS acceptance+code during our window (monitor
  monthly; accelerate P2; the "first RELEASED artifact" claim is time-sensitive).
- NEW: BUILD-LAW eval-path trap (#3) — cheap to fix, catastrophic to miss.
- NEW: circular depth metric (#11) — must fix before any reconstruction claim.
- UNCHANGED: EndoGSLAM license (C1); CRCD sub-SNR framing; A100-only rasterizer builds.

## 5. Reading list (before building)
1. NRGS-SLAM 2602.17182 — full read (the LOCKED precondition stands; now doubly so).
2. 4DTAM 2505.22859 + repo — replay/ARAP mechanics; possible baseline run.
3. Free-DyGS 2409.01003 — partial activation + RDR recipe.
4. T2GS (MICCAI 2025) — ETMM decomposition + tool-region metric protocol.
5. WildGS-SLAM 2504.03886 — exact loss + detach discipline.
6. Endo-2DTAM 2501.19319 — the calibrated surfel upgrade path.
7. Deform3DGS audit findings (raw doc §deform3dgs) — the lift contract.
