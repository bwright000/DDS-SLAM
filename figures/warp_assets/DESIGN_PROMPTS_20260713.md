# Claude Design handoff — two thesis figures (2026-07-13)

Attach the assets listed under each prompt. Both figures must be **vector output
(SVG + PDF)**, white background, and match the house style of the existing
Co-SLAM figure: dashed rounded grey group-boxes with bold serif group titles,
light-blue rounded nodes with thin blue outlines, mid-grey arrows, LaTeX
Computer-Modern-style serif labels, generous whitespace. Target width: full
`\linewidth` of an A4 single-column thesis (~160 mm), so keep smallest text
legible at that size.

---

## PROMPT 1 — BDDS-SLAM architecture (augment the Co-SLAM figure)

> You are given `coslam_architecture.svg` — the editable vector source of our
> Co-SLAM pipeline figure (dashed group boxes "Tracking process" and "Scene
> representation"; RGB-D stream → ray sampling; a 3D tissue point-cloud render;
> a sparse-hash-grid vertex diagram; encoder/decoder blocks). **Do not redraw
> it — augment it.** Produce ONE figure showing three architectural layers,
> visually distinguished:
>
> **Layer 1 — Co-SLAM base (keep, restyle to muted grey-blue):** everything in
> the given SVG. Joint encoding `γ(x) = [γ_ob(x), γ_hash(x)]` → geometry
> decoder `g_θ → (s, h)` → colour decoder `f_θ → c` → SDF volume rendering →
> rendered colour/depth supervising tracking (frame-to-model, frozen map) and
> mapping (field + keyframe BA).
>
> **Layer 2 — DDS-SLAM addition (one accent colour, e.g. amber; small "DDS"
> corner tag):** a deformation-warp block inserted BEFORE the joint encoding.
> Inputs: query point `x`, timestamp `t` (drawn from the ray). Frequency
> encodings `γ_freq(x), γ_freq(t)` feed a small MLP `D_φ` producing a
> displacement `Δx`; output `x_canon = x + Δx` flows into `γ(·)`. Annotate
> "anchored: Δx ≡ 0 at t = 0" and "supervised only through the render loss".
> Show the timestamp `t` also being attached to each sampled ray in the
> tracking box.
>
> **Layer 3 — BDDS-SLAM additions, OURS (second accent colour, e.g. teal;
> "ours" corner tags). Three flag-gated blocks:**
> 1. **Uncertainty head** — small head hanging off the geometry feature `h`
>    producing per-ray variance `σ²(r)`; trained by NLL of the photometric
>    residual; two consumer arrows: "mapping: down-weight residuals" and
>    "tracking: weight 1/σ²". Use `uncertainty_downweight.png` as a small
>    inset (σ² heat-map on a surgical frame).
> 2. **Motion attribution / freeze gate** — a compact left-to-right strip:
>    input frame → frozen ViT (base DINOv3-B/16) patch features → k-means
>    **semantic districts** (inset: `pipe_districts.png`) → RAFT flow votes,
>    one arrow per district (inset: `pipe_votes.png`) → tiny egomotion fit
>    `f_k ≈ u + v·ζ_k + d·r_k·ζ_k` → two statistics `q10` (minority) and
>    `dis3` (epipolar majority) → **freeze ⇔ q10 < 2.5 px ∨ dis3 > 0.5** →
>    gate on the pose solve: "still → copy pose, skip solve" / "moving →
>    track". Add a small arrow "re-pin frozen keyframes after every BA cycle".
> 3. **BA noise-floor fix** — small block on the bundle-adjustment path:
>    "corrected keyframe pose step (removes optimiser-induced drift)".
>
> Add a small legend mapping the three colours to *Co-SLAM base / DDS-SLAM /
> this thesis*. Keep every mathematical symbol exactly as written above (they
> must match the thesis equations). The 3D tissue render (`crcd_3dscene.png`)
> stays as the scene-representation visual.

**Assets for Prompt 1:** `coslam_architecture.svg` (+ `fig_coslam_architecture.png`
as raster reference), `bdds_overview.png` (previous overview — supersede, don't
copy), `crcd_3dscene.png`, `pipe_frame.png`, `pipe_districts.png`,
`pipe_votes.png`, `pipe_flow.png`, `uncertainty_downweight.png`,
`fig_dino_comparison.png` (only if a DINO-PCA inset is wanted).

---

## PROMPT 2 — DDS-SLAM deformation warp (`figures/deformation_warp.pdf`)

> Produce ONE horizontal figure explaining a canonical-space deformation warp,
> three zones left→right:
>
> **Left — "Live frame, time t":** use `warp_deformed.png` (a real endoscopic
> frame with a bent white grid showing tissue deformation; the variant
> `warp_deformed_arrows.png` additionally shows yellow displacement arrows —
> choose whichever reads better small). Mark one query point `x` on the tissue
> with a small circle and label "query point x at time t".
>
> **Middle — the warp network:** `x` and `t` each pass through a frequency
> positional encoding (`γ_freq(x)`, `γ_freq(t)`, drawn as small sinusoid
> stacks), concatenate into a small MLP block `D_φ`, which outputs `Δx`.
> A ⊕ node forms `x_canon = x + Δx`. Two callouts: **"Δx ≡ 0 at t = 0 —
> canonical space anchored to frame 0"** and **"no displacement target:
> supervised only through the rendering loss"**. Optional small inset:
> `warp_field_magnitude.png` (displacement-magnitude heat map) behind or
> beside `D_φ`.
>
> **Right — "Canonical configuration (frame 0)":** use `warp_canonical.png`
> (same scene, straight grid). Show `x_canon` landing at the corresponding
> tissue point (dashed arc from the ⊕ node), then a compact static-SDF stack:
> `x_canon → γ(·) → g_θ → s` labelled "static SDF, evaluated once, in
> canonical space".
>
> A single long annotation beneath: *"the warp absorbs all temporal variation;
> geometry is represented once, in canonical space."* Style must match the
> Co-SLAM/BDDS architecture figure (same node style, serif math). Export as
> `deformation_warp.pdf` + `.svg`.

**Assets for Prompt 2:** `warp_assets/warp_canonical.png`,
`warp_assets/warp_deformed.png`, `warp_assets/warp_deformed_arrows.png`,
`warp_assets/warp_field_magnitude.png` (all generated 2026-07-13 from CRCD
C1_001 frame 1 with a synthetic smooth displacement field, max |Δx| ≈ 42 px —
illustrative, not a trained field; say so in the thesis caption only if asked).

---

Thesis caption for Prompt 2 (already in `thesis.tex`, do not change):
"The DDS-SLAM deformation warp. A frequency-encoded network D_φ predicts, for a
query point x at time t, a displacement Δx mapping it to its location in the
canonical (frame-0) configuration, where the static SDF is evaluated. Anchored
to zero at t = 0, the warp absorbs all temporal variation so the geometry is
represented once, in canonical space."
