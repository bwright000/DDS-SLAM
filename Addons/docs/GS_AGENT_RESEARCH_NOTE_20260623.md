# Research Note → GS Agent (2026-06-23)

**From:** the NeRF/DDS-SLAM side. **Purpose:** what we ran and learnt that should change GS decisions. Addendum to `GS_MIGRATION_HANDOVER_20260622.md`. All on CRCD `c1_001` (raw-left, sub-SNR), single-seed unless noted.

---

## TL;DR — the four findings that matter for GS

1. **To "chase" deformation you must touch GEOMETRY, not appearance.**
   We built a "up-weight the loss on moving/deforming regions" lever and it was a **total no-op** — flat PSNR, byte-identical video — *even on frames where the motion signal fired on 55% of the frame*. Root cause: we up-weighted the **RGB(5)+depth(0.1) losses ≈ 0.5% of the budget**, while the **SDF/geometry loss (weight 1000, ≈99%)** — the thing that actually moves the rendered surface — was a flat global mean we never touched. **GS analog:** a photometric/color-loss reweight will *not* model deformation. The lever that moves the surface is the **Gaussian positions / depth supervision**. Put any "chase the deforming region" weight on the *geometry* path. (We've since wired the geometry version, `sdfup` — result pending.)

2. **The deformation is REAL and large on CRCD — it's worth modeling, you're not chasing noise.**
   Raw per-pixel optical-flow (Sampson) residual vs the fitted camera motion: **P99 up to 106 px, max 118 px, 17/36 sampled frames with P99 > 3 px**, concentrated in the last third. Key reframe: **CRCD is sub-SNR for the CAMERA** (GT camera motion ~18 mm → tracking is hard) **but the SCENE/tool moves a LOT relative to the near-static camera.** So the deformation/moving-tool problem is genuine and measurable here — the deformation field + object-model are justified, not over-engineering.

3. **Two signals, two jobs: per-pixel flow for MOTION-MAGNITUDE, learnt DINO features for IDENTITY.**
   Our route already groups by **DINOv2 k-means** (`region_route` mode `'region'` — the TriGauge "what-kind": learnt, label-free, preferred over GT masks) and flags a region if its **median** Sampson > 3 px. Three refinements (GS attribution 06-23 + our diagnostics):
   - **Magnitude must be PER-PIXEL.** The region-MEDIAN missed localized motion — frame 28: 7.87% of pixels moved >3 px, yet the region logged `moving-frac = 0`. Drive deform-magnitude / which-Gaussians-to-update from the **per-pixel** residual (`flow_residual` / mode `'pixel'`), not a patch median.
   - **Flow is magnitude, NOT identity.** Even per-pixel, flow classifies tool-vs-tissue at only **AUC 0.59–0.67** (vs the GT tool). So get identity from the **learnt DINO grouping**, magnitude from per-pixel flow. DINO groups tissue too → no GT mask needed; masks = eval ground-truth only.
   - **Two traps in the current region mode:** 14 px patches **bleed / miss partial deformation**, and the **fundamental fit is degenerate on forward/zoom** (flags zoom as deformation). The `'pixel'` homography mode fixes both but drops DINO grouping → **the upgrade is DINO grouping + per-pixel homography residual combined.**

4. **σ² → pose, flow → deformation. Don't cross them.**
   Learned aleatoric σ² (WildGS-style) is excellent for the **tracking pose down-weight** (we get −23% ATE / Pearson 0.81→0.97, it kills the deformation-moment jitter). **But σ² is appearance-blind to motion** — it flags specular/textureless/edges, *not* the moving tool. So: use σ² for **pose robustness / which pixels to trust for tracking**; use **optical flow** for **what's deforming / rigid-vs-deform routing**. Expecting σ² to route deformation is a known dead end (it becomes a contrast detector).

5. **Pick the snippet by what you develop.** **C1_001 = deformation** — broad tissue deformation over *half the scene in the last third*, and **short (359 frames → fast)**. It scores WEAK on *tool*-attribution (REAL 1.92 vs NULL 1.73) only because the tool isn't separable from that broad deformation — it's a strong, cheap **deformation** testbed. **C2_001 / C3_001 = clean tool-as-object** (GO, the spatial null collapses: 2.43/1.52 and 1.96/1.16) — use for the rigid-tool / articulation work, but they're **2–4× longer** (729 / 1526 frames). So: **deformation field → c1_001 (fast); tool-as-object → C2/C3 (clean but slow).**

---

## What we ran (NeRF side)

| Cell | What | Result |
|---|---|---|
| `improved` | `curmap100` (cur_frame_iters 0→100) + bigger decoder | **+4.12 PSNR** (21.98→26.10). The per-frame fast map re-fit is the workhorse. |
| `geo` (σ²) | Inc-1 NLL σ² head + Inc-2 tracking down-weight | **−23% ATE**, the proven tracking/jitter win |
| `geo_flow_agree_baf` | + optical-flow camera-vs-scene gate + freeze_ba | the tracking/path-length winner |
| `upweight` | flow → **RGB+depth** map up-weight | **NO-OP** (wrong loss) |
| `sdfup` | flow → **SDF/geometry** map up-weight | the right-knob fix — **pending** |
| `best` | improved + tracking stack | **running** |

**Critical mechanism:** DDS-SLAM's deformation field is **dead** (Δx→0; starved — no displacement supervision). The "decent deformation renders" it produces come entirely from the **per-frame map re-fitting to the current deformed depth** (the SDF↔depth chase), *not* from the field. In a static representation, **the per-frame chase IS how deformation gets rendered.**

---

## What it means for GS (actionable)

- **The static-map "chase" has a hard ceiling we hit empirically.** You cannot reweight a static representation to render a moving rigid tool or sharp deformation — we closed that door (the no-op above). This is positive evidence *for* the GS plan: the **Deform3DGS RBF field** (true 4D) and the **NRGS-style object-model** (tool = rigid SE(3) + articulation) are the right escalation, not a loss tweak.
- **Geometry-first principle.** Whenever a representation supervises geometry strongly (SDF-from-depth, or Gaussian-positions-from-depth), the appearance loss is a *weak* lever on the scene. To change what's rendered — especially deformation — move the geometry. Carry this into how you wire σ²/flow weights in GS: prefer the depth/position path.
- **Routing:** per-pixel flow residual → deform-vs-rigid; σ² → pose trust. Two signals, two jobs.
- **Data:** use **rectified** CRCD + **metric-anchored** depth (stereo SGBM every 120 frames, ramped, `sc_factor=1`) so hardcoded geometry constants apply. The anchor can fail silently (we had a snippet come out 12.5× off); our robust version does **cross-anchor consensus + a loud fail gate** — reuse it.

---

## Tools you can reuse directly

- `Addons/motion/diag_flow_residual.py` — prints the **raw per-pixel Sampson motion residual** (P50/P90/P99/max in px) per frame for any sequence. **Run it on your GS dataset first** to see where and how much deformation there is before designing the deform routing (we found P99 up to 106 px on c1_001).
- `Addons/motion/flow_track.py` — `flow_residual` (RAFT + RANSAC-fundamental + Sampson, **per-pixel** — use this for magnitude); `region_route` mode `'region'` = the learnt **DINO grouping** (identity), mode `'pixel'` = full-res **homography** residual (fixes the forward/zoom false-positive + the region-median coarseness). **Mask a frame-border band before routing** — RAFT + the fit blow up at the image edge (the attribution panels light up the borders); use the `edge_mask` option or routing chases the edges, not the tissue/tool.
- `Addons/depth/stereo120_metric_anchor.py` — metric depth anchoring with the robust per-anchor reliability + cross-anchor consensus + loud-fail gate.
