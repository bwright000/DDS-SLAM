# HANDOVER — DDS-SLAM → Gaussian-Splatting Migration (2026-06-22)

> **For the adjacent agent picking up the GS migration.** Self-contained. Read this top-to-bottom once,
> then the three reference docs in §9. Your first concrete job is in §5 (Phase-0 CRCD adapter).

---

## 1. Mission (one paragraph)
Migrate DDS-SLAM's backend from a neural-SDF (Co-SLAM) renderer to **3D Gaussian Splatting**, on the
verified base **EndoGSLAM** (already cloned at the workspace root `EndoGSLAM/`). The end goal is a surgical
GS-SLAM that **renders moving instruments ("tools") sharply by tracking them as objects** — a capability a
canonical SDF *structurally cannot have*. Build it **staged and metric-gated**: Phase-0 stand up the
unmodified base on our data → Phase-1 port our proven uncertainty/flow wins → an empirical tool-entry probe →
Phase-2/3 the net-new deformation + object-level tool work. **Your immediate task is Phase-0 only** (§5);
everything past it is scoped in §6 but gated on Phase-0 passing.

## 2. WHY GS (the reasoning that led here — don't re-litigate it)
The neural-SDF stores **one value per world location and has no time axis**. Consequences we diagnosed and
verified in code:
- A **moving tool** is forced into the static map at every position it swept → the map stores the
  **time-average** → a **ghost/blur**. (Trace: the tool occupies world spot L at frame x, background at L in
  other frames → contradictory `sdf_loss` targets → the map's optimal compromise *is* the blur.)
- A **new tool entering** mid-sequence, an **occlusion reveal**, or a **cut** are **topology changes** —
  *adding/removing* content. A deformation field can only **warp existing** geometry; it **cannot add/remove**.
  So the SDF *structurally* cannot represent surgery's defining events.
- **Gaussian Splatting can**: densification **spawns** Gaussians for new content and **prunes** departed ones,
  and a moving rigid object = a Gaussian group with its own **SE(3)** — composited, not smeared.
- Separately, we proved the SDF deformation field is **dead/redundant on our mild data** and can only ever
  *do-no-harm* on render; it is not the path to a render win. The render win lives in **topology/tool
  handling**, which is GS-native. (Full SDF analysis: `DDS_MAPPING_MECHANISM_20260622.md`.)

**The thesis contribution = render the tool as a tracked object** (per-object SE(3) + enter/exit lifetime),
with **rigid-vs-deformable routing at the tool–tissue contact boundary**. The verification confirmed this is
the **open niche** (unclaimed by NRGS-SLAM and every 2025–26 surgical GS paper — they all stop at masking).

## 3. Verified state (the GO-WITH-CONDITIONS verdict — `GS_MIGRATION_VERIFICATION_20260622.md`)
- **EndoGSLAM is real and viable**: MICCAI-2024, code released (`Loping151/EndoGSLAM`), true online
  unknown-pose SLAM (`use_gt_poses=False`), photometric+depth render-and-compare tracking (≈ our
  `tracking_render`), **full densification (clone/split/prune)**. **No base change.**
- **BLOCKER B1 (the Phase-0 gate):** EndoGSLAM's depth term is **L1, scale-sensitive**, and Gaussian
  scale-init is metric-sensitive. Our **MoGe-2 depth is UP-TO-SCALE** → must apply **per-snippet scale
  calibration** (α) so the depth loss isn't dominated by scale error. Same fix we used in the NeRF world.
- **BLOCKER B2 (later):** online **deformable** GS is **RESEARCH, not a port** — Deform3DGS is
  offline/batch/known-pose; no released online-deformable surgical GS exists. Budget Phase-2 as a 4–8 wk
  research milestone with a kill-criterion. **Phase-1 uncertainty (replicate −23% ATE in GS) is the
  guaranteed shippable result** so a Phase-2 stall isn't fatal.
- **LOAD-BEARING UNKNOWN:** the "GS beats SDF on tool ghosting" render win is **mechanically sound but
  UNPROVEN on surgical tool data**. → an **early tool-entry probe** is an explicit GO/NO-GO gate before
  Phase-3. If it fails, descope to **GS + uncertainty + honest-negative-on-topology** (still publishable).
- **NOVELTY RE-SCOPE:** NRGS-SLAM (arXiv 2602.17182, Feb 2026, **no code**) already does per-Gaussian
  rigid/deform *routing* via a Bayesian gate. **Do NOT headline "learned routing."** Headline **RBF-basis
  deformation + object-level tool SE(3) lifetimes**. Read NRGS in full before any novelty claim.
- **CAUTIONS:** EndoGSLAM has **no LICENSE file** (`license=null`; contact authors or build on the licensed
  `diff-gaussian-rasterization-w-depth` submodule). sm_80 rasterizer rebuild is Colab-only (local GTX 970 =
  sm_52, can't compile).

## 4. What EndoGSLAM HAS vs is MISSING (inventoried 2026-06-22)
**Workspace root** `C:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\` contains the whole GS ecosystem:
`EndoGSLAM/` (base), `SemGauss-SLAM/` (rasterizer w/ sem channels), `WildGS-SLAM/` (σ² uncertainty),
`SGS-SLAM/`, plus our `DDS-SLAM/` (the NeRF fork, source of contributions to port).

**✅ HAS (a complete, runnable online surgical GS-SLAM):**
- `EndoGSLAM/scripts/main.py:rgbd_slam` — the SLAM loop (tracking/mapping/keyframe); `initialize_first_timestep`, `get_loss`, `add_new_gaussians`.
- `EndoGSLAM/utils/slam_external.py` — **densify (clone/split/prune)** + `prune_gaussians` (the spawn/prune).
- `EndoGSLAM/datasets/gradslam_datasets/basedataset.py` — **generic** base: loads color/depth/intrinsics/poses, handles **PNG *and* `.npy`** depth (`np.load`, ~line 294), divides by `png_depth_scale`, optional `load_embeddings` hook (repurpose for DINO/seg). Working loader patterns: `c3vd.py`, `endoslam.py`.
- `EndoGSLAM/scripts/calc_metrics.py`, `utils/eval_helpers.py` — PSNR/SSIM + pose eval. `viz_scripts/` — recon + video.
- Rasterizer = pip `diff-gaussian-rasterization-w-depth` (Luiten; RGB+depth) — fine for Phase 0–1.

**❌ MISSING (what you build/port):**
1. **CRCD + SemSup dataset loaders** (only C3VD/EndoSLAM) — *Phase-0, your first task*.
2. **MoGe-2 up-to-scale depth handling** + per-snippet α calibration — *Phase-0, the B1 gate*.
3. **Seg-mask loading** (no loader reads instrument masks) — needed for the tool; use the `load_embeddings` hook pattern.
4. **Port our contributions:** Sim3-ATE eval (`DDS-SLAM/Addons/eval/sim3_ate.py`), 6-panel video (`Addons/viz/generate_video.py`), flow gate (`Addons/motion/flow_track.py`), DINO baker (`Addons/dino/`), σ² uncertainty (lift `WildGS-SLAM/.../dyn_uncertainty/`).
5. **Deformation** (Deform3DGS RBF basis — EndoGSLAM is rigid) — *Phase-2, research*.
6. **Object-level tool rendering** (segment + SE(3) + lifetime + composite) — *the contribution*.
7. **SemGauss rasterizer** swap (RGB+depth+sem+deform channels) for per-Gaussian routing — *Phase-3*.

## 5. YOUR FIRST TASK — Phase-0 CRCD adapter (the de-risked first step)
**Goal:** run *unmodified* EndoGSLAM online on **CRCD c1_001 (raw-left, 360-GT)** with **MoGe-2 depth**, and
measure it against the NeRF-SDF canon — *before any net-new code*. This isolates B1 (depth scale) + base
viability.

**Steps:**
1. **CRCD loader** — copy `EndoGSLAM/datasets/gradslam_datasets/c3vd.py` → `crcd.py`. Point `get_filepaths()` at
   the staged CRCD-Published raw-left frames + MoGe-2 `.npy` depth; `load_poses()` at the **360-row** GT
   (`groundtruth.txt`). Set intrinsics (CRCD calib) + `png_depth_scale`. (SemSup loader next, same pattern.)
2. **MoGe-2 scale (B1)** — MoGe depth is up-to-scale. Empirically fit a per-snippet scale α (start: align median
   depth to the scene metric; verify the L1 depth loss isn't scale-dominated). The depth `.npy` path already
   works in `basedataset` (`np.load`).
3. **Repoint eval** — replace EndoGSLAM's pose eval with our **`sim3_ate.py`** (scale-aware; **never** rigid
   Horn `output.txt` — it inverts A/Bs on up-to-scale depth). Run our `eval_rendering.py` + `generate_video.py`
   on the GS renders.
4. **Build the rasterizer for sm_80 on Colab** (the Luiten dep; ~15 min) — local can't (sm_52).
5. **Run** EndoGSLAM online (unknown pose) end-to-end on CRCD c1_001.

**PASS criterion:** converges, non-degenerate trajectory + renders, **Sim3 ATE in the same order of magnitude
as DDS-SLAM-Base (~3.15 mm)** and render PSNR comparable. **Pass → Phase 1. Fail → stop, reconsider the base
(do NOT proceed to net-new).** Report {Sim3 ATE_mean/max, seed-std, est/GT path-ratio, |Pearson|dom, PSNR,
SSIM, LPIPS, depth-L1} + the 6-panel video. n=3 seeds (seed coin-flip is real).

## 6. The phased plan past Phase-0 (scope; do NOT start until Phase-0 passes)
- **Phase 1 — port the proven wins.** Lift WildGS image-space DINO→σ² (`dyn_uncertainty/`) → mapping NLL +
  **Inc-2** tracking down-weight; port the **flow camera-vs-scene gate**. **Target: replicate the −23% ATE win
  in GS form** = the guaranteed result. All flag-gated default-off vs the Phase-0 base.
- **Tool-entry PROBE (gate).** Cheap A/B on CRCD/STIR-with-tool-masks: does GS densify render a mid-sequence
  tool/reveal **sharply** vs the SDF ghost? **This validates the whole migration's premise.** Fail → descope.
- **Phase 2 — deformation (research).** Implement a per-Gaussian **RBF/FDM motion basis** *inside* the
  consolidated rasterizer + EndoGSLAM's online loop (Deform3DGS is reference only — **do NOT ship its batch
  loop**). Re-root our deformation teacher (Δx* → nearest-Gaussian trajectory; replay/gauge-fix carry over).
- **Phase 3 — the CONTRIBUTION: object-level tool rendering + routing.** Segment the tool → track a per-frame
  rigid **SE(3)** → store the tool in its own canonical Gaussian group → **composite** in front of tissue →
  **enter/exit lifetime** (spawn on entry, prune on exit). **Routing** = rigid-vs-deformable at the tool–tissue
  **contact boundary**. Swap to the **SemGauss rasterizer** (sem+deform channels). Head-to-head vs NeRF-SDF-DDS.

**Tool–tissue interaction taxonomy (the honest scope of Phase 3):** *free-space tool* → clean decomposition
works; *contact, tissue visible* → routing handles it (the contribution); *tissue occluded behind the tool* →
**fundamental limit**, inferred-not-reconstructed (state it, don't pretend to solve); *cutting* → needs
spawn/prune, and the new surface is only seen *after* it opens. Model **observed motion, not contact physics**.

## 7. Methodology (carries over UNCHANGED from the NeRF world)
- **Metrics are the only arbiter.** Tracking = **Sim3 ATE** (`sim3_ate.py`) + est/GT path-ratio + dominant-axis
  |Pearson| (CRCD is sub-SNR — always quote these, never bare ATE, never rigid `output.txt`). Render =
  **PSNR/SSIM/LPIPS + depth-L1**. **n=3 seeds** (a win must clear base seed-std).
- **Base-first:** the unmodified EndoGSLAM @ flags-off is the **new eternal reference** (replaces
  `DDS-SLAM-Base`). Every change runs FROM it in isolation, flag-gated default-off. New regression gate vs it.
- **Every result ships TWO diagnostic sets by default:** the metric table **and** the 6-panel inline video.
- **Datasets:** **CRCD c1_001 raw-left (Published 360-GT)** = render + ATE (the only one with both). **SemSup
  trail3** = render-only (its pose GT is fictional/identity — **never** headline SemSup ATE). STIR (if ported)
  = the deformation/tool proof.
- **Env:** Colab A100, **sm_80** rasterizer build (no local CUDA). Stage CRCD-Published (360, NOT the stale
  271-row local copy). CRCD = **raw-left** for all working experiments.

## 8. Do-NOT / gotchas
- 🚨 **Never headline rigid `output.txt` ATE** — inverts A/Bs on up-to-scale MoGe depth. Sim3 only.
- 🚨 **Stale CRCD GT:** `DDS-SLAM/data/CRCD/.../groundtruth.txt` is a 271-row subset. Use **CRCD-Published 360**.
- 🚨 **Don't headline "learned routing"** as novelty (NRGS has it). Lead with RBF-basis + object-tool SE(3) lifetimes.
- 🚨 **Don't ship Deform3DGS's batch loop/rasterizer** — keep the RBF *insight* only.
- 🚨 **Don't revive the deformation-as-render-win idea** — the SDF field analysis proved it's do-no-harm at best on mild data; GS's win is **topology/tools**, not tissue stretch.
- Confirm the **EndoGSLAM license** before any public release.

## 9. Reference docs (read in this order) + key paths
1. `Addons/docs/GS_MIGRATION_VERIFICATION_20260622.md` — the GO/NO-GO verdict + blockers + first step.
2. `Addons/docs/GS_MIGRATION_PLAN_20260621.md` — the full stack + component port map + phases + risks.
3. `Addons/docs/DDS_MAPPING_MECHANISM_20260622.md` — *why* the SDF fails (the reasoning behind the migration).
- Memory canon: `project_gs_migration_plan_20260621` (has the verification verdict appended),
  `project_combine_routing_model_20260619`, `CLAUDE.local.md` (workspace paths/run cycle).
- Our contributions to port live in `DDS-SLAM/Addons/{eval,viz,motion,dino,depth}/`.
- The GS ecosystem repos are workspace-root siblings: `EndoGSLAM/`, `SemGauss-SLAM/`, `WildGS-SLAM/`, `SGS-SLAM/`.

---
**TL;DR for the adjacent agent:** Build the **CRCD loader + MoGe-2 scale calibration + Sim3-ATE/video repoint**
so *unmodified* EndoGSLAM runs on CRCD c1_001 and we can measure GS-vs-NeRF (§5). Pass that gate before any
net-new. The thesis is **object-level tool rendering** (§6 Phase 3); everything is metric-first, base-first,
n=3 (§7); mind the depth-scale and GT-row gotchas (§8).
