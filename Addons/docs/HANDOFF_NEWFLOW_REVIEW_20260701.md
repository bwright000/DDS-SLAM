# HANDOFF — new flow-supervisor (Mode A `solve_pnp` + Mode B `depth_pool`) for adjacent-agent review

**Branch:** `diagnosis-live` (remote `bwright000/DDS-SLAM`). **Repo root under review:** `DDS-SLAM/`.
**Date:** 2026-07-01. **Status:** built, CPU-smoke-passing, committed + pushed, NOT yet run on GPU.
**Governing rule:** METRICS are the only arbiter (render PSNR/SSIM/LPIPS + tracking Sim3-ATE). `DDS-SLAM-Base/`
is the pristine reference; every addition is flag-gated default-off so `flow_track.enable=false` == pristine.

---

## 0. TL;DR — what the reviewer should check (ranked)
1. **Correctness of the PnP init-only override math + OpenGL/OpenCV convention** (`ddsslam.py` `solve_pnp` branch). A sign/convention error here silently injects wrong poses. The prior flow bug was exactly this (see §2).
2. **Is `solve_pnp` a re-weighter in disguise, or a real estimator?** Confirm the `cur_c2w` override actually reaches the pose optimiser init and isn't overwritten before use.
3. **The honest null-result risk (§8.1):** init-only may not clamp the jitter if the SDF tracker over-travels regardless. Sanity-check the reasoning; suggest the cheapest disproof.
4. **v1 tool-exclusion is a NO-OP on CRCD (§8.2)** — verify, and judge whether RANSAC-only is enough for a first E3 read.
5. **`flow_diag.py` PASS criteria** — are D1 (activation timing) / D2 (path-ratio) the right, non-gameable checks on sub-SNR data?
6. **Base byte-identity** — confirm nothing runs when `flow_track.enable=false` / `mode` unset.

---

## 1. Context, goal, metric conventions
- **Project:** DDS-SLAM = a Co-SLAM fork (neural-SDF SLAM) for deformable **endoscopic** surgery. We track the
  camera (endoscope) frame-to-frame while tissue deforms and a tool moves independently.
- **The tracking problem being fixed:** on CRCD the tracker **over-travels / jitters** — even when the camera
  holds still it wobbles, and it exaggerates motion (path-ratio ~1.9–2.8 vs GT). A tool moving in-frame used to
  hijack the old camera/scene gate.
- **Data:** monocular RGB; **MoGe-2 depth = up-to-scale, per-frame, noisy**, stereo-anchored per snippet
  (rect-bench bakes ~metric sc=1, re-anchored every 120 frames). Poses + ray dirs are **OpenGL** convention.
- **Metric conventions (critical, get these right when judging):**
  - CRCD tracking = `Addons/eval/sim3_ate.py` (**Sim3** = scale-corrected). ALWAYS quote **path-ratio +
    ALIGNED Pearson**, never the pipeline's rigid `output.txt` (scale-confounded, inverts A/Bs).
  - CRCD GT is **sub-SNR** (E3_005 GT ≈ 0.05 mm/frame, below the tracker floor) → ATE alone is misleading;
    path-ratio + Pearson + `flow_diag` are the real signals.
  - Eval against **CRCD-Published 360-row GT** (`F:/Datasets/CRCD-Published/<ep>/snippet_NNN/groundtruth.txt`),
    never the stale repo copy. E3_005 → `E_3/snippet_005/`, 265 rows.

---

## 2. Lineage — what we tried, what failed, WHY (so the new design makes sense)
| Approach | Aggregation | Depth? | Fate / why |
|---|---|---|---|
| `flow_residual` (per-pixel Sampson → per-pixel weight) | none | no (2D F) | **failed — per-pixel residual too noisy to threshold** (RAFT+depth noise). This is why we moved to DINO regions. |
| `agreement_gate` (F-matrix + per-DINO-region Sampson + **binary gate**) | ~12 regions | no (2D F) | **failed** — (a) F **hijacked by the moving tool** (camera-vs-object ambiguity), (b) self-defeating `cam_mag>2 AND disagree<=0.2` **froze the camera ~every frame** (staircase jitter + masked σ²), (c) hardcoded 3px deadband. |
| **L0** `residual=rigid` (depth-predicted rigid-flow residual → per-DINO-region soft weight) | ~12 regions | yes | **built + run.** Two bugs found + fixed: (1) OpenGL/OpenCV convention made the residual ~0 everywhere (no-op) — commit `186b307`; (2) `sim3_ate` Pearson was computed on the UN-aligned est (rotation-confounded) — commit `f786551`. After fixes, L0 is a valid **re-weighter** but does **NOT** fix scale (it moved Sim3 s the WRONG way 0.583→0.384) and its per-frame residual is a ~24px **noise floor** (true parallax <5px on near-static E3). Verdict: a re-weighter can't cure the disease. |
| **Mode B** `depth_pool` (NEW) | ~12 regions | yes | pose-free; the user's "scale everything to a common plane" idea. |
| **Mode A** `solve_pnp` (NEW) | **ALL pixels → 1 SE3** | yes | robust global solve; the first thing that **estimates the camera motion** rather than re-weighting. |

**The lesson thread:** "don't decide per-pixel; aggregate." DINO regions were step 1 (average over ~12 regions).
Mode A takes it furthest (a global fit over thousands of pixels + RANSAC), and keeps DINO regions for the
**trust weight**. So A is the *least* per-pixel thing here, not a regression to the old failure.

**Two reusable GOTCHAS burned in (a reviewer should assume these bite any new geometric code):**
- **OpenGL convention:** `datasets/utils.py:get_camera_rays` default `type='OpenGL'` (z=−1, y-flipped) and
  `dataset.load_poses` negates the y,z axes. Any back-projection/reprojection must convert GL→CV
  (`c2w @ diag(1,-1,-1,1)`) or every point lands behind the camera. This silently no-op'd L0 once.
- **sim3_ate Pearson (now fixed):** must correlate the **Sim3-aligned** est vs GT, not the raw est.

---

## 3. What's built (commits on `diagnosis-live`)
| Commit | Content |
|---|---|
| `d0399c5` | `flow_track.py`: `depth_pooled_weight` (B) + `rigid_solve_pnp` (A). Pure additions. |
| `6517154` | `Addons/regression/test_newflow_smoke.py` — CPU smoke (mocks RAFT). |
| `e0f5b42` | `ddsslam.py`: `_mode` dispatch (solve_pnp/depth_pool) + `_flowlog`; `datasets/dataset.py` seg-load gate. |
| `417a74c` | configs `crcd_abl_{dpool,pnp}_rect.yaml`; `Addons/eval/flow_diag.py`; rect-bench ARM_TMPL += dpool,pnp; `Addons/colab/e3_newflow_ablation_20260701.sh`. |
(Earlier relevant: `186b307` OpenGL fix, `f786551` sim3_ate Pearson fix, `5bdda7b` the abl_base/l0 configs.)

**Smokes (run `python Addons/regression/test_newflow_smoke.py` from repo root, no GPU):** A recovers a known
SE3 exactly (t_err~0, R_err~0) and with a tool blob + tool_mask still recovers the camera; B collapses 0.4 m
and 0.8 m regions to one consensus (w~1, distance-invariant) and flags an independent mover (w~w_min). PASS.

---

## 4. Mode A `solve_pnp` — algorithm, wiring, what to scrutinise
**Function:** `Addons/motion/flow_track.py::rigid_solve_pnp(ref_bgr, cur_bgr, ref_depth, fx,fy,cx,cy, model, tf, device, tool_mask, flow_advance_px, reproj_px, min_inliers)`
1. RAFT flow ref→cur → matches `p ↔ p'=p+flow`. If `median|flow| < flow_advance_px` → **return None** (below the
   noise floor; caller falls back to const-velocity).
2. Back-project **ref only**: `X_r = D_r · K⁻¹·p̃` (OpenCV dirs, z=+1). Correspondences = `(X_r [3D], p' [2D])`.
   **Hard-exclude `tool_mask` pixels** before solving.
3. `cv2.solvePnPRansac(objectPoints=X_r, imagePoints=p', K, reprojErr=reproj_px)` → `rvec,tvec,inliers`.
   Return None if fail / `<min_inliers`.
4. Return `(T_rel_cv [4×4] OpenCV ref-cam→cur-cam, resid_px [H,W] per-pixel reprojection residual, info)`.
   **Up-to-scale** (ref depth is relative) → NOT a metric anchor.

**Wiring:** `ddsslam.py` `tracking_render`, the `if _mode == 'solve_pnp':` branch (search "MODE A"). On a successful
solve: **INIT-ONLY** override
`cur_c2w = c2w_ref_gl @ GL2CV @ inv(T_rel) @ GL2CV` (`GL2CV=diag(1,-1,-1,1)`), then
`region_soft_weight(resid, dino_grid(cur_bgr))` → `track_w_map`. `cur_c2w` then feeds `get_pose_param_optim`
(the SDF tracker init); the tracker refines freely and `tracking.best` can reject it. On None → no override
(const-velocity) + no weight. σ² still multiplies `track_w_map` at `model/scene_rep.py:597-600`.

**⚠️ REVIEW HOTSPOTS:**
- The convention chain in the override: `T_rel` maps ref-cam(CV)→cur-cam(CV), so `c2w_cur_cv = c2w_ref_cv @
  inv(T_rel)` and both wraps by `GL2CV`. Verify by hand — a wrong `inv`/side/GL2CV silently injects a bad init.
  (The smoke checks `rigid_solve_pnp` recovers SE3 but does NOT check the ddsslam override composition.)
- Confirm `cur_c2w` isn't recomputed/overwritten between the override and `get_pose_param_optim(cur_c2w[None,...])`.
- `ref_stride=8` (not 1): the reference is 8 frames back, so `T_rel` is an 8-frame motion; the override sets the
  ABSOLUTE cur pose (fine), but the review recommended an **accumulated-flow ref-advance** instead (deferred, §8.3).

---

## 5. Mode B `depth_pool` — algorithm, wiring
**Function:** `flow_track.py::depth_pooled_weight(...)`. Pool flow AND depth into DINO k-means regions;
`v_k = median_flow_k · median_depth_k` brings every region to a common plane (translation parallax `f·t/Z · Z =
f·t` = distance-invariant); weight = soft MAD-thresholded **deviation from the robust median consensus** (floor =
`w_floor_px · median_depth`). Pose-free (no rigid-flow prediction → sidesteps L0's prior-noise floor); median
consensus is tool-robust. **Caveat:** exact only for translation — camera **rotation** flow is depth-independent,
so `×depth` mis-scales it (contaminates rotation-heavy frames). Wiring: `if _mode == 'depth_pool':` → `track_w_map`.

---

## 6. Diagnostics — `Addons/eval/flow_diag.py` (auto per run, the headline judge)
Inputs: `est_c2w_data.txt` + GT (+ optional `trust_log.csv`). Outputs `flow_diag.json` + `.png`. Two checks
(Sim3-aligned, rank-based → robust on sub-SNR):
- **D1 CAMERA-ACTIVATION TIMING:** Spearman ρ(est per-frame step, GT per-frame step) + moving/still ratio.
  **PASS: ρ ≥ 0.5 AND ratio ≥ 2.0.** ("moves when GT moves, still when GT still.")
- **D2 OVER-TRAVEL:** path-ratio (est·s / GT). **PASS: ∈ [0.7, 1.4].** (est-still-jitter mm reported as info.)
**E3_005 baselines to beat (all currently FAIL):** abl_base ρ0.24/pr2.84 ; l0 ρ0.32/1.90 ; l0sig ρ0.30/1.42.
Per-frame `trust_log.csv` also logs (solve_pnp) `t_norm, rot_deg, inlier_frac, reproj_med` and (depth_pool) the
consensus stats. **Watch these FIRST — not ATE/PSNR.**

---

## 7. Key design verdicts (internal-review workflow, folded in)
1. **Metric scale anchor is DEAD on monocular depth** — forward(z) camera motion ≈ a global depth-scale error;
   MoGe z-noise (14–35 mm) swamps the ~0.05 mm GT signal; L0 empirically moved Sim3 s the WRONG way. A true
   metric anchor is **stereo-only**. BUT we DON'T need it — scores are Sim3 scale-corrected, so the disease is
   **RELATIVE jitter** and **relative (up-to-scale) depth (which MoGe gives) is the right tool** (need the
   ratios, not just ordinal further/closer). ∴ `solve_pnp` is deliberately up-to-scale + **init-only**.
2. **2D-3D PnP, not 3D-3D Procrustes** — Procrustes uses `D_cur` too (double depth-noise, corrupts translation
   1:1); PnP uses ref depth only and reads forward motion from radial looming (60–270× better on translation in
   the review's Monte-Carlo).
3. **Init-only, NO persistent prior** — a soft SE3 prior is a wrong-attractor with no self-correction; init is
   transient and `best`-loss can reject it.
4. **Tool exclusion is mandatory but v1 can't do it on CRCD** — see §8.2.

---

## 8. Known caveats / risks (the honest list — please scrutinise)
### 8.1 Null-result risk (the strategic one)
Init-only PnP may **not** move the metric: the disease may be the SDF tracker's photometric/geometric gradient
**over-travelling every iteration regardless of the init** (L0 the re-weighter came back flat: base ATE 3.65 → l0
3.96). If `flow_diag` D2 path-ratio stays ~1.9 on E3, init-only hasn't touched the disease → the fix is in the
SDF **refinement**, not the seed (next lever: an uncertainty-scaled R+lateral prior on high-confidence frames, or
stereo depth). **This is a legitimate possible outcome; the build is the cheapest honest way to learn it.**

### 8.2 v1 tool-exclusion is a NO-OP on CRCD
`solve_pnp` masks `batch['seg']==2`, but CRCD's `StereoMISDataset.semantic_paths` glob the **binary** `masks/`
(`datasets/dataset.py:187-189`), so `_attach_seg`'s tool(raw 3)→canonical 2 remap never fires → `seg` is all-zeros
→ `tool_mask` empty → **RANSAC-only tool rejection**. On tool-dominated E3 the tool could still hijack. **v2 fix
(deferred):** add `data.seg_label_subdir` and set it to `semantic_class` (the 4-class map), decoupling the tool
label from the binary edge-field masks (must confirm the rect-bench stages `semantic_class/`). Watch the trust
map / D1 ρ to detect a hijack.

### 8.3 Other deferred hardening (from the review, not in v1)
- **Accumulated-flow ref-advance** (advance the reference only when accumulated flow clears the noise floor, then
  interpolate) — v1 uses fixed `ref_stride=8` + a per-frame `flow_advance_px` skip.
- **Photometric-not-worse gate** (adopt the PnP init only if it beats const-velocity on the SDF's own loss_iter0)
  — the most robust gate; v1 gates only on solve-success + flow floor + `min_inliers`.
- **Frame-to-frame depth-scale consistency + weak-forward-signal** are the real remaining physical risks (not
  metric-vs-relative); `flow_diag` per-frame `t_norm` + the scale panels are the watch.

### 8.4 Parity
All new code is behind `flow_track.enable` + `_mode`; `_mode` unset → the legacy gate/residual dispatch is
unchanged; `flow_track.enable=false` → the block is skipped entirely (base byte-identical). Please confirm.

---

## 9. How to run + judge
```bash
# fresh Colab (Drive mounted, repo cloned to /content/DDS-SLAM):
cd /content/DDS-SLAM && git pull && bash Addons/colab/e3_newflow_ablation_20260701.sh
#   arms = abl_base l0 dpool pnp  on E3_005 (n=1 SCREEN). Outputs -> MyDrive/Outputs/rect_bestbase_newflow_20260701/
```
- **Read FIRST:** each cell's `flow_diag.json` (D1 ρ + D2 path-ratio). For `pnp`, `grep -a '\[solve_pnp\]' run.log`
  — is it *solving* (not all `const-velocity fallback`) and are `|t|`/`inl` sane.
- **Then:** `sim3_metrics.txt` (Sim3 ATE + path-ratio + aligned Pearson dom + mean-xyz), `render_eval.txt`,
  `depth_l1.txt`, `panels.mp4` (has the Trust Weight panel).
- **n=1 is a SCREEN.** The winner earns **n=3 + the full C1/C2/C3/G3 bench** before any claim (seed-std is the
  noise floor; a win must clear it). E3-only can mislead the other way too (a tool-frame win can cost elsewhere).

---

## 10. Open questions for the reviewer
1. Is init-only PnP the right first experiment, or should we go straight to the photometric-not-worse gate (§8.3)
   to avoid a likely null result?
2. Given the metric-scale-dead finding (§7.1), is there any monocular way to damp the *relative* over-travel that
   we've missed, short of stereo? (e.g., a flow-consistency regulariser in the SDF loss itself, since the disease
   may live there — §8.1.)
3. Is `depth_pool` (B, pose-free) worth keeping given its rotation caveat (§5), or does `pnp` (A) subsume it?
4. Is the `flow_diag` D1/D2 PASS bar defensible on sub-SNR data, or gameable (e.g., a frozen tracker trivially
   passes D2 path-ratio→0)? — note D2 currently requires ∈[0.7,1.4], so a frozen (pr→0) tracker FAILS, and D1
   requires ρ≥0.5 which a frozen tracker also fails; please confirm this can't be gamed.

## Appendix — key file:line + memory pointers
- `Addons/motion/flow_track.py`: `rigid_solve_pnp`, `depth_pooled_weight`, `region_soft_weight`,
  `rigid_flow_residual` (L0), `agreement_gate` (old F-gate).
- `ddsslam.py` `tracking_render`: the `_mode` dispatch (search "MODE A" / "MODE B"); `_flowlog`.
- `model/scene_rep.py:585-600` (σ² + track_ray_w multiply), `:627/:645` (rgb/depth loss), `:676` (sdf loss).
- `datasets/dataset.py:187-189` (semantic_paths=binary masks — the §8.2 no-op), `:218` (seg-load gate),
  `:161-162` (canonical seg remap tool→2).
- `Addons/eval/sim3_ate.py` (fixed aligned Pearson), `Addons/eval/flow_diag.py`.
- configs: `configs/CRCD/crcd_abl_{base,l0,l0aggr,l0sig,l0sigaggr,dpool,pnp}_rect.yaml`.
- MEMORY (in `~/.claude/.../memory/`): `project_flow_gate_inverts_tool_e3005_20260627.md` (the live canon for this
  work — read it first), `feedback_sim3_ate_misleading_subsnr.md` (the Pearson bug + sub-SNR rules),
  `project_inc1inc2_build_wildgs_20260615.md` (σ² canon, now caveated), `reference_disagreement_measure_research_20260627.md`.
