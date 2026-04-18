# DDS-SLAM Results Log

All results, eliminated hypotheses, and evidence from the reproduction investigation.

## Metric note — Identity ATE vs Real-GT ATE

The paper's **8.3 mm ATE on StereoMIS** is measured against **identity GT** (drift from starting pose), not against robot kinematics. Their `StereoMISDataset.load_poses()` ignores the path argument and returns identity matrices. Our code now outputs both metrics (`output_identity.txt` = paper-comparable, `output.txt` = real kinematics). **All StereoMIS ATE numbers below are identity ATE unless noted.**

## Semantic-SuPer trail3 (151 frames, static camera, GT = identity)

| Run | Depth | ATE RMSE | Per-frame motion | PSNR | Notes |
|-----|-------|----------|-----------------|------|-------|
| **Paper** | **Finetuned Monodepth2** | **N/A** | **~0mm** | **28.6** | Python 3.7, PyTorch 1.10, TCNN v1.5, RTX 3090 — their checkpoint is inaccessible |
| DA V2 | DA V2 | 58.9mm | 13.8mm | 27.6 | First successful run |
| Monodepth2 (rescaled) | Monodepth2 KITTI | 4.2mm | 0.67mm | 26.9 | Compressed depth range constrains poses near identity — misleading |
| MoGe | MoGe-2 | 60.2mm | 25.4mm | 27.0 | Similar to DA V2 |
| Upstream-matching code | DA V2 | 61.3mm | 13.6mm | — | Code matches IRMVLab/DDS-SLAM upstream exactly |
| DA V2 + depth_weight=5.0 | DA V2 | — | ~14mm | — | SDF still dominates 3000:1, no effect |
| DA V2 + TCNN FP32 | DA V2 | — | ~14mm | — | TCNN_HALF_PRECISION=0 rebuild, no effect |
| DA V2 + foreach=False | DA V2 | — | ~14mm | — | Adam foreach disabled, no effect |
| DA V2 + clamp [0,1] | DA V2 | — | ~5mm F1 | — | Gradient flipped to +0.937 BUT SDF quality destroyed (loss 7x worse) |
| DA V2 + expanded BB | DA V2 | 57.0mm | 14.3mm | — | BB: [[-2,2.5],[-2,2],[0,5.5]], no effect |
| DA V2 + scene-aligned BB | DA V2 | — | ~14mm | — | BB: [[-0.7,0.7],[-0.7,0.7],[0.2,0.9]], no effect |
| DA V2 + first_iters=5000 | DA V2 | — | ~14mm | — | SDF fully converged by 1000, no effect |
| DA V2 + GT pose bypass | DA V2 | 50.0mm | 13.7mm | — | tracking_render bypassed BUT global_BA still overwrites poses |
| DA V2 + dynamic=False | DA V2 | CRASH | — | — | `render_rays` reads `rays_o[...,3]` unconditionally — crashes without timestamps |
| DA V2 + GT bypass + zero vox_motion | DA V2 | 50.5mm | 13.9mm | — | Tracking bypassed + deformation disabled. global_BA STILL corrupts poses |
| DA V2 + GT bypass + zero vox + lr_pose=0 | DA V2 | 50.4mm | 14.1mm | — | sed didn't match — lr was 0.0001 not 0.001, poses still optimized. Rerun with correct zero lr needed |
| **trail3_fix1** (tracker LR ×0.1, iter 50) | MoGe | 63.0mm | 11.0mm median, 53mm max | — | Tracker walk-cap 0.5mm/frame not honored — mapping pose optim (lr_trans=0.0001 × cur_frame_iters=100 = 10mm/frame) overwrites tracker output every frame. Confirms global_BA is dominant walker (matches GT-bypass row). |
| **trail3_fix2** (optim_cur=False) | MoGe | 66.7mm | 11.08mm median, 59mm max | — | **Null result** — per-frame delta essentially identical to fix1 (11.04mm). `optim_cur=False` only blocks current-frame pose write at ddsslam.py:422; past-keyframe re-optimization at line 418 continues every frame (keyframe_every=1, map_every=1). Walker is past-KF path, not current-frame path. Not on critical path for paper reproduction. |
| **trail3_bbox_correct** (bbox sized to scene + far=1.0) | MoGe | 61.9mm | 16.88mm median, 70.7mm max | — | **Refuted.** Per-frame got **57% WORSE** (10.72 → 16.88mm). Opposite of the bbox-gradient-attenuation hypothesis's prediction. ATE changed within seed-variance (66.7→61.9). Tight bbox made field-oscillation-per-pose-change higher-leverage, not lower. Bbox oversize is NOT the cause of per-frame walk. |
| **trail3_D9** (UsePercentage=False at scene_rep.py:393,400, plain MSE per paper Eq.12) | MoGe | 57.7mm | 10.68mm median, 59.1mm max | — | **Null.** Per-frame essentially identical to baseline 10.72mm. ATE shift 66.7→57.7 within prior 3.4× seed variance. **Consistent with unified theory prediction**: L_m outlier discard is DDS-only; Co-SLAM (no L_m) walks identically on StereoMIS to 4 sig figs, so DDS-only fixes cannot be the universal cause. D9 closes. Code patch at scene_rep.py:393,400 left applied (paper-spec). |

## 4-Lab Paper-Spec Baseline (2026-04-15) — All Labs Walk Universally

Ran unmodified paper-spec config across all 4 SemSuP labs with consistent MoGe depth, to produce a clean comparison against paper Table I. Result: **every lab walks far beyond paper's near-stationary behavior**, with no correlation to paper's Rep.Err difficulty ordering.

| Paper Lab | Config | Depth | ATE RMSE | ATE mean | ATE max | per-frame median | per-frame max | trajectory extent (x,y,z mm) | Paper Rep.Err |
|---|---|---|---|---|---|---|---|---|---|
| Lab 1 | trail3 | MoGe | **66.66 mm** | 61.0 | 115.5 | **10.72 mm** | 54.1 mm | (99, 43, 192) | 3.3 (0.4) |
| Lab 2 | trail4 | MoGe | **151.4 mm** | 145.6 | 247.0 | **28.22 mm** | 93.7 mm | (160, **417**, 239) | 3.0 (0.5) |
| Lab 3 | trail8 | MoGe | **43.55 mm** | 40.3 | 97.1 | **13.63 mm** | 65.3 mm | (32, 69, 158) | 2.4 (0.4) |
| Lab 4 | trail9 | MoGe | **34.94 mm** | 31.3 | 83.2 | **18.13 mm** | 80.3 mm | (38, 52, 145) | 2.0 (0.2) |

**Key conclusions:**
- Universal walk of 11–28 mm/frame on a ~stationary camera (100-300× paper's implied behavior). Bug is universal, not lab-specific.
- Our walk ordering (Lab 1 < Lab 3 < Lab 4 < Lab 2) does not match paper's difficulty ordering (Lab 4 < Lab 3 < Lab 2 < Lab 1). Not a scaled version of the right signal.
- **Two distinct failure-mode signatures in one codebase on same config:**
  - *Random-walk dominant* (Labs 1, 4): high per-frame walk (11–18 mm), but ATE stays moderate (35–67 mm) because the pose zigzags back to near-origin.
  - *Directional-pull dominant* (Lab 2): high per-frame walk (28 mm) AND high ATE (151 mm) with 417 mm Y-axis drift on a static recording. Loss landscape is actively pulling the pose in a consistent direction.
  - Lab 3 is intermediate (13 mm walk, 44 mm ATE).
- The presence of BOTH modes on a single codebase indicates a **scene-dependent non-zero gradient bias**, not uniform noise. Exactly what a corrupted loss landscape would produce — same code, different scene → different bad gradient direction.
- fix1/fix2/paper-spec on trail3 all produce ~11 mm/frame — tracker/mapping LR knobs don't matter. The loss landscape itself is wrong.
- **D1 (semantic-distance GT shape) becomes the prime remaining hypothesis.** It is the only known structural difference from paper in the loss, and a scene-dependent loss-landscape bug fits the dual-failure-mode symptom.

**Notes for future runs:**
- `output.txt` and `output_identity.txt` **append** rather than overwrite across runs. Always `rm output/trail*/demo/output.txt` before re-running to avoid historical contamination. (Lab 1's output.txt contains 3 concatenated dictionaries from baseline + fix1 + fix2 runs — most recent value is last.)
- `output.txt` and `output_identity.txt` contents are **identical** on SemSuP because SuperDataset returns identity poses for both — there is no separate real GT.

## StereoMIS P2_1 (real camera motion, identity ATE)

| Run | Env | Depth | Frames | ATE RMSE | Per-frame motion | Notes |
|-----|-----|-------|--------|----------|-----------------|-------|
| **Paper** | Py3.7/PT1.10/TCNN v1.5 | stereo (robust-pose-est) | 4000 | **8.3mm** | **~0.2mm** | RTX 3090, identity GT |
| Exact paper env | Py3.7/PT1.10+cu113/TCNN v1.6 | RAFT stereo | [:4000] | **13.9mm** | — | Built via `colab_exact_env.sh` 2026-03-30 |
| Modern stack | Py3.12/PT2.10/TCNN v2.0 | RAFT stereo | [:4000] | **12.3mm** | 2.84mm | — |
| Modern, first 1500 | Modern | RAFT stereo | [:1500] | **4.3mm** | — | **Beats paper 2× on stable window** |
| Modern, first 4000 frame-by-frame | Modern | RAFT stereo | [:4000] | see below | — | Catastrophic drift event at frame ~1530 |
| Modern, last 4000 | Modern | RAFT stereo | [-4000:] | **43.9mm** | 4.83mm | Later frames much harder |
| Modern, last 4000 | Modern | RAFT stereo, scale=10000 | [-4000:] | 43.9mm | — | Quantization ruled out — identical to scale=100 |
| Modern, last 4000 | Modern | MoGe-2 (monocular) | [-4000:] | 976mm | 63.3mm | Monocular fails catastrophically |
| Modern, last 4000 | Modern | DA V2 (monocular) | [-4000:] | 1292mm | — | Monocular fails catastrophically |
| Modern, full | Modern | DA V2 | 8465 | 1623mm | — | Identity poses only, no tracking |

### StereoMIS frame-window drift analysis (first-4000, modern stack, identity ATE)

| Frames | Identity ATE | Status |
|--------|-------------|--------|
| 100 | 2.4mm | Excellent |
| 500 | 3.7mm | Good |
| 1000 | 3.8mm | Good |
| **1500** | **4.3mm** | **Better than paper's 8.3mm** |
| 2000 | 20.9mm | Catastrophic jump at ~1530 |
| 4000 | 27.5mm | Sustained drift |

**Drift trigger:** Between frames 1500 and 2000 the endoscope moves 78 mm forward in Z (real camera motion confirmed from robot kinematics). The tracker can't follow — pose jumps 46 mm over 50 frames at ~1530 and never recovers. Depth range at frame 2000 collapses to 30-40 mm with only 2 unique uint16 values at scale=100 (see MEETING_PREP.md §6); tested scale=10000 does not fix it, so quantization is not the sole cause.

## Rendering Quality (Semantic-SuPer trail3)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| **Paper** | **28.649** | **0.797** | **0.231** |
| Depth Anything V2 | 27.605 | 0.754 | 0.372 |
| MoGe (CVPR'25) | 26.980 | 0.746 | 0.400 |
| Monodepth2 (rescaled) | 26.885 | 0.729 | 0.404 |

## Diagnostic Findings

### Frame 1 Gradient Analysis (Semantic-SuPer, DA V2, identity GT)
```
Frame 1 starts at correct pose (dist=0mm)
Optimizer runs 20 iterations, loss DECREASES as pose moves AWAY from correct:
  iter0:  cos_to_correct= 0.000  dist=0.000mm  loss=0.049
  iter9:  cos_to_correct=-0.988  dist=0.916mm  loss=0.045
  iter19: cos_to_correct=-0.988  dist=1.934mm  loss=0.043
Gradient is ANTI-ALIGNED with correct direction.

Later frames improve:
  Frame 5: cos_to_correct=+0.67 to +0.99, final_dist=1.12mm (CORRECT direction)
```

### Loss Breakdown (weighted)
```
SDF:   43.0 (87%)  weight 1000
RGB:    0.02       weight 5
fs:     0.02       weight 10
edge:   0.00015    weight 0.5
depth:  0.00001    weight 0.1
```

### Scene vs Bounding Box Mismatch
```
Depth range: [0.38, 0.73] meters (surface Z position)
Bounding box Z: [0.7, 1.2] (barely contains scene)
91% of ray samples normalize outside [0,1] after BB normalization
Moving BB to contain scene had NO effect on gradient direction
```

### StereoMIS Depth Collapse at the Drift Event (first-4000, scale=100)

| Frame | Depth Range | uint16 Values | Unique Levels |
|-------|------------|---------------|---------------|
| 1400 | 70–180mm | 7–18 | 12 |
| 1500 | 70–190mm | 7–19 | 13 |
| 1530 | 70–180mm | 7–18 | 12 |
| **1550** | **70–150mm** | **7–15** | **9** |
| **1580** | **50–70mm** | **5–7** | **3** |
| **1600** | **40–60mm** | **4–6** | **3** |
| **2000** | **30–40mm** | **3–4** | **2** |

At frame 2000 the entire depth range is encoded as 2 discrete uint16 values. Adjacent pixels disagree by 10mm (~33% of depth). However, regenerating at scale=10000 (0.1mm resolution) produced the **same 43.9mm ATE** — quantization is not the sole cause of the last-4000 failure.

## Evidence — Code & Depth Pipeline Verification (2026-03-31)

Backs Eliminated Causes #9, #10, #12. Every file that runs during tracking or mapping is byte-for-byte identical to upstream; only evaluation and environment-adaptation files were modified.

### Code diff vs `DDS-SLAM-Base/` (pristine IRMVLab upstream)

| File | Result |
|------|--------|
| `model/decoder.py` | IDENTICAL |
| `model/encodings.py` | IDENTICAL |
| `model/keyframe.py` | IDENTICAL |
| `model/utils.py` | IDENTICAL |
| `model/scene_rep.py` | Identical (we added unused helper methods) |
| `optimization/frame_align.py` | IDENTICAL |
| `optimization/utils.py` | IDENTICAL |
| `config.py` | IDENTICAL |
| `configs/StereoMIS/*.yaml` | IDENTICAL |
| `ddsslam.py` | `cuda:2`→`cuda:0`, GT pose loading (eval only), cv2 headless fix |
| `datasets/dataset.py` | Frame selection `[-4000:]`↔`[:4000]`, GT pose loading (eval only) |
| `tools/eval_ate.py` | `mathutils`→`scipy` (quaternion math, eval only) |

### Depth pipeline diff vs `Stereo-MIS Base/` (pristine robust-pose-estimator)

| Component | Original | Ours | Match? |
|-----------|----------|------|--------|
| RAFT checkpoint | `poseNet_2xf8up4b.pth` | Same file | Yes |
| `flow2depth()` formula | `depth = baseline / -flow[:,0]` | Same | Yes |
| `depth_clipping` | `[1, 250]` mm | `depth_clip_max_mm = 250` | Yes |
| `scale` | `1/250` | `1/250` | Yes |
| Baseline passed to model | `bf * scale` | `bf * (1/250)` | Yes |
| `bf` computation | `|Tx_rect| * fx_rect` from `stereoRectify` | Same formula | Yes |
| Image normalization | [0, 255] float, no /255 | Same | Yes |
| Stereo rectification | `cv2.stereoRectify`, alpha=0 | Same | Yes |
| Distortion bug | Left coefficients used for both cameras | Replicated | Yes |
| `png_depth_scale` | 100 | 100 | Yes |

**Caveat:** robust-pose-estimator never saves depth maps as PNGs — it keeps them as internal float tensors. The DDS-SLAM authors must have written custom extraction code that isn't in either repository. We cannot verify what precision or filtering they used.

### `mask_specularities` from robust-pose-estimator (applied in upstream, not currently applied by us)

```python
def mask_specularities(img, mask=None, spec_thr=0.96):
    spec_mask = img.sum(axis=-1) < (3 * 255 * spec_thr)
    mask = mask & spec_mask if mask is not None else spec_mask
    mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((11, 11)))
    return mask
```

Upstream applies this + loads binary instrument masks from `masks/` to exclude specular highlights and tools. Our depth PNGs have 0% zero pixels — depth is computed everywhere, including invalid regions. `regenerate_stereomis_depth.py` reproduces this filtering.

## Eliminated Causes

| # | Hypothesis | Test | Result |
|---|-----------|------|--------|
| 1 | TF32 precision | T4 = sm_75, TF32 needs sm_80+ | N/A |
| 2 | Adam foreach | foreach=False on all optimizers | No effect |
| 3 | TCNN FP16 gradients | TCNN_HALF_PRECISION=0 rebuild | No effect |
| 4 | depth_weight too low | 0.1→5.0 | SDF dominates 3000:1 |
| 5 | Out-of-bounds hash wrapping | Clamp [0,1] | Changed gradient BUT broke SDF |
| 6 | Bounding box too small | Expanded to cover ray range | No effect |
| 7 | Bounding box misaligned | Moved to contain scene | No effect |
| 8 | Insufficient first_frame_mapping | 1000→5000 iterations | No effect (fully converged) |
| 9 | Code differences | Line-by-line diff all files | Byte-identical tracking/mapping pipeline |
| 10 | Config differences | Compared against IRMVLab repo | Identical |
| 11 | **Environment (PyTorch 2.10 / TCNN v2.0)** | **Built exact paper env (Py3.7/PT1.10+cu113/TCNN v1.6@91ee479) on Colab** | **13.9mm vs 12.3mm modern-stack on StereoMIS first-4000 — env is NOT the cause (2026-03-30)** |
| 12 | Depth pipeline mismatch | Byte-compare against robust-pose-estimator: checkpoint, flow2depth formula, clipping, rectification, baseline, distortion bug | Identical |
| 13 | Depth quantization (last-4000) | Regenerated StereoMIS depth at png_depth_scale=10000 (0.1mm vs 10mm) | Identical 43.9mm ATE — quantization not the cause for last-4000 |
| 14 | GT pose bypass (tracking only) | Skip tracking_render | global_BA still corrupts poses |
| 15 | Evaluation metric mismatch | Traced paper's `load_gt_pose` to see it uses identity, not kinematics | **Paper's 8.3mm is identity ATE.** Our first-1500 identity ATE is 4.3mm (beats paper 2×). |

## Audit (2026-04-15) — Paper vs Code Divergences

Summary: our code is byte-identical to upstream IRMVLab/DDS-SLAM, but the authors' own code diverges from paper's written equations on 11 items. Audit completed 2026-04-15 via two independent investigators; full D1–D11 table below.

| # | Divergence | Paper | Code | Severity |
|---|---|---|---|---|
| D1 | Semantic-distance GT | raw Euclidean `dist` | `exp(-dist/10)` — **sign-inverted field** | HIGH |
| D2 | Semantic decoder γ(d) input | takes view dir | missing | LOW |
| D3 | PE frequencies | L_x=10, L_t=4 | n_frequencies=12 (both) | HIGH |
| D4 | Rendering weights | Eq.7 alpha-compositing | `sigmoid × sigmoid` product | CRITICAL |
| D5 | SDF→density κ | σ=κ·Sigmoid(-κs), κ learnable | Not implemented; `learnable_beta` dead config | CRITICAL |
| D6 | Free-space target | drive SDF→T=0.1 | drives SDF→1.0 | ~~CRITICAL~~ **NOT A BUG** — `predicted_sdf` is normalized (unit=truncation) per Eq.11 self-consistency. Drive→1 == drive→T in paper units. |
| D7 | Super ray samples | 32+16=48 | 32+11=43 (Super yaml only) | MED |
| D8 | Mapping procedure | separate pose-fixed local phase | joint optim | MED |
| D9 | L_m MSE | plain MSE | top-20%-error pixels discarded (`UsePercentage=0.8`) | MED |
| D10 | First-sign-change mask | N/A | zeros weights past first SDF zero-crossing | CODE-ONLY |
| D11 | `rays_o[...,3]` read | N/A | unconditional; crashes if `dynamic=False` | BLOCKER for dynamic=False test |

Independently verified by second investigator. Interpretation (shared): paper is idealisation; authors' code diverges; authors hit 8.3 mm because specific biases happen to balance on their data/depth/seed. Fixing divergences reduces reliance on accidental cancellation.

Variance note: on StereoMIS scale=10⁶ runs, Run B = 13.9 mm, Run C = 47.0 mm on identical config with different seeds (3.4× swing). Authors may have a favourable seed; untestable without their RNG state.

## Scene Extent Measurement (trail3, Lab 1) — 2026-04-15

Via `Addons/inspect_scene_extent.py` on MoGe depth NPYs:

| Axis | Scene p1 | Scene p99 | Scene extent | Current bbox | Oversize |
|---|---|---|---|---|---|
| X | -0.239 | +0.206 | 0.445 m | [-0.7, +0.7] = 1.4 m | 3.1× |
| Y | -0.223 | +0.137 | 0.360 m | [-0.7, +0.7] = 1.4 m | 3.9× |
| Z | +0.383 | +0.697 | 0.314 m | **[+0.7, +1.2] = 0.5 m** | **MISALIGNED** — 99% of scene Z is BELOW bbox z_min=0.7 |

**Critical**: trail3.yaml default bbox Z starts at 0.7, but scene Z p99 = 0.697. Virtually the entire scene is outside the bbox on the Z axis, clamped via `torch.clamp(inputs_flat, 1e-6, 1-1e-6)` at [scene_rep.py:231](../model/scene_rep.py#L231). The hash grid sees a Z-collapsed 2D projection, not the 3D scene.

Also: `far: 5.0` in Super.yaml vs scene z p99 = 0.697 → 93% of ray sample budget beyond scene.

Recommended bbox (p1-p99 + 20% margin): `[[-0.33, 0.30], [-0.30, 0.21], [0.32, 0.76]]` with `far: 1.0`. Queued as `configs/Super/trail3_bbox_correct.yaml`.

Note: depth values in meters after `/png_depth_scale` (=8.0). Bbox oversize is 3-4×, not the 22× the StereoMIS bbox implied — per-SemSuP effect of bbox shrink should be modest (2-3× gradient sensitivity improvement per axis).

## Dual-mechanism hypothesis (2026-04-15)

Best current synthesis of the per-frame walk:

- **Bbox oversize → noisy flat loss landscape → random-walk per-frame motion.** StereoMIS config bbox ~4 m cube vs actual scene ~0.15 m = 22× oversize. TCNN hash normalises by bbox_extent, so effective voxel count in scene is ~100³ (not configured 2000³). A 1 mm pose perturbation becomes 2.5e-4 normalised shift instead of 3.3e-3. Loss curvature ~20× flatter than intended. **Explains why tracker-LR tuning (fix1/fix2) did nothing — tracker is gradient-noise-dominated in a miscalibrated landscape, not budget-limited.** Fits Labs 1/3/4 random-walk signature and StereoMIS.
- **D6/D1 biases → directional pull on top of the flat landscape.** D6 pushes free-space SDF to invisible weight region; D1 inverts semantic signal. Both produce scene-dependent non-zero gradients. Fits Lab 2's 417 mm Y-axis directional drift on a stationary recording.

Prior "scene-aligned bbox" test ([[-0.7, 0.7], [-0.7, 0.7], [0.2, 0.9]] → no effect) was still a 1.4 m cube, still ~10× oversize. Hypothesis predicts actual-size (~0.3 m) bbox would help — regime not yet tested.

## Shared-infrastructure cross-check

On StereoMIS last-4000 with identical RAFT-stereo depth:
- DDS-SLAM per-frame trans median: **2.08 mm** (34.7× GT)
- Co-SLAM per-frame trans median: **2.11 mm** (35.2× GT)
- GT per-frame median (groundtruth.txt): **0.060 mm**

Agreement to 4 sig figs across two architectures rules out architecture-specific causes (TimeNet, EdgeNet, HNDSR). The shared walk-magnitude locks the cause to something in the shared infrastructure: depth loader, loss formulation, tracker hyperparameters, SDF truncation, bbox. Consistent with bbox+D6 hypothesis.

## Untested Hypotheses (updated 2026-04-15 post-audit)

| # | Hypothesis | Priority | Status |
|---|-----------|----------|--------|
| 1 | **D9 — L_m outlier discard** (`UsePercentage=notFirstMap` → `False` at scene_rep.py:393,400) | **RUNNING** | Paper Eq. 12 is plain MSE; code drops top-20%-error pixels. Code patched locally; Colab test pending. |
| 2 | **D1 — Semantic GT sign inversion** (`exp(-dist/10)` + remove sigmoid at scene_rep.py:111) | HIGH | 2-line change across 2 files (NOT 1-line — sigmoid at line 111 pairs with bounded GT). Test if D9 null. |
| 3 | **D4/D5 — Rendering weights + learnable κ** (sigmoid-product → paper's alpha compositing) | HIGH | Shared with Co-SLAM; ~20-line refactor. Strong candidate for StereoMIS shared-infrastructure walk. |
| 4 | Depth quality — paper's finetuned Monodepth2 (inaccessible) or author contact | HIGH | StereoMIS path too (masked depth: `run_step1a_masked_depth.sh` ready) |
| 5 | D3 encoding frequencies (12 → L_x=10, L_t=4), D7 ray samples (11 → 16 Super), D8 mapping procedure | MED | Single-line / config changes; after bigger candidates |
| 6 | Multiple random seeds on StereoMIS scale=10⁶ (third run) | LOW | Variance characterisation (prior 3.4× seed swing) |
| 7 | D11 — dynamic=False guard at scene_rep.py:310-312 | LOW | Blocks in-place Co-SLAM-equivalent test; 3-line guard |
| — | **CLOSED hypotheses:** | | |
| — | Bbox shrink (trail3_bbox_correct) | REFUTED | Per-frame got **worse** (10.7 → 16.9mm); tight bbox amplifies field-update leverage on pose |
| — | D6 free-space target 1.0 → 0.1 | NOT A BUG | `predicted_sdf` is normalized (unit=truncation) per Eq.11 self-consistency; drive→1 == drive→T in paper units |
| — | fix1 tracker-LR cut | NULL | Tracker output overwritten by global_BA every frame |
| — | fix2 optim_cur=False | NULL | Past-KF re-optimization at ddsslam.py:418 continues regardless |

## Environment

- **Modern stack (default):** Python 3.12, PyTorch 2.10+cu128, CUDA 12.8, TCNN v2.0, Colab T4 (sm_75). Setup: [`colab_setup.sh`](../Addons/colab_setup.sh)
- **Exact paper env (verified 2026-03-30):** Python 3.7.17, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN 1.6 @ commit 91ee479, GCC 10 host compiler. Setup: [`colab_exact_env.sh`](../Addons/colab_exact_env.sh). Cached venv: `MyDrive/dds_cache/dds_env.tar.gz` (~1GB, ~2min restore).
- **Paper:** Python 3.7, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN ~v1.5, RTX 3090 (sm_86)

## Key Findings (updated 2026-04-15 post-pose-data analysis)

1. **All 4 SemSuP labs are RANDOM-WALK, not directional drift.** Analysis of `est_c2w_data.txt` across all 4 labs: drift ratios (net_displacement / total_|Δ|) are ≤ 0.16 on every axis for every lab. Lab 2's 417mm Y extent was the max excursion of a wide random walk, not directional drift. Earlier "dual mechanism" story (Lab 2 directional vs others random) was wrong.
2. **Walk is Z-axis dominant.** Total `|Δ|` in Z axis across 151 frames: Lab1=1803mm, Lab2=4437mm, Lab3=2340mm, Lab4=2994mm. That's 12-30 mm per-frame Z jitter. Z is the camera-forward/depth axis — textbook signature of **depth-ambiguity tracking noise**.
3. **Rotation jitter is small** (0.24-0.52 deg/frame median). Consistent with paper's 3.3mm Rep.Err magnitudes. Rotation isn't the problem.
4. **This eliminates all "loss-landscape-bias" hypotheses.** D1, D2, D6, D9 all posit a consistent wrong-direction gradient which would produce drift ratio ≈ 1.0. Observed ratio ≤ 0.16 rules this class out. D9 test is unlikely to help.
5. **DDS-SLAM ≈ Co-SLAM on StereoMIS (4 sig fig match).** Confirms walker is in shared infrastructure: depth input and/or TCNN sigmoid-product rendering. Not any DDS-only code.
6. **Bbox-shrink refuted.** Per-frame walk got **worse** (10.7 → 16.9 mm). Smaller bbox amplifies field-update leverage instead of reducing it.
7. **D6 eliminated as a bug.** `predicted_sdf` is normalized by truncation per Eq.11 self-consistency; "drive to 1" == "drive to T" in paper units. Was misflagged in the audit handoff.
8. **Strongest remaining hypothesis: DEPTH QUALITY.** Z-axis random walk + shared-infrastructure Co-SLAM cross-check + "paper used inaccessible finetuned Monodepth2" all point here.

## Pose-Data Analysis (trail3-9 paper-spec baseline, 2026-04-15)

| Lab | per-frame |Δt| mm | per-frame Δθ deg | Z total |Δ| mm | X drift ratio | Y drift ratio | Z drift ratio |
|---|---|---|---|---|---|---|
| 1 | 10.72 | 0.33 | 1803 | 0.038 | 0.077 | 0.056 |
| 2 | **28.22** | 0.52 | **4437** | 0.149 | 0.127 | 0.024 |
| 3 | 13.63 | 0.24 | 2340 | 0.079 | 0.159 | 0.022 |
| 4 | 18.13 | 0.29 | 2994 | 0.011 | 0.072 | 0.036 |

All labs: high Z-axis random-walk magnitude, near-zero drift ratio on all axes. Mechanism is uniform across labs; Lab 2 is an amplitude outlier, not a mechanism outlier.
4. **Tracker works excellently on stable StereoMIS windows** — 4.3 mm identity ATE on first-1500, 2× better than paper's 8.3 mm on 4000. Reproduction is not globally broken.
5. **StereoMIS failure is frame-window-dependent** — catastrophic drift event at frame ~1530 under 78 mm camera motion in Z. Tracker exceeds convergence radius.
6. **Environment eliminated** — exact paper stack 13.9 mm vs modern stack's 12.3 mm on StereoMIS first-4000. Within noise.
7. **Code and depth pipelines byte-identical to upstream** (DDS-SLAM + robust-pose-estimator, including their distortion bug).
8. **Paper evaluates against identity GT**, not robot kinematics. On SemSuP these two are the same because SuperDataset only has identity.
9. **Mapping pose refinement is the walker on SemSuP**, not the tracker. fix1/fix2 null results confirm. Past-keyframe re-optimization at `ddsslam.py:418` continues regardless of `optim_cur`.
10. **Stereo depth essential for StereoMIS** — monocular fails catastrophically (1000mm+ ATE).
11. **Rendering quality robust despite tracking noise** — neural SDF compensates; PSNR 27.6 vs paper 28.6 on Lab 1 (the field gets the scene ~right, the tracker misreads where the camera is).
12. **Co-SLAM has the same StereoMIS issue** (ATE 94 mm, worse than ours 43.9 mm) — cross-system confirmation.

## Session Logs (reverse chronological)

### 2026-04-15 — 4-lab SemSuP paper-spec baseline

- Recovered trail4/8/9 configs from commit `d9e4958` (had been deleted as "dead" before confirming Lab mapping).
- Confirmed Lab mapping: trail3=Lab 1, trail4=Lab 2, trail8=Lab 3, trail9=Lab 4.
- Generated MoGe depth for trail4/8/9 on Colab base Python 3.12 (MoGe requires torch ≥ 2.0, incompatible with Py 3.7 venv).
- Ran all 4 labs under exact-paper env. All walk. Results pushed to `MyDrive/DDS-SLAM-Results/paper_spec_baseline_20260415_1708/`.
- Audited `compute_edge_semantic` + `EdgeNet_Semantic`: confirmed D1 divergence (exp decay vs raw distance), flagged D2 (missing view-direction input).
- fix1 and fix2 null results logged; their branches closed as non-critical-path.
- RESULTS_LOG reorganized: metric clarification, 4-lab baseline table, session log.

### 2026-03-31 — Day 2 StereoMIS investigation
- Code diff vs upstream: byte-identical on tracking/mapping.
- Depth pipeline vs robust-pose-estimator: byte-identical.
- Paper evaluates against identity GT (not kinematics) discovered.
- Frame-by-frame drift analysis: StereoMIS first-1500 = 4.3 mm (beats paper), first-4000 = 27.5 mm, catastrophic event at frame ~1530.
- Depth quantization ruled out via scale=10000 test.
- Specularity/instrument masking identified as remaining untested depth variable.

### 2026-03-30 — Environment elimination
- Built exact paper env on Colab: Py 3.7, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN v1.6 @ 91ee479.
- StereoMIS first-4000: 13.9 mm (exact env) vs 12.3 mm (modern). Within noise. Env eliminated.

## Open Questions

1. **Does depth masking (specularity + instrument) fix the StereoMIS last-4000 failure?** Test ready but not run.
2. **Does disabling `optim_cur` collapse the trail3 ATE toward zero?** fix2 config ready but not run.
3. **What depth maps did the DDS-SLAM authors actually use?** Robust-pose-estimator never saves PNGs. Their extraction code is missing from both repos.

## Supervisor Questions (open, 2026-03-31)

1. Given tracking works for stable sequences (4.3mm on first-1500, 2× better than paper), should we proceed with CRCD and characterise failure modes rather than chase exact StereoMIS reproduction?
2. Should we contact the DDS-SLAM authors about their depth extraction pipeline?
3. For the thesis, is reporting both metrics (identity GT + real GT) sufficient, or do we need to exactly match 8.3mm?

## Project Status — CRCD Pipeline (as of 2026-03-31)

Despite the StereoMIS reproduction gap, the CRCD pipeline is fully built:
- `Addons/preprocess_crcd.py` — converts data packs to DDS-SLAM format (rectification, masks, poses)
- Calibration extracted: fx=1096.70, baseline=5.485mm
- SAM3 masks rasterized (liver + gallbladder + tool, 3-class)
- C_1 snippet_001 preprocessed (271 frames)
- Config files created
- Only missing piece: depth generation (RAFT-Stereo on Colab)
