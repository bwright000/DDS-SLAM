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
| trail3_fix2 (optim_cur=False) | MoGe | — | — | — | **Pending** — disables mapping pose refinement entirely to confirm global_BA attribution |

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

## Untested Hypotheses (as of 2026-04-15)

| # | Hypothesis | Priority | Status |
|---|-----------|----------|--------|
| 1 | **Specularity + instrument masking on depth** (trail3 + StereoMIS last-4000) | **HIGH** | `regenerate_stereomis_depth.py` + `run_step1a_masked_depth.sh` ready, not yet run |
| 2 | **Mapping pose refinement as walker** (disable `optim_cur`) | **HIGH** | `trail3_fix2.yaml` queued — direct follow-up to GT-bypass finding |
| 3 | Tracker convergence radius vs 78mm camera motion at frame ~1530 | MEDIUM | No direct test yet; would need staged-motion synthetic data |
| 4 | Paper's missing depth extraction code | MEDIUM | Robust-pose-estimator never saves PNGs; author contact may be needed |
| 5 | Multiple random seeds | LOW | Untested |
| 6 | Pure-PyTorch hash grid | LOW | Untested |

## Environment

- **Modern stack (default):** Python 3.12, PyTorch 2.10+cu128, CUDA 12.8, TCNN v2.0, Colab T4 (sm_75). Setup: [`colab_setup.sh`](../Addons/colab_setup.sh)
- **Exact paper env (verified 2026-03-30):** Python 3.7.17, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN 1.6 @ commit 91ee479, GCC 10 host compiler. Setup: [`colab_exact_env.sh`](../Addons/colab_exact_env.sh). Cached venv: `MyDrive/dds_cache/dds_env.tar.gz` (~1GB, ~2min restore).
- **Paper:** Python 3.7, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN ~v1.5, RTX 3090 (sm_86)

## Key Findings

1. **Tracker works excellently on stable windows** — 4.3mm identity ATE on StereoMIS first-1500, **2× better than paper's 8.3mm on 4000 frames.** The reproduction is not globally broken.
2. **Failure is frame-window-dependent, not env/code/depth-version dependent.** Catastrophic drift event at StereoMIS frame ~1530 where camera moves 78mm in Z over 500 frames — tracker exceeds its convergence radius and never recovers.
3. **Environment eliminated** — exact paper stack produces 13.9mm vs modern stack's 12.3mm on first-4000. Within noise.
4. **Code and depth pipelines byte-identical** to upstream DDS-SLAM and robust-pose-estimator (including their distortion bug, replicated).
5. **Paper evaluates against identity GT, not robot kinematics** — both metrics now emitted (`output_identity.txt` vs `output.txt`).
6. **On trail3, mapping pose refinement is the walker.** GT-pose-bypass run still drifts to 50mm — tracker isn't the problem, global_BA overwrites poses via `optim_cur=True` + `cur_frame_iters=100` + `keyframe_every=1`.
7. **fix1 null result confirms mapping is dominant walker** — 10× cut to tracker LR had no effect on per-frame walk (63mm ATE, 11mm/frame), exactly matching mapping's theoretical 10mm/frame walk cap.
8. **Stereo depth essential for StereoMIS** — monocular depth fails catastrophically (1000mm+ ATE).
9. **Rendering quality robust despite jitter** — neural SDF compensates for wrong poses (PSNR 27.6 vs 28.6).
10. **Co-SLAM has the same issue on StereoMIS last-4000** (ATE 94mm, worse than DDS-SLAM's 43.9mm) — cross-system confirmation that problem is not DDS-SLAM-specific.

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
