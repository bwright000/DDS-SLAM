# DDS-SLAM Results Log

All results from our reproduction attempts, tracked by git commit and run conditions.

## Semantic-SuPer trail3 (151 frames, static camera, GT = identity)

| Run | Depth | ATE RMSE | Per-frame motion | PSNR | Notes |
|-----|-------|----------|-----------------|------|-------|
| **Paper** | **Finetuned Monodepth2** | **N/A** | **~0mm** | **28.6** | Python 3.7, PyTorch 1.10, TCNN v1.5, RTX 3090 |
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
| DA V2 + GT bypass + zero vox + lr_pose=0 | DA V2 | **PENDING** | **PENDING** | — | **Currently running — BA can't move poses either** |

## StereoMIS P2_1 (real camera motion)

| Run | Depth | Frames | ATE RMSE | Per-frame motion | Notes |
|-----|-------|--------|----------|-----------------|-------|
| **Paper** | **stereo (robust-pose-est)** | **4000** | **8.3mm** | **~0.2mm** | Python 3.7, PyTorch 1.10, TCNN v1.5, RTX 3090 |
| RAFT stereo first 4000 | RAFT stereo | [:4000] | **12.3mm** | 2.84mm | Best result, 1.49x paper |
| RAFT stereo last 4000 | RAFT stereo | [-4000:] | 43.9mm | 4.83mm | Later frames much harder |
| MoGe last 4000 | MoGe-2 | [-4000:] | 976mm | 63.3mm | Monocular fails catastrophically |
| DA V2 last 4000 | DA V2 | [-4000:] | 1292mm | — | Monocular fails catastrophically |
| DA V2 all 8465 | DA V2 | all | 1623mm | — | Full sequence, identity poses |

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
| 9 | Code differences | Line-by-line diff all files | Identical pipeline |
| 10 | Config differences | Compared against IRMVLab repo | Identical |
| 11 | Docker exact env | Docker won't run on Colab | Blocked |
| 12 | Conda PyTorch 1.10 | TCNN won't compile | Blocked |
| 13 | TCNN v1.6 build | Won't compile against PyTorch 2.10 | Blocked |
| 14 | GT pose bypass (tracking only) | Skip tracking_render | global_BA still corrupts poses |

## Untested Hypotheses

| # | Hypothesis | Priority | Status |
|---|-----------|----------|--------|
| 1 | TimeNet (dynamic=False) | **HIGH** | **RUNNING NOW** |
| 2 | PyTorch nn.Linear backward | MEDIUM | Untested |
| 3 | Multiple random seeds | MEDIUM | Untested |
| 4 | Pure-PyTorch hash grid | LOW | Untested |

## Environment

- **Ours:** Python 3.12, PyTorch 2.10+cu128, CUDA 12.8, TCNN v2.0, Colab T4 (sm_75)
- **Paper:** Python 3.7, PyTorch 1.10.1+cu113, CUDA 11.3, TCNN ~v1.5 (commit 91ee479), RTX 3090 (sm_86)

## Key Findings

1. **Stereo depth essential for StereoMIS** — monocular depth fails catastrophically (1000mm+ ATE)
2. **First 4000 frames >> last 4000** — paper likely used first 4000
3. **Rendering quality robust despite jitter** — neural SDF compensates for wrong poses (PSNR 27.6 vs 28.6)
4. **SDF gradient anti-aligned** on frame 1 — optimizer moves pose AWAY from correct position
5. **Both tracking AND global_BA** produce wrong gradients (GT pose bypass test)
6. **Gradient direction improves** over frames 2-5 as SDF gets updated by BA
7. **Co-SLAM has same reproducibility issue** (GitHub Issue #57)
8. **TCNN v2.0 JIT compilation** changed kernel execution vs v1.5, but FP32 rebuild had no effect
9. **Cannot reproduce paper's exact environment on Colab** — CUDA toolkit mismatch
