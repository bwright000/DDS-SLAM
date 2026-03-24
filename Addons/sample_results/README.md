# DDS-SLAM Reproduction — Sample Results

## Trail3 (Lab1) — 151 frames on Colab T4

### Rendering Quality Comparison

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| **Paper (DDS-SLAM)** | **28.649** | **0.797** | **0.231** |
| Depth Anything V2 | 27.605 | 0.754 | 0.372 |
| Monodepth2 (rescaled) | 26.885 | 0.729 | 0.404 |

### Frame 0 (First Frame)
| Ground Truth | Depth Anything V2 | Monodepth2 |
|:---:|:---:|:---:|
| ![GT](frame000_ground_truth.png) | ![DA](frame000_depth_anything.jpg) | ![MD](frame000_monodepth2.jpg) |

### Frame 75 (Mid Sequence)
| Ground Truth | Depth Anything V2 | Monodepth2 |
|:---:|:---:|:---:|
| ![GT](frame075_ground_truth.png) | ![DA](frame075_depth_anything.jpg) | ![MD](frame075_monodepth2.jpg) |

### Frame 150 (Final Frame)
| Ground Truth | Depth Anything V2 | Monodepth2 |
|:---:|:---:|:---:|
| ![GT](frame150_ground_truth.png) | ![DA](frame150_depth_anything.jpg) | ![MD](frame150_monodepth2.jpg) |

### Estimated Camera Trajectories
| Depth Anything V2 | Monodepth2 |
|:---:|:---:|
| ![Pose DA](pose_depth_anything.png) | ![Pose MD](pose_monodepth2.png) |

## Notes
- Depth maps generated using alternative models (paper uses Python-SuPer finetuned Monodepth2, checkpoints inaccessible)
- Depth Anything V2 provides metric depth; Monodepth2 KITTI-pretrained required manual rescaling to surgical range
- RAFT-Stereo (stereo pairs) not yet evaluated
- Full details in [session log](../memory/session_20260323.md)
