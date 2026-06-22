# GS Phase-0 — fresh-boot run on a Colab T4 (CRCD c1_001, RECTIFIED)

Run unmodified EndoGSLAM on CRCD c1_001 with MoGe-2 depth and judge it vs the GS peer (SGS-SLAM 3.31 mm,
rectified) and the NeRF canon (3.15 mm, raw-left cross-pipeline ref). T4 = **sm_75**. Terminal commands
(VS Code tunnel). **RECTIFIED input** — locked policy for pinhole CRCD SLAM methods
([[project_benchmark_onboarding_audit_20260617]]); we reuse the SAME tested staging as the SGS benchmark
(`preprocess_crcd_published.py` + `crcd_assemble_sgs.py --mode rectified`), so K + depth-scale come from the
calib, never a hardcode.

## 0. One-time, on your Windows box — push so Colab can pull
```bash
git -C "C:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM" push origin diagnosis-live
```

## 1. Fresh Colab T4 — get repo + bootstrap EndoGSLAM
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv          # expect Tesla T4, ~15360 MiB
cd /content
git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM.git
cd /content/DDS-SLAM
bash Addons/gs/setup_endogslam_crcd.sh                          # clones EndoGSLAM@pin + applies the adapter
```

## 2. Env — deps + Gaussian rasterizer for sm_75 (~10 min)
```bash
pip -q install natsort imageio kornia lpips pytorch-msssim torchmetrics open3d trimesh opencv-python scipy plyfile matplotlib pyyaml
TORCH_CUDA_ARCH_LIST=7.5 pip -q install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/
python -c "import torch, diff_gaussian_rasterization; print('rasterizer OK |', torch.cuda.get_device_name(0))"
```
*(Don't `pip install -r EndoGSLAM/requirements.txt` — its numpy/open3d pins can break Colab's torch.)*

## 3. Stage CRCD c1_001 — RECTIFIED, via the canonical ARM-4 pipeline
Ensure Drive is mounted at `/content/drive`. This is the exact staging `run_sgsslam.sh` uses (so the
EndoGSLAM scene == the SGS scene). c1_001 → episode `C_1`, `snippet_001`.
```bash
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
DRIVE_MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2
CALIB=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
SNIP=$DRIVE_CRCD/C_1/snippet_001          # raw: rgb/ semantic_instance/ groundtruth.txt intrinsics.yaml
MOGE=$DRIVE_MOGE/C_1/snippet_001/depth    # uint16 MoGe-2 depth (scale 10000), raw-space
LOCAL=/content/data/CRCD/c1_001           # local copy (Drive FUSE is slow on long runs)
STAGED=/content/data/CRCD_staged/C1_001   # rectified preprocess output
SCENE=/content/EndoGSLAM/data/CRCD/C1_001 # assembled scene EndoGSLAM reads

# 3a. copy raw snippet + MoGe depth local
mkdir -p $LOCAL/rgb $LOCAL/semantic_instance $LOCAL/depth
cp -rn $SNIP/rgb/.               $LOCAL/rgb/
cp -rn $SNIP/semantic_instance/. $LOCAL/semantic_instance/
cp -rn $MOGE/.                   $LOCAL/depth/
cp -f  $SNIP/groundtruth.txt     $LOCAL/
cp -f  $SNIP/intrinsics.yaml     $LOCAL/ 2>/dev/null || true

# 3b. rectify (left map only; right not needed) -> staged video_frames + semantic_class + rectified_calib.txt
python /content/DDS-SLAM/Addons/preprocess/preprocess_crcd_published.py \
   --snippet_dir $LOCAL --calib_pkl $CALIB --output_dir $STAGED

# 3c. assemble -> EndoGSLAM scene (frames/depths/semantic_ids/traj.txt) + emit the data yaml (RECTIFIED K + scale)
python /content/DDS-SLAM/Addons/colab/crcd_assemble_sgs.py --mode rectified \
   --staged $STAGED --moge_depth $LOCAL/depth --calib_pkl $CALIB \
   --out $SCENE --depth_scale 10000 --n_classes 4 \
   --emit_yaml /content/EndoGSLAM/configs/data/crcd.yaml
sed -i "s/dataset_name: 'replica'/dataset_name: 'crcd'/" /content/EndoGSLAM/configs/data/crcd.yaml

# 3d. verify (want frames == depths == traj lines, all 360)
echo "frames=$(ls $SCENE/frames/*.jpg|wc -l) depths=$(ls $SCENE/depths/*.png|wc -l) traj=$(wc -l < $SCENE/traj.txt)"
grep -E "image_height|image_width|fx|png_depth_scale" /content/EndoGSLAM/configs/data/crcd.yaml
```

## 4. B1 depth audit — scale stability (before training)
```bash
cd /content/EndoGSLAM && python - <<'PY'
import glob, cv2, numpy as np
ds = sorted(glob.glob('data/CRCD/C1_001/depths/*.png'))
m = [float(np.median(cv2.imread(p, -1).astype(np.float32))) / 10000.0 for p in ds]   # /scale -> metres
print(f"{len(ds)} frames | median depth(m) min {min(m):.3f} max {max(m):.3f} ratio {max(m)/min(m):.2f}")
print("ratio ~1 -> good (per-snippet scale stable; Sim3 absorbs the global up-to-scale factor)")
PY
```

## 5. Run — n=3 seeds (native rectified res; DOWNSAMPLE=2 if a longer snippet OOMs the T4)
```bash
cd /content/EndoGSLAM
for S in 0 1 2; do echo "=== seed $S ==="; SEED=$S python scripts/main.py configs/crcd/crcd_base.py; done
# outputs: experiments/CRCD_base/C1_001_s0|_s1|_s2 ; c1_001=360 frames fits T4 native (SGS confirmed)
```

## 6. Eval — Sim3 (our ruler) + render
```bash
cd /content/EndoGSLAM
for S in 0 1 2; do echo "=== seed $S ==="; \
  python scripts/eval_sim3_crcd.py --params experiments/CRCD_base/C1_001_s$S/params.npz \
                                   --gt data/CRCD/C1_001/traj.txt; done
python scripts/calc_metrics.py --gt data/CRCD/C1_001 --render experiments/CRCD_base/C1_001_s0 --test_single
```

## 7. THE GATE
PASS = converges + non-degenerate trajectory/renders, **Sim3 ATE in the SGS/NeRF ballpark (SGS rectified
3.31 mm; NeRF raw-left 3.15 mm)**, **path-ratio ≈ 1 and |Pearson|_dom high** (sub-SNR — judge together, never
bare ATE), render PSNR comparable (SGS c1_001 ≈ 22.6). Report {ATE mean/max, seed-std n=3, scale, path-ratio,
|Pearson|dom, PSNR/SSIM/LPIPS}. **PASS → Phase 1 (port the uncertainty win). FAIL → stop, reconsider the base.**

### If it breaks
- rasterizer import error → re-run the `TORCH_CUDA_ARCH_LIST=7.5 pip install …` (add `pip install ninja`).
- OOM on T4 → `DOWNSAMPLE=2 SEED=0 python scripts/main.py configs/crcd/crcd_base.py` (halves res).
- `traj poses != frames` → re-run 3c (assembly is index-paired; don't mix a stale GT).
- `No assembled CRCD frames` → 3b/3c didn't run or wrote elsewhere; check `$SCENE/frames`.
- depth median not ~O(1) → wrong `png_depth_scale`; MoGe PNG is scale 10000.
