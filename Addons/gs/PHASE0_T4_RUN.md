# GS Phase-0 — fresh-boot run on a Colab T4

Run unmodified EndoGSLAM on CRCD c1_001 (MoGe-2 depth) and judge it against the NeRF canon (~3.15 mm).
T4 = **sm_75**. Everything below is terminal commands (VS Code tunnel). Background:
`DDS-SLAM/Addons/docs/GS_MIGRATION_HANDOVER_20260622.md` §5/§14.

## 0. One-time, on your Windows box — push so Colab can pull
```bash
git -C "C:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM" push origin diagnosis-live
```

## 1. Fresh Colab T4 — get our repo + build EndoGSLAM
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv      # expect: Tesla T4, 15360 MiB

cd /content
git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM.git || (cd DDS-SLAM && git pull)

# clones EndoGSLAM @ pinned base + lays down the CRCD adapter (idempotent; patches, doesn't vendor)
cd /content/DDS-SLAM
bash Addons/gs/setup_endogslam_crcd.sh                      # -> /content/EndoGSLAM ready
```

## 2. Env — deps + Gaussian rasterizer for sm_75 (once, ~10 min)
Do NOT `pip install -r EndoGSLAM/requirements.txt` (it pins numpy 1.21 / open3d 0.16 and can break Colab's torch).
Install the unpinned set on top of Colab's torch instead:
```bash
pip -q install natsort imageio kornia lpips pytorch-msssim torchmetrics open3d trimesh opencv-python scipy plyfile matplotlib
TORCH_CUDA_ARCH_LIST=7.5 pip -q install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/
python -c "import torch, diff_gaussian_rasterization; print('rasterizer OK |', torch.cuda.get_device_name(0))"
```

## 3. Data — stage CRCD-Published c1_001 (360) into the layout the loader expects
🚨 Use **CRCD-Published (360)**, never the stale 271-row copy. Ensure Drive is mounted at `/content/drive`
(if your tunnel terminal can't see it, run `from google.colab import drive; drive.mount('/content/drive')` in
one Colab cell). First LOOK at the layout, then copy (don't assume folder names):
```bash
SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/C_1/snippet_001
MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/C_1/snippet_001
find "$SNIP" -iname "*l.png" | head; find "$SNIP" -iname "groundtruth.txt"
find "$MOGE" -iname "*.npy" | head

DST=/content/EndoGSLAM/data/CRCD/C1_001
mkdir -p "$DST/video_frames" "$DST/depth/moge2"
# ↓↓↓ adjust the 3 source globs to what `find` showed above ↓↓↓
cp "$SNIP"/<frames_dir>/*l.png        "$DST/video_frames/"
cp "$MOGE"/depth/*.npy                "$DST/depth/moge2/"
cp "$SNIP"/groundtruth.txt            "$DST/"
# must be 360 == 360 == 360:
echo "frames=$(ls $DST/video_frames/*l.png | wc -l)  depth=$(ls $DST/depth/moge2/*.npy | wc -l)  gt=$(grep -vc '^#' $DST/groundtruth.txt)"
```
(If frames/depth are named differently, either rename or edit the globs in
`/content/EndoGSLAM/configs/data/crcd.yaml` `paths:` — the loader is config-driven.)

## 4. B1 depth audit — the only depth-scale risk that bites (do BEFORE training)
```bash
cd /content/EndoGSLAM && python - <<'PY'
import glob, numpy as np
ds = sorted(glob.glob('data/CRCD/C1_001/depth/moge2/*.npy'))
m = [float(np.median(np.load(p))) for p in ds]
print(f"{len(ds)} frames | median depth min {min(m):.4f} max {max(m):.4f} ratio {max(m)/min(m):.2f}")
print("median ~O(1) & ratio~1 -> keep png_depth_scale:1.0 | hundreds -> set png_depth_scale | ratio>>1.5 -> MoGe scale flicker (the real B1)")
PY
```

## 5. Run — n=3 seeds (each writes its own dir; seed coin-flip is real)
```bash
cd /content/EndoGSLAM
for S in 0 1 2; do echo "=== seed $S ==="; SEED=$S python scripts/main.py configs/crcd/crcd_base.py; done
# outputs: experiments/CRCD_base/C1_001_s0 | _s1 | _s2
```

## 6. Eval — the arbiter (Sim3, never rigid Horn)
```bash
cd /content/EndoGSLAM
for S in 0 1 2; do echo "=== seed $S ==="; \
  python scripts/eval_sim3_crcd.py --params experiments/CRCD_base/C1_001_s$S/params.npz \
                                   --gt data/CRCD/C1_001/groundtruth.txt; done
# render PSNR/SSIM/LPIPS (their own metric; report alongside):
python scripts/calc_metrics.py --gt data/CRCD/C1_001 --render experiments/CRCD_base/C1_001_s0 --test_single
```

## 7. THE GATE
PASS = converges + non-degenerate trajectory/renders, **Sim3 ATE ~ same order of magnitude as DDS-SLAM-Base
(~3.15 mm)**, **path-ratio ≈ 1 and |Pearson|_dom high** (CRCD is sub-SNR — judge these together, never bare
ATE), render PSNR comparable. Report {ATE mean/max, seed-std over n=3, scale, path-ratio, |Pearson|dom,
PSNR/SSIM/LPIPS}. **PASS → Phase 1 (port the uncertainty win). FAIL → stop, reconsider the base.**

### If it breaks
- rasterizer import error → re-run the `TORCH_CUDA_ARCH_LIST=7.5 pip install …` line; ensure `pip install ninja`.
- OOM on T4 (16 GB) → lower `desired_image_height/width` in `configs/crcd/crcd_base.py` (e.g. 270×480).
- loader errors `GT (271) != frames (360)` → you staged the stale GT; use CRCD-Published 360.
- `No CRCD frames matched` → fix `paths.color_glob` in `configs/data/crcd.yaml` to your frame naming.
