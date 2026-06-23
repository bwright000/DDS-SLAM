# v1 flow_map (mapping catch-up) — fresh-boot A/B on a Colab T4

Runs the v1 A/B: **base vs flow_map vs uniform-control**, parity-gated, held-out-dynamic metric.
Same proven bootstrap as Phase-0 (`PHASE0_T4_RUN.md`) — `setup_endogslam_crcd.sh` now ALSO applies the
v1 flow_map patches (parity-safe; default-off == base). T4 = **sm_75**. Smoke on c1_001 first (already a
clean snippet, motion in the final ~30 frames), then headline on **C2_001** (max attribution P99).

## 1. Fresh Colab T4 — repo + EndoGSLAM + v1 patches
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv          # Tesla T4, ~15360 MiB
cd /content
git clone -b diagnosis-live https://github.com/bwright000/DDS-SLAM.git
cd /content/DDS-SLAM
bash Addons/gs/setup_endogslam_crcd.sh                          # EndoGSLAM@pin + CRCD adapter + v1 flow_map
```

## 2. Env — deps + Gaussian rasterizer (sm_75) + RAFT (~10 min)
```bash
pip -q install natsort imageio kornia lpips pytorch-msssim torchmetrics open3d trimesh opencv-python scipy plyfile matplotlib pyyaml scikit-learn
TORCH_CUDA_ARCH_LIST=7.5 pip -q install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth/
python -c "import torch, diff_gaussian_rasterization; from torchvision.models.optical_flow import raft_large; print('rasterizer+RAFT OK |', torch.cuda.get_device_name(0))"
```
*(Don't `pip install -r EndoGSLAM/requirements.txt` — its pins break Colab's torch. RAFT-large weights auto-download ~20 MB on first run.)*

## 3. Stage a snippet — RECTIFIED, the canonical ARM-4 pipeline (Drive mounted at /content/drive)
Set `NAME`/`EP`/`SN` for the snippet. **c1_001** = `C_1/snippet_001`; **C2_001** = `C_2/snippet_001`.
```bash
NAME=C1_001; EP=C_1; SN=snippet_001                 # <-- change to C2_001 / C_2 / snippet_001 for the headline
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
DRIVE_MOGE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2
CALIB=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
SNIP=$DRIVE_CRCD/$EP/$SN; MOGE=$DRIVE_MOGE/$EP/$SN/depth
LOCAL=/content/data/CRCD/$NAME; STAGED=/content/data/CRCD_staged/$NAME
SCENE=/content/EndoGSLAM/data/CRCD/$NAME

mkdir -p $LOCAL/rgb $LOCAL/semantic_instance $LOCAL/depth
cp -rn $SNIP/rgb/.               $LOCAL/rgb/
cp -rn $SNIP/semantic_instance/. $LOCAL/semantic_instance/
cp -rn $MOGE/.                   $LOCAL/depth/
cp -f  $SNIP/groundtruth.txt     $LOCAL/
cp -f  $SNIP/intrinsics.yaml     $LOCAL/ 2>/dev/null || true

python /content/DDS-SLAM/Addons/preprocess/preprocess_crcd_published.py \
   --snippet_dir $LOCAL --calib_pkl $CALIB --output_dir $STAGED
python /content/DDS-SLAM/Addons/colab/crcd_assemble_sgs.py --mode rectified \
   --staged $STAGED --moge_depth $LOCAL/depth --calib_pkl $CALIB \
   --out $SCENE --depth_scale 10000 --n_classes 4 \
   --emit_yaml /content/EndoGSLAM/configs/data/crcd.yaml
sed -i "s/dataset_name: 'replica'/dataset_name: 'crcd'/" /content/EndoGSLAM/configs/data/crcd.yaml
echo "frames=$(ls $SCENE/frames/*.jpg|wc -l) depths=$(ls $SCENE/depths/*.png|wc -l) traj=$(wc -l < $SCENE/traj.txt)"   # want all equal
```

## 4. The v1 A/B  (SMOKE first: 1 seed on c1_001 -> validates parity + the whole pipeline)
```bash
cd /content/EndoGSLAM
SNIPPETS="C1_001" SEEDS="0" bash Addons/gs/flow_map_ab_20260623.sh
```
Watch, in order:
- `>>> INC0 PARITY PASS` — the hard gate (off == base). If it FAILS the runbook ABORTS before any run.
- `[flow_map] ENABLED lam=… deadband=…` on the flowmap/unictrl arms; `FM_HB frame N/M …` heartbeat.
- `DECISION` block at the end (1-seed) — base/flowmap/unictrl dynPSNR + the KEEP/escalate verdict.

Then the **headline** (after staging C2_001 via §3 with `NAME=C2_001 EP=C_2`):
```bash
cd /content/EndoGSLAM
SNIPPETS="C2_001" bash Addons/gs/flow_map_ab_20260623.sh           # full n=3 (SEEDS defaults to 0 1 2)
```

## 5. What it produces / how to read it
- Per run: `experiments/CRCD_base/<NAME>_<arm>_s<seed>/metrics_split.json` (held-out∩dynamic = HEADLINE,
  held-out∩static = GUARD) + the 6-panel video. Auto-shipped to `MyDrive/Outputs/GS_flowmap_ab_20260623/`.
- **DECISION rule (printed):** KEEP v1 IFF flowmap held-out-dynamic PSNR beats **base** AND the **uniform
  control** (n=3 non-overlapping) AND the guard holds (static PSNR ≥ base − τ). Beat-base-but-not-uniform =
  a global Adam LR effect, NOT localized catch-up → not a win.
- KEEP → proceed to v2 (deform field). Not-a-win → static-chase ceiling → v2 or retune `FM_LAMBDA`/`FM_DEPTH_DB`.

### If it breaks
- rasterizer import error → re-run the `TORCH_CUDA_ARCH_LIST=7.5 pip install …` (`pip install ninja` first).
- `INC0 PARITY FAIL` → read `_flowmap_logs/parity.log`; means a main.py edit wasn't guarded (drift) — re-run setup.
- OOM on T4 → `DOWNSAMPLE=2` env on the runbook (halves res); c1_001/C2_001 native usually fit.
- RAFT slow → `FM_RAFT_SMALL=1` (raft_small) for a faster smoke.
- `traj poses != frames` → re-run §3 assemble (index-paired; don't mix a stale GT).
