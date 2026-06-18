> **Resolved decisions apply.** See *Resolved Benchmark Decisions* in [00_COMMON.md](00_COMMON.md) — they override any “escalate”/“open question” below: Depth-L1 = input-vs-output per model; periodic ~100f stereo rescaling; per-paper semantic mirroring on CRCD 4 classes; missing configs (e3_005/c3_001/g3_001) are the agent's job.

# Agent Run Book — SGS-SLAM (Semantic Gaussian Splatting SLAM)

> **Document class:** Standing orders for a fresh autonomous agent. You will be handed (a) the SGS-SLAM paper, (b) the official SGS-SLAM repo, and (c) this DDS-SLAM repo (the shared CRCD harness) **later**. Do not assume you already have them. Read this entire document before touching a keyboard. Where this document and a handed-over artifact disagree on a load-bearing fact, **STOP and escalate** (Section 0.2) — do not guess.
>
> **Two repos, two roles.** "SGS-SLAM repo" = the method under test (clone of `github.com/ShuhongLL/SGS-SLAM`). "DDS-SLAM repo" = THIS repo (`/content/DDS-SLAM`), used **only** as the shared CRCD preprocessing + evaluation + video harness. SGS-SLAM's own inline eval is used **only** for Phase A (own-dataset repro); CRCD metrics in Phase B come from the DDS-SLAM `Addons/eval/*` harness, exactly as every other method in the benchmark.
>
> **Important note on verified facts.** Every concrete file path, config field, npz key, function name, and command in this book was verified against a clone of the SGS-SLAM repo (`scripts/slam.py`, `datasets/gradslam_datasets/{basedataset,replica}.py`, `configs/replica/slam.py`, `configs/data/replica.yaml`, `utils/{common_utils,slam_external,eval_helpers}.py`) and against this DDS-SLAM repo's `Addons/`. The pinned SGS-SLAM commit you build MUST match these. If any field/key below is absent in the commit you are handed, **STOP and escalate** — do not improvise.

---

## 0. Mission & non-negotiables

### 0.1 Mission
Produce, for SGS-SLAM, a reproducible two-phase result set for a **published** benchmarking paper:

- **PHASE A** — reproduce SGS-SLAM on its **own** dataset (Replica, with GT semantics), evaluated **exactly as the SGS-SLAM paper does**, until a **DECISION GATE** passes.
- **PHASE B** — run SGS-SLAM on the **5 fixed CRCD snippets** (`C_1/001`, `E_3/005`, `C_3/001`, `G_3/001`, `C_2/001`), adapting CRCD to SGS-SLAM's RGB-D + GT-semantic input contract, and evaluate with the **fixed CRCD metric set** (ATE-sim3, PSNR, SSIM, LPIPS, Depth-L1) + the canonical 6-panel video, via the DDS-SLAM harness.

Everything must be driven by a single `run_sgsslam.sh` line (Section 7) with per-stage logging, failure isolation, and resumability.

### 0.2 Non-negotiables (precision; NO ASSUMPTIONS)
1. **No silent guessing on load-bearing facts.** If a fact is unknown, a choice changes results, or a handed-over artifact contradicts this book, **STOP** and escalate.
2. **Clarification Protocol.** To escalate: (a) write a line to `/content/drive/MyDrive/Outputs/sgsslam/_escalations.log` in the form `ESCALATE [<phase>] <question> | options: A=... B=... | blocking=yes|no`; (b) if `blocking=yes`, exit the affected stage with the reserved code **`42`** (Section 7) and do **not** fabricate a downstream result; (c) if `blocking=no`, record the assumption you are *temporarily* parking, continue, and surface it in the final deliverable's "Open questions" (Section 10). The pre-known open questions are already enumerated in Section 10 — treat any **new** ambiguity the same way.
   - **blocking vs non-blocking is a binary, decided up front.** An item is `blocking=yes` ONLY if no defensible default exists and proceeding would corrupt a headline result. An item with a stated safe interim default is `blocking=no`: you park it, continue, and surface it. Never both escalate-blocking AND continue on the same item — that is a protocol violation. Section 10 fixes the classification of every known open question; honor it.
3. **Failure isolation.** One method/snippet/stage failing must never abort the rest. Catch, log to `_failures.log`, continue.
4. **Reproduce-the-paper bar is tied to THIS paper's reporting granularity** (Section 2.6 / Section 4.5). Do not invent a stricter or looser bar.
5. **VRAM gate (this is a 3DGS method — strict).** SGS-SLAM is VRAM-heavy 3D Gaussian Splatting. Detect GPU via `nvidia-smi`. See Section 3.4 for the **empirically grounded, resolution-aware** gate (the headline floor is not a single number — it depends on snippet length and resolution). Record GPU name + VRAM at the top of every stage log.
6. **Determinism.** Seed everything via SGS-SLAM's **native `seed` config field** (Section 8.1); never edit method-logic source to inject a seed. The DDS-SLAM `run_cell.sh` override-YAML pattern does **NOT** apply to SGS-SLAM (it has no YAML/`inherit_from` run-config); SGS-SLAM configs are Python dicts with a top-level `seed` key.
7. **Outputs location is fixed:** everything ships to `/content/drive/MyDrive/Outputs/sgsslam/...`. CRCD raw must be **copied F:/ → Drive → /content** first (Colab cannot read `F:/`).

---

## 1. Orientation / required reading (do this first, in order)

### 1.1 Shared harness docs (AUTHORITATIVE — read first)
- **`CONTRACT.md` and `00_COMMON.md` both EXIST** at `benchmarking/runbooks/` and are AUTHORITATIVE. (A previous version of this book wrongly claimed 00_COMMON was absent — **corrected 2026-06-17**.) **Read [`CONTRACT.md`](CONTRACT.md) in full FIRST:** it governs the CLI signature, output layout, artifact filenames, exit codes, and aggregation, and **SUPERSEDES Section 7 of this document** wherever they differ — notably escalate exit code `42`→**`20`**, the calib path (`cam_calib/`, not `_calib/`), and the render-metric filenames (`render_eval.{csv,txt}`, not `render_metrics.txt`). Then read [`00_COMMON.md`](00_COMMON.md) §0 (resolved decisions) + the metric law. Where this book disagrees with CONTRACT.md on plumbing, CONTRACT.md wins; where they disagree on a load-bearing scientific fact, STOP and escalate (Section 0.2, code **20**).

### 1.2 SGS-SLAM paper (read these sections, in this order)
- **Abstract + Method** — confirm the per-Gaussian "semantic color" channel, the second rasterization pass (`render_mode='semantic_color'`), and the joint color+depth+semantic loss. Note the **tracking seg-loss weight 0.05** and **mapping seg-loss weight 0.1** (matches `configs/replica/slam.py`: `tracking.loss_weights.seg=0.05`, `mapping.loss_weights.seg=0.1`).
- **Experiments → Replica tables**: Table 1 (per-scene PSNR/SSIM/LPIPS + **Avg**, **training-view** rendering), Table 2 (per-scene ATE RMSE + Depth-L1 + Avg, **units = cm**), Table 3 (mIoU on a Replica subset).
- **Limitations** — quote it in your final report: *"SGS-SLAM relies on depth and 2D semantic signal inputs for tracking and mapping. In scenarios where this information is scarce or difficult to access, the system's effectiveness will be compromised."* This is the justification for the CRCD-adaptation choices and the deformable-tissue caveat.

### 1.3 SGS-SLAM repo (read before building — verify these exact facts)
- `environment.yml` (authoritative pins) **and** `README.md` (note the README↔env mismatches; Section 3).
- `configs/replica/slam.py` — a **Python dict** named `config` (NOT a YAML). Top-level `seed`, `scene_name`, `map_every`, `keyframe_every`, `tracking_iters`, `mapping_iters`; nested `data.{basedir,gradslam_data_cfg,sequence,desired_image_height=680,desired_image_width=1200,start,end,stride,num_frames,load_semantics=True,num_semantic_classes=101}`; `scene_radius_depth_ratio=3`, `mean_sq_dist_method`, `tracking.loss_weights.{im,depth,seg}`, `mapping.loss_weights.{im,depth,seg}`, `mapping.pruning_dict`, `mapping.densify_dict`. **There is NO `mapping.bound`, NO `marching_cubes_bound`, NO `inherit_from`, NO `timesteps`, NO `tcnn_encoding`/`lr_embed`/`lr_decoder`** — those are DDS-SLAM neural-implicit fields and DO NOT EXIST here.
- `configs/data/replica.yaml` — the **intrinsics + depth-scale** live here: `camera_params.{image_height,image_width,fx,fy,cx,cy,png_depth_scale=6553.5,crop_edge}`. `dataset_name: 'replica'` selects the loader.
- `datasets/gradslam_datasets/replica.py::ReplicaDataset.get_filepaths()` — globs `frames/frame*.jpg`, `depths/depth*.png`, **`semantic_ids/semantic_id*.png`**, **`semantic_colors/semantic_color*.png`** (note the filename prefixes `semantic_id` and `semantic_color`, not bare `NNNNNN.png`). `load_poses()` reads `traj.txt` (4×4 row-major c2w per line).
- `datasets/gradslam_datasets/basedataset.py` — verified depth contract: `_preprocess_depth` returns `depth / self.png_depth_scale` where `png_depth_scale = config_dict["camera_params"]["png_depth_scale"]`. `_preprocess_semantic_id` resizes with **`cv2.INTER_NEAREST`** and `__getitem__` loads it as `np.int64`. `_preprocess_semantic_color` resizes NEAREST and (if `normalize_color`) divides by 255. **There is NO `sc_factor` field anywhere in SGS-SLAM** (sc_factor is a DDS-SLAM `datasets/dataset.py` concept only).
- `scripts/slam.py` — `get_dataset(config_dict, ...)` dispatches on `config_dict["dataset_name"].lower()` (a chain of `elif`s: `replica`, `scannet`, `tum`, ...). The final `eval(...)` call (≈line 1087) passes `eval_every=config['eval_every'], save_frames=True`.
- `utils/eval_helpers.py::eval(...)` — the Table-1 metric loop is `for time_idx in range(num_frames): if time_idx != 0 and (time_idx+1) % eval_every != 0: continue`. **So the reported average is over a STRIDED subset when `eval_every>1`** (Replica default `eval_every=5`). Renders are written by `save_frames=True` as `gs_{time_idx:04d}.png` into `eval/rendered_rgb/`, `eval/rendered_depth/` (JET colormap, NOT raw metric), `eval/rendered_sem/`.
- `utils/common_utils.py::save_params` — writes `params.npz` (and `params.ply`). Pose keys: `cam_unnorm_rots` shape `(1,4,num_frames)` (quaternion, real-part-first `[w,x,y,z]`, unnormalized), `cam_trans` shape `(1,3,num_frames)`. These are **relative w2c to the first camera frame** (world = first camera).
- `utils/slam_external.py::build_rotation(q)` — confirms quaternion order `q=[r,x,y,z]` = `[w,x,y,z]`.
- The diff-gaussian-rasterization-w-depth submodule pin: commit `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110` (JonathonLuiten).

### 1.4 DDS-SLAM harness files you WILL invoke (read their headers)
- `Addons/preprocess/preprocess_crcd_published.py` — rectifies CRCD-Published. **Requires `--calib_pkl`** (the ECM_STEREO L2R rectification pickle; keys `ecm_map_left_x/y`, `ecm_map_right_x/y` consumed by `load_stereo_maps`). Emits `video_frames/{l,r}.png`, `masks/` (binary tool mask), `semantic_class/NNNNNN.png` (uint8 = `np.clip(sem_rect,0,255)`, NEAREST-rectified, value = `coco_id+1`), `groundtruth.txt` (TUM), `rectified_calib.txt` (space-separated `key val`, keys `fx fy cx cy baseline_m ...`).
- `Addons/depth/generate_depth_moge.py` — MoGe-2 metric depth → `<fid>-left_depth.npy` (float32, value = `depth_m * depth_scale`). **No PNG output.**
- `Addons/depth/moge_npy_to_png.py` — the npy→uint16-PNG converter (`--in <npy_dir> --out <png_dir>`); writes `<fid>.png` = `clip(npy,0,65535).uint16`. **Call this by name** — the run book's depth step is `generate_depth_moge.py` then `moge_npy_to_png.py`.
- `Addons/colab/run_crcd_4snippets.sh` — **the structural template** for the whole CRCD loop (env, stage, rectify, MoGe depth, stereo `sc_factor`, GT sanity, SLAM, render eval, trajectory metrics, Drive ship, combined table). Your `run_sgsslam.sh` mirrors its phase skeleton.
- `Addons/colab/crcd_depth_gen_remainder_20260616.sh` — depth-only batch. (a) Its `SNIPPETS=(...)` table is the **authoritative source for frame counts**: `E3_005=265`, `C3_001=1527`, `G3_001=1987`. (b) **Lines ~115–141 contain the AUTHORITATIVE stereo-SGBM `sc_factor` block** (inline heredoc). This — NOT `generate_depth_stereo.py` — is the sc_factor logic referenced throughout this book. It is reproduced verbatim in Section 5.3.
- `Addons/eval/sim3_ate.py` — `load_est` parses **12 floats (3×4 row-major c2w) or 16 (4×4)** per line, translation = cols `[3,7,11]`. Pairs `est[i] ↔ GT[i]` by line order; resamples with a warning if counts differ.
- `Addons/eval/eval_rendering.py` — `PAPER_REFERENCES` **already contains** `'CRCD (C1_001)'`, `'(C2_001)'`, `'(C3_001)'`, `'(E3_005)'`, `'(G3_001)'` (all `None` refs). **No harness edit is needed**; pass any of these 5 keys via `--sequence` directly.
- `Addons/eval/aggregate_crcd_generic.py` — **the correct, method-agnostic cross-snippet + cross-method aggregator.** Reads `<ROOT>/<NAME>/{render_eval.csv|render_eval.txt, sim3_metrics.txt}` (or `payload.tgz`). Supports `--root` (single method) and `--method-roots` (master table). **Use this, NOT `aggregate_ab.py`** (which is hardcoded to the DDS-SLAM base/geo/dino A/B/C study, globs `<cell>_s<seed>/payload.tgz`, and parses `render_metrics.txt` — wrong layout for this run book).
- `Addons/viz/generate_video.py`, `Addons/env/colab_setup.sh`.
- The DDS-SLAM `configs/CRCD/*.yaml` are **NICE-SLAM/HashGrid neural-implicit configs** (`inherit_from`, `mapping.bound`, `marching_cubes_bound`, `tcnn_encoding`, `dataset: 'stereomis'`). **They are NOT usable as SGS-SLAM configs and must NOT be copied into SGS-SLAM.** Their per-snippet depth-range numbers are useful ONLY as a sanity reference for the metric depth range (Section 5.5).

---

## 2. Method dossier

### 2.1 One-paragraph summary
SGS-SLAM (ECCV 2024) is the **first semantic 3D-Gaussian-Splatting dense RGB-D SLAM**, built on the **SplaTAM** lineage. It adds a per-Gaussian **"semantic color"** property: each semantic class is encoded as a fixed RGB color, and the per-Gaussian semantic color is rendered through the **same** standard `diff-gaussian-rasterization-w-depth` rasterizer in a **second pass** (`render_mode='semantic_color'`) — **no custom extra-channel rasterizer compile is required**. It jointly optimizes appearance, depth, and semantic losses for tracking + mapping, producing dense reconstruction and 3D semantic segmentation.

### 2.2 Method class
3D Gaussian Splatting (3DGS) dense **RGB-D** SLAM. Multi-channel per-Gaussian optimization. **Static-scene assumption; no deformation model; no monocular fallback** (depth is a mandatory input, loss weight 1.0 in both tracking and mapping).

### 2.3 Own datasets + how to obtain
| Dataset | Obtain | Gating | Sequences |
|---|---|---|---|
| **Replica (w/ GT semantic masks)** | Authors' Dropbox (pre-packaged with `semantic_ids/` + `semantic_colors/`) → default `./data/Replica`. URL in repo README; canonical link: `https://www.dropbox.com/scl/fo/a93xhcpsteumsmw8oq4jc/ALD5oq6MfkKTpT_7K5cDqhQ?rlkey=hblzvi1m9pcqmksgzs9ydwdxp&dl=0` | Free | Room0-2, Office0-4 (8, headline) |
| ScanNet | `preprocess/scannet/run.py` after access request | **ToS-gated** | scene0000/0059/0106/0181/0207 |
| ScanNet++ | undistort per their tooling, SplaTAM split | **ToS-gated** | 8b5caf3398, b20a261fdf |

> **CRITICAL:** the vanilla Replica / iMAP / SplaTAM Replica downloads **do NOT** contain the `semantic_ids/` + `semantic_colors/` folders SGS-SLAM needs. Use the **authors' Dropbox** build. Plan the repro around **Replica only** (the other two are access-gated).

### 2.4 Pretrained weights
**None.** SGS-SLAM trains per-scene Gaussians from scratch; semantics come from GT label maps, not a learned segmentation model. No license gating on the method side. (CRCD-side externals — MoGe-2 for depth — belong to the DDS-SLAM depth harness, not SGS-SLAM.)

### 2.5 Semantic-input requirement (THIS method)
SGS-SLAM **REQUIRES dense 2D semantic GT label maps as input**, consumed during both tracking and mapping. The loader reads two per-frame folders: **`semantic_ids/semantic_id*.png`** (integer class-id PNGs, loaded as `int64`) and **`semantic_colors/semantic_color*.png`** (fixed-palette color PNGs). Replica config sets `num_semantic_classes=101`.

**Good news for CRCD:** CRCD-Published ships per-pixel `semantic_instance/` masks (uint16, pixel = `coco_id + 1`) with a known **4-class** scheme `{0=bg, 1=Liver, 2=Gallbladder, 3=Tool}` per `info_semantic.json`; `preprocess_crcd_published.py` rectifies these (NEAREST) and emits `semantic_class/NNNNNN.png`. So SGS-SLAM's GT-semantic requirement can be satisfied **directly from CRCD GT** — **no DINO / pretrained seg head / predicted labels** required. Adaptation work (Section 5.4): a script that emits `semantic_ids/` + `semantic_colors/`, plus a CRCD loader subclass with **`num_semantic_classes=4`** (NOT 101). **ESCALATE** if any of the 5 snippets lacks `semantic_instance/` masks at rectified resolution, or if the on-disk values are not exactly `{0,1,2,3}` (Section 5.4, Section 10 Q1/Q12).

### 2.6 How the paper REPORTS results (granularity → acceptance bar)
- **Headline = Replica, per-scene with an `Avg.` column.** Table 1 = per-scene rendering (**training views**), Table 2 = per-scene ATE RMSE + Depth-L1 (**cm**), Table 3 = mIoU on a subset.
- **Faithful-repro bar for THIS paper:** Replica is per-scene → **matching 1–2 representative Replica scenes within tolerance is acceptable** (you do **not** need all 8 unless you are reproducing the full `Avg.`).
- **Headline Replica `Avg.` reference numbers** (for the GATE in Section 4.5): ATE RMSE **0.41 cm**, Depth-L1 **0.36 cm**, PSNR **34.66 dB**, SSIM **0.973**, LPIPS **0.096**, mIoU **92.72%**. ScanNet `Avg` ATE RMSE ≈ 9.87 cm (secondary).
- **Rendering metrics are TRAINING-view** (Table 1 caption), not held-out novel views. `eval_novel_view.py`/`eval_nvs` is a separate, secondary protocol — do **not** use it for the GATE.
- **Frame set for the average (BLOCKING prerequisite of the gate — Q8):** `eval()` skips frames per `eval_every` (Replica default 5 ⇒ strided average). Before claiming a faithful Table-1 PSNR match you MUST determine whether the paper's reported average is over the strided eval set or every frame. Read `utils/eval_helpers.py::eval` + how the paper's table was generated. This is **blocking** for the gate (Section 4.5): a "PASS" computed over the wrong frame set is not a real reproduction.

---

## 3. Environment setup on Colab

> SGS-SLAM is built **inside its own clone with its own env** — it is **NOT** the DDS-SLAM modern stack. Build it separately. The DDS-SLAM `Addons/eval` harness needs only Python+numpy+opencv+`lpips`, satisfiable in either env.

### 3.1 Pins (resolve the repo's internal mismatch FIRST)
The repo disagrees with itself: README prose says **py3.9 / CUDA 11.8**; `environment.yml` pins **py3.10 / cudatoolkit 11.6 / torch 1.12.1 / torchvision 0.13.1 / torchaudio 0.12.1**. **Prefer `environment.yml`.** The rasterizer must be built against the **same CUDA that torch links** — that is the single binding constraint.

- python **3.10**, torch **1.12.1**, cudatoolkit **11.6** (per `environment.yml`).
- If the legacy stack fails to build against the Colab-assigned GPU/driver, escalate (Section 10, Q6) before improvising a torch bump — a torch/CUDA change is load-bearing for the rasterizer.

### 3.2 Build steps
```bash
git clone --recursive https://github.com/ShuhongLL/SGS-SLAM.git /content/SGS-SLAM
cd /content/SGS-SLAM
# verify the rasterizer submodule pin:
git -C submodules/diff-gaussian-rasterization-w-depth rev-parse HEAD
#   expect cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110
conda env create -f environment.yml        # or micromamba; py3.10 / cu116 / torch 1.12.1
conda activate sgs-slam                     # use the env name from environment.yml
pip install -r requirements.txt
# Build the depth-aware RGB rasterizer against the assigned GPU arch:
export TORCH_CUDA_ARCH_LIST="8.0"           # A100 sm_80; T4 -> "7.5"; detect first (3.4)
pip install submodules/diff-gaussian-rasterization-w-depth
```
**Key deps** (from `environment.yml` / requirements): `diff-gaussian-rasterization-w-depth` (CUDA build), `open3d==0.16.0`, `faiss-gpu`, `kornia`, `opencv`, `lpips`, `torchmetrics`, `plyfile`, `natsort`, `imageio`, `wandb`, `cyclonedds`.

### 3.3 Dependency landmines
1. **The one real landmine:** compiling `diff-gaussian-rasterization-w-depth` against the installed CUDA — needs `nvcc`, matching CUDA, and either a GPU at build time or `TORCH_CUDA_ARCH_LIST` set. This is the **standard depth-aware RGB rasterizer** (SplaTAM lineage), **NOT** a custom semantic-channel rasterizer — **no special semantic build**.
2. Legacy torch 1.12.1 / cu11.6 may conflict with Colab's current driver. Detect GPU arch first (3.4) and set `TORCH_CUDA_ARCH_LIST` accordingly: **A100 → `8.0`**, **T4 → `7.5`**.
3. `wandb` will prompt for login — run it **offline** (`WANDB_MODE=offline`) so it never blocks an autonomous run. (The config default is `use_wandb=True`; either set `WANDB_MODE=offline` or set `use_wandb=False` in your config copy.)

### 3.4 GPU detection + VRAM gate (run at the top of every phase)
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
python -c "import torch;p=torch.cuda.get_device_properties(0);print(p.name,round(p.total_memory/1024**3,1),'GB','sm_%d%d'%(p.major,p.minor))"
```
**Resolution-aware VRAM policy (reconciled with the real footprint).** SGS-SLAM is dense 3DGS: VRAM scales with the number of live Gaussians, which scales with frame count × resolution × scene complexity. Empirical expectation for surgical CRCD content:

| Snippet | Frames | Full-res (1280×720) peak VRAM (expected) | Verdict |
|---|---|---|---|
| C1_001 (360), E3_005 (265), C2_001 (730) | ≤730 | ~10–16 GB | T4-16GB **feasible at full res** (monitor; reduce densification if it creeps past 15 GB) |
| C3_001 (1527), G3_001 (1987) | ~1500–2000 | **>16 GB likely (≈18–28 GB)** | **A100 REQUIRED at full res.** On T4: either downscale (below) or **escalate** |

Policy:
1. **Detect VRAM.** If **< 12 GB → ABORT+escalate (code 42)** for all snippets (SGS-SLAM cannot run dense 3DGS reliably below this).
2. **12–16 GB (e.g. T4):** the 3 short snippets (≤730 frames) may run at full res with monitoring. For the **2 long snippets (C3_001, G3_001) on a ≤16 GB GPU**, you MUST either (a) **escalate (blocking) to request an A100** for those two, OR (b) apply the **documented T4 fallback** below AND record in the deliverable that those two snippets used reduced settings (render-metric comparability caveat).
3. **A100-40GB:** recommended for all 5; run everything at full res.
4. Set `TORCH_CUDA_ARCH_LIST` from `compute_cap` (sm_80 / sm_75) for the rasterizer build.

**Documented T4 fallback for the long snippets (only if A100 is unavailable AND the operator waives full-res comparability):** in the snippet config set `data.desired_image_height=384`, `data.desired_image_width=683` (half-res, preserves aspect), and reduce Gaussian growth: `mapping.pruning_dict.removal_opacity_threshold` ↑ to `0.05` and `mapping.mapping_window_size` ↓ to `16`. **Downscaling changes the effective intrinsics** — the loader rescales fx/fy/cx/cy automatically from the data-yaml base intrinsics to `desired_*`, but you MUST note in the report that render PSNR/SSIM/LPIPS for those snippets are at reduced resolution and therefore **not directly comparable** to full-res snippets/other methods. Log the chosen resolution in `gpu_runtime.json`.

### 3.5 Verification smoke test (must pass before Phase A)
```bash
python -c "import torch, diff_gaussian_rasterization; print('rasterizer OK', torch.__version__, torch.version.cuda)"
python -c "import lpips, open3d, faiss, kornia, torchmetrics; print('deps OK')"
# tiny end-to-end: 1 Replica scene, ~20 frames, ~1 min — proves data loader + both rasterizer passes + seg loss wire up
python scripts/slam.py configs/replica/slam_smoke.py   # see frame-cap mechanism below
```
**Frame-cap mechanism (SGS-SLAM-native — NOT the DDS-SLAM override-YAML).** SGS-SLAM has no override-config system. To cap frames for the smoke test, copy `configs/replica/slam.py` → `configs/replica/slam_smoke.py` and set `data.num_frames=20` (or `data.end=20`). These are real fields in the `config['data']` dict (`num_frames=-1` means all; `end` is a slice bound). Do **not** attempt a DDS-SLAM `inherit_from`/override YAML — it does not exist for SGS-SLAM.

A green smoke test = rasterizer built, deps import, the **semantic second pass** runs without a missing-folder error. If the smoke run errors on `semantic_ids/`/`semantic_colors/`, the Dropbox-semantic Replica build is wrong (Section 2.3).

---

## 4. PHASE A — Reproduce on SGS-SLAM's OWN dataset (Replica)

### 4.1 Obtain data + weights
- Download **Replica-with-GT-semantics** from the authors' Dropbox into `/content/SGS-SLAM/data/Replica` (cache the tarball to `/content/drive/MyDrive/Datasets/SGS-SLAM-Replica/` so re-runs rehydrate from Drive, not the network).
- Weights: **none** (Section 2.4).
- **Verify** each chosen scene contains `frames/` (RGB), `depths/`, `traj.txt`, **`semantic_ids/`** **and** `semantic_colors/`. Missing semantic folders ⇒ wrong Replica build ⇒ STOP (do not run vanilla Replica).

### 4.2 Choose scenes (bar = per-scene; pick representatives)
Run **`room0` and `office0`** as the two representative scenes (one room-class, one office-class). This satisfies the per-scene bar (Section 2.6) without committing to all 8. If the user wants the full `Avg.`, run all 8 and report the average.

### 4.3 Configs (per-scene — there are NO shipped per-scene .py files)
**Verified:** the repo ships **only `configs/replica/slam.py`** (a base config with `scene_name="room0"` hardcoded at the top). There is **no `room0.py`/`office0.py`**. To run a specific scene you must produce a per-scene config copy:
```bash
# author a per-scene config by copying the base and editing scene_name (one line)
cp configs/replica/slam.py configs/replica/slam_room0.py     # edit: scene_name = "room0"
cp configs/replica/slam.py configs/replica/slam_office0.py   # edit: scene_name = "office0"
```
`scene_name` flows into `data.sequence`, `run_name`, and the loader's `input_folder = basedir/sequence`. Keep `num_semantic_classes=101`, `load_semantics=True`, seg-loss weights (track 0.05 / map 0.1), `scene_radius_depth_ratio=3` **as shipped** — Phase A is a *faithful* repro, so do not retune. Set the seed via the config `seed` field (Section 8.1). For the **faithful Table-1 average**, set `eval_every` to whatever the paper's table uses (Q8, blocking — Section 2.6); for per-frame render dumps used in diagnostics set `eval_every=1`.

### 4.4 EXACT run + eval commands
```bash
cd /content/SGS-SLAM && conda activate sgs-slam
export WANDB_MODE=offline PYTHONHASHSEED=0
for SCENE in room0 office0; do
  python scripts/slam.py configs/replica/slam_${SCENE}.py 2>&1 | tee /content/drive/MyDrive/Outputs/sgsslam/phaseA/${SCENE}/slam.log
done
```
- ATE RMSE / Depth-L1 / PSNR / SSIM / LPIPS are produced **inline** by `utils/eval_helpers.py::eval` during/after SLAM (writes `eval/psnr.txt` etc.). mIoU comes from the semantic eval path (`evaluate_miou`/`evaluate_label_miou`).
- **Rendering metrics MUST be on TRAINING views** (Section 2.6) — `eval()` evaluates training frames; do **not** call `eval_nvs`/`scripts/eval_novel_view.py` for the GATE.
- **Before claiming a faithful PSNR match**, confirm the exact frame set the Table-1 average uses (every frame vs `eval_every`-strided) — **Q8 is BLOCKING for the gate** (Section 2.6 / 4.5).

### 4.5 Acceptance bar + DECISION GATE
Bar is per-scene (Section 2.6). For each chosen scene, the reproduced numbers must land within tolerance of the paper's **per-scene** values (read them from Table 1/2 for that exact scene). Use these **tolerances** (sanity-gate, escalate if exceeded):

| Metric | Tolerance vs paper per-scene |
|---|---|
| ATE RMSE | within **±0.3 cm** (or ≤ 2× paper value, whichever is looser at sub-cm) |
| PSNR | within **±1.0 dB** |
| SSIM | within **±0.01** |
| LPIPS | within **±0.02** |
| Depth-L1 | within **±0.3 cm** |
| mIoU | within **±3 %** |

**GATE — must pass to proceed to Phase B:**
- [ ] **Q8 resolved (BLOCKING):** the frame set used for the reproduced rendering average matches the paper's (else the PASS is not real). Resolve before evaluating, not after.
- [ ] Both representative scenes ran to completion (no crash; full frame count).
- [ ] ATE, PSNR, SSIM, LPIPS, Depth-L1, mIoU each within tolerance for **both** scenes (or, if reproducing `Avg.`, the 8-scene average within tolerance of the headline numbers in Section 2.6).
- [ ] Rendering metrics confirmed computed on **training views**, over the paper's frame set.
- [ ] Seed + GPU + runtime logged; determinism check (Section 8.1) shows two seeded re-runs agree within numerical noise.

**If the GATE fails:** STOP, write `ESCALATE [phaseA] gate fail: <metric> reproduced=<x> paper=<y> | blocking=yes` to `_escalations.log`, exit code 42. Do **not** proceed to CRCD on a failed repro.

### 4.6 Artifacts to record (under `/content/drive/MyDrive/Outputs/sgsslam/phaseA/<scene>/`)
`slam.log`, the inline metrics dump (`eval/psnr.txt` and ATE/Depth-L1/mIoU console values), the `params.npz`/`params.ply` reconstruction (or a pointer + size if too large for Drive), the config used, `seed.txt`, `gpu_runtime.json` (GPU name, VRAM, wall-time, peak VRAM), and a `gate.json` (`{scene, metric, reproduced, paper, within_tol, eval_every, frame_set}`).

---

## 5. PHASE B — CRCD adaptation

> **Gate dependency:** Phase B for SGS-SLAM only runs after the Phase-A GATE passes (or the user explicitly waives it).

### 5.1 The 5 target snippets (fixed)
`C_1/001`, `E_3/005`, `C_3/001`, `G_3/001`, `C_2/001`. DDS-SLAM config-naming maps these to stems `c1_001`, `e3_005`, `c3_001`, `g3_001`, `c2_001`; staged data dirs are CAPS-no-slash `C1_001`, `E3_005`, `C3_001`, `G3_001`, `C2_001`. **Frame counts** (authoritative from the depth-gen table): C1_001 = 360, **E3_005 = 265**, **C3_001 = 1527**, **G3_001 = 1987**, C2_001 = 730.

> **CONFIG NOTE.** SGS-SLAM configs are Python dicts (Section 5.5). You author **one CRCD `.py` config per snippet** plus **one shared CRCD data-yaml** (intrinsics + depth scale). There is no `mapping.bound`/`marching_cubes_bound` to derive — that is the DDS-SLAM neural-implicit config system and **does not apply to SGS-SLAM**. The surgical close-range geometry is handled by SplaTAM's `scene_radius_depth_ratio` (Section 5.5), not a 3-axis bound box.

### 5.2 F:/ → Drive → /content staging
CRCD lives on the user's local **F:/** drive; Colab cannot read F:/. Required flow:
1. **(User/operator step, off-Colab):** copy CRCD-Published from `F:/Datasets/CRCD-Published` to `/content/drive/MyDrive/Datasets/CRCD-Published/`. **Staging manifest (ALL of these must be present per snippet):**
   - `<EP>/snippet_<SID>/{rgb, rgbright, semantic_instance, groundtruth.txt, intrinsics.yaml}`
   - **The ECM_STEREO L2R rectification pickle** (required by `preprocess_crcd_published.py --calib_pkl`). On the operator's machine it lives at `C:/Users/benli/sam3facebook/cam_cali/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`. **Stage it to a fixed Drive path: `/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`.** Its keys (`ecm_map_left_x`, `ecm_map_left_y`, `ecm_map_right_x`, `ecm_map_right_y`) are exactly what `load_stereo_maps` reads.
   - **Pre-flight check (BLOCKING):** if the calib pickle OR any snippet's `semantic_instance/` is absent, escalate (`blocking=yes`, code 42) — rectification and every downstream step cannot run without them. Do not fabricate.
2. **Stage to /content** per snippet exactly as `run_crcd_4snippets.sh` Phase 1: prefer a per-snippet tarball (≈3–5 min) over per-item FUSE cp (≈78 min/8 GB). Verify `N_RGB ≥ frames`.
3. **Rectify** with:
   ```bash
   python Addons/preprocess/preprocess_crcd_published.py \
     --snippet_dir /content/CRCD-Published/<EP>/snippet_<SID> \
     --calib_pkl   /content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl \
     --output_dir  data/CRCD/<NAME>
   ```
   → produces `video_frames/{l,r}.png`, `masks/`, `semantic_class/NNNNNN.png`, `groundtruth.txt`, `rectified_calib.txt`. Verify `rectified_calib.txt` exists and left/right/GT counts match.

### 5.3 Depth: MoGe-2 + stereo-scaled (the CRCD depth contract)
SGS-SLAM is **hard RGB-D** with depth-loss weight 1.0 — **depth quality directly drives ATE.** CRCD ships no usable depth; generate it exactly as the DDS-SLAM pipeline:
1. Symlink rectified left frames as MoGe inputs: `_moge_in/<fid>-left.png`.
2. **MoGe-2 metric depth** → `.npy`:
   ```bash
   python Addons/depth/generate_depth_moge.py --rgb _moge_in --out _moge_npy \
     --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0 --model_id Ruicheng/moge-2-vitl
   ```
   (`temporal_window 1` = smoothing off for CRCD; metric-direct, no `--ref`.)
3. **`.npy → uint16 PNG`** via the named converter:
   ```bash
   python Addons/depth/moge_npy_to_png.py --in _moge_npy --out data/CRCD/<NAME>/depth
   ```
   writes `depth/<fid>.png = clip(npy,0,65535).uint16` (value = depth_m × 10000).
4. **Stereo anchor scale (the AUTHORITATIVE frame-0 SGBM `sc_factor` block — verbatim from `crcd_depth_gen_remainder_20260616.sh` lines ~115–141; NOT `generate_depth_stereo.py`, which is RAFT-based and parses a different calib format):**
   ```python
   import cv2, numpy as np, os, sys, glob
   STAGED=sys.argv[1]
   calib={}
   for line in open(f'{STAGED}/rectified_calib.txt'):
       k,v=line.strip().split(); calib[k]=float(v)
   baseline_m=calib['baseline_m']; fx=calib['fx']
   left_path=sorted(glob.glob(f'{STAGED}/video_frames/*l.png'))[0]
   fid=os.path.basename(left_path).replace('l.png','')
   left=cv2.imread(left_path,cv2.IMREAD_GRAYSCALE)
   right=cv2.imread(f'{STAGED}/video_frames/{fid}r.png',cv2.IMREAD_GRAYSCALE)
   moge=cv2.imread(f'{STAGED}/depth/{fid}.png',cv2.IMREAD_UNCHANGED)
   if left is None or right is None or moge is None: print('FATAL read frame0',file=sys.stderr); sys.exit(1)
   moge_m=moge.astype(np.float32)/10000.0; mvalid=moge_m>0.01
   sgbm=cv2.StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=7,P1=8*49,P2=32*49,
       disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=32,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
   disp=sgbm.compute(left,right).astype(np.float32)/16.0; vs=disp>0.5
   if vs.sum()<1000: print('FATAL few stereo px',file=sys.stderr); sys.exit(2)
   sd=np.zeros_like(disp); sd[vs]=baseline_m*fx/disp[vs]
   vj=vs&(sd>0.05)&(sd<3.0)&mvalid
   if vj.sum()<500: print('FATAL few joint px',file=sys.stderr); sys.exit(2)
   sc=float(np.median(sd[vj]/moge_m[vj]))
   print(f'  stereo median={np.median(sd[vj]):.3f}m MoGe median={np.median(moge_m[vj]):.3f}m -> sc_factor={sc:.4f}')
   open(f'{STAGED}/.sc_factor','w').write(f'{sc:.6f}\n')
   ```
   Save the frame-0 SGBM depth too (you will reuse it as the **Depth-L1 reference**, Section 6.3): write `stereo_ref/<fid>.npy` (metres, NaN where invalid) for every frame where you compute SGBM (see cadence note below).
5. **Apply the scale — SGS-SLAM has NO `sc_factor` config field.** The DDS-SLAM "patch `data.sc_factor` into the config" step is a **no-op/error for SGS-SLAM and must NOT be used.** Instead, **bake the scale into the depth PNGs at generation time**: regenerate (or rescale in place) the depth PNGs so that `depth/<fid>.png = clip(depth_m * sc_factor * 10000, 0, 65535).uint16`. Concretely, multiply the MoGe `.npy` metres by `sc_factor` **before** `moge_npy_to_png.py`, OR re-write the PNGs as `clip(round(load(png).astype(f32) * sc_factor), 0, 65535).uint16`. Then the SGS-SLAM loader's `png_depth_scale=10000` (Section 5.6) recovers correctly-scaled metric depth, and `cam_trans`/ATE are in true metres. Record the applied `sc_factor` in the snippet's `config_used` header and `gpu_runtime.json`.
6. **Stereo cadence (Q2 — RESOLVED here, do not silently default).** The fixed global spec says "stereo-match ~every 100 frames." The repo's existing pipeline computes a **single frame-0 anchor**. This is a real methodology divergence. **Default for THIS run: compute the SGBM anchor every 100 frames** (frame 0, 100, 200, …), take the per-window `sc_factor`, and apply a single robust scale = **median of the per-window sc_factors** to the whole snippet (SGS-SLAM cannot consume a time-varying scale — a per-frame scale would break the static-Gaussian world frame). Save each window's SGBM depth into `stereo_ref/`. **If the per-window sc_factors disagree by >20%** (scene depth genuinely drifting), escalate (`blocking=yes`): a single global scale is unsafe and the operator must decide. If the operator has issued a written waiver to use frame-0-only, honor it and record the waiver reference. Do not proceed on a silent single-anchor default.
7. **Cache** depth PNGs + `.sc_factor` + `stereo_ref/` to `/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/` so re-runs rehydrate.

### 5.4 Semantic-input handling (satisfiable from CRCD GT — no DINO)
SGS-SLAM needs `semantic_ids/semantic_id*.png` + `semantic_colors/semantic_color*.png`. Two concrete deliverables:

**(a) Emit the two folders.** Extend `preprocess_crcd_published.py` (or add a small post-step script `Addons/preprocess/crcd_emit_sgsslam_semantics.py`) that, for each frame `i`, reads the rectified `semantic_class/NNNNNN.png` and writes:
- `data/CRCD/<NAME>/semantic_ids/semantic_id_<i>.png` — **exact copy** of `semantic_class/NNNNNN.png` (single-channel uint8 class id). Filename prefix `semantic_id_` and natsort-able index to match `ReplicaDataset.get_filepaths()` glob `semantic_id*.png`.
- `data/CRCD/<NAME>/semantic_colors/semantic_color_<i>.png` — the **4-color LUT** applied to the class map (3-channel uint8). Filename prefix `semantic_color_`.
- **4-color LUT (deterministic, matches the DDS-SLAM `generate_video.py --seg_classmap` palette; values are RGB):** `0=bg → (0,0,0)`, `1=Liver → (255,0,0)`, `2=Gallbladder → (0,255,0)`, `3=Tool → (0,0,255)`. Write PNGs in the same channel order the SGS-SLAM loader expects (it loads via `imageio.imread`; store as RGB).

**(b) Verify the on-disk value set (Q12 — escalate if not {0,1,2,3}).** `preprocess_crcd_published.py` writes `semantic_class` as `np.clip(sem_rect,0,255).uint8` where pixel = `coco_id+1`. The 4-class assumption holds ONLY if the actual coco_ids are `{0,1,2}` (→ on-disk `{0,1,2,3}` after the bg=0 / +1 convention). **Before exposing as `semantic_ids/`, run `np.unique` over `semantic_class/*.png` for all 5 snippets.** If the value set is exactly `{0,1,2,3}` → proceed with `num_semantic_classes=4`. If it contains other/non-contiguous ids → **remap to contiguous 0..3** via a fixed lookup AND escalate (`blocking=yes`) so the operator confirms the class mapping. Record the observed unique values in the deliverable.

**(c) CRCD loader subclass + registration (net-new code — exact files).**
- New file: `datasets/gradslam_datasets/crcd.py` containing `class CRCDDataset(GradSLAMDataset)`, modeled on `ReplicaDataset`. Override **only**:
  - `__init__`: set `self.input_folder = os.path.join(basedir, sequence)` and `self.pose_path = os.path.join(self.input_folder, "groundtruth.txt")`; call `super().__init__(...)` passing through `load_semantics`, `num_semantic_classes`, `desired_height/width`.
  - `get_filepaths()`: return `(color_paths, depth_paths, semantic_id_paths, semantic_color_paths, embedding_paths)` globbing `video_frames/*l.png` (color), `depth/*.png` (depth), `semantic_ids/semantic_id*.png`, `semantic_colors/semantic_color*.png`, `embedding_paths=None`. Use `natsorted`. **Assert all four non-empty lists are equal length** (basedataset raises if semantic counts ≠ color counts; assert here for a clearer error).
  - `load_poses()`: read `groundtruth.txt` (TUM `ts tx ty tz qx qy qz qw`), build a 4×4 **c2w** per row (`scipy` quaternion → R; translation), return a list of `torch.float` 4×4. (Replica's `load_poses` reads a 4×4-per-line `traj.txt`; CRCD GT is TUM, hence this override.)
- Registration (two edits — these are dataset-registration, NOT method-logic edits; see the note in 8.1 on the "no source edits" constraint):
  1. `datasets/gradslam_datasets/__init__.py`: add `from .crcd import CRCDDataset`.
  2. `scripts/slam.py::get_dataset`: add `elif config_dict["dataset_name"].lower() in ["crcd"]: return CRCDDataset(config_dict, basedir, sequence, **kwargs)` (and import `CRCDDataset` in the `from datasets.gradslam_datasets import (...)` block).
- Set `dataset_name: 'crcd'` in the CRCD data-yaml (Section 5.6) so `get_dataset` dispatches to your loader.
- **Resize discipline:** basedataset already resizes `semantic_id`/`semantic_color` with `INTER_NEAREST` (verified) — good, no fractional class ids. Depth uses bilinear in basedataset by default; for surgical depth that is acceptable, but if you observe class/edge artifacts, flag it (non-blocking diagnostic).

> **ESCALATE conditions (blocking):** (a) any of the 5 snippets lacks `semantic_instance/` masks; (b) `np.unique(semantic_class)` is not `{0,1,2,3}` after the +1 convention (Q12); (c) the benchmark owner has not confirmed `semantic_instance` (coco_id+1) as the agreed semantic source for SGS-SLAM (Q1). Do not substitute the DDS-SLAM **binary tool mask** or a DINO signal without explicit instruction — SGS-SLAM's contract is **dense multi-class GT**, which the 4-class `semantic_class` map satisfies and the binary tool mask does not.

### 5.5 Per-snippet config authoring (SGS-SLAM Python-dict schema)
For each of the 5 snippets author a `configs/crcd/<stem>.py` (copy `configs/replica/slam.py` and edit). **There is no bound box.** Set:

**Shared CRCD data-yaml** `configs/data/crcd.yaml` (intrinsics + depth scale, read from `rectified_calib.txt`; CRCD rectified is H=720, W=1280):
```yaml
dataset_name: 'crcd'
camera_params:
  image_height: 720
  image_width: 1280
  fx: 1096.696
  fy: 1096.696
  cx: 622.808
  cy: 383.126
  png_depth_scale: 10000.0     # matches depth/<fid>.png = depth_m * 10000 (after sc_factor baked in)
  crop_edge: 0
```

**Per-snippet `.py`** — fields to set (everything else inherited from the Replica base dict):
| Field | Value | Note |
|---|---|---|
| `seed` | from CLI (Section 8.1) | top-level dict key |
| `scene_name` / `run_name` | `<NAME>` | cosmetic + output naming |
| `data.gradslam_data_cfg` | `./configs/data/crcd.yaml` | selects CRCD loader + intrinsics + png_depth_scale |
| `data.basedir` | `data/CRCD` | loader joins `basedir/sequence` |
| `data.sequence` | `<NAME>` (e.g. `C1_001`) | the staged dir name |
| `data.desired_image_height/width` | `720` / `1280` (or T4 fallback 384/683, §3.4) | **NOT** Replica's 680×1200 |
| `data.num_frames` | snippet frame count (360/265/1527/1987/730) | or `-1` for all |
| `data.load_semantics` | `True` | |
| `data.num_semantic_classes` | **`4`** | NOT 101 |
| `scene_radius_depth_ratio` | **`3` → tune** (see below) | replaces the bound box for close-range scaling |
| `mean_sq_dist_method` | `"projective"` | as Replica |
| `eval_every` | `1` for full per-frame renders (Section 6) | strided only if matching paper for Phase A |
| `mapping.mapping_window_size`, `pruning_dict`, `densify_dict` | as Replica, adjust only on OOM/explosion | §3.4 / §8.2 |
| `viz.load_semantics` | `True` | enables semantic-color render pass |

**Close-range scaling (the SplaTAM analog of a "bound").** `scene_radius_depth_ratio` sets the pruning/densification scene radius as `max_first_frame_depth / ratio`. Surgical CRCD depth is ~0.6–0.9 m (vs Replica rooms at several metres). The **DDS-SLAM `configs/CRCD/*.yaml` bounds are a sanity reference only** for that depth range (e.g. c1_001 Z∈[0.68,0.90] m, c2_001 Z∈[0.68,0.92] m, e3_001 Z∈[0.62,0.89] m) — confirm your frame-0 MoGe depth (after sc_factor) falls in a comparable band. Keep `scene_radius_depth_ratio=3` as the faithful default; if Gaussians mis-initialize / explode on a snippet (NaN loss, runaway count), **raise the ratio** (e.g. 5–8, shrinking the scene radius for a tighter scene) and record the change + justification in the config header. If you cannot find a stable value, escalate (Q3) rather than ship a diverged run.

### 5.6 EXACT run commands (per snippet)
```bash
cd /content/SGS-SLAM && conda activate sgs-slam
export WANDB_MODE=offline PYTHONHASHSEED=0
python scripts/slam.py configs/crcd/<stem>.py 2>&1 | tee <OUT>/slam.log
```
where `<stem> ∈ {c1_001, e3_005, c3_001, g3_001, c2_001}`. Each snippet runs independently; a crash logs to `_failures.log` and the loop continues.

### 5.7 Trajectory extraction — `params.npz → est_c2w_data.txt` (REQUIRED converter)
SGS-SLAM writes poses **inside `params.npz`**, not as a text file. No existing repo script extracts them. `Addons/eval/kitti_to_tum.py` consumes an **already-12-float** est file — it does NOT read the npz. You MUST run this converter (author it as `Addons/eval/sgsslam_npz_to_est.py`, or inline it) **before** any trajectory metric:

```python
#!/usr/bin/env python3
# sgsslam_npz_to_est.py — extract per-frame c2w from SGS-SLAM params.npz to est_c2w_data.txt
# Verified against SGS-SLAM commit cb65e4b... : cam_unnorm_rots (1,4,N) quaternion [w,x,y,z] (unnormalized),
# cam_trans (1,3,N), both = relative w2c to the first camera (world = first cam). c2w = inv(w2c).
import argparse, numpy as np

def quat_wxyz_to_R(q):                      # mirrors utils/slam_external.py build_rotation, order [w,x,y,z]
    q = q / (np.linalg.norm(q) + 1e-12)
    r, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-r*z),   2*(x*z+r*y)],
        [2*(x*y+r*z),   1-2*(x*x+z*z), 2*(y*z-r*x)],
        [2*(x*z-r*y),   2*(y*z+r*x),   1-2*(x*x+y*y)]], dtype=np.float64)

ap = argparse.ArgumentParser()
ap.add_argument('--npz', required=True)          # <OUT>/params.npz
ap.add_argument('--out', required=True)          # <OUT>/est_c2w_data.txt
a = ap.parse_args()
P = np.load(a.npz, allow_pickle=True)
rots  = P['cam_unnorm_rots']                     # (1,4,N)
trans = P['cam_trans']                           # (1,3,N)
N = rots.shape[-1]
with open(a.out, 'w') as f:
    for i in range(N):                           # one line per source frame, in source order -> est[i] <-> GT[i]
        q = rots[0, :, i].astype(np.float64)     # [w,x,y,z]
        t = trans[0, :, i].astype(np.float64)
        w2c = np.eye(4); w2c[:3,:3] = quat_wxyz_to_R(q); w2c[:3,3] = t
        c2w = np.linalg.inv(w2c)
        f.write(' '.join(f'{v:.9f}' for v in c2w[:3,:].reshape(-1)) + '\n')  # 12 floats, 3x4 row-major
print(f'wrote {N} c2w rows to {a.out}')
```
Run: `python Addons/eval/sgsslam_npz_to_est.py --npz <OUT>/params.npz --out <OUT>/est_c2w_data.txt`. This produces exactly the 12-float row-major c2w lines `sim3_ate.py::load_est` expects (translation cols [3,7,11]). It is now `kitti_to_tum.py`-consumable too if evo pairing is wanted. **`cam_trans` is in metres only because the depth PNGs were sc_factor-baked (Section 5.3 step 5);** if you skipped that, ATE scale is wrong.

### 5.8 Frame / index alignment (renders ↔ GT frames ↔ GT poses ↔ depth) — VERIFY before any metric
This is load-bearing. The DDS-SLAM metrics pair by index/line-order:
- `eval_rendering.py` pairs renders to GT by the **numeric filename stem** against `video_frames/*l.png`.
- `sim3_ate.py` pairs `est[i] ↔ GT[i]` by **line order** (resamples + warns on count mismatch).
- `generate_video.py` pairs by index.

SGS-SLAM's `eval(save_frames=True)` writes renders as `gs_{time_idx:04d}.png` **only for frames it evaluates** — i.e. frame 0 plus every `eval_every`-th frame. To get one render per source frame:
1. **Set `eval_every=1`** in the CRCD config so `eval()` renders ALL frames (or explicitly document a stride and apply the same stride to GT poses + GT frames).
2. **Rename renders to the source frame index** matching `video_frames/NNNNNNl.png`: `gs_0000.png → 000000.png`, etc. The `time_idx` in the SGS-SLAM render filename IS the source frame index (the loader is `stride=1`, `start=0`), so the mapping is `time_idx → the i-th `video_frames` stem`. Build the rename from the natsorted `video_frames/*l.png` list so the stems match exactly. Write renamed RGB to `renders_rgb/` (pattern `[0-9]*.png`).
3. **`est_c2w_data.txt` has exactly one line per source frame in source order** (the converter in 5.7 emits N = `cam_unnorm_rots.shape[-1]` lines; with `start=0,stride=1,num_frames=full` this equals the source frame count).
4. **VERIFICATION GATE (escalate on mismatch, blocking):** assert `len(renders_rgb) == len(est_c2w_data lines) == len(video_frames/*l.png) == len(groundtruth.txt poses)`. If they differ (e.g. SGS-SLAM rendered only keyframes, or `eval_every>1` slipped through), STOP and escalate — do not let the metrics silently resample/mis-pair.

### 5.9 Output layout (under Drive)
```
/content/drive/MyDrive/Outputs/sgsslam/
  phaseA/<scene>/...                              # Section 4.6
  phaseB/<NAME>/                                  # NAME ∈ C1_001,E3_005,C3_001,G3_001,C2_001
    params.npz              # raw SGS-SLAM output (pose source)
    est_c2w_data.txt        # 12-float row-major c2w, one per source frame (Section 5.7)
    renders_rgb/            # rendered RGB, renamed to source-frame index NNNNNN.png (Section 5.8)
    rendered_depth_raw/     # raw metric rendered depth .npy per frame (for Depth-L1; see 6.3)
    stereo_ref/             # frame-0/periodic SGBM metric depth .npy (Depth-L1 reference)
    semantic/              # optional rendered semantic-color frames
    recon.ply / params.ply  # or pointer if oversized
    slam.log, config_used.py, crcd.yaml, seed.txt, gpu_runtime.json
    render_eval.csv, render_eval.txt, sim3_metrics.txt, depth_l1.txt, summary.txt
    <NAME>_6panel.mp4
    .DONE
  _render_summary.csv  _failures.log  _escalations.log  COMBINED_SUMMARY.txt  runbook.log
```
> The DDS-SLAM `sim3_ate.py` reads **12/16-float row-major c2w per line**; the Section 5.7 converter produces exactly that, one pose per source frame, ordered by frame index (so `est[i] ↔ GT[i]`, matching `groundtruth.txt` row order).

---

## 6. CRCD evaluation (DDS-SLAM harness — same for all methods)

Run from the **DDS-SLAM repo** (`/content/DDS-SLAM`) so the harness imports resolve. `pip install lpips` if missing.

### 6.1 ATE (Sim3, the headline CRCD trajectory metric)
```bash
python Addons/eval/sim3_ate.py \
  --est /content/drive/MyDrive/Outputs/sgsslam/phaseB/<NAME>/est_c2w_data.txt \
  --gt  data/CRCD/<NAME>/groundtruth.txt \
  --name "SGS-SLAM <NAME>" \
  --out /content/drive/MyDrive/Outputs/sgsslam/phaseB/<NAME>/sim3_metrics.txt
```
Reports **Sim3 ATE rmse/mean/median/max (mm)**, recovered scale `s`, est/GT path ratio, |Pearson| dominant axis (scale-free), and the RIGID ATE explicitly labeled **"do NOT headline."** Because MoGe-2 depth is up-to-scale (we bake a stereo-derived `sc_factor`, but it is a single global estimate), **headline the Sim3 number.** This is intentionally a **different protocol** from the paper's own cm-rigid Replica ATE; keep the two separate in reporting.

### 6.2 PSNR / SSIM / LPIPS (rendering)
```bash
python Addons/eval/eval_rendering.py \
  --gt_dir data/CRCD/<NAME>/video_frames \
  --render_dir /content/drive/MyDrive/Outputs/sgsslam/phaseB/<NAME>/renders_rgb \
  --name "SGS-SLAM <NAME>" \
  --output_csv  .../<NAME>/render_eval.csv \
  --summary_csv /content/drive/MyDrive/Outputs/sgsslam/_render_summary.csv \
  --sequence "CRCD (<NAME>)"
```
- GT pairs by filename index against `video_frames/*l.png`. Renders must be renamed to source-frame index (Section 5.8) so `[0-9]*.png` matches `video_frames/NNNNNNl.png`.
- The harness SSIM is a **simple 11×11 Gaussian-window approx** and LPIPS is **alex** — **accept these for cross-method consistency** (do not swap in SGS-SLAM's own torchmetrics SSIM for the CRCD table; that would make the table non-comparable across methods).
- **All 5 CRCD keys already exist in `eval_rendering.py:PAPER_REFERENCES`** (`CRCD (C1_001/C2_001/C3_001/E3_005/G3_001)`, all `None` refs). **Pass `--sequence "CRCD (<NAME>)"` directly. No harness edit is required** (and none should be made — it is a shared published-benchmark harness).

### 6.3 Depth-L1 (REQUIRED metric — reference fixed here)
**There is no Depth-L1 script in the repo and CRCD ships no GT depth.** The only CRCD-derived metric depth is the **stereo-SGBM** depth you already compute (Section 5.3 step 4/6). **Headline Depth-L1 reference = rendered SLAM depth vs the stereo-SGBM depth**, masked to stereo-valid pixels, in **cm** (to match the paper's units). Author `Addons/eval/depth_l1_crcd.py`:
```python
#!/usr/bin/env python3
# depth_l1_crcd.py — Depth-L1 (cm) of rendered SLAM depth vs stereo-SGBM reference, stereo-valid pixels only.
import argparse, glob, os, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument('--rendered_dir', required=True)   # rendered_depth_raw/*.npy (metres)
ap.add_argument('--stereo_dir', required=True)     # stereo_ref/*.npy (metres, NaN=invalid)
ap.add_argument('--out', required=True)
a = ap.parse_args()
refs = sorted(glob.glob(os.path.join(a.stereo_dir, '*.npy')))
errs = []
for rp in refs:
    fid = os.path.splitext(os.path.basename(rp))[0]
    rend_p = os.path.join(a.rendered_dir, fid + '.npy')
    if not os.path.exists(rend_p): continue
    ref = np.load(rp).astype(np.float64); rend = np.load(rend_p).astype(np.float64)
    if ref.shape != rend.shape: continue
    m = np.isfinite(ref) & (ref > 0.01) & (rend > 0.01)
    if m.sum() < 100: continue
    errs.append(np.abs(rend[m] - ref[m]).mean() * 100.0)   # metres -> cm
errs = np.array(errs)
with open(a.out, 'w') as f:
    f.write(f'Depth-L1 (cm) vs stereo-SGBM, stereo-valid px: mean={errs.mean():.3f} median={np.median(errs):.3f} n_frames={len(errs)}\n')
print(open(a.out).read())
```
Requirements this implies (wire them up):
- **Dump raw metric rendered depth** as `rendered_depth_raw/<fid>.npy` (metres). SGS-SLAM's `save_frames` writes only a JET colormap PNG — that is NOT usable. Render raw depth via the depth-silhouette pass (`rastered_depth` in `eval_helpers.py`) and save it as `.npy`; do this in your eval-time render-all step, named by source frame index.
- **Stereo reference** = the `stereo_ref/<fid>.npy` from Section 5.3 (frame-0 and every ~100 frames). Pairing is by frame id; only frames with a stereo reference contribute.
- **Self-consistency (rendered vs MoGe input) is a DIAGNOSTIC ONLY, never the headline Depth-L1** — it measures fit to the supervising prior, not accuracy.
- If, for a snippet, no frame yields ≥100 stereo-valid pixels (e.g. specular-dominated), Depth-L1 is **undefined**: write `n_frames=0`, mark the metric `N/A` for that snippet, and surface it — do not silently substitute self-consistency.

### 6.4 Canonical 6-panel video
```bash
python Addons/viz/generate_video.py \
  --rgb_input_dir data/CRCD/<NAME>/video_frames --rgb_input_pattern '*l.png' \
  --rgb_output_dir .../phaseB/<NAME>/renders_rgb --rgb_output_pattern '[0-9]*.png' \
  --depth_input_dir data/CRCD/<NAME>/depth --depth_output_dir .../phaseB/<NAME>/depth --depth_norm robust \
  --seg_dir data/CRCD/<NAME>/semantic_class --seg_pattern '*.png' --skip_raw_seg --seg_classmap \
  --trajectory_est .../phaseB/<NAME>/est_c2w_data.txt --trajectory_gt data/CRCD/<NAME>/groundtruth.txt --trajectory_raw \
  --output .../phaseB/<NAME>/<NAME>_6panel.mp4 --fps 15
```
Panels: (1) Input RGB, (2) Rendered RGB, (3) Input Depth, (4) Output Depth (robust p2–p98), (5) Seg overlay (4-class palette composited on rendered RGB), (6) Trajectory raw — plus the Sim3-aligned trajectory panel (default-on). SGS-SLAM has no uncertainty head, so omit `--uncert_dir`. (For panel 4, supply a normalized depth render dir if available, or reuse `depth/` input — the renderer normalizes robustly.)

### 6.5 Recommended diagnostics (EXPERT INPUT — produce these)
- **Trajectory overlay vs GT** (Sim3-aligned + raw) — already in the 6-panel; also dump a standalone PNG.
- **Per-frame ATE curve** (Sim3-aligned residual vs frame index) and **per-frame PSNR curve** (`eval/psnr.txt` is already per-eval-frame) — catches mid-sequence tracking loss / a single bad-frame skew that an average hides.
- **Depth-error heatmap** (rendered − stereo reference depth, per frame, robust colormap) — localizes where depth-loss-1.0 is failing on specular/deforming tissue.
- **Semantic-overlay frames** (rendered seg-color vs GT `semantic_class`) at ~10 strided frames — sanity that the 4-class supervision is being consumed, not ignored.
- **Keyframe-coverage plot** (which frames became keyframes per `keyframe_every`; spatial coverage of inserted Gaussians).
- **GT motion-profile sanity** (Phase-2 of the template): extent (mm), path length, per-frame motion, active fraction, sub-SNR sentinels. **Most CRCD snippets are sub-SNR for tracking** (C1_001 active-median 0.115 mm/f ≈ 8.7× below noise floor) — these are **REJECT for tracker benchmarking, keep for render-quality + diagnostics**. C2_001 (≈29 mm extent) is the best-case tracker candidate.

---

## 7. Standard CLI contract — `run_sgsslam.sh`

> Matches the master orchestrator's calling convention. **This is the authoritative interim contract (Section 1.1, NON-BLOCKING); reconcile with `00_COMMON.md` if/when supplied.**

**Invocation:** `bash run_sgsslam.sh <stage> [snippet|scene] [seed]`

**Positional / args:**
- `<stage>` ∈ `env | smoke | phaseA | phaseB | eval | video | all`
- `[snippet|scene]` optional filter: a Replica scene (`room0`) for phaseA, or a CRCD `NAME` (`C1_001`) for phaseB/eval/video; omitted ⇒ all.
- `[seed]` optional int (default `0`).

**Env vars (read, with defaults):**
| Var | Default | Meaning |
|---|---|---|
| `SGS_REPO` | `/content/SGS-SLAM` | method clone |
| `DDS_REPO` | `/content/DDS-SLAM` | harness repo |
| `OUT_ROOT` | `/content/drive/MyDrive/Outputs/sgsslam` | all outputs |
| `CRCD_DRIVE` | `/content/drive/MyDrive/Datasets/CRCD-Published` | staged raw (incl. `cam_calib/...pkl`) |
| `SEED` | `0` | determinism (→ config `seed`) |
| `GPU_MIN_VRAM_GB` | `12` | hard abort floor (§3.4) |
| `TORCH_CUDA_ARCH_LIST` | auto from `compute_cap` | rasterizer build |
| `WANDB_MODE` | `offline` | no login prompt |

**Exit codes:** `0` success · `1` generic failure (logged, isolated) · `2` env/build failure · `3` data-staging failure · `4` preprocess failure · `5` depth-gen failure · **`42` STOP-and-escalate (Clarification Protocol; blocking ambiguity or gate fail or VRAM abort)**.

**Stage markers (stdout, grep-able):** `[PHASE <id>] <ts> -- <desc>` (mirror `run_crcd_4snippets.sh:phase()`). Per-snippet sentinels: `.STAGED`, `.PREPROCESSED`, `depth/.DONE`, `.sc_factor`, `.DONE`. Resumability: skip a snippet if `<DRIVE_DST>/.DONE` exists; skip SLAM if `est_c2w_data.txt` line-count ≥ frames.

**Failure isolation:** each snippet/scene in its own subshell guarded by `|| { echo "FAILED_<stage>_<NAME>" >> $OUT_ROOT/_failures.log; continue; }`. A single `run_sgsslam.sh all` runs Phase A → GATE → all 5 CRCD snippets → eval → video → combined table, never aborting the batch on one failure. **The final stage runs the cross-snippet aggregation (Section 8.1, `aggregate_crcd_generic.py`).**

---

## 8. Failure modes, determinism, checkpointing, logging

### 8.1 Determinism / seeding + aggregation
- **Seed via the SGS-SLAM-native `seed` config field** (top-level `config['seed']`, propagated to `run_name`). SGS-SLAM internally calls `seed_everything(seed)` (`utils/common_utils.py`). Set it from `$SEED`. Also export `PYTHONHASHSEED=0`. Disable TF32 by adding to your config copy's preamble (top of the `.py` config, which is imported before SLAM runs) `import torch; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False` — this is config-level, not a method-logic edit.
- **On the "never edit method source" constraint:** the ONLY source touches allowed are the **dataset registration** edits in Section 5.4(c) (`__init__.py` import + one `elif` branch in `get_dataset`). These add a new dataset, they do not change SLAM/loss/optimizer logic. Treat them as scoped, declared adaptations: log them to `_escalations.log` as `ESCALATE [phaseB] registered CRCDDataset (loader + get_dataset elif) | blocking=no` and note them in the deliverable. If the operator forbids ANY source edit, the alternative is to monkey-patch via a wrapper `scripts/run_crcd.py` that imports `get_dataset` and registers the class at runtime — escalate to choose. Do not silently edit deeper logic.
- **Determinism check (Phase A):** run one scene twice with the same seed; metrics must agree within numerical noise. Log both; flag if they diverge (3DGS densification can introduce nondeterminism — record it, don't hide it).
- 3DGS Gaussian-splitting/cloning may not be bit-exact across runs even seeded; if so, report **mean ± std over ≥2 seeds** on CRCD.
- **Cross-snippet + cross-method aggregation — use `aggregate_crcd_generic.py` (NOT `aggregate_ab.py`):**
  ```bash
  # single-method cross-snippet table:
  python Addons/eval/aggregate_crcd_generic.py \
    --root /content/drive/MyDrive/Outputs/sgsslam/phaseB \
    --names C1_001 E3_005 C3_001 G3_001 C2_001 \
    --out  /content/drive/MyDrive/Outputs/sgsslam/COMBINED_SUMMARY.txt
  # cross-method master table (orchestrator):
  python Addons/eval/aggregate_crcd_generic.py \
    --method-roots SGS-SLAM=/content/drive/MyDrive/Outputs/sgsslam/phaseB \
                   SemGauss=/content/drive/MyDrive/Outputs/SemGauss-SLAM/crcd_<DATE> \
    --names C1_001 E3_005 C3_001 G3_001 C2_001 \
    --out /content/drive/MyDrive/Outputs/CROSS_METHOD_TABLE.txt
  ```
  It reads `<ROOT>/<NAME>/{render_eval.csv|render_eval.txt, sim3_metrics.txt}` — exactly what Section 5.9/6.x emit. **Do not produce `render_metrics.txt` / `<cell>_s<seed>/payload.tgz`** (that layout is for the obsolete `aggregate_ab.py`).

### 8.2 Failure modes (catch + log, don't abort batch)
| Symptom | Likely cause | Action |
|---|---|---|
| rasterizer import error | `TORCH_CUDA_ARCH_LIST` wrong / CUDA mismatch | rebuild for detected arch; if persists, escalate Q6 |
| missing `semantic_ids/`/`semantic_colors/` (Replica) | vanilla Replica, not authors' Dropbox | re-download correct build |
| loader error "Number of semantic ids images ... must be the same" | emitted folder count ≠ frame count | re-run the semantic-emit step (5.4a); assert counts |
| Gaussians explode / NaN loss on CRCD | room-scale `scene_radius_depth_ratio` on close-range scene | raise ratio (5.5); if still NaN, escalate Q3 |
| ATE huge but Sim3 small | scale confound from up-to-scale depth / wrong sc_factor | check sc_factor baked into PNGs (5.3 step 5); headline Sim3 |
| OOM on C3_001/G3_001 | dense 3DGS VRAM on long hi-res snippet | A100, or T4 fallback (§3.4); if VRAM<12 GB, abort+escalate |
| render count ≠ GT count | `eval_every>1` or keyframe-only renders | set `eval_every=1`, re-render all, re-run 5.8 verification |
| Depth-L1 `n_frames=0` | no stereo-valid pixels | mark N/A, surface; do NOT substitute self-consistency |

### 8.3 Checkpointing / resume
- Cache: Replica tarball + CRCD-Published-MoGe-2 depth/`.sc_factor`/`stereo_ref` on Drive.
- SGS-SLAM writes its own `params<idx>.npz` checkpoints (`save_checkpoints=True`, `checkpoint_interval=500`); to resume a long snippet set `load_checkpoint=True`, `checkpoint_time_idx=<last>`.
- Per-phase sentinels (Section 7). On restart, skip any snippet whose `.DONE` is present and any SLAM whose `est_c2w_data.txt` line-count already meets `frames`.
- Ship payloads to Drive **before** marking `.DONE` (the template's lesson: explicitly move renders into `renders_rgb/` and rename to source index before tarring, else they are silently dropped/mis-paired).

### 8.4 Logging conventions
- Master log `tee`'d to `$OUT_ROOT/runbook.log`. Per-snippet `slam.log`. `gpu_runtime.json` per run: GPU name, total + **peak** VRAM, wall-time, tracking/mapping fps, chosen resolution, applied `sc_factor`. Every escalation to `_escalations.log`; every isolated failure to `_failures.log`.

---

## 9. Deliverables checklist (must exist in `Outputs/sgsslam/` when done)

- [ ] `phaseA/<scene>/` for each repro scene: `slam.log`, inline metrics (`eval/psnr.txt` + ATE/Depth-L1/mIoU), `params.npz`/`params.ply` (or pointer), `config_used.py`, `seed.txt`, `gpu_runtime.json`, `gate.json`.
- [ ] **`phaseA/GATE.md`** — per-scene reproduced vs paper, within-tolerance verdict, PASS/FAIL, training-view + frame-set (Q8) confirmation, determinism check.
- [ ] `phaseB/<NAME>/` for all 5 snippets: `params.npz`, `est_c2w_data.txt`, `renders_rgb/` (renamed to source index), `rendered_depth_raw/`, `stereo_ref/`, `sim3_metrics.txt`, `render_eval.csv`/`.txt`, `depth_l1.txt` (vs stereo-SGBM; or `N/A` with reason), `summary.txt`, `<NAME>_6panel.mp4`, diagnostics (per-frame ATE/PSNR curves, depth-error heatmaps, semantic-overlay frames, keyframe-coverage, traj overlay), `gpu_runtime.json`, `.DONE`.
- [ ] CRCD configs authored for all 5 snippets (`configs/crcd/<stem>.py`) + shared `configs/data/crcd.yaml`, `num_semantic_classes=4`, with any `scene_radius_depth_ratio` / resolution changes + justification in the config header.
- [ ] The CRCD loader (`datasets/gradslam_datasets/crcd.py`) + registration edits, and the semantic-emit step output (`semantic_ids/`, `semantic_colors/`).
- [ ] The two authored eval helpers: `Addons/eval/sgsslam_npz_to_est.py`, `Addons/eval/depth_l1_crcd.py`.
- [ ] Cross-snippet: `_render_summary.csv`, `_failures.log`, `_escalations.log`, `COMBINED_SUMMARY.txt` (from `aggregate_crcd_generic.py --root`), `runbook.log`.
- [ ] Cross-method contribution: outputs in the `<NAME>/{render_eval.csv, sim3_metrics.txt}` layout that `aggregate_crcd_generic.py --method-roots` ingests for the master table.
- [ ] A short `METHOD_REPORT.md` (return as text to the orchestrator, not a buried file): Phase-A gate result, CRCD per-snippet Sim3-ATE/PSNR/SSIM/LPIPS/Depth-L1, the deformable-tissue limitation caveat (quote the paper), resolution/sc_factor caveats, and every open question carried forward.

---

## 10. Open questions to escalate (explicit list — with fixed blocking classification)

| # | Question | Default / interim | blocking? |
|---|---|---|---|
| 1 | Confirm CRCD `semantic_instance` (coco_id+1, 4-class) is the agreed SGS-SLAM semantic input, exposed as `semantic_ids/`+`semantic_colors/`, `num_semantic_classes=4`. Do NOT substitute binary tool mask or DINO. | proceed with 4-class GT (Section 5.4) | **yes** before Phase B |
| 2 | Stereo-scaling cadence: global spec says ~every 100 frames; resolve to periodic SGBM windows → median global scale (Section 5.3 step 6). | periodic (~100f) median scale; escalate if windows disagree >20% | resolved (escalate only on >20% drift) |
| 3 | Per-snippet `scene_radius_depth_ratio` for close-range surgical scenes; who approves a deviation from the faithful default of 3. | default 3; raise on explosion, record | **yes** if a stable value cannot be found |
| 4 | SGS-SLAM depth contract: confirmed `png_depth_scale` lives in the data-yaml and depth is loaded as `depth/png_depth_scale`. Set `png_depth_scale=10000` in `crcd.yaml`; SGS-SLAM has NO `sc_factor` — scale is baked into PNGs (Section 5.3 step 5). | resolved (verified in basedataset.py) | no |
| 5 | SGS-SLAM has no `range_d/near/far/depth_trunc/sc_factor` fields, so the DDS-SLAM "sc_factor rescale bug" does not apply. Baked-in scale is the contract. | resolved — bake scale into PNGs | no |
| 6 | Legacy stack build on the assigned GPU: torch 1.12.1 / cu11.6 / py3.10 building the rasterizer on A100 (sm_80) vs T4 (sm_75). If it fails, is a torch/CUDA bump authorized (load-bearing — deviates from `environment.yml`)? | build per env.yml; escalate on failure | **yes** if build fails |
| 7 | Depth-L1 reference fixed: rendered SLAM depth vs frame-0/periodic stereo-SGBM depth, stereo-valid px, cm (Section 6.3). Self-consistency is diagnostic only. | resolved (stereo-SGBM reference) | no (confirm with operator at report time) |
| 8 | Table-1 frame set for the PSNR/SSIM/LPIPS average: every training frame vs `eval_every`-strided. Read `utils/eval_helpers.py::eval` + the paper's table generation. | must determine before the gate | **yes** (gate prerequisite) |
| 9 | (REMOVED) eval_rendering.py sequence keys — all 5 CRCD keys already exist in PAPER_REFERENCES; no edit needed. | — | n/a |
| 10 | VRAM policy for SGS-SLAM specifically: A100 for C3_001/G3_001 at full res, or documented T4 downscale (Section 3.4). Confirm full-res-comparability waiver if T4 fallback is used. | A100 for long snippets; T4 fallback only with waiver | **yes** if only T4 available for long snippets |
| 11 | `00_COMMON.md` absent → Section 7 is the interim contract (Section 1.1). | use Section 7 | no |
| 12 | Confirm `np.unique(semantic_class)` is exactly `{0,1,2,3}` for all 5 snippets; remap to contiguous 0..3 if not (Section 5.4b). | verify before exposing semantic_ids/ | **yes** if not {0,1,2,3} |
