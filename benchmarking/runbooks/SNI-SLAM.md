> **Resolved decisions apply.** See *Resolved Benchmark Decisions* in [00_COMMON.md](00_COMMON.md) — they override any “escalate”/“open question” below: Depth-L1 = input-vs-output per model; periodic ~100f stereo rescaling; per-paper semantic mirroring on CRCD 4 classes; missing configs (e3_005/c3_001/g3_001) are the agent's job.

# SNI-SLAM — Agent Run Book (Benchmark Standing Orders)

> **Method:** SNI-SLAM (Semantic Neural Implicit SLAM), CVPR 2024 — IRMVLab.
> **Class:** NeRF / feature-plane (tri-plane, ESLAM-lineage) dense **RGB-D semantic** SLAM with SDF/TSDF volume rendering. **NOT 3D Gaussian Splatting.**
> **Your job:** Reproduce SNI-SLAM's published Replica result (PHASE A), then adapt it to the 5 CRCD surgical snippets and evaluate through the DDS-SLAM harness (PHASE B).
> **You are a fresh autonomous agent.** You will be handed the SNI-SLAM paper PDF, the cloned SNI-SLAM repo, the `sni` env spec, the Replica data, and `00_COMMON.md` *later*. These are standing orders to follow once you have them. **When a load-bearing fact is unknown, STOP and escalate — never guess (see §0).**
> **Pre-flight invariant (read once):** several artifacts named below are handed/created later (SNI-SLAM repo, `sni` env, Replica data, `00_COMMON.md`). Every path that points into the SNI-SLAM repo or into Replica is a **runtime-verify item, not a pre-verified fact**. Before Phase A you MUST `ls`/`grep` each such path and escalate any mismatch (§4.0). By contrast, every `Addons/...` path in this book has been verified against the live DDS-SLAM repo on the authoring date.

---

## 0. Mission & non-negotiables

### 0.1 Mission
Produce a reproducible, publishable benchmark entry for SNI-SLAM with two deliverable halves:
1. **PHASE A — Own-dataset reproduction.** Run SNI-SLAM on **Replica** exactly as the paper does, with the paper's own eval scripts, and pass a **Decision Gate** tied to the paper's reporting granularity (an **8-scene average**, §2.4).
2. **PHASE B — CRCD adaptation.** Run SNI-SLAM on 5 prescribed CRCD snippets (C_1/001, E_3/005, C_3/001, G_3/001, C_2/001) with MoGe-2 + stereo-scaled depth, a CRCD-appropriate semantic-input decision, and evaluate via the DDS-SLAM harness (ATE-sim3, PSNR, SSIM, LPIPS, Depth-L1) + the canonical 6-panel video.

### 0.2 Non-negotiables (precision rules)
- **No assumptions on load-bearing choices.** If a number, path, license term, dataset mapping, or method behaviour is unknown AND it changes the result or its interpretation, you **STOP and escalate** (Clarification Protocol below). Do not silently default.
- **Reproduce exactly as the paper does in PHASE A.** Use SNI-SLAM's *own* `eval_ate.py` (rigid Horn, no scale), *own* recon eval, *own* mIoU eval. Do **not** substitute the DDS-SLAM sim3 harness in Phase A.
- **Use the DDS-SLAM harness exactly in PHASE B.** All CRCD metrics flow through `Addons/eval/*` and `Addons/viz/generate_video.py`. SNI-SLAM's native eval scripts are **not** used for CRCD.
- **Failure isolation.** One scene/snippet failing must never abort the rest. Catch, log to `_failures.log`, continue.
- **Resumability.** Every expensive stage is sentinel-gated; a re-run skips completed work.
- **Determinism.** Seed everything; log the seed; log GPU + VRAM + wall-time per stage (§8).
- **GPU policy.** SNI-SLAM is NeRF/feature-plane, **not** VRAM-heavy like the 3DGS methods. It runs on a T4 (16 GB) with CPU offload, but is slow. Detect GPU via `nvidia-smi`; **A100 recommended for throughput, not required**. Record the exact min-VRAM floor empirically and escalate the abort threshold (§10, OQ-7).

### 0.3 Clarification Protocol (how/when to escalate)
**Trigger an escalation when any of these is true:**
- A required input does not exist on CRCD (e.g. surgical semantic labels) and the substitute would change which *method* is being benchmarked.
- A config/data mapping is ambiguous (e.g. which CRCD snippet maps to which config; 3 of 5 configs do not exist — OQ-3).
- A metric has no implementation or reference (Depth-L1 — OQ-1) or two definitions conflict (alignment conventions).
- Scope is unclear (full 8×5 Replica runs vs 8×1; ScanNet/TUM in/out — OQ-2, OQ-8).
- A known bug (sc_factor co-scaling — OQ-5) could corrupt published numbers.
- A directive in this book conflicts with the GLOBAL benchmark spec (e.g. stereo cadence — OQ-4).

**How to escalate:**
1. Write a single message titled `CLARIFICATION NEEDED — SNI-SLAM — <topic>`.
2. State: (a) the exact decision point, (b) 2–4 concrete options with consequences, (c) your recommended default and why, (d) what is blocked until answered.
3. **Do not proceed past the blocked step.** Continue with *independent* work that is not blocked. Record the open item in the run's `OPEN_QUESTIONS.md` under `Outputs/`.
4. All §10 questions are pre-identified escalations — raise the relevant ones **before** the stage that needs them, not after a failed run. Items marked **HARD GATE** abort the affected snippet/scene until resolved; they are not "flag-and-continue".

---

## 1. Orientation / required reading

**Read first, before touching anything:**
1. **`00_COMMON.md`** (shared harness contract for all 4 methods). **Not present in the local working copy — obtain it from the user before Phase A; if absent, escalate (blocked, exit 3).** It defines the master orchestrator interface, the Drive layout `/content/drive/MyDrive/Outputs`, and the `run_<method>.sh` contract you must satisfy (§7). Re-validate §7's stage-marker names against it once you have it.
2. **SNI-SLAM paper** (`https://openaccess.thecvf.com/content/CVPR2024/papers/Zhu_SNI-SLAM_Semantic_Neural_Implicit_SLAM_CVPR_2024_paper.pdf`), in this order:
   - **Table 1** — Replica recon + localization, *averaged over all 8 scenes, 5 runs each* (THE headline; defines the Phase-A gate).
   - **Sec. 3 (Method)** — feature-plane representation, cross-attention fusion, hierarchical semantic rep.
   - **Eq. 9** — the semantic loss is **gradient-detached** from geometry/appearance (semantics do not drive tracking).
   - **Sec. 4 / Implementation** — losses, weights, training GPU (RTX 4090), FPS (Table 5).
   - Tables 2 (mIoU), 3 (ScanNet RMSE), 4 (TUM RMSE), 5 (runtime/params) — context; 3 & 4 are **out of public-repo scope**.
3. **SNI-SLAM repo** (handed later), in this order:
   - `environment.yaml` (the brittle py3.7 / torch1.11 / cu113 / pytorch3d0.7.1 stack — §3).
   - `configs/SNI-SLAM.yaml` (model config; **hardcoded absolute paths** to `seg/dinov2_replica.pth` — must be edited).
   - `configs/Replica/*.yaml` + `configs/Replica/replica.yaml` (the only shipped configs).
   - `src/networks/dinov2_seg.py` (DINO2SEG head; `mode='mapping'` → 16-D feature map, `mode='train'` → argmax pseudo-label). **Read the patch-size / img_size constants here** (§5.4).
   - `src/utils/datasets.py` `BaseDataset.__getitem__` (unconditionally reads `semantic_class/*.png` + `semantic_classes.pkl` — the CRCD blocker). **Read which config keys it consumes for depth scale and bounds** (§5.4, OQ-3/OQ-5).
   - `Mapper.py` (semantic self-supervision; **find where `estimate_c2w_list` is written into the ckpt** — confirms OQ-10), `Tracker.py`.
   - `run.py` (entry), `src/tools/eval_ate.py`, `src/tools/eval_segmentation.py`, `src/tools/eval_recon.py`, `visualizer.py`.
   - GitHub Issues #11, #20, #27, #29 (install/repro gotchas).
4. **DDS-SLAM harness** (this repo, the CRCD eval/viz backend — all paths below verified present):
   - `Addons/colab/run_crcd_4snippets.sh` (canonical end-to-end CRCD batch — structural template; **its SNIPPETS array does NOT match the benchmark 5**, §5.0).
   - `Addons/colab/crcd_depth_gen_remainder_20260616.sh` (depth-gen for E3_005/C3_001/G3_001 among others; the **source of truth for the depth pipeline**, §5.3).
   - `Addons/colab/run_cell.sh` (single-run wrapper: train→video→metrics→ship).
   - `Addons/eval/sim3_ate.py` (Sim3 ATE; `load_est` reads est_c2w_data.txt), `Addons/eval/eval_rendering.py` (PSNR/SSIM/LPIPS).
   - `Addons/eval/sni_export_traj.py` (**new — authored for this run book**; ckpt `.tar` → `est_c2w_data.txt`, §5.7).
   - `Addons/eval/aggregate_crcd_generic.py` (**new — authored for this run book**; general `<NAME>/payload.tgz` aggregator, §7).
   - `Addons/depth/generate_depth_moge.py` (writes only `*-left_depth.npy`), `Addons/depth/moge_npy_to_png.py` (**new — extracted block**, §5.3), `Addons/depth/generate_depth_stereo.py` (full per-frame stereo generator — NOT a periodic anchor; relevant to OQ-4).
   - `Addons/preprocess/preprocess_crcd_published.py`, `Addons/viz/generate_video.py`, `Addons/env/colab_setup.sh`.

---

## 2. Method dossier

### 2.1 One-paragraph summary
SNI-SLAM (CVPR 2024) is a NeRF-based dense RGB-D semantic SLAM system built on a feature-plane (tri-plane, ESLAM-lineage) scene representation with separate appearance/geometry/semantic planes at coarse+fine resolution. It fuses appearance, geometry, and semantic features via **cross-attention**, then jointly renders color, TSDF/depth, and semantics via SDF-based volume rendering. Semantics are **self-supervised** by a pretrained **DINOv2 + conv segmentation head** (`use_gt_semantic: False` by default): at each mapping step the head produces a 16-D dense semantic feature map (fusion + feature loss) and an argmax class map (cross-entropy pseudo-target). Crucially, the semantic loss is **gradient-detached** (paper Eq. 9), so semantics do **not** improve tracking — tracking accuracy comes from RGB+depth+SDF+feature losses.

### 2.2 Method class
Neural-implicit (NeRF-style) dense semantic RGB-D SLAM. Feature-plane representation, SDF/TSDF volume rendering. **Strictly RGB-D** (needs per-frame depth for tracking AND mapping). **Not** Gaussian-splatting → moderate VRAM.

### 2.3 Own datasets + how to obtain
| Dataset | Status | How to obtain |
|---|---|---|
| **Replica** (semantic, vMAP/Semantic-NeRF render) | **In scope (Phase A).** | Author subset via Google Drive folder `https://drive.google.com/drive/u/0/folders/1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h` → `./data/replica`. Full 8 scenes need `data_generation/` (Habitat-Sim + Replica meshes). Each scene: `rgb/rgb_*.png`, `depth/depth_*.png` (`png_depth_scale=1000`), `semantic_class/semantic_class_*.png`, `traj.txt`. Requires `seg/semantic_classes.pkl` + `seg/num_semantic_class.pkl`. **8 scenes:** room0/1/2, office0/1/2/3/4. |
| **ScanNet** | **OUT of scope** (confirm OQ-8). | No config, no exercised loader, no ScanNet seg head shipped. License-gated ToU at `http://www.scan-net.org/`. Table 3 not reproducible from public repo. |
| **TUM RGB-D** | **OUT of scope** (confirm OQ-8). | No config/loader; paper used SAM-based DEVA for labels. `https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download`. Table 4 not reproducible. |

### 2.4 How the paper REPORTS results (defines the Phase-A bar)
- **Canonical headline = Table 1**, averaged over **all 8 Replica scenes**, each averaged over **5 runs**:
  - Depth-L1 **0.766 cm**, Acc **1.942 cm**, Comp **1.702 cm**, Comp.Ratio **96.624 %**, ATE-Mean **0.397 cm**, ATE-RMSE **0.456 cm**.
  - Per-scene numbers are in the **supplementary**, not the main paper.
- Table 2 = semantic mIoU on 4 Replica scenes (avg **87.41 %**). Table 3 = ScanNet RMSE (avg 6.54). Table 4 = TUM RMSE. Table 5 = runtime/params (~6.2 M params, ~0.86–2.13 FPS).
- **FAITHFUL-REPRO BAR:** the headline is a **dataset-wide 8-scene average**, so a faithful Replica reproduction must run **all 8 scenes** and match the Table 1 averages within tolerance. Matching only 1–2 scenes is **NOT sufficient**. (Compute-adaptive compromise: 8 scenes × 1 run — but this must be **confirmed** by the user, OQ-2.)

### 2.5 Pretrained weights + license gating
All from the **same Google Drive folder** as the Replica data (`.../1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h`), **not** license-gated:
| File | Required? | Purpose |
|---|---|---|
| `dinov2_replica.pth` | **YES** | DINO2SEG seg head, Replica-domain, **52 classes**. Produces semantic feature map + self-supervision pseudo-labels at runtime. Path set in `configs/SNI-SLAM.yaml` `model.cnn.pretrained_model_path` (currently hardcoded `/data0/nerf/sni-slam/seg/dinov2_replica.pth` — **must edit**). |
| `facebookresearch_dinov2_main.zip` | **YES** | DINOv2 ViT backbone code. Unzip exactly to `seg/facebookresearch_dinov2_main` (added to `sys.path` by `dinov2_seg.py`). Backbone blocks 0–3 frozen, ≥4 fine-tuned. |
| `seg/semantic_classes.pkl` + `seg/num_semantic_class.pkl` | **YES** | Remap raw Replica semantic ids → contiguous. Without them `BaseDataset.__getitem__` crashes. |

### 2.6 Semantic-input requirement (this method's input)
- **Operational input for tracking/mapping = DINOv2-head prediction on the RGB frame** (`use_gt_semantic: False`). GT masks are loaded by the Replica loader but used **only for mIoU eval**, not training.
- The shipped head is **Replica-domain (furniture/room, 52 classes)** — it produces **garbage on surgical tissue/instruments**. This is the central CRCD blocker (§5.4, OQ-6 — escalate before any CRCD run).

---

## 3. Environment setup on Colab

> **Landmine:** SNI-SLAM's stack (py3.7 / torch1.11 / cu113 / pytorch3d0.7.1) is **old and brittle** and **incompatible** with Colab's default and with the DDS-SLAM modern stack (torch≥2 + tinycudann). **You need TWO environments:**
> - **`sni` env** — SNI-SLAM training/eval (Phase A entirely; Phase B SLAM run + `sni_export_traj.py`).
> - **DDS-SLAM env** — the harness (`Addons/eval/*`, `Addons/viz/*`, `Addons/depth/*`), built via `Addons/env/colab_setup.sh`.
> Never mix them in one interpreter. Run SNI-SLAM in `conda activate sni`; run harness scripts in the DDS-SLAM env. (`sni_export_traj.py` only needs `torch`+`numpy`; run it in `sni` so it can `torch.load` the ckpt.)

### 3.1 Pins (from `environment.yaml`, name: `sni`)
- Python **3.7.11**; pytorch **1.11.0** (`py3.7_cuda11.3_cudnn8.2.0`), torchvision **0.12.0**, torchaudio **0.11.0**; cudatoolkit **11.3.1**; **pytorch3d 0.7.1** (cu113/pyt1110).
- numpy 1.21.5, opencv-python 4.5.5.64, open3d 0.13.0, trimesh 3.10.7, scikit-image 0.19.2, openexr 1.3.7 (+ system `libopenexr-dev` + pyembree 0.1.6), matplotlib 3.4.3, scipy 1.7.3, pandas 1.3.5, timm, wandb (pip), tabulate, yacs, gdown 4.4.0.

### 3.2 Build steps
```bash
# 1. system deps (BEFORE env create)
sudo apt-get update && sudo apt-get install -y libopenexr-dev
# 2. conda env
conda env create -f environment.yaml          # name: sni
conda activate sni
# 3. unzip DINOv2 backbone EXACTLY as named (sys.path append is hardcoded)
unzip facebookresearch_dinov2_main.zip -d seg/    # -> seg/facebookresearch_dinov2_main/
# 4. EDIT hardcoded absolute paths in configs/SNI-SLAM.yaml:
#    model.cnn.pretrained_model_path : /data0/nerf/sni-slam/seg/dinov2_replica.pth
#    -> <repo>/seg/dinov2_replica.pth   (and any sibling seg/ path)
```

### 3.3 Dependency landmines (ordered by likelihood)
1. **pytorch3d 0.7.1 / cu113 / torch1.11 / py3.7 mismatch** — the #1 install failure. If conda can't resolve, fetch the exact prebuilt wheel matching `py3.7_cu113_pyt1110`. Verify `python -c "import pytorch3d; print(pytorch3d.__version__)"` → `0.7.1`.
2. **Colab default CUDA/torch mismatch** — a clean conda env is mandatory; do not `pip install` into base Colab python.
3. **openexr / pyembree** — need apt `libopenexr-dev` *first*. Used by mesh culling/eval (`eval_recon.py`).
4. **`seg/facebookresearch_dinov2_main` on `sys.path`** — `dinov2_seg.py` appends it; unzip name must match exactly.
5. **Hardcoded absolute paths** in `configs/SNI-SLAM.yaml` — runs fail immediately if not edited.
6. **`torch.multiprocessing` / `share_memory_()` fragility** (Issues #11, #27) — Mapper/Tracker use shared CUDA memory across processes; if it hangs, reduce processes / check `keyframe_device`/`feature_device: cpu` (defaults push storage to CPU to cut VRAM).

### 3.4 Verification smoke test (must pass before Phase A)
```bash
conda activate sni
python - <<'PY'
import torch, pytorch3d, cv2, open3d, trimesh, skimage, timm
assert torch.__version__.startswith('1.11'), torch.__version__
assert torch.cuda.is_available(), 'no CUDA'
assert pytorch3d.__version__ == '0.7.1', pytorch3d.__version__
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))
print('pytorch3d', pytorch3d.__version__, '| OK')
PY
# DINO2SEG head + weights load:
python - <<'PY'
import sys, os; sys.path.append(os.path.abspath('seg/facebookresearch_dinov2_main'))
import torch
sd = torch.load('seg/dinov2_replica.pth', map_location='cpu')
print('seg head keys:', len(sd) if isinstance(sd, dict) else type(sd))
for f in ['seg/semantic_classes.pkl','seg/num_semantic_class.pkl']:
    assert os.path.exists(f), f'MISSING {f}'
print('seg pkls present | OK')
PY
```
Record GPU name + total VRAM (`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader`) into the run log.

---

## 4. PHASE A — Reproduce on Replica (own dataset)

> **Goal:** match Table 1's **8-scene-average** within tolerance using SNI-SLAM's own eval. **END WITH A DECISION GATE.**

### 4.0 Pre-Phase-A path re-validation (mandatory; these are NOT pre-verified)
Before any run, confirm-or-escalate each of the following against the *actual* handed repo. Do **not** assume.
```bash
test -f 00_COMMON.md || echo "ESCALATE: 00_COMMON.md absent (exit 3)"
test -d "$SNI_REPO" || echo "ESCALATE: SNI-SLAM repo absent"
# the eval entrypoints — names/args may differ from this book; confirm each exists:
for f in run.py src/tools/eval_ate.py src/tools/eval_segmentation.py src/tools/eval_recon.py visualizer.py; do
  test -f "$SNI_REPO/$f" || echo "ESCALATE: $f not found at assumed path — re-discover before Phase A"
done
# the recon mesh path + external recon-eval repo are ASSUMED — pin them now (see §4.4):
grep -rn "final_mesh_eval_rec_culled\|eval_rec\|\.ply" "$SNI_REPO/src/tools/eval_recon.py" || \
  echo "ESCALATE: confirm the actual mesh filename SNI-SLAM writes"
```
Any line that prints `ESCALATE` is a blocker for the dependent stage. Record in `OPEN_QUESTIONS.md` and raise before running.

### 4.1 Obtain data + weights
1. Download the Google-Drive folder → `./data/replica` (subset) + `seg/` weights (§2.5).
2. **Determine scene coverage.** Inspect `./data/replica`: which of the 8 scenes are present?
   - If all 8 present → proceed.
   - If a subset → **generate the missing scenes** via `data_generation/` (Habitat-Sim + Replica meshes) OR **escalate** (OQ-2): is 8×1 the target, and if scenes are missing, is generation in scope? **Do not claim an 8-scene-average reproduction from a subset.**
3. Verify each scene dir has `rgb/`, `depth/`, `semantic_class/`, `traj.txt`, and the `seg/*.pkl` remap files exist.

### 4.2 Configs
- Use the shipped `configs/Replica/<scene>.yaml` (room0/1/2, office0–4) inheriting `configs/Replica/replica.yaml` + `configs/SNI-SLAM.yaml`.
- Confirm `func.use_gt_semantic: False` (default), `model.cnn.n_classes: 52`, `scale: 1`, `png_depth_scale: 1000`, and that `model.cnn.pretrained_model_path` now points at the local `seg/dinov2_replica.pth`.
- Set `seed` explicitly and log it (§8).

### 4.3 EXACT run commands (per scene)
```bash
conda activate sni
SCENE=room0    # iterate over: room0 room1 room2 office0 office1 office2 office3 office4
python -W ignore run.py configs/Replica/${SCENE}.yaml
# outputs land under the config's output dir; per-frame estimated poses in ckpts/*.tar (estimate_c2w_list)
```
**Run-loop requirements:** per-scene timer, `nvidia-smi` VRAM sample (peak) logged, failure caught → append `FAILED_REPLICA_<scene>` to `_failures.log` and **continue**. If OQ-2 resolves to 5 runs/scene, wrap in an outer `for run in 1..5` loop with distinct seeds.

### 4.4 Evaluate EXACTLY as the paper does (pin every command before running)
> The recon eval uses an **external** repo and an **assumed** mesh filename — both are runtime-verify items. **Pin the exact commit + command + I/O paths in `OPEN_QUESTIONS.md` and confirm before running** (do not leave "run neural_slam_eval" as prose).
```bash
# (a) ATE — rigid Horn, NO scale (paper convention). Reports cm (x100 internally).
python src/tools/eval_ate.py configs/Replica/${SCENE}.yaml

# (b) 3D recon (Acc / Comp / Comp.Ratio @5cm) + 2D Depth-L1 — EXTERNAL repo, PIN IT:
#   repo : https://github.com/JingwenWang95/neural_slam_eval   (RECORD the exact commit hash used)
#   mesh : $OUTPUT/mesh/<MESH>.ply   <- CONFIRM the actual filename from eval_recon.py (§4.0);
#                                       'final_mesh_eval_rec_culled.ply' is an ASSUMPTION until verified.
#   cmd  : <pin the exact eval_recon.py / neural_slam_eval invocation, its --config / --gt-mesh /
#           --rec-mesh args, and the expected stdout metric lines>  -> ESCALATE if any arg is ambiguous.
#   (calc_3d_metric: ICP-aligned, 450k pts, thresh 0.05 m, *100 -> cm; calc_2d_metric -> Depth-L1 cm)

# (c) semantic mIoU + pixAcc (every 50 frames) — CONFIRM whether it needs a config arg:
python src/tools/eval_segmentation.py configs/Replica/${SCENE}.yaml   # if the script ignores argv, drop it; verify first

# (d) optional vis.mp4:
python visualizer.py configs/Replica/${SCENE}.yaml
```
**Do not** use `Addons/eval/sim3_ate.py` here — Phase A must match the paper's rigid-only alignment.

### 4.5 Acceptance bar (tied to Table 1's 8-scene-average granularity)
Compute the **mean over the 8 scenes** (mean over runs first if multi-run) for each metric, then compare to Table 1:

| Metric | Paper (8-scene avg) | Pass band (recommend) |
|---|---|---|
| ATE-RMSE (cm) | 0.456 | ≤ 0.456 × 1.25 ≈ **0.57** |
| ATE-Mean (cm) | 0.397 | ≤ **0.50** |
| Depth-L1 (cm) | 0.766 | ≤ **0.96** |
| Acc (cm) | 1.942 | ≤ **2.43** |
| Comp (cm) | 1.702 | ≤ **2.13** |
| Comp.Ratio (%) | 96.624 | ≥ **94.0** |
| Semantic mIoU (%) (4 scenes) | 87.41 | ≥ **83.0** |

- Tolerance rationale: Issue #29 confirms **real per-scene variance** (paper averages 5 runs). With 8×1 runs, expect deviation; **±25 %** on error metrics is the recommended default. **The exact tolerance and run count are escalation items (OQ-2)** — confirm before declaring PASS/FAIL.
- **ScanNet / TUM are NOT part of the gate** (not reproducible from the public repo). Confirm dropped (OQ-8).

### 4.6 Artifacts to record (Phase A)
Under `Outputs/SNI-SLAM/phaseA_replica/`:
- `phaseA_per_scene.csv` (scene, run, ATE-RMSE, ATE-Mean, Depth-L1, Acc, Comp, CompRatio, mIoU, wall-s, peak-VRAM-MB, seed).
- `phaseA_8scene_average.txt` (the averaged row vs Table 1, with pass/fail per metric).
- per-scene `ckpts/*.tar`, `mesh/<MESH>.ply` (confirmed name), eval stdout logs (incl. the pinned recon-eval command + commit).
- `env_smoke.txt`, `gpu.txt`, `_failures.log`.

### 4.7 ⛔ DECISION GATE
**PASS** (all of):
1. All 8 Replica scenes ran to completion (or the user-confirmed compute-adaptive subset, OQ-2).
2. The **8-scene-average** ATE-RMSE, ATE-Mean, Depth-L1, Acc, Comp, Comp.Ratio fall in the pass bands (§4.5) at the user-confirmed tolerance.
3. (If mIoU evaluated) the 4-scene mIoU avg ≥ band.
4. No silent misconfig: sanity gates in §8.5 all green.

**On PASS** → proceed to Phase B. **On FAIL** → STOP, escalate with the per-scene table, the failing metrics, and your hypothesis (install mismatch? missing scenes? variance?). **Do not start Phase B** until the gate is resolved or the user explicitly waives it.

---

## 5. PHASE B — CRCD adaptation

> **`Addons/colab/run_crcd_4snippets.sh` is the STRUCTURAL template only** — its phase ordering (stage → preprocess → MoGe-2 depth → stereo sc_factor → SLAM → eval → ship) is what you mirror. **It is NOT reusable verbatim for snippet coverage** (§5.0). The depth pipeline source-of-truth for the 3 missing snippets is `crcd_depth_gen_remainder_20260616.sh`.

### 5.0 Snippet coverage — DO NOT "reuse verbatim"
- `run_crcd_4snippets.sh:41-45` has `SNIPPETS=(F3_007, C1_001, C2_001, F1_002)`. It **lacks E3_005, C3_001, G3_001** (3 of the benchmark 5) and **includes F1_002, F3_007** (not in the benchmark 5).
- **Action:** replace the SNIPPETS array with the **5 benchmark snippets** (§5.1). For **E3_005, C3_001, G3_001**, depth comes from the cache produced by **`crcd_depth_gen_remainder_20260616.sh`** (its `SNIPPETS` array at line 30-35 includes all three: `E3_005 E_3 005 265`, `C3_001 C_3 001 1527`, `G3_001 G_3 001 1987`), **not** from the 4-snippet script. Staging + preprocess + config authoring for those three must be **added** (they are not in the 4-snippet script).

### 5.1 The 5 prescribed snippets → config mapping
| Snippet | DDS config stem | Staged dir | Frames | Config exists? |
|---|---|---|---|---|
| C_1/001 | `c1_001` | `data/CRCD/C1_001` | 360 | **YES** (`c1_001_paperfaith_lrfix.yaml`) |
| C_2/001 | `c2_001` | `data/CRCD/C2_001` | 730 | **YES** (`c2_001_paperfaith_lrfix.yaml`) |
| E_3/005 | `e3_005` | `data/CRCD/E3_005` | 265 | **NO — must author** |
| C_3/001 | `c3_001` | `data/CRCD/C3_001` | 1527 | **NO — must author** |
| G_3/001 | `g3_001` | `data/CRCD/G3_001` | 1987 | **NO — must author** |
> Frame counts above are echoed from `crcd_depth_gen_remainder_20260616.sh` (E3_005=265, C3_001=1527, G3_001=1987). **ESCALATE (OQ-3)** to confirm the snippet→config naming and counts, and who derives `mapping.bound` for the 3 missing snippets (§5.6).

### 5.2 F:/ → Drive → /content staging
- CRCD lives on the user's **local F:/** drive. Colab cannot read F:/. **The user must first copy CRCD-Published to Drive** at `/content/drive/MyDrive/Datasets/CRCD-Published/` (raw `<EP>/snippet_<SID>/{rgb,rgbright,semantic_instance,groundtruth.txt,intrinsics.yaml}` + `cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`). If absent → **escalate** (blocked, exit 3).
- Per snippet, stage Drive→`/content` (tarball-first; per-item cp fallback) exactly as `run_crcd_4snippets.sh` Phase 1. Sentinel: `.STAGED`.

### 5.3 Preprocess + MoGe-2 + stereo-scaled depth
Run in the **DDS-SLAM env**. Per snippet. **For E3_005/C3_001/G3_001, if the Drive cache from `crcd_depth_gen_remainder_20260616.sh` exists, REHYDRATE it and skip steps 2–3** (§5.3 step 5).

1. **Preprocess (rectify):** `python Addons/preprocess/preprocess_crcd_published.py --snippet_dir <RAW> --calib_pkl <CALIB_PKL> --output_dir data/CRCD/<NAME>`. **Exact output names** (verified): `video_frames/NNNNNNl.png` + `video_frames/NNNNNNr.png` (rectified L/R), `masks/NNNNNN.png` (1=tool), `semantic_class/NNNNNN.png` (note: **no `l` suffix on masks/semantic**), `groundtruth.txt`, `rectified_calib.txt`. Sentinel `.PREPROCESSED`.
2. **MoGe-2 metric depth.** Symlink `video_frames/*l.png` → `_moge_in/<fid>-left.png` where `fid = basename minus 'l.png'` (e.g. `000123l.png` → `_moge_in/000123-left.png`) — this matches `generate_depth_moge.py`'s `*-left.png` glob and its `<fid>-left_depth.npy` output:
   ```bash
   cd data/CRCD/<NAME>; mkdir -p _moge_in _moge_npy
   for f in video_frames/*l.png; do fid=$(basename "$f" l.png); ln -sf "$PWD/$f" "_moge_in/${fid}-left.png"; done
   python "$DDS_REPO/Addons/depth/generate_depth_moge.py" --rgb _moge_in --out _moge_npy \
     --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0
   # generate_depth_moge.py writes ONLY _moge_npy/<fid>-left_depth.npy (float32 = depth_m*10000). NO PNG.
   # npy -> uint16 PNG via the NAMED extracted script (no inline block to copy):
   python "$DDS_REPO/Addons/depth/moge_npy_to_png.py" --in _moge_npy --out depth
   #   -> depth/<fid>.png ; clip(.,0,65535). At depth_scale 10000 + max_depth_m 5.0, max value = 50000 < 65535.
   ```
   Sentinel `depth/.DONE`.
3. **Stereo anchor → sc_factor.** **The GLOBAL benchmark spec mandates periodic (~every 100 frames) stereo re-scaling; the existing harness computes only a single frame-0 SGBM anchor.** This is a **directive conflict → HARD ESCALATION (OQ-4) before running any snippet** — do not silently default to frame-0.
   - The frame-0 SGBM block (verbatim source: `crcd_depth_gen_remainder_20260616.sh:117-150`) computes `disp = StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=7,...)`, `depth = baseline_m*fx/disp`, mask `0.05<d<3.0 & MoGe-valid`, `sc_factor = median(stereo/MoGe)` → `.sc_factor`.
   - **Periodic option (if the user requires the spec-mandated cadence):** `Addons/depth/generate_depth_stereo.py` is a *full per-frame* stereo generator, NOT a periodic anchor — the periodic-rescaling procedure does **not exist yet** and must be authored: run SGBM every ~100 frames, compute per-window `median(stereo/MoGe)`, and apply the per-window scale to the MoGe npy maps of that window before the npy→png step. **Specify and get sign-off on this procedure before implementing.** Default posture is **escalate, not run.**
4. **Apply sc_factor — HARD GATE (OQ-5).** Known co-scaling defect: with the DDS-SLAM SLAM loader, only `trunc` rescales with `sc_factor`; `range_d`, `near`, `far`, `depth_trunc` do **NOT**. When `sc_factor` deviates from 1, the SDF truncation/sampling bounds become inconsistent and reconstruction/depth can be silently corrupted. **Therefore:**
   - If `abs(log(sc_factor)) > 0.1` → **ABORT the snippet** and escalate (OQ-5). Do **not** complete a run and emit numbers behind this bug.
   - To proceed past the band, EITHER the user explicitly constrains snippets to `|log(sc_factor)| < 0.1`, OR you apply the **co-scaling patch**: every one of `range_d`, `near`, `far`, `depth_trunc`, `trunc` must be multiplied by `sc_factor` consistently in whichever config the SLAM actually reads (§5.4 resolves which config family that is). State the patched keys explicitly in the run log.
   - Whichever consumer is chosen for sc_factor (config patch vs baked-into-PNG, §5.4/§5.5), the patch target must be **the config/loader the SNI-SLAM run actually consumes**, never a DDS-SLAM config that SNI-SLAM never reads.
5. **Cache / rehydrate depth + sc_factor on Drive** at `/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth/*.png,.sc_factor}`. `crcd_depth_gen_remainder_20260616.sh` already generates this for **E3_005, C3_001, G3_001** — **rehydrate that cache, do not regenerate.**

### 5.4 Semantic-input AND data-plumbing — **CORE BLOCKERS, ESCALATE FIRST**
SNI-SLAM is run via its **own** repo (`python run.py configs/SNI-SLAM_CRCD_<NAME>.yaml`) with its **own** `BaseDataset`/loader — **NOT** `ddsslam.py`. The DDS-SLAM SLAM config family (`data.sc_factor`, `png_depth_scale` read at `datasets/dataset.py:195` as `png/png_depth_scale*sc_factor`, `mapping.bound`/`marching_cubes_bound`, `configs/CRCD/*paperfaith_lrfix.yaml`) is **only** consumed by `ddsslam.py`. **Two config families exist and must not be conflated.** Before Phase B you must resolve, against SNI-SLAM's actual loader (read `src/utils/datasets.py`), and ESCALATE any ambiguity:

**(P1) Depth-scale + sc_factor plumbing into SNI-SLAM (extend OQ-3/OQ-5).** Determine from the SNI-SLAM loader:
   - (a) **Which config key carries `png_depth_scale`** and confirm it accepts **10000** (Replica default is 1000 — surgical MoGe PNGs are ×10000). If the key/name differs, set it correctly in the SNI-SLAM config.
   - (b) **How `sc_factor` is applied.** SNI-SLAM has **no `data.sc_factor` field.** Choose ONE and state it: (i) **bake `sc_factor` into the depth PNGs at generation time** (multiply the MoGe npy by `sc_factor` before `moge_npy_to_png.py`, so the PNGs are already metric and `png_depth_scale=10000` alone suffices — simplest, recommended), OR (ii) **add a loader patch** that multiplies read depth by `sc_factor`. If (i), the OQ-5 co-scaling concern instead lands on SNI-SLAM's own near/far/trunc keys — co-scale those (see (P2)).
   - (c) **Which SNI-SLAM keys set the workspace bound vs marching-cubes bound** (Replica uses its own `bound`/`marching_cubes_bound` schema; find the equivalents). Set them to the surgical volume (§5.6).
   - **Until (a)–(c) are resolved, the SNI-SLAM CRCD run cannot consume the stereo-scaled MoGe depth at correct metric scale — this is a blocker, not a default.**

**(P2) CRCD dataset class + semantic decoupling.** `BaseDataset.__getitem__` hard-reads `semantic_class/*.png` + `semantic_classes.pkl`/`num_semantic_class.pkl`. Regardless of the semantic decision you must:
   1. **Author a CRCD dataset class** (a new file in the SNI-SLAM repo, e.g. `src/utils/datasets_crcd.py`, registered like the Replica loader) reading CRCD `video_frames/*l.png` + generated `depth/*.png` (×10000) + `groundtruth.txt`, **decoupled from the Replica `semantic_class` requirement** OR shipping **placeholder** `semantic_class/*.png` + a generated `semantic_classes.pkl`/`num_semantic_class.pkl` so `__getitem__` doesn't crash. Map CRCD intrinsics (H=720, W=1280, fx=fy=1096.696, cx=622.808, cy=383.126) replacing Replica's. Set `png_depth_scale` per (P1a) and the bounds per (P1c).
   2. **Pre-flight H/W vs patch-14 / img_size-518 assertion (DINO2SEG).** **720 is NOT a multiple of 14** (720/14 = 51.43) → the seg-head reshape can fail. Before any SLAM step, assert the (H,W) fed to the head are compatible with the patch size and `img_size=518` upsample, accounting for `crop_edge`/`edge=10`. If incompatible, resolve explicitly: crop H to the nearest multiple of 14 (e.g. 714 or 700) and/or resize, and apply the SAME crop/resize to RGB, depth, and (placeholder) semantics consistently. Read the actual patch/img_size constants in `dinov2_seg.py` (do not assume 14/518 if the repo differs). Document the chosen crop.

**(P3) Semantic-input decision — ESCALATE (OQ-6); pick ONE, do not guess. Per chosen option the deliverable is concrete:**
   - (a) **Supply surgical semantic masks.** CRCD-Published has a binary tool `masks/` and a 4-class `semantic_class/` from preprocess. Confirm whether either is the agreed SNI-SLAM input and how to map it to a 52→N-class head target. *Deliverable: the mapping + the `semantic_classes.pkl`/`num_semantic_class.pkl` regenerated for N classes.*
   - (b) **Train a new DINOv2 seg head on CRCD labels.** Needs CRCD semantic labels (likely absent). *Deliverable: training data spec + new `dinov2_crcd.pth`.*
   - (c) **No-semantic ablation.** Zero the semantic stream (`w_semantic`/`w_feature`→0). **This deviates from the paper method** and must be labelled an ablation, not a faithful SNI-SLAM run. *Deliverable: the zeroed keys + the placeholder semantics from (P2.1) so the loader runs.*
   - Also confirm **mIoU is dropped for CRCD** (CRCD metrics = ATE/PSNR/SSIM/LPIPS/Depth-L1 only).
   - **Inherent mismatch to REPORT (not a bug):** CRCD has deformable tissue + breathing/tool motion; SNI-SLAM assumes a **static scene**. Expect tracking/mapping degradation; document it.

### 5.5 Per-snippet config authoring (3 missing configs + keeping families consistent)
- **DDS-harness-side staging config** (only for data layout/bounds, used by harness expectations): create `configs/CRCD/<stem>_paperfaith_lrfix.yaml` from the c1/c2 template — set `timesteps` (E3_005=265, C3_001=1527, G3_001=1987), `mapping.bound` + `marching_cubes_bound` (hand-derived, §5.6), `data.datadir: data/CRCD/<NAME>`, `data.output`, `data.exp_name: demo`.
- **SNI-SLAM-side config** (the one `run.py` actually consumes): `configs/SNI-SLAM_CRCD_<NAME>.yaml` referencing the new CRCD dataset class (§5.4 P2), the staged data, `png_depth_scale` per (P1a), `sc_factor` consumer per (P1b), and the bound keys per (P1c). **Keep the two families consistent** (same datadir, same metric scale, same surgical bounds). The sc_factor co-scaling patch (§5.3 step 4) applies to **this** config's near/far/trunc keys when sc_factor is baked into PNGs.

### 5.6 Hand-deriving mapping.bound (procedure)
For each new snippet, after depth is generated:
```bash
python - <<'PY'
import cv2, numpy as np, glob
S='data/CRCD/<NAME>'
d=cv2.imread(sorted(glob.glob(S+'/depth/*.png'))[0],cv2.IMREAD_UNCHANGED).astype(np.float32)/10000.0
v=d[d>0.01]; print('depth p2/median/p98 (m):',np.percentile(v,2),np.median(v),np.percentile(v,98))
g=np.loadtxt(S+'/groundtruth.txt',comments='#')[:,1:4]
print('GT extent (m):', (g.max(0)-g.min(0)))
PY
# Z bound ~[p2-pad, p98+pad] of depth; X/Y bound to GT extent + ~10cm pad. (cf. C1_001 [[-0.08,0.13],[-0.02,0.18],[0.68,0.90]])
```
**Escalate the derived bounds for confirmation (OQ-3)** — wrong bounds silently destroy reconstruction. Set the SAME values in both config families (§5.5).

### 5.7 EXACT SLAM run + trajectory export (all 5 snippets)
Run SNI-SLAM in the `sni` env, one snippet at a time, with failure isolation. **SNI-SLAM stores poses in the ckpt `.tar` as `estimate_c2w_list`, NOT as `est_c2w_data.txt`. You MUST export them** with the new exporter — `kitti_to_tum.py` does the OPPOSITE (reads est_c2w_data.txt → TUM pairs) and cannot create the file.
```bash
conda activate sni
for NAME in C1_001 E3_005 C3_001 G3_001 C2_001; do
  T0=$(date +%s)
  if ! python -W ignore run.py configs/SNI-SLAM_CRCD_${NAME}.yaml; then
     echo "FAILED_SLAM_$NAME" >> "$DRIVE_OUT/_failures.log"; echo "[$NAME] crashed — continuing"; continue
  fi
  # export estimated trajectory from the ckpt .tar -> est_c2w_data.txt (16 floats/line, c2w, t at [3,7,11]):
  CKPT=$(ls -t "$SNI_OUT_${NAME}"/ckpts/*.tar | head -1)
  RUN="$SNI_OUT_${NAME}/demo"; mkdir -p "$RUN"
  python "$DDS_REPO/Addons/eval/sni_export_traj.py" --ckpt "$CKPT" --out "$RUN/est_c2w_data.txt" \
     || { echo "FAILED_EXPORT_$NAME" >> "$DRIVE_OUT/_failures.log"; continue; }
  echo "[$NAME] elapsed $(( ($(date +%s)-T0)/60 )) min"
done
```
- `sni_export_traj.py` loads `estimate_c2w_list` ([N,4,4] or [N,3,4]), flattens row-major to the format `sim3_ate.load_est`/`generate_video.py` consume, and **warns + tells you to escalate (OQ-10) if far fewer than N frames have a non-trivial estimated pose** (i.e. SNI-SLAM does not keep per-frame poses for all frames). Confirm via OQ-10 before headlining ATE.
- **Renders for PSNR/SSIM/LPIPS + depth panel:** the harness expects rendered RGB as `<OUT>/[0-9]*.jpg` and rendered depth as `<OUT>/depth/<frame:04d>.png` (uint16 × 10000). **OQ-10: confirm SNI-SLAM emits per-frame rendered RGB+depth in this form; if it does not, the rendering metrics and the rendered-depth panel cannot be produced — escalate before Phase B eval.**

### 5.8 Output layout under MyDrive/Outputs
```
/content/drive/MyDrive/Outputs/SNI-SLAM/crcd_<DATE>/
  <NAME>/                      # one per snippet (C1_001, E3_005, C3_001, G3_001, C2_001)
    render_eval.txt, render_eval.csv
    sim3_metrics.txt
    summary.txt
    depth_l1.txt               # number once OQ-1 confirms the reference; else 'PENDING — OQ-1'
    <NAME>_6panel.mp4
    payload.tgz                # demo/ (est_c2w_data.txt) + ckpts/ + depth/ + renders_rgb/ + the metric .txt/.csv
    .DONE
  _render_summary.csv
  _failures.log
  COMBINED_SUMMARY.txt         # written by aggregate_crcd_generic.py (§7)
  runbook.log
  OPEN_QUESTIONS.md
```

---

## 6. CRCD evaluation (DDS-SLAM harness)

> All in the **DDS-SLAM env**. `<OUT>` = SNI-SLAM's per-snippet output root (with renders + `depth/`), `<RUN>` = `<OUT>/demo` (holds the exported `est_c2w_data.txt`), `<NAME>` ∈ {C1_001,…}, `<DST>` = `$DRIVE_OUT/crcd_<DATE>/<NAME>`.

### 6.1 ATE (sim3-aligned) — headline trajectory metric
```bash
python Addons/eval/sim3_ate.py \
  --est <RUN>/est_c2w_data.txt \
  --gt  data/CRCD/<NAME>/groundtruth.txt \
  --name "SNI-SLAM <NAME>" --out <DST>/sim3_metrics.txt
```
Reports Sim3 ATE rmse/mean/median/max (mm), recovered scale s, est/GT path ratio, |Pearson| dom axis (also prints rigid ATE labelled "do NOT headline"). The exporter must have produced `est_c2w_data.txt` first (§5.7).

### 6.2 PSNR / SSIM / LPIPS
> **Required code edit (already applied in this repo):** `Addons/eval/eval_rendering.py` enforces `--sequence` against `choices=list(PAPER_REFERENCES.keys())`. The keys now include **`CRCD (C1_001)`, `CRCD (C2_001)`, `CRCD (C3_001)`, `CRCD (E3_005)`, `CRCD (G3_001)`** (the 3 new ones were added so all 5 snippets pass argparse — previously E3_005/C3_001/G3_001 would `SystemExit`). If running against an unpatched copy, add those three `None/None/None` entries first.
```bash
pip install -q lpips    # if missing
python Addons/eval/eval_rendering.py \
  --gt_dir data/CRCD/<NAME>/video_frames \
  --render_dir <OUT> \
  --name "SNI-SLAM" --sequence "CRCD (<NAME>)" \
  --output_csv <DST>/render_eval.csv \
  --summary_csv $DRIVE_OUT/crcd_<DATE>/_render_summary.csv 2>&1 | tee <DST>/render_eval.txt
```
GT pairs by filename index against `video_frames/*l.png`; renders are `<OUT>/[0-9]*.jpg`. CRCD has **no paper reference** (post-dates the paper) → reference values print as None.

### 6.3 Depth-L1 — fixed metric; reference unconfirmed (OQ-1, escalate BEFORE Phase B)
Depth-L1 is one of the **five fixed CRCD metrics**, but the repo has **no Depth-L1 script** and CRCD has **no GT depth** (MoGe-2 is generated). A `PENDING` placeholder is a gap to **surface**, not bury.
- **Recommended candidate (state this to the user when escalating):** rendered SLAM depth vs the **MoGe-2 input depth** (self-consistency), masked to valid pixels (`d>0.01` AND inside the tool mask is excluded), mean |Δ| in mm. Alternatives: vs frame-0 stereo-SGBM depth, or vs held-out stereo depth.
- **Pairing + script:** pair `<OUT>/depth/<frame>.png` (rendered, ×10000) with `data/CRCD/<NAME>/depth/<fid>.png` (MoGe input, ×10000) by frame index; both divided by 10000 to metres; report `mean(|rendered-moge|)` over valid pixels in mm. This is ~15 lines; **author it as `Addons/eval/depth_l1.py` once the user confirms the reference**, then write the number to `<DST>/depth_l1.txt`. **Until confirmed, emit `depth_l1.txt = "PENDING — OQ-1"` and escalate; do not fabricate a number.**

### 6.4 Canonical 6-panel video
```bash
UNC=""; [ -d "<OUT>/uncert" ] && UNC="--uncert_dir <OUT>/uncert"
python Addons/viz/generate_video.py \
  --rgb_input_dir data/CRCD/<NAME>/video_frames --rgb_input_pattern '*l.png' \
  --rgb_output_dir <OUT> --rgb_output_pattern '[0-9]*.jpg' \
  --depth_input_dir data/CRCD/<NAME>/depth --depth_output_dir <OUT>/depth \
  --depth_norm robust --png_depth_scale 10000 \
  --seg_dir data/CRCD/<NAME>/semantic_class --seg_pattern '*.png' --skip_raw_seg --seg_classmap $UNC \
  --trajectory_est <RUN>/est_c2w_data.txt --trajectory_gt data/CRCD/<NAME>/groundtruth.txt --trajectory_raw \
  --output <DST>/<NAME>_6panel.mp4 --fps 15
```
- **`--png_depth_scale 10000` is passed explicitly** (its default is `None`): both input and output depth panels then divide by the same metric scale before robust-normalizing, so they are comparable. `generate_video.py` applies `--png_depth_scale` in `colormap_depth(...)` for both `depth_input_dir` and `depth_output_dir`.
- `--rgb_input_pattern '*l.png'` and `--rgb_output_pattern '[0-9]*.jpg'` override the script defaults (`*_gt.png` / `[0-9]*.png`) — required so input frames match `NNNNNNl.png` and SLAM renders match `*.jpg`.
- Panels: (1) Input RGB, (2) Rendered RGB, (3) Input Depth, (4) Output Depth, (5) Seg overlay, (6) Trajectory raw + Sim3-aligned.

### 6.5 Recommended diagnostics (run for every snippet)
- Trajectory overlay vs GT (Sim3-aligned, in the 6-panel + standalone PNG).
- Per-frame ATE curve and per-frame PSNR curve (from the eval CSVs).
- Depth-error heatmap (rendered vs MoGe-2 input — same pairing as §6.3, pending OQ-1).
- Sampled semantic-overlay frames (documents the domain mismatch if option (c)).
- Keyframe coverage histogram.
- GT motion-profile sanity (extent, path length, per-frame motion, active fraction, sub-SNR sentinels) — reuse `run_crcd_4snippets.sh` Phase 2 block. Most CRCD snippets are sub-SNR; C_2/001 is best-case (largest extent).

---

## 7. Standard CLI contract — `run_sni_slam.sh`

The master orchestrator calls one bash line per method. Provide `run_sni_slam.sh` with this interface (re-validate the stage-marker names against `00_COMMON.md` once obtained).

**Invocation:**
```bash
bash run_sni_slam.sh <phase> [snippet|scene] [seed]
#   <phase> : all | phaseA | phaseB | <single CRCD NAME, e.g. C1_001>
#   seed    : default 0
```
**Environment variables (read, with defaults):**
| Var | Default | Meaning |
|---|---|---|
| `SNI_REPO` | `/content/SNI-SLAM` | SNI-SLAM clone path |
| `DDS_REPO` | `/content/DDS-SLAM` | harness path |
| `DRIVE_OUT` | `/content/drive/MyDrive/Outputs/SNI-SLAM` | output root |
| `DRIVE_CRCD` | `/content/drive/MyDrive/Datasets/CRCD-Published` | raw CRCD on Drive |
| `SEED` | `0` | determinism seed |
| `GPU_ABORT` | `0` | if `1`, abort when VRAM < floor (OQ-7) |
| `REPLICA_RUNS` | `1` | runs/scene in Phase A (set 5 for full-faithful, OQ-2) |

**Stage markers** (grep-able by the orchestrator):
```
[STAGE] sni:env:start / sni:env:ok
[STAGE] sni:phaseA:<scene>:start / :ok / :fail
[STAGE] sni:phaseA:gate:PASS|FAIL
[STAGE] sni:phaseB:<NAME>:stage|preprocess|depth|scfactor|slam|export|eval|video|ship  (each :start/:ok/:fail)
[STAGE] sni:phaseB:<NAME>:DONE
[STAGE] sni:aggregate:ok
```
**Exit codes:** `0` all requested stages done (per-item failures isolated); `2` env/setup failure; `3` data/weights/`00_COMMON.md` missing; `4` Phase-A gate FAILED; `5` blocked on an unresolved escalation (semantic input OQ-6, missing config/plumbing OQ-3, sc_factor band OQ-5, Depth-L1 OQ-1, stereo cadence OQ-4).
> **Failure isolation:** a single scene/snippet failure → `_failures.log` + continue; never non-zero exit unless *every* requested item failed.

**Aggregation (cross-snippet and, for the master, cross-method):**
> **Do NOT use `aggregate_ab.py`** — it is hardcoded to the DDS-SLAM A/B/C cells (`c1_001_canon_base` / `trail3_*`), globs `<cell>_s<seed>/payload.tgz`, and parses `render_metrics.txt`. None of that matches this layout (`<NAME>/payload.tgz`, files `render_eval.csv` / `sim3_metrics.txt`). Running it yields an empty/irrelevant table.
> Use the **new general aggregator** authored for this run book, which is parameterized over snippet names and reads the files this book actually emits:
```bash
# single method, cross-snippet:
python Addons/eval/aggregate_crcd_generic.py \
  --root $DRIVE_OUT/crcd_<DATE> \
  --names C1_001 E3_005 C3_001 G3_001 C2_001 \
  --out  $DRIVE_OUT/crcd_<DATE>/COMBINED_SUMMARY.txt
# cross-method (master): pass one METHOD=PATH per method to --method-roots.
```
It reads `<NAME>/render_eval.csv` (or `render_eval.txt`, or the file inside `payload.tgz`) for PSNR/SSIM/LPIPS and `<NAME>/sim3_metrics.txt` for Sim3 ATE + |Pearson|, and prints a per-snippet table + `MEAN+/-STD` row.

---

## 8. Failure modes, determinism, checkpointing, logging

### 8.1 Determinism / seeding
- Set and **log** the seed for python `random`, `numpy`, `torch` (`manual_seed` + `cuda.manual_seed_all`), and DataLoader workers. Disable TF32 in the harness step (`run_cell.sh` pattern: `torch.backends.cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=False`).
- SNI-SLAM uses `torch.multiprocessing` + shared memory → **full bitwise determinism is not guaranteed** (Issue #29). Record the seed; report variance rather than claiming determinism. For Phase A with `REPLICA_RUNS>1`, vary seed per run and report mean±std.

### 8.2 GPU / VRAM / runtime logging (required every stage)
At each stage start: `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader`. During SLAM, sample peak `memory.used` (background poller). Log `name, total_VRAM, peak_used_VRAM, wall_seconds` to the per-item CSV → empirical min-VRAM floor for OQ-7.

### 8.3 Checkpointing / resume
- **Sentinels:** `.STAGED`, `.PREPROCESSED`, `depth/.DONE`, `.sc_factor`, per-snippet `.DONE`. Phase A: per-scene `ckpts/*.tar` + `<scene>.DONE`.
- **Skip logic:** check the sentinel and (for SLAM) `est_c2w_data.txt` line-count ≥ frames (the `run_crcd_4snippets.sh:419-425` pattern) before each stage. Re-runs skip completed work.
- Depth cache rehydration from `/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/...` avoids regenerating MoGe-2 depth (esp. E3_005/C3_001/G3_001).

### 8.4 Logging conventions
- One `runbook.log` per run via `exec > >(tee -a "$LOG") 2>&1`.
- `phase()` banners with timestamps (mirror `run_crcd_4snippets.sh`).
- `_failures.log` accumulates `FAILED_<STAGE>_<NAME>` lines.
- All escalations also written to `OPEN_QUESTIONS.md`.

### 8.5 Sanity gates (must run; all green to trust a result)
1. **Frame-count alignment:** `#rgb == #depth == #semantic == #GT`. For CRCD: `#video_frames/*l.png == #depth/*.png`, and `est_c2w_data.txt` lines ≥ frames. Mismatch → fail loud.
2. **Index alignment RGB↔depth↔semantics↔GT:** frame *i* ↔ GT[*i*] by index; verify file-id ordering is identical across `video_frames`, `depth`, `semantic_class`, and GT row count matches. Any subsample/`trainskip` applied **consistently** to all four streams.
3. **Sim3 alignment correctness:** after `sim3_ate.py`, assert recovered scale `s` finite and `0.05 < s < 20`; wildly off `s` (with frame-0 stereo anchor applied) → suspect depth-scale plumbing (§5.4 P1). Rigid ATE ≫ Sim3 ATE = pure global-scale confound (expected on monocular-depth scale).
4. **sc_factor band (HARD):** if `|log(sc_factor)| > 0.1` AND the OQ-5 co-scaling patch is not applied → **the snippet was aborted in §5.3 step 4**; this gate must confirm no snippet emitted numbers behind the unfixed bug.
5. **Bound covers the scene:** assert GT extent + frame-0 depth p2..p98 fall inside the SNI-SLAM bound keys (§5.4 P1c); else reconstruction silently collapses.
6. **Depth scale consistency:** confirm SNI-SLAM's `png_depth_scale` key = **10000** (matches `--depth_scale 10000`), and that sc_factor is applied exactly once (baked into PNG OR loader patch — never both).
7. **Semantic stream state:** log whether the DINO2SEG head ran and on which domain; for option (c) confirm `w_semantic`/`w_feature` are zeroed (result honestly labelled an ablation). Also confirm the (H,W) patch-14/img_size-518 pre-flight (§5.4 P2.2) passed.
8. **Render outputs exist:** before eval, assert `<OUT>/[0-9]*.jpg` count > 0 and `<OUT>/depth/*.png` count > 0 (`run_crcd_4snippets.sh:529-543` render-relocation pattern; move top-level `*.jpg` into `renders_rgb/` before tar).
9. **Trajectory export sanity:** `sni_export_traj.py` written-pose count ≥ 0.9 × frames; if far below → OQ-10 (SNI-SLAM not keeping per-frame poses) before headlining ATE.

### 8.6 Common failure modes & responses
| Symptom | Likely cause | Response |
|---|---|---|
| `import pytorch3d` fails | wheel/CUDA mismatch | rebuild `sni` env with exact cu113/pyt1110 wheel (§3.3) |
| crash reading `semantic_class/*.png` on CRCD | Replica loader hard-requires masks | author CRCD dataset class / placeholder PNGs (§5.4 P2) |
| seg-head reshape error on CRCD | H/W not multiple of patch-14 (720%14≠0) | apply the §5.4 P2.2 crop/resize; verify crop_edge/edge=10 |
| hang in Mapper/Tracker | `share_memory_()` fragility (#11/#27) | reduce processes; `keyframe_device/feature_device: cpu` |
| Sim3 ATE huge, rigid huge, s far from 1 | depth-scale plumbing wrong | re-check png_depth_scale key + sc_factor consumer (§5.4 P1; gates §8.5.3/6) |
| `est_c2w_data.txt` missing after SLAM | exporter not run / wrong ckpt key | run `sni_export_traj.py`; check ckpt key (§5.7) |
| 0 rendered frames despite render | renders at output root | move top-level `*.jpg` into `renders_rgb/` before tar (§8.5.8) |
| PSNR/SSIM/LPIPS empty | SNI-SLAM didn't render per-frame RGB | OQ-10 — method may not emit renders |
| `aggregate_*` empty table | used `aggregate_ab.py` (hardcoded) | use `aggregate_crcd_generic.py` (§7) |

---

## 9. Deliverables checklist

**Phase A — `Outputs/SNI-SLAM/phaseA_replica/`:**
- [ ] `env_smoke.txt`, `gpu.txt`
- [ ] `phaseA_per_scene.csv` (all 8 scenes × runs, all 6 metrics + mIoU + wall-s + peak-VRAM + seed)
- [ ] `phaseA_8scene_average.txt` (averaged row vs Table 1, pass/fail per metric, tolerance used)
- [ ] per-scene `ckpts/*.tar`, confirmed `mesh/<MESH>.ply`, eval stdout logs (incl. pinned recon-eval commit+command)
- [ ] **Decision-Gate verdict** (PASS/FAIL) recorded

**Phase B — `Outputs/SNI-SLAM/crcd_<DATE>/`:**
- [ ] per-snippet (×5): `render_eval.{txt,csv}`, `sim3_metrics.txt`, `summary.txt`, `depth_l1.txt` (number or `PENDING — OQ-1`), `<NAME>_6panel.mp4`, `payload.tgz` (incl. `demo/est_c2w_data.txt`), `.DONE`
- [ ] diagnostics: trajectory overlay, per-frame ATE/PSNR curves, depth-error heatmap (pending OQ-1), sampled semantic overlays, keyframe coverage, GT motion-profile sanity
- [ ] `_render_summary.csv`, `_failures.log`, `runbook.log`, `OPEN_QUESTIONS.md` (escalations + resolutions)
- [ ] `COMBINED_SUMMARY.txt` produced by **`aggregate_crcd_generic.py`** (not `aggregate_ab.py`) + cross-method row for the master table

---

## 10. Open questions to escalate (explicit)

Raise each **before** the stage that depends on it; log all in `OPEN_QUESTIONS.md`. Items marked **HARD GATE** abort the affected snippet/scene until resolved.

- **OQ-1 (Depth-L1 reference) — HARD GATE for the metric.** Fixed metric, no script, no CRCD GT depth. Recommended candidate: rendered SLAM depth vs MoGe-2 input (self-consistency), masked. Confirm the reference → author `Addons/eval/depth_l1.py` (pairing in §6.3) → emit the number; until then `depth_l1.txt = PENDING — OQ-1`.
- **OQ-2 (Phase-A scope + tolerance).** Full **8×5** Replica average (paper-faithful, ~40 runs) or **8×1** compute-adaptive? Confirm the numeric tolerance band (§4.5). If `./data/replica` is a subset, is Habitat-Sim scene generation in scope? *Blocks the gate definition.*
- **OQ-3 (Missing configs + bounds + SNI-SLAM plumbing keys).** Confirm snippet→config mapping + counts (E3_005=265, C3_001=1527, G3_001=1987). Who derives/confirms `mapping.bound` for the 3 missing snippets (§5.6)? **AND** confirm SNI-SLAM's loader keys for `png_depth_scale` (accepts 10000?) and bound/marching-cubes (§5.4 P1a/P1c). *Blocks 3 of 5 CRCD runs + depth plumbing.*
- **OQ-4 (Stereo-scaling cadence) — HARD GATE; directive conflict.** GLOBAL spec mandates **periodic (~per-100-frame)** stereo re-scaling; harness implements only a **single frame-0 anchor** (and `generate_depth_stereo.py` is a full per-frame generator, not a periodic anchor). The periodic procedure does not exist. **Do not default to frame-0** — get an explicit user waiver for frame-0, OR sign-off + spec for the periodic procedure (§5.3 step 3). *Blocks depth scaling.*
- **OQ-5 (sc_factor co-scaling bug) — HARD GATE.** When sc_factor deviates from 1, only `trunc` rescales; `range_d/near/far/depth_trunc` don't → silent corruption. **Abort any snippet with `|log(sc_factor)|>0.1`** unless the co-scaling patch (all five keys, in the config the SLAM reads) is applied or the user constrains snippets to the band. *Affects numeric validity.*
- **OQ-6 (CRCD semantic input) — HARD GATE; THE blocker.** Choose (a) supply surgical masks (source + class mapping), (b) train a CRCD seg head (labels?), or (c) no-semantic ablation (documented deviation). Per choice the deliverable is in §5.4 P3. Confirm mIoU dropped for CRCD. *Determines as-published vs ablation.*
- **OQ-7 (GPU/VRAM policy).** Confirm the min-VRAM floor + abort threshold for SNI-SLAM (fits T4 with CPU offload). Set `GPU_ABORT` accordingly.
- **OQ-8 (ScanNet/TUM scope).** Confirm Tables 3 & 4 are **dropped** from the gate (not reproducible from the public repo).
- **OQ-9 (CVPR vs TPAMI).** Is the fixed target **CVPR-2024 SNI-SLAM** (shipped code) or does **TPAMI-2025 SNI-SLAM++** supersede it for this benchmark?
- **OQ-10 (Render + pose export). HARD GATE for rendering metrics.** Does SNI-SLAM (i) keep per-frame estimated poses for ALL frames in `estimate_c2w_list` (verify via `sni_export_traj.py` written-count), and (ii) emit per-frame rendered RGB (`<OUT>/[0-9]*.jpg`) + depth (`<OUT>/depth/*.png`)? If (ii) is no, PSNR/SSIM/LPIPS + the rendered-depth panel cannot be produced — escalate before Phase B eval.
