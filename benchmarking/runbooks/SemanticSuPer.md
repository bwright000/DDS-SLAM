> **Resolved decisions apply.** See *Resolved Benchmark Decisions* in [00_COMMON.md](00_COMMON.md) — they override any “escalate”/“open question” below: Depth-L1 = input-vs-output per model; periodic ~100f stereo rescaling; per-paper semantic mirroring on CRCD 4 classes; missing configs (e3_005/c3_001/g3_001) are the agent's job.

# Agent Run Book — Semantic-SuPer (Semantic-aware Surgical Perception Framework)

> Benchmark Run Book — one document, one method. Standing orders for a **fresh autonomous agent** that will be handed the paper + the local DDS-SLAM repo **later**. Read this entire document before touching anything. When this document and a future instruction conflict, **STOP and escalate** (Section 0.3).
>
> **Repo facts in this runbook were verified against the working tree on branch `diagnosis-live`** (commit `a92f381`). Every path/glob/filename below was confirmed to exist (or confirmed absent) at authoring time. Where the repo and the global benchmark spec disagree, the disagreement is named explicitly and routed to an escalation — never silently reconciled.

---

## 0. Mission & non-negotiables

### 0.1 Mission
Benchmark **Semantic-SuPer** for a **published** comparison paper across two regimes:
1. **PHASE A — Own-dataset reproduction**: reproduce the method's published headline result on its **own** dataset, evaluated **exactly as the paper does**, and pass a hard **decision gate**.
2. **PHASE B — CRCD adaptation**: run the method on **5 CRCD surgical snippets** (`C_1/001`, `E_3/005`, `C_3/001`, `G_3/001`, `C_2/001`), evaluated with the **DDS-SLAM CRCD harness** (ATE-sim3, PSNR, SSIM, LPIPS; Depth-L1 is **BLOCKED**, Section 6) + the canonical **6-panel video**.

This is one of **four** method run books (Semantic-SuPer, SNI-SLAM, SemGauss-SLAM, SGS-SLAM). A master orchestrator calls every method through the **identical CLI contract** in Section 7.

### 0.2 Non-negotiables (precision, no assumptions)
- **NO ASSUMPTIONS.** Every load-bearing fact (a path, a class ontology, a frame count, a metric definition, a scale factor, a tolerance) must be **read from the paper, the repo, or the data** — never guessed. If it cannot be found, it is an **escalation**, not a default.
- **Reproducibility first.** Pin versions (and commit SHAs for cloned upstreams), seed RNGs, log GPU/VRAM/runtime, record every command exactly as run.
- **Failure isolation.** One snippet or one phase failing must **never** abort the rest. Catch, log to `_failures.log`, continue.
- **Resumability.** Every expensive stage is sentinel-gated (`.STAGED` / `.PREPROCESSED` / `depth/.DONE` / `.sc_factor` / `.DONE`). Re-running must skip completed work.
- **Do not edit the repo model to inject seeds/flags.** Use override-config files (the `run_cell.sh` pattern), never in-place edits to `ddsslam.py`. Harness *eval* scripts may receive the **trivial, behavior-preserving** edits explicitly authorized in this runbook (Section 6, Q-SEQCHOICE) — every such edit is recorded as a diff in `${DRIVE_ROOT}/_harness_diffs/`.

### 0.3 Clarification Protocol (how/when to escalate)
When you hit a **load-bearing unknown or a load-bearing choice**, do **not** pick one. Instead:
1. **STOP** the affected unit of work (one snippet / one phase). Keep other units running.
2. Write a numbered entry to `${DRIVE_ROOT}/_escalations.md` containing: (a) the exact decision, (b) why it is load-bearing (what downstream result it changes), (c) the concrete options with trade-offs, (d) what you need from the user to proceed.
3. Mark the unit `BLOCKED_ESCALATION` in `_failures.log` and emit stage marker `STAGE_BLOCKED` (Section 7.4).
4. Continue with all non-blocked units.

**Escalate immediately (do not guess) for any item in Section 10.** The most important for this method:
- **Q1 — Which code path = "reproduce Semantic-SuPer"** — upstream `Python-SuPer/run_semantic_super.py` (native pixel reprojection error, the **only** path that can support a "reproduce Semantic-SuPer" claim) **vs** DDS-SLAM's data-port through `ddsslam.py` (a render-PSNR comparison vs a *different* method's numbers, **NOT** a reproduction of Semantic-SuPer Table I). See Section 4.0 — this also determines whether Phase A is publishable at all.
- **Q2 — CRCD semantic ontology** — Semantic-SuPer's 3 classes `{tool, beef, chicken}` do not exist on CRCD.

---

## 1. Orientation / required reading

Read these **before** running anything. Do not skim.

### 1.1 The paper (arXiv:2210.16674, "Semantic-SuPer")
Exact sections to read and what to extract:
- **Sec III (Method)** — surfel model + ED graph; **Sec III-A/B** for the loss terms (point-to-plane ICP, **JSD-weighted semantic ICP**, **semantic-boundary morphing loss**, ARAP/face/quaternion regularization, SSIM render loss via Pulsar). Extract the **loss weights**: `lambda_morph=10`, `lambda_sim=1`, `lambda_reg=10`.
- **Sec IV-A (Experimental setup)** — hardware (Intel i9-7940X + **RTX 2080**), the **dVRK ex-vivo dataset** (4 trials, 150 frames each, rectified 640×480, green-pin GT), depth = **Monodepth2 stereo** fine-tuned on-domain, segmentation = **DeepLabV3+ (3 classes)** trained K-fold.
- **Table I (headline)** — **reprojection error in PIXELS**, per-trial (Lab1–Lab4), two numbers per cell ("all points" / "edge points"), each `mean(std)`. **This is the reproduction target.** There is **no** dataset-wide average.
- **Table II** — loss-term ablation. **Table III** — segmentation quality (Hausdorff / F1) vs reproj error.
- Note explicitly: **no ATE, no sim3, no PSNR** in the paper. The native metric is **2D pixel reprojection error** against green-pin tracks. The camera is **static** per trial.

### 1.2 The official repo (`github.com/ucsdarclab/Python-SuPer`) — NOT present locally
**Confirmed absent from this repo** (`find . -name run_semantic_super.py` → empty; `Python-SuPer/` does not exist). Path A is therefore **not executable until you clone it** — see Section 4.A.0. Read first, in order:
- `README.md` — both install paths (Docker, conda), the **Google-Drive links** for data + DeepLabV3+ ckpt + Monodepth2/RAFT weights, and the warning that conda env build "may take 7 hours".
- `run_semantic_super.py` — the entry point + its **flags** (`--load_seg --seg_dir`, `--load_depth`, `--num_classes 3`, `--mesh_step_size`, `--edge_ids`, `--sf_soft_seg_point_plane`, `--sf_bn_morph`, `--mesh_rot --mesh_face`, `--render_loss`, `--tracking_gt_file`, `--use_derived_gradient`).
- `seg/` — `train.sh`, `inference.sh`, and the `DeepLabV3+/` output layout.
- `resources/environment.yaml` and `docker/super_docker.Dockerfile` — the env pins.

### 1.3 The local DDS-SLAM repo (the **harness** for Phase B + the CRCD data port)
- **There is no `00_COMMON.md` checked in.** The shared spec lives in the runbook scripts. Read these as the authoritative shared harness contract (paths verified present):
  - `Addons/colab/run_crcd_4snippets.sh` — the canonical **end-to-end CRCD batch** (Phases 0–7). **Use it as a STRUCTURAL REFERENCE ONLY.** Its `SNIPPETS` array hardcodes `F3_007, C1_001, C2_001, F1_002` — **only 2 of our 5 targets overlap**, and it writes render metrics as `render_eval.txt`/`render_eval.csv` and **does not tar them into `payload.tgz`** (Section 6 fixes both). Do **not** run it verbatim.
  - `Addons/colab/run_cell.sh` — the canonical **single-run wrapper** (train → 6-panel video → render PSNR/SSIM/LPIPS into **`render_metrics.txt`** → Sim3 ATE → ship). This is the file whose `render_metrics.txt` name the aggregator expects. **This is the per-snippet primitive your orchestrator should call** (it already TF32-off + seeds + always emits metrics+video).
  - `Addons/colab/crcd_depth_gen_remainder_20260616.sh` — depth-only batch that **already lists `E3_005`(265), `C3_001`(1527), `G3_001`(1987)** with the exact MoGe-2 + stereo-anchor methodology. **This is your depth template for the 3 missing-config snippets.**
  - `Addons/eval/sim3_ate.py`, `Addons/eval/eval_rendering.py`, `Addons/eval/aggregate_ab.py` — the metric scripts.
  - `Addons/eval/compute_rep_err.py` — a DDS-SLAM **reimplementation** of pixel reprojection error (frame-0-depth anchoring + nearest-surfel-flow approximation). **NOT identical to the paper's metric.** See Section 4.A.5.
  - `Addons/viz/generate_video.py` — the 6-panel video.
  - `Addons/preprocess/preprocess_crcd_published.py` — CRCD staging (rectify, `masks/`, `semantic_class/`, `groundtruth.txt`, `rectified_calib.txt`).
  - `Addons/depth/generate_depth_moge.py` and `Addons/depth/generate_depth_stereo.py` — depth generation. **Note:** `run_crcd_4snippets.sh`/`run_cell.sh` do **NOT** call `generate_depth_stereo.py`; they use **inline frame-0 SGBM** (Section 5.3, Q-CADENCE).
  - `Addons/env/colab_setup.sh` — Colab env build/verify.
  - `Addons/RESULTS_LOG.md` — **read in full**. It records the `trial→Lab` mapping, the `output.txt` **append bug**, the data-port **per-frame walk** finding (Section 4.B caveat), the hash-grid **Z-collapse** failure on bad bounds (Section 5.5), and (critically) that this repo **ports Semantic-SuPer data into its own neural field rather than running the upstream tracker**.
  - `datasets/dataset.py` — **the loader factory**. `get_dataset()` maps `dataset:'super'`→`SuperDataset`, `dataset:'stereomis'`→`StereoMISDataset`. **CRCD configs declare `dataset:'stereomis'`** (`configs/CRCD/crcd.yaml:1`), so CRCD runs through `StereoMISDataset`, **not** `SuperDataset`. This determines which mask dir is read (Section 5.4).
  - `configs/CRCD/crcd.yaml`, `configs/CRCD/crcd_paperfaith_lrfix.yaml`, `configs/CRCD/c1_001_paperfaith_lrfix.yaml`, `configs/CRCD/c2_001_paperfaith_lrfix.yaml` — the CRCD config inheritance chain you will mirror for the missing snippets.
  - `configs/Super/trail{3,4,8,9}.yaml` — the data-port (Path B) configs. **WARNING: their `datadir` is `data/Super/trail_N`, which DOES NOT EXIST** (Section 4.B).

---

## 2. Method dossier

**One-paragraph summary.** Semantic-SuPer is a **semantic-aware deformable surfel + Embedded-Deformation-graph tracker/reconstructor** for endoscopic tissue. It fuses **Monodepth2 stereo depth**, surface normals, and a **DeepLabV3+ 3-class soft semantic map** into a surfel model (pos/normal/color/radius/confidence/timestamp) driven by a sparse ED graph. Per-frame ED-node + global SE3 transforms are estimated by minimizing **point-to-plane ICP + JSD-weighted semantic ICP + semantic-boundary "morphing" loss + SSIM render loss (Pulsar differentiable renderer) + ARAP/face/quaternion regularization**, optimized with **Adam (PyTorch autodiff)** by default (`--use_derived_gradient` switches to Levenberg–Marquardt). The novel contribution is using **soft** semantic labels to guide data association.

**Method class.** Classical deformable surfel + ED-graph tracker. **NOT** NeRF, **NOT** Gaussian Splatting. → It does **not** need an A100; the authors used a single **RTX 2080** (~8–11 GB VRAM at 640×480).

**Own datasets + how to obtain.**
- **Semantic-SuPer dVRK ex-vivo (chicken-on-beef)** — authors' Google Drive (README link `https://drive.google.com/file/d/1JItjKdimx29MnHbqOV0LJCHNmElxc0Ef/view`). 4 trials **Lab1–Lab4**, 150 frames each, rectified stereo 640×480, ~30–60 green-pin GT tracks per trial.
  - **Repo trial↔Lab mapping (this DDS-SLAM repo):** `trial_3=Lab1`, `trial_4=Lab2`, `trial_8=Lab3`, `trial_9=Lab4` (per `Addons/RESULTS_LOG.md` and `configs/Super/trail{3,4,8,9}.yaml`). The actual local data lives at **`data/v2_data/trial_{3,4,8,9}/`** (verified), each containing `rgb/`, `seg/` (`DeepLabV3+/` + `png_masks/`), `seg_gt/`, and the green-pin GT `rgb/trial_N_l_pts.npy`. **There is NO `depth/` dir and NO `*left_depth.npy`** in any trial. **The upstream Drive folder names may differ — verify the mapping against the actual download before comparing to Table I (Q3).**
- **SuPer V1 dataset** — separate Drive link; only for original (non-semantic) SuPer; **not needed** for the Semantic-SuPer headline.

**Pretrained weights + URLs + license gating.**
- **DeepLabV3+ seg ckpts** (segmentation-models-pytorch): `https://drive.google.com/drive/folders/1qzv0KKo_t0VfVQkeNvbeB--4klCDXKiU` — not license-gated on the page; **assume research-use, confirm before redistribution** (Q4).
- **Monodepth2 (stereo, fine-tuned) + RAFT-Stereo**: `https://drive.google.com/file/d/1ptCS9YM5rdA1nXu3bTtdmLEqHa_87TTX/view` — same license caveat.

**Semantic-input requirement (this method).** A per-frame **soft** semantic map from a **pretrained CNN segmentation network** (DeepLabV3+ default), **3 classes** `{surgical tool, beef tissue, chicken tissue}`. **Soft softmax outputs are essential** — the paper's `NoSoftLabel` ablation is markedly worse; supplying hardened argmax masks **silently disables the method's core contribution**. NOT GT masks at inference; NOT DINOv2. Loaded via `--load_seg --seg_dir seg/DeepLabV3+ --num_classes 3`. Produce masks with `seg/inference.sh` (provided ckpt) or train with `seg/train.sh`.

**How the paper reports results.** **PER-SEQUENCE.** Table I = reprojection error (px) per trial, `mean(std)` for "all points" and "edge points". Published Semantic-SuPer values: Lab1 `7.5(6.1)/6.7(5.7)`; Lab2 `8.6(7.6)/9.2(7.8)`; Lab3 `6.0(4.9)/5.9(4.8)`; Lab4 `4.3(3.8)/4.3(3.4)`. **Faithful-repro bar (this method): reproduce Table I reproj error (px) on at least 1–2 of the 4 trials within tolerance of the published `mean(std)`** — matching per-trial numbers, not an average. **This bar is achievable ONLY under Path A (Section 4.A).**

---

## 3. Environment setup on Colab

Two **separate, non-co-installable** environments are required. **Never co-install them.**

### 3.1 ENV-SS — Semantic-SuPer (Phase A Path A, the upstream surfel tracker)
Old, brittle stack. Pins (from `resources/environment.yaml`):
- Python **3.8**, torch **1.11.0** + torchvision **0.12.0**, **cudatoolkit 11.3.1** (cu113).
- **pytorch3d 0.6.2** (provides the **Pulsar** renderer — the **top build landmine**).
- segmentation-models-pytorch 0.3.0, torch-scatter 2.0.9, torch-sparse 0.6.14, torch-geometric 2.0.4, open3d 0.15.2, opencv 4.5.3, numpy 1.23.1, kornia 0.6.6, moviepy 1.0.3, Pillow 9.4.0; Monodepth2 + RAFT-Stereo vendored.

**Build strategy on Colab (in priority order):**
1. **Docker image** `docker/super_docker.Dockerfile` if the Colab runtime allows it (needs NVIDIA Container Toolkit). Preferred — avoids the pytorch3d/Pulsar build.
2. **Pinned wheels** for torch 1.11+cu113 and a **prebuilt pytorch3d 0.6.2** wheel matching that torch/CUDA. Verify Pulsar imports before proceeding.
3. **conda env create -f resources/environment.yaml** — last resort; README warns **~7 hours** and frequent conflicts.

> **Escalate (Q-ENV)** before spending hours: only relevant if Q1 = Path A. If Q1 = Path B (data-port), ENV-SS is **not needed** — use ENV-DDS (3.3) only.

**ENV-SS smoke test (must pass before Phase A Path A):**
```bash
python - <<'PY'
import torch, torchvision, pytorch3d
from pytorch3d.renderer.points.pulsar import Renderer  # Pulsar — the landmine
import segmentation_models_pytorch as smp
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("pytorch3d", pytorch3d.__version__)
r = Renderer(64, 64, n_channels=3, n_track=512)   # historical >512-point crash
print("Pulsar OK")
PY
```

### 3.2 ENV-DEPTH — MoGe-2 (Phase B depth generation only)
MoGe-2 needs **torch ≥ 2.0, Python ≥ 3.10** — **incompatible with ENV-SS**. Built/activated by `Addons/env/colab_setup.sh`; MoGe installed on demand:
```bash
pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub
python -c 'from moge.model.v2 import MoGeModel'   # must succeed
```

### 3.3 ENV-DDS — DDS-SLAM harness (Phase B SLAM + all metrics + video; also Phase A Path B)
The CRCD SLAM run, sim3 ATE, render eval, and 6-panel video all run here. Activate/verify exactly as the runbooks do:
```bash
bash Addons/env/colab_setup.sh --skip-data --skip-tunnel   # builds modern stack if missing
python -c "import torch, tinycudann, marching_cubes; assert torch.cuda.is_available()"
pip install -q lpips   # required by eval_rendering.py
```

### 3.4 GPU policy (this method)
Semantic-SuPer is a **classical surfel tracker** → **fits a T4** comfortably (≤8–11 GB at 640×480). **Do NOT abort on non-A100** for this method. (The A100-mandatory abort applies only to the VRAM-heavy 3DGS methods SemGauss-SLAM / SGS-SLAM.) Detect and **log** the GPU regardless:
```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
```
**Escalate (Q-GPU)** the exact min-VRAM number/abort threshold per method if the orchestrator demands a uniform policy.

---

## 4. PHASE A — Reproduce on the method's OWN dataset

### 4.0 Decide the code path FIRST (BLOCKING escalation — Q1)
`Addons/RESULTS_LOG.md` and the loader factory confirm: **this DDS-SLAM repo does not run the upstream Semantic-SuPer surfel tracker.** `configs/Super/*` feed Semantic-SuPer **data** through DDS-SLAM's own neural field (`ddsslam.py`), which reports **PSNR / ATE**, not the paper's **pixel reprojection error**. These are **non-comparable**.

→ **STOP and escalate (Q1):** does "reproduce Semantic-SuPer" mean
- **(A) Upstream `run_semantic_super.py`** → native **Table I reprojection error (px)** = the **only** path that supports a *"reproduce Semantic-SuPer"* claim; **or**
- **(B) DDS-SLAM data-port** through `ddsslam.py` on `trail3/4/8/9` → DDS-SLAM render PSNR/SSIM/LPIPS (the values in `eval_rendering.py:PAPER_REFERENCES`, e.g. `Lab1 PSNR 28.649`). **These are a DIFFERENT paper's (DDS-SLAM's) numbers.** Path B is a *same-data render-PSNR comparison*, **NOT a reproduction of Semantic-SuPer Table I**, and the paper's native metric (pixel reprojection error) is **never reproduced** under Path B.

**Publishability note (must be in the escalation):** the global benchmark bar is "reproduce the method's own paper." Path B **cannot satisfy that bar by construction.** Additionally `RESULTS_LOG.md` documents that the data-port **walks** on these *static-camera* trials (per-frame motion 11–28 mm where the camera is static; Lab2 drifts 417 mm; all 4 labs walk universally with ordering uncorrelated to the paper; root-cause bbox/loss-landscape **UNRESOLVED**), so even the Path-B render PSNR may not be trustworthy. **Ask the user explicitly whether a Path-B-only gate is acceptable for publication** before running it. Do not let Path B silently stand in for reproduction.

Run the path the user picks (`SS_CODE_PATH=A|B`). Both protocols are specified below.

---

### 4.A Path A — Upstream Python-SuPer (paper-faithful, pixel reprojection error)

**A.0 Clone + pin the upstream repo (REQUIRED — it is not in this repo).**
```bash
SS_UP=/content/Python-SuPer
git clone https://github.com/ucsdarclab/Python-SuPer "$SS_UP"
cd "$SS_UP"
git checkout <PINNED_SHA>     # Q-UPSHA: pin a commit; record it in ${DRIVE_ROOT}/_versions.txt
```
- **Q-UPSHA (BLOCKING for Path A):** the upstream repo has no released tag; pin an explicit commit SHA and record it. Do not run an unpinned `HEAD`.
- **Working directory for the run command is `$SS_UP`** (the script resolves `seg/`, vendored Monodepth2/RAFT, and `run_semantic_super.py` relative to repo root). All `python run_semantic_super.py ...` calls in A.4 execute from `cd "$SS_UP"`.

**A.1 Obtain data + weights.** Download to Drive (one-time), then stage to `/content`:
- dVRK ex-vivo dataset (Drive `1JItjKdimx29MnHbqOV0LJCHNmElxc0Ef`) → trials with `rgb/`, `depth/`, `seg/DeepLabV3+/`, and `rgb/<trial>_l_pts.npy` (green-pin GT).
- DeepLabV3+ seg ckpts (Drive `1qzv0KKo_...`) and Monodepth2/RAFT weights (Drive `1ptCS9YM5...`).
- **Verify the trial↔Lab mapping** in the actual download against Table I order (Q3). Record it in `${DRIVE_ROOT}/_mapping.txt`.

**A.2 Produce the semantic maps** (if not shipped): `bash seg/inference.sh` with the provided DeepLabV3+ ckpt → `seg/DeepLabV3+/`. **Confirm soft outputs are written (not argmax).** If you must train: `bash seg/train.sh` (K=4 fold, 50 epochs, Adam lr 1e-4).

**A.3 Configs / per-trial knobs.** `--mesh_step_size` is tuned **per trial** so mean ED-graph edge length ≈ **5 mm** (depends on camera-to-scene distance; Lab4 is furthest). A single fixed step will **not** reproduce all trials — set it per trial. Use the paper loss weights (`lambda_morph=10`, `lambda_sim=1`, `lambda_reg=10`) and the **default Adam optimizer** (do **not** pass `--use_derived_gradient`).

**A.4 Exact run command (per trial, from `cd "$SS_UP"`):**
```bash
python run_semantic_super.py \
  --data_dir <trial_dir> \
  --load_depth \
  --load_seg --seg_dir seg/DeepLabV3+ --num_classes 3 \
  --sf_soft_seg_point_plane \      # JSD-weighted semantic ICP (soft labels)
  --sf_bn_morph \                  # semantic-boundary morphing loss
  --mesh_rot --mesh_face \         # ARAP / face regularization
  --render_loss \                  # SSIM via Pulsar
  --mesh_step_size <per_trial> \   # set so mean ED edge ~5mm
  --tracking_gt_file rgb/<trial>_l_pts.npy \
  --edge_ids <boundary_point_ids>  # from the trial's GT metadata
```
This emits **all-points** and **edge-points** reprojection mean/std (px) + TensorBoard viz. Record stdout to `${DRIVE_ROOT}/phaseA/<Lab>/reproj.txt`.

**A.5 Evaluate exactly as the paper — and pin the metric source.** The Table-I-comparable reprojection error is the one computed **inside the upstream `run_semantic_super.py`** (a labeled point's flow = average of its 3 nearest tracked surfels' flows; 2D distance to the green-pin GT track, in **pixels**, on rectified 640×480; **no 3D alignment, no sim3, no ATE**). **Use the upstream internal number for the gate.**
> The repo's `Addons/eval/compute_rep_err.py` is a **NON-IDENTICAL reimplementation** (it anchors each GT point from frame-0 depth, approximates flow via nearest-surfel, then pixel-L2 to the green-pin track) and is **NOT authorized** to stand in for the Table-I match. Its docstring example points at non-existent paths (`data/Super/trail_3/depth/ref/000000.npy`). **Use it only as a cross-check, never as the gate number, and only if the user authorizes the approximation (Q-REPERR).** If you must invoke it as a cross-check, valid arguments on the local data are:
> ```bash
> python Addons/eval/compute_rep_err.py \
>   --est_c2w <upstream_or_port_est_c2w_data.txt> \
>   --pts     data/v2_data/trial_3/rgb/trial_3_l_pts.npy \
>   --depth0  <existing_frame0_depth.npy>   # NOTE: data/Super/... does not exist; supply a real frame-0 depth
> ```
> **Escalate (Q-REPERR):** confirm the upstream-internal number is the gate, and whether the `compute_rep_err.py` approximation is acceptable as a cross-check at all.

**A.6 Artifacts to record:** per-trial `reproj.txt` (all + edge px `mean(std)`), `--mesh_step_size` used, seg-source provenance, depth-source provenance, the Lab↔trial mapping, the pinned upstream SHA, runtime + peak VRAM (Section 8.2), TensorBoard logs.

**A.7 Acceptance bar (tied to Table I granularity).** Reproduce **≥ 1–2 of the 4 trials** within tolerance of the published `mean(std)` for **both** all-points and edge-points. Default tolerance: **within the published std band, or ≤ +30% of the published mean** — **confirm the exact numeric tolerance with the user (Q-TOL) before declaring PASS.**

---

### 4.B Path B — DDS-SLAM data-port (only if the user chooses (B) AND accepts it is NOT a reproduction)

**B.0 The configs are BROKEN as shipped — fix the data path + depth FIRST.** Confirmed: `configs/Super/trail3.yaml` sets `datadir: data/Super/trail_3`, but **`data/Super/` does not exist** — the data is at **`data/v2_data/trial_3/`**. `SuperDataset` (used because `dataset:'super'`) globs:
- images `rgb/*left.png` (present),
- depth `{depth_subdir}/*left_depth.npy` with **default `depth_subdir='rgb'`** — **NO `*left_depth.npy` exists anywhere** (verified), so the depth glob is empty → crash,
- semantics `seg/png_masks/*left.png` (present).

→ **STOP and escalate (Q-TRAILDIR + Q-TRAILDEPTH):**
1. **Q-TRAILDIR:** the canonical trail datadir — `data/Super/trail_N` (config) vs `data/v2_data/trial_N` (actual). Confirm which to use; do not silently rewrite the config without confirmation.
2. **Q-TRAILDEPTH:** Path B needs `*left_depth.npy`. None exist. Either (a) generate them for the trail dataset (MoGe-2 → `*left_depth.npy` in a chosen `depth_subdir`, mirroring Section 5.3), or (b) point `depth_subdir` at an existing depth source — **but none exists locally**, so generation is required. The repo's `trail3_moge2.yaml` sets `depth_subdir: depth/moge2` yet still points at the nonexistent `data/Super/trail_3` and that depth dir is also not present.

**B.1 Pre-flight glob assertion (must pass before launching SLAM):** assert all three globs are **non-empty** on the chosen datadir:
```bash
ls <datadir>/rgb/*left.png            | head -1   # images
ls <datadir>/<depth_subdir>/*left_depth.npy | head -1   # depth (generated)
ls <datadir>/seg/png_masks/*left.png  | head -1   # semantics
```
If any is empty → `STAGE_BLOCKED`, do not run.

**B.2 Run** (after B.0/B.1 resolved) via `Addons/colab/run_cell.sh`:
```bash
bash Addons/colab/run_cell.sh configs/Super/trail3.yaml phaseA_trail3_s0 0   # Lab1
# repeat for trail4 (Lab2), trail8 (Lab3), trail9 (Lab4)
```
`run_cell.sh` auto-detects `super`, renders, runs `eval_rendering.py` against the hardcoded `PAPER_REFERENCES` (Lab1 PSNR 28.649 / SSIM 0.797 / LPIPS 0.231, etc.), and writes **`render_metrics.txt`**. **Acceptance bar = these PSNR/SSIM/LPIPS within user-set tolerance (Q-TOL).**

> **Caveat to surface prominently in the report:** Path B is **NOT a Semantic-SuPer reproduction** (it compares same-data render PSNR vs a *different* method, DDS-SLAM). `RESULTS_LOG.md` documents the data-port **walks** on these static-camera trials (per-frame 11–28 mm, Lab2 ATE 417 mm, ordering uncorrelated to the paper, root-cause UNRESOLVED), so the render PSNR itself may be untrustworthy and **any ATE from Path B is meaningless.** The Path-B headline is render PSNR only, and even that is flagged.

---

### 4.Z DECISION GATE (must pass before any CRCD work)
```
GATE = (code path chosen by user via Q1; Path-B-as-gate explicitly approved if B — see 4.0)
       AND ( Path A: ≥1–2 trials within Q-TOL of Table I px (upstream-internal metric)
           | Path B: PSNR within Q-TOL of PAPER_REFERENCES — recorded as NON-reproduction )
       AND (semantic maps verified SOFT, num_classes=3 [Path A])
       AND (Lab↔trial mapping verified and recorded)
       AND (Path A: upstream SHA pinned & recorded | Path B: trail datadir + depth source confirmed)
       AND (runtime + peak VRAM logged)
```
- **PASS** → write `${DRIVE_ROOT}/phaseA/.GATE_PASS` (record path chosen, matched trials + numbers, tolerance, and — if Path B — the explicit "NOT a reproduction" flag) → proceed to Phase B.
- **FAIL or any input unverified** → **STOP**, escalate per Section 0.3, do **not** start Phase B.

---

## 5. PHASE B — CRCD adaptation

> **Per-snippet primitive:** call `Addons/colab/run_cell.sh` (it writes `render_metrics.txt` — the name the aggregator reads — and the 6-panel video + Sim3 ATE). Use `run_crcd_4snippets.sh` only as a **structural reference** for the stage ordering (env → stage → preprocess → depth → scfactor → config → slam → eval → ship); **do not run its SNIPPETS loop** (wrong snippets) and apply the Section 6 filename/payload reconciliation. CRCD runs through **ENV-DDS (3.3)** via `StereoMISDataset` (`dataset:'stereomis'`). The harness CRCD metrics are **harness-imposed**, not Semantic-SuPer-paper-comparable (the method has no ATE/PSNR native metric and was built for a **static** camera). State this caveat in the report.

### 5.0 The 5 target snippets → exact per-snippet rows
Use **exactly** these rows. Do **not** reuse `run_crcd_4snippets.sh`'s array (`F3_007/C1_001/C2_001/F1_002`) — only `C1_001`/`C2_001` overlap. Row format: `NAME  EP  SID  config-stem  frames`.
| Snippet | NAME | EP | SID | Staged dir | Config stem | Frames | Config status |
|---|---|---|---|---|---|---|---|
| `C_1/001` | `C1_001` | `C_1` | `001` | `data/CRCD/C1_001` | `c1_001_paperfaith_lrfix` | 360 | **EXISTS** |
| `C_2/001` | `C2_001` | `C_2` | `001` | `data/CRCD/C2_001` | `c2_001_paperfaith_lrfix` | 730 | **EXISTS** |
| `E_3/005` | `E3_005` | `E_3` | `005` | `data/CRCD/E3_005` | `e3_005_paperfaith_lrfix` | 265 | **MISSING — author it (Q-BOUNDS)** |
| `C_3/001` | `C3_001` | `C_3` | `001` | `data/CRCD/C3_001` | `c3_001_paperfaith_lrfix` | 1527 | **MISSING — author it (Q-BOUNDS)** |
| `G_3/001` | `G3_001` | `G_3` | `001` | `data/CRCD/G3_001` | `g3_001_paperfaith_lrfix` | 1987 | **MISSING — author it (Q-BOUNDS)** |

Frame counts for the 3 missing snippets (265/1527/1987) are taken from `crcd_depth_gen_remainder_20260616.sh` but are **provisional** — confirm against the staged GT line-count (Section 5.1/5.2), not the stale local copy.

### 5.1 F:/ → Drive copy (Colab cannot read F:/) — Q-COPY
CRCD-Published lives on the user's local **F:/** drive. `data/CRCD/` **does not exist locally** (verified) — every Phase-B unit depends on this copy. **One-time:** the **user** copies F:/ → Drive at `/content/drive/MyDrive/Datasets/CRCD-Published/` preserving the layout the runbook expects: per snippet `<EP>/snippet_<SID>/{rgb,rgbright,semantic_instance,groundtruth.txt,intrinsics.yaml}` and `cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`. **Prefer per-snippet staging tarballs** `<EP>_snippet_<SID>_staging.tar` (Drive-FUSE per-item cp is ~78 min/8 GB; one sequential tar read is ~3–5 min).
- **The agent cannot read F:/ itself.** If the copy is absent, emit `STAGE_BLOCKED ... stage (escalation: Q-COPY)` and **exit 30 cleanly** (Section 7.5) — do **not** `STAGE_FAIL`.

### 5.2 Stage + preprocess (rectify) per snippet
Mirror `run_crcd_4snippets.sh` Phase 1/1b:
```bash
python Addons/preprocess/preprocess_crcd_published.py \
  --snippet_dir /content/crcd_raw/<EP>_snippet_<SID> \
  --calib_pkl   $DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl \
  --output_dir  data/CRCD/<NAME>
```
Produces `video_frames/{NNNNNN l,r}.png` (rectified 1280×720), **`masks/`** (binary tool), **`semantic_class/`** (4-class index map for the video panel), `groundtruth.txt` (TUM), `rectified_calib.txt`.
- **Verify the authoritative frame count from the staged `groundtruth.txt` line-count** (the local config note for c1_001 warns the *local* copy is stale: "360 NOT 271"). Use the staged GT count, not the table above, for the resume check.
- Verify `len(video_frames left) ≥ frames`, `rectified_calib.txt` exists.

### 5.3 Depth — MoGe-2 + stereo-scaled metric (per snippet)
Mirror `run_crcd_4snippets.sh` Phases 1.5 + 1.6 (identical methodology in `crcd_depth_gen_remainder_20260616.sh`, which **already covers E3_005/C3_001/G3_001**). Rehydrate from the Drive cache if present (`/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth/*.png,.sc_factor}`).
1. Symlink left frames → `_moge_in/<fid>-left.png`.
2. MoGe-2 metric depth:
```bash
python Addons/depth/generate_depth_moge.py \
  --rgb _moge_in --out _moge_npy \
  --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0   # metric-direct (NO --ref)
```
3. `npy → depth/<fid>.png` (uint16, clip to 65535).
4. **Stereo anchor (frame-0 SGBM, inline — NOT generate_depth_stereo.py)** → `sc_factor = median(stereo_depth_m / moge_depth_m)` using `rectified_calib.txt` (`baseline_m`, `fx`), `cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7, P1=8·49, P2=32·49, …, MODE_SGBM_3WAY)`, masked `0.05<d<3.0 m` ∧ MoGe-valid. Write `data/CRCD/<NAME>/.sc_factor`.
5. If `abs(log(sc_factor)) > 0.1`, patch `data.sc_factor: <val>` into the snippet config (the loader multiplies loaded depth and scales GT translations by it).

> **Two BLOCKING reconciliations here.**
> **(a) Q-CADENCE — spec vs repo mismatch.** The global spec mandates stereo-matching **~every 100 frames** via **`Addons/depth/generate_depth_stereo.py`**. Confirmed: the Phase-B scripts (`run_crcd_4snippets.sh`, `run_cell.sh`) **do NOT call `generate_depth_stereo.py`** — they compute a **single frame-0 SGBM `sc_factor`** inline. (`generate_depth_stereo.py` is referenced only by `overnight_20260605.sh` and `run_depth_gen_colab.sh`, not the Phase-B path.) **Reconcile before any `sc_factor` is computed:** either the spec's named script is wrong for this method (keep single frame-0 inline, matching all existing results), or the agent must switch to periodic re-scaling via `generate_depth_stereo.py`. **This changes every depth map and every translation-scaled GT comparison**, so it MUST be resolved first; on sub-SNR/largely-static snippets a single frame-0 anchor may be unrepresentative. Do not pick silently.
> **(b) Q-SCBUG — `sc_factor` threshold bug** (`run_crcd_4snippets.sh:372`): when `sc_factor` is far from 1.0, only `trunc` rescales — `range_d/near/far/depth_trunc` stay unscaled → per-threshold inconsistency. Confirm whether to fix before publishing or constrain snippets to `sc_factor` within 10% of 1.0.

### 5.4 Semantic-input handling (BLOCKING — the method's core; Q2 + Q-SEMSRC)
Semantic-SuPer's contribution is **soft semantic labels guiding association** over `{tool, beef, chicken}` — a phantom ontology that **does not exist on CRCD** (porcine cholecystectomy: liver/gallbladder/fat/connective/instrument). CRCD-Published ships GT `semantic_instance` (uint16 = `coco_id+1`, Tool=id 3); `preprocess_crcd_published.py` writes both a **binary tool mask** (`masks/`) and a **4-class `semantic_class/`** map.

**Which directory the CRCD run actually consumes (traced, no longer ambiguous):** CRCD configs declare `dataset:'stereomis'` → **`StereoMISDataset`** (`datasets/dataset.py:120`). That class globs semantics from **`{basedir}/masks/*.png`** (`dataset.py:136-138`) — i.e. the **binary tool mask**. `semantic_class/` is consumed **only** by the 6-panel video, **not** by SLAM. So on the data-port the semantic signal driving the edge-semantic loss is the **binary tool mask** by default.

→ **STOP and escalate.** A binary tool mask is a **HARD** mask, which (Section 2 / 8.6) **silently disables Semantic-SuPer's soft-label contribution** — any such run is **non-representative of the method**. Keep every snippet `BLOCKED_ESCALATION` on the semantic stage until resolved:
- **Q2 (ontology):** (1) re-train DeepLabV3+ on CRCD's own classes (most faithful to "predicted soft labels guide association"); (2) collapse to a CRCD-native class set from GT masks (skips the net; breaks "predicted, not GT"); (3) 2-class tool/tissue proxy.
- **Q-SEMSRC (signal source):** confirm whether the run consumes the **binary tool mask** (`masks/`, the traced loader default — hard mask, disables the contribution), the **4-class `semantic_class/`** (would require code change — StereoMISDataset does not read it), or a DINO-feature uncertainty variant.

Proceed only if the user **explicitly authorizes** the loader-default binary tool mask as an interim — and the report must then flag the run as **soft-label contribution DISABLED → non-representative of Semantic-SuPer**.

### 5.5 Author the 3 missing configs (`e3_005`, `c3_001`, `g3_001`) — Q-BOUNDS (BLOCKING)
Copy the `c1_001_paperfaith_lrfix.yaml` template. Each per-snippet file overrides **only** `timesteps`, `mapping.bound`, `marching_cubes_bound`, `data.datadir`, `data.output`, `data.exp_name`, inheriting `configs/CRCD/crcd_paperfaith_lrfix.yaml`:
```yaml
inherit_from: configs/CRCD/crcd_paperfaith_lrfix.yaml
timesteps: 265                       # E3_005 (265); C3_001 → 1527; G3_001 → 1987 — CONFIRM vs staged GT count
mapping:
  bound: [[?,?],[?,?],[?,?]]                 # HAND-DERIVE from frame-0 depth + GT extent — DO NOT GUESS
  marching_cubes_bound: [[?,?],[?,?],[?,?]]
data:
  datadir: data/CRCD/E3_005
  trainskip: 1
  output: output/CRCD/E3_005_paperfaith_lrfix
  exp_name: demo
```
`mapping.bound`/`marching_cubes_bound` were **hand-derived** for c1/c2 from frame-0 depth + GT extent (c1: `[[-0.08,0.13],[-0.02,0.18],[0.68,0.90]]`) and are **NOT known** for these three.
- **BLOCK each missing-config snippet at the `config` stage** (`STAGE_BLOCKED ... config (escalation: Q-BOUNDS)`) until the user provides bounds + confirms frame counts. **Never run SLAM with placeholder/guessed bounds.**
- **Bounds sanity gate (mandatory, mirrors the RESULTS_LOG Z-collapse finding):** after staging+depth, compute frame-0 depth **p1–p99** and the GT translation **extent**; **refuse to proceed** (`STAGE_BLOCKED`) if the proposed bounds do not cover both. `RESULTS_LOG.md` documents that out-of-bbox scenes make the hash grid **Z-collapse** and silently corrupt the run rather than fail loudly.

### 5.6 Run the SLAM (per snippet, failure-isolated)
Per snippet, after stage/preprocess/depth/sc_factor/config are ready and Q2/Q-SEMSRC + Q-BOUNDS are resolved:
```bash
bash Addons/colab/run_cell.sh configs/CRCD/<stem>.yaml <NAME>_s0 0 \
  || { echo "FAILED_SLAM_<NAME>" >> "$DRIVE_ROOT/_failures.log"; continue; }
```
(Use `run_cell.sh` — it emits `render_metrics.txt` + 6-panel + Sim3 in one shot and never forgets metrics. A bare `python ddsslam.py` produces neither.)
**Resumability:** skip if `output/CRCD/<NAME>_paperfaith_lrfix/demo/est_c2w_data.txt` has ≥ staged-GT-count lines. **Before any re-run, `rm` stale `demo/output.txt`** (it **appends** — `RESULTS_LOG.md`).

### 5.7 Output layout under MyDrive/Outputs
- `DRIVE_ROOT=/content/drive/MyDrive/Outputs/semantic_super_crcd_<DATE>`.
- Per snippet `DRIVE_DST=$DRIVE_ROOT/<NAME>`: **`render_metrics.txt`** (the aggregator-expected name; Section 6), `render_eval.csv` (per-frame), `sim3_metrics.txt`, `summary.txt`, `<NAME>_6panel.mp4`, `payload.tgz` (**must include `render_metrics.txt`** — Section 6), `.DONE`.
- Cross-run aggregates at `DRIVE_ROOT`: `_render_summary.csv`, `_failures.log`, `_escalations.md`, `_harness_diffs/`, `COMBINED_SUMMARY.txt`, `runbook.log`.

---

## 6. CRCD evaluation

Run via the DDS-SLAM harness. `OUT=output/CRCD/<NAME>_paperfaith_lrfix`, `STAGED=data/CRCD/<NAME>`.

**ATE (sim3-aligned) — headline trajectory metric.**
```bash
python Addons/eval/sim3_ate.py \
  --est "$OUT/demo/est_c2w_data.txt" --gt "$STAGED/groundtruth.txt" \
  --name "<NAME>" --out "$DRIVE_DST/sim3_metrics.txt"
```
Reports **Sim3 (Umeyama with-scale) ATE rmse/mean/median/max (mm)**, recovered scale `s`, est/GT path ratio, `|Pearson|` dominant axis (scale-free). It **also** prints the rigid ATE labeled "do NOT headline" — **never headline the rigid number**; on up-to-scale MoGe depth it is dominated by scale mismatch.

**PSNR / SSIM / LPIPS — with the REQUIRED harness fixes.**
Two confirmed blockers and their fixes (record both diffs in `${DRIVE_ROOT}/_harness_diffs/`):

1. **Filename reconciliation (CRITICAL).** `aggregate_ab.py` (Section 9) extracts **`render_metrics.txt`** from each `payload.tgz` (it greps `PSNR:/SSIM:/LPIPS:/Rendered:`). But `run_crcd_4snippets.sh` writes **`render_eval.txt`** and never tars any render-metrics file into `payload.tgz` (its `SHIP_ITEMS=demo/ckpts/depth/renders_rgb` only). **Standardize Phase B on `render_metrics.txt`** by driving each snippet through **`run_cell.sh`** (which already writes `render_metrics.txt`), AND ensure that file is **added to `payload.tgz`** before shipping:
```bash
cp "$DRIVE_DST/render_metrics.txt" "$OUT/render_metrics.txt"   # so it lands inside the tar
# include render_metrics.txt in SHIP_ITEMS when building payload.tgz
```
Without this, `aggregate_ab.py` parses `None` for every snippet's PSNR/SSIM/LPIPS and the final table is silently empty.

2. **`--sequence` argparse crash for 3 snippets (Q-SEQCHOICE) — authorized trivial fix.** Confirmed: `eval_rendering.py:73-74` uses `--sequence choices=list(PAPER_REFERENCES.keys())`, and `PAPER_REFERENCES` (lines 46-58) contains only `CRCD (C1_001/C2_001/F1_002/F3_007)` — **NOT** `E3_005/C3_001/G3_001`. Any render-eval for those three exits non-zero before computing PSNR. **Add three `None`-ref entries** (behavior-preserving, matching the existing `None` pattern) and record the diff:
```python
'CRCD (E3_005)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
'CRCD (C3_001)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
'CRCD (G3_001)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
```
Eval call (after the fix; CRCD has no paper reference):
```bash
python Addons/eval/eval_rendering.py \
  --gt_dir "$STAGED/video_frames" --render_dir "$OUT" --name "<NAME>" \
  --output_csv "$DRIVE_DST/render_eval.csv" \
  --summary_csv "$DRIVE_ROOT/_render_summary.csv" --sequence "CRCD (<NAME>)"
```
Renders pair to `video_frames/*l.png` by filename index. LPIPS needs `pip install lpips`.

**Depth-L1 — BLOCKED_ESCALATION (no implementation, no GT). NOT a runnable stage; NOT a required deliverable.**
Confirmed: the repo has **no** Depth-L1 script (only `compute_rep_err.py` for reprojection), and CRCD ships **no GT depth** (MoGe-2 is generated). The `eval_depthl1` stage is therefore **permanently `BLOCKED_ESCALATION` pending Q-DEPTHL1** — it must **not** be marked FAIL and must **not** gate-deadlock the unit. **Remove Depth-L1 from the "must exist" deliverables** until the user supplies the script + reference definition.
- **Q-DEPTHL1:** (a) what is the Depth-L1 **reference** — rendered SLAM depth vs MoGe-2 input (self-consistency) / vs frame-0 stereo-SGBM / vs held-out stereo? (b) the user must provide/approve the Depth-L1 script + exact pairing. Do **not** fabricate a Depth-L1 number.

**Canonical 6-panel video** (`run_cell.sh:53-59`):
```bash
python Addons/viz/generate_video.py \
  --rgb_input_dir "$STAGED/video_frames" --rgb_input_pattern '*l.png' \
  --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' \
  --depth_input_dir "$STAGED/depth" --depth_output_dir "$OUT/depth" --depth_norm robust \
  --seg_dir "$STAGED/semantic_class" --seg_pattern '*.png' --skip_raw_seg --seg_classmap \
  --trajectory_est "$OUT/demo/est_c2w_data.txt" --trajectory_gt "$STAGED/groundtruth.txt" --trajectory_raw \
  --output "$DRIVE_DST/<NAME>_6panel.mp4" --fps 15
# add: --uncert_dir "$OUT/uncert"  if the run wrote sigma^2
```
Panels: (1) Input RGB, (2) Rendered RGB, (3) Input Depth, (4) Output Depth, (5) Seg overlay (`semantic_class`, video-only), (6) Trajectory raw, plus the Sim3-aligned trajectory (on by default unless `--skip_horn_traj`).

**Recommended diagnostics (Section 8.5):** trajectory overlay vs GT, per-frame ATE/PSNR curves, depth self-consistency heatmaps (note: NOT Depth-L1 until Q-DEPTHL1), semantic-overlay frames, keyframe-coverage plot, GT motion-profile sanity.

---

## 7. Standard CLI contract — `run_semantic_super.sh`

The master orchestrator calls **every** method through one interface. Author `run_semantic_super.sh` at the repo root (mirror `run_crcd_4snippets.sh` *structure*; per snippet call `run_cell.sh`; emit the stage markers below; apply the Section 6 fixes).

### 7.1 Invocation
```bash
bash run_semantic_super.sh --phase <A|B|all> [--snippet <NAME|all>] [--seed <int>] [--drive-root <path>]
```

### 7.2 Arguments
| Arg | Meaning | Default |
|---|---|---|
| `--phase` | `A` (own-dataset gate), `B` (CRCD), or `all` | `all` |
| `--snippet` | one of `C1_001,C2_001,E3_005,C3_001,G3_001` or `all` (Phase B) | `all` |
| `--seed` | RNG seed (Section 8.1) | `0` |
| `--drive-root` | output root override | `/content/drive/MyDrive/Outputs/semantic_super_crcd_<DATE>` |

### 7.3 Environment variables (read if set)
`SS_GPU_ABORT_GB` (min VRAM; permissive — do not hard-abort on T4), `SS_SKIP_DEPTH=1` (rehydrate cached depth only), `SS_DRY_RUN=1` (print plan, run nothing), `SS_CODE_PATH=A|B` (Q1), `SS_SEM_ONTOLOGY` (resolution of Q2; absent ⇒ stays `BLOCKED_ESCALATION`), `SS_UP_SHA` (pinned upstream SHA for Path A; absent ⇒ Path A blocks on Q-UPSHA).

### 7.4 Stage markers (one per line, parseable)
```
STAGE_BEGIN <phase> <snippet> <stagename>
STAGE_OK    <phase> <snippet> <stagename> <elapsed_s>
STAGE_FAIL  <phase> <snippet> <stagename> <exit_code>
STAGE_SKIP  <phase> <snippet> <stagename> (sentinel)
STAGE_BLOCKED <phase> <snippet> <stagename> (escalation: <id>)
```
Stages: `env, stage, preprocess, depth, scfactor, config, semantic, slam, eval_ate, eval_render, video, ship`.
- **`eval_depthl1` is NOT in the stage list** — it is permanently BLOCKED (Section 6, Q-DEPTHL1) and must not deadlock the gate.
- `stage` emits `STAGE_BLOCKED ... (escalation: Q-COPY)` + exit 30 if `data/CRCD/<NAME>` staging inputs are absent.
- `config` emits `STAGE_BLOCKED ... (escalation: Q-BOUNDS)` for the 3 missing configs until bounds provided.
- `semantic` emits `STAGE_BLOCKED ... (escalation: Q2/Q-SEMSRC)` until ontology+source resolved.

### 7.5 Exit codes
`0` all requested units done (or cleanly SKIP). `10` Phase-A gate FAIL. `20` ≥1 CRCD snippet failed but others succeeded (isolated; details in `_failures.log`). `30` BLOCKED on escalation (nothing runnable) — **also the clean exit when Q-COPY staging is absent.** `1` environment/setup fatal. **A single snippet failure must yield `20`, never abort the batch.**

### 7.6 Idempotency
Re-invoking skips any unit with its `.DONE`/`.GATE_PASS` sentinel and intact outputs. Sentinels live next to outputs on Drive.

---

## 8. Failure modes, determinism, checkpointing, logging

### 8.1 Determinism / seeding
- **Phase B / Path B:** seed via the **override-config** pattern (`run_cell.sh` already writes `_cell_<NAME>.yaml` with `inherit_from` + `seed`) — never edit the repo model. TF32 off (`run_cell.sh` already sets `allow_tf32=False`).
- **Phase A / Path A determinism is an OPEN item until the upstream is cloned (Q-UPSHA / Q-SEED).** `run_semantic_super.py` is not present, so its RNG surface (numpy/torch/cuda seeds, Adam init, surfel sampling, Pulsar) cannot be enumerated yet. **After cloning (Section 4.A.0):** enumerate the exact seed points, add a thin wrapper that seeds numpy+torch+cuda before `runpy`-loading the entry (mirroring `run_cell.sh`'s no-model-edit pattern), and record them. **Do not claim Path-A determinism is settled until this is done.**
- Run the **decision-gate trial(s)** with **≥ 2 seeds** to bound variance; `RESULTS_LOG.md` documents up to ~3.4× ATE seed variance on the data-port — report `mean ± std`, not a single number. (For Path A this requires the seed wrapper above.)

### 8.2 GPU / VRAM / runtime logging (per stage)
Log at every stage start/end: `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader`, wall-clock elapsed, and **peak VRAM** via `torch.cuda.max_memory_allocated()` (reset per stage). Runtime for this method is an **open question (Q-RT)** — measure empirically; record per snippet to `${DRIVE_ROOT}/_runtime.csv`.

### 8.3 Sim3-alignment correctness checks (do not trust ATE blindly)
- **Always** report the **Sim3** ATE; treat rigid ATE as a confound indicator only.
- If `recovered scale s` is implausible (off 1.0 by >2× after `sc_factor` applied) → flag depth-scale problem, not tracking quality.
- `est/GT path ratio > 3` ⇒ **tracker jittering, not tracking** (warn).
- `|Pearson| dom < 0.1` ⇒ trajectory shape uncorrelated with GT (warn). On a **moving-endoscope** CRCD sequence fed to a tracker built for a **static** camera, expect low Pearson — **a documented domain gap, not necessarily a bug** (state it).
- Confirm `len(est) == len(gt)` for 1:1 pairing; if resampled, log it.

### 8.4 Frame-subsampling / index alignment (RGB ↔ depth ↔ semantics ↔ GT)
`StereoMISDataset` globs `video_frames/*l.png`, `depth/*.png`, **`masks/*.png`**, pairs GT by frame index. **Before SLAM, assert** equal counts across `video_frames(left)`, `depth`, `masks`, `semantic_class`, and GT lines; with `trainskip>1` ensure the **same** subsample across all modalities. Any mismatch ⇒ **STOP**. Record counts in `summary.txt`.

### 8.5 Recommended diagnostics (emit per snippet)
- **Trajectory overlay** est-vs-GT (Sim3 + raw) — from the 6-panel video.
- **Per-frame ATE curve** + **per-frame PSNR curve** (from `render_eval.csv`).
- **Depth self-consistency heatmap** (rendered vs MoGe-2 input) — explicitly NOT Depth-L1 until Q-DEPTHL1 fixes the reference.
- **Semantic-overlay frames** — visually confirm the ontology/source actually used (Q2/Q-SEMSRC).
- **Keyframe-coverage plot.**
- **GT motion-profile sanity:** extent, path length, per-frame motion, active fraction, sub-SNR sentinels. **Most CRCD snippets are sub-SNR** for tracker benchmarking (c1_001: active median 0.115 mm/f, 8.7× sub-SNR, sentinel verdict REJECT-for-tracking) — surface this so ATE is read correctly.

### 8.6 Sanity gates that catch silent misconfiguration
- **Soft-label gate (Path A):** assert seg maps are multi-channel softmax, not argmax (else the contribution is disabled).
- **Hard-mask flag (Phase B):** if the consumed semantic signal is the binary tool mask (`masks/`, the StereoMISDataset default), **tag the run `SOFTLABEL_DISABLED` / non-representative** (Section 5.4).
- **`num_classes` gate:** assert seg-map channel count matches the agreed ontology (3 for Path A; CRCD ontology for Phase B per Q2).
- **`output.txt` append gate:** `rm` stale `demo/output.txt` before every re-run.
- **`sc_factor` consistency gate:** if `abs(log(sc_factor))>0.1` and Q-SCBUG unresolved, warn loudly and tag `SC_THRESHOLD_SUSPECT`.
- **Bounds-coverage gate (Phase B):** refuse SLAM if proposed bounds don't cover frame-0 depth p1–p99 + GT extent (Section 5.5; Z-collapse).
- **Mesh-step-size gate (Path A):** verify mean ED edge ≈ 5 mm per trial.
- **Render-count gate:** assert `render_freq:1` (set in `crcd_paperfaith_lrfix.yaml:42`) so the video / per-frame metrics aren't sparse.
- **Glob non-empty gate (Path B):** Section 4.B.1.

### 8.7 Logging conventions
`exec > >(tee -a "$DRIVE_ROOT/runbook.log") 2>&1`; one `[PHASE x.<NAME>] HH:MM:SS — desc` per stage; failures → `_failures.log`; escalations → `_escalations.md`; authorized harness edits → `_harness_diffs/`. Mirror everything to Drive.

### 8.8 Checkpointing / resume
Sentinels: `.STAGED`, `.PREPROCESSED`, `depth/.DONE`, `.sc_factor`, per-snippet `.DONE`, Phase-A `.GATE_PASS`. SLAM resume = `est_c2w_data.txt` line-count ≥ staged-GT count. Depth rehydrates from the Drive MoGe-2 cache. Re-runs idempotent.

---

## 9. Deliverables checklist (must exist in Outputs when done)

**Phase A (`${DRIVE_ROOT}/phaseA/`):**
- [ ] `_mapping.txt` (verified Lab↔trial mapping), chosen code path (Q1), and — if Path A — pinned upstream SHA in `_versions.txt`.
- [ ] Path A: per-trial `reproj.txt` (all + edge px `mean(std)`, **upstream-internal metric**) for matched trials + `--mesh_step_size`; **OR** Path B: per-trial `render_metrics.txt` (PSNR/SSIM/LPIPS) vs `PAPER_REFERENCES`, **flagged NOT-a-reproduction**.
- [ ] Seed runs (≥2) with `mean ± std` (Path A only after seed wrapper exists, Section 8.1); runtime + peak VRAM.
- [ ] `.GATE_PASS` with matched numbers, tolerance (Q-TOL), and the Path-B non-reproduction flag if applicable.

**Phase B (`${DRIVE_ROOT}/<NAME>/` for each of the 5 snippets):**
- [ ] `sim3_metrics.txt` (Sim3 ATE + scale + path ratio + `|Pearson|`).
- [ ] `render_metrics.txt` (PSNR/SSIM/LPIPS summary — the aggregator-expected name) + `render_eval.csv` (per-frame).
- [ ] `<NAME>_6panel.mp4`.
- [ ] `summary.txt` (raw/SE3/Sim3 ATE, path ratio, Pearson, modality counts, GT motion-profile sanity).
- [ ] `payload.tgz` (demo/ + ckpts/ + depth/ + renders_rgb/ + **`render_metrics.txt`**), `.DONE`.
- [ ] `data.sc_factor` recorded; semantic ontology + signal source actually used recorded (Q2/Q-SEMSRC); `SOFTLABEL_DISABLED` tag if binary mask used.
- [ ] The 3 authored configs (`e3_005/c3_001/g3_001`) recorded with their **user-provided** bounds (Q-BOUNDS).
- [ ] **Depth-L1: NOT required** — record `BLOCKED_ESCALATION` (Q-DEPTHL1) instead.

**Cross-cutting (`${DRIVE_ROOT}/`):**
- [ ] `_render_summary.csv`, `_runtime.csv`, `_failures.log`, `_escalations.md`, `_harness_diffs/`, `_versions.txt`, `runbook.log`, `COMBINED_SUMMARY.txt`.
- [ ] Cross-method aggregation via `python Addons/eval/aggregate_ab.py <DRIVE_ROOT>`. **Validate before declaring the table done:** run it against **one finished snippet** and confirm non-`None` PSNR/SSIM/LPIPS (proves the Section-6 filename/payload reconciliation worked). An all-`--` table means `render_metrics.txt` was not tarred — fix before reporting.

---

## 10. Open questions to escalate (explicit)

1. **Q1 (BLOCKING) — Phase-A code path & publishability:** upstream `run_semantic_super.py` (Table I px, the only reproduction-capable path) vs DDS-SLAM data-port (PSNR vs a *different* method = NOT a reproduction). Confirm whether a Path-B-only gate is publishable at all (Section 4.0).
2. **Q-UPSHA (BLOCKING, Path A) — Pin the upstream commit SHA** for `git clone github.com/ucsdarclab/Python-SuPer`; record it. No unpinned HEAD.
3. **Q2 (BLOCKING) — CRCD semantic ontology:** retrain DeepLabV3+ on CRCD classes vs collapse to CRCD-GT proxy vs 2-class tool/tissue.
4. **Q-SEMSRC (BLOCKING) — CRCD semantic signal:** traced default is the **binary tool mask** (`masks/`, via StereoMISDataset) = a HARD mask that disables the soft-label contribution. Confirm whether to use it (and accept non-representative runs), wire `semantic_class/` into the loader (code change), or a DINO variant.
5. **Q-BOUNDS (BLOCKING) — Missing-config bounds:** user must hand-derive `mapping.bound`/`marching_cubes_bound` for `e3_005/c3_001/g3_001` and confirm frame counts (265/1527/1987). Snippets stay BLOCKED at `config` until provided; bounds-coverage gate enforced.
6. **Q-COPY (BLOCKING) — F:/ → Drive copy** of CRCD-Published must be done by the user (Colab/agent cannot read F:/). Absent ⇒ `STAGE_BLOCKED` + exit 30.
7. **Q-CADENCE — Stereo-scaling cadence & named script:** repo uses **single frame-0 inline SGBM**, NOT the spec's `generate_depth_stereo.py` ~every-100-frame re-scaling. Reconcile before any `sc_factor` is computed (changes every depth + every translation-scaled GT comparison).
8. **Q-DEPTHL1 (BLOCKED) — Depth-L1 has no implementation and CRCD has no GT depth:** define the reference + provide/approve the script. Stage permanently BLOCKED; removed from required deliverables until resolved.
9. **Q-TOL — Acceptance tolerance** vs Table I `mean(std)` (Path A) / `PAPER_REFERENCES` PSNR (Path B).
10. **Q-REPERR — Reprojection-error metric source:** confirm the **upstream-internal** number is the Table-I gate; whether `compute_rep_err.py` (a non-identical frame-0-anchor + nearest-surfel approximation) is acceptable even as a cross-check.
11. **Q-SCBUG — `sc_factor` threshold bug** (`range_d/near/far/depth_trunc` unscaled when `sc_factor`≠1): fix before publishing, or constrain snippets to `sc_factor` within 10% of 1.0.
12. **Q-SEQCHOICE — `eval_rendering.py` `--sequence` choices** lack `CRCD (E3_005/C3_001/G3_001)` → argparse-crash for 3 of 5. **Authorized trivial fix:** add `None`-ref entries (Section 6); record the diff.
13. **Q3 — Trial↔Lab mapping** in the actual Drive download vs this repo's `trial_3/4/8/9 → Lab1/2/3/4`. Verify before comparing to Table I.
14. **Q4 — Dataset/ckpt license/terms** (Drive links not explicit). Assume research-use; confirm before redistribution.
15. **Q-GPU — Min-VRAM / abort policy:** this method fits a T4 (do not abort on non-A100); confirm the exact per-method min-VRAM number the orchestrator expects.
16. **Q-RT — Runtime:** per-trial / per-snippet wall time unstated; measure empirically; confirm whether offline (non-real-time) is acceptable.
17. **Q-ENV — ENV-SS build feasibility** (pytorch3d-0.6.2/Pulsar/torch-1.11 on Colab); confirm Docker availability before committing hours. Only relevant if Q1 = Path A.
18. **Q-TRAILDIR — Path-B trail datadir:** `data/Super/trail_N` (config, MISSING) vs `data/v2_data/trial_N` (actual). Confirm before rewriting configs.
19. **Q-TRAILDEPTH — Path-B trail depth:** no `*left_depth.npy` exists; depth must be generated for the trail dataset (or `depth_subdir` pointed at a source that does not yet exist). Resolve before any Path-B run.
20. **Q-SEED (Path A) — RNG seed points** of `run_semantic_super.py` (numpy/torch/cuda + tracker-specific sampling/Pulsar): enumerate after cloning; Path-A determinism is open until then.
