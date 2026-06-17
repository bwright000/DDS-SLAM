> **Resolved decisions apply.** See *Resolved Benchmark Decisions* in [00_COMMON.md](00_COMMON.md) — they override any “escalate”/“open question” below: Depth-L1 = input-vs-output per model; periodic ~100f stereo rescaling; per-paper semantic mirroring on CRCD 4 classes; missing configs (e3_005/c3_001/g3_001) are the agent's job.

# Agent Run Book — SemGauss-SLAM (Dense Semantic Gaussian Splatting SLAM)

> Benchmark Run Book set, document **04_SEMGAUSS_SLAM.md**.
> Standing orders for a **fresh autonomous agent** that will be handed the SemGauss-SLAM paper + the official repo **later**. You also have the DDS-SLAM repo (this checkout) as the shared evaluation harness. Read `00_COMMON.md` first; this document only covers what is method-specific or where SemGauss-SLAM deviates from the common contract.

---

## 0. Mission & non-negotiables

**Mission.** Produce, for the published benchmark paper, two things for SemGauss-SLAM:
1. **Phase A** — a faithful reproduction on SemGauss-SLAM's **own** dataset (Replica, 8-scene average), evaluated **exactly as that paper does**, passing a decision **GATE**.
2. **Phase B** — SemGauss-SLAM run on the **5 CRCD snippets** (C_1/001, E_3/005, C_3/001, G_3/001, C_2/001), evaluated with the **DDS-SLAM harness** (ATE-sim3, PSNR, SSIM, LPIPS, Depth-L1) + the canonical 6-panel video.

### 0.1 HARD BLOCKERS — read this first

Two things make most of this run book **non-executable as of authoring**, and a fresh agent must treat them as STOP gates, not warnings:

- **BLOCKER 1 — the SemGauss-SLAM repo is NOT in this checkout.** Verified: `git remote -v` shows only `DDS-SLAM`; there is **no `.gitmodules`**, no `sem_gauss.py`, no `configs/replica/*.py`, no `diff-gaussian-rasterization-w-depth_sem_gauss/`, nothing matching `*semgauss*`. **Every command in this document that invokes the SemGauss code is a TEMPLATE, not a verified invocation.** Specifically `python sem_gauss.py <cfg>` is a **placeholder for an entry point whose real name/CLI is unknown**. Phase A and the SLAM step (Phase 3) of Phase B are **BLOCKED** until the repo is provided. See §2.1 for the exact "repo handoff" procedure that must run before any SLAM work.
- **BLOCKER 2 — the single entry point `Addons/colab/run_semgauss_slam.sh` does NOT exist.** Verified absent. The END-GOAL "single bash line runs everything" **cannot run** until the agent **authors** this script (§7). It is a deliverable-to-build, mirroring `Addons/colab/run_crcd_4snippets.sh`, and it **cannot be end-to-end tested** until BLOCKER 1 is cleared and the SemGauss config schema is known.

Additional **blocked-by-clarification** items (each escalates per §0.2 and is tracked in §10): the **Depth-L1** reference (no implementation exists; §6, §10 q8), the **conda/CUDA-11.6 build feasibility on Colab** (§3, §10 q15), the **CRCD semantic-label space** (§5.3, §10 q7/q16), the **stereo-scaling cadence** (§5.2, §10 q9), the **cross-method aggregator contract** (§7.3, §10 q17), and the **GATE tolerances + VRAM floor** (§4.6/§3.1, §10 q12/q13).

### 0.2 Non-negotiables (do not violate)

- **Precision over progress.** This is a published methodology. Every load-bearing number (iteration counts, LR, resolution, class count, DINOv2 variant, frames-per-scene, alignment convention) must come from the paper text or the repo code — **never from memory or assumption**.
- **No silent guessing — Clarification Protocol.** When a fact is unknown, a choice is load-bearing, or a repo artifact is missing/contradictory, **STOP and escalate to the user**. Do not improvise a value to keep the pipeline moving. Escalation = (a) append a structured entry to `${DRIVE_ROOT}/_clarifications.log` with the exact question, the options you see, and what is blocked; (b) write the affected stage's sentinel as `.BLOCKED` (not `.DONE`); (c) **continue with other independent snippets/methods** (failure isolation), then surface the blocked items in your final report; (d) the orchestrator returns **exit 30** if any BLOCKING clarification remains open. Section 10 is the pre-seeded escalation list — re-confirm each before relying on it.
- **The custom CUDA rasterizer is the single largest silent-failure risk.** SemGauss-SLAM ships its own `diff-gaussian-rasterization-w-depth_sem_gauss` pinned to **CUDA 11.6 / torch 1.12.1**. The DDS harness env (`Addons/env/colab_setup.sh`, verified) is **100% pip on Colab native Python 3.10 / torch 2.x / CUDA 12.x — there is NO conda anywhere in the Addons path**. Building a cu116 toolchain + compiling a custom rasterizer on a CUDA-12.x host driver is **high-risk and may not be possible on the assigned runtime**. See §3 — this requires **user confirmation of the runtime before Phase A**.
- **VRAM gate is mandatory for this method.** SemGauss-SLAM is a **3DGS** method (16-channel semantic embedding per Gaussian + DINOv2 feature maps) — VRAM-heavy. Detect GPU via `nvidia-smi`; **abort** below the documented floor; **recommend A100**. See §3.1 — the practical floor for the headline full-Replica GATE is **24 GB**, not 16 GB (paper ran on RTX 4090 = 24 GB).
- **Determinism is required.** Seed everything; log GPU/VRAM/runtime; record every deviation. Bit-exact 3DGS reproducibility is **not** expected (densification is stochastic) — use ≥3 seeds + mean±std. See §8.
- **Failure isolation + resumability.** One scene/snippet failing must not abort the rest. Per-stage sentinels; resume on reconnect. See §7, §8.
- **CRCD lives on F:/ (user's local drive).** Colab cannot read F:/. CRCD must be copied F:/ → Drive first, then staged to `/content`. See §5.

---

## 1. Orientation / required reading

**Read in this order before touching anything.**

1. **`00_COMMON.md`** (the shared contract): Colab+Drive layout, `/content/drive/MyDrive/Outputs` conventions, the `run_<method>.sh` CLI contract, the CRCD layout, the depth pipeline, and the metrics harness. This run book assumes it.
2. **The SemGauss-SLAM paper** (arXiv:2403.07494, *SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM*). Read with intent to extract the facts in §2 and the **open questions** in §10:
   - **Method / Sec III** — the per-Gaussian 16-channel semantic embedding `e`; the mapping objective and its weights (feature `λ_f`, semantic `λ_s`, color `λ_c`, depth `λ_d` — **read the exact values from the config, do not trust paper-text recall**); semantic-informed bundle adjustment; the tracking objective (RGB+depth only, silhouette region, **semantic losses excluded** — confirm in code).
   - **Sec on the segmentation network** — "universal feature extractor DINOv2, followed by a pretrained classifier." Extract (or flag as open) the DINOv2 variant, the native-dim→16 projection mechanism, **and whether features are extracted internally at train time vs consumed precomputed** (§5.3, §10 q1).
   - **Tables I–IV** — the **reporting granularity** (this is the acceptance bar). Confirm each table's exact numbers from the paper PDF before populating the GATE (§4.6); the headline is Replica 8-scene average (Table I), novel-view mIoU on 4 scenes (Table III), input-view mIoU (Table IV), ScanNet per-scene ATE (Table II).
3. **The official SemGauss-SLAM repo** — **does not exist in this checkout yet (BLOCKER 1).** Once provided, read first, in this order, and record findings into `configs_extracted.json` (§2.1):
   - `README.md` (data + weights download links, exact env, exact run command, DINOv2 `.pth` filename).
   - `sem_gauss.py` **or whatever the actual entry point is** — read its `argparse`/`main` to confirm: the **exact CLI**, the config file format (`.py` vs `.yaml`), the **`input_folder` layout**, the **intrinsics keys**, the **depth-scale key**, the **class-count key**, the **seed key (if any)**, and the **trajectory output cadence + file format**.
   - `configs/replica/*.py` and `configs/scannet/*.py` — the source of truth for iteration counts, LR, resolution, frames, class count.
   - the **dataloader** module the entry point imports — confirm RGB glob, depth glob, depth scale, intrinsics, mask-input support.
   - `segmentation/` — the DINOv2 backbone + classifier seg head; identify the exact variant, the 16-d projection, and whether it runs at train time.
   - `diff-gaussian-rasterization-w-depth_sem_gauss/` — the custom rasterizer (build target).
   - `eval_mesh/` / eval utils — mesh + ATE/render/mIoU metrics; identify the **ATE alignment convention** in code.
4. **The DDS-SLAM harness files you will call** (verified present in this checkout):
   - `Addons/eval/sim3_ate.py`, `Addons/eval/eval_rendering.py`, `Addons/eval/aggregate_ab.py`, `Addons/eval/kitti_to_tum.py`
   - `Addons/viz/generate_video.py`
   - `Addons/depth/generate_depth_moge.py`, `Addons/depth/generate_depth_stereo.py`
   - `Addons/dino/generate_dino_features.py`
   - `Addons/preprocess/preprocess_crcd_published.py`
   - `Addons/colab/run_crcd_4snippets.sh` (**the canonical CRCD batch template** — mirror its phase/sentinel structure), `Addons/colab/run_cell.sh` (single-run wrapper — note it writes `render_metrics.txt`, the name `aggregate_ab.py` expects), `Addons/colab/a100_improve_ab_20260616.sh` (the aggregator-compatible payload+naming reference), `Addons/colab/crcd_depth_gen_remainder_20260616.sh` (depth-only, covers E3_005/C3_001/G3_001), `Addons/env/colab_setup.sh`.
   - `configs/CRCD/crcd.yaml`, `configs/CRCD/crcd_paperfaith_lrfix.yaml`, `configs/CRCD/c1_001_paperfaith_lrfix.yaml`, `configs/CRCD/c2_001_paperfaith_lrfix.yaml` (templates for the **CRCD on-disk layout + intrinsics + sc_factor mechanics**).

---

## 2. Method dossier

**One-paragraph summary.** SemGauss-SLAM is an **RGB-D 3D Gaussian Splatting semantic SLAM** system. Every 3D Gaussian carries, in addition to color/opacity/scale/rotation, a **16-channel semantic feature embedding `e`**. During **mapping** it is supervised by two semantic signals — a **feature-level loss** `L_f` (DINOv2 2D features vs the rasterized 16-channel feature map) and a **cross-entropy semantic loss** `L_s` on labels from a pretrained classifier applied to the rendered features — plus color + depth losses. **Tracking** uses RGB+depth only over a silhouette region; **semantic-informed bundle adjustment** refines poses. Headline results are an **8-scene Replica average** plus semantic mIoU on novel/input views. **All loss weights, iteration counts, and the silhouette threshold are load-bearing and must be read from the shipped configs, not recalled (§10 q3).**

**Method class.** 3D Gaussian Splatting (3DGS) semantic SLAM, **RGB-D (depth mandatory, no monocular fallback)**, differentiable rasterization with depth + semantic channels.

**Own datasets + how to obtain.**
| Dataset | Sequences | How obtained |
|---|---|---|
| **Replica** (semantic, NICE-SLAM renders) | room0/1/2, office0/1/2/3/4 (8 scenes) | Pre-rendered RGB-D from the project Google Drive in the README. Place under the config's `input_folder`. Native semantic GT. **Frames-per-scene = OPEN until read from `configs/replica/*.py` (§10 q3).** |
| **ScanNet** | per-scene ATE (count per Table II) | Register at scan-net.org, extract color+depth from `.sens`, official semantic labels. |

> Project Google Drive (data + weights + GT meshes): link is in the repo `README.md`. **Do not hardcode a folder ID until the README is read (§10 q3).**

**Pretrained weights + license gating.**
| Weight | Purpose | Source | Gated? |
|---|---|---|---|
| DINOv2 backbone + pretrained classifier (seg head) | 2D feature extraction (`L_f`) + label prediction (`L_s`) | Project Google Drive → into `segmentation/` (exact subpath + filename per README) | Confirm in README |
| Replica GT meshes | mesh reconstruction quality | Project Google Drive | Confirm in README |

> **Landmine:** the DINOv2 `.pth` must be **manually placed** before the first run or the seg head silently fails. Pin its exact filename from the README and verify presence in the smoke test (§3.3).

**Semantic-input requirement (this method's defining trait).** SemGauss-SLAM needs **two** semantic signals at mapping time: (1) DINOv2 2D features (class-agnostic, for `L_f`); (2) per-pixel **class labels** for the cross-entropy `L_s`, sourced either from the pretrained classifier ("Ours") or from dataset GT labels ("Ours (GT)"). On its own datasets, native semantic GT exists. **On CRCD this is load-bearing — see §5.3 and §10 q1/q7.** A critical open question is whether the repo extracts DINOv2 features **internally at train time** (most likely) — if so, the harness DINO bake is **unnecessary and would double-supply** (§5.3).

**How the paper REPORTS results (the acceptance bar).**
- **Canonical headline = Replica AVERAGE over all 8 scenes (Table I).** Faithful reproduction = **run all 8 Replica scenes** and match the **dataset-wide average** within tolerance for geometry/render metrics.
- **Semantic mIoU** is reported over **4 scenes** (Table III) → 4-scene matching acceptable for the semantic table.
- **ScanNet ATE is per-scene** (Table II) → **1–2 representative sequences** within tolerance suffice (secondary table).

**Own-data eval protocol (paper).** Tracking ATE (cm); reconstruction Depth-L1 (cm) + mesh accuracy/completion; rendering PSNR/SSIM/LPIPS; semantic mIoU (novel + input views). **The ATE alignment convention (rigid SE3 vs Sim3) and the Table I render view type (train vs held-out) must be read from the eval code/paper — both OPEN (§10 q4/q5).** Use the **repo's own eval outputs** for Phase A, not the DDS harness.

### 2.1 Repo handoff procedure (RUN BEFORE any SLAM work — clears BLOCKER 1)

When the SemGauss-SLAM repo is provided, **before authoring any config or run command**, do the following and **STOP/escalate if any cannot be answered**:

1. Check out the repo to `${SEMGAUSS_REPO}` (default `/content/SemGauss-SLAM`). Record the commit hash.
2. Read the actual entry point's `argparse`/`main`. Record into `${DRIVE_ROOT}/phaseA/configs_extracted.json`:
   - `entry_point` (real script name), `cli_pattern` (exact invocation), `config_format` (`.py`/`.yaml`).
   - `input_folder_key`, `rgb_glob`, `depth_glob`, `depth_scale_key` + default value, `intrinsics_keys`, `class_count_key`, `seed_key_or_null`.
   - `traj_output_path`, `traj_format` (KITTI 3×4 / 4×4 / TUM quat / other), `traj_cadence` (per-frame vs per-keyframe).
   - `mask_input_supported` (does the loss accept a per-pixel exclusion mask?).
   - `dino_internal_vs_precomputed`, `dino_variant`, `native_dim`, `dim16_projection_mechanism`.
   - all loss weights, mapping/tracking iters, LR, resolution, frames-per-scene, semantic class count, silhouette threshold.
3. Only after `configs_extracted.json` is complete may you author any CRCD config (§5.4) or any run command. **The string `python sem_gauss.py …` used throughout this document is a placeholder; replace it with the verified `cli_pattern`.**
4. Build/feasibility-test the rasterizer (§3.2) and escalate the build-feasibility question (§10 q15) **before** committing to Phase A.

---

## 3. Environment setup on Colab

> **Two-stack reality.** SemGauss-SLAM is pinned to **Python 3.10 / torch 1.12.1 / CUDA 11.6**, while DDS-SLAM's harness (`Addons/env/colab_setup.sh`, verified) runs on Colab's **native stack** (pip-only, torch 2.x / CUDA 12.x, tinycudann, pytorch3d). **Do not run both in one env.** Build a dedicated SemGauss conda env for **training**; use the **base** Colab Python for the **DDS-SLAM harness** (depth gen, DINO bake, eval, video). Each cross-stack handoff is a **file artifact on disk**, never an in-process import.

### 3.1 GPU / VRAM gate (run FIRST, before any build)

```bash
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "GPU: $GPU"
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
# Policy for the VRAM-heavy 3DGS methods. The paper ran on RTX 4090 = 24 GB, so 24 GB is the
# PRACTICAL floor for the headline FULL-Replica GATE (16-ch embedding + DINOv2 maps per Gaussian).
#   PHASE A (full Replica GATE):  < 24000 MB -> ABORT (cannot run the GATE; require A100/4090)
#   PHASE B (CRCD short snippets): < 16000 MB -> ABORT;  16000-23999 -> WARN (risky, short only)
#   A100 40/80 -> RECOMMENDED for both phases.
VRAM_ABORT_MB=${VRAM_ABORT_MB:-24000}   # Phase A default; Phase B may lower to 16000 via env
if [ "$VRAM_MB" -lt "$VRAM_ABORT_MB" ]; then
  echo "ABORT: $VRAM_MB MB < floor $VRAM_ABORT_MB MB. SemGauss-SLAM full-Replica needs >=24 GB (RTX 4090). Use A100."
  exit 78
fi
[[ "$GPU" =~ A100 ]] || echo "WARN: non-A100. Confirm VRAM headroom; prefer A100."
```
> The **24 GB Phase-A floor** is derived from the paper's 4090 hardware; **the 16 GB Phase-B floor and the warn-vs-abort band are still pre-seeded — confirm both via Clarification Protocol before publishing the gate (§10 q12).** Do NOT run the 8-scene Phase-A GATE on <24 GB (near-certain OOM, wastes a full pass).

### 3.2 SemGauss conda env (training) — HIGH-RISK BUILD, see §10 q15

> **There is no conda in this repo's Colab path.** Miniconda is **not preinstalled** on Colab. You must bootstrap it. cu116 nvcc against a CUDA-12.x host driver routinely fails to build custom extensions. **Do not begin Phase A until the user confirms the rasterizer compiles on the assigned runtime (§10 q15).**

**Step 1 — bootstrap Miniconda (not preinstalled):**
```bash
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
bash /tmp/mc.sh -b -p /opt/miniconda
source /opt/miniconda/etc/profile.d/conda.sh
conda --version    # verify
```

**Step 2 — create the pinned env (verify each wheel actually resolves; escalate if not):**
```bash
conda create -y -n sem_gauss python=3.10 && conda activate sem_gauss
# nvcc 11.6 toolkit inside the env (the host driver stays 12.x — this is the risk):
conda install -y -c "nvidia/label/cuda-11.6.0" cuda-toolkit
nvcc --version    # must report 11.6; if it reports the host 12.x, STOP and escalate
# torch 1.12.1 + cu116 — CONFIRM these wheels still install (they are old; may be delisted):
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 \
  -c pytorch -c conda-forge
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda.startswith('11.6')"
cd "${SEMGAUSS_REPO}"
pip install -r requirements.txt           # use the REPO's pinned versions verbatim; do not substitute
export WANDB_MODE=offline                  # headless Colab — wandb must not block
```

**Step 3 — build the custom rasterizer (the #1 failure mode):**
```bash
# Submodule names are UNKNOWN until .gitmodules is read (BLOCKER 1). Read it first:
cat .gitmodules 2>/dev/null || echo "NO .gitmodules — confirm rasterizer/simple-knn are vendored, not submodules"
git submodule update --init --recursive   # only if .gitmodules exists
export TORCH_CUDA_ARCH_LIST="8.0"          # A100 = sm_80; set to the detected GPU's arch
pip install ./diff-gaussian-rasterization-w-depth_sem_gauss   # use the REAL dir name from the repo
[ -d ./simple-knn ] && pip install ./simple-knn || echo "NOTE: simple-knn presence UNVERIFIED (§10 q6)"
```
> **Build-failure detection + escalation (mandatory):** after the build, run the §3.3 import smoke test. **If `import diff_gaussian_rasterization` fails, STOP and escalate (§10 q15) — do NOT** silently rebuild against the host CUDA 12.x (it changes numerics) and do NOT pip a generic rasterizer. Capture the full nvcc error into `${DRIVE_ROOT}/env_semgauss_buildfail.log`. Ask the user whether a prebuilt cu116 rasterizer wheel / container exists.

**Step 4 — place the DINOv2 `.pth`** from the project Google Drive into the exact `segmentation/` subpath + filename **per the README** (do not guess the filename).

### 3.3 Verification smoke test (must pass before Phase A)

```bash
conda activate sem_gauss && cd "${SEMGAUSS_REPO}"
python - <<'PY'
import torch, glob
assert torch.cuda.is_available(), "no CUDA"
assert torch.version.cuda.startswith("11.6"), f"expected cu116, got {torch.version.cuda}"
print("torch", torch.__version__, "cuda", torch.version.cuda)
import diff_gaussian_rasterization as dgr            # custom rasterizer importable
print("rasterizer OK:", dgr.__file__)
seg = glob.glob("segmentation/**/*.pth", recursive=True)
assert seg, "DINOv2/classifier .pth NOT found in segmentation/ — download from project Drive"
print("seg weights:", seg)
PY
```
Record `torch.__version__`, `torch.version.cuda`, GPU name, VRAM, the rasterizer `__file__`, and the repo commit hash into `${DRIVE_ROOT}/env_semgauss.txt`. If any line fails → **fix or escalate; do not proceed**.

---

## 4. PHASE A — Reproduce on SemGauss-SLAM's OWN dataset (Replica)

> **BLOCKED until BLOCKER 1 is cleared and §2.1 `configs_extracted.json` is complete.** Do not run Phase A on a guessed CLI/config schema.

**Goal:** match the **Replica 8-scene average** within tolerance. This is the GATE to Phase B.

### 4.1 Obtain data + weights (deterministic, with integrity checks)
```bash
DATA0="${SEMGAUSS_REPO}/$(jq -r .input_folder_root ${DRIVE_ROOT}/phaseA/configs_extracted.json)"  # real path from config
mkdir -p "$DATA0"
# Pull from the README's project Drive folder. gdown --folder OFTEN PARTIAL-FAILS on large folders.
gdown --folder "<README_FOLDER_URL>" -O /content/semgauss_drive || echo "gdown folder failed — see fallback"
# FALLBACK on gdown folder failure: pull each scene by direct file ID (gdown <FILE_ID>) or rsync from Drive mount.
```
**Integrity verification (fail-fast — a partial download silently corrupts the 8-scene GATE):**
```bash
EXPECTED_FRAMES=<from configs/replica/*.py>   # §10 q3 — do not proceed until known
for S in room0 room1 room2 office0 office1 office2 office3 office4; do
  D="$DATA0/$S"
  for sub in <rgb_subdir> <depth_subdir> <semantic_subdir>; do
    [ -d "$D/$sub" ] || { echo "MISSING $S/$sub — download incomplete, ABORT"; exit 1; }
  done
  N=$(ls "$D/<rgb_subdir>" | wc -l)
  [ "$N" -eq "$EXPECTED_FRAMES" ] || { echo "$S frame count $N != $EXPECTED_FRAMES — ABORT"; exit 1; }
  [ -f "$D/<gt_traj_file>" ] || { echo "$S missing GT traj — ABORT"; exit 1; }
done
echo "Replica integrity OK"
```
> Move GT meshes to a known dir for mesh eval. **Resolve `EXPECTED_FRAMES`, the subdir names, the GT-traj filename, and the DINOv2 `.pth` filename from the configs/README before declaring the download complete (§10 q3).**

### 4.2 Configs — read, never assume
Open every `configs/replica/<scene>.py` and **append to `configs_extracted.json`**: mapping iters, tracking iters, LRs, resolution, frames-per-scene, **semantic class count**, DINOv2 variant, the 16-d projection, the **default semantic path (predicted vs GT)**. **Do not change them** for the faithful run — only set the per-scene `input_folder`/output paths. If a config references a missing path/weight → **escalate**.

### 4.3 EXACT run commands (8-scene loop)
There is **no built-in "run all"**. Script it with per-scene isolation + sentinels. **Replace `${CLI}` with the verified `cli_pattern` from §2.1:**
```bash
SCENES="room0 room1 room2 office0 office1 office2 office3 office4"
PA=${DRIVE_ROOT}/phaseA; mkdir -p "$PA"
for S in $SCENES; do
  M="$PA/$S"; mkdir -p "$M"
  [ -f "$M/.DONE" ] && { echo "skip $S"; continue; }
  ( set -o pipefail; SECONDS=0
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 30 > "$M/vram.log" &  VPID=$!
    ${CLI} "configs/replica/${S}.py" 2>&1 | tee "$M/run.log"     # ${CLI} from configs_extracted.json
    kill $VPID 2>/dev/null; echo "elapsed_sec=$SECONDS" >> "$M/run.log"
  ) || { echo "FAILED_$S" >> "$PA/_failures.log"; echo "$S crashed — continuing"; continue; }
  touch "$M/.DONE"
done
```
> **Reproduce which semantic path?** The paper reports both "Ours" (predicted) and "Ours (GT)". Default to whatever `configs/replica/*.py` ships; **record which** in `configs_extracted.json`. If ambiguous → **escalate (§10)**.

### 4.4 Evaluate exactly as the paper does
- **Use the metrics computed inside the SemGauss-SLAM pipeline/utils** (the repo's own eval), not the DDS harness. Parse per-scene numbers into `$PA/<scene>/metrics.json`.
- **Mesh eval** — run the repo's mesh eval after editing its hardcoded paths (1–2 scenes is enough to sanity-check; not part of the headline gate).
- **Compute the 8-scene average** yourself (no aggregator in the repo): mean of ATE(cm), Depth-L1(cm), PSNR, SSIM, LPIPS → `$PA/replica_average.json`.
- **Confirm and record the ATE alignment convention** the repo uses (rigid SE3 vs Sim3) by reading the eval code (§10 q4).

### 4.5 Artifacts to record (Phase A)
`env_semgauss.txt`; `configs_extracted.json`; per-scene `run.log`, `vram.log`, `metrics.json`, recovered mesh; `replica_average.json`; (optional) 1–2 ScanNet per-scene ATE; build-fail log if applicable.

### 4.6 DECISION GATE (must pass before Phase B)

> **The Table I paper targets and the pass tolerances below are NOT yet user-confirmed. Per §0.2, write `GATE_PASS`/`GATE_FAIL` ONLY after the user confirms (a) the exact Table I numbers from the PDF and (b) the numeric tolerances (§10 q13). Until then this stage is `.BLOCKED`.**

| Metric (8-scene Replica avg) | Paper Table I target | Pass tolerance (pre-seeded — confirm) |
|---|---|---|
| ATE RMSE (cm) | `<from PDF>` | abs margin TBD |
| Depth-L1 (cm) | `<from PDF>` | abs margin TBD |
| PSNR (dB) | `<from PDF>` | lower-bound TBD |
| SSIM | `<from PDF>` | lower-bound TBD |
| LPIPS | `<from PDF>` | upper-bound TBD |
| Semantic mIoU, 4-scene novel-view (Table III) | `<from PDF>` | lower-bound TBD |

**GATE rule (also subject to user confirmation, §10 q13):** geometry+render (ATE, Depth-L1, PSNR, SSIM, LPIPS) must pass on the 8-scene average. **Whether semantic mIoU is blocking for Phase B geometry is a policy choice the user has not confirmed — escalate it (§10 q13); do not unilaterally carve it out.** On 3DGS, also define an **acceptable run-to-run std** (densification stochasticity) so the GATE is not failed by noise (§8.1). If the GATE fails: do **not** publish CRCD numbers — escalate with the per-scene table + config diff. Write `$PA/GATE_PASS` or `$PA/GATE_FAIL` (or leave `.BLOCKED`).

---

## 5. PHASE B — CRCD adaptation

> Mirror the **phase/sentinel structure of `Addons/colab/run_crcd_4snippets.sh`** exactly. Staging, depth, semantic prep, eval, and video reuse the harness; **only the SLAM step (Phase 3) is BLOCKED until BLOCKER 1 + §2.1 are cleared.** Run staging/depth/sanity for all snippets even while SLAM is blocked (failure isolation).

### 5.0 The 5 target snippets
| CRCD snippet | config stem | staged dir | frames |
|---|---|---|---|
| C_1/001 | `c1_001` | `data/CRCD/C1_001` | 360 |
| E_3/005 | `e3_005` | `data/CRCD/E3_005` | 265 |
| C_3/001 | `c3_001` | `data/CRCD/C3_001` | 1527 |
| G_3/001 | `g3_001` | `data/CRCD/G3_001` | 1987 |
| C_2/001 | `c2_001` | `data/CRCD/C2_001` | 730 |

> Frame counts from `crcd_depth_gen_remainder_20260616.sh` (E3_005=265, C3_001=1527, G3_001=1987) and the existing c1/c2 configs (360/730). **Confirm episode→snippet→frame counts before authoring configs (§10 q10).**

### 5.1 F:/ → Drive → /content staging
CRCD is on the user's **F:/** drive (Colab cannot read F:/). One-time: user copies `CRCD-Published` to `/content/drive/MyDrive/Datasets/CRCD-Published/`. Then per snippet, **reuse the staging from the 4-snippet runbook** (tarball-first, cp/rsync fallback): stage `rgb/`, `rgbright/`, `semantic_instance/`, `groundtruth.txt`, `intrinsics.yaml` to `/content/crcd_raw/<EP>_snippet_<SID>`, sentinel `.STAGED`.

**Preprocess** (rectify, write the on-disk CRCD layout):
```bash
python Addons/preprocess/preprocess_crcd_published.py \
  --snippet_dir /content/crcd_raw/<EP>_snippet_<SID> \
  --calib_pkl   /content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl \
  --output_dir  data/CRCD/<NAME>
```
This produces `video_frames/{NNNNNNl,NNNNNNr}.png`, `masks/`, `semantic_class/`, `groundtruth.txt` (TUM), `rectified_calib.txt`.

> **VERIFIED LABEL-SPACE LANDMINE (do not trust §-prose claims of a clean 4-class map).** `preprocess_crcd_published.py` lines 135/142-143 write:
> - `masks/<id>.png` = `(sem_rect == tool_pixel_value) * 255` → **values are {0, 255}, NOT {0,1}**.
> - `semantic_class/<id>.png` = `np.clip(sem_rect, 0, 255)` of the **raw `coco_id+1` instance map** → this is **NOT guaranteed to be the index set {0=bg,1=Liver,2=Gallbladder,3=Tool}**. If `semantic_instance` contains instance IDs or coco_ids outside {0,1,2,3}, both path-A `L_s` supervision, the `generate_video.py` seg overlay (`CLASS_PALETTE` defines only keys 0-3, verified line 158), and a 4-class classifier-head resize all break.
>
> **Mandatory pre-flight check (run on a sample of each of the 5 snippets BEFORE relying on path A):**
> ```bash
> python - <<'PY'
> import cv2, glob, numpy as np
> for f in glob.glob("data/CRCD/<NAME>/semantic_class/*.png")[:20]:
>     print(f, np.unique(cv2.imread(f, cv2.IMREAD_UNCHANGED)))
> PY
> ```
> If the union of values is **not exactly ⊆ {0,1,2,3}** on every snippet → **STOP, add an explicit `coco_id → {0,1,2,3}` remap step in preprocessing, confirm `tool_pixel_value` yields class 3, and escalate if the label space differs across snippets (§10 q16).** Also confirm whether SemGauss expects mask values **255 vs 1** (§10 q16) and whether it wants class labels as 0/255 or a 0..K index. Sentinel `.PREPROCESSED` only after this check passes.

### 5.2 Depth: MoGe-2 + stereo-scaled (mandatory — SemGauss is RGB-D)
Reuse the **exact** pipeline (Phases 1.5 + 1.6 of `run_crcd_4snippets.sh`; the depth-only batch `crcd_depth_gen_remainder_20260616.sh` already covers **E3_005, C3_001, G3_001** and caches to `/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth,.sc_factor}` — **rehydrate from cache** if present).

1. Symlink left frames so MoGe sees `*-left.png`: `for f in video_frames/*l.png; do ln -sf "$PWD/$f" "_moge_in/$(basename $f l.png)-left.png"; done`
2. MoGe-2 metric depth (base stack). **Note `generate_depth_moge.py` WRITES `<fid>-left_depth.npy` (float32 npy, value = depth_m × depth_scale) — NOT a PNG** (verified docstring lines 11/19):
   ```bash
   python Addons/depth/generate_depth_moge.py --rgb _moge_in --out _moge_npy \
     --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0      # metric-direct (no --ref)
   ```
3. **npy → uint16 PNG.** There is **no standalone npy→PNG script**; the conversion exists **only as an inline `PYEOF` block inside `run_crcd_4snippets.sh` (verified Phase 1.5, lines 227-241)**. Either **call that block verbatim** or **extract it into a reusable step** `Addons/depth/npy_to_png16.py` and reference it (preferred for reproducibility — record that you authored it). The block writes `data/CRCD/<NAME>/depth/<fid>.png` as uint16 = `depth_m × 10000`.
   > **uint16 clipping headroom:** uint16 max 65535 / 10000 = **6.5535 m**; with `max_depth_m=5.0` there is headroom, so no clipping. **Assert this holds for any snippet whose config raises `max_depth_m` above 6.55 (§8.6).**
4. **Stereo anchor scale.** **VERIFIED CADENCE CONFLICT:** the repo computes a **single frame-0 SGBM anchor applied globally** (`run_crcd_4snippets.sh` Phase 1.6, SGBM on frame 0 only, `sc_factor = median(stereo/MoGe)`), which **directly contradicts the GLOBAL spec's "stereo-match ~every 100 frames."** **This is a hard pre-Phase-B decision — escalate (§10 q9):** keep the single anchor (matches all prior repo results) vs implement periodic (~every 100 frames) re-scaling via `Addons/depth/generate_depth_stereo.py`. **Do NOT author CRCD SemGauss configs until this is settled.** Frame-0 SGBM parameters (for reference): `StereoSGBM_create(minDisparity=0,numDisparities=128,blockSize=7,P1=8*49,P2=32*49,disp12MaxDiff=1,uniquenessRatio=10,speckleWindowSize=100,speckleRange=32,MODE_SGBM_3WAY)`, `depth = baseline_m*fx/disp`, mask `0.05<d<3.0` ∧ MoGe-valid.
5. **Apply sc_factor — VERIFIED "F7 latent bug":** in DDS-SLAM the config-patch logic scales only `trunc`; `near/far/range_d/depth_trunc` stay **unscaled** (verified warning at `run_crcd_4snippets.sh` line 372). **For SemGauss the analogous depth-threshold parameters are UNKNOWN until the config is read.** After reading SemGauss's config, **enumerate every depth-threshold parameter and confirm ALL are scaled by sc_factor (not just one)**, or **constrain snippets to `|log(sc_factor)| ≤ 0.1`**. Escalate if `sc_factor` is far from 1.0 (§10 q11).

### 5.3 Semantic-input handling on CRCD (LOAD-BEARING — escalate before baking anything)
SemGauss-SLAM needs class labels for `L_s` and DINOv2 features for `L_f`.

- **DINOv2 features (`L_f`) — DO NOT bake until §2.1 answers `dino_internal_vs_precomputed`.** **Most 3DGS-semantic systems extract DINOv2 internally at train time.** If SemGauss does, **the harness DINO bake is unnecessary and would DOUBLE-SUPPLY/mismatch the feature space** — skip it. Only if precompute is genuinely required:
  - Determine SemGauss's **exact** backbone/patch and its native-dim→16 projection (PCA vs learned linear vs decoder) from `segmentation/` + model code (§10 q1). **Reproduce ITS projection — do not invent a 384→16 mapping.**
  - `Addons/dino/generate_dino_features.py` defaults to `dinov2_vits14` (C=384). **If SemGauss uses ViT-B/L or a different patch size, regenerate with that backbone**, not `vits14`.
  ```bash
  python Addons/dino/generate_dino_features.py \
    --rgb_dir data/CRCD/<NAME>/video_frames --rgb_glob '*l.png' \
    --out_dir data/CRCD/<NAME>/dino --backbone <SEMGAUSS_BACKBONE>
  ```
- **Class labels (`L_s`) — the paper's indoor classifier is INVALID for surgical anatomy.** Two paths:
  - **(A) Drive `L_s` from CRCD's 4-class GT masks** (`semantic_class/`, after the §5.1 remap check). Matches the paper's **"Ours (GT)"** path. **Recommended.** Requires resizing the classifier's final layer / class-indexed buffers to **4 classes** (and confirming the label encoding SemGauss expects, §10 q16).
  - **(B) Retrain a small CRCD-specific seg head** on frozen DINOv2 for the 4 classes. Matches **"Ours"** but needs labeled training frames.
  - **DO NOT silently reuse the paper's indoor classifier. Escalate the A-vs-B choice (§10 q7).** Until resolved, mark the snippet's semantic stage `.BLOCKED`.

### 5.4 Per-snippet config authoring
> **BLOCKED until §2.1 `configs_extracted.json` is complete** (the SemGauss config schema — `input_folder` layout, intrinsics keys, depth-scale key, class-count key, mask-input support — cannot be known until the repo is read). **Do not author any SemGauss-side CRCD config before this.**

**Only c1_001 and c2_001 have `_paperfaith_lrfix` configs.** You must author the DDS-side staging configs for **e3_005, c3_001, g3_001** (so depth/sc_factor/masks/sem are produced identically), from the c1/c2 template:

```yaml
# configs/CRCD/<stem>_paperfaith_lrfix.yaml
inherit_from: configs/CRCD/crcd_paperfaith_lrfix.yaml
timesteps: <FRAMES>                                   # e3_005:265  c3_001:1527  g3_001:1987
mapping:
  bound: [[?,?],[?,?],[?,?]]                          # HAND-DERIVE; see below
  marching_cubes_bound: [[?,?],[?,?],[?,?]]
data:
  datadir: data/CRCD/<NAME>
  trainskip: 1
  output: output/CRCD/<NAME>_paperfaith_lrfix
  exp_name: demo
```
> **`mapping.bound` is load-bearing and must NOT be guessed.** Derive each snippet's bound from its staged frame-0 depth (`depth/<fid>.png` p10/median/p90) + GT extent (the Phase-2 motion sanity prints extent). **Escalate who signs off the bounds (§10 q10).**

For the **SemGauss-SLAM run config** (in the SemGauss repo's own format, per §2.1), override: `input_folder = data/CRCD/<NAME>` (RGB=`video_frames/*l.png`, depth=`depth/`), intrinsics (H=720, W=1280, fx=fy=1096.696, cx=622.808, cy=383.126 — confirm against `rectified_calib.txt`), the **depth-scale key set to 10000** (the verified §5.2 MoGe scale; key name from §2.1, §10 q-depth), the **4-class** count, the chosen `L_s` source (A/B), and the `sc_factor` applied per §5.2 step 5. **If SemGauss supports a per-pixel loss mask, wire `masks/` (tool pixels) out of tracking/mapping losses; if it does NOT (`mask_input_supported=false` in §2.1), escalate (§10 q14) — you cannot exclude tools.** SemGauss assumes a **static rigid scene**; CRCD tissue **deforms** and tools move — record deformation as an out-of-distribution caveat (§10 q14).

### 5.5 EXACT run commands (all 5 snippets, isolated + resumable)
> **`python sem_gauss.py …` below is a PLACEHOLDER** — replace with the verified `cli_pattern` (§2.1). The output dir naming `<NAME>_s<SEED>` is **mandatory** for `aggregate_ab.py` (§7.3).
```bash
DATE=$(date +%Y%m%d)
DRIVE_ROOT=/content/drive/MyDrive/Outputs/semgauss_crcd_5snippets_${DATE}; mkdir -p "$DRIVE_ROOT"
SNIPPETS=( "C1_001 C_1 001 360" "E3_005 E_3 005 265" "C3_001 C_3 001 1527" "G3_001 G_3 001 1987" "C2_001 C_2 001 730" )
for ROW in "${SNIPPETS[@]}"; do
  read -r NAME EP SID FRAMES <<< "$ROW"
  CELL="${NAME,,}_semgauss"                      # aggregator cell name (see §7.3)
  DST="$DRIVE_ROOT/${CELL}_s${SEED}"; mkdir -p "$DST"     # <cell>_s<seed> — REQUIRED by aggregate_ab.py
  [ -f "$DST/.DONE" ] && { echo "skip $NAME"; continue; }
  OUT=output/CRCD/${NAME}_semgauss_s${SEED}
  ( set -o pipefail; SECONDS=0
    conda activate sem_gauss
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 30 > "$DST/vram.log" & VPID=$!
    ${CLI} configs/crcd/${NAME,,}_semgauss.<ext> 2>&1 | tee "$DST/run.log"   # ${CLI}+ext from §2.1
    kill $VPID 2>/dev/null; echo "elapsed_sec=$SECONDS" >> "$DST/run.log"
  ) || { echo "FAILED_SLAM_$NAME" >> "$DRIVE_ROOT/_failures.log"; echo "$NAME crashed — continuing"; continue; }
  # ... then §6 eval + video, then payload.tgz (§7.3) + touch "$DST/.DONE"
done
```
> **Trajectory adapter (VERIFIED gap — `kitti_to_tum.py` only REFORMATS, it does NOT upsample keyframes).** `sim3_ate.py` (verified) takes est lines of **12 (3×4 row-major) or 16 (4×4) floats, translation cols [3,7,11]**, pairs `est[i]↔GT[i]` by index. `generate_video.py` (verified line ~135) truncates GT to `len(est)` with **no validity check**. **3DGS SLAM systems commonly emit one pose per KEYFRAME, not per frame** — if so, ATE and the trajectory panel are **silently corrupted**.
> 1. Determine SemGauss's actual `traj_cadence` + `traj_format` (§2.1).
> 2. If **per-frame** and KITTI/4×4: place at `$OUT/demo/est_c2w_data.txt`; if quaternion-TUM, reformat to cols [3,7,11].
> 3. If **per-keyframe**: author an **explicit expansion to per-frame** — either interpolate keyframe poses to every frame, OR subsample GT by the identical keyframe indices — **BEFORE `sim3_ate.py`** (`kitti_to_tum.py` cannot do this). Record the chosen method.
> 4. Place rendered RGB at `$OUT/<frame:04d>.{jpg|png}` and rendered depth at `$OUT/depth/<frame:04d>.png` (uint16 ×10000).

### 5.6 Output layout (under MyDrive/Outputs)
```
/content/drive/MyDrive/Outputs/semgauss_crcd_5snippets_<DATE>/
  <cell>_s<SEED>/  run.log  vram.log  frame_alignment.txt
                   render_metrics.txt          # NAME REQUIRED by aggregate_ab.py (NOT render_eval.txt)
                   render_eval.csv  sim3_metrics.txt  depthL1.txt(.BLOCKED)  summary.txt
                   <NAME>_6panel.mp4  est_c2w_data.txt  diag/  payload.tgz  .DONE
  _render_summary.csv  _failures.log  _clarifications.log  COMBINED_SUMMARY.txt  runbook.log  env_semgauss.txt
```

---

## 6. CRCD evaluation (DDS-SLAM harness)

Run from the **base** Colab Python (the harness env), per snippet. **Output to `render_metrics.txt` (the name `aggregate_ab.py` reads), not `render_eval.txt` (§7.3).**

**Phase-0 prerequisite — patch `eval_rendering.py` choice list BEFORE any eval (VERIFIED hard crash otherwise):** `eval_rendering.py` uses `argparse choices=list(PAPER_REFERENCES.keys())` (verified line 74); the keys present are only `CRCD (C1_001)`, `CRCD (C2_001)`, `CRCD (F1_002)`, `CRCD (F3_007)` (verified lines 55-58). Passing `--sequence "CRCD (E3_005)"` / `C3_001` / `G3_001` **hard-crashes with argparse exit-2 before any eval runs.** This is a **required setup STEP**, not an inline note: add three keys to `PAPER_REFERENCES` with `None` values:
```python
'CRCD (E3_005)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
'CRCD (C3_001)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
'CRCD (G3_001)': {'PSNR': None, 'SSIM': None, 'LPIPS': None},
```
**This is a repo edit — commit it and record it for reproducibility.** The §7 smoke test must `grep` these three keys and fail if absent.

**ATE (sim3) — headline trajectory metric:**
```bash
python Addons/eval/sim3_ate.py \
  --est "$OUT/demo/est_c2w_data.txt" --gt "data/CRCD/$NAME/groundtruth.txt" \
  --name "$NAME" --out "$DST/sim3_metrics.txt" | tee -a "$DST/sim3_metrics.txt"
```
Reports Sim3 ATE (mm), recovered scale `s`, est/GT path ratio, |Pearson|dom. **Headline Sim3 for CRCD** (MoGe depth is up-to-scale); never headline the rigid/raw RMSE (scale-confounded).

**PSNR / SSIM / LPIPS — output to `render_metrics.txt`:**
```bash
python Addons/eval/eval_rendering.py \
  --gt_dir "data/CRCD/$NAME/video_frames" --render_dir "$OUT" --name "SemGauss-SLAM" \
  --output_csv "$DST/render_eval.csv" --summary_csv "$DRIVE_ROOT/_render_summary.csv" \
  --sequence "CRCD ($NAME)" > "$DST/render_metrics.txt" 2>&1
grep -E "Rendered:|PSNR:|SSIM:|LPIPS:" "$DST/render_metrics.txt"
```
CRCD has no paper PSNR reference (dataset post-dates the paper) — expected; the `None` entries suppress the comparison line.

**Depth-L1 — NO implementation exists; the FIXED metric set requires it; therefore this is a HARD GATE, not a skip (VERIFIED).** There is **no `depth_l1`/Depth-L1 code anywhere in `Addons/`** and CRCD ships **no GT depth** (MoGe is generated). Per §0.2 and the global "CRCD metric set is fixed/non-negotiable":
1. **Escalate the Depth-L1 reference choice to the user BEFORE Phase B** (this is a hard gate on the metric, §10 q8). Candidate references: rendered SLAM depth vs (a) MoGe-2 input depth (self-consistency), (b) frame-0 stereo-SGBM depth, (c) held-out stereo depth.
2. Once the reference is fixed, **author the Depth-L1 script + the rendered↔reference pairing** (reuse `sim3_ate.py`'s index-pairing convention).
3. **Until resolved, write `depthL1.txt = "BLOCKED: no GT depth reference (see _clarifications.log)"` and mark the metric `.BLOCKED` — do NOT silently skip, do NOT invent one.**

**Canonical 6-panel video:**
```bash
UNC=""; [ -d "$OUT/uncert" ] && UNC="--uncert_dir $OUT/uncert"
python Addons/viz/generate_video.py \
  --rgb_input_dir "data/CRCD/$NAME/video_frames" --rgb_input_pattern '*l.png' \
  --rgb_output_dir "$OUT" --rgb_output_pattern '[0-9]*.jpg' \
  --depth_input_dir "data/CRCD/$NAME/depth" --depth_output_dir "$OUT/depth" --depth_norm robust \
  --seg_dir "data/CRCD/$NAME/semantic_class" --seg_pattern '*.png' --skip_raw_seg --seg_classmap $UNC \
  --trajectory_est "$OUT/demo/est_c2w_data.txt" --trajectory_gt "data/CRCD/$NAME/groundtruth.txt" --trajectory_raw \
  --output "$DST/${NAME}_6panel.mp4" --fps 15
```
> The seg panel uses `CLASS_PALETTE` (keys 0-3 only). **It silently mis-colors any class ID >3** — the §5.1 remap check is a prerequisite for a correct seg panel.

**Extended trajectory summary** — reuse the inline Phase-6 block of `run_crcd_4snippets.sh` (raw RMSE, SE3 ATE, Sim3 ATE + scale, est/GT path ratio, per-axis Pearson) → `summary.txt`.

**Recommended diagnostics (author per snippet, save under `$DST/diag/`):** trajectory overlay vs GT (Sim3-aligned 3D + top-down), per-frame ATE & per-frame PSNR curves, depth-error heatmaps (pending the Depth-L1 reference decision), semantic-overlay frames, keyframe-coverage map.

---

## 7. Standard CLI contract — `run_semgauss_slam.sh` (DELIVERABLE TO BUILD — does not exist yet)

> **BLOCKER 2:** `Addons/colab/run_semgauss_slam.sh` is **verified absent**. The agent must **author it from scratch**, mirroring `run_crcd_4snippets.sh`'s phase/sentinel/isolation structure. It **cannot be end-to-end tested until BLOCKER 1 + §2.1 are cleared** (the SLAM CLI/config schema is required for Phases A and B-3). Author the staging/depth/eval/video/aggregate stages now; leave the SLAM-invocation stage parameterized by `${CLI}` from §2.1.

### 7.1 Interface
```bash
bash Addons/colab/run_semgauss_slam.sh <PHASE> [SNIPPET] [SEED]
#   PHASE   : env | smoke | phaseA | phaseB | all
#   SNIPPET : (phaseB only) C1_001 | E3_005 | C3_001 | G3_001 | C2_001 | ALL   (default ALL)
#   SEED    : integer (default 0)
```

### 7.2 Env vars (read, with defaults)
| Var | Default | Meaning |
|---|---|---|
| `DRIVE_ROOT` | `/content/drive/MyDrive/Outputs/semgauss_crcd_5snippets_$(date +%Y%m%d)` | output root |
| `SEMGAUSS_REPO` | `/content/SemGauss-SLAM` | the official repo checkout (BLOCKER 1) |
| `DDS_REPO` | `/content/DDS-SLAM` | this harness checkout |
| `VRAM_ABORT_MB` | `24000` (Phase A) / `16000` (Phase B) | abort floor (confirm §10 q12) |
| `SEED` | `0` | determinism seed |
| `SEMANTIC_PATH` | `GT` | `GT` (CRCD 4-class masks) or `HEAD` (retrained seg head) — §5.3; `HEAD` requires user go-ahead |
| `WANDB_MODE` | `offline` | headless |

### 7.3 Cross-method aggregation — the VERIFIED contract (this was broken as previously specified)

**VERIFIED facts about `aggregate_ab.py`:** it (a) **hardcodes** cell names (`c1_001_canon_base/uncert/uncert_dino` + `trail3_moge2_*`, lines 71-74), (b) globs `<cell>_s<seed>` dirs (line 53), and (c) reads `sim3_metrics.txt` + **`render_metrics.txt`** from **inside `payload.tgz`** (lines 85-89). **VERIFIED incompatibilities:** `run_crcd_4snippets.sh` writes `render_eval.txt` (line 440) and packs `payload.tgz` with only `demo/ ckpts/ depth/ renders_rgb/` — it does **NOT** include `render_metrics.txt` or `sim3_metrics.txt`, so **`run_crcd_4snippets.sh` is itself incompatible with `aggregate_ab.py`.** The aggregator-compatible producers are `run_cell.sh` / the `a100_*` scripts, which write `render_metrics.txt`.

**Chosen approach (a) — make SemGauss outputs match `aggregate_ab.py`'s contract EXACTLY** (do all four):
1. Name each output dir `<cell>_s<SEED>` where `<cell> = <name_lower>_semgauss` (e.g. `c1_001_semgauss_s0`). (§5.5 already does this.)
2. Run `eval_rendering.py` redirected to a file named **`render_metrics.txt`** (not `render_eval.txt`). (§6 already does this.)
3. **Pack `render_metrics.txt` + `sim3_metrics.txt` INTO `payload.tgz`** (in addition to `demo/ ckpts/ depth/ renders_rgb/`):
   ```bash
   cp "$DST/render_metrics.txt" "$DST/sim3_metrics.txt" "$OUT/" 2>/dev/null || true
   SHIP=(render_metrics.txt sim3_metrics.txt demo); [ -d "$OUT/ckpts" ] && SHIP+=(ckpts); [ -d "$OUT/depth" ] && SHIP+=(depth)
   ls "$OUT"/*.jpg >/dev/null 2>&1 && { mkdir -p "$OUT/renders_rgb"; mv "$OUT"/*.jpg "$OUT/renders_rgb/"; SHIP+=(renders_rgb); }
   tar czf "$DST/payload.tgz" -C "$OUT" "${SHIP[@]}"
   ```
4. **Extend `aggregate_ab.py`'s hardcoded cell list** to include the 5 SemGauss cells (`c1_001_semgauss`, `e3_005_semgauss`, `c3_001_semgauss`, `g3_001_semgauss`, `c2_001_semgauss`). **Confirm with the user whether to extend the existing script or write a new SemGauss-aware aggregator (§10 q17)** — the existing CRCD block parses Sim3 ATE + |Pearson| + render from `payload.tgz`, which is exactly the per-snippet contract above.

**Confirm the intended cross-method path end-to-end** with the user before publishing (since `run_crcd_4snippets.sh` is incompatible, the cross-method comparison cannot simply reuse prior CRCD outputs — they must be re-emitted with `render_metrics.txt` in `payload.tgz`, or a new aggregator written; §10 q17).

### 7.4 Stage markers (stdout, grep-able)
```
[STAGE] env.start / env.ok / env.FAIL / env.BLOCKED(rasterizer)
[STAGE] smoke.ok / smoke.FAIL
[STAGE] phaseA.<scene>.start / .done / .FAIL   ;  phaseA.gate.PASS / .FAIL / .BLOCKED
[STAGE] phaseB.<NAME>.stage / .preprocess / .depth / .semantic / .slam / .eval / .video / .done / .FAIL / .BLOCKED
[STAGE] aggregate.done
```

### 7.5 Exit codes
| Code | Meaning |
|---|---|
| 0 | all requested stages done (or cleanly isolated-skipped) |
| 10 | Phase A GATE failed |
| 20 | one or more Phase B snippets failed (others may have succeeded — isolation honored) |
| 30 | a load-bearing clarification is BLOCKING (see `_clarifications.log`) — includes BLOCKER 1/2 unresolved |
| 78 | GPU/VRAM below abort floor |
| 1 | unexpected fatal (env/build) |

> The script **must not** abort the whole run on a single snippet failure: wrap each snippet in a subshell, log to `_failures.log`, continue.

---

## 8. Failure modes, determinism, checkpointing, logging — plus EXPERT INPUT

### 8.1 Determinism / seeding
- **Defer the exact mechanism until §2.1 reveals the entry point and `seed_key`.** If SemGauss has a seed config key, inject the seed there. If not, use a **minimal non-invasive wrapper** (the `run_cell.sh` runpy + override-yaml trick is **DDS-SLAM-specific and will NOT transfer to an unknown config schema** — do not assume it). Set, before any CUDA work: `torch.manual_seed; np.random.seed; random.seed; torch.cuda.manual_seed_all`; `PYTHONHASHSEED=$SEED`; TF32 off (`allow_tf32=False`) — **without editing the repo model in place** if at all possible.
- **3DGS caveat:** Gaussian init + adaptive densification are stochastic; **bit-exact reproducibility is NOT expected** even with fixed seeds. **Run ≥3 seeds** on the headline CRCD snippet (C2_001, best-case) and report **mean±std** via the aggregator. **Define an acceptable run-to-run std** for the GATE (§4.6, §10 q13) so noise does not fail a valid reproduction.

### 8.2 GPU / VRAM / runtime logging
- Per run, background `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv -l 30 > vram.log`; capture wall time via `SECONDS`; log peak VRAM (a **reportable result** for this method). If OOM: log it, reduce resolution **only with user approval** (resolution is load-bearing for Table I), else mark `.FAIL`.

### 8.3 Sim3-alignment correctness checks
- After `sim3_ate.py`: assert recovered scale `s` finite and > 0; for CRCD expect `s` measurably off 1.0 — if `s ≈ 1.000` exactly AND rigid≈sim3, suspect est was already metric or GT/est got swapped. **|Pearson|dom near 0** ⇒ shape doesn't track GT (sub-SNR or broken tracking) — flag. `est/GT path ratio > 3` ⇒ jitter, not tracking. Headline Sim3; never the rigid/raw RMSE.

### 8.4 Frame-subsampling / index alignment (HARD GATE — fails the snippet)
This is the silent-failure hot spot. Enforce **before** the SLAM run and as a **hard gate** (not a logged note):
- `count(video_frames/*l.png) == count(depth/*.png) == count(semantic_class/*.png) == count(masks/*.png) == GT-pose-count` (or a documented, intentional offset). The harness pairs `est[i]↔GT[i]` by index.
- DINO features (if precomputed): `count(dino/*) == count(video_frames/*l.png)`, same stem ordering.
- After SLAM: **`count(est poses) == frames`** (or apply the documented keyframe stride to GT, §5.5). Write a `frame_alignment.txt` per snippet with all counts; **if counts are unequal and no intentional offset is documented, write `.FAIL` and skip eval for that snippet** (do not let `sim3_ate.py`/`generate_video.py` silently mis-pair via truncation).

### 8.5 Recommended diagnostics — see §6 list (save under `$DST/diag/`).

### 8.6 Sanity gates that catch silent misconfiguration (run BEFORE trusting any number)
- **GT motion-profile sanity** (Phase 2 of `run_crcd_4snippets.sh`): extent, path length, per-frame motion, active fraction. Most CRCD snippets are **sub-SNR** (active median < 0.5 mm/f) — tracker quality then **untrustable**, flag not headline. **C1_001 is explicitly REJECT-for-tracker / keep-for-render; C2_001 (29 mm extent) is the best case.**
- **Static-scene vs deformation (VERIFIED concern):** SemGauss assumes a rigid static scene; CRCD tissue **deforms** (not maskable) and tools move. Tool pixels can only be excluded if `mask_input_supported` (§2.1); deforming tissue corrupts the 3DGS map and ATE on exactly the sub-SNR/REJECT snippets above. **Confirm with the user (a) deformation is reported as an OOD caveat and (b) which CRCD metrics are headline-able per snippet given the REJECT verdicts already encoded in c1/c2 (§10 q14).**
- **Tool-mask wiring:** if SemGauss supports a mask, **verify tool pixels are actually excluded from its losses** (read the loss code); if it does not, escalate (§10 q14) — moving instruments will corrupt the "static" map.
- **sc_factor sanity:** if `|log(sc_factor)| > 0.1`, the §5.2-step-5 threshold-scaling inconsistency applies — flag and escalate (§10 q11).
- **Depth scale consistency (HARD pre-run gate):** assert MoGe PNGs are uint16 at scale 10000 AND SemGauss's depth-scale key (§2.1) = 10000; assert `max_depth_m × 10000 ≤ 65535` (no uint16 clipping). A mismatch silently destroys geometry + ATE.
- **Render-count gate:** if `count($OUT/*.jpg) < frames`, render eval is partial — `aggregate_ab.py` already excludes <100-frame seeds (verified line 100); replicate that guard.

### 8.7 Checkpointing / resume & logging conventions
- Per-stage sentinels: `.STAGED / .PREPROCESSED / depth/.DONE / .sc_factor / .DONE`; `.BLOCKED` for escalated stages. SLAM resume by checking `est_c2w_data.txt` line-count ≥ frames. Rehydrate depth from the Drive MoGe cache.
- Single combined log per run via `exec > >(tee -a "$DRIVE_ROOT/runbook.log") 2>&1`. Naming per `Addons/docs/NAMING_CONVENTION.md`: `SemGauss-SLAM_<changes>_<YYYYMMDD>`.

---

## 9. Deliverables checklist (must exist in Outputs when done)

**Pre-Phase-A (blockers/escalations):**
- [ ] `${SEMGAUSS_REPO}` checked out + commit recorded (BLOCKER 1 cleared)
- [ ] `Addons/colab/run_semgauss_slam.sh` authored (BLOCKER 2 cleared)
- [ ] `phaseA/configs_extracted.json` complete (entry point, CLI, config schema, traj cadence/format, mask support, dino internal/precompute, all hyperparams)
- [ ] rasterizer build confirmed on the runtime, OR `env_semgauss_buildfail.log` + escalation (§10 q15)
- [ ] `eval_rendering.py` patched with E3_005/C3_001/G3_001 keys (committed + grep-verified in smoke)
- [ ] Depth-L1 reference decided + script authored, OR `.BLOCKED` recorded (§10 q8)
- [ ] cross-method aggregator path decided (extend vs new) + tested (§10 q17)

**Phase A (`${DRIVE_ROOT}/phaseA/`):**
- [ ] `env_semgauss.txt`; `configs_extracted.json`
- [ ] per-scene `run.log`, `vram.log`, `metrics.json` for all 8 Replica scenes
- [ ] `replica_average.json`; 4-scene novel-view mIoU; (optional) ScanNet ATE; (optional) mesh-eval output
- [ ] `GATE_PASS` / `GATE_FAIL` (only after tolerances user-confirmed, else `.BLOCKED`)

**Phase B (`${DRIVE_ROOT}/<cell>_s<SEED>/` for all 5 snippets):**
- [ ] `run.log`, `vram.log`, `frame_alignment.txt`
- [ ] `est_c2w_data.txt` (12/16-float c2w, per-frame or documented stride), `sim3_metrics.txt`
- [ ] `render_metrics.txt` (the aggregator-expected name), `render_eval.csv`
- [ ] `depthL1.txt` (value **or** explicit `BLOCKED` with the escalation ref)
- [ ] `summary.txt`; `<NAME>_6panel.mp4`; `diag/`
- [ ] `payload.tgz` **containing `render_metrics.txt` + `sim3_metrics.txt`** (§7.3), `.DONE`

**Cross-run (`${DRIVE_ROOT}/`):**
- [ ] `_render_summary.csv`, `_failures.log`, `_clarifications.log`, `COMBINED_SUMMARY.txt`, `runbook.log`
- [ ] cross-method aggregation table (extended/new aggregator output)

---

## 10. Open questions to escalate (Clarification Protocol — confirm before relying)

> Items marked **HARD GATE** block the named stage and must be resolved (or the stage `.BLOCKED`) before proceeding.

1. **DINOv2 internal-vs-precomputed, variant, + 16-d projection.** Does SemGauss extract DINOv2 at train time (then the harness bake is unnecessary/double-supply) or consume precomputed `.npy`? Variant (S/B/L, patch size) and native-dim→16 mechanism (PCA vs learned vs decoder)? Read `segmentation/` + model code. **HARD GATE on `L_f` (§5.3).**
2. **Semantic class count** for Replica + ScanNet. Affects classifier output dim + mIoU class set.
3. **Mapping/tracking iters, LRs, resolution, frames-per-scene, loss weights, silhouette threshold** for Replica/ScanNet — from `configs/*.py`. **HARD GATE on Phase A (§4.2).**
4. **ATE alignment convention** in the repo eval code (rigid SE3 vs Sim3). Affects Phase A reporting + the A-vs-CRCD narrative.
5. **Table I render view type** — train/mapping vs held-out for PSNR/SSIM/LPIPS. Affects how you reproduce the render numbers.
6. **simple-knn / rasterizer submodule names** — read `.gitmodules`; required to build. **HARD GATE on build (§3.2).**
7. **CRCD semantic path: (A) GT 4-class masks ["Ours (GT)"] vs (B) retrained DINOv2 seg head ["Ours"].** Recommended (A). **HARD GATE on `L_s` (§5.3).**
8. **Depth-L1 has NO implementation and CRCD has no GT depth.** Define the reference (rendered vs MoGe input / frame-0 stereo / held-out stereo) and author the script + pairing. **HARD GATE on the fixed Depth-L1 metric (§6).**
9. **Stereo-scaling cadence:** single frame-0 anchor (repo, matches all prior results) vs ~per-100-frame (global spec). **HARD GATE — do not author CRCD configs until settled (§5.2).**
10. **Missing configs for e3_005/c3_001/g3_001** (only c1/c2 exist). Confirm episode→snippet→frame counts and **who derives/signs off `mapping.bound`** (hand-derived from frame-0 depth + GT extent).
11. **sc_factor threshold scaling:** after reading SemGauss's config, enumerate EVERY depth-threshold param that must scale by sc_factor and confirm ALL scale (the DDS "F7 bug" scaled only `trunc`), or constrain snippets to `|log(sc_factor)| ≤ 0.1` (§5.2 step 5).
12. **GPU/VRAM policy:** confirm the Phase-A 24 GB abort floor + the Phase-B 16 GB floor + the warn-vs-abort band (§3.1).
13. **Phase A GATE: exact Table I numbers from the PDF + numeric tolerances + the acceptable 3DGS run-to-run std + whether mIoU is blocking for Phase B** (§4.6, §8.1). **HARD GATE — write GATE verdict only after this (§0.2).**
14. **Deformation + tool-mask:** CRCD deforms (SemGauss assumes static rigid); does SemGauss support a per-pixel loss mask at all (§2.1)? Confirm deformation is reported as an OOD caveat and which CRCD metrics are headline-able per snippet given the c1/c2 REJECT-for-tracker verdicts (§8.6).
15. **cu116 rasterizer build feasibility on the assigned Colab runtime** (nvcc 11.6 vs CUDA-12.x host driver; torch 1.12.1+cu116 wheels still installable; prebuilt wheel/container available?). **HARD GATE — confirm runtime before Phase A (§3.2).**
16. **CRCD label space:** verify `semantic_class` values are ⊆ {0,1,2,3} on all 5 snippets; if not, add a `coco_id→{0,1,2,3}` remap; confirm mask encoding SemGauss expects (255 vs 1) and label encoding (0/255 vs 0..K index). **HARD GATE on path A + seg panel (§5.1).**
17. **Cross-method aggregator:** extend `aggregate_ab.py`'s hardcoded cell list vs write a new SemGauss-aware aggregator; confirm the full cross-method path (note `run_crcd_4snippets.sh` writes `render_eval.txt` and is itself incompatible — prior CRCD outputs may need re-emission with `render_metrics.txt` in `payload.tgz`). **HARD GATE on the final aggregation table (§7.3).**
