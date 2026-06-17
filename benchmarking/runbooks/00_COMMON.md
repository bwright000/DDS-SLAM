# 00_COMMON.md — Shared Run Book for the CRCD Semantic-SLAM Benchmark

## ⚙️ Resolved Benchmark Decisions (authoritative — override any "escalate"/"open question" below)

Made by the user on 2026-06-17. These SUPERSEDE any conflicting "Clarification Protocol" / escalation text later in this document or in any method run book.

1. **Depth-L1 = input-vs-output, per model.** Depth-L1 = mean L1 between each method's rendered/predicted depth (output) and the depth that model was GIVEN as input on that snippet (the MoGe2 + stereo-scaled map). Per-model self-consistency. Agent deliverable: author `Addons/eval/depth_l1.py` pairing rendered-depth to input-depth by frame index, masking invalid pixels, reporting mean/median L1 (mm) + a per-frame curve.

2. **Periodic (~100-frame) stereo re-scaling.** Replace the repo's single frame-0 SGBM `sc_factor` with per-segment estimation: run SGBM (existing frame-0 settings) every ~100 frames, `sc_factor_k = median(stereo_depth / moge_depth)` per segment, apply piecewise (each ~100-frame block scaled by its own factor). Keep the >10%-off-1.0 gate per segment; log the per-segment scale series as a drift diagnostic. New code the agent writes; frame-0 path in `run_crcd_4snippets.sh` is the starting point. NOTE the known bug: only `trunc` rescales with sc_factor (range_d/near/far/depth_trunc do not) — the agent must fix this when applying per-segment scales.

3. **CRCD semantics = mirror each paper's OWN mechanism** (do NOT impose a uniform source), on CRCD's 4 classes {0=bg, 1=Liver, 2=Gallbladder, 3=Tool}:
   - GT-mask methods (**SGS-SLAM**) → feed CRCD's 4-class GT masks directly.
   - DINOv2-seg-head methods (**SNI-SLAM, SemGauss-SLAM**) → retrain/fit the SAME seg-head architecture on CRCD's 4 classes (never reuse the Replica indoor-class head).
   - Predicted-label methods (**Semantic-SuPer** / DeepLabV3+) → retrain DeepLabV3+ on CRCD's 4 classes.

4. **Missing snippet configs are the AGENT's job.** Configs for `e3_005`, `c3_001`, `g3_001` do not exist. The executing agent authors them from the `c1_001`/`c2_001` `_paperfaith_lrfix` template, deriving `timesteps` (E3_005=265, C3_001=1527, G3_001=1987) and `mapping.bound` / `marching_cubes_bound` from GT extent + frame-0 depth inspection. Neither the user nor the planner pre-derives these.

---


> **Status:** authoritative shared specification. Every per-method run book
> (`10_DDS-SLAM.md`, `20_SemGauss-SLAM.md`, `30_SGS-SLAM.md`, `40_SNI-SLAM.md`)
> **imports this file** and must not contradict it. If a per-method run book and
> this file disagree, **STOP and escalate** (see §1, Clarification Protocol).
>
> **Audience:** a fresh autonomous agent that will be handed *the paper + the
> local repo later* and must execute its method end-to-end on Colab with zero
> prior context. This document is a STANDING-ORDERS contract — read it fully
> before touching anything.
>
> **Harness root:** `/content/DDS-SLAM` (the DDS-SLAM repo, staged to Colab).
> All file paths cited below are **relative to that repo root** unless prefixed
> with `/content/...` or `/content/drive/...`. All paths in agent output and
> escalations must be **absolute**.

---

## 0. Scope of this document

This file fixes everything that is **shared** across the four benchmarked
methods, so the per-method docs only have to specify what is *method-specific*.
It defines:

- §1 The Clarification Protocol (the no-assumptions escape hatch).
- §2 The per-method run-book TEMPLATE (the 11 mandatory sections every method
  doc must contain, with the expert-input checklist baked in).
- §3 Colab + Drive bootstrap, GPU detection + VRAM abort policy, env hygiene,
  the `F:/ → Drive` CRCD copy.
- §4 CRCD data-prep + the MoGe-2 + stereo-scaling depth pipeline.
- §5 The shared CRCD evaluation harness (ATE-sim3, PSNR/SSIM/LPIPS, Depth-L1)
  and the canonical 6-panel video.
- §6 The Output directory convention under `/content/drive/MyDrive/Outputs`.
- §7 The standard per-method CLI contract (`run_<method>.sh`) and the master
  orchestrator (`RUN_ALL.md`) — **both of which the agent must AUTHOR**.
- §8 Logging, determinism/seeding, checkpoint/resume conventions.
- §9 The shared open-questions register (must be reconciled before publishing).

**The benchmark in one sentence:** for each of 4 methods, (a) reproduce the
paper on the method's OWN dataset and pass a DECISION GATE, then (b) run 5 CRCD
snippets — `C_1/001`, `E_3/005`, `C_3/001`, `G_3/001`, `C_2/001` — and evaluate
all with the shared harness; **a single bash line** (§7.0) runs all 4 ×
{own-repro + 5 CRCD} with failure isolation, resumability, and a final
cross-method table.

### 0.1 What the agent must AUTHOR (these files DO NOT exist yet)

The repo ships **no** end-to-end orchestrator and **no** per-method runner whose
snippet list matches this benchmark. The agent must create them:

| File to author | Status in repo today | What it must do |
|---|---|---|
| `Addons/colab/run_<method>.sh` (one per method) | **does not exist** | implement the CLI contract in §7; snippet list **exactly** `C1_001 C2_001 E3_005 C3_001 G3_001`; seed loop; per-snippet subshell+trap (§7.3); produce the exact files the aggregator consumes (§5.4) |
| `benchmarking/runbooks/RUN_ALL.md` + its driver `Addons/colab/run_all.sh` | **does not exist** | the single bash line (§7.0): loop the 4 methods × {repro + 5 CRCD}, isolate failures per method, then run the cross-method aggregator (§5.4) |

> **`run_crcd_4snippets.sh` is a STARTING TEMPLATE ONLY — not a drop-in.** Its
> hardcoded `SNIPPETS` list (`run_crcd_4snippets.sh:41-45`) is
> `F3_007, C1_001, C2_001, F1_002` — it is **missing** `E3_005, C3_001, G3_001`
> and includes `F3_007, F1_002` which are **not** in this benchmark. When the
> agent authors `run_<method>.sh` it must **edit**: (1) the snippet list to the
> exact five above, (2) add the seed loop (§8.2 — the template has none), (3)
> change the per-snippet failure handling from SLAM-only `continue` to a full
> per-snippet subshell+trap (§7.3), and (4) produce `sim3_metrics.txt` +
> `render_metrics.txt` **inside** `payload.tgz` (§5.4) — the template does not.

---

## 1. The Clarification Protocol (no assumptions — ever)

This is a **PUBLISHED** benchmarking paper. A wrong silent assumption that ships
in a number is worse than a stalled run. Therefore:

> **RULE:** When a fact required to proceed is unknown, ambiguous, or a
> load-bearing *choice* (one that changes a reported metric, a config value, a
> data source, or a pass/fail gate), the agent **MUST STOP and escalate to the
> user**. It must **never** guess, never invent a default, never "pick the
> reasonable option and note it."

### 1.1 What counts as load-bearing (escalate)
- Any numeric config value not present in a committed config or the paper
  (e.g. a per-snippet `mapping.bound`, `timesteps`, a learning rate).
- The semantic-input source/format for a method on CRCD (see §9 Q4).
- The Depth-L1 reference target (see §9 Q1 — there is NO implementation).
- The own-dataset repro target sequences, metric, and **numeric tolerance** for
  the DECISION GATE (see §9 Q6).
- The min-VRAM floor / abort threshold for a method (see §9 Q7).
- The stereo-scaling cadence (single frame-0 anchor vs periodic, see §9 Q2).
- Whether to fix a flagged bug before publishing (see §9 Q5, sc_factor) — and
  any per-snippet `sc_factor` that fails the §4.4 band gate.
- Any dataset/weight whose **license gating** is unclear or requires a manual
  acceptance click.

### 1.2 What does NOT require escalation (proceed)
- Mechanical facts already in this file, a committed config, or the paper.
- Re-running an idempotent step after a transient failure.
- Reading files, listing dirs, running `nvidia-smi`, dry sanity checks.

### 1.3 How to escalate (the exact format)
Emit a block to stdout **and** append it to the run's `runbook.log`, then halt
the affected stage (not the whole orchestrator — see failure isolation, §7):

```
########## CLARIFICATION REQUIRED ##########
METHOD   : <method>
STAGE    : <PHASE A | PHASE B snippet C3_001 | eval | env>
BLOCKER  : <one-line statement of the missing fact / choice>
WHY LOAD-BEARING : <what reported number / gate / config this changes>
OPTIONS  : (a) ... (b) ... (c) ...   # if discrete; else "value needed: <name>"
EVIDENCE : <file:line or paper section that surfaced the ambiguity>
DEFAULT  : NONE — will not proceed without an answer.
###########################################
```

Write a sentinel `<stage_dir>/.NEEDS_CLARIFICATION` so the orchestrator records
the stage as **BLOCKED** (distinct from FAILED) and continues with other
methods/snippets, and return **exit 20** (§7.3).

---

## 2. The per-method run-book TEMPLATE (mandatory structure)

Every `NN_<method>.md` MUST be a self-contained standing-orders document with
**exactly these sections** (same numbering). A section may say "N/A — see
00_COMMON §X" but may not be omitted.

> **Section 0 — Mission & non-negotiables.** Restate: precision; no assumptions;
> point to §1 here for the Clarification Protocol. State this method's one
> non-negotiable (e.g. "do NOT headline rigid ATE on up-to-scale depth").
>
> **Section 1 — Orientation / required reading.** Exact paper sections to read
> *before acting* (method formulation, datasets, metrics, eval protocol);
> exact repo files/dirs to read first; **"read 00_COMMON.md in full."**
>
> **Section 2 — Method dossier.** ≤1-paragraph summary; method class
> (NeRF/hash-grid vs 3DGS vs other); own datasets + how to obtain (URLs +
> license gating); pretrained weights + URLs + license; the method's
> **semantic-input requirement, stated concretely** (GT N-class masks? binary
> mask? 4-class map? pretrained head on a frozen DINOv2? feature grid?) —
> then **either confirm CRCD-Published `semantic_instance` (coco_id+1) maps to
> it, giving the exact class remapping, OR escalate** (§9 Q4); and **how the
> paper REPORTS results** (dataset-wide average vs per-sequence — this sets the
> gate granularity).
>
> **Section 3 — Environment setup on Colab.** python/CUDA/torch pins; build
> steps; **the explicit `lpips` + MoGe-2 install (§3.3 — these are NOT in
> `colab_setup.sh`)**; dependency landmines; a **verification smoke test** that
> imports `lpips` and `moge` and proves the env before any long run.
>
> **Section 4 — PHASE A (reproduce on own dataset).** Obtain data+weights;
> configs; EXACT run commands; evaluate **exactly as the paper does** (their
> metrics, their eval scripts); the **numeric acceptance bar** (paper's reported
> number + tolerance) tied to the paper's reporting granularity; artifacts to
> record. **END WITH A DECISION GATE** with an agent-evaluable pass/fail
> criterion (§9 Q6) — pass ⇒ proceed to Phase B; fail ⇒ exit 10, escalate, do
> not proceed.
>
> **Section 5 — PHASE B (CRCD adaptation).** `F:/ → Drive` copy (§3.4 here);
> MoGe-2 + stereo-scaled depth (§4 here); semantic-input handling **+ extend
> preprocess to emit the method's required label if the binary tool mask +
> 4-class map are insufficient, or escalate** (§9 Q4); per-snippet config
> authoring (§4.0); exact run commands for all 5 snippets; output layout
> (§6 here).
>
> **Section 6 — CRCD evaluation.** Invoke the shared harness (§5 here) for
> ATE-sim3 / PSNR / SSIM / LPIPS / Depth-L1; render the canonical 6-panel video;
> recommended diagnostics (§2.1 expert checklist).
>
> **Section 7 — Standard CLI contract.** The `run_<method>.sh` interface the
> master orchestrator calls — args, env vars, exit codes, stage markers (§7
> here, verbatim). The agent **authors** this file (§0.1).
>
> **Section 8 — Failure modes, determinism, checkpoint/resume, logging** (§8
> here).
>
> **Section 9 — Deliverables checklist** (what must exist in Outputs when done —
> §6.3 here).
>
> **Section 10 — Open questions to escalate** (this method's slice of §9 here,
> plus any method-specific unknowns).

### 2.1 EXPERT INPUT the agent must add to EVERY method doc

These are not optional. Each method doc must concretely specify all of:

1. **Determinism / seeding.** Set `seed:` in the run config (default `0`).
   `ddsslam.py:73-78` (`seed_everything`) seeds `random`, `PYTHONHASHSEED`,
   `numpy`, `torch`, `torch.cuda`. **TF32 must be OFF** (set in
   `run_cell.sh:44`). Run **n=3 seeds** {0,1,2} for any headline number and
   report mean ± std (the aggregator, §5.4, requires `<cell>_s<seed>` dirs —
   the authored runner MUST loop seeds and name dirs accordingly, §6.1). Note:
   hash encoders / CUDA atomics are not bit-exact; std across seeds is the
   honest uncertainty — report it, don't hide it.

2. **GPU / VRAM / runtime logging + ABORT.** At stage start log `nvidia-smi
   --query-gpu=name,memory.total,memory.used,driver_version --format=csv`,
   wall-clock per stage, and peak VRAM (`torch.cuda.max_memory_allocated()`).
   Record into the stage log AND the per-snippet `summary.txt`. **Enforce the
   method's VRAM floor (§9 Q7, must be answered with a concrete number before
   Phase B): read `GPU_FLOOR_GB`, compare to `memory.total`; for the 3DGS
   methods (SemGauss/SGS) if total < floor `exit 40` (do NOT continue); for
   DDS-SLAM (hash-grid) warn-and-log the floor decision and continue.** A
   warn-only check does **not** satisfy the abort requirement (§3.3).

3. **Sim3-alignment correctness checks.** Headline trajectory metric is
   **Sim3 ATE** (`Addons/eval/sim3_ate.py`, Umeyama *with scale*). Verify:
   (a) `recovered scale s` is finite and `est/GT path ratio` ∈ a sane band —
   a path ratio ≫ 1 or ≪ 1 means the depth scale is wrong, escalate;
   (b) `|Pearson| dom` (scale-free) is reported alongside — it is immune to
   scale artefacts; (c) the printed **RIGID ATE is never headlined** (it is
   scale-confounded on MoGe depth — the script labels it "do NOT headline",
   `sim3_ate.py:111`). Sanity gate: if `len(est) != len(gt)` the script
   *resamples with a warning* — treat any resample as a **frame-alignment
   defect to investigate**, not a clean number.

4. **Frame-subsampling / index alignment** (RGB ↔ depth ↔ semantics ↔ GT).
   The loader globs each stream independently and slices the tail:
   `video_frames/*l.png[-4000:]`, `depth/*.png[-4000:]`,
   `masks/*.png[-2000:]` (`datasets/dataset.py:133-138`). **The streams a method
   consumes must be 1:1 by sorted filename index.** Apply the method-aware
   stream-alignment gate (§4.5) before any run. A mismatch is the #1
   silent-corruption source — it makes ATE pair the wrong frames.

5. **Recommended diagnostics** (produce for every CRCD snippet, alongside the
   6-panel video): trajectory overlay est-vs-GT (Sim3-aligned + raw — both
   panels are in the 6-panel video, §5.5); per-frame ATE curve and per-frame
   PSNR curve; depth-error heatmap (rendered depth vs the reference chosen in
   §9 Q1); semantic-overlay frames (the seg panel); keyframe-coverage plot
   (which frames became keyframes / map coverage). These catch failures the
   scalar means hide.

6. **Sanity gates that catch silent misconfiguration.** At minimum: §4.5
   stream-count gate (method-aware); the GT-motion sentinel (extent / path
   length / active fraction — `run_crcd_4snippets.sh:381-413`, see §4.6); the
   est-line-count ≥ frames resume check (`run_crcd_4snippets.sh:419-425`); the
   sc_factor band **hard gate** (§4.4 — STOP+escalate if out of band, do NOT
   patch-and-continue); and the min-rendered-frames eval gate (§5.2 — assert
   eval saw ≥ 100 frames, do not accept a silent WARN). Any gate failure ⇒
   escalate, do not paper over it.

---

## 3. Colab + Google Drive bootstrap

### 3.1 Mount Drive
```python
# (Colab cell, run first)
from google.colab import drive
drive.mount('/content/drive')
```
Verify: `/content/drive/MyDrive` exists. If not, **STOP** — nothing downstream
works without Drive (all outputs go there).

### 3.2 Stage the repo to /content
The repo MUST live at `/content/DDS-SLAM` (every Addons script hard-codes
`REPO=/content/DDS-SLAM`, e.g. `run_cell.sh:11`). Clone/copy it there, then:
```bash
cd /content/DDS-SLAM && git rev-parse --abbrev-ref HEAD   # confirm branch
```

### 3.3 GPU detection + VRAM abort + env build
```bash
# GPU detection helper — log it, branch on it.
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
```

**VRAM abort policy (enforced, not advisory).** Read `memory.total` (GiB) and
compare to the method's `GPU_FLOOR_GB` (§9 Q7 — must be a concrete number before
Phase B). The env phase must implement this gate:
```bash
TOTAL_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
TOTAL_GB=$(( TOTAL_GB / 1024 ))
if [ "$IS_3DGS" = "1" ] && [ "$TOTAL_GB" -lt "$GPU_FLOOR_GB" ]; then
  echo "[GPU] $TOTAL_GB GiB < floor $GPU_FLOOR_GB GiB for 3DGS method -> ABORT"
  exit 40                              # SemGauss / SGS: hard abort, do NOT continue
fi
# DDS-SLAM (hash-grid): warn-and-continue, but LOG the floor decision.
[ "$TOTAL_GB" -lt "$GPU_FLOOR_GB" ] && echo "[GPU] WARN: $TOTAL_GB GiB < floor $GPU_FLOOR_GB; continuing (hash-grid)."
```
> The canonical template's GPU logic (`run_crcd_4snippets.sh:110-114`) only
> **warns** ("non-A100 … Continuing anyway") and never aborts; it does **not**
> read a VRAM floor or `exit 40`. That warn-only check does **not** satisfy this
> requirement — the authored runner must implement the gate above.

**Build/activate the environment** with the canonical setup script:
```bash
bash Addons/env/colab_setup.sh        # build + activate + verify
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}   # tinycudann needs this on Colab
```
**What `colab_setup.sh` actually installs (verified):** torch (Colab native) +
common pips (`PyYAML scipy trimesh matplotlib opencv-contrib-python tqdm yacs
Cython ninja gdown transformers Pillow`) + **tinycudann** + **pytorch3d** +
the **marching_cubes** extension. It does **NOT** install `lpips` or `MoGe-2`
(grep confirms neither token appears in the file). Its smoke test imports only
`torch / tinycudann / pytorch3d / marching_cubes`.

**Therefore the env phase MUST also install lpips + MoGe-2 explicitly** (these
are otherwise installed only lazily inside `run_crcd_4snippets.sh` via
`ensure_lpips` (line 82-83) and `ensure_moge` (line 88-93) — so an agent that
runs only `bash run_<method>.sh env` would have NEITHER, and PSNR/LPIPS eval and
the entire depth pipeline would silently break):
```bash
pip install -q lpips
pip install -q git+https://github.com/microsoft/MoGe.git huggingface_hub
# extend the env smoke test to prove BOTH imports:
python -c "import lpips; print('lpips OK')"
python -c "import moge; print('moge OK')"
```
A failed import here ⇒ exit 30 (env/build failure).

**Env hygiene:** do NOT edit the repo model/configs to inject seed/TF32; the
canonical wrapper writes a *generated override yaml* instead (`run_cell.sh:35-46`),
keeping the committed repo pristine. Pin nothing ad-hoc — pins live in each
method's §3.

**GPU policy summary (shared default; per-method §9 Q7 sets the numbers):**
- DDS-SLAM (NeRF/hash-grid via tinycudann): runs on T4 (warn: ~16-20h wall;
  `run_crcd_4snippets.sh:112-114`). Log the floor; warn-and-continue allowed.
- **SemGauss-SLAM / SGS-SLAM (3DGS, VRAM-heavy): state min VRAM, RECOMMEND
  A100, and `exit 40` if `memory.total < GPU_FLOOR_GB`.** Do not
  warn-and-continue these.

### 3.4 The `F:/ → Drive` CRCD copy (mandatory, manual, escalate-aware)
CRCD-Published lives on the **user's local `F:/` drive**. Colab **cannot read
`F:/`**. The data must be copied to Drive *by the user* before any agent step:

1. User uploads CRCD-Published from `F:/Datasets/CRCD-Published` to
   `/content/drive/MyDrive/Datasets/CRCD-Published/` (snippets + the calib
   pickle `cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`, expected at
   `crcd_depth_gen_remainder_20260616.sh:20`).
2. Agent verifies presence:
   ```bash
   ls /content/drive/MyDrive/Datasets/CRCD-Published/ \
      /content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
   ```
3. If missing ⇒ **escalate** (the agent cannot fetch from `F:/`).

> The local working copy only has `data/v2_data` (SemSup). `data/CRCD/*` and
> `data/P2_1` are **NOT present** — this confirms the copy step is required.

---

## 4. CRCD data-prep + DEPTH pipeline (shared)

CRCD is a stereo surgical/endoscopic dataset that **ships no usable depth**.
The pipeline: rectify → MoGe-2 metric depth → stereo-anchor scale → cache.
All commands below reuse committed Addons; the canonical end-to-end driver is
`Addons/colab/run_crcd_4snippets.sh` (Phases 1.5/1.6) and the depth-only batch
is `Addons/colab/crcd_depth_gen_remainder_20260616.sh`.

### 4.0 Snippet → name → config mapping (FIXED)
| CRCD snippet | staged dir | config stem | datadir | frames |
|---|---|---|---|---|
| `C_1/001` | `C1_001` | `c1_001` | `data/CRCD/C1_001` | 360 |
| `C_2/001` | `C2_001` | `c2_001` | `data/CRCD/C2_001` | 730 |
| `E_3/005` | `E3_005` | `e3_005` | `data/CRCD/E3_005` | 265 |
| `C_3/001` | `C3_001` | `c3_001` | `data/CRCD/C3_001` | 1527 |
| `G_3/001` | `G3_001` | `g3_001` | `data/CRCD/G3_001` | 1987 |

Benchmark config variant: `<stem>_paperfaith_lrfix.yaml`
(inherits `configs/CRCD/crcd_paperfaith_lrfix.yaml` → `crcd.yaml`); output dir
`output/CRCD/<NAME>_paperfaith_lrfix`. Frame counts for E3/C3/G3 are from the
depth-gen snippet table (`crcd_depth_gen_remainder_20260616.sh:30-35`).

> **⚠ MISSING CONFIGS — escalate before Phase B (see §9 Q3).** Only
> `c1_001_paperfaith_lrfix.yaml` and `c2_001_paperfaith_lrfix.yaml` exist
> (verified: zero configs of any kind for `e3_005`, `c3_001`, `g3_001`). They
> must be authored from the c1/c2 template, which needs per-snippet `timesteps`
> (from the table) **and `mapping.bound` / `marching_cubes_bound`**.
>
> **The agent must NOT invent bounds.** Two acceptable routes — the agent must
> use exactly one and record which in `runbook.log`:
>
> - **(a) Escalate bound derivation to the user (DEFAULT).** State that the
>   user must supply the `bound`/`marching_cubes_bound` for the three new
>   snippets, and provide the c1_001 reference as the format example
>   (`c1_001_paperfaith_lrfix.yaml:18-19`):
>   `bound = [[-0.08,0.13],[-0.02,0.18],[0.68,0.90]]`,
>   `marching_cubes_bound` = identical to `bound`. Do not proceed without the
>   answer.
> - **(b) Pin the deterministic recipe (only if the user authorizes it).**
>   After depth-gen, from frame-0 valid (post-`sc_factor`) MoGe depth in
>   **metres** compute per-axis bounds in the camera frame: for the depth (Z)
>   axis `Z ∈ [P1(d) − M, P99(d) + M]`; for X,Y back-project the valid pixels
>   with the rectified intrinsics and take `[P1 − M, P99 + M]` per axis.
>   Fix `M = 0.02 m` (2 cm margin) and `P1/P99 = 1st/99th percentile`. Set
>   `marching_cubes_bound = bound`. These constants are pinned so the result is
>   reproducible; record the computed numbers in the snippet config comment and
>   `runbook.log`. **"Inspect frame-0 depth" without these exact constants is
>   forbidden** (it is the very assumption §1 prohibits).

### 4.1 Rectify / stage (preprocess) — REQUIRED args, one snippet per call
`preprocess_crcd_published.py` **requires** `--snippet_dir`, `--calib_pkl`,
`--output_dir` (`preprocess_crcd_published.py:79-81`) and processes **exactly
one** snippet per invocation — there is **no** built-in loop. The agent must
author the loop over all 5 snippets (pattern from
`crcd_depth_gen_remainder_20260616.sh:85`):
```bash
CALIB_PKL=/content/drive/MyDrive/Datasets/CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
# rows: NAME  EP   SID
for ROW in "C1_001 C_1 001" "C2_001 C_2 001" "E3_005 E_3 005" "C3_001 C_3 001" "G3_001 G_3 001"; do
  read -r NAME EP SID <<< "$ROW"
  SNIP=/content/drive/MyDrive/Datasets/CRCD-Published/${EP}/snippet_${SID}
  OUT=data/CRCD/${NAME}
  [ -f "$OUT/.PREPROCESSED" ] && continue
  python Addons/preprocess/preprocess_crcd_published.py \
    --snippet_dir "$SNIP" --calib_pkl "$CALIB_PKL" --output_dir "$OUT" \
    || { echo "preprocess FAIL $NAME" >> "$DRIVE_ROOT/_failures.log"; continue; }
  [ -f "$OUT/groundtruth.txt" ] || cp "$SNIP/groundtruth.txt" "$OUT/groundtruth.txt"
  touch "$OUT/.PREPROCESSED"
done
```
Produces per snippet `data/CRCD/<NAME>/`:
- `video_frames/NNNNNNl.png` + `NNNNNNr.png` (rectified L/R; **left is the SLAM
  input**, globbed `*l.png[-4000:]`, `dataset.py:133`)
- `masks/NNNNNN.png` (binary tool mask, 1=tool — the semantic input the loader
  reads for the edge-semantic loss, `dataset.py:136-138,192`; `--tool_pixel_value`
  defaults to 3 per `info_semantic.json`)
- `semantic_class/NNNNNN.png` (4-class index 0=bg/1=Liver/2=Gallbladder/3=Tool —
  **video panel only**, NOT read by the SLAM loader)
- `groundtruth.txt` (TUM, copied verbatim)
- `rectified_calib.txt` (fx, fy, cx, cy, baseline_m, baseline_mm, width, height —
  `preprocess_crcd_published.py:59-71`; consumed by the stereo anchor)

> **If a method needs an N-class GT semantic map** (not the binary tool mask or
> the video-only 4-class map), `preprocess_crcd_published.py` must be **extended
> to emit it** from CRCD-Published `semantic_instance` (coco_id+1) with that
> method's class remapping — or the requirement escalated (§9 Q4). Do not run
> Phase B for such a method against the binary-mask-only output.

### 4.2 MoGe-2 metric depth (the CRCD depth source)
```bash
# 1) symlink left frames as MoGe inputs (generate_depth_moge.py globs *-left.png)
#    for each video_frames/*l.png -> _moge_in/<fid>-left.png
# 2) MoGe-2 metric-direct (no --ref => writes <fid>-left_depth.npy = depth_m * depth_scale)
python Addons/depth/generate_depth_moge.py \
  --rgb _moge_in --out _moge_npy \
  --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0
# 3) npy -> uint16 PNG: cv2.imwrite('depth/<fid>.png', clip(npy,0,65535).astype(uint16))
```
Model `Ruicheng/moge-2-vitl`, `resolution_level=9`, `temporal_window=1`
(smoothing OFF for CRCD), metric-direct (`generate_depth_moge.py:221-223`).
Result: `data/CRCD/<NAME>/depth/<fid>.png` (uint16, value = depth_m × 10000).

### 4.3 Stereo-anchor scale (frame-0 SGBM)
MoGe-2 is scale-ambiguous; recover metric scale by stereo matching the rectified
L/R pair. The committed pipeline computes a **single frame-0 SGBM `sc_factor`**
per snippet (`run_crcd_4snippets.sh:265-334`):
- read `rectified_calib.txt` (`baseline_m`, `fx`);
- `cv2.StereoSGBM_create(minDisparity=0, numDisparities=128, blockSize=7,
  P1=8*49, P2=32*49, disp12MaxDiff=1, uniquenessRatio=10, speckleWindowSize=100,
  speckleRange=32, mode=MODE_SGBM_3WAY)` on frame-0 left/right;
- `depth = baseline_m * fx / disparity`; mask `0.05 < d < 3.0 m` AND MoGe-valid;
- `sc_factor = median(stereo_depth / moge_depth_m)` → write
  `data/CRCD/<NAME>/.sc_factor`.

> **⚠ CADENCE MISMATCH — escalate (see §9 Q2).** The global brief says
> "stereo-match periodically (~every 100 frames)". The committed code does a
> **single frame-0 anchor applied globally** (one scalar). There is no
> every-100-frames re-estimation in the repo. Confirm with the user whether to
> (a) keep the frame-0 anchor (matches all existing results) or (b) implement
> periodic re-scaling. **(b) is NOT a config tweak** — no such script exists; it
> requires authoring + pinning a new periodic-stereo script, re-deriving the
> per-snippet bounds (§4.0) and depth thresholds, and it materially changes
> runtime (a stereo solve every ~100 frames). **Do not silently implement (b).**

### 4.4 Apply the scale + the threshold band — HARD GATE
At load time `dataset.py:195` applies `depth = png/png_depth_scale * sc_factor`,
and `load_poses` scales GT translations by `sc_factor` (`dataset.py:252,279`) —
so est and GT live in the same scaled frame.

> **⚠ KNOWN BUG — do NOT patch-and-continue (see §9 Q5).** When `sc_factor` is
> far from 1.0, the template patches `data.sc_factor` into the config and only
> `trunc` is rescaled; `range_d / near / far / depth_trunc` are **NOT**
> (`run_crcd_4snippets.sh:367-376`), which corrupts depth gating. The template
> emits a WARN and **continues** — that violates the no-silent-assumption bar.
>
> **Required behaviour (replaces the template's WARN+continue):** after
> computing each snippet's `sc_factor`, if `abs(log(sc_factor)) > 0.1` (i.e.
> >~10% off 1.0) then **STOP and escalate** (§1) — do not patch, do not run.
> The escalation must offer: (a) fix the bug to also rescale
> `range_d/near/far/depth_trunc`, or (b) confirm this snippet is excluded.
> `c1_001`/`c2_001` are known within-band; the three new snippets' `sc_factor`
> is unknown until depth-gen runs, so this gate is the only thing preventing a
> silently mis-thresholded run on E3_005/C3_001/G3_001.

### 4.4b Depth cache (resumability)
Depth PNGs + `.sc_factor` cache to Drive so future runs rehydrate:
`/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth/*.png,.sc_factor}`
(`crcd_depth_gen_remainder_20260616.sh:21,147-153`).

### 4.5 Stream-alignment gate (MANDATORY before any SLAM run — METHOD-AWARE)
Assert 1:1 alignment **only over the streams the running method actually
consumes**. RGB + depth + GT are required for all four methods; the binary tool
mask (`masks/*.png`) is consumed **only by DDS-SLAM** (`dataset.py:136-138`).
A method that does not read `masks/` must NOT be gated on it (else the gate
fails spuriously when `masks/` is absent, or passes vacuously). The method doc
declares `USES_MASK=0|1` (and, if it reads an N-class GT map, gates that
stream too).
```bash
N_RGB=$(ls data/CRCD/<NAME>/video_frames/*l.png 2>/dev/null | wc -l)
N_DEP=$(ls data/CRCD/<NAME>/depth/*.png        2>/dev/null | wc -l)
N_GT=$(grep -vc '^#' data/CRCD/<NAME>/groundtruth.txt)
# ALWAYS REQUIRE: N_RGB == N_DEP ; timesteps <= N_RGB ; N_GT == N_RGB (GT[i]<->frame i)
if [ "$USES_MASK" = "1" ]; then
  N_MSK=$(ls data/CRCD/<NAME>/masks/*.png 2>/dev/null | wc -l)
  # REQUIRE additionally: N_MSK == N_RGB
fi
```
Any mismatch on a consumed stream ⇒ **escalate** (silent ATE corruption
otherwise — see §2.1 item 4).

### 4.6 GT-motion sanity sentinel
Run the GT motion profile (`run_crcd_4snippets.sh:381-413`): trajectory extent,
path length, per-frame motion, active fraction, sub-SNR sentinels. Several CRCD
snippets are **sub-SNR / high-static** (e.g. C1_001: 18.55 mm GT path, 41.8%
active, median 0.115 mm/frame ≈ 8.7× sub-SNR — flagged REJECT for tracker
benchmarking, keep for render/diagnostic). Record the verdict in `summary.txt`;
a sub-SNR snippet's ATE is reported but **not headlined as tracking quality**.

### 4.7 Depth-source reference table (do not cross the wires)
| Script | Use | NOT for |
|---|---|---|
| `generate_depth_moge.py` | **CRCD** metric depth (this pipeline) | — |
| `generate_depth_stereo.py` | StereoMIS own-dataset (RAFT-stereo, `poseNet_2xf8up4b.pth`) | CRCD |
| `generate_depth_for_ddsslam.py` | SemSup/Super (Monodepth2 + median-scale-match) | CRCD |

---

## 5. Shared CRCD evaluation harness

All metrics are computed **exactly** as in `run_crcd_4snippets.sh` Phases 5-6
and `run_cell.sh` step 3 — same Addons scripts. `<OUT>` =
`output/CRCD/<NAME>_paperfaith_lrfix`; `<RUN>` = `<OUT>/demo`;
`<NAME>` ∈ {C1_001, C2_001, E3_005, C3_001, G3_001}.

> **STRICT STAGE ORDERING (load-bearing — §5.7).** Run `eval_rendering.py` and
> `sim3_ate.py` on the **live `<OUT>` dir BEFORE** moving any top-level `*.jpg`
> into `renders_rgb/` for tarring. `eval_rendering.py --render_dir <OUT>` globs
> `[0-9]*.jpg` at the OUT root; if renders are moved first it finds **zero
> frames and only WARNs** (no PSNR). The producer must: render → eval (both
> scripts, writing the files below into `<OUT>`) → assert frame count
> (§5.2) → then move jpgs + tar.

### 5.1 ATE (Sim3) — the headline trajectory metric — write to `sim3_metrics.txt`
```bash
python Addons/eval/sim3_ate.py \
  --est <RUN>/est_c2w_data.txt \
  --gt  data/CRCD/<NAME>/groundtruth.txt \
  --name "<NAME>" \
  --out <OUT>/sim3_metrics.txt          # REQUIRED: aggregator reads this filename from payload.tgz
```
- est: `est_c2w_data.txt`, 12 or 16 floats/line, translation cols [3,7,11]
  (`load_est`, `sim3_ate.py:31-39`).
- gt: TUM, cols [1,2,3] (`load_gt_tum:42-49`).
- `--out` **appends** the report to the named file (`sim3_ate.py:99,117`). It
  MUST be `<OUT>/sim3_metrics.txt` so the aggregator (§5.4) finds it.
- Umeyama **with scale** ⇒ Sim3 ATE rmse/mean/median/max (mm), recovered scale
  `s`, est/GT path ratio, `|Pearson| dom` (scale-free). Also prints the RIGID
  ATE explicitly labelled **"do NOT headline."**
- Pairing: 1:1 when `len(est)==len(gt)`; else uniform resample **with a
  warning** (treat any resample as a defect to investigate, §2.1 item 3).

### 5.2 PSNR / SSIM / LPIPS — render quality — capture stdout to `render_metrics.txt`
```bash
python Addons/eval/eval_rendering.py \
  --gt_dir     data/CRCD/<NAME>/video_frames \
  --render_dir <OUT> \
  --name "<NAME>" \
  --output_csv  <OUT>/render_eval.csv \
  --summary_csv <DRIVE_ROOT>/_render_summary.csv \
  --sequence "CRCD (<NAME>)" \
  2>&1 | tee <OUT>/render_metrics.txt    # REQUIRED: aggregator parses PSNR/SSIM/LPIPS/Rendered from this
```
- Renders: `<OUT>/[0-9]*.jpg` (or `.png`). CRCD GT pairing = Mode 2, separate
  dir by filename index against `video_frames/*l.png`.
- PSNR (`compute_psnr`), SSIM (11×11 Gaussian), LPIPS (`lpips` alex net — must
  already be installed, §3.3).
- The script prints `Rendered: <N> images` (`eval_rendering.py:141`) and the
  PSNR/SSIM/LPIPS means; `aggregate_ab.py` parses these tokens out of
  `render_metrics.txt` — so **stdout must be tee'd to `<OUT>/render_metrics.txt`**.
- **Min-frame gate (not a silent WARN):** if `Rendered < 100`, treat as FAIL for
  this snippet and append to `_failures.log` — mirror the aggregator's own
  exclusion (`aggregate_ab.py:100-104`). `eval_rendering.py:91-92` only prints
  "No rendered images found" and exits 0, so the runner must assert the count.
- CRCD sequences have **None** paper-reference (`eval_rendering.py:55-60` — CRCD
  post-dates the paper; all five CRCD entries map to `None`), so the gate is
  *relative across methods*, not vs a paper baseline.
- Output: per-frame CSV + one summary row appended to `--summary_csv`
  (`method,sequence,n_frames,psnr_mean/std,ssim_mean/std,lpips_mean/std`).

### 5.3 Depth-L1 — **NO IMPLEMENTATION EXISTS (escalate, §9 Q1)**
The fixed CRCD metric set requires Depth-L1, but **the repo has no Depth-L1
script** (verified). `compute_rep_err.py` computes pixel **reprojection** error
against Semantic-SuPer point-track GT (SemSup-only), not depth-map L1. CRCD
ships no GT depth (MoGe-2 is generated). **The agent must STOP and escalate** to
obtain: (a) the Depth-L1 reference — rendered SLAM depth vs the MoGe-2 input
depth (self-consistency)? vs frame-0 stereo-SGBM depth? vs a held-out stereo
depth? — and (b) the exact pairing. **Do not author a Depth-L1 number from a
self-chosen reference.**
- Until resolved, the orchestrator emits the Depth-L1 cell as the literal string
  `BLOCKED` in the final table — never silently empty.
- Depth-L1 being BLOCKED must **not** block the other four metrics: the table
  renderer (§5.4) tolerates a `BLOCKED` cell and still prints ATE/PSNR/SSIM/LPIPS.
  Confirm this when authoring the renderer.

### 5.4 Cross-method / cross-seed aggregation — producer/consumer CONTRACT
The current `aggregate_ab.py` is **hardcoded** to specific cells
(`c1_001_canon_base/_uncert/_uncert_dino` and SemSup `trail3_*`,
`aggregate_ab.py:60-99`) and does **not** iterate the 5 snippets or 4 methods.
It is therefore **not usable as-is** for the final table. The agent must
reconcile producer and consumer; pick **exactly one** route and pin it:

**Route 1 (recommended) — generalize the aggregator.** Rewrite/extend
`aggregate_ab.py` (or author `aggregate_bench.py`) to take the
method × snippet × seed matrix as input and iterate it, reusing the existing
parsers it already has: `sim3_vals()` (reads `sim3_metrics.txt`,
`aggregate_ab.py:40-49`) and `num()` over `render_metrics.txt` for
`PSNR/SSIM/LPIPS/Rendered`. It reads those two filenames from inside each
`<cell>_s<seed>/payload.tgz` and applies the `<100`-frame exclusion
(`aggregate_ab.py:100-104`). Keep `BLOCKED` as the Depth-L1 cell.

**Route 2 — make the producer match the existing consumer's globs.** Have the
authored `run_<method>.sh`: (1) **loop `SEEDS`**; (2) name each output dir
`<NAME>_s<SEED>` (the consumer globs `<cell>_s*`, `aggregate_ab.py:51-52` — the
template instead names dirs `$DRIVE_ROOT/$NAME` with **no seed**, line 130, which
the glob never matches); (3) write `sim3_metrics.txt` (§5.1) and
`render_metrics.txt` (§5.2) into `<OUT>`; (4) tar **both** files into
`payload.tgz` (the template ships only `demo/+ckpts/+depth/+renders_rgb/`,
lines 535-547 — neither metrics file — so the consumer finds nothing). Then a
thin wrapper iterates the matrix.

Whichever route: **the filenames `sim3_metrics.txt` + `render_metrics.txt`, the
dir convention `<NAME>_s<SEED>`, and "both files are inside `payload.tgz`" are
the contract.** Producer and consumer MUST agree on all three before any
aggregate number is reported. Invocation after reconciliation:
```bash
python Addons/eval/aggregate_ab.py /content/drive/MyDrive/Outputs/<DRIVE_ROOT>   # Route 1: generalized
```

### 5.5 The canonical 6-panel video (always produced) — use the CRCD branch
```bash
python Addons/viz/generate_video.py \
  --rgb_input_dir  data/CRCD/<NAME>/video_frames --rgb_input_pattern '*l.png' \
  --rgb_output_dir <OUT> --rgb_output_pattern '[0-9]*.jpg' \
  --depth_input_dir data/CRCD/<NAME>/depth --depth_output_dir <OUT>/depth --depth_norm robust \
  --seg_dir data/CRCD/<NAME>/semantic_class --seg_pattern '*.png' --skip_raw_seg --seg_classmap \
  [--uncert_dir <OUT>/uncert] \
  --trajectory_est <RUN>/est_c2w_data.txt --trajectory_gt data/CRCD/<NAME>/groundtruth.txt --trajectory_raw \
  --output <DST>/<NAME>_6panel.mp4 --fps 15
```
> **Copy the CRCD branch of `run_cell.sh`, NOT the SemSup branch.** This
> benchmark runs CRCD only. The CRCD branch (`run_cell.sh:60-67`) reads depth
> from `data/CRCD/<NAME>/depth` and **omits** `--skip_horn_traj` (so the
> Sim3-aligned trajectory panel is on). The SemSup branch (`run_cell.sh:68-75`)
> is different in two ways that would corrupt a CRCD run if copied: it reads
> depth from `$DDIR/depth/moge2` (different subdir) and it **passes**
> `--skip_horn_traj` (line 65) — do NOT cross these wires.

Auto 2×3 layout (`generate_video.py:481-489`). The 6 CRCD panels
(`run_cell.sh:60-67`): (1) Input RGB, (2) Rendered RGB, (3) Input Depth
(robust norm), (4) Output Depth (robust p2–p98 colormap), (5) Seg Overlay
(4-class palette composited on rendered RGB; raw-seg panel suppressed), (6)
Trajectory Raw (no alignment). On the CRCD branch the **Sim3-aligned
trajectory** panel is on by default (alignment = same Umeyama-with-scale as
`sim3_ate.py`, `generate_video.py:140-154`). If the run wrote `<OUT>/uncert`,
add `--uncert_dir` for an inferno Uncertainty panel.

### 5.6 The single-run shortcut (every manual run)
For ANY manual/single run, use the canonical wrapper — it trains (TF32-off +
seed via generated override, no repo edit), then **always** produces the
6-panel video + render metrics + (CRCD) Sim3 ATE + ships to Drive:
```bash
bash Addons/colab/run_cell.sh configs/CRCD/<stem>_paperfaith_lrfix.yaml <NAME>_v0 [seed]
```
It auto-detects CRCD vs SemSup vs StereoMIS from the datadir
(`run_cell.sh:16-30`). A bare `python ddsslam.py` produces *neither metrics nor
video* — never run bare in the benchmark.

### 5.7 DDS-SLAM launch + output paths (reference; ordering is load-bearing)
```bash
python ddsslam.py --config configs/CRCD/<stem>_paperfaith_lrfix.yaml
# args: --config (required), --input_folder, --output (optional cfg override) — ddsslam.py:936-940
```
Writes (`ddsslam.py`): `<RUN>/est_c2w_data.txt` (trajectory, :813),
`<RUN>/config.json`, a copy of `ddsslam.py` (:952-955), `<RUN>/output.txt`
(pipeline rigid-ATE), `<RUN>/checkpoint{i}.pt` (:804);
`<OUT>/<frame:04d>.jpg` (rendered RGB **at the OUT ROOT**, :890, gated by
`render_freq`=1 in paperfaith_lrfix); `<OUT>/depth/<frame:04d>.png` (rendered
depth uint16, value = depth_m × `output_depth_scale`=10000, :905-912);
`<OUT>/uncert/<frame:04d>.png` (σ², only if uncertainty head on, :921-927).

> Because renders land at the OUT root as `<frame>.jpg`, and a validated bug
> (`run_crcd_4snippets.sh:535-543`) once produced a `payload.tgz` with 0
> renders, the runner moves top-level `*.jpg` into `renders_rgb/` before tar.
> That move MUST happen **after** §5.1/§5.2 eval (which expect renders at the
> root), per the strict ordering note at the top of §5.

---

## 6. Output directory convention (under Drive)

### 6.1 Roots
- Batch root: `DRIVE_ROOT=/content/drive/MyDrive/Outputs/<method>_crcd_<DATE>`
  (e.g. `.../dds_crcd_20260617`).
- **Per snippet × seed:** `DRIVE_DST=$DRIVE_ROOT/<NAME>_s<SEED>` — the `_s<seed>`
  suffix is **mandatory** (the aggregator globs `<NAME>_s*`, §5.4). The runner
  MUST loop `SEEDS` and name dirs this way; the template's seedless
  `$DRIVE_ROOT/$NAME` (`run_crcd_4snippets.sh:130`) is **wrong** for aggregation.
- Single manual runs: `/content/drive/MyDrive/Outputs/manual_cells/<NAME>/`.
- All Outputs live under `/content/drive/MyDrive/Outputs` — nothing
  headline-bearing stays only on ephemeral `/content`.

### 6.2 Per-snippet-seed `DRIVE_DST` contents
`render_eval.csv`, `render_metrics.txt` (eval stdout — §5.2), `sim3_metrics.txt`
(§5.1), `summary.txt` (extended trajectory summary + GT motion verdict +
GPU/VRAM/runtime), the 6-panel `<NAME>_6panel.mp4`, `payload.tgz` (tars `demo/`
+ ckpts + `depth/` + `renders_rgb/` **+ `sim3_metrics.txt` + `render_metrics.txt`**
— the last two are required by §5.4 and are NOT in the template's tar), and a
**`.DONE`** sentinel.

### 6.3 Cross-run aggregates at `DRIVE_ROOT`
`_render_summary.csv`, `_failures.log`, `COMBINED_SUMMARY.txt`, `runbook.log`.
The cross-seed dirs `<NAME>_s<seed>` (each containing `payload.tgz`) are what
`aggregate_ab.py` consumes.

**Deliverables checklist (per method, §9 of each method doc):** Phase-A repro
log + gate verdict; per-snippet-seed `DRIVE_DST` with all of §6.2 for all 5
snippets × {0,1,2}; the cross-method aggregate row in `COMBINED_SUMMARY.txt` /
`_render_summary.csv`; any `.NEEDS_CLARIFICATION` sentinels resolved or escalated.

---

## 7. Orchestration: the single bash line + the per-method CLI contract

### 7.0 The END-GOAL single command (the agent AUTHORS `run_all.sh` + `RUN_ALL.md`)
There is **no** orchestrator in the repo today (§0.1). The agent must author
`Addons/colab/run_all.sh` (documented by `benchmarking/runbooks/RUN_ALL.md`) so
that the entire benchmark runs from **one line**:
```bash
bash Addons/colab/run_all.sh 2>&1 | tee /content/drive/MyDrive/Outputs/RUN_ALL_$(date +%Y%m%d).log
```
`run_all.sh` MUST:
- iterate the 4 methods `dds semgauss sgs sni`, and for each call its authored
  `run_<method>.sh all` (which does `env → repro → gate → crcd(×5 snippets ×
  SEEDS) → eval`);
- **isolate failures per method**: wrap each method call so a nonzero exit
  (10/20/30/40/1) is recorded to `$DRIVE_ROOT/_failures.log` and the loop
  **continues** to the next method;
- after all methods, run the cross-method aggregator (§5.4) and write
  `COMBINED_SUMMARY.txt` with the final table (ATE-sim3 / PSNR / SSIM / LPIPS /
  Depth-L1[=BLOCKED until §9 Q1]) × {4 methods × 5 snippets}, mean±std over seeds.

### 7.1 Per-method invocation
```
bash run_<method>.sh <phase> [snippet]
```
- `<phase>` ∈ `env | repro | crcd | eval | all`
  - `env`   build/verify the method's Colab env (§3) — **includes the lpips +
            MoGe-2 install and the VRAM-floor abort (§3.3)**.
  - `repro` Phase A own-dataset reproduction + DECISION GATE.
  - `crcd`  Phase B: stage+depth+SLAM for `[snippet]` (or all 5 if omitted),
            looping `SEEDS`.
  - `eval`  run the shared harness (§5) for `[snippet]` (or all 5), all seeds.
  - `all`   `env → repro → (gate) → crcd → eval` for all 5 snippets × seeds.
- `[snippet]` ∈ the exact 5 NAMEs `C1_001 C2_001 E3_005 C3_001 G3_001`
  (omit for "all snippets"). The hardcoded `F3_007`/`F1_002` of the template are
  NOT part of this benchmark.

### 7.2 Environment variables (read, with defaults)
| Var | Default | Meaning |
|---|---|---|
| `DRIVE_ROOT` | `/content/drive/MyDrive/Outputs/<method>_crcd_<DATE>` | batch output root |
| `SEEDS` | `0 1 2` | space-sep seed list (headline = 3 seeds) |
| `GPU_FLOOR_GB` | per-method (§9 Q7, concrete number required) | `exit 40` if `memory.total < this` for 3DGS methods; warn-only for DDS |
| `RESUME` | `1` | honor `.DONE`/`.STAGED`/`.PREPROCESSED`/`.sc_factor` sentinels |
| `SKIP_GATE` | `0` | **never set 1 for a published run** (gate is mandatory — and its numeric threshold must be defined per §9 Q6 before it can be enforced) |

### 7.3 Exit codes
| Code | Meaning | Orchestrator action |
|---|---|---|
| `0` | stage succeeded | continue |
| `10` | DECISION GATE failed (Phase A) | skip this method's CRCD; record; continue others |
| `20` | clarification required (`.NEEDS_CLARIFICATION` written) | mark BLOCKED; continue others |
| `30` | env/build failure (incl. lpips/moge import fail) | mark FAILED; continue others |
| `40` | GPU below floor (abort, 3DGS) | mark SKIPPED-VRAM; continue others |
| `1`  | any other runtime failure | mark FAILED; continue others |

**Failure isolation is non-negotiable, AND the template does NOT provide it.**
`run_crcd_4snippets.sh` uses `set -euo pipefail` (line 24) and isolates **only**
the SLAM-crash case with `if ! python ddsslam.py …; then continue`
(lines 428-432). Every other per-snippet phase (staging rsync, MoGe depth gen,
stereo anchor, eval_rendering, sim3, tar) is **un-trapped** — under `set -e` any
of them failing aborts the **entire batch**, killing all remaining snippets.

The authored `run_<method>.sh` MUST instead run **each snippet (and each seed)
in its own subshell with an ERR trap**, so ANY phase failure isolates to that
snippet and the loop continues:
```bash
run_one_snippet() {   # body NOT under set -e; returns nonzero on any failure
  local NAME=$1 SEED=$2
  set +e
  ( set -E
    trap 'echo "[TRAP] $NAME s$SEED failed at line $LINENO" \
          | tee -a "$DRIVE_ROOT/_failures.log"; exit 1' ERR
    # ... stage / depth / sc_factor gate / SLAM / eval / sim3 / video / tar ...
  )
  local rc=$?
  set -e
  return $rc
}

for NAME in C1_001 C2_001 E3_005 C3_001 G3_001; do
  for SEED in $SEEDS; do
    run_one_snippet "$NAME" "$SEED" || {
      echo "FAILED $NAME s$SEED (rc=$?)" >> "$DRIVE_ROOT/_failures.log"
      continue                       # isolate: never abort the rest
    }
  done
done
```
A nonzero exit from one method/snippet/seed MUST NOT abort the rest.

### 7.4 Stage markers (stdout + `runbook.log`)
Emit machine-greppable markers around every stage so the orchestrator can parse
progress and the aggregator can scope:
```
[STAGE_BEGIN] method=<m> phase=<p> snippet=<NAME> seed=<s> ts=<iso>
... stage output ...
[STAGE_END]   method=<m> phase=<p> snippet=<NAME> seed=<s> status=<OK|FAIL|BLOCKED|SKIP> dt=<s> peak_vram_gb=<x>
```
On success write the stage sentinel (`.STAGED`/`.PREPROCESSED`/`.DONE`); on
clarification write `.NEEDS_CLARIFICATION` (+ exit 20).

---

## 8. Logging, determinism, checkpoint/resume

### 8.1 Logging
- Every stage writes to both stdout and `$DRIVE_ROOT/runbook.log` (use
  `exec > >(tee -a "$LOG") 2>&1` as in `crcd_depth_gen_remainder_20260616.sh:23`).
- Prefix lines with `[HH:MM:SS]`. Failures additionally append to
  `$DRIVE_ROOT/_failures.log` (method, snippet, seed, stage, last 20 lines).
- Always log GPU name/total/used VRAM, driver, per-stage wall time, and
  `torch.cuda.max_memory_allocated()` peak.

### 8.2 Determinism / seeding
- Set `seed:` in the run config; `ddsslam.py:73-78` seeds `random`,
  `PYTHONHASHSEED`, `numpy`, `torch`, `torch.cuda` (it was a no-op until the
  2026-06-14 fix — the call is now wired, `ddsslam.py:60`).
- **TF32 OFF** (`run_cell.sh:44`) — never rely on TF32 for reported numbers.
- Headline numbers = n=3 seeds {0,1,2}; report mean ± std via the aggregator
  (§5.4), which **requires** the `<NAME>_s<seed>` dirs (§6.1). The template has
  **no seed loop** — the authored runner must add one (§7.3). Hash-grid/CUDA
  atomics are not bit-exact — std is the honest uncertainty; report it.

### 8.3 Checkpoint / resume
Resumability is per-phase via sentinels and counters:
- `.STAGED` (rectify/stage done), `.PREPROCESSED` / `.sc_factor` (depth+scale
  done), `.DONE` (snippet-seed shipped).
- SLAM-completion check: `est_c2w_data.txt` line-count ≥ `timesteps`
  (`run_crcd_4snippets.sh:419-425`).
- Checkpoints `<RUN>/checkpoint{i}.pt` (`ddsslam.py:804`).
- With `RESUME=1` (default) a stage with its sentinel present is skipped. A
  partial SLAM run (line-count < frames) is **not** treated as done — it reruns.

---

## 9. Shared open-questions register (reconcile before publishing)

These are blocking ambiguities surfaced by the harness audit. Each is
**load-bearing** and must be escalated (§1) before the affected number is
published. Per-method docs inherit the relevant subset in their Section 10.

- **Q1 — Depth-L1 has no implementation.** Required metric, no script; CRCD has
  no GT depth. Escalate: (a) the Depth-L1 reference (rendered vs MoGe input
  self-consistency / vs frame-0 SGBM / vs held-out stereo); (b) author + pin the
  exact pairing. Until resolved, Depth-L1 cells = literal `BLOCKED`, and the
  table renderer must still print the other four metrics (§5.3).
- **Q2 — Stereo-scaling cadence mismatch.** Brief says ~every-100-frames; repo
  does a single frame-0 anchor. Confirm keep-frame-0 vs implement-periodic.
  Periodic ⇒ new script + re-derived bounds/thresholds + runtime hit, NOT a
  config tweak (§4.3).
- **Q3 — Missing configs** for `e3_005`, `c3_001`, `g3_001` (no config at all).
  Must author from c1/c2 template; needs per-snippet `timesteps` (265/1527/1987)
  + **bounds via §4.0 route (a) escalate or (b) the pinned recipe**. Who derives
  bounds; confirm frame counts (§4.0).
- **Q4 — Semantic-input ambiguity** (per method). For DDS-SLAM the loader reads
  the **binary tool mask** (`masks/*.png`, `dataset.py:136-138`) for the
  edge-semantic loss — NOT the 4-class `semantic_class` map (video only). The
  other three methods need different inputs (GT N-class masks vs
  pretrained-head-on-DINOv2 vs DINO-feature uncertainty, cf.
  `c1_001_canon_uncert_dino.yaml`). Per method, the dossier (§2) must state the
  exact required input and **either confirm CRCD-Published `semantic_instance`
  (coco_id+1) maps to it (with class remapping) or escalate**; if an N-class GT
  map is needed, extend `preprocess_crcd_published.py` to emit it (§4.1).
- **Q5 — sc_factor threshold bug.** Far-from-1 sc_factor rescales only `trunc`,
  not `range_d/near/far/depth_trunc` (`run_crcd_4snippets.sh:367-376`). The
  runner must **hard-gate** (`abs(log(sc))>0.1` ⇒ STOP+escalate, §4.4), not
  patch-and-continue. Fix the bug or exclude the offending snippet.
- **Q6 — Paper-repro target + GATE for each method.** `eval_rendering.py`
  hardcodes Super/SemSup (trail3/4/8/9) + StereoMIS P2_1 PSNR/SSIM/LPIPS paper
  baselines (`eval_rendering.py:47-51`) but **no ATE baselines and no
  tolerance**, and CRCD has none. Per method pin: the exact repro
  dataset+sequences, which metric(s) gate, the paper's reported numbers, and a
  **numeric tolerance** (e.g. PSNR within X dB, ATE within Y%). The mandatory
  gate (exit 10) cannot be enforced while its threshold is undefined — escalate
  if unknown (Phase A).
- **Q7 — GPU/VRAM policy per method.** DDS-SLAM (hash-grid) fits T4
  (warn-and-continue, ~16-20h, `run_crcd_4snippets.sh:112-114`); SemGauss/SGS
  (3DGS) are VRAM-heavy — state min VRAM, recommend A100, **`exit 40` below
  floor** (§3.3). Escalate the exact `GPU_FLOOR_GB` per method **before Phase B**
  — without a concrete number the abort gate cannot run.

---

### Appendix A — Reusable Addons quick index
| Path | Role |
|---|---|
| `Addons/env/colab_setup.sh` | Colab env build/activate/verify — torch/tinycudann/pytorch3d/marching_cubes ONLY (NO lpips, NO MoGe-2 — install those per §3.3) |
| `Addons/preprocess/preprocess_crcd_published.py` | Rectify CRCD-Published → on-disk layout, ONE snippet/call, required args (§4.1) |
| `Addons/depth/generate_depth_moge.py` | **CRCD** MoGe-2 metric depth (§4.2) |
| `Addons/depth/generate_depth_stereo.py` | StereoMIS RAFT-stereo depth (own-dataset; NOT CRCD) |
| `Addons/depth/generate_depth_for_ddsslam.py` | SemSup/Super Monodepth2 median-scale-match (NOT CRCD) |
| `Addons/colab/crcd_depth_gen_remainder_20260616.sh` | Depth-only batch (incl. E3_005/C3_001/G3_001); preprocess-loop example at :85 |
| `Addons/eval/sim3_ate.py` | **Headline** Sim3 ATE; `--out sim3_metrics.txt` (§5.1) |
| `Addons/eval/eval_rendering.py` | PSNR/SSIM/LPIPS; tee stdout → `render_metrics.txt` (§5.2) |
| `Addons/eval/compute_rep_err.py` | Reprojection error (SemSup-only; NOT Depth-L1) |
| `Addons/eval/aggregate_ab.py` | Cross-seed aggregator — **hardcoded to c1_001/SemSup cells; must be generalized for the bench matrix** (§5.4) |
| `Addons/eval/kitti_to_tum.py` | Trajectory format converter (KITTI 3×4 → TUM) |
| `Addons/viz/generate_video.py` | 6-panel video generator (§5.5) |
| `Addons/viz/generate_6panel_sweep.sh` | Batch 6-panel driver (Super template; mirror for CRCD) |
| `Addons/colab/run_cell.sh` | **Canonical single-run** wrapper — copy the CRCD branch, not SemSup (§5.5/§5.6) |
| `Addons/colab/run_crcd_4snippets.sh` | **STARTING TEMPLATE ONLY** for the authored `run_<method>.sh` — wrong snippet list, no seed loop, no per-snippet trap, metrics not in payload (§0.1, §7.3) |
| `Addons/dino/generate_dino_features.py` | DINOv2/v3 feature precompute (uncertainty.mode:dino variant only) |
| **`Addons/colab/run_<method>.sh`** (×4) | **TO AUTHOR** — per-method runner, exact 5-snippet list, seed loop, per-snippet trap, payload contract (§0.1, §7) |
| **`Addons/colab/run_all.sh` + `benchmarking/runbooks/RUN_ALL.md`** | **TO AUTHOR** — the single-line master orchestrator + cross-method table (§0.1, §7.0) |
