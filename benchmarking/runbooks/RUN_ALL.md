> **Resolved decisions apply.** See *Resolved Benchmark Decisions* in [00_COMMON.md](00_COMMON.md) — they override any “escalate”/“open question” below: Depth-L1 = input-vs-output per model; periodic ~100f stereo rescaling; per-paper semantic mirroring on CRCD 4 classes; missing configs (e3_005/c3_001/g3_001) are the agent's job.

# RUN_ALL.md — Master Orchestration Run Book

> **Scope.** Top-level run book for the 4-method semantic-SLAM benchmark on the *published-paper* methodology. It defines the single entrypoint `bench.sh`, the run matrix, robust failure isolation, resumability, GPU/VRAM preflight, the F:/→Drive→/content data pipeline, and the final cross-method aggregation.
>
> **CURRENT EXECUTABLE SCOPE (read this first).** As of 2026-06-17 only **DDS-SLAM** is runnable end-to-end from this tree. The other three repos (`SemanticSuPer`, `SNI-SLAM`, `SemGauss-SLAM`, `SGS-SLAM`) are **not synced, have no `run_<method>.sh` wrapper, no env build, and no repro dataset staged** — so "one bash line runs all 4 methods" is **not true today** (critic gap 10). `bench.sh` therefore defaults to `METHODS=ddsslam`. The full 4-method matrix is **descoped** until §11's onboarding checklist is completed for each repo; running them today produces immediate `FAIL(no_wrapper)` and is intentionally gated off by default.
>
> Even within DDS-SLAM, **3 of the 5 CRCD snippets cannot run today** because their configs do not exist (critic gap 3, §9 item 3) — this is a hard blocker, not a soft clarification.
>
> **It does not redefine per-method internals.** Each method has a wrapper `run_<method>.sh` conforming to the *Common CLI Contract* (§3). `bench.sh` only orchestrates wrappers.
>
> **No-assumptions rule.** Wherever a fact is unknown or a choice is load-bearing, `bench.sh` must **STOP and escalate** (Clarification Protocol, §9). It never silently guesses. Some items below are *code-level decisions the agent must make and record* (not user clarifications) — those are marked **[CODE]**; the rest are **[USER]** clarifications.

---

## 1. The single command

After §2 setup (Drive mounted, DDS-SLAM synced, raw CRCD copied F:/→Drive, depth cache warmed):

```bash
bash /content/bench/bench.sh 2>&1 | tee -a /content/drive/MyDrive/Outputs/bench_master_$(date +%Y%m%d).log
```

Defaults run **only DDS-SLAM** (the sole executable method today). Everything else (env, staging, depth/semantic prep, training, eval, video, aggregation) is driven inside `bench.sh`.

Run-control overrides (env):

```bash
# DDS-SLAM CRCD legs only (the runnable subset):
METHODS=ddsslam STAGES=crcd bash /content/bench/bench.sh

# Force re-run of one leg (clears its out_dir first, see §6):
FORCE=1 METHODS=ddsslam SNIPPETS=c1_001 bash /content/bench/bench.sh

# Preflight only — print matrix + GPU report + clarification gate, run nothing:
DRYRUN=1 bash /content/bench/bench.sh

# Opt IN to the not-yet-onboarded methods (will FAIL(no_wrapper) until §11 done):
METHODS="ddsslam semsuper snislam semgauss sgsslam" bash /content/bench/bench.sh
```

---

## 2. One-time setup (ordered; do BEFORE the single command)

Ordering is load-bearing. Colab **cannot read F:/** (a path on the user's local Windows machine), so the raw dataset must reach Drive first, and depth/semantic prep must precede SLAM.

| # | Step | Why it is first | Owner |
|---|------|-----------------|-------|
| 0 | Mount Drive (`drive.mount('/content/drive')`). | All outputs + staged data live on Drive. | user (Colab cell) |
| 1 | **Copy CRCD `F:/` → Drive** at `/content/drive/MyDrive/Datasets/CRCD-Published/` (raw per-snippet dirs + `cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`). Prefer per-snippet `*_staging.tar`. | Colab can't see `F:/`; manual local→Drive upload. `bench.sh` aborts if calib pickle or a needed snippet raw is missing. | **user (local machine)** |
| 2 | Sync DDS-SLAM under `/content/DDS-SLAM/` (this repo). *(Other repos: see §11 — not required for the default run.)* | `bench.sh` dispatches to `$REPO/run_ddsslam.sh`. | user (git clone) |
| 3 | Build DDS-SLAM env once: `bash Addons/env/colab_setup.sh --skip-data --skip-tunnel`. | First run slow; reused after. `run_ddsslam.sh` re-checks/rebuilds idempotently. | `run_ddsslam.sh` |
| 4 | **Depth + sc_factor for ALL 5 CRCD snippets**, cached to `/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/<EP>/snippet_<SID>/`. Run `Addons/colab/crcd_depth_gen_all5_20260617.sh` (the orchestrator-owned script that **includes c1_001 + c2_001**, unlike the old `_remainder_` script). `bench.sh` also invokes the per-snippet prep at run time; pre-running just warms the cache. | Depth must exist before SLAM. **The old `_remainder_` script EXCLUDES c1/c2** (critic gap 4) — never rely on it for c1/c2. | `crcd_depth_gen_all5` / `bench.sh` |

**Cache prerequisite assertion (critic gap 4).** `bench.sh` asserts, for every requested CRCD snippet, that **either** the Drive depth cache `CRCD-Published-MoGe-2/<EP>/snippet_<SID>/{depth/.DONE,.sc_factor}` exists **or** the raw snippet is present on Drive so prep can regenerate it. For `c1_001`/`c2_001` specifically — which the legacy `_remainder_` script cannot regenerate — if **neither** the cache nor a regeneration path exists, `bench.sh` escalates (CLARIFY) and skips that leg rather than crashing.

---

## 3. Common CLI Contract (every `run_<method>.sh` MUST honor)

```bash
bash <REPO>/run_<method>.sh <stage> <leg> <out_dir> <gpu_tier>
```

| Arg | Meaning | Allowed values |
|-----|---------|----------------|
| `<stage>` | matrix half | `repro` (own-dataset paper repro) or `crcd` (one snippet) |
| `<leg>` | sequence / snippet | repro: method-defined id (e.g. `stereomis_p2_1`, `replica_all`, `replica_room0`). crcd: `c1_001 e3_005 c3_001 g3_001 c2_001` |
| `<out_dir>` | absolute Drive dir | `/content/drive/MyDrive/Outputs/bench_<DATE>/<method>/<stage>/<leg>/` |
| `<gpu_tier>` | detected tier | `A100` \| `L4` \| `T4` \| `OTHER` |

**Contract:**

1. **Exit code is truth.** `0` only if required artifacts were produced; non-zero otherwise.
2. **Required artifacts in `<out_dir>` (uniform names):**
   - `metrics.json` — flat object with whatever of `{ate_mm, psnr, ssim, lpips, depth_l1}` the leg computed (missing = `null`; CRCD ate is **Sim3** ATE mm). repro legs also carry the method's paper-table value + paper reference for the **gate** (§4).
   - `est_c2w_data.txt` (or trajectory) + `groundtruth.txt` copy.
   - `render/` — rendered RGB (+ depth if available).
   - `render_metrics.txt` — **eval_rendering.py STDOUT captured verbatim** (this is the file the metrics emitter parses; see §7 / critic gap 1).
   - CRCD: `<leg>_6panel.mp4` (§7).
   - `status.txt` — one line `PASS|FAIL|GATE_FAIL|SKIP <reason>`.
   - `.DONE` — written **last**, only on success.
3. **GPU-adaptive + abort.** Honor `<gpu_tier>`; below the method's floor (§5) write `status.txt=FAIL insufficient_vram` and exit non-zero **without** starting training.
4. **Self-contained logging.** All to stdout/stderr; never swallow tracebacks.
5. **Resumable.** If `<out_dir>/.DONE` exists and `FORCE` unset, exit `0` immediately.
6. **Idempotent env.** Check/rebuild env at top.

DDS-SLAM's `run_ddsslam.sh` (Appendix A) is a thin adapter over the repo's canonical scripts. Its `crcd` path **runs the full prep itself** (stage → rectify → MoGe → sc_factor → config patch → copy depth to `/content`), then trains + evals + renders the video.

---

## 4. The run matrix + the decision GATE

**Today's runnable matrix = 1 method × (1 repro + 2 CRCD configs that exist).** The full design is 4 methods × (1 repro + 5 CRCD) = 24 legs, but 16 legs are descoped (§11) and 3 DDS-SLAM CRCD legs lack configs (§9 item 3). The count is therefore **not** "20 legs"; it is reported honestly per §8 as runnable/skipped/descoped.

| Method | `repro` leg | Repro granularity (paper-faithful) | GATE metric + reference |
|--------|-------------|-------------------------------------|-------------------------|
| **DDS-SLAM** | `stereomis_p2_1` (primary) — **its actual paper dataset** | Per-sequence render metrics, matching `eval_rendering.py:PAPER_REFERENCES['StereoMIS (P2_1)']` | PSNR 22.513, SSIM 0.592, LPIPS 0.496 (+ Sim3 ATE auditable). Reuses `Addons/colab/run_stereomis_p2_1.sh`. SemSup Lab1–4 (`Lab1 (trail3)` etc., PSNR 28.649/29.678/27.230/27.340) are an **optional** secondary repro via `repro_semsup_breakthrough_20260608.sh`. |
| Semantic-SuPer | `super_trail3` (+trail4/8/9 opt.) | Per-trial reprojection (Table I) | Reproj err px (Table I); repo `compute_rep_err.py`. **DESCOPED (§11).** |
| SNI-SLAM | `replica_all` (8 scenes) | 8-scene avg (Table 1) | Depth-L1 0.766 cm, ATE 0.456 cm, … **DESCOPED (§11).** |
| SemGauss-SLAM | `replica_all` (8 scenes) | 8-scene avg (Table I) | ATE 0.33 cm, Depth-L1 0.50 cm, PSNR 35.03, SSIM 0.982, LPIPS 0.062. **DESCOPED (§11).** |
| SGS-SLAM | `replica_room0` (+1) | Per-scene w/ Avg | ATE 0.41 cm, Depth-L1 0.36 cm, PSNR 34.66, SSIM 0.973, LPIPS 0.096. **DESCOPED (§11).** |

**DDS-SLAM repro is now WIRED (critic gap 5).** `REPRO_LEG[ddsslam]=stereomis_p2_1` (was wrongly `super_trail3`). `run_ddsslam.sh repro stereomis_p2_1` runs the StereoMIS P2_1 target and evals with `--sequence "StereoMIS (P2_1)"`, whose reference exists in `PAPER_REFERENCES`. The gate compares `metrics.json.psnr/ssim/lpips` to that reference.

**The GATE.** A method's CRCD legs are attempted **only if its repro PASSED** (within tolerance). `gate_check()` reads `metrics.json` (path resolved from `REPRO_LEG[m]`, not `listdir()[0]` — critic gap 12) vs the reference. On `GATE_FAIL`, that method's CRCD legs become `SKIP gate_fail`; **other methods continue**. Numeric tolerance is **[USER] §9 item 6**; until set, the gate is **report-only** (records pass/fail, does not block) and emits a CLARIFY line.

**Multi-scene repro collapse (critic gap 12).** For `replica_all` (8 scenes) the wrapper writes per-scene metrics into `<out_dir>/scenes/<scene>/metrics.json` **and** a single rolled-up `<out_dir>/metrics.json` whose values are the **mean over the 8 scenes** (matching each paper's dataset-wide average). The gate and aggregator read the rolled-up file only; per-scene files are retained for audit.

CRCD legs (DDS-SLAM): `c1_001  e3_005  c3_001  g3_001  c2_001` ↔ `C_1/001 E_3/005 C_3/001 G_3/001 C_2/001`. **Today only `c1_001` and `c2_001` have configs and can run** (§9 item 3).

---

## 5. GPU / VRAM preflight + adaptive policy

`bench.sh` reads `nvidia-smi` once and classifies the tier. Comparison is **margin-aware in MiB** (critic gap 13): a leg aborts only if `total_MiB < floor_MiB + HEADROOM_MiB` (HEADROOM=1024), so floored-GiB rounding (e.g. T4 reports 15109 MiB) never spuriously passes or aborts an exactly-at-floor card.

| Method | Class | Hard floor (MiB) | Recommended | Policy |
|--------|-------|------------------|-------------|--------|
| DDS-SLAM | NeRF / hash-grid | 14000 (fits T4 16 GB) | A100 (T4 wall ~16–20 h) | warn if not A100, continue |
| Semantic-SuPer | surfel + ED graph | 8000 | T4 / 2080 | warn <8000, continue |
| SNI-SLAM | NeRF / feature-plane | 12000 (w/ CPU offload) | 4090 / A100 | warn <12000, continue |
| **SemGauss-SLAM** | **3DGS** | **24000** (paper used 4090 24 GB; 16 GB OOMs mid-run) | **A100 40 GB** | **ABORT** if `< 24000 + headroom` |
| **SGS-SLAM** | **3DGS** | **16000** | **A100 40 GB** | **ABORT** if `< 16000 + headroom` |

SemGauss floor **raised 16→24 GB** to match the paper's actual hardware (critic gap 13). The exact floors remain **[USER] §9 item 8** (run-book-derived); until confirmed, `bench.sh` uses these and emits a CLARIFY line on every VRAM abort so a wrong floor is visible.

---

## 6. Resumability, logging, isolation, timeouts, retries

- **Sentinels.** Done iff `<out_dir>/.DONE`. Skipped (`SKIP already_done`) unless `FORCE=1`. **On `FORCE=1`, `run_leg()` clears `<out_dir>` (`rm -rf` its contents) before re-running** so no stale `est`/`render`/`sim3_metrics.txt`/`metrics.json` survive into aggregation (critic gap 11). DDS-SLAM CRCD prep keeps finer cache sentinels (`.STAGED`,`.PREPROCESSED`,`depth/.DONE`,`.sc_factor`) on Drive — those are the *prep* cache, not the *leg* output, and are kept across FORCE.
- **Per-stage logging.** Each leg → `<out_dir>/leg.log`; orchestrator → `$BENCH_ROOT/bench.log`; master `tee` (§1) is outermost.
- **Isolation.** Each leg runs in a subshell via `run_leg()`; no `set -e` around dispatch. A crash appends to `_failures.log` and sets the status cell to `FAIL`. One leg never aborts the loop.
- **Timeouts.** repro 600 min, crcd 360 min (T4 bump 600). Timeout → `FAIL timeout`.
- **Retries.** Idempotent prep (Drive copy, MoGe, env) retries once. Training is **not** auto-retried.
- **PASS/FAIL table.** `$BENCH_ROOT/STATUS.md` rewritten after every leg.

---

## 7. CRCD prep ordering + metrics + 6-panel video (shared harness)

Per CRCD leg, prep order is fixed and **executed by the wrapper** (DDS-SLAM owns the canonical implementation; the `crcd_prep()` function in Appendix A factors `crcd_depth_gen_all5`'s logic):

1. **Stage** raw `EP_snippet_SID` from Drive (tar-first) → `/content/crcd_raw/...` (`.STAGED`).
2. **Rectify** via `Addons/preprocess/preprocess_crcd_published.py` → `data/CRCD/<NAME>/{video_frames,masks,semantic_class,groundtruth.txt,rectified_calib.txt}` (`.PREPROCESSED`). `masks/`=binary tool mask (1=tool); `semantic_class`=4-class map (overlay only); `semantic_instance`=coco_id+1 (raw).
3. **MoGe-2 depth** (`generate_depth_moge.py --temporal_window 1 --depth_scale 10000 --max_depth_m 5.0`) → `depth/*.png` (`depth/.DONE`). **Rehydrate from the Drive cache first**; only generate if absent. Copy the depth PNGs to local `data/CRCD/<NAME>/depth/` so `ddsslam.py` (which requires them on `/content`) can read them (critic gap 2).
4. **Periodic stereo scale** (critic gap 9). Run StereoSGBM **every `STEREO_EVERY=100` frames** (frames 0,100,200,…), compute per-anchor `s_k = median(stereo_depth / MoGe_depth)`, and write `<NAME>/.sc_segments` (`frame_idx,s_k`) plus a global `.sc_factor = median(s_k)`. The single global `sc_factor` is what `ddsslam.py` consumes (the config has only one knob); the per-segment table is recorded for the paper's methodology audit and for the drift check below. **If the per-segment spread `max(s_k)/min(s_k) > 1.5`** the wrapper writes `status.txt=FAIL stereo_scale_drift` and escalates (a single global scale is unsafe on that deforming sequence) rather than silently averaging. *(If the user signs off on frame-0-only via `clarify/answers.env:CLARIFY_STEREO_CADENCE=frame0`, set `STEREO_EVERY=999999` to reproduce the legacy single-anchor behavior — that deviation must be recorded in the paper.)*
5. **sc_factor application + hard block (critic gap 7).** Patch `data.sc_factor` into the config. **Known correctness defect:** the code scales the depth maps (`dataset.py:195`), GT pose translations (`:252/:279`), and the SDF `trunc` weight (`scene_rep.py:107/401/544`) by `sc_factor`, but does **NOT** scale `training.range_d`, `cam.near`, `cam.far`, or `cam.depth_trunc` (`scene_rep.py:361/363/366/371`, `ddsslam.py:299/673/761`, `keyframe.py:37`). When `sc_factor` is far from 1, the depth-guided sampling band and validity masks become inconsistent with the rescaled depth → biased geometry/ATE/Depth-L1 on **exactly the published snippets** (MoGe est is ~8× GT metric scale per `sim3_ate.py` docstring). **The wrapper therefore HARD-BLOCKS** any leg with `|log(sc_factor)| > 0.1` (`status.txt=FAIL sc_factor_unscaled_fields`) until one of two fixes is applied and recorded in `clarify/answers.env:CLARIFY_SCFACTOR_FIX`:
   - **(A) Rescale the depth PNGs** so the global `sc_factor ≈ 1` (multiply each MoGe depth by the global `s` before writing, set `sc_factor: 1.0`). Then no config field is mis-scaled. *(preferred — keeps all downstream fields valid)*
   - **(B) Patch the config** to also scale `training.range_d`, `cam.near`, `cam.far`, `cam.depth_trunc` by `sc_factor` consistently.
   This is **not a deferrable warning** — publishing CRCD geometry with the half-scaled config is a correctness defect.

CRCD metrics (DDS-SLAM harness, identical for every method's CRCD leg):

- **ATE (Sim3):** `Addons/eval/sim3_ate.py --est <traj> --gt <NAME>/groundtruth.txt --name "<NAME>" --out "$OUT/sim3_metrics.txt"`. **`--out` APPENDS** (`sim3_ate.py:117`), so the wrapper `rm -f "$OUT/sim3_metrics.txt"` **before** the call (critic gap 11). Headline = Sim3 mean mm + recovered scale + |Pearson|dom; rigid ATE printed but never headlined.
- **PSNR/SSIM/LPIPS (critic gap 1 — the central fix):** `eval_rendering.py` writes the **means only to STDOUT** as `PSNR:  X (std: Y)` / `SSIM:  X` / `LPIPS:  X` (note the **two spaces** after the colon, `eval_rendering.py:196-199`). Its `--output_csv` writes **per-frame rows** with the lowercase header `frame,psnr,ssim[,lpips]` and contains **no** `PSNR:` literal; its `--summary_csv` header is `psnr_mean,psnr_std,…`. The previous emitter grepped the per-frame CSV for `PSNR:` → always `None` → all render metrics null. **Fix:** capture STDOUT to `render_metrics.txt` (`… > "$OUT/render_metrics.txt" 2>&1`, exactly as `run_cell.sh:73`) and the metrics emitter parses **that** file with the two-space-anchored regex `^PSNR:\s+([0-9.]+)`. A fallback averages the lowercase `psnr/ssim/lpips` columns of `render_eval.csv` if the stdout file is unparseable.
- **Depth-L1 (critic gap 8) — [CODE] DECISION, now defined.** CRCD has no measured GT depth (MoGe-2 is generated), so Depth-L1 cannot be "rendered vs GT depth". The benchmark adopts a concrete, reproducible proxy and **states it explicitly in the paper**: **Depth-L1 = mean |rendered_depth − stereo_SGBM_depth|** evaluated **only on the periodic stereo anchor frames** (frames 0,100,200,…) where an independent StereoSGBM depth exists (the same SGBM from prep step 4, in metres, after applying the global scale). Pairing: for each anchor frame `k`, mask to pixels where both rendered depth `>0` and SGBM depth ∈ (0.05, 3.0) m; L1 in **mm**; report the mean over anchors. Implemented as `Addons/eval/depth_l1_stereo.py` (**to author**) called from `run_ddsslam.sh`. If the user instead chooses to **drop** Depth-L1 from the CRCD metric set, set `clarify/answers.env:CLARIFY_DEPTH_L1=drop`; the wrapper then writes `depth_l1: null` and the paper's metric table omits the column with a stated reason ("no measured GT depth on CRCD"). The current behavior is **not** "emit null + CLARIFY and move on" — it is either compute the stereo-anchor Depth-L1 or explicitly drop it, decided before the run.

**Semantic input (critic gap 6) — [CODE] DECISION, now stated.** DDS-SLAM's loader reads the **binary tool mask** `masks/*.png` (`dataset.py:138`, `[-2000:]`); the 4-class `semantic_class` and raw `semantic_instance` are **never** read by the loader (only the video overlay uses `semantic_class`). So DDS-SLAM's "semantic SLAM" supervision on CRCD is **a binary instrument/tissue mask**, contradicting the base config comment that claims `semantic_instance`. The agent's recorded decision: **DDS-SLAM consumes `masks/` (binary tool mask); this satisfies the semantic-SLAM bar for this method** (its semantic loss is a per-pixel class term and binary is a valid 2-class instance). Two code facts must be verified and recorded once:
  - **Slice alignment.** Images/depth use `[-4000:]` (`dataset.py:133-134`); semantic uses `[-2000:]` (`:138`). All 5 snippets are ≤ 2108 frames (C1=360, C2≈730, E3_005=265, C3_001=1527, G3_001=1987), so both slices keep the full sequence and `len(semantic)==len(images)` ⇒ the `>=` branch (`:172-173`) gives 1:1 pairing. **The `index//2` halving fallback (`:175`) does NOT fire for any of the 5 snippets.** This is confirmed and harmless *for these lengths*; any future snippet > 2000 frames would truncate semantic and silently halve supervision — assert `len(semantic_paths) >= len(img_files)` in the wrapper preflight and FAIL if violated.
  - If the user requires the **4-class** map instead (`clarify/answers.env:CLARIFY_SEMANTIC_INPUT=semantic_class`), the loader/config must change to read `semantic_class/` — that is a code change, gated as a CLARIFY, not a silent default.

6-panel video (`Addons/viz/generate_video.py`, exactly `run_cell.sh:53-59` CRCD block): Input RGB, Rendered RGB, Input Depth, Output Depth, Seg Overlay (`semantic_class` palette, `--skip_raw_seg --seg_classmap`), Trajectory raw + Sim3-aligned. Optional uncertainty panel if `<OUT>/uncert` exists.

---

## 8. Final cross-method aggregation

After the matrix (fully or partially) completes, `bench.sh` aggregates:

1. **`aggregate.csv`** columns `method,stage,leg,status,ate_mm,psnr,ssim,lpips,depth_l1` — one row per leg from `<out_dir>/metrics.json`. Missing legs/metrics → `SKIP`/`null` (never silently dropped). **repro-leg discovery is driven by `REPRO_LEG[m]`**, not `listdir()[0]` (critic gap 12).
2. **`AGGREGATE.md`** — headline table, CRCD legs grouped by snippet, plus repro-gate pass/fail column, plus a **scope line** stating how many legs are RUN vs SKIP(config) vs DESCOPED so the table is never mistaken for a complete 4-method result.
3. **`_render_summary.csv`** — appended per leg by `eval_rendering.py --summary_csv`; basis for `aggregate_ab.py <root>` when multiple seeds run.
4. **Video + diagnostics gather** — copy every `<leg>_6panel.mp4` into `_videos/`, every trajectory `summary.txt`/sim3 report + `.sc_segments` into `_diagnostics/`.

---

## 9. Clarification Protocol — UNRESOLVED items that GATE the run

`bench.sh` reads `clarify/answers.env` (key=value). For any unanswered **[USER]** item, the affected leg is **not started**; `bench.sh` prints `CLARIFY[n]` and records `SKIP clarify_open`. **[CODE]** items are decisions the agent has made/recorded above and are wired into the wrappers (not user-blocking) but listed for traceability. **No item may be guessed.**

| # | Item | Type | Blocks | What is needed |
|---|------|------|--------|----------------|
| 1 | **Depth-L1 definition.** No measured GT depth on CRCD. | [CODE] resolved (§7) | depth_l1 column | Compute stereo-anchor Depth-L1 (`depth_l1_stereo.py`, to author) **or** set `CLARIFY_DEPTH_L1=drop`. Not "emit null silently". |
| 2 | **Stereo-scaling cadence.** Spec = ~every 100 frames; legacy repo = frame-0 only. | [CODE] resolved (§7 step 4) | depth scale, Depth-L1 | Periodic 100-frame SGBM is now the default; legacy frame-0 only via `CLARIFY_STEREO_CADENCE=frame0` (deviation must be documented). |
| 3 | **Missing configs for 3 of 5 snippets.** `c1_001`,`c2_001` exist; **`e3_005`,`c3_001`,`g3_001` do not exist at all** (no `_paperfaith_lrfix`, no raw). | **HARD BLOCK (not a clarification)** | CRCD legs `e3_005,c3_001,g3_001` | **The benchmark cannot report these 3 DDS-SLAM CRCD snippets until configs are authored.** Each needs: `timesteps` (confirmed E3_005=265, C3_001=1527, G3_001=1987 from the depth-gen frame table), `data.datadir/output`, and per-snippet `mapping.bound`/`marching_cubes_bound` **derived from frame-0 rectified depth** (procedure below). `bench.sh` exits `SKIP missing_config` for these until the YAMLs exist. *(Authoring requires the staged data; the agent must run prep, inspect frame-0 depth percentiles, set bounds = [p2,p98] per axis + 10% pad, then author the config — it must NOT guess bounds.)* |
| 4 | **Semantic input.** Loader reads **binary `masks/`** (`dataset.py:138`), not the 4-class map. | [CODE] resolved (§7) | semantic supervision | Default = binary tool mask (recorded). To switch to 4-class: `CLARIFY_SEMANTIC_INPUT=semantic_class` (code change). |
| 5 | **sc_factor unscaled fields.** `range_d/near/far/depth_trunc` not scaled. | **HARD BLOCK if `|log(sc)|>0.1`** | any CRCD leg with off-metric scale | Apply fix (A) rescale depth PNGs to `sc≈1`, or (B) patch all 4 fields; record in `CLARIFY_SCFACTOR_FIX`. Cannot publish half-scaled. |
| 6 | **Repro gate tolerance.** | [USER] | the GATE (and thus whether CRCD runs) per method | Numeric tolerance per metric. Until set, gate is report-only (non-blocking). *(DDS-SLAM repro **target** is now defined: `stereomis_p2_1` vs `StereoMIS (P2_1)` reference — critic gap 5.)* |
| 7 | **Other-method repro datasets.** SNI/SemGauss/SGS need Replica (+ seg heads / GT masks) staged. | [USER] | repro legs for the 3 descoped 3DGS methods | Per §11 onboarding. Moot while those methods are descoped. |
| 8 | **Per-method VRAM floor.** | [USER] | GPU aborts (§5) | Confirm min-VRAM per method (the §5 values are run-book-derived). |
| 9 | **Other-method onboarding.** No repos/wrappers/envs for semsuper/snislam/semgauss/sgsslam. | **DESCOPED** | 16 of the full matrix's legs | Complete §11 per repo (clone URL, `run_<method>.sh`, env build, dataset locations) before opting them in. |

---

## 10. `bench.sh` skeleton

> Concrete for DDS-SLAM (synced). Other methods dispatch to wrappers that do not yet exist and are gated off by default (§11). Save as `/content/bench/bench.sh`.

```bash
#!/bin/bash
# ============================================================================
# bench.sh — master orchestrator. DEFAULT METHODS=ddsslam (only runnable method).
#   Matrix: {ddsslam}(+descoped) x { repro + 5 CRCD snippets }.
#   Failure-isolated, resumable, GPU-adaptive.  NOT `set -e` at top level.
# ============================================================================
set -uo pipefail
DATE=$(date +%Y%m%d)
BENCH_ROOT=/content/drive/MyDrive/Outputs/bench_${DATE}
mkdir -p "$BENCH_ROOT"/{_videos,_diagnostics}
BENCH_LOG="$BENCH_ROOT/bench.log"; FAILLOG="$BENCH_ROOT/_failures.log"; STATUS_MD="$BENCH_ROOT/STATUS.md"
exec > >(tee -a "$BENCH_LOG") 2>&1
echo "=== bench.sh start $(date -Iseconds) -- BENCH_ROOT=$BENCH_ROOT ==="

# ---- run-control knobs ----------------------------------------------------
METHODS=${METHODS:-"ddsslam"}                 # default: only runnable method (critic gap 10)
STAGES=${STAGES:-"repro crcd"}
SNIPPETS=${SNIPPETS:-"c1_001 e3_005 c3_001 g3_001 c2_001"}
FORCE=${FORCE:-0}; DRYRUN=${DRYRUN:-0}
HEADROOM_MIB=1024

declare -A REPO=(
  [ddsslam]=/content/DDS-SLAM
  [semsuper]=/content/SemanticSuPer  [snislam]=/content/SNI-SLAM
  [semgauss]=/content/SemGauss-SLAM  [sgsslam]=/content/SGS-SLAM)   # DESCOPED until §11
declare -A REPRO_LEG=(                          # critic gap 5: ddsslam -> its OWN dataset
  [ddsslam]=stereomis_p2_1 [semsuper]=super_trail3
  [snislam]=replica_all [semgauss]=replica_all [sgsslam]=replica_room0)
declare -A VRAM_FLOOR_MIB=(                      # critic gap 13: MiB + headroom
  [ddsslam]=14000 [semsuper]=8000 [snislam]=12000 [semgauss]=24000 [sgsslam]=16000)

ANSWERS=/content/bench/clarify/answers.env; [ -f "$ANSWERS" ] && source "$ANSWERS"

clarify_blocked() {  # $1=method $2=stage $3=leg -> echo reason if blocked
  if [ "$2" = crcd ]; then
    case "$3" in
      e3_005|c3_001|g3_001)   # critic gap 3: configs DO NOT EXIST -> hard skip
        local cfg="${REPO[$1]}/configs/CRCD/${3}_paperfaith_lrfix.yaml"
        [ -f "$cfg" ] || { echo "missing_config[3]"; return; } ;;
    esac
    [ "${CLARIFY_SEMANTIC_INPUT:-masks}" = open ] && { echo "clarify_open[4:semantic]"; return; }
  fi
  if [ "$2" = repro ]; then case "$1" in
      snislam|semgauss|sgsslam) [ "${CLARIFY_REPRO_DATASET:-open}" = open ] && { echo "clarify_open[7:repro_dataset]"; return; } ;;
  esac; fi
  echo ""
}

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
case "$GPU_NAME" in *A100*) GPU_TIER=A100;; *L4*) GPU_TIER=L4;; *T4*) GPU_TIER=T4;; *) GPU_TIER=OTHER;; esac
echo "  GPU: $GPU_NAME (${GPU_MIB} MiB) tier=$GPU_TIER"
[ -d /content/drive/MyDrive ] || { echo "FATAL: Drive not mounted"; exit 1; }

declare -A STATUS
write_status_md(){ { echo "# bench status — $(date -Iseconds)"; echo;
  echo "GPU: $GPU_NAME ($GPU_MIB MiB, $GPU_TIER) | METHODS=$METHODS"; echo;
  printf '| method | repro | %s |\n' "$(echo $SNIPPETS|sed 's/ / | /g')";
  printf '|%s\n' "$(for _ in $(seq $(( $(echo $SNIPPETS|wc -w)+2 ))); do printf -- '---|'; done)";
  for m in $METHODS; do row="| $m | ${STATUS[$m:repro]:-—} |";
    for s in $SNIPPETS; do row="$row ${STATUS[$m:$s]:-—} |"; done; echo "$row"; done; } > "$STATUS_MD"; }

# ---- CRCD cache prerequisite assert (critic gap 4) ------------------------
crcd_cache_ok(){  # $1=snippet -> 0 if cache OR raw available
  local s=$1; local EP SID
  case "$s" in c1_001) EP=C_1 SID=001;; c2_001) EP=C_2 SID=001;; e3_005) EP=E_3 SID=005;;
               c3_001) EP=C_3 SID=001;; g3_001) EP=G_3 SID=001;; *) return 1;; esac
  local CACHE=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2/$EP/snippet_$SID
  local RAW=/content/drive/MyDrive/Datasets/CRCD-Published
  [ -f "$CACHE/depth/.DONE" ] && [ -f "$CACHE/.sc_factor" ] && return 0
  [ -f "$RAW/${EP}_snippet_${SID}_staging.tar" ] || [ -d "$RAW/$EP/snippet_$SID" ] && return 0
  return 1
}

run_leg(){  # $1=method $2=stage $3=leg
  local m=$1 stage=$2 leg=$3
  local out="$BENCH_ROOT/$m/$stage/$leg"
  local key="$m:$( [ "$stage" = repro ] && echo repro || echo "$leg" )"
  # FORCE clears the leg out_dir (critic gap 11)
  if [ "$FORCE" = 1 ] && [ -d "$out" ]; then rm -rf "$out"; fi
  mkdir -p "$out"
  if [ "$FORCE" != 1 ] && [ -f "$out/.DONE" ]; then STATUS[$key]="SKIP(done)"; echo "[$m/$stage/$leg] .DONE -> skip"; return 0; fi
  local blk; blk=$(clarify_blocked "$m" "$stage" "$leg")
  if [ -n "$blk" ]; then STATUS[$key]="SKIP($blk)"; echo "CLARIFY/SKIP [$m/$stage/$leg] $blk"; echo "SKIP $blk" > "$out/status.txt"; return 0; fi
  if [ "$stage" = crcd ] && ! crcd_cache_ok "$leg"; then
    STATUS[$key]="SKIP(no_depth_cache)"; echo "CLARIFY[4:cache] [$m crcd $leg] no Drive depth cache AND no raw to regenerate -> skip"; echo "SKIP no_depth_cache" > "$out/status.txt"; return 0; fi
  local floor=${VRAM_FLOOR_MIB[$m]:-0}
  if [ "$floor" -gt 0 ] && [ "$GPU_MIB" -lt $(( floor + HEADROOM_MIB )) ]; then
    STATUS[$key]="FAIL(vram<$floor)"; echo "ABORT: [$m] needs ${floor}+${HEADROOM_MIB} MiB, have $GPU_MIB"; echo "CLARIFY[8]: floor unconfirmed";
    echo "FAIL_VRAM $m/$stage/$leg" >> "$FAILLOG"; echo "FAIL insufficient_vram" > "$out/status.txt"; return 1; fi
  if [ "$DRYRUN" = 1 ]; then STATUS[$key]="DRY"; echo "[dry] would run $m/$stage/$leg"; return 0; fi
  local tmo=360; [ "$stage" = repro ] && tmo=600; [ "$GPU_TIER" = T4 ] && tmo=600
  local wrapper="${REPO[$m]}/run_${m}.sh"
  if [ ! -f "$wrapper" ]; then STATUS[$key]="FAIL(no_wrapper)"; echo "DESCOPED/FAIL: missing $wrapper (see §11)"; echo "MISSING_WRAPPER $m" >> "$FAILLOG"; echo "SKIP no_wrapper(descoped)" > "$out/status.txt"; return 0; fi
  echo "[$m/$stage/$leg] -> $wrapper (timeout ${tmo}m, $GPU_TIER)"
  ( timeout "${tmo}m" bash "$wrapper" "$stage" "$leg" "$out" "$GPU_TIER" ) 2>&1 | tee "$out/leg.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ -f "$out/.DONE" ]; then STATUS[$key]="PASS"
  elif [ "$rc" -eq 124 ]; then STATUS[$key]="FAIL(timeout)"; echo "TIMEOUT $m/$stage/$leg" >> "$FAILLOG"; echo "FAIL timeout" > "$out/status.txt"
  else STATUS[$key]="FAIL(rc=$rc)"; echo "FAIL_$rc $m/$stage/$leg" >> "$FAILLOG"; [ -f "$out/status.txt" ] || echo "FAIL rc=$rc" > "$out/status.txt"; fi
  write_status_md; return 0
}

gate_check(){  # $1=method ; report-only until tolerance set (critic gap 12: REPRO_LEG-driven)
  local m=$1 out="$BENCH_ROOT/$m/repro/${REPRO_LEG[$m]}"
  [ -f "$out/metrics.json" ] || { echo "GATE[$m]: no metrics.json @ $out -> FAIL"; return 1; }
  if [ "${CLARIFY_GATE_TOLERANCE:-open}" = open ]; then echo "CLARIFY[6]: gate tol unset for $m -> REPORT-ONLY"; return 0; fi
  python /content/bench/gate_check.py "$out/metrics.json" "$m"
}

for m in $METHODS; do
  echo ""; echo "################ METHOD: $m ################"; GATE_OK=1
  if [[ " $STAGES " == *" repro "* ]]; then run_leg "$m" repro "${REPRO_LEG[$m]}"; gate_check "$m" || GATE_OK=0; fi
  if [[ " $STAGES " == *" crcd "* ]]; then
    if [ "$GATE_OK" -ne 1 ]; then echo "[$m] repro GATE_FAIL -> SKIP CRCD"; for s in $SNIPPETS; do STATUS[$m:$s]="SKIP(gate_fail)"; done
    else for s in $SNIPPETS; do run_leg "$m" crcd "$s"; done; fi; fi
done

echo ""; echo "=== AGGREGATION ==="
python - "$BENCH_ROOT" "$METHODS" "$STAGES" "$SNIPPETS" <<'PY'
import sys, os, json, csv
root, methods, stages, snippets = sys.argv[1], sys.argv[2].split(), sys.argv[3].split(), sys.argv[4].split()
REPRO_LEG={'ddsslam':'stereomis_p2_1','semsuper':'super_trail3','snislam':'replica_all',
           'semgauss':'replica_all','sgsslam':'replica_room0'}     # critic gap 12: explicit, not listdir()[0]
rows=[]
for m in methods:
    legs=[]
    if 'repro' in stages: legs.append(('repro', REPRO_LEG.get(m,'repro')))
    if 'crcd' in stages: legs += [('crcd', s) for s in snippets]
    for stage, leg in legs:
        d=os.path.join(root,m,stage,leg); mj=os.path.join(d,'metrics.json')
        met=json.load(open(mj)) if os.path.isfile(mj) else {}
        stf=os.path.join(d,'status.txt')
        st=open(stf).read().strip() if os.path.isfile(stf) else ('SKIP' if not os.path.isfile(mj) else 'PASS')
        rows.append(dict(method=m,stage=stage,leg=leg,status=st,ate_mm=met.get('ate_mm'),
            psnr=met.get('psnr'),ssim=met.get('ssim'),lpips=met.get('lpips'),depth_l1=met.get('depth_l1')))
cols=['method','stage','leg','status','ate_mm','psnr','ssim','lpips','depth_l1']
with open(os.path.join(root,'aggregate.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,cols); w.writeheader(); w.writerows(rows)
def cell(v): return '—' if v is None else (f'{v:.3f}' if isinstance(v,float) else str(v))
run=sum(r['status'].startswith('PASS') for r in rows); skp=sum(r['status'].startswith('SKIP') for r in rows)
with open(os.path.join(root,'AGGREGATE.md'),'w') as f:
    f.write(f'# Cross-method aggregate\n\n_SCOPE: {run} legs RUN, {skp} SKIP/DESCOPED of {len(rows)} listed._\n\n')
    f.write('| '+' | '.join(cols)+' |\n|'+ '---|'*len(cols)+'\n')
    for r in rows: f.write('| '+' | '.join(cell(r[c]) for c in cols)+' |\n')
print(open(os.path.join(root,'AGGREGATE.md')).read()); print('CSV ->', os.path.join(root,'aggregate.csv'))
PY

find "$BENCH_ROOT" -name '*_6panel.mp4' -exec cp -n {} "$BENCH_ROOT/_videos/" \; 2>/dev/null || true
find "$BENCH_ROOT" \( -name 'sim3_metrics.txt' -o -name '.sc_segments' -o -name 'summary.txt' \) -path '*crcd*' \
     -exec sh -c 'cp -n "$1" "$0/_diagnostics/$(echo "$1"|tr / _)"' "$BENCH_ROOT" {} \; 2>/dev/null || true
python /content/DDS-SLAM/Addons/eval/aggregate_ab.py "$BENCH_ROOT" 2>/dev/null || echo "  (aggregate_ab.py: single-seed, skipped)"
write_status_md
echo ""; echo "=== bench.sh DONE $(date -Iseconds) ==="
echo "Status:$STATUS_MD  CSV:$BENCH_ROOT/aggregate.csv  MD:$BENCH_ROOT/AGGREGATE.md  Fails:$FAILLOG"
```

---

## Appendix A — DDS-SLAM `run_ddsslam.sh` (concrete adapter, contract §3)

Save as `/content/DDS-SLAM/run_ddsslam.sh`. The `crcd` path **now performs the full prep** (critic gap 2) and the metrics emitter **parses `render_metrics.txt`** (critic gap 1).

```bash
#!/bin/bash
# run_ddsslam.sh <stage> <leg> <out_dir> <gpu_tier>   (Common CLI Contract §3)
set -uo pipefail
STAGE=$1; LEG=$2; OUT=$3; TIER=${4:-OTHER}
REPO=/content/DDS-SLAM; cd "$REPO"; mkdir -p "$OUT"
[ "${FORCE:-0}" != 1 ] && [ -f "$OUT/.DONE" ] && { echo "done"; exit 0; }
bash Addons/env/colab_setup.sh --skip-data --skip-tunnel >/dev/null 2>&1 || true
STEREO_EVERY=${STEREO_EVERY:-100}
DRIVE_CRCD=/content/drive/MyDrive/Datasets/CRCD-Published
CALIB_PKL=$DRIVE_CRCD/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl
CACHE_ROOT=/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2

# ---- reusable CRCD prep (factored from crcd_depth_gen_all5; critic gaps 2,4,7,9) ----
crcd_prep(){  # $1=NAME(e.g.C1_001) $2=EP $3=SID $4=FRAMES ; sets $SC_GLOBAL ; exits non-zero on fail
  local NAME=$1 EP=$2 SID=$3 FR=$4
  local RAW=/content/crcd_raw/${EP}_snippet_${SID} STAGED=$REPO/data/CRCD/$NAME CACHE=$CACHE_ROOT/$EP/snippet_$SID
  [ -f "$CALIB_PKL" ] || { echo "FAIL calib_pickle_missing" > "$OUT/status.txt"; return 1; }
  # stage
  if [ ! -f "$RAW/.STAGED" ]; then mkdir -p "$RAW"; local T=$DRIVE_CRCD/${EP}_snippet_${SID}_staging.tar
    if [ -f "$T" ]; then tar xf "$T" -C "$RAW"||{ echo "FAIL stage_tar">"$OUT/status.txt";return 1;}
    elif [ -d "$DRIVE_CRCD/$EP/snippet_$SID" ]; then cp -r "$DRIVE_CRCD/$EP/snippet_$SID/." "$RAW/"||{ echo "FAIL stage_cp">"$OUT/status.txt";return 1;}
    else echo "FAIL raw_not_on_drive">"$OUT/status.txt"; return 1; fi; touch "$RAW/.STAGED"; fi
  # rectify
  if [ ! -f "$STAGED/.PREPROCESSED" ]; then rm -rf "$STAGED"; mkdir -p "$STAGED"
    python Addons/preprocess/preprocess_crcd_published.py --snippet_dir "$RAW" --calib_pkl "$CALIB_PKL" --output_dir "$STAGED" \
      || { echo "FAIL rectify">"$OUT/status.txt"; return 1; }
    [ -f "$STAGED/groundtruth.txt" ] || cp "$RAW/groundtruth.txt" "$STAGED/groundtruth.txt"; touch "$STAGED/.PREPROCESSED"; fi
  local N_L; N_L=$(find "$STAGED/video_frames" -name '*l.png'|wc -l)
  # depth: rehydrate from cache first, else generate
  if [ ! -f "$STAGED/depth/.DONE" ] || [ "$(ls "$STAGED/depth"/*.png 2>/dev/null|wc -l)" -lt "$N_L" ]; then
    if [ -f "$CACHE/depth/.DONE" ] && [ "$(ls "$CACHE/depth"/*.png 2>/dev/null|wc -l)" -ge "$N_L" ]; then
      mkdir -p "$STAGED/depth"; cp -n "$CACHE/depth"/*.png "$STAGED/depth/"; cp -n "$CACHE/.sc_factor" "$STAGED/.sc_factor" 2>/dev/null; touch "$STAGED/depth/.DONE"
    else  # generate (reuses crcd_depth_gen_all5 inner logic)
      bash Addons/colab/crcd_depth_gen_all5_20260617.sh "$NAME" "$EP" "$SID" "$FR" || { echo "FAIL moge_depth">"$OUT/status.txt"; return 1; }
      cp -n "$CACHE/depth"/*.png "$STAGED/depth/" 2>/dev/null; cp -n "$CACHE/.sc_factor" "$STAGED/.sc_factor" 2>/dev/null; fi; fi
  # periodic stereo scale (critic gap 9): writes .sc_segments + global .sc_factor; drift -> FAIL
  python Addons/depth/generate_depth_stereo.py --staged "$STAGED" --calib "$STAGED/rectified_calib.txt" \
     --every "$STEREO_EVERY" --segments_out "$STAGED/.sc_segments" --global_out "$STAGED/.sc_factor" \
     || { echo "FAIL stereo_scale">"$OUT/status.txt"; return 1; }
  SC_GLOBAL=$(cat "$STAGED/.sc_factor")
  cp "$STAGED/.sc_segments" "$OUT/" 2>/dev/null || true
  # sc_factor hard block (critic gap 7)
  local OFF; OFF=$(python -c "import math;print(1 if abs(math.log(float('$SC_GLOBAL')))>0.1 else 0)")
  if [ "$OFF" = 1 ] && [ "${CLARIFY_SCFACTOR_FIX:-open}" = open ]; then
    echo "FAIL sc_factor_unscaled_fields (sc=$SC_GLOBAL; range_d/near/far/depth_trunc not scaled — §7/§9.5)" > "$OUT/status.txt"; return 3; fi
  # semantic 1:1 assertion (critic gap 6)
  python - "$STAGED" "$N_L" <<'PY' || { echo "FAIL semantic_slice_mismatch">"$OUT/status.txt"; exit 4; }
import glob,os,sys; s=len(glob.glob(os.path.join(sys.argv[1],'masks','*.png'))); n=int(sys.argv[2])
assert s>=n, f"semantic {s} < frames {n}: index//2 halving would fire (§7)"
PY
  return 0
}

if [ "$STAGE" = crcd ]; then
  NAME=$(echo "$LEG" | tr '[:lower:]' '[:upper:]')   # c1_001 -> C1_001
  CFG="configs/CRCD/${LEG}_paperfaith_lrfix.yaml"
  [ -f "$CFG" ] || { echo "FAIL missing_config $CFG (§9.3 — author config first)" > "$OUT/status.txt"; exit 3; }
  case "$LEG" in c1_001) EP=C_1 SID=001 FR=360;; c2_001) EP=C_2 SID=001 FR=730;;
                 e3_005) EP=E_3 SID=005 FR=265;; c3_001) EP=C_3 SID=001 FR=1527;; g3_001) EP=G_3 SID=001 FR=1987;;
                 *) echo "FAIL unknown_snippet">"$OUT/status.txt"; exit 5;; esac
  crcd_prep "$NAME" "$EP" "$SID" "$FR" || exit $?
  # patch sc_factor into config (apply fix B fields too if CLARIFY_SCFACTOR_FIX=patch_fields)
  python Addons/colab/patch_sc_factor.py --config "$CFG" --sc "$SC_GLOBAL" --mode "${CLARIFY_SCFACTOR_FIX:-trunc_only}"
  DDIR="data/CRCD/$NAME"; ODIR="output/CRCD/${NAME}_paperfaith_lrfix"; RUN="$ODIR/demo"
  python ddsslam.py --config "$CFG" || { echo "FAIL slam" > "$OUT/status.txt"; exit 6; }
  [ -f "$RUN/est_c2w_data.txt" ] || { echo "FAIL no_trajectory" > "$OUT/status.txt"; exit 6; }
  rm -f "$OUT/sim3_metrics.txt"   # critic gap 11: --out APPENDS
  python Addons/eval/sim3_ate.py --est "$RUN/est_c2w_data.txt" --gt "$DDIR/groundtruth.txt" --name "$NAME" --out "$OUT/sim3_metrics.txt"
  # capture eval_rendering STDOUT (critic gap 1): means live ONLY on stdout
  python Addons/eval/eval_rendering.py --gt_dir "$DDIR/video_frames" --render_dir "$ODIR" \
     --name "$NAME" --output_csv "$OUT/render_eval.csv" \
     --summary_csv "$BENCH_ROOT/_render_summary.csv" --sequence "CRCD ($NAME)" \
     > "$OUT/render_metrics.txt" 2>&1 || echo "WARN render-eval"
  # Depth-L1 stereo-anchor proxy (critic gap 8) unless dropped
  if [ "${CLARIFY_DEPTH_L1:-stereo}" != drop ]; then
    python Addons/eval/depth_l1_stereo.py --render_depth "$ODIR/depth" --staged "$DDIR" \
       --segments "$DDIR/.sc_segments" --sc "$SC_GLOBAL" --out "$OUT/depth_l1.txt" || echo "WARN depth_l1"; fi
  python Addons/viz/generate_video.py \
     --rgb_input_dir "$DDIR/video_frames" --rgb_input_pattern '*l.png' \
     --rgb_output_dir "$ODIR" --rgb_output_pattern '[0-9]*.jpg' \
     --depth_input_dir "$DDIR/depth" --depth_output_dir "$ODIR/depth" --depth_norm robust \
     --seg_dir "$DDIR/semantic_class" --seg_pattern '*.png' --skip_raw_seg --seg_classmap \
     --trajectory_est "$RUN/est_c2w_data.txt" --trajectory_gt "$DDIR/groundtruth.txt" --trajectory_raw \
     --output "$OUT/${LEG}_6panel.mp4" --fps 15 2>&1 | tail -3 || echo "WARN video"
  cp "$RUN/est_c2w_data.txt" "$DDIR/groundtruth.txt" "$OUT/" 2>/dev/null || true
  # emit metrics.json — parse render_metrics.txt two-space format (critic gap 1)
  python - "$OUT" <<'PY'
import sys,os,re,json
o=sys.argv[1]
def rd(p): return open(os.path.join(o,p)).read() if os.path.isfile(os.path.join(o,p)) else ''
def g(p,s): m=re.search(p,s,re.M); return float(m.group(1)) if m else None
t=rd('sim3_metrics.txt')          # "Sim3 ATE  rmse/mean/median/max : R / M / .. mm" -> mean
ate=g(r'Sim3 ATE[^:]*:\s*[0-9.]+\s*/\s*([0-9.]+)', t)
rm=rd('render_metrics.txt')       # "PSNR:  X (std: Y)" — two spaces (eval_rendering.py:196)
psnr=g(r'^PSNR:\s+([0-9.]+)', rm); ssim=g(r'^SSIM:\s+([0-9.]+)', rm); lpips=g(r'^LPIPS:\s+([0-9.]+)', rm)
if psnr is None:                  # fallback: average lowercase per-frame CSV columns
  import csv
  c=os.path.join(o,'render_eval.csv')
  if os.path.isfile(c):
    rows=list(csv.DictReader(open(c)))
    col=lambda k:[float(r[k]) for r in rows if r.get(k) not in (None,'')]
    av=lambda k:(sum(col(k))/len(col(k))) if col(k) else None
    psnr,ssim,lpips=av('psnr'),av('ssim'),av('lpips')
dl1=None
dt=rd('depth_l1.txt')
if dt: dl1=g(r'Depth-?L1[^0-9]*([0-9.]+)', dt)
json.dump(dict(ate_mm=ate,psnr=psnr,ssim=ssim,lpips=lpips,depth_l1=dl1),
          open(os.path.join(o,'metrics.json'),'w'))
PY
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"
else
  # repro: DDS-SLAM's OWN dataset = StereoMIS P2_1 (critic gap 5)
  case "$LEG" in
    stereomis_p2_1) SEQ="StereoMIS (P2_1)"; bash Addons/colab/run_stereomis_p2_1.sh "$OUT" || { echo "FAIL repro_slam">"$OUT/status.txt"; exit 6; } ;;
    super_trail3)   SEQ="Lab1 (trail3)";    bash Addons/colab/repro_semsup_breakthrough_20260608.sh "$OUT" || { echo "FAIL repro_slam">"$OUT/status.txt"; exit 6; } ;;
    *) echo "FAIL unknown_repro_leg $LEG (§9.6)" > "$OUT/status.txt"; exit 6 ;;
  esac
  python Addons/eval/eval_rendering.py --gt_dir "$OUT/render" --render_dir "$OUT/render" \
     --name "$LEG" --sequence "$SEQ" --summary_csv "$BENCH_ROOT/_render_summary.csv" \
     > "$OUT/render_metrics.txt" 2>&1 || echo "WARN render-eval"
  python - "$OUT" "$SEQ" <<'PY'
import sys,os,re,json
o,seq=sys.argv[1],sys.argv[2]; rm=open(os.path.join(o,'render_metrics.txt')).read()
def g(p): m=re.search(p,rm,re.M); return float(m.group(1)) if m else None
ref={'StereoMIS (P2_1)':(22.513,0.592,0.496),'Lab1 (trail3)':(28.649,0.797,0.231)}.get(seq)
d=dict(psnr=g(r'^PSNR:\s+([0-9.]+)'),ssim=g(r'^SSIM:\s+([0-9.]+)'),lpips=g(r'^LPIPS:\s+([0-9.]+)'),
       ate_mm=None,depth_l1=None,
       ref_psnr=ref[0] if ref else None, ref_ssim=ref[1] if ref else None, ref_lpips=ref[2] if ref else None)
json.dump(d,open(os.path.join(o,'metrics.json'),'w'))
PY
  echo "PASS" > "$OUT/status.txt"; sync; touch "$OUT/.DONE"
fi
```

> **New scripts this appendix requires (to author, all under `Addons/`):**
> - `Addons/colab/crcd_depth_gen_all5_20260617.sh` — depth-gen that **includes c1_001 + c2_001** (the `_remainder_` script excludes them; critic gap 4), callable per-snippet `<NAME> <EP> <SID> <FRAMES>`.
> - `Addons/depth/generate_depth_stereo.py` extension — `--every` periodic SGBM writing `.sc_segments` + global `.sc_factor`, with `max(s_k)/min(s_k)>1.5` drift abort (critic gap 9).
> - `Addons/colab/patch_sc_factor.py` — patches `data.sc_factor` (`--mode trunc_only`) **or** also `range_d/near/far/depth_trunc` (`--mode patch_fields`) **or** rescales depth PNGs to `sc≈1` (`--mode rescale_png`) (critic gap 7 / §7 step 5).
> - `Addons/eval/depth_l1_stereo.py` — rendered-depth vs stereo-SGBM on the periodic anchor frames, L1 in mm (critic gap 8 / §7).

---

## 11. Onboarding the descoped methods (semsuper / snislam / semgauss / sgsslam)

Until **all** of the following exist per repo, that method stays out of `METHODS` and `bench.sh` reports its legs as `SKIP no_wrapper(descoped)` (critic gap 10). "One bash line runs everything" becomes true only after this is complete for each:

1. **Clone URL + commit** pinned under `/content/<Repo>/` (record exact upstream).
2. **`run_<method>.sh`** conforming to §3 (the same artifact set incl. `render_metrics.txt`, `metrics.json`, `.DONE`).
3. **Env build** command (3DGS methods compile CUDA rasterizers).
4. **Repro dataset on Drive** (Replica for SNI/SemGauss/SGS; SemSup trials) + the method's **semantic source** (GT masks / pretrained seg-head-on-DINOv2) staged and located. *(Note: SNI-SLAM ScanNet/TUM and the 3DGS seg-heads may not be reproducible from public repos — that is §9 item 7, [USER].)*
5. **CRCD semantic adapter** — define how that method's required semantic input is supplied from CRCD (`semantic_instance`/`semantic_class`/binary `masks`), escalating if CRCD lacks the source.

---

### Files this run book points at (all absolute, in this repo)
- Master orchestrator (to author): `/content/bench/bench.sh` — §10.
- DDS-SLAM adapter (to author): `/content/DDS-SLAM/run_ddsslam.sh` — Appendix A.
- New helper scripts (to author): `Addons/colab/crcd_depth_gen_all5_20260617.sh`, `Addons/colab/patch_sc_factor.py`, `Addons/eval/depth_l1_stereo.py`, periodic-`--every` extension of `Addons/depth/generate_depth_stereo.py`.
- Reused as-is: `/Users/benwright/Desktop/DDS-SLAM-BEN/DDS-SLAM/Addons/colab/run_cell.sh`, `run_crcd_4snippets.sh`, `run_stereomis_p2_1.sh`, `repro_semsup_breakthrough_20260608.sh`, `Addons/preprocess/preprocess_crcd_published.py`, `Addons/depth/generate_depth_moge.py`, `Addons/eval/sim3_ate.py`, `Addons/eval/eval_rendering.py`, `Addons/eval/aggregate_ab.py`, `Addons/viz/generate_video.py`, `Addons/env/colab_setup.sh`.
- **Do NOT use for c1/c2 depth:** `Addons/colab/crcd_depth_gen_remainder_20260616.sh` (its `SNIPPETS` array starts at F3_004; comment line 29 "C/F done already" — it cannot regenerate c1_001/c2_001; critic gap 4).
- Configs present: `configs/CRCD/c1_001_paperfaith_lrfix.yaml`, `c2_001_paperfaith_lrfix.yaml`, base `crcd_paperfaith_lrfix.yaml`. **Absent — HARD BLOCK until authored (§9.3):** `e3_005_paperfaith_lrfix.yaml` (timesteps=265), `c3_001_paperfaith_lrfix.yaml` (1527), `g3_001_paperfaith_lrfix.yaml` (1987); each needs per-snippet `mapping.bound`/`marching_cubes_bound` derived from frame-0 rectified depth ([p2,p98]±10% pad) — derived, never guessed. Until then DDS-SLAM reports only c1_001 + c2_001.
