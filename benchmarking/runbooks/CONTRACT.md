# CONTRACT.md — the single benchmark contract (Arm 4)

> **Authority.** This file is the ONE source of truth for the **CLI signature, output
> layout, artifact filenames, exit codes, and aggregation** of the CRCD semantic-SLAM
> benchmark. Where [00_COMMON.md](00_COMMON.md) §7, [RUN_ALL.md](RUN_ALL.md), or any
> per-method run book (`SemGauss-SLAM.md`, `SGS-SLAM.md`, `SNI-SLAM.md`,
> `SemanticSuPer.md`) disagree with this file on those things, **this file wins** — the
> others were authored at different times and drifted into three incompatible CLI/output/
> exit-code schemes (the #1 onboarding blocker found by the 2026-06-17 audit).
>
> **Inherited, not restated here.** The *scientific* decisions still live in
> [00_COMMON.md §0](00_COMMON.md) "Resolved Benchmark Decisions" (Depth-L1 = input-vs-output;
> periodic ~100-frame stereo rescaling; CRCD semantics mirror each paper's own mechanism on
> the 4 classes; missing configs are the agent's job) and the metric law in §5 below. This
> file governs the *plumbing* so all four methods' wrappers are byte-compatible with the
> shared DDS-SLAM harness and the cross-method aggregator.
>
> **Grounding.** Every "EXISTS" / "TO-BUILD" claim in §7 was verified against the live repo
> on 2026-06-17 (not copied from the audit summary, which over-stated the build scope —
> several "must author" items already exist).

Created: 2026-06-17 · Arm 4 item **A4-0.1**. Companion: [ARM4_build_queue.md](../ARM4_build_queue.md), [AUDIT_onboarding_3methods_20260617.md](../AUDIT_onboarding_3methods_20260617.md).

---

## 1. CLI signature (every `run_<method>.sh`)

```bash
bash run_<method>.sh <phase> [snippet]
```

| Token | Values | Meaning |
|---|---|---|
| `<phase>` | `env` · `repro` · `crcd` · `eval` · `all` | `env` = build+verify the method env incl. `lpips`+MoGe-2 and the VRAM-floor gate (§5). `repro` = Phase-A own-dataset repro + decision gate. `crcd` = stage+depth+SLAM for `[snippet]` (all 5 if omitted), looping `SEEDS`. `eval` = shared harness (§3) for `[snippet]`. `all` = `env → repro → (gate) → crcd → eval`. |
| `[snippet]` | `C1_001 C2_001 E3_005 C3_001 G3_001` (UPPERCASE; omit = all 5) | The benchmark's fixed snippet set. `F1_002`/`F3_007` are NOT in this benchmark. |

The `<stage> <leg> <out_dir> <gpu_tier>` 4-arg form in RUN_ALL.md §3 and the 3-arg form in
SemGauss-SLAM.md §7 are **DEPRECATED**. The orchestrator passes only `<phase> [snippet]`; the
wrapper derives its own `out_dir` from §2 (it is NOT supplied by the caller).

Read-with-defaults env vars: `SEEDS` (default `0 1 2`), `DRIVE_ROOT` (default per §2), `FORCE`
(default `0`; `1` clears the leg dir before re-run), `GPU_FLOOR_GB` (per §5/§8).

---

## 2. Output layout (canonical)

- **Batch root:** `DRIVE_ROOT = /content/drive/MyDrive/Outputs/<method>_crcd_<DATE>`
  (e.g. `SGS-SLAM_crcd_20260618`). `<method>` ∈ `DDS-SLAM SGS-SLAM SNI-SLAM SemGauss-SLAM`.
- **Per snippet × seed:** `DRIVE_ROOT/<NAME>_s<SEED>/` — the `_s<SEED>` suffix is **mandatory**
  for the n=3 protocol (§5). Single-seed debugging may use `DRIVE_ROOT/<NAME>/`.
- **Single manual runs:** `/content/drive/MyDrive/Outputs/manual_cells/<NAME>/`.
- **`.DONE` is written LAST, only on success.** Resumable: if `<dir>/.DONE` exists and `FORCE`
  unset, the leg exits 0 immediately. On `FORCE=1` the wrapper `rm -rf`s the leg dir first.

### 2.1 Required contents of each `<NAME>_s<SEED>/`
| File | Producer | Notes |
|---|---|---|
| `render_eval.csv` | `eval_rendering.py --output_csv` | **per-frame** rows, header `frame,psnr,ssim[,lpips]`. The aggregator's PREFERRED input. |
| `render_eval.txt` | `eval_rendering.py` STDOUT, tee'd | human summary (`PSNR: …`, `Rendered: N`). Aggregator FALLBACK. |
| `sim3_metrics.txt` | `sim3_ate.py --out` | **`rm -f` it before the call** — `--out` APPENDS. Format must match `sim3_ate.py` (Sim3 ATE rmse/mean/median/max mm + `\|Pearson\| dom axis`). |
| `depth_l1.txt` | `depth_l1.py --out` | line `Depth-L1 (mm): mean=… median=… frames=…` (§3). |
| `depth_l1_curve.csv` | `depth_l1.py --csv` | per-frame curve (diagnostic). |
| `<NAME>_6panel.mp4` | `generate_video.py` | CRCD 6-panel (§5). |
| `summary.txt` | wrapper | GPU name+VRAM+driver, wall-clock, peak VRAM, GT-motion verdict (§5). |
| `payload.tgz` | wrapper | tars `demo/` + ckpts + `depth/` + renders **+ `render_eval.{csv,txt}` + `sim3_metrics.txt` + `depth_l1.txt`** (the metrics files MUST be inside, the aggregator reads them from the tar if loose copies are absent). |
| `.DONE` | wrapper, last | success sentinel. |

> **FILENAME law (resolves the §5.4 producer/consumer conflict):** the canonical render-metric
> filenames are **`render_eval.csv` + `render_eval.txt`**, because the consumer we use
> (`aggregate_crcd_generic.py`) reads exactly those. The old `render_metrics.txt` name (00_COMMON
> §5.2 / RUN_ALL §3) belonged to `aggregate_ab.py` (the DDS A/B/C study) and is **deprecated** here.

---

## 3. Metric files — exact formats (so the aggregator parses them)

- **Trajectory (Sim3 ATE):** `sim3_ate.py --est <traj> --gt <NAME>/groundtruth.txt --name <NAME> --out <dir>/sim3_metrics.txt`. The est trajectory must be `est_c2w_data.txt`-format: one pose/line, **12 floats (3×4 row-major c2w) or 16 floats (4×4)**, translation at cols `[3,7,11]` (`sim3_ate.load_est`). Methods that store poses elsewhere convert first (DDS writes it natively; SNI uses `sni_export_traj.py`; SGS/SemGauss need a `npz→est` adapter — §7).
- **Render (PSNR/SSIM/LPIPS):** `eval_rendering.py --gt_dir <NAME>/video_frames --render_dir <OUT> --name <NAME> --sequence "CRCD (<NAME>)" --output_csv <dir>/render_eval.csv  > <dir>/render_eval.txt 2>&1`. The 5 CRCD `--sequence` keys already exist in `PAPER_REFERENCES` (all `None` refs — relative-across-methods, no paper baseline). `lpips` must be installed (`env` phase). Renders MUST be `[0-9]*.jpg/png` at the `--render_dir` root with int-parseable stems (non-DDS methods rename — §7).
- **Depth-L1 (input-vs-output, mm):** `depth_l1.py --render_depth_dir <OUT>/depth --input_depth_dir <NAME>/depth --render_scale <Sr> --input_scale <Si> --sc_factor <SC> --out <dir>/depth_l1.txt --csv <dir>/depth_l1_curve.csv`. Greppable line: `Depth-L1 \(mm\): mean=([0-9.]+)`. Per 00_COMMON §0 Decision 1 this is self-consistency (rendered depth vs the depth the model was GIVEN), NOT vs measured GT — state in the paper.

---

## 4. Exit codes (canonical)

| Code | Meaning |
|---|---|
| `0` | success (required artifacts produced; `.DONE` written) |
| `10` | Phase-A repro **gate FAIL** (below bar) — do not proceed to CRCD for this method |
| `20` | **BLOCKED / needs clarification** — write `<dir>/.NEEDS_CLARIFICATION`, escalate, do not fabricate downstream |
| `30` | env/build failure (incl. failed `lpips`/MoGe import, rasterizer compile) |
| `40` | **VRAM floor abort** (3DGS hard abort below floor; §5) |
| `1` | generic/uncaught failure |

DEPRECATED per-doc schemes mapped to the above: SGS-SLAM.md `42` (escalate) → **20**;
SemGauss-SLAM.md `78` (VRAM) → **40**; RUN_ALL.md ad-hoc `3/4/6/124` → `124` stays
"timeout" (orchestrator), the rest → `1` (or the specific code above when applicable).

---

## 5. Metric law + run protocol (inherited; non-negotiable)

- **Sim3 only.** Headline trajectory metric is **Sim3 ATE via `sim3_ate.py`** (+ recovered scale,
  est/GT path-ratio, `\|Pearson\| dom`). The pipeline's own rigid `output.txt` / `tools/eval_ate.py`
  is **NEVER headlined** (it inverts A/Bs on up-to-scale depth).
- **Headline tracking only on C2_001.** The other CRCD snippets are sub-SNR (reject-for-tracker,
  keep-for-render). Report their ATE but do not headline it as tracking quality. Always quote
  path-ratio + Pearson alongside any ATE.
- **n=3 seeds {0,1,2}** for every headline number; report mean ± std (the seed coin-flip is the
  noise floor). TF32 OFF; seed via each method's native seed field (never edit model source).
- **VRAM abort.** Log `nvidia-smi` (name/total/used/driver) + wall-clock + peak VRAM per stage.
  3DGS methods (SemGauss, SGS): if `memory.total < GPU_FLOOR_GB` → **exit 40, do not train**.
  Hash-grid/NeRF (DDS, SNI): warn-and-continue but log the floor decision.
- **6-panel video** (`generate_video.py`, CRCD branch): Input RGB · Rendered RGB · Input Depth ·
  Output Depth · Seg Overlay (4-class palette) · Trajectory (raw + Sim3-aligned). Always produced.

---

## 6. Aggregation

`aggregate_crcd_generic.py` is the canonical cross-snippet / cross-method aggregator (method-agnostic;
reads `<ROOT>/<NAME>/{render_eval.csv|render_eval.txt, sim3_metrics.txt}` or `payload.tgz`):

```bash
# single method, cross-snippet:
python Addons/eval/aggregate_crcd_generic.py --root <DRIVE_ROOT> --names C1_001 C2_001 E3_005 C3_001 G3_001 --out <DRIVE_ROOT>/COMBINED_SUMMARY.txt
# cross-method master table:
python Addons/eval/aggregate_crcd_generic.py --method-roots DDS-SLAM=<r0> SGS-SLAM=<r1> SNI-SLAM=<r2> SemGauss=<r3> --names ... --out CROSS_METHOD_TABLE.txt
```

> **KNOWN LIMIT (Arm 4 item A4-4.1 = EXTEND, not author):** as written it reads one dir per snippet
> (`<NAME>/`) and reports mean±std **across the 5 snippets**, NOT across seeds, and does **not** read
> `depth_l1.txt`. To honor the n=3 protocol + the Depth-L1 column it needs a small extension:
> accept `--seeds 0 1 2`, read `<NAME>_s<SEED>/`, compute **per-snippet over-seed** mean±std, and add a
> Depth-L1 column parsed from `depth_l1.txt`. Until extended, run it per-seed-root and combine by hand.

---

## 7. Harness inventory — EXISTS vs TO-BUILD (verified 2026-06-17)

**EXISTS and reusable as-is (audit was STALE in calling some of these "to author"):**
- `Addons/eval/sim3_ate.py`, `eval_rendering.py` (CRCD keys C1/C2/C3/E3/G3 present, `None` refs),
  `aggregate_crcd_generic.py` (§6), `sni_export_traj.py` (SNI ckpt → `est_c2w_data.txt`),
  `kitti_to_tum.py`, `compute_rep_err.py`.
- `Addons/depth/generate_depth_moge.py`, `moge_npy_to_png.py`, `generate_depth_stereo.py`
  (**RAFT-stereo for StereoMIS — NOT the CRCD SGBM sc_factor path**), `generate_depth_for_ddsslam.py`.
- `Addons/preprocess/preprocess_crcd_published.py` (emits `video_frames/`, `masks/`, `semantic_class/`,
  `groundtruth.txt`, `rectified_calib.txt`; coco_id+1, NEAREST rectify).
- `Addons/viz/generate_video.py`, `Addons/env/colab_setup.sh` (installs tcnn/pytorch3d/marching_cubes;
  does **NOT** install `lpips`/MoGe — wrappers must, §1 `env`).
- `Addons/colab/run_crcd_4snippets.sh` (structural template + the **inline** CRCD stereo-SGBM
  `sc_factor` heredoc — the authoritative single-anchor logic A4-0.4 must extract+make periodic).

**TO-BUILD (verified ABSENT):**
- `Addons/eval/depth_l1.py` → **DONE this turn (A4-0.2).**
- Periodic (every-~100-frame) stereo left-frame depth — **DELEGATED to another agent (A4-0.4, 2026-06-18),
  not built here.** Output is **METRIC-SCALED (user-confirmed) → `sc_factor = 1.0` everywhere**, so the
  `range_d/near/far/depth_trunc` field-scaling bug is **MOOT** and `patch_sc_factor.py` is **not needed**.
  Consume interface (confirm exact dir/scale with that agent): `data/CRCD/<NAME>/depth/<fid>.png` uint16
  (value = depth_m × scale); each method's `png_depth_scale` must equal that scale; `depth_l1.py` runs with
  `--sc_factor 1.0`; per-method config `sc_factor`/`data.sc_factor` = `1.0`.
- Shared DINOv2 4-class seg-head trainer → `dinov2_crcd.pth` — **CODE DONE (`Addons/seg/train_dinov2_crcd.py`,
  A4-0.3); run pending Colab.** Head-shape reconciliation RESOLVED: SNI's & SemGauss's `DINO2SEG` are the
  same class → **ONE `dinov2_crcd.pth` serves both** IFF both CRCD configs use `n_classes=4, c_dim=16,
  crop_edge=0` (cross-method constraint — SNI already conforms; SemGauss CRCD config must). SGS needs no head.
- `patch_sc_factor.py` (config patcher) — referenced by RUN_ALL but ABSENT. **NOT needed** (metric-scaled
  delegated depth → `sc_factor=1.0` is a static config value, no per-snippet patching).
- The aggregator seed+Depth-L1 extension — A4-4.1 (§6).
- Per-method `run_<method>.sh` wrappers + the master `run_all.sh` — Stages 1-3 / 4.
- The 3 missing CRCD configs `e3_005/c3_001/g3_001` (per method) — A4-0.5. The bound-derivation tool
  `Addons/preprocess/derive_crcd_bounds.py` is **DONE** (pinned recipe; only SNI/NeRF needs bounds; run it
  on the delegated metric depth, and RE-DERIVE c1/c2 too — their committed bounds are stale up-to-scale).

---

## 8. OPEN [USER] PARAMETERS (must be set before the affected leg's headline ships)

These are genuine user decisions; until answered the wrapper uses the interim default shown and
the gate is **report-only** (records pass/fail, does not block). Not silent — surfaced here + in
each leg's `summary.txt`.

| Param | Interim default | Needed for |
|---|---|---|
| Per-method **repro gate tolerance** (Phase A pass bar) | report-only (no block) | the decision gate; whether CRCD runs |
| Per-method **VRAM floor (GB)** | DDS 14 · SNI 12 · SGS 16 · SemGauss 24 (audit §5, run-book-derived) | the `exit 40` abort (3DGS) |

**RESOLVED (do NOT re-ask):** stereo cadence = periodic ~100 frames (00_COMMON §0 Decision 2);
Depth-L1 = input-vs-output (Decision 1); CRCD semantics = mirror each paper's own mechanism on the
4 classes (Decision 3); missing configs = agent authors them (Decision 4).
