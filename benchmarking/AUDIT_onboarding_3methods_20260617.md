# Onboarding Audit — SemGauss-SLAM, SGS-SLAM, SNI-SLAM into the DDS-SLAM benchmark

> Produced 2026-06-17 by a deep multi-agent audit (36 agents; claims extracted from the run books, verified file-by-file against the actual `SemGauss-SLAM/`, `SGS-SLAM/`, `sni-slam/` repos + the DDS-SLAM shared harness; every BLOCKER/MAJOR gap adversarially refuted — 18 survived, 6 dropped; 16 spec contradictions found).

> ⚠️ **POST-AUDIT CORRECTION (2026-06-17, Stage 0).** This audit's harness inventory was partly STALE: `Addons/eval/aggregate_crcd_generic.py`, `sni_export_traj.py`, `Addons/depth/moge_npy_to_png.py`, and the 5 CRCD `--sequence` keys in `eval_rendering.py` **already exist** — the §4/§5/§6 "must author" claims for those are wrong. The authoritative, verified EXISTS-vs-TO-BUILD inventory now lives in [runbooks/CONTRACT.md §7](runbooks/CONTRACT.md). The §4.2 SNI semantic conclusion below was also corrected (CRCD seg-head is REQUIRED). Treat CONTRACT.md + [ARM4_build_queue.md](ARM4_build_queue.md) as the live source; this file is the historical research record.

All file references are relative to the workspace root `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM`. Findings marked **FACT** are grounded in verified file:line reads; **INFERENCE** marks reasoning over those facts.

---

## 1. Bottom line

Yes — all three can be onboarded. All three repos are **present and checked out** (this refutes the run books' single largest premise for SemGauss, which claimed the repo was absent). None is plug-and-play. Each is a per-scene-trained method (no pretrained-weight gate except a downloadable seg-head for SemGauss/SNI) whose internal eval is **rigid-Horn ATE with no scale** — incompatible with the harness's Sim3, exactly the rigid-vs-Sim3 trap your metric rules warn about, so CRCD ATE must always route through [sim3_ate.py](DDS-SLAM/Addons/eval/sim3_ate.py).

The single biggest obstacle is **not** any method — it is the **harness contract itself**. [00_COMMON.md](DDS-SLAM/benchmarking/runbooks/00_COMMON.md) and [RUN_ALL.md](DDS-SLAM/benchmarking/runbooks/RUN_ALL.md) disagree on CLI signature, runner location, master entrypoint, method-id token, snippet casing, output-dir layout, seed loop, Depth-L1 definition, and exit codes. A `run_<method>.sh` built to one spec silently breaks the other. **Pick ONE spec (recommend 00_COMMON §6/§7 + the §0 resolved decisions win) before authoring anything.**

---

## 2. Readiness matrix

| Method | Class | can_repro | can_crcd_semantics | can_emit_harness_artifacts | overall_readiness |
|---|---|---|---|---|---|
| **SGS-SLAM** | 3DGS | YES | YES | PARTIAL (2 thin adapters) | NEEDS_WORK |
| **SNI-SLAM** | NeRF / tri-plane | PARTIAL | NO (needs CRCD seg-head trainer — absent) | PARTIAL (1 adapter + dir fix) | NEEDS_WORK |
| **SemGauss-SLAM** | 3DGS | PARTIAL | PARTIAL (seg-head trainer absent) | NO (2 adapters + seg trainer) | NEEDS_WORK |

Ordering top-to-bottom = least-to-most work. None ships READY; shipping any as READY would be dishonest.

---

## 3. The shared-harness fit problem

**Verdict (INFERENCE, grounded in the SHARED HARNESS PLUG facts):** A 3DGS/NeRF method can drive [sim3_ate.py](DDS-SLAM/Addons/eval/sim3_ate.py) + [eval_rendering.py](DDS-SLAM/Addons/eval/eval_rendering.py) + [generate_video.py](DDS-SLAM/Addons/viz/generate_video.py) **with zero harness edits**, but only if it emits exactly two plain-text/image artifacts in the right place:

1. `$OUT/demo/est_c2w_data.txt` — one pose/line, 12 space-separated floats (3×4 row-major c2w), translation at cols 3/7/11, ordered by frame index.
2. `$OUT/NNNN.jpg` — 4-digit, int-parseable stem, at the **OUT root** (not a subdir).

The aggregator [aggregate_ab.py](DDS-SLAM/Addons/eval/aggregate_ab.py) is the lone method-name-coupled script (hardcoded CRCD/SemSup cell lists). Per-run evals do not need it — run the two eval scripts directly and read stdout.

**None of the three methods emits either artifact natively.** All store poses inside a `params.npz` / ckpt `.tar` (quaternion + translation, rigid w2c relative to frame 0), and all write renders to a subdir with a non-int-parseable `gs_NNNN.png` stem, strided by an `eval_every` (5 by default). Concretely:

| Method | Trajectory native? | Render native? | Adapter required |
|---|---|---|---|
| **SemGauss** | NO — `params.npz` `cam_unnorm_rots`+`cam_trans` (FACT: `common_utils.py:34-41`, `sem_gauss.py:115-118`, decode `eval_utils.py:464-468`) | NO — `eval_dir/rendered_rgb/gs_{:04d}.png`, every 5 frames (FACT: `eval_utils.py:418,282`) | npz→est extractor + render rename + `eval_every=1` |
| **SGS** | NO — `params.npz` `cam_unnorm_rots` (quat `[w,x,y,z]`, unnormalized) + `cam_trans` (FACT: `scripts/slam.py:160-163`, decode `utils/eval_helpers.py:759-763`) | NO — `eval/rendered_rgb/gs_{:04d}.png`, `eval_every=5` (FACT: `eval_helpers.py:633,724`; `configs/replica/slam.py:30`) | `sgsslam_npz_to_est.py` + render rename + `eval_every=1` |
| **SNI** | NO — ckpt `.tar` `estimate_c2w_list` (FACT: `src/utils/Logger.py:50-65`, `SNI_SLAM.py:70-71`) | NO — native viz is a multi-panel debug montage `{idx:05d}_{iter:04d}.jpg` (FACT: `src/utils/Frame_Visualizer.py:109-160`) | trajectory exporter + point harness at the `rendered/` subdir |

**SNI is the closest fit:** it already ships [render_all_frames_sni.py](sni-slam/Addons/viz/render_all_frames_sni.py) which writes DDS-named `rendered/{idx:04d}.jpg` + `rendered/depth/{idx:04d}.png` uint16 (FACT: `render_all_frames_sni.py:234-243`) — but to a `rendered/` **subdir**, so the harness must be pointed there (or `--output_dir` set to the OUT root). SNI's trajectory exporter (`sni_export_traj.py`), which the run book claimed exists, is **genuinely absent** in both repos and must be authored.

**Important refutation (do not chase this):** the earlier claim that SNI's `sni_export_traj.py` "does not exist" was itself wrong for the SNI repo at large — the file **does exist, is git-tracked and fully implemented** (it loads the ckpt, flattens `estimate_c2w_list` row-major, writes 16-float lines that `sim3_ate.load_est` accepts at cols 3/7/11, and includes the OQ-10 sparse-pose guard). It writes 16 floats/line not 12, but the consumer accepts both. Treat SNI's trajectory export as **present**, not a blocker. (The SemGauss/SGS npz→est extractors, by contrast, are confirmed absent and required.)

---

## 4. Per method

### 4.1 SGS-SLAM (3DGS, ECCV 2024) — the strongest fit

**Repro (YES):** Replica-with-GT-semantics is downloadable (FACT: `README.md:83` Dropbox, license-click gated); trains per-scene from scratch, no weights needed; in-repo eval emits all 6 paper metrics (PSNR/MS-SSIM/LPIPS/Depth-L1-cm/ATE-RMSE-cm/mIoU via `utils/eval_helpers.py::eval`). A numeric gate target exists (Replica Avg ATE 0.41cm / PSNR 34.66 / SSIM 0.973 / LPIPS 0.096 / Depth-L1 0.36cm / mIoU 92.72 — UNVERIFIABLE from repo, INFERENCE: plausible SplaTAM-class).

**CRCD semantics (YES — best of the three):** semantics are GT-driven per-Gaussian colors with **no learned predictor** (FACT: `scripts/slam.py:117-120,152-157`). Feeding CRCD's 4-class GT masks as `semantic_ids/` (int64 id PNG) + `semantic_colors/` (4-color LUT) is mechanically identical to the Replica path; `num_semantic_classes=4` is benign metadata (the model regresses a 3-ch color, not N-way logits). This cleanly satisfies §0 Decision 3 for a GT-mask method.

**Confirmed gaps:**

| Gap | Sev | Effort | Evidence (FACT) |
|---|---|---|---|
| `run_sgsslam.sh` wrapper does not exist anywhere | BLOCKER | HIGH | Glob `SGS-SLAM/**/*.sh` → none; no `Addons/colab/`; `README.md:159-162` only documents bare `python scripts/slam.py` |
| Trajectory adapter `sgsslam_npz_to_est.py` absent | BLOCKER | LOW | `find *npz*est*` → 0; poses only in `params.npz` via `common_utils.py:42-43`; decode recipe `eval_helpers.py:759-763`; quat order `[w,x,y,z]` `gs_external.py:29-32` |
| Render rename + `eval_every=1` (renders → `eval/rendered_rgb/gs_NNNN.png`, strided) | MAJOR | LOW | `eval_helpers.py:724` (gs_ stem, subdir); `scripts/slam.py:1090` + `configs/replica/slam.py:30` (eval_every=5). NOTE the harder break is `generate_video.py:306` default `[0-9]*.png` → **zero match** on `gs_*`; `eval_rendering.py:98` int(stem) fails → misaligning positional fallback (`:126,132-134,170`), not a crash |
| Depth-L1 metric: §0 Decision 1 = input-vs-output via `Addons/eval/depth_l1.py` — absent; SGS saves rendered depth only as JET colormap (no uint16/.npy) so a source-touch is needed to dump raw depth | MAJOR | MED | `00_COMMON.md:3-7`; SGS repo has no `Addons/`; `eval_helpers.py:721-725` COLORMAP_JET PNG; runbook headlines a THIRD def (stereo-SGBM, cm) at `SGS-SLAM.md:463-464,493` contradicting §0 (mm) |
| CRCD dataset+config not authored (`CRCDDataset` + 2 registration edits + 5 `configs/crcd/*.py` + `configs/data/crcd.yaml`) | MAJOR | MED | `scripts/slam.py:47-71` get_dataset has 11 elif + ValueError, no CRCD; `__init__.py:1-13` 12 loaders, no CRCD; `configs/data/` = replica.yaml+scannet.yaml only; template uses PLURAL `semantic_ids/semantic_colors` (`replica.py:51-52`) |
| 4-class semantic emission not authored (`semantic_ids/` 4-class remap + `semantic_colors/` LUT + `np.unique=={0,1,2,3}` assert) | MAJOR | MED | consumed at `basedataset.py:51-52`, `slam.py:117-120,152-157`; current [preprocess_crcd_published.py](DDS-SLAM/Addons/preprocess/preprocess_crcd_published.py) only writes `semantic_class/` (`:142-143` clip 0..255), no `semantic_ids/`/`semantic_colors/`/assert. Heavy lifting (NEAREST rectify, coco_id+1 scheme `:132-139`) already present — work is rename+LUT+assert |
| Env: own clone+env, rasterizer pin `cb65e4b` absent on disk, must compile for sm_80 | MAJOR | MED | `environment.yml:7-11` (py3.10/torch1.12.1/cu11.6) vs `README.md:70-73` (py3.9/torch2.0.1/cu11.8) — **two conflicting specs**; rasterizer installed via pip VCS (`requirements.txt:17`, `environment.yml:29` has a malformed `/tree/` URL needing rewrite to `.git@<sha>`); hard import at `scripts/slam.py:44` etc. INFERENCE: prefer the README torch2.0.1/cu11.8 path — more likely to compile on A100 |
| Phase-A gate tolerance unset at contract level | MAJOR (mitigated) | LOW | `00_COMMON.md:919-926` (Q6: gate "cannot be enforced while its threshold is undefined — escalate"); runbook proposes its own at `SGS-SLAM.md:200-210` but unratified; it is a **documented open BLOCKING escalation**, not a silent hole |
| `aggregate_ab.py` hardcoded to DDS cells | MINOR | LOW | aggregator_contract; per-run eval works without it |

### 4.2 SNI-SLAM (NeRF / tri-plane, CVPR 2024, IRMVLab) — further along than the run book implies

**Repro (PARTIAL):** gate target fully defined and verified (FACT: [scripts/PAPER_BASELINES.md](sni-slam/scripts/PAPER_BASELINES.md) gives the exact 8-scene avg — Depth-L1 0.766, ATE-RMSE 0.456, etc.); env exactly pinned (`environment.yaml`); data+weights public on Drive folder `1BCu8bCGKG9HmnLFbyx7DIHI0slgkeo4h`. **But** nothing is bundled (no `data/`, no `seg/`, no `.pth`/`.pkl` present), full 8-scene needs Habitat-Sim, and the numeric tolerance is still an unset [USER] item.

**CRCD semantics (NO — a CRCD-trained seg-head is REQUIRED; the GT-mask path does not avoid it):**

> **CORRECTED 2026-06-17 (user-flagged, re-verified against code).** My first pass framed this as an optional "accept GT-mask path" vs "retrain head" choice. That was wrong. The GT-mask path **still fuses a Replica-trained feature** — `use_gt_semantic` only swaps the *label*, not the *feature*. (FACT, [Mapper.py:471-485](sni-slam/src/Mapper.py#L471-L485)): the semantic FEATURE `sem_feat` is computed UNCONDITIONALLY via `model_manager.set_mode_feature()` + `cnn(frame_rgb)` ([:474](sni-slam/src/Mapper.py#L474)), passed to `optimize_mapping`, and stored per keyframe ([:501](sni-slam/src/Mapper.py#L501)) — it IS the semantic signal SNI fuses/renders. `use_gt_semantic:True` only sets `gt_sem_label = gt_semantic` (the LABEL, [:476-477](sni-slam/src/Mapper.py#L476-L477)). In `'mapping'` mode `DINO2SEG.forward` runs `segmentation_conv[0..2]` = Upsample → **Conv2d(768→16)** → Upsample ([dinov2_seg.py:82-105](sni-slam/src/networks/dinov2_seg.py#L82-L105)); the 768→16 conv is `n_classes`-independent, so it is NOT among the shape-mismatched keys dropped at [model_manager.py:53-56](sni-slam/src/networks/model_manager.py#L53-L56) → it loads its **Replica-trained weights** intact, on top of the Replica-fine-tuned DINOv2 backbone, under `torch.no_grad()`, never optimized → frozen Replica-domain feature, baked in. Only the FINAL conv `[3]` (16→n_class) is dead. So the config comment ([crcd_sni_base.yaml:31-33](sni-slam/configs/CRCD/crcd_sni_base.yaml#L31-L33)) calling the Replica head "unused dead code under use_gt_semantic" is WRONG for the feature path.

**Therefore §0 Decision #3 ("retrain the seg-head on CRCD's 4 classes; never reuse the Replica indoor head", [00_COMMON.md:11-13](DDS-SLAM/benchmarking/runbooks/00_COMMON.md)) is REQUIRED, not optional.** Fix: author a `DINO2SEG` training loop — freeze backbone blocks 0-3, train blocks 4+ + `segmentation_conv` on CRCD `(RGB, 4-class GT mask)` with cross-entropy (`DINO2SEG(mode='train')` already emits the class logits, [dinov2_seg.py:103-108](sni-slam/src/networks/dinov2_seg.py#L103-L108)) → save `dinov2_crcd.pth` → repoint `model.cnn.pretrained_model_path`. **No training code exists in-repo** (load-only at [model_manager.py:43](sni-slam/src/networks/model_manager.py#L43); `eval_segmentation.py` is eval-only). This is the **same missing capability SemGauss needs (§4.3)** → author ONE shared DINOv2-seg-head trainer for both methods.

**Other confirmed gaps:**

| Gap | Sev | Effort | Evidence (FACT) |
|---|---|---|---|
| CRCD seg-head trainer absent — `sem_feat` is ALWAYS Replica-trained even under `use_gt_semantic` (above) | BLOCKER | HIGH | `Mapper.py:474,501`; `dinov2_seg.py:82-105,103-108`; `model_manager.py:43,53-56`; `00_COMMON.md:11-13`. Shared with SemGauss §4.3 |
| `run_sni_slam.sh` wrapper absent + the two-doc CLI conflict | BLOCKER | MED | no wrapper in `sni-slam/` or `DDS-SLAM/Addons/colab/` |
| Render/depth go to a `rendered/` SUBDIR not `$OUT` root | MAJOR | LOW | `render_all_frames_sni.py:13,22-23,118` (out=`cfg.output/rendered`); harness globs at root |
| Repro data+weights not bundled; full 8-scene needs Habitat-Sim | MAJOR | MED | `data/`+`seg/` absent; `README.md:32-33` Drive; `PAPER_BASELINES.md` notes 8-scene generation |
| Phase-A gate tolerance unset | MAJOR | LOW | `00_COMMON.md` Q6; `PAPER_BASELINES.md` supplies targets |
| 3 missing CRCD configs (e3_005=265, c3_001=1527, g3_001=1987) | MAJOR | MED | `configs/CRCD/` has c1/c2/f1/f3 only; c1_001 bound `[[-0.08,0.13],[-0.02,0.18],[0.68,0.90]]` is the template |
| Periodic ~100-frame stereo rescaling not implemented + trunc-only sc_factor bug | MAJOR | MED | §0 Decision #2; SNI has no `data.sc_factor` field |
| Depth-L1 `Addons/eval/depth_l1.py` to author (§0 Decision #1) | MAJOR | LOW | render_all_frames already emits rendered depth in needed format |
| `aggregate_ab.py` hardcoded → need `aggregate_crcd_generic.py` | MINOR | LOW | aggregator_contract |
| Env: harness env needs explicit `lpips` + MoGe-2 | MINOR | LOW | colab_setup installs neither; LPIPS silently skipped if import fails |
| VRAM floor unconfirmed (non-3DGS → warn-and-continue) | MINOR | LOW | CPU-offload defaults `SNI-SLAM.yaml:4-5` |
| H=720 not a multiple of patch-14 → DINOv2 reshape risk | MINOR | LOW | backbone feature runs even under `use_gt_semantic:True` (`Mapper.py:471-481`) |

Note: the run book's "must author `src/utils/datasets_crcd.py`" and "must author CRCD dataset class" are **STALE/WRONG** — the CRCD loader already exists at `src/utils/datasets.py:324-385` (registered in `dataset_dict`). Also the hardcoded weight path is the **relative** `seg/dinov2_replica.pth` (`configs/SNI-SLAM.yaml:95`), not the absolute `/data0/...` the run book claims.

### 4.3 SemGauss-SLAM (3DGS, IROS 2025) — the most work

**Repro (PARTIAL):** code+configs present (FACT: `sem_gauss.py:989-996` real entry, `configs/replica/replica.py`, vendored rasterizer dir all exist — **refuting the run book's central BLOCKER 1**); data + per-scene `dinov2_replica.pth` + GT mesh on one author Drive (`README.md:74,81,89`), ScanNet TOS-gated. But **no Table-I PDF numbers exist** so the gate has no enforceable target, and the cu116-rasterizer-on-CUDA12-host build is a real feasibility risk.

**CRCD semantics (PARTIAL):** semantics need a per-scene pretrained DINOv2 seg-head that emits both 16-d features (L_f) and N-class logits (L_s). The repo only **loads** a `.pth` (`dinov2_seg.py:151`); there is **no training code** (grep for seg_net optimizer/backward/save = 0). Critically, the L_f feature target is unavoidable even when `use_gt_semantic=True` (FACT: `sem_gauss.py:600` assigns `se_fe` from the seg-head in the GT branch; L_f at `:265/271/277`) — so a CRCD-fit `.pth` is mandatory and must be trained from scratch.

**Confirmed gaps:**

| Gap | Sev | Effort | Evidence (FACT) |
|---|---|---|---|
| No seg-head TRAINING code for CRCD 4-class adaptation | BLOCKER | HIGH | `dinov2_seg.py:151` load-only; `sem_gauss.py:447` Segmentation() runs unconditionally at startup; `:600` se_fe assigned in GT branch; no CRCD config/n_classes=4 (only `replica.py:45`/`scannet.py:45`) |
| Trajectory adapter (npz `cam_unnorm_rots`+`cam_trans` → `est_c2w_data.txt`) | BLOCKER | MED | grep est_c2w_data = 0; `common_utils.py:34-41`; per-frame storage `sem_gauss.py:115-118`; decode `eval_utils.py:464-468`. NOTE poses are per-frame → no keyframe expansion needed |
| CRCD dataset loader absent + dispatcher rejects unknown names | BLOCKER | MED | `sem_gauss.py:37-43` get_dataset → ValueError else; `__init__.py:4-5` exports only Replica/Scannet. Fix needs loader **+ export + elif branch** (not just the subclass) |
| Per-method wrapper + master orchestrator absent; THREE incompatible CLI contracts | BLOCKER | MED | `find` → none; `SemGauss-SLAM.md:21,462` (BLOCKER 2); 00_COMMON §7.1 (2-arg) vs RUN_ALL §3 (4-arg) vs SemGauss §7.1 (3-arg `run_semgauss_slam.sh`, exit 78 not 40) |
| Repro gate has no enforceable target (Table-I `<from PDF>`, tolerances TBD) | HIGH (downgraded from BLOCKER) | LOW | `SemGauss-SLAM.md:263-268`; `00_COMMON.md:919-926` (Q6). NOT a pipeline blocker — orchestrator runs the gate REPORT-ONLY when tolerance unset (`RUN_ALL.md:108,310,318`); and Phase A is already blocked upstream by the absent wrapper |
| Replica data + `.pth` + mesh on one Drive; hardcoded `/data0/...` must be repointed; `.pth` is a hard Phase-A dependency | MAJOR | MED | `dinov2_seg.py:151` no fallback; `replica.py:45,53` `/data0/...`; `README.md:74,81,89,85`; no vendored weights, no `data0/` tree |
| Missing DDS-side configs e3_005/c3_001/g3_001 | MAJOR | MED | `configs/CRCD/` has c1/c2/f1/f3 only; `00_COMMON.md:16,385` (Decision 4, "verified: zero configs"); frame counts match [crcd_depth_gen_remainder_20260616.sh](DDS-SLAM/Addons/colab/crcd_depth_gen_remainder_20260616.sh):31,34 |
| Depth-L1 script absent; SemGauss's own Depth-L1 is rendered-vs-GT (different def, and CRCD has no GT depth) | MAJOR | MED | `00_COMMON.md:7` (input-vs-output, mm); no `depth_l1.py`; `eval_utils.py:350,357-359` |

Note: the run book's claim that `requirements.txt` is pip-installable is wrong — it is a **conda explicit-spec file** (header `conda create --file`). But this was DROPPED as a standalone gap because the run book's own §3.2 already prescribes the correct conda-then-pip sequence; it is a NIT, not MAJOR. Similarly the render-format/`Rendered>=100`-gate gap was **refuted** (no such 100-frame gate exists in `eval_rendering.py`, and the path is not currently wired).

---

## 5. Spec drift / contradictions in the run books — fix BEFORE onboarding

These are **harness-wide blockers independent of any method.** They survived adversarial verification.

1. **CLI / output / exit-code contract is defined THREE incompatible ways** (FACT):
   - **00_COMMON §7.1** (`:774`): `run_<method>.sh <phase> [snippet]` (2 args); phase `env|repro|crcd|eval|all`; UPPERCASE snippets; dirs `<NAME>_s<SEED>` + `payload.tgz` containing `sim3_metrics.txt`+`render_metrics.txt`; mandatory n=3 seed loop; exit `0/10/20/30/40/1`; orchestrator `Addons/colab/run_all.sh`.
   - **RUN_ALL.md §3** (`:64`): `<REPO>/run_<method>.sh <stage> <leg> <out_dir> <gpu_tier>` (4 args); stage `repro|crcd` only; lowercase legs; caller-supplied `out_dir` (no `_s<seed>`); `metrics.json`+`status.txt` (no payload.tgz); no seed loop; ad-hoc exit codes (3/4/6/124); orchestrator `/content/bench/bench.sh`.
   - **Per-method §7** (e.g. `SemGauss-SLAM.md:466`): a third signature + exit 78 (VRAM).
   They disagree on arg count (2/3/4), phase vocabulary, runner location (`Addons/colab/` vs repo root), method-id token, snippet casing, output container, seed-loop ownership, exit-code scheme, and orchestrator name.

2. **Depth-L1 defined THREE different ways** (FACT): §0 Decision 1 = input-vs-output (mm) via `Addons/eval/depth_l1.py`; §5.3 still emits literal BLOCKED; RUN_ALL §7 + SGS-SLAM §6.3 = rendered-vs-stereo-SGBM (cm) via `depth_l1_crcd.py`. RUN_ALL.md is even **internally** self-contradictory: its header (`:1`) imports §0's input-vs-output as overriding while its body (`:160,187`) defines stereo-SGBM. **§0 wins** (it explicitly supersedes, `00_COMMON.md:5`).

3. **Hallucinated absolute Mac path** (FACT): `RUN_ALL.md:526` cites `/Users/benwright/Desktop/DDS-SLAM-BEN/...` for "reused as-is" scripts — fabricated; the real user is `benli` (Windows) and the harness root is `/content/DDS-SLAM`. Would break every cited reuse.

4. **Filename mismatches** (FACT): 00_COMMON cites `10_DDS-SLAM.md / 20_SemGauss-SLAM.md / 30_SGS-SLAM.md / 40_SNI-SLAM.md`; actual files are `SemGauss-SLAM.md`, `SGS-SLAM.md`, `SNI-SLAM.md`, `SemanticSuPer.md` (no numeric prefixes, **no `10_DDS-SLAM.md` at all**).

5. **Calib pickle path specified THREE ways** (FACT): canonical `.../CRCD-Published/cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl` vs `SGS-SLAM.md:239` operator path `C:/Users/benli/sam3facebook/...` staged to `_calib/` (not `cam_calib/`). Unify before staging.

6. **SGS-SLAM.md §1.1 falsely claims `00_COMMON.md` does not exist** (FACT) and tells the agent to treat its own §7 as authoritative — so the SGS agent would never consult the canonical spec. Must be repaired.

7. **§0 resolved-decision vs stale-escalation drift** (FACT): §0 (2026-06-17) resolves Depth-L1, stereo cadence, semantics, and missing configs, but §4.3/§4.4/§5.3/§9 and all of RUN_ALL still encode the pre-resolution "escalate/BLOCKED/frame-0/DDS-only" state. The stale text must be deleted or the authoritative header re-asserted at each call site.

**Recommendation (INFERENCE):** adopt **00_COMMON §6/§7 + the §0 resolved decisions** as canonical; rewrite RUN_ALL.md + `bench.sh` to match; fix SGS-SLAM.md §1.1; delete the hallucinated Mac path; unify the calib path. Do this first — it unblocks all three wrappers at once.

---

## 6. Ranked build order — minimal critical path to a runnable 3-method benchmark

Easiest-win first. Effort is rough (LOW ≈ hours, MED ≈ 1-2 days, HIGH ≈ days).

**Phase 0 — harness unification (blocks all wrappers; do once)**
- 0a. Reconcile CLI/output/exit/Depth-L1 to ONE spec (00_COMMON §6/§7 + §0 win); rewrite RUN_ALL.md/bench.sh; fix SGS-SLAM.md §1.1; remove hallucinated Mac path; unify calib path. — **MED**
- 0b. Author shared `Addons/eval/depth_l1.py` (§0 Decision 1, input-vs-output, mm) — reused by all three. — **LOW**
- 0c. Author ONE shared DINOv2-seg-head trainer (CRCD 4-class) — `dinov2_crcd.pth` for SNI (`DINO2SEG`) and the SemGauss head; required by both (§4.2/§4.3), not a per-method afterthought. — **HIGH**
- 0d. Escalate the unresolved [USER] items together: gate tolerances (per method), stereo cadence, VRAM floors. — **LOW**

**Phase 1 — SGS-SLAM (least work, clean GT-mask fit → fastest first metric)**
1. Clone + env (prefer README torch2.0.1/cu11.8 path); fetch+build rasterizer `cb65e4b` for sm_80 (rewrite the `/tree/` URL). — **MED**
2. Phase-A repro on room0+office0; gate vs `PAPER_BASELINES`-class numbers at ratified tolerance. — **MED**
3. Author the two thin adapters: `sgsslam_npz_to_est.py` (npz→est) + render renamer (`gs_NNNN.png`→`$OUT/NNNN.png`, `eval_every=1`). Add the rendered-depth-raw `.npy` dump (source-touch, exceeds the 2 registration edits → covered by source-edit escalation). — **LOW**
4. `CRCDDataset` + 2 registration edits + 5 `configs/crcd/*.py` + `configs/data/crcd.yaml`; extend `preprocess_crcd_published.py` to emit `semantic_ids/`+`semantic_colors/` with `np.unique=={0,1,2,3}` assert. — **MED**
5. `run_sgsslam.sh` to the unified spec (seed loop, payload.tgz, `.DONE` last). — **HIGH**

**Phase 2 — SNI-SLAM (loader already exists; resolve the semantic conflict first)**
6. Train the CRCD `DINO2SEG` head (Phase 0c shared trainer) → `dinov2_crcd.pth`; repoint `model.cnn.pretrained_model_path`. REQUIRED — the GT-mask path still fuses a Replica-domain `sem_feat` (§4.2). — **HIGH (shared with SemGauss)**
7. Build `sni` env + harness env (+`lpips`, MoGe-2); download Drive subset into `./data/replica`+`./seg`. — **MED**
8. Phase-A repro; gate vs `PAPER_BASELINES.md` 8-scene avg. — **MED**
9. Point harness at the `rendered/` subdir (or set `render_all_frames_sni.py --output_dir` to OUT root); author 3 missing configs; apply patch-14 H-crop. SNI's trajectory exporter already exists — verify, don't re-author. — **MED**
10. `run_sni_slam.sh` to the unified spec. — **MED**

**Phase 3 — SemGauss-SLAM (hardest; needs a from-scratch seg-head trainer)**
11. Build the `sem_gauss` conda env; smoke-test the cu116 rasterizer **before** Phase A (STOP/escalate on fail). — **HIGH**
12. Download author Drive (Replica + `dinov2_replica.pth` + mesh); repoint `/data0/...`; escalate for Table-I numbers + tolerances. — **MED**
13. Phase-A repro; gate (REPORT-ONLY until tolerance ratified). — **MED**
14. **Author the seg-head training script** (freeze `dinov2_vitb14`, train DINO2SEG 16-d feature + 4-class head on CRCD GT masks → `dinov2_crcd.pth`). — **HIGH**
15. `CRCDGradSLAMDataset` + export + dispatcher elif; CRCD config; trajectory + render adapters; 3 missing configs. — **MED**
16. `run_semgauss_slam.sh` to the unified spec. — **HIGH**

**Phase 4 — aggregate**
17. Author `Addons/eval/aggregate_crcd_generic.py` (or extend `aggregate_ab.py` cell lists) → `_render_summary.csv` + `COMBINED_SUMMARY.txt` (mean±std over n=3). — **LOW**

**Critical-path rationale (INFERENCE):** SGS-SLAM reaches a first real CRCD metric fastest because its semantics need **no learned head** (GT masks fed directly) — its only true blockers are a wrapper + two LOW-effort adapters. SNI and SemGauss BOTH require a CRCD-trained DINOv2 seg-head (no training code exists in either repo); building it once (Phase 0c) serves both. SNI is then second (CRCD loader + renderer already shipped, only adapters + the head left); SemGauss is last (it additionally needs trajectory/render adapters, a CRCD `gradslam` loader + dispatcher edits, and a riskier cu116-on-A100 rasterizer build). The `depth_l1.py` and the contract unification (Phase 0) are shared prerequisites — doing them once amortizes across all three. Headline render PSNR/SSIM/LPIPS + Depth-L1 for every snippet; quote Sim3 ATE + path-ratio + |Pearson| only for C2_001 (the sub-SNR rules make the other CRCD snippets reject-for-tracker, keep-for-render).
