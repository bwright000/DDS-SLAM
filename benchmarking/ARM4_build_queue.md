# Arm 4 — Benchmark onboarding: SemGauss-SLAM / SGS-SLAM / SNI-SLAM on CRCD

> **Source of truth:** [AUDIT_onboarding_3methods_20260617.md](AUDIT_onboarding_3methods_20260617.md) (36-agent audit, claim-verified, adversarially refuted) + the SNI seg-head re-verification (2026-06-17). Effort: **LOW ≈ hours · MED ≈ 1–2 days · HIGH ≈ days.**
>
> **Goal of Arm 4:** a one-line-runnable multi-method CRCD semantic-SLAM benchmark — metrics `{Sim3 ATE, PSNR, SSIM, LPIPS, Depth-L1}` + canonical 6-panel video, **n=3 seeds**, every method vs DDS-SLAM on the 5 snippets `C1_001 C2_001 E3_005 C3_001 G3_001`.
>
> **Metric law (carry through every leg):** headline render PSNR/SSIM/LPIPS + Depth-L1 for all snippets; quote **Sim3 ATE only via `sim3_ate.py`** (+ recovered scale + est/GT path-ratio + |Pearson|dom), and headline tracking quality **only on C2_001** — the others are sub-SNR (reject-for-tracker, keep-for-render). **Never headline rigid ATE.**

---

## STAGE 0 — shared foundations (block all three methods; do once)

- [x] **A4-0.1 — Unify the harness contract.** ✅ **DONE 2026-06-17.** Authored [`runbooks/CONTRACT.md`](runbooks/CONTRACT.md) = single source of truth (CLI `<phase> [snippet]`; output `<NAME>_s<SEED>`; files `render_eval.{csv,txt}`+`sim3_metrics.txt`+`depth_l1.txt`; exit `0/10/20/30/40/1`; aggregator `aggregate_crcd_generic.py`; verified EXISTS/TO-BUILD inventory §7). Fixed SGS-SLAM.md §1.1, calib `_calib/`→`cam_calib/`, RUN_ALL.md Mac path; added supersede-banners to RUN_ALL.md + 00_COMMON.md. *Deferred:* full `bench.sh`/`run_all.sh` rewrite (needs the not-yet-built wrappers).
- [x] **A4-0.2 — Shared Depth-L1.** ✅ **DONE 2026-06-17.** Authored [`Addons/eval/depth_l1.py`](../Addons/eval/depth_l1.py) (input-vs-output mm; frame-id pairing, NEAREST resize, valid-pixel mask, mean/median + per-frame CSV; greppable summary). Parses clean; full run pending Colab data. Aggregator must be extended to surface the column (A4-4.1).
- [x] **A4-0.3 — Shared DINOv2 4-class seg-head trainer.** ✅ **CODE DONE 2026-06-17** ([`Addons/seg/train_dinov2_crcd.py`](../Addons/seg/train_dinov2_crcd.py)) — freezes DINOv2 vitb14 blocks 0-3, trains 4+ + `segmentation_conv` on CRCD `(video_frames/*l.png, semantic_class)` with CE; backbone init from official DINOv2 (torch.hub) — REQUIRED, aborts on init fail; per-class + mean IoU; asserts labels ⊆{0,1,2,3}; saves `dinov2_crcd.pth`. Parses clean. *Run pending Colab + staged CRCD.*
  - ✅ **HEAD-SHAPE RECONCILIATION RESOLVED (this turn, code-verified):** SNI `DINO2SEG` (`sni-slam/src/networks/dinov2_seg.py`) and SemGauss `DINO2SEG` (`SemGauss-SLAM/utils/dinov2_seg.py`) are the **SAME class** (dinov2_vitb14, blocks-4+-trainable, `segmentation_conv` 768→16→n_cls). c_dim=16 in BOTH; the only diffs (Upsample sizes, mode-enum names, crop_edge) are **parameter-free**. → **ONE `dinov2_crcd.pth` loads strict into SemGauss `Segmentation.get_dinov2` AND (strict=False, 0 dropped) into SNI `ModelManager.get_dinov2`** — given both CRCD configs use **n_classes=4, c_dim=16, crop_edge=0** (SNI already does; SemGauss CRCD config in A4-3.5 must match). This also FIXES the SNI Replica-feature bug (AUDIT §4.2). **A4-3.4 collapses to "point `pretrained_model_path` at the shared `.pth`" — no SemGauss-specific trainer needed.** SGS uses GT masks, no head (confirmed: zero seg/dino files in SGS-SLAM).
- [~] **A4-0.4 — Periodic stereo depth → DELEGATED to another agent (2026-06-18); field-fix DROPPED.** Another agent generates the CRCD **left-frame depth maps via stereo calibration every ~100 frames** (= §0 Decision 2). **Not built here.** ✅ **Depth is METRIC-SCALED (user-confirmed 2026-06-18) → `sc_factor = 1.0` everywhere → the `range_d/near/far/depth_trunc` field-scaling bug is MOOT; no DDS model-code edit, no `patch_sc_factor.py`.** Bonus: metric depth ≈ GT metric scale, so Sim3 recovered-scale/path-ratio should sit ≈1 (cleaner ATE than the old ~8× MoGe mismatch). — **EXTERNAL**
  - *Consume interface the benchmark needs (confirm exact dir/scale with that agent):* per-snippet left-frame depth `data/CRCD/<NAME>/depth/<fid>.png` (uint16, value = depth_m × scale). Each method's `png_depth_scale` (or equivalent) must equal that scale; `depth_l1.py` runs with `--input_scale <scale> --sc_factor 1.0`; per-method config `sc_factor`/`data.sc_factor` = `1.0`.
- [~] **A4-0.5 — Bound-derivation tool ✅ CODE DONE; configs pending depth.** Authored [`Addons/preprocess/derive_crcd_bounds.py`](../Addons/preprocess/derive_crcd_bounds.py) (pinned recipe 00_COMMON §4.0(b): frame-0 metric depth → per-axis `[P1−M, P99+M]`, M=0.02 m; `marching_cubes_bound = bound`; OpenCV back-projection). Parses + synthetic smoke test correct (flat 0.05 m → Z=[0.03,0.07]). *Pending the delegated metric depth (A4-0.4):* run it for each SNI snippet. 🚨 **Re-derive ALL SNI bounds (c1/c2 too — the committed ones are stale up-to-scale MoGe values).** Depth-gen itself is the other agent's job now (not ours). SGS/SemGauss (3DGS) need **no** bound — only timesteps/datadir/intrinsics + `png_depth_scale`=delegated scale. — **MED**
- [ ] **A4-0.6 — Escalate open [USER] gates.** Repro gate tolerances per method, periodic-vs-frame0 stereo cadence sign-off, per-method VRAM floors. Until answered: gate = report-only. — **LOW**

> **Parallelism:** SGS (Stage 1) needs **0.1/0.2/0.4/0.5** but **NOT 0.3** (it feeds GT masks directly). So start the 0.3 seg-head trainer in the background and drive SGS to the first real CRCD metric in parallel.

---

## STAGE 1 — SGS-SLAM (3DGS, ECCV 2024) — fastest first metric, no seg-head needed

- [ ] **A4-1.1 — Env + rasterizer.** Clone; build env (prefer README torch 2.0.1 / cu11.8); fetch + compile diff-gauss rasterizer pin `cb65e4b` for sm_80 (rewrite the malformed `/tree/` URL → `.git@<sha>`). — **MED**
- [ ] **A4-1.2 — Phase-A repro + gate.** Replica (GT-semantics) room0+office0; compare to paper Avg (ATE 0.41 cm / PSNR 34.66 / SSIM 0.973 / LPIPS 0.096 / Depth-L1 0.36 cm / mIoU 92.72) at the A4-0.6 tolerance. — **MED**
- [ ] **A4-1.3 — Harness adapters.** `sgsslam_npz_to_est.py` (`params.npz` `cam_unnorm_rots`[w,x,y,z]+`cam_trans` → `est_c2w_data.txt`, 12-float c2w rows); render rename `gs_NNNN.png` → `$OUT/NNNN.png` with `eval_every=1`; add a raw rendered-depth `.npy/uint16` dump (source-touch — current is JET colormap only). — **LOW**
- [ ] **A4-1.4 — CRCD data path.** `CRCDDataset` + 2 registration edits (`get_dataset` elif + `__init__` export) + 5 `configs/crcd/*.py` + `configs/data/crcd.yaml`; extend `preprocess_crcd_published.py` to emit `semantic_ids/` (4-class) + `semantic_colors/` (4-colour LUT) with `np.unique=={0,1,2,3}` assert. — **MED**
- [ ] **A4-1.5 — `run_sgsslam.sh`.** To the A4-0.1 contract: seed loop {0,1,2}, `<NAME>_s<seed>` dirs, `payload.tgz` incl. `sim3_metrics.txt`+`render_metrics.txt`, VRAM floor abort, `.DONE` written last. — **HIGH**
- [ ] **A4-1.6 — Run + eval.** 5 snippets × n=3 → shared harness (Sim3 ATE, render metrics, Depth-L1, 6-panel video). — **—**

---

## STAGE 2 — SNI-SLAM (NeRF / tri-plane, CVPR 2024) — consumes the 0.3 seg-head

- [ ] **A4-2.1 — Env.** `sni` env (`environment.yaml`) + harness env explicit `lpips` + MoGe-2 (colab_setup installs neither). — **MED**
- [ ] **A4-2.2 — Phase-A repro + gate.** 8-scene Replica vs `scripts/PAPER_BASELINES.md` (Depth-L1 0.766 cm, ATE 0.456 cm, …) at tolerance. Needs Habitat-Sim for full 8-scene. — **MED**
- [ ] **A4-2.3 — CRCD seg-head (REQUIRED).** Point `model.cnn.pretrained_model_path` → `dinov2_crcd.pth` from **A4-0.3**. ⚠️ `use_gt_semantic:True` swaps only the *label*; the fused feature `sem_feat` ([Mapper.py:474,501](../../sni-slam/src/Mapper.py)) is ALWAYS the seg-head's output → the Replica head must NOT be used. Verify `sem_feat` is now CRCD-domain. — **LOW** (given 0.3)
- [ ] **A4-2.4 — Harness adapters.** Renders/depth currently land in a `rendered/` **subdir** — set `render_all_frames_sni.py --output_dir` to the OUT root (or point the harness there); verify the existing trajectory exporter (`estimate_c2w_list` → 16-float rows, accepted by `sim3_ate.load_est`); apply patch-14 H-crop (720 not divisible by 14). — **MED**
- [ ] **A4-2.5 — Per-snippet configs.** 5 CRCD configs inheriting `crcd_sni_base.yaml`; bounds from A4-0.5. (CRCD loader already exists at `src/utils/datasets.py:324-385` — do NOT re-author.) — **MED**
- [ ] **A4-2.6 — `run_sni_slam.sh`.** To the A4-0.1 contract (seeds, dirs, payload, floors). — **MED**
- [ ] **A4-2.7 — Run + eval.** 5 × n=3 → shared harness. — **—**

---

## STAGE 3 — SemGauss-SLAM (3DGS, IROS 2025) — hardest; consumes 0.3 (adapted)

- [ ] **A4-3.1 — Env + rasterizer smoke.** Build `sem_gauss` conda env; smoke-test the vendored cu116 rasterizer on the A100 (CUDA 12 host) **before** Phase A — STOP/escalate on build fail (real feasibility risk). — **HIGH**
- [ ] **A4-3.2 — Repro data + weights.** Download author Drive (Replica + per-scene `dinov2_replica.pth` + GT mesh); repoint hardcoded `/data0/...`. — **MED**
- [ ] **A4-3.3 — Phase-A repro + gate.** Report-only until Table-I numbers + tolerance supplied (no PDF numbers in repo → A4-0.6 escalation). — **MED**
- [ ] **A4-3.4 — CRCD seg-head = REUSE A4-0.3 (collapsed to LOW).** No SemGauss-specific trainer: SemGauss's `DINO2SEG` == the one A4-0.3 trains. Just set `model.pretrained_model_path` → the shared `dinov2_crcd.pth` and the SemGauss CRCD config to `n_classes=4, c_dim=16, crop_edge=0` so `Segmentation.get_dinov2`'s `load_state_dict(strict=True)` matches. Its `set_mode_get_feature` (16-d `L_f`) + `get_semantic` (logits `L_s`) both then run on the CRCD-trained head. — **LOW**
- [ ] **A4-3.5 — CRCD data + adapters.** `CRCDGradSLAMDataset` + export + `get_dataset` elif; CRCD config (`n_classes=4`); trajectory adapter (`params.npz` → `est_c2w_data.txt`) + render rename; 5 per-snippet configs (bounds from A4-0.5). — **MED**
- [ ] **A4-3.6 — `run_semgauss_slam.sh`.** To the A4-0.1 contract; 3DGS VRAM floor = **hard abort** (≈24 GB). — **HIGH**
- [ ] **A4-3.7 — Run + eval.** 5 × n=3 → shared harness. — **—**

---

## STAGE 4 — aggregate

- [ ] **A4-4.1 — Aggregator: EXTEND (not author).** `Addons/eval/aggregate_crcd_generic.py` **already EXISTS** (method-agnostic; reads `<ROOT>/<NAME>/{render_eval.csv|txt, sim3_metrics.txt}`|`payload.tgz`; `--root`/`--method-roots`). Audit's "author" was stale. Remaining: extend it to (a) accept `--seeds 0 1 2` → read `<NAME>_s<SEED>` → per-snippet **over-seed** mean±std (currently it does cross-snippet mean±std, single-seed), and (b) parse `depth_l1.txt` → add the Depth-L1 column. — **LOW**
- [ ] **A4-4.2 — Final deliverable.** Cross-method table {Sim3 ATE, PSNR, SSIM, LPIPS, Depth-L1} × {3 methods + DDS-SLAM} × 5 snippets, mean±std; per-snippet 6-panel videos gathered. — **—**

---

## Critical path (one line)

**0.1 → (0.2/0.4/0.5 ∥ 0.3 trainer) → SGS (first metric) → SNI (uses 0.3) → SemGauss (uses 0.3-adapted) → aggregate.**
SGS needs no seg-head, so it produces the first real CRCD number while the shared seg-head trainer (0.3) runs in the background — that trainer then unblocks both SNI and SemGauss.
