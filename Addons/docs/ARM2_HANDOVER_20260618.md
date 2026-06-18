# ARM-2 HANDOVER — Deformation-field FIX (TEACHER-SUPERVISED build)

**Date:** 2026-06-18 · **Branch:** `diagnosis-live` · **Remote:** `bwright000/DDS-SLAM`

Arm-2 = revive DDS-SLAM's **dead deformation field**. This doc is self-contained: read it + the
memories in §9 and you can continue without re-deriving. Personality rules still apply (caveman-direct,
question the user, scientific rigour, **commit after each unique change**, metrics+video every result).

---

## 0. TL;DR — where it stands RIGHT NOW
- **Diagnosis is SETTLED** (§1): the field trains to **exactly |Δx|=0** (dead), from **starvation** (no dedicated supervision), NOT weight-decay. The CODE is faithful to the PAPER — the paper's field is starved too → the fix is **new science, not a bug-fix**.
- **Design is LOCKED by the user** (§2): **TEACHER-SUPERVISED** — bake a self-supervised target `Δx* = X₀ − X_k` (depth + DINO-correspondence), regress `TimeNet` toward it via one default-off loss; keep the MLP only as the dense off-surface interpolator.
- **Stage 0 (zero-GPU commitment gate) PASSED** (§4): the teacher recovers real pin motion **+82%, cos +0.92**, hollow-failure structurally ruled out. SemSup build = **GO**; CRCD viability still open.
- **BUILT so far:** the JUDGE, the Stage-0 gate, the target BAKER, the target VALIDATOR (§3).
- **IMMEDIATE NEXT STEP** (§5): bake the targets → validate they reproduce the gate → then write the **Stage-1 model loss** (NOT yet written; exact code in §6).

---

## 1. THE DIAGNOSIS (settled — do not relitigate)
- The deformation field = static canonical hash-map (t=0, **time-blind**) + a free-learned MLP **`TimeNet`** (`model/decoder.py:6-53`, 3-layer bias-free) producing an **additive** warp `Δx`, queried as `pts + vox_motion` in the canonical map (`model/scene_rep.py` ~195-231; anchor `where(t==0,0,Δx)` at :217-218).
- **It is DEAD:** proven on real deformation GT (SemSup green pins) via the field-warped pin EPE — trained field outputs **|Δx| = 0.00000** (mean & max). Settled 3 ways: render (field_off≈paper), pose (field_off≈field_on trajectory), and the pins.
- **Cause = STARVATION:** only the shared render/depth/sdf loss reaches the field (via the warped-coord chain rule); the time-blind map fits a per-frame time-average and **wins the gradient race** → field collapses to 0. **wd EXONERATED** (`wd0` cell collapsed identically).
- **Paper-vs-code:** the code is FAITHFUL. The paper has **5 render-only losses, NO Δx supervision**, and *deliberately forgoes* correspondence. Table III "field helps" is n=1, sub-SNR, **capacity-confounded** (never probes |Δx|). ⇒ a dedicated teacher is a **genuine methodological contribution**.
- **The "hollow" trap:** a non-zero **but wrong-direction** field made pins **2.7× WORSE** (`field_off` init |Δx|=0.028). ⇒ the teacher must produce **CORRECT** motion, not just non-zero |Δx|. The `cos>0` check guards this.

## 2. THE LOCKED DESIGN — TEACHER-SUPERVISED
- **Target:** `Δx* = X₀ − X_k` (observed-at-k → canonical-frame-0 **pull-back**), derived from **depth + DINO-feature correspondence**. This IS the user's "derive from depth+point-difference" intuition.
- **Keep `TimeNet`** as the dense **off-surface interpolator** — the renderer needs Δx at every ray sample, but the teacher only knows it at surface points; the MLP turns the sparse surface teacher into a dense field. (This is why we don't *replace* the MLP.)
- **One loss `def_sup`** regresses `TimeNet(X_k, t_k) → Δx*`, weighted by per-target `trust`. Default-off flag `deformation_sup_weight: 0` → base stays **bit-identical** (parity gate passes).
- **Tools:** CORE = depth, DINO-correspondence, seg (tissue-only validity), poses. Modifier = σ² (trust-weight, Stage 3). Deferred = optical flow (only if DINO too coarse). **CUT = SNI fusion, DINO-as-field-input** (they decorate a still-untaught MLP, don't address starvation). **NEVER train on the pins** (held-out judge).
- **Injection site:** `current_frame_mapping` (`ddsslam.py` ~352-377) FIRST — `indice_h, indice_w` are in scope there (it already samples `dino_grid`), so the target looks up by `(cur_frame_id, indice_h, indice_w)` with **zero keyframe-DB surgery**. Escalate to the dominant `global_BA` trainer only if Stage-1 reduction is positive-but-weak. ⚠️ **`global_BA` has NO pixel key** (`keyframe.py:33,44` discards `idxs`; `self.rays` = `[dir3,rgb3,depth1,edge1]`) → needs pixel-recovery from `rays_d_cam` (invert `get_camera_rays`).

## 3. WHAT'S BUILT (committed, branch `diagnosis-live`)
| File | Purpose | Key commits |
|---|---|---|
| `Addons/eval/field_warped_pin_epe.py` | **THE JUDGE** — the only field-sensitive metric. Warps pins through the field, reduction vs rigid + `cos(D,X₀−Xk)` direction + |Δx| + anchor + shuffle controls. Uses the model's OWN embedders (no convention drift). | f61f7c5, 9c27190 |
| `Addons/eval/teacher_vs_pin_epe.py` | **Stage-0 gate** — does DINO+depth correspondence recover pin motion (before any GPU). Pin gate (binned by magnitude, `--ref_stride` for small motion) + generic-tissue cycle. `--est_c2w` optional (identity default). | fb092ab, ff51f62 |
| `Addons/deform/generate_deform_targets.py` | **The BAKER** — dense `Δx*` targets at patch res, `<stem>_deform.npz {dx,valid,trust}`. Vectorised chunked match (OOM-fixed: `--chunk 64`). | f1209bf, 1b9d7ca |
| `Addons/deform/validate_deform_targets.py` | **Target VALIDATOR** — confirms baked Δx* reproduces the gate (+82%) before any model edit. | aca38a2 |
| `Addons/eval/gt_pins/trial_3_l_pts.npy` | **Pin GT** (committed; data/ is gitignored). dict['gt'], 151 frames × 32 pins (u,v,valid). | — |

## 4. STAGE-0 RESULT (the gate that greenlit the build)
Run on SemSup trail_3 (DINO vits14 baked, MoGe depth, identity poses):
- **PIN gate, cumulative (ref=0 — what the field learns):** rigid 0.01254 → teacher 0.00221, **reduction +82.4%, cos +0.92**. Holds on ALL magnitude bins incl **small/CRCD-scale +74.1%, cos +0.89**.
- **Stress (ref=k−3, tiny per-frame motion):** overall +77%, but the smallest bin (|gt|<0.0015) degrades to **+6.5%, cos +0.59** — the resolution floor. **cos stays POSITIVE ⇒ noisy-not-wrong ⇒ hollow failure structurally avoided.**
- **Generic non-pin tissue (GT-free cycle):** 2–4.5 px cycle, Lowe ratio median 0.66–0.71 ⇒ DINO discriminative on plain tissue, generalises off the salient pins.
- **Verdict:** SemSup build **GO**. CRCD viability **open** (no pins → judge later via generic-cycle probe + STIR).

## 5. IMMEDIATE NEXT STEP (the open task)
The user chose **"validate baked targets first."** On Colab (SemSup staged, DINO baked):
```bash
# 1. bake the dense Δx* targets (pure numpy, no GPU, a few min):
python Addons/deform/generate_deform_targets.py \
  --dino_dir data/Super/trail_3/dino --dino_glob '*_dino.npy' \
  --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy' \
  --out_dir data/Super/trail_3/deform
# 2. validate they reproduce the gate (+82%, cos>0.85) BEFORE touching the model:
python Addons/deform/validate_deform_targets.py \
  --pts Addons/eval/gt_pins/trial_3_l_pts.npy --deform_dir data/Super/trail_3/deform \
  --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy'
```
**PASS = baker overall valid ≳50–70% + validator valid-only reduction within ~10 pts of +82%, cos>0.85.** Then build Stage 1 (§6).

## 6. STAGE-1 — the model loss (NOT YET BUILT; exact code)
**New method in `model/scene_rep.py`** (mirrors the judge's `field_D` — zero convention-bug class):
```python
def deform_teacher_loss(self, Xk, t, dx_target, w):
    # Xk [N,3] surface point (world/field frame); t [N,1]; dx_target [N,3]; w [N,1] trust (detached)
    h = torch.cat([self.embed_time(t), self.embed_fre_pos(Xk)], -1)
    D = self.time_net(h)
    _hb = self.config.get('deform_hardbound', 0)
    if _hb and _hb > 0: D = _hb * torch.tanh(D / _hb)
    if not self.config.get('deformation_anchor_off', False):
        D = torch.where(t == 0, torch.zeros_like(D), D)
    return (w * (D - dx_target) ** 2).sum() / w.sum().clamp_min(1)
```
**Inject in `ddsslam.py` `current_frame_mapping` (~:362-364):** after `target_d` is sampled, look up `dx_target,w` from the baked `.npz` by `(cur_frame_id, indice_h, indice_w)` (bilinear-sample the patch-res maps; same indices `sample_dino_grid` uses); build `Xk = rays_o[...,:3] + rays_d * target_d`; call `deform_teacher_loss`; add to total:
```python
_ds_w = self.config['training'].get('deformation_sup_weight', 0)
if _ds_w > 0 and ret.get('def_sup') is not None:
    loss += _ds_w * ret['def_sup']
```
**Default `deformation_sup_weight: 0` → no key → byte-identical → `Addons/regression/test_inc0_bitidentical.py` passes unchanged.** Teacher arm flips it on.

**GATE (n=3 seeds, non-overlapping):** field pin-EPE reduction `>>` shuffled AND `|Δx|>0` AND `cos>0` AND anchor≈0 AND **no regression** in render PSNR/SSIM/LPIPS or Sim3 ATE.

## 7. STAGE 2-3 (contingent, each its own pin-EPE gate)
- **Stage 2 — escalate to `global_BA`** ONLY if Stage-1 reduction is positive-but-weak (current_frame_mapping is the minority trainer). Needs pixel-recovery (§2). Must beat Stage 1.
- **Stage 3 — ablations, stop at first that clears the bar:** (a) static-first warm-up (`deform_warmup_frames`); (b) surface-smoothness leash **across neighbouring rays** (NOT along-ray — that's near-dead code); (c) seg-route via existing `oracle_w`; (d) σ² trust-weight. **Map-throttle ships ONLY if it raises EXTERNAL pin-EPE** (never internal residual-Pearson — the documented unfalsifiable trap).
- **Cheap pre-steps from the paper reading** (thesis-bulletproofing, when GPU free): wire `time_smoothness_weight` as a **negative control** (confirm field stays dead); set `freq_n_t=4`/`freq_n_xyz=10` + enable `time_normalize` in the teacher arm so it's graded on a clean substrate.

## 8. CONVENTIONS & GOTCHAS (load-bearing — get these right)
- **Warp direction:** observed→canonical **pull-back**. `Xk + Δx* = X₀`. Target `Δx* = X₀ − Xk`. Getting this backwards = the 2.7× hollow failure. The `cos>0` check catches it.
- **Back-projection:** **OpenGL** rays (`datasets/utils.py:50`): `X = c2w_t + d·(c2w_R @ [(u-cx)/fx, -(v-cy)/fy, -1])`. The judge + baker + validator all use this; the field's `embed_fre_pos` takes RAW world pts (not bbox-normalised — that's only for the MAP at scene_rep.py:235).
- **Time:** `t = k/num_frames if time_normalize else k` (`ddsslam.py:285`). Read from config; match what the checkpoint trained with.
- **Pins are the HELD-OUT JUDGE — NEVER train on them** (leakage). Teacher is self-supervised (DINO+depth) so it transfers to CRCD unchanged.
- **SemSup pin GT is the ONLY deformation GT we have.** SemSup pose GT is FICTIONAL (use render + pins, not ATE). CRCD has NO deformation GT (STIR is the future un-gameable check).
- **Depth:** SemSup `png_depth_scale=8` (`/8`). MoGe up-to-scale → depth cancels in the relative pin comparison, so the dead-field/teacher verdicts are robust to it.
- **The field forward to replicate exactly** (scene_rep.py:205-218): `embed_time(t)` ⊕ `embed_fre_pos(pts)` → `time_net` → hardbound (`deform_hardbound`) → anchor `where(t==0,0,·)`. Reuse the loaded model's own modules — never re-implement the encoders.

## 9. READ THESE (memories — `…/memory/`)
1. `project_field_warped_pin_epe_verdict_20260618.md` — the dead-field verdict + the Stage-0 gate PASS (the core result).
2. `project_paper_vs_code_deformation_20260618.md` — why the teacher is new science, not a bug-fix.
3. `project_arm2_deformation_field_diagnosis_20260617.md` — the full diagnosis lineage.
4. `project_inc1inc2_build_wildgs_20260615.md` — the build context (TEACHER-SUPERVISED, Inc-1/2).
5. `feedback_metric_first_base_first_20260615.md` — GOVERNING methodology (metrics-only arbiter; FIX vs IMPROVE branches kept separate).
6. `project_stir_dataset_measurability_20260614.md` — STIR (the future CRCD/transfer un-gameable GT).
7. `reference_arm2_uncertainty_citations_20260615.md` — write-up citations.
8. `CLAUDE.local.md` (repo root) — paths, datasets, run cycle.

## 10. OPEN DECISIONS (the USER's calls — don't decide unilaterally)
1. **Injection site for v0:** cheap `current_frame_mapping` first (recommended) vs straight to `global_BA`.
2. **CRCD motion-floor go/no-go:** if CRCD's |Δx*| sits at/below the depth-noise floor → ship SemSup-only & park CRCD until STIR, OR invest in sub-pixel/RAFT correspondence. (Resolve with the generic-cycle probe on CRCD once its DINO is baked.)
3. **STIR sequencing:** build the STIR EPE loader before the teacher (un-gameable transfer GT) vs bolt it on after.

---
**Substrate stays pristine** (DDS-SLAM-Base = eternal reference). Everything Arm-2 is flag-gated default-off so base == pristine. The judge (§3) is the arbiter of every stage.
