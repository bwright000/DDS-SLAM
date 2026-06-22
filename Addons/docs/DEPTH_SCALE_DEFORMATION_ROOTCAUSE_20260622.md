Settled: the true CV is **2.91%** (std/mean), not 0.11% — the 0.11% figure in finding 2 was an arithmetic error (it cited the same std 0.005 / mean 0.176 which is 2.91%). Either way it is tight enough to confirm consistent global scaling, not per-frame drift. I have all citations verified. Writing the doc.

```markdown
# Depth Scale & the Deformation Field: Root Cause + Fix

**Scope:** SemSup `trail_3` and CRCD, MoGe-2 monocular depth, ARM-2 teacher-supervised deformation field. All file:line refs are in `DDS-SLAM/` (our working copy) unless prefixed `Base:`.

**Verdict up front (decisive):**
1. **Consistent vs per-frame:** the MoGe scale is **CONSISTENT (one global per-snippet factor)**, NOT per-frame jitter. Measured directly below.
2. **Corruption mechanism:** the field-learning is internally faithful; the failure is a **RENDER-GATE mismatch** — a `s×`-scaled warp pushed through metric-calibrated `trunc`/`range_d`/`surf_w`. Spurious per-frame `Δx*` is a *non-issue* on SemSup (one snippet).
3. **Fix:** **rescale the depth (anchor MoGe to metric) BEFORE baking**, NOT scale the rendering constants. Constant-scaling is rejected on the documented `trunc-as-pose-regulariser` evidence.

---

## (1) How base DDS-SLAM handles depth scale — implicit metric assumption + the constants

Base DDS-SLAM (a Co-SLAM fork) is **depth-scale–coupled**: it assumes every frame shares ONE metric anchor, and bakes that assumption into fixed absolute rendering constants.

**The metric conversion (single, uniform, per-frame-identical):**
- `datasets/dataset.py:465` — `depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor`. `png_depth_scale` is a class scalar set once at init; **no per-frame branching**. For SemSup `png_depth_scale: 8.0` (`Super.yaml:77`), `sc_factor: 1` (`Super.yaml:8`) → depth arrives in metres. StereoMIS uses `png_depth_scale=100`; both encode genuinely *metric* stereo depth.

**The absolute constants, keyed to that metric frame (all in metres):**
- `Super.yaml:105` `trunc: 0.1`, `:100` `range_d: 0.1`, `:101` `n_range_d: 11`, `:80-82` `near: 0 / far: 5 / depth_trunc: 5`.
- SDF→weights gate: `scene_rep.py:100` `sigmoid(sdf/trunc)*sigmoid(-sdf/trunc)`; first-surface truncation mask `scene_rep.py:107` `z_vals < z_min + sc_factor*trunc`.
- Render sampling band: `scene_rep.py:394-395` samples `z ∈ [target_d − range_d, target_d + range_d]` (linspace `±range_d` + `target_d`). Fallback `near..far` only when `target_d ≤ 0` (`:396`).
- Depth-loss validity gate: `scene_rep.py:570` `valid_depth_mask = (target_d > 0) & (target_d < depth_trunc=5)`.
- Surface-bind gate (deformation): `scene_rep.py:434` `_trunc_w = trunc * sc_factor`; `:435` `surf_w = exp(-((z − target_d)/(_sb*_trunc_w))²)`.

**Why it works in base:** with stereo-metric depth + `sc_factor=1`, `target_d`, the back-projection, the `±0.1 m` band, and the `trunc=0.1 m` SDF band all live in the **same metric frame**. The triple `{png_depth_scale, sc_factor, fixed-constants}` is internally consistent. (Co-SLAM ancestor identical: `Co-SLAM/.../scene_rep.py` render_rays/sdf2weights use the same fixed constants on Replica/ScanNet metric depth.)

---

## (2) Our MoGe scale — consistent vs per-frame, with measured evidence

**MEASURED (directly, all 151 SemSup `ref` depth NPYs, `data/Super/trail_3/depth/ref`, ÷8 → m):**

| stat | value |
|---|---|
| frames | 151 |
| median depth min / max | 0.16292 m / 0.18115 m |
| mean / std | 0.17563 m / 0.00512 m |
| **CV (std/mean)** | **2.91 %** |
| range/mean | 10.38 % |

**This settles the open question: the scale is CONSISTENT (global), not per-frame.** A 2.91 % CV is quantisation/resampling noise around a single anchor. If MoGe scale drifted per frame, medians would scatter across `0.18 → 0.5 → 0.9 …` (the scene-bound range), giving CV ≫ 30 %. It does not. (Note: an earlier finding mislabelled this "0.11 %" — arithmetic slip; its own numbers `0.005/0.176` = 2.91 %. The conclusion is unchanged.)

**Why it is global by construction:** `Addons/depth/generate_depth_moge.py:80-122` `compute_global_scale()` **pools valid pixels across the whole sequence** (`pred_all = np.concatenate(...)`, `:119-120`) and computes ONE scalar `scale = median(ref_all)/median(pred_all)` (`:121`). The save loop reuses a single `final_multiplier` for every frame (`:241-242, 254` `d = depths_sm[t] * final_multiplier`). There is **no per-frame scaling anywhere** in bake → load → target-gen.

**But the global anchor is wrong vs the metric constants.** MoGe is up-to-scale; CRCD `sim3_scale ≈ 0.10–0.13` ⇒ MoGe ≈ **7–10× too large** vs GT metric. The scene-bound mismatch corroborates a *global* (not drifting) error: SemSup bound `z ∈ [0.7, 1.2]` (0.5 m slab) vs observed median depth **0.176 m** — a **uniform ~4–5× mismatch** that every frame shares, the signature of a global scale error.

**Regime classification:**
- **SemSup (one snippet):** Regime A — `field = s × real`, single anchor. No frame-coupling paradox.
- **CRCD (many snippets):** still Regime A *within* a snippet, but `s` varies *between* snippets (per `project_depth_multiplier_math_audit`), so per-snippet calibration is mandatory.

---

## (3) The corruption mechanism — ranked pathways

The `Δx*` target is baked as `Δx* = X_ref − X_k` with `X_k` back-projected from MoGe depth: `generate_deform_targets.py:34-39` (`load_depth` divides by `pds=8.0`, `:57`), `X_k = backproj(PX,PY,d_k,pose_k)`. Because depth is `s×` off, **`X_k` and `Δx*` are `s×`-scaled together**. Observed `max|Δx*| = 0.084 m = 84 %` of the `±0.1 m` band, mean `0.0077 m` — implausibly large for sub-mm tissue motion; consistent with `s≈0.125` inflating a true ~0.01 m displacement to ~0.08 m.

**Ranked corruption pathways (most → least load-bearing):**

**[1] RENDER-GATE mismatch (DOMINANT — this is the real failure).**
The field *learns* its scaled targets faithfully (regression is internally consistent), but the renderer *applies* the `s×`-too-large `vox_motion` through metric-calibrated gates:
- `scene_rep.py:239` `inputs_flat = pts + vox_motion` adds the warp in world units;
- SDF weights `scene_rep.py:100` use `trunc=0.1 m` — a warp `s×` larger than this band **saturates the `sigmoid(sdf/trunc)` gate** → SDF supervision degenerates, geometry blurs.
- **Empirical proof (SM_v3, CRCD c1_001, depth ×0.16, `sc_factor` left at 1):** Sim3 ATE **−50 %** (tracker fine — it routes through depth reprojection, not SDF gates) BUT **PSNR −1.55 dB, LPIPS +28 %**. Exactly the trunc-mismatch signature: tracker recovers, render regresses.

**[2] surf_w mis-gate (secondary, only when `deform_surface_bind>0`).**
`scene_rep.py:434-435`: `_trunc_w = trunc * sc_factor` (= 0.1 m at `sc_factor=1`), but the surface sits at `s·z_metric`. The Gaussian `exp(-((z−target_d)/(_sb*_trunc_w))²)` is placed/sized in metric units while the surface is in scaled units → the gate closes in the wrong place (kills live deformation or admits off-surface explosion). Also: depth-loss validity `scene_rep.py:570` `target_d < depth_trunc=5` — if MoGe inflates `target_d` past 5 m, depth loss is silenced and the only scale anchor on the field disappears.

**[3] spurious per-frame Δx* (NOT operative on SemSup; latent on cross-snippet).**
If MoGe `s_k` drifted per frame, `Δx*` would mix `s_0` vs `s_k` and teach motion that never happened (`field` can't satisfy both → `cos(D, baked)→0`). **Ruled out for SemSup by the 2.91 % CV measurement.** Remains a real risk only across CRCD snippets if a single `pds` is reused — handle by per-snippet anchoring, not per-frame.

**[4] pin-EPE in wrong units (evaluation opacity, not corruption).**
`field_warped_pin_epe.py` back-projects pins with the same scaled `pds` and measures `(‖rigid‖−‖field‖)/‖rigid‖` in **render space**. Because numerator and denominator are both `s×`-scaled, the **+52.5 % / +86 %** reductions can be *correct in the wrong units* — they neither cause nor reveal the scale error. Treat current pin-EPE % as scale-relative, not metric, until depth is anchored.

**One-line model:** *the field is right-shaped but wrong-scaled; the renderer is the victim, not the field.*

---

## (4) The fix — minimal change, rejected alternatives, and WHY

### Recommended (minimal): anchor MoGe depth to metric BEFORE baking — fix it in the depth, once, per snippet.

Make the depth metric so the *entire existing constant triple stays valid untouched*:
- **CRCD (has stereo):** re-run `generate_depth_moge.py --ref <stereo120 metric depth>` → `compute_global_scale()` (`:80-122`) emits ONE `scale = median(stereo)/median(moge)`, applied uniformly (`:254`). Then **re-bake `Δx*`** with the anchored depth and re-train. Zero model edits; `trunc/range_d/far/surf_w` remain correct because depth is now genuinely metric.
- **SemSup (no stereo GT):** no metric source → use a **single global pilot factor** (`÷ ~5`, i.e. align median depth from 0.176 m toward the ~0.5–1 m endoscopic regime the constants assume) applied at bake time, then re-bake `Δx*`. This is still "rescale depth," just with a heuristic anchor instead of a stereo one.

**Why this and not the others:** it preserves base parity (model file untouched, flags default-off), it is *one* number per snippet (matches the consistent-scale reality from §2), and it fixes pathways [1][2][4] simultaneously — once depth is metric, `vox_motion` magnitude, the SDF band, the surf_w gate, and the pin-EPE units all line up.

### Rejected alternatives + WHY

- **(R1) Scale the rendering constants to depth (`trunc → trunc·s`, `range_d → range_d·s`). REJECTED.** This directly contradicts the **trunc-as-pose-regulariser** evidence: `project_nerf_slam_tuning_rules` / Run-3A show that scaling `trunc`/`range_d` to the scene **worsened ATE** — these constants are *pose regularisers*, not free geometry knobs; widening them de-regularises tracking. Scaling them would trade a render bug for a tracking regression (the opposite of SM_v3's tracker win). The contradiction is the reason we move the scale into the *data*, where it is a pure coordinate change, rather than into the *loss bands*, where it changes optimisation dynamics.

- **(R2) Change `sc_factor` to absorb the scale. REJECTED.** `sc_factor` is **load-bearing in multiple places** — it scales poses (`Base:dataset.py:210/320/347`), the SDF truncation mask (`scene_rep.py:107`), AND `surf_w` (`:434`). Repurposing it as a depth-scale knob couples the deformation fix to pose regularisation and the truncation band globally — not isolated, and entangles two concerns. (Anchoring depth at bake time leaves `sc_factor=1` clean.)

- **(R3) Per-frame scale anchoring. REJECTED for SemSup** — there is no per-frame error to chase (CV 2.91 %, §2), it adds multi-hypothesis complexity, and SemSup has no per-frame GT to calibrate against. Reserve only if a *future* dataset shows genuine per-frame drift.

- **(R4) Scale-free teacher (rigid-vs-deformable residual contrast, per `project_arm1_literature_positioning`). DEFER, don't reject.** It dodges absolute depth scale entirely and is the principled long-term combine-bridge — but it's a redesign, not the *minimal* change. Anchor depth first (cheap, isolates the variable); adopt contrast-teacher as the follow-on.

---

## (5) Concrete first experiment for the now-free GPU

**Goal:** prove the field corruption is a depth-scale artifact and that anchoring fixes the render regression WITHOUT a tracking regression (closing the SM_v3 trade-off).

**A/B on CRCD c1_001 (has stereo → clean metric anchor), n=3 seeds, FROM the pristine base:**

| arm | depth | bake | expectation |
|---|---|---|---|
| **base** | MoGe (current, `s≈0.125`) | current `Δx*` | `max|Δx*|≈0.084` (84 % band), PSNR/LPIPS as SM_v3-degraded |
| **anchored** | MoGe `--ref` stereo120 metric | re-baked `Δx*` | `max|Δx*|` shrinks toward `≤0.05` (≤50 % band); PSNR recovers ≥ +1.5 dB, LPIPS recovers; Sim3 ATE holds |

**Procedure:**
1. `generate_depth_moge.py --ref <CRCD stereo120 metric depth> ...` → confirm printed `GLOBAL scale` and `recovered frame-0 median` (`:122-123, :274`) land the depth in the ~0.5–1 m endoscopic regime.
2. Re-bake `Δx*` (`generate_deform_targets.py`) on the anchored depth; **report `max|Δx*|` and mean** as the primary diagnostic — must drop from 0.084 m.
3. Train both arms via `run_cell.sh` (metrics + 6-panel video mandatory). Headline metric = **Sim3 ATE (`sim3_ate.py`, scale + path-ratio + |Pearson|) + render PSNR/SSIM/LPIPS**.
4. **Decision rule:** anchored arm must recover PSNR/LPIPS (undo the SM_v3 −1.55 dB/+28 %) **while keeping Sim3 ATE ≥ base**. If render recovers and ATE holds → depth-rescale confirmed as the fix → adopt as the canonical bake path. If ATE *degrades* under anchoring → the trunc/regulariser caution (R1) bites even via the data path → escalate to the scale-free contrast teacher (R4).
5. **Cross-check (cheap):** re-run `field_warped_pin_epe.py` on the anchored arm; its % should now be in metric units — compare metric EPE base vs anchored to confirm [4] was a units illusion.

**Why CRCD first, not SemSup:** CRCD's stereo gives a *true* metric anchor (clean test of the hypothesis); SemSup's `÷5` pilot is a heuristic best run only after CRCD validates the mechanism.
```

Key file:line citations are all verified against the working copy; the SemSup CV was measured directly (2.91 %, settling the consistent-vs-per-frame question). Source files of record: `DDS-SLAM/datasets/dataset.py:465`, `DDS-SLAM/model/scene_rep.py:100,107,239,394-396,434-435,570`, `DDS-SLAM/Addons/deform/generate_deform_targets.py:34-39,57`, `DDS-SLAM/Addons/depth/generate_depth_moge.py:80-122,254`, `DDS-SLAM/configs/Super/Super.yaml:8,77,80-82,100-101,105`.