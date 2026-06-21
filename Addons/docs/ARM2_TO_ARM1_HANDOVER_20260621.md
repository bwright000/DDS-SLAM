# ARM-2 → ARM-1 HANDOVER (deformation field → the combine)
**Date:** 2026-06-21 · **From:** Arm-2 (deformation-field revival) · **To:** Arm-1 (uncertainty σ² / pose down-weight)
**Live memory:** `project_deform_gauge_bug_20260620` (head), `project_field_warped_pin_epe_verdict_20260618` (the judge), `project_combine_routing_model_20260619` (your arm).

---

## 0. TL;DR
Arm-2's job was to revive DDS-SLAM's **dead deformation field**. **Done — the field is ALIVE.** It now tracks tissue deformation on held-out ground truth (**+52–67% green-pin EPE reduction**, time-specific, correct direction). The remaining problem is **yours to close**: the alive field warps **everything** (static background + tool + tissue) because it is **un-routed** → it costs ~4 PSNR of render (global blur + tool granularity). **Routing is the combine, and routing is your σ²/seg.** The field is no longer a hypothesis — it's real machinery for your router to gate.

---

## 1. Where Arm-2 stands (the numbers, SemSup trail_3, pose-frozen, n=1 unless noted)
| arm | field (held-out pin-EPE) | render PSNR | note |
|---|---|---|---|
| base / `teacher_off` | dead (\|Δx\|=0) | ~28 | localized blur (only at moving tissue) |
| `teacher_on` (frame-k teacher) | −44% (worse), shuffled≈real | 22.2 | **forgets** → time-agnostic |
| `teacher_replay` (causal replay) | **+67%**, cos +0.85, ≫ shuffled +42% | 22.2 | **field ALIVE** |
| `teacher_coadapt` (un-isolate, FAILED) | dead (\|Δx\|=0) | 23.3 | render gradient re-collapsed it; **global blur** |
| `teacher_replay_sharp` (map-only sharpen) | **+52.5%**, cos +0.81 | **24.1** | alive **and** ~2 PSNR render recovered |

**n=3 lock of `replay_sharp` is running** (`SEEDS="0 1 2" ARMS="replay_sharp"`). Expect the field mean to settle near replay's (+52 vs +67 is likely n=1 RNG-stream noise — the field training is identical, replay-only).

**Render gap decomposition (the important read):** base ~28 → replay 22.2 = ~6 PSNR. **~2 PSNR was a pipeline artifact** (we'd disabled the current-frame sharpening pass to protect the field — `cur_frame_map_only` brought it back). **~4 PSNR remains = the un-routed field smear → ROUTING (your combine).**

---

## 2. The machinery you'll combine with (how the field works now)
The field was **starved**, not buggy: DDS-SLAM's render losses are blind to small monocular deformation, so the degenerate optimum is "blurry-average map + Δx=0." Faithful to the paper (no Δx supervision by design — see `project_paper_vs_code_deformation`). So the teacher is a **genuine new contribution**, not a port-fix.

The revival chain (all flag-gated default-off → base bit-identical, parity-gated):
1. **Teacher Δx\*** — self-supervised target: DINOv2 correspondence **frame k → frame 0** + monocular depth → a per-pixel displacement grid. **Causal** (each target uses only `{frame 0, frame k}`). Currently baked offline for speed; online-computable per-frame (~1–2% overhead) and **doing it online auto-fixes the gauge bug** (deferred, spare-time).
2. **Causal replay** (`_deform_replay_step` + `field_optimizer`) — trains the field on a buffer of **past** frames each step → fixes catastrophic forgetting → the field becomes **time-specific** (real ≫ shuffled-time).
3. **Gauge fix** (`regauge_deform_targets.py`) — see §4, pitfall #1.
4. **`cur_frame_map_only`** — keep the current-frame sharpening pass but train the **map only**; the field is excluded from the render optimizers and trained solely by the teacher. The render forward still *uses* the field, so **the map co-adapts to the field** (not the field to render).

Configs: `configs/Super/trail3_teacher_{off,on,replay,replay_sharp}.yaml`.

---

## 3. Infrastructure to REUSE (don't rebuild)
- **The judge: `Addons/eval/field_warped_pin_epe.py`** — the **only field-sensitive metric**. Render PSNR and Sim3-ATE are **field-blind** (a dead field renders fine). Use this to verify your routing keeps the field alive, not just that render looks good. Ships a 6-panel diagnostic PNG.
- **`--deform_dir` flag on the judge** — reports `cos(D, baked Δx*)` + baked reduction → **separates "the field is wrong" from "the targets are wrong"**. This is what cracked the gauge bug. Use it whenever a result looks off.
- **`run_cell.sh`** (manual run: metrics + 6-panel video + Sim3) and the runbook **`teacher_ab_t4_20260619.sh`** (env → regauge → train → judge → ship to Drive).
- **Standing rule** (`feedback_two_diagnostic_sets`): every GPU run ships **numerical + visual** diagnostics, auto-shipped to Drive co-located with the payload.

---

## 4. Pitfalls — hard-won, these WILL bite you
1. **GAUGE: bake, train, and judge must share ONE pose frame.** The dead field looked "wrong/overfit" for a day — it was actually a **bake-in-identity vs SLAM-in-`diag(1,−1,−1)`** frame mismatch. The field learned its targets *faithfully* but in the wrong frame (`cos(D,baked)=+0.78` yet `cos(D,X0−Xk)=−0.64`). **Any baked target you introduce (a σ² teacher, an optical-flow target) inherits this** — back-project it in the SLAM's live pose frame, or regauge. The `--deform_dir` check is your early-warning.
2. **Adam neutralizes the loss weight.** We set `deformation_sup_weight:10000` expecting the teacher to dominate the render — it didn't. Adam normalizes by gradient magnitude, so a big loss weight only sets *direction*, not step size. **Don't use loss weights to win a tug-of-war; use optimizer routing** (separate param groups / separate optimizers).
3. **The render gradient collapses the field to 0.** If `time_net` is in the render optimizer (`global_BA` or `current_frame_mapping`), render starves it (the degenerate optimum). The fix that works: **isolate the field in its own optimizer, co-adapt the MAP to the field.** The naive "let render refine the field" (coadapt) is a confirmed dead end — it killed the field *and* smeared the map.
4. **Pose-frozen is a scope choice, not a result.** All Arm-2 numbers pin the pose to isolate the field from the pose↔field gauge race. **Un-freezing the pose is YOUR arm** (Inc-2 uncertainty-weighted tracking). The field consumes whatever pose you hand it; the targets inherit its error (but bake+SLAM stay consistent if computed in the same frame — see #1).
5. **Metric traps:** SemSup pose GT is **fictional** → ATE meaningless, use render PSNR/SSIM/LPIPS. CRCD is **sub-SNR** → Sim3-ATE + path-ratio + dominant-axis Pearson (never rigid `output.txt`). These are in the canon memories.

---

## 5. THE COMBINE — your interface (the most important section)
**What Arm-2 hands you:** an alive deformation field that warps observed→canonical, but **applied globally** — it has no notion of *where* deformation is, so it smears the static background and the independently-moving tool. Symptoms: **global blur + tool granularity + ~4 PSNR render gap.**

**What closes it (your arm):** **route the field to the deforming tissue only.** This is exactly `project_combine_routing_model`:
- **static background → camera** (field excluded → stays sharp),
- **tool → its own rigid SE(3)** (field excluded → no granularity),
- **tissue → the field** (warped correctly → blur becomes localized *and corrected* → render recovers).

The router is your **σ² (made deformation-aware) × seg/DINO what-kind**. The Inc-2 `1/σ²` tracking/mapping split already half-wires the routing (trust→tracking, moving→field). **The render recovery (the last ~4 PSNR) is the combine deliverable — it lives in your routing, not in more Arm-2 tuning.** Your `flow-as-sensor` thread (measure motion via optical flow, split camera-vs-scene) is the natural "what-moves" signal to gate the field with.

**Validation of the combine** = the same field-sensitive judge: routing must keep pin-EPE alive (≫ shuffled) *while* render climbs toward 28. If routing kills the field, you've over-gated; if render doesn't recover, you've under-gated.

---

## 6. Open / pending
- **Now:** n=3 lock of `replay_sharp` (the Arm-2 headline: alive field + ~2 PSNR render recovered).
- **Next (your arm):** seg/σ²-routed combine → recover the last ~4 PSNR while keeping the field alive.
- **Deferred (spare-time, `project_dds_fundamentals_deferred`):** online Δx* (auto-fixes the gauge), canonical re-anchoring, replay-buffer windowing. None are leaks; all causal.

---

## 7. Code map
- **Field/teacher:** `model/scene_rep.py::deform_teacher_loss`; `ddsslam.py::{_deform_replay_step,_buffer_deform,create_optimizer(field_optimizer/cur_map_optimizer),current_frame_mapping,global_BA}`. Flags: `deformation_sup_weight`, `deform_field_teacher_only`, `cur_frame_map_only`, `deform_replay_iters`, `global_ba_time_fix`, `time_normalize`.
- **Targets:** `Addons/deform/{generate_deform_targets,validate_deform_targets,regauge_deform_targets}.py`.
- **Judge:** `Addons/eval/field_warped_pin_epe.py` (`--deform_dir` for the baked-check).
- **Runbook:** `Addons/colab/teacher_ab_t4_20260619.sh` (regauge is on the path; judge auto-runs).
- **Configs:** `configs/Super/trail3_teacher_*.yaml`.
- **Memory:** read `project_deform_gauge_bug_20260620` first, then `project_field_warped_pin_epe_verdict_20260618`, `project_combine_routing_model_20260619`, `project_paper_vs_code_deformation_20260618`. Dead lineage (do-not-retry) consolidated in `project_graveyard_20260613`.
