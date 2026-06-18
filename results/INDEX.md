# Results Index — DDS-SLAM run archive

The result **payloads** (videos, checkpoints, renders, zips) live in the dated folders below but are
**gitignored** (GBs, over GitHub's 100 MB limit). Only this INDEX is tracked. Reverse-chronological.

> `results/_misc/` holds stray/junk (CRCD fragments, temp extracts, non-run docs — see bottom).
> _(The redundant `.zip` archives were deleted 2026-06-18 — the extracted folders are the live copy.)_

| Folder | Date | What it is | Key result |
|---|---|---|---|
| `2026-06-18_field-diag+depth_overnight/` | 06-18 | **Arm-2 field diag** (field off/on/wd0 ×3, SemSup) **+ Arm-3 stereo120 metric depth** (CRCD c1_001) | `field_off` crashed (`dynamic:False` bug → fixed to `deformation_off:True`); render metrics empty (LPIPS GPU-OOM → fixed CPU); pin rep-err skipped (GT now committed). **Depth: MoGe ~5.3× too large, 18.7% scale drift, metric scene ≈7 cm — the ×9–10 fix validated.** |
| `2026-06-17_wgs-manual-runs/` | 06-17 | **WildGS-faithful** `dino_wgs` vs `base_wgs` (cur:10), via `run_cell` | DINO improves BOTH: ATE 2.89→**2.42** mm, PSNR 22.89→**23.35**. First CRCD render metrics. 6-panel videos here. |
| `2026-06-17_dinov2-first/` | 06-17 | First **DINOv2 (v2)** uncertainty, canon cur:0 (apples-to-apples vs geo) | dino #1 Sim3 ATE_mean **2.17** mm, beats geo (2.34) — n=1, the open A/B/C question. |
| `2026-06-16_crcd-uncert/` | 06-16 | CRCD geo(v1) uncertainty hedge runs (uncert s1/s2) + `c1_001_canon_uncert.mp4` | geo hedge on CRCD; feeds the n=3 canon below. |
| `2026-06-15_inc1inc2-hedge-canon/` | 06-15 | **🎯 Inc-1/2 hedge CANON (n=3)** — the headline Arm-1 win | base → **+uncert(geo)**: Sim3 ATE 3.15→**2.43** mm (−23%), max 14.79→**9.43**, \|Pearson\| 0.81→0.97, all non-overlapping. |
| `2026-06-14_map-probe/` | 06-14 | Gradient-attribution probe (map vs field — who gets the update) | "map absorbs the gradient not the motion" hypothesis (later folded into the Arm-2 diagnosis). |
| `2026-06-14_battery7-stabilise-falsified/` | 06-14 | Battery-7 (n=3) — STABILISE-FIRST test | **Falsified**: raw field "live" but HOLLOW (moves empty space, Pearson≈0). |
| `2026-06-13_battery6/` | 06-13 | Battery-6 — reg0.003/lr0.1 reproduction | **NO-GO + reproducibility failure** (came back DEAD vs battery-5 LIVE — seed coin-flip). |
| `2026-06-13_prebuild/` | 06-13 | Prebuild batteries (T0.1/T1.2/T1.3) + `deform_viz_*.html` | T1.2 time-fix works; field bistable, stable band not found. |
| `2026-06-13_hypconfirm3/` | 06-13 | Hypothesis-confirm 3 | field-diagnosis iteration. |
| `2026-06-12_hypconfirm2/` | 06-12 | Hypothesis-confirm 2 (two runs) | field-diagnosis iteration. |
| `2026-06-11_hypconfirm-dead-field/` | 06-11 | **Dead-field verdict** (SemSup) | gauge-RACE falsified; field inert (deformoff ≈ baseline) → structural, not the pose race. |

## `_misc/`
- `crcd-stray/` — `C_1`, `E_3`, `F_3` snippet dirs (CRCD data fragments, not run outputs).
- `_hyp_extract/`, `_hyp2x/` — temp extracts of the hypconfirm experiments (super_baseline/deformoff/posefrozen). Redundant with the dated `hypconfirm*` folders.
- `medical_segmentation_datasets_survey*.docx`, `annotated_masks.replay.json` — non-run docs.

_Labels are best-effort from the run logs/canon; rename any folder freely — only this INDEX is tracked, so update the table to match._
