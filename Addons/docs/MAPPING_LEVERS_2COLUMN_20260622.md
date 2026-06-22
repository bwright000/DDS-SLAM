# Mapping Levers — Config-Tuning vs Our-Contribution (cross-validated vs memory)

> Built from `DDS_MAPPING_MECHANISM_20260622.md` (the mapping deep-read) cross-validated against project
> memory (wf w8b9eg7us). Two columns per the thesis distinction: **CONFIG** = tuning the authors' existing
> DDS-SLAM (a fair baseline, NOT a contribution) vs **CONTRIBUTION** = our novel mechanisms. Memory verdict:
> NOVEL (untried) / WON (already banked) / GRAVEYARD (do-not-retry) / CONTRADICTS.

## COLUMN A — CONFIG CHANGES (tuning DDS-SLAM; the *fair baseline*, not a contribution)
| Lever | Memory verdict | EV / note | Use in 48h? |
|---|---|---|---|
| **`cur_frame_iters` 0→100 (curmap100)** | **NOVEL** (never run) | **biggest mapping lever** — CRCD's 0 is a StereoMIS copy artifact @096d253; 100 = authors' own surgical value | **YES — the fair baseline** |
| `map_every`/`iters`/`keyframe_every` (CRCD 5/20/5 → denser) | CONFIRMED, deferred | CRCD ~8k rays/frame = **75× under** SemSup | with curmap100 |
| FM3 keyframe-windowing **wiring** (their dead code) | NOVEL (dead code, unwired) | render-tail; lowest-effort | cheap add |
| Decoder MLP width/depth `32→64` / `2→3` | NOVEL (decoder never swept) | +0.5–1 unproven | cheap probe |
| `hash16→19` | **WON** (30.148 PSNR) | already in `crcd.yaml` | banked — don't re-spend |
| `n_range11→16` | **WON** (paper typo) | banked | banked |
| `lr_trans 1e-3→1e-4` | **WON** (−39% ATE, backlog #0) | the standing baseline | banked |
| `voxel_sdf 0.001` | CONFIRMED (locked w/ hash19) | 0.001@hash16 = 788 mm | already in force |
| `sdf_weight:rgb` ratio sweep | NOVEL, low-conf | SDF dominates ~3000:1 → likely swamps | probe only |
| `depth_weight↑` sweep | **GRAVEYARD** | 0.1→5.0 = zero effect (SDF 3000:1) | **NO** |
| `smooth_weight↑` | **GRAVEYARD** | TV is on the SDF grid, never on `vox_motion` | **NO** |
| scale `trunc`/`range_d`/`far` | **CONTRADICTS** | Run-3A regressed ATE 2.20→5.80 mm | **NO** |

## COLUMN B — IMPROVEMENTS (OUR contributions; the thesis novelty)
| Lever | Memory verdict | EV / note | State |
|---|---|---|---|
| Flow-as-sensor **tracking gate** (`flow_agree_baf`) | WON | the tracking win (0.82 mm) | **shipped** |
| `map_route` + **`protect`/`attend`** (flow-routed static map) | ours, NOVEL | **~0.5 PSNR on the DEAD field** (sampling-level symptom-patch) | running now |
| **B2: `route_w_ba` into `global_BA.forward`** (co-adapt map→field) | NOVEL (principled fix) | **+1.5–2.5 PSNR — but ONLY with an ALIVE field**; no-op on the dead field | wired (`route_ba`), inert until field alive |
| **Δx\* teacher** (decouple field from `map_optimizer` + supervise) | WON (field ALIVE, +67% pin-EPE) | render-neutral; revives the field | **parked behind depth-anchor** |
| `deform_surface_bind` (confine warp to surface band) | WON (built) | +0.5–1 (with a live, routed warp) | built, default-off |
| Inc-1/2 uncertainty (geo / DINO σ²) | ours | tracking down-weight | done |

---

## The strategic problem the split exposes
**The biggest *novel* mapping lever is in Column A, not B.** `curmap100` (fix the starvation) is the single highest-EV untried mapping move — and it's **config-tuning the authors' model**, not our contribution.

**Our novel mapping *contribution* (Column B) is gated:**
- **B2 co-adaptation** is the principled contribution fix — but it co-adapts the map *to the field's warp*, and our field is **dead**, so **B2 is a no-op right now**. Its +1.5–2.5 PSNR is an *alive-field* number.
- Reviving the field (the **teacher**) is **WON** but **parked behind the depth-scale anchor** (the field learns `s×real`, mis-scaled vs the render gates).
- The only Column-B mapping lever that *acts on the dead field today* is **`protect`/`attend`**, which the deep-read itself calls a **symptom-patch (~0.5 PSNR)**.

So, plainly: **an easy mapping render-win exists (curmap100, config), but a mapping win attributable to our contribution needs the field alive — which needs the depth-anchor first.** The two are not the same deliverable.

## Graveyard guardrail (do NOT spend 48h here)
`depth_weight↑`, `smooth_weight↑` (both dead — SDF dominates / TV not on the field), `voxel_sdf 0.001` standalone (788 mm), blanket `trunc/range_d/far` scaling (regressed ATE, open contradiction), and re-spending the already-banked `hash19`/`n_range16`/`lr_trans`/teacher-revival.

## Recommended 48h plan (honest, two-column)
1. **Set the fair config baseline (Column A): `curmap100` + the banked config.** Report it as "DDS-SLAM *properly configured*" — this is the honest baseline our contribution must beat, and it's the most likely single render mover.
2. **Measure our contribution (Column B) *on top of* that tuned base:** the combine (tracking gate + `protect`/`attend`). This isolates the contribution delta against a fair, not crippled, baseline.
3. **The real contribution mapping win = revive the field (teacher) behind the depth-anchor, then B2.** That is the thesis-valid mapping win, but it almost certainly exceeds 48h. Decide explicitly whether to (a) ship the *config-tuned base + small contribution delta* as the 48h deliverable, or (b) gamble the 48h on the depth-anchor→teacher→B2 chain for the bigger contribution win.
