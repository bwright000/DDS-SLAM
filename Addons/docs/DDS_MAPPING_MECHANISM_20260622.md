Resolved. The CRCD `mapping.iters: 20` is correct (finding 1), not 200 (findings 2/3 conflated it with Super.yaml). All other code claims verified. I now have what I need to write the doc decisively.

```markdown
# DDS-SLAM Mapping: Mechanism, the Deformation-vs-Mapping Competition, and the Levers

> Scope: OUR working copy `DDS-SLAM/`, parity-gated to IRMVLab `009977b` (all additions flag-gated default-off → base run == pristine). Line numbers cite the verified source. Where the four source findings disagreed on a number, the value below is the one **read directly from the file** — the most consequential correction is in §0.

---

## 0. Numbers reconciled (read this first)

Two findings inflated the CRCD mapping budget by 10× (conflating it with SemSup/`Super.yaml`). The file is decisive:

| Knob | CRCD (`crcd.yaml`) | SemSup (`Super.yaml`) | Source of confusion |
|---|---|---|---|
| `mapping.iters` (global_BA loop) | **20** (`crcd.yaml:24`) | 200 | findings 2&3 said "200 CRCD" — **WRONG**; it is 20 |
| `mapping.first_iters` | 1000 (`crcd.yaml:33`) | 1000 | agreed |
| `mapping.cur_frame_iters` | **0** (`crcd.yaml:25`) | 100 | agreed (CRCD current-frame mapping is a NO-OP) |
| `map_every` / `keyframe_every` | 5 / 5 (`crcd.yaml:29,28`) | 1 / 1 | agreed |
| `hash_size` | 19 (`crcd.yaml:66`) | 16 (legacy) | agreed |
| `voxel_sdf` | 0.001 (`crcd.yaml:74`) | 0.004 | agreed |

This changes the relative-budget story but **not the conclusion**: even at iters=20, global_BA is still the dominant field-trainer on CRCD because `cur_frame_iters=0` means current_frame_mapping never runs there — global_BA is the *only* mapper, period.

`time_net` capacity (decoder.py + `crcd.yaml:88-89`): `hidden_dim_time=128`, `num_layers_time=4`, inputs `freq(t)⊕freq(x)`. Finding 3's "~85k–130k params" is the right order; the hash table (`2^19 × feat × levels`) is ~1–4M floats — **map capacity ≫ field capacity by 1–2 orders of magnitude**, which matters for §3.

---

## 1. How the map is built and updated — the complete picture

### 1.1 Representation
Neural-SDF, Co-SLAM lineage. A point query is: `pts → (optional warp) → tinycudann HashGrid embed_fn ⊕ OneBlob pos-enc → SDF/color MLP decoder`.
- **Hash grid** (`grid.enc:'HashGrid'`, `hash_size:19`, `voxel_sdf:0.001`, `crcd.yaml:54-77`) — high-frequency local geometry. `voxel_sdf:0.001` is only safe *because* `hash_size:19`; at 16 it regressed ATE 2.20→788 mm (collision pressure, `crcd.yaml:68-73`).
- **OneBlob coordinate encoding** (`pos.enc:'OneBlob'`, `n_bins:16`) — low-frequency surface coherence.
- **Decoder** (`crcd.yaml:80-91`): SDF net 2-layer × 32-hidden → `[sdf:1, geo_feat:15]`; color net 2-layer × 64-hidden; `tcnn_network:false` (bias-free MLP). TCNN-normalized to `[0,1]` in the bounding box (`scene_rep.py:243-244`).
- **Deformation field** `time_net` (128-hidden × 4-layer) lives *inside* this path — see §1.5.

### 1.2 The three update loops (per frame, `ddsslam.py` run loop)
Order per frame: `tracking_render` (pose only, **frozen map**) → `if i % map_every == 0:` `global_BA` → `current_frame_mapping` → `_deform_replay_step` → `rendering` → `add_keyframe`.

1. **`first_frame_mapping`** (`ddsslam.py:305`, frame 0 only, `first_iters=1000`). Canonical anchor: random 2048 px from GT `c2w[0]`, render loss, `map_optimizer.step()` (no pose). At `t=0` the field is hard-zeroed (`scene_rep.py:217-218`) so it gets ~zero gradient here.
2. **`global_BA`** (`ddsslam.py:508`) — **THE dominant mapper.** Fires every `map_every=5` frames (so **80% of CRCD frames map zero times**). Loops `mapping.iters=20` (`ddsslam.py:555`). KF[0] is the fixed gauge (`ddsslam.py:532`); KF[1:]+current poses are jointly optimized. Rays drawn **uniformly** over the whole keyframe pool (`sample_global_rays`, `keyframe.py:85-98`) + ≥`min_pixels_cur=1024` current-frame px injected. `map_optimizer.step()` every step; `pose_optimizer.step()` every `pose_accum_step=5`.
3. **`current_frame_mapping`** (`ddsslam.py:390`) — **DEAD on CRCD** (`cur_frame_iters=0` → early-return). On SemSup (100 iters) it sharpens the current frame via `cur_map_optimizer`.

### 1.3 Optimizers (`create_optimizer`, `ddsslam.py:949-1002`)
- **`map_optimizer`** (`:976-979`): `_dec_groups(include_timenet = not deform_field_teacher_only)` + `embed_fn`. **By default `time_net` IS in it.** Drives first_frame_mapping + global_BA. `betas=(0.9,0.99)`, `lr_decoder=0.01`.
- **`cur_map_optimizer`** (`:982-990`, only if `cur_frame_iters>0`): same, excludes `time_net` iff `cur_frame_map_only:true`.
- **`field_optimizer`** (`:996-1002`, only if `deformation_sup_weight>0`): Adam over `time_net` alone. The replay-teacher path. **Never created on base CRCD/SemSup.**

### 1.4 Sampling (`render_rays`, `scene_rep.py:382-412`)
Depth-anchored: `z = linspace(-range_d, +range_d, n_range_d) + target_d` (`range_d=0.05`, `n_range_d=16` CRCD) → samples crowd a ±0.1 truncation band around measured depth; merged with `n_samples_d=32` uniform samples and sorted; jittered (`perturb=1`). Importance sampling is **dead code** (`n_importance=0`). Consequence: SDF gradients are dominated by the on-surface band, so off-surface field excursions are weakly penalized.

### 1.5 Where the field enters the render (`run_network`, `scene_rep.py:194-240`)
When `dynamic:true`: split `pts, frame_time` → `embed_time ⊕ embed_fre_pos → time_net → vox_motion[N,3]`. Then, in order: optional `tanh` hardbound (`deform_hardbound`, default 0), **`t=0` anchor** (`:217-218`, hard `where`), optional `detach()` if `deform_field_teacher_only` (`:226-227`), optional `oracle_w` route gate (`:232-234`), optional `surf_w` surface bind (`:237-238`), then **`inputs_flat = pts + vox_motion`** (`:239`) and `def_reg = (vox_motion**2).mean()` (`:240`). **The field is in EVERY forward** (mapping, BA, current, tracking) and warps points *before* the hash query.

---

## 2. The shared-loss gradient economy

`get_loss_from_ret` (`ddsslam.py:248-303`) is the single loss assembler for all mapping. Every render term backprops through `pts → time_net` via `pts+vox_motion`, so the map (hash+MLP) and the field minimize **the identical loss**.

| Term | Weight (CRCD) | Code | Trains map? | Trains field? |
|---|---|---|---|---|
| `rgb_loss` | `rgb_weight=5.0` | `:264` | yes (direct) | yes (indirect, via warp) |
| `depth_loss` | `depth_weight=0.1` | `:266` | yes | yes |
| `sdf_loss` | `sdf_weight=1000` | `:268` | yes (dominant magnitude) | yes |
| `fs_loss` (free-space) | `fs_weight=10` | `:270` | yes | yes |
| `edge_loss` | `0.5·rgb_weight` (`edge=False` default) | `:272` | — | — |
| `edge_semantic_loss` | `0.1·rgb_weight=0.5` | `:274` | yes | yes |
| `def_reg` = ‖Δx‖² | `deformation_reg_weight` **=0 (absent)** | `:280-281` | no | (would be field-only) |
| `nll` (Inc-1) | `nll_weight=0` | `:288` | off | off |
| `whatkind_loss` | `whatkind_weight=0` | `:295-297` | off | off |
| `smoothness` (SDF-TV) | `smooth_weight=1e-6` (only when `smooth=True`, i.e. global_BA) | `:299-302` | yes (negligible) | no (TV is on the SDF grid, **not** on `vox_motion`) |

**How the field receives gradient.** Only through `∂L_render/∂Δx = ∂L/∂sdf · ∂sdf/∂Δx` (and the rgb/color analogue). It has **no direct output target** by default — `def_reg` is off (weight 0), and the named regularizers `time_smoothness_weight`, `l1_time_planes`, `plane_tv_weight` (present in `crcd.yaml:111-113`) are **dangling config knobs with zero code references** — the comment at `ddsslam.py:276-277` openly admits the authors "named … but never connected" them. So the field's entire training signal is the residual the map has not yet absorbed.

**Two `vox_motion` gradient channels.** (i) `pts+vox_motion` into the hash/SDF query — vanishes wherever the SDF is locally flat (`∂sdf/∂Δx → 0`) or wherever the static map already fits (residual → 0). (ii) `def_reg` — **disconnected by default**. There is no third channel without the teacher.

---

## 3. WHY the map ALWAYS wins — ranked, with code evidence

The field is **not incapable** — under teacher supervision it learns (held-out pin-EPE +52.5% ≫ shuffled +41.6%, per the ARM-2 bridge). It collapses to |Δx|≈0 because, by default, every mechanism that could feed it is off and one mechanism actively starves it. Ranked, least-to-most dominant:

**Cause 5 — Capacity asymmetry (contextual).** Hash table ~1–4M floats vs `time_net` ~10⁵ params (§0). The static SDF can carve the *swept volume* of mild deformation alone. Real but secondary — finding 3 is right that the collapse would happen even at matched capacity.

**Cause 4 — Indirect, slope-gated field gradient.** Map gradient is direct `∂L/∂map`; field gradient is `∂L/∂sdf · ∂sdf/∂Δx`, which decays as the render loss converges (small residual) and vanishes on flat SDF (`scene_rep.py:239,249`; `sdf_weight=1000` drives `sdf_loss→0` fast). After convergence the field's signal is `O(slope × residual) ≈ 0`.

**Cause 3 — Un-routed field smears static content.** `oracle_w=None`, `route_w=None` by default → `inputs_flat = pts + vox_motion` warps **bg/tool/tissue indiscriminately** (`scene_rep.py:239`). Even a *live* field minimizes render loss by smearing rigid regions, not by modeling tissue. `route_w_ba` is assembled but gated off (`map_route.route_ba` default False, `ddsslam.py:560-564`) so global_BA trains the SDF on **un-warped** rays while render applies the warp — a co-adaptation mismatch.

**Cause 2 — Update-frequency/order dominance.** global_BA (and on CRCD it is the *only* mapper) runs and `map_optimizer.step()`s `time_net` **before** any teacher acts; `current_frame_mapping`/replay run *after* (`run loop`), and on CRCD `cur_frame_iters=0` so they never run at all. The map has stepped the field toward the render minimum before correction is possible. The code comment is explicit (`ddsslam.py:969-972`): *"v0 showed global_BA collapses the field to 0 before the teacher acts."*

**Cause 1 (DOMINANT) — `time_net` shares `map_optimizer` and minimizes the same render loss, which prefers Δx→0.** By default `deform_field_teacher_only=False` → `_dec_groups(include_timenet=True)` (`ddsslam.py:973-976`) → the field sits in `map_optimizer` with the hash/MLP at **identical lr (0.01)**. The render loss is a *geometry-fitting* loss: a sharper static SDF lowers it just as well as a correct Δx, and the map — with more capacity (C5), a direct gradient (C4), and more/earlier steps (C2) — gets there first. Once `L→0`, `∂L/∂Δx→0`, momentum dies, and the field drifts back toward its bias-free ≈0 init. With `wd=1e-6` (negligible) and no `def_reg`, **nothing holds it up**. This is starvation-by-shared-optimizer, and it is the root: the file's own fix for the dead field is to *remove `time_net` from `map_optimizer`* (`deform_field_teacher_only`, `:969-973`) — which only makes sense if membership is the cause.

---

## 4. The levers — ranked by EV, each tied to its cause; plus the protect/attend verdict

| # | Lever | Targets cause | Status | Est. render EV | Risk |
|---|---|---|---|---|---|
| 1 | **B2: thread `route_w_ba` into `global_BA.forward`** (co-adapt the SDF to warped queries) | **C3 + C1** | designed, ~15 lines, **not built** (gate at `ddsslam.py:560-564,634`; storage at `keyframe.py:59-98`) | **+1.5–2.5 PSNR** | low (sparsity; standard in NRGS/StereoMIS) |
| 2 | `deform_surface_bind > 0` (Gaussian falloff, `scene_rep.py:237-238,430-436`) — confine warp to surface band | C3 + C4 | code exists, default off | +0.5–1 PSNR (necessary, insufficient alone) | low |
| 3 | **Capacity: MLP width** `hidden_dim 32→64`, `num_layers 2→3` (`hash_size:19` already in force) | C5 | config-only | +0.5–1.0 PSNR (CRCD; SemSup likely saturated) | low (monotonic) |
| 4 | Loss reweighting: sweep `depth_weight∈[0.01,1]`, lock `rgb_weight=5` | sharpness floor | n=3 A/B, 9 runs | +0.2–0.5 PSNR | medium (depth↔rgb trade) |
| 5 | `deform_field_teacher_only:true` + `deformation_sup_weight>0` + replay — decouple & supervise field | **C1 + C2** (revives field, not render) | flags built | render-neutral; **revives field** | medium (needs baked Δx* targets) |
| 6 | Keyframe windowing / recency weight (`sample_global_keyframe`/`sample_overlap_keyframe` are **dead code**, `keyframe.py:100-192`) | tail-frame forgetting | not wired | +0.2–0.5 PSNR | low |
| 7 | `t=0` anchor alternative | identifiability only | flag exists | **0 (not a mapping lever)** | — |

**Verdict on the `map_route` protect/attend mechanism.** It is a **symptom-patch, not the cure.** `protect>0` excludes high-route rays from the global_BA *batch* and `attend>0` over-samples them (`ddsslam.py:616-632`) — but this is **batch-level filtering, not forward-path routing**: the map is still queried at `pts+vox_motion` on the rays that *do* pass, so the un-co-adaptation of Cause 3 persists (the "Lensing" the map must still fit). It reduces the map's *exposure* to corrupted regions (~0.5 PSNR) but does not make the map *learn geometry consistent with the warp*. The well-founded fix is **B2** (lever 1): route into `global_BA.forward` so the SDF co-adapts to the routed warp. `attend` additionally risks amplifying outliers. So: protect/attend is a defensible stopgap, but the principled mechanism is forward-path co-adaptation, and shipping protect/attend in its place would be patching the symptom that B2 removes at the source.

---

## 5. The single highest-EV next move for a near-term render win

**Build B2: thread the per-keyframe field route `route_w_ba` into `global_BA`'s forward pass.** It directly attacks the dominant pathology (Cause 3, the un-routed/un-co-adapted map, which compounds Cause 1) and is the cheapest decisive change available: the seams already exist —
- assembly at `ddsslam.py:560-564` (currently gated off by `map_route.route_ba`),
- the `forward(..., route_w=route_w_ba)` call at `ddsslam.py:634` already accepts it,
- per-keyframe route storage at `keyframe.py:59-98` (`add_keyframe` + `sample_global_rays(with_route=True)`).

≈15 lines wire the route through; **no new hyperparameters, parity-safe (default-off)**. Mechanistically it makes the static SDF learn its zero-crossing at the *warped* coordinates instead of fighting them, which is exactly the mismatch behind the inverted render ladder (un-routed 24.1 < routed < dead 27.70). Estimated **+1.5–2.5 PSNR** on SemSup toward the ≥28 target, and the largest headroom on CRCD (22 PSNR) where deformation (5–6 px flow) is an order larger than SemSup's ~1.5%-of-slab.

Pair it immediately with `deform_surface_bind>0` (lever 2, already-built, ~+0.5–1 PSNR) so the routed warp cannot extrapolate off-surface. Run it **from the pristine base, isolated, n=3**, judged on the live-render PSNR/SSIM/LPIPS (+ Sim3 ATE on CRCD) per the governing metric rule. Capacity (lever 3) and loss reweighting (lever 4) are deferred to a final tune — they are saturated/marginal next to the co-adaptation fix.
```