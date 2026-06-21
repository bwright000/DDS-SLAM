The two most load-bearing anchors verified: the `oracle_w` seam is exactly at scene_rep.py:223-225 (`vox_motion = vox_motion * ow`), and the run loop gating (`i%map_every==0` → global_BA → current_frame_mapping → _deform_replay_step) is at ddsslam.py:990-993. Both match the findings precisely. Writing the document.

# DDS-SLAM Mapping: Research + Design Canon (the Inverted-Flow Mapping Push)

**Status:** design canon, 2026-06-21. **Branch context:** IMPROVE + COMBINE. **Governing rule:** METRICS are the arbiter; pristine `DDS-SLAM-Base/` is the eternal reference; every addition is flag-gated default-off (parity-gated == pristine base). **All paths absolute under** `c:\Users\benli\OneDrive\Documents\GitHub\DDS-SLAM\DDS-SLAM\`.

This document threads ONE spine: **mapping is the inverse of tracking.** The same per-region camera-vs-scene flow-disagreement signal that the tracking gate uses to *down-weight pose* must be *inverted* to *route the map* — static→sharp, deforming-tissue→the alive field, tool→excluded. The decisive engineering fact is that the routing seam **already exists** (`oracle_w` at `model/scene_rep.py:223-225`), so the whole field-routing reduces to threading one kwarg.

---

## 1. How Neural-SLAM mapping works (the foundation)

Neural SLAM is **online continual learning of a scene representation jointly with camera pose.** A neural map — here a Co-SLAM-style multi-resolution hash-grid plus a tiny SDF/color MLP decoder — is trained *incrementally*: each incoming frame is tracked against the *frozen* map, then the map (and a window of recent poses) is refined by a mapping/bundle-adjustment step over a keyframe database. Five structural facts govern everything below.

**1.1 Track-then-map, online not offline.** The run loop alternates TRACK (pose vs frozen map) then MAP (refine map + recent poses). DDS-SLAM's `run()` is literally this (`ddsslam.py:957-1001`): frame 0 → `first_frame_mapping`; every later frame → `tracking_render` (pose only), then on the mapping cadence `global_BA` → `current_frame_mapping` → `_deform_replay_step`, then `rendering`, then `add_keyframe`. iMAP framed real-time SLAM as "online continual learning" needing both *plasticity* (learn the new frame) and *stability* (don't forget old frames) — the governing tension of all neural mapping. Offline SfM/NeRF instead see all frames jointly and have no forgetting problem; the online setting's sequential distribution shift is what makes everything below hard. (iMAP, arXiv:2103.12352; "Learn to Memorize and to Forget", arXiv:2407.13338.)

**1.2 The map is a network → catastrophic forgetting → replay is the cure.** Because the map is a shared MLP+grid, training on the newest frame *overwrites* old frames. The dominant countermeasure is **keyframe REPLAY**: re-sample old keyframes every map step. **Global bundle adjustment IS the replay engine.** iMAP: a global representation suffers "severe catastrophic forgetting if keyframe-based replay is not deployed." Other countermeasures: windowing/co-visibility selection (NICE-SLAM's small active set), local feature grids (less forgetting), regularization/distillation (UNIKD), submaps (MIPS-Fusion). Co-SLAM's specific claim is that its *joint* encoding lets it replay ALL keyframes rather than a small active window.

**1.3 The map representation: Co-SLAM hybrid encoding.** Multi-resolution **hash-grid** features (Instant-NGP: fast convergence, high-frequency local detail) + a **one-blob coordinate** encoding (surface coherence, hole-filling in unobserved areas) → tiny MLP decoder. Co-SLAM's thesis: pure parametric (hash) over-fits/forgets, pure coordinate (MLP) is slow; the joint encoding gets *both* fast convergence AND smooth completion — which is exactly what lets it run global BA over *all* keyframes. DDS-SLAM inherits this: `run_network`/`render_rays`/`forward` in `model/scene_rep.py`; the decoder (SDF + color + the added σ²/edge heads) in `model/decoder.py`. Map *flavors* differ by decoder output: SDF/TSDF (DDS-SLAM/Co-SLAM, surface = zero-crossing) vs occupancy (NICE-SLAM) vs radiance/density (iMAP). The 2024-25 alternative is 3D Gaussian Splatting — explicit primitives, differentiable rasterization, real-time, no per-point network query (relevant to §3's structural ceiling). (Co-SLAM, arXiv:2304.14377; SLAM-meets-NeRF survey, MDPI 2032-6543/15/3/85; "How NeRFs and 3DGS are Reshaping SLAM", arXiv:2402.13255.)

**1.4 Depth is the backbone, not RGB.** Depth supervision (a) **anchors render samples to the surface** and (b) **supplies the SDF + free-space losses that carry most of the geometry gradient.** RGB alone is weak/ambiguous. In DDS-SLAM, `render_rays` puts `n_range_d` samples in a `target_d ± range_d` band *around the measured depth surface* (`model/scene_rep.py:385-386`); `get_sdf_loss` splits those samples into within-truncation (SDF target = signed distance to surface) and beyond-truncation (free-space loss pushes SDF to ±tr). Per the flaws catalog, raising `depth_weight` does *not* help (SDF dominates) — depth's value is *structural* (sample placement + SDF target), not a loss-weight knob. DDS-SLAM's depth is **MoGe-2 monocular, up-to-scale** → the chronic Sim3/scale trap that makes rigid ATE invert A/Bs.

**1.5 Bundle adjustment = joint map+pose inverse rendering, and tracking/mapping are DECOUPLED.** Neural-SLAM BA is *not* classical feature reprojection: it renders rays from a window of keyframe poses, then backprops a photometric+SDF loss to **both** the map *and* those poses (first KF fixed as a gauge anchor). Crucially, tracking optimizes *only* pose against a frozen map; mapping optimizes the map against fixed-or-jointly-refined poses. **This decoupling is the seam the thesis lives in** (§4): you can drive the two loops with the *same* sensor at *opposite* polarity.

**1.6 The static-map premise breaks under deformation.** The entire pipeline assumes one geometry explains all frames. Moving content poisons both halves: the map *averages* motion into blur, and pose *drifts*. DDS-SLAM's revived deformation field warps *everything* un-routed → ~4 PSNR render gap + tool ghosting. The literature's answer is to **separate motion from camera** (optical-flow + multi-view-geometry residual — the same signal the user already built for tracking) and route moving regions to a time-conditioned map. Dynamic-SLAM work (DIO-SLAM; "Learn to Memorize and to Forget") makes exactly this camera-flow-vs-scene-flow split and applies it *oppositely* in the two loops — the literature validation of the inverse-of-tracking spine (§4, §5).

---

## 2. How DDS-SLAM maps specifically — the mechanistic reality

This section dispels the **"check-frame, map-frame"** mental model. Mapping is **NOT one-frame-in / one-frame-out.** All file:line below verified against the working copy.

### 2.0 The run loop (the "NOT check-frame map-frame" structure) — `ddsslam.py:957-1001`

Per frame `i` from a `DataLoader` (`num_workers=1`, one frame/batch):

- `i==0`: `first_frame_mapping(batch, first_iters)` then `rendering` (`ddsslam.py:980-983`).
- `i>0` (`ddsslam.py:986-1001`): `tracking_render` (pose) runs **ALWAYS**; but the mapper runs **only when `i % map_every == 0`** → `global_BA` → `current_frame_mapping` → `_deform_replay_step` (`ddsslam.py:990-993`, verified). A keyframe is added only when `i % keyframe_every == 0` (`:999-1001`); `rendering` only when `i % render_freq == 0` (`:995-996`).

So mapping is **gated, not per-frame**, and the **dominant mapper is `global_BA`** — a bundle adjustment that uniformly samples rays from the *whole keyframe DB* plus a small current-frame injection. `current_frame_mapping` is a separate, *usually-OFF* single-frame polish (`cur_frame_iters<=0` early-returns). **"Check-frame, map-frame" is wrong on three counts:** (1) ~80% of frames (CRCD `map_every:5`) trigger no map step at all; (2) when a map step fires it touches *many* keyframes, not the current frame; (3) the per-frame polish that *would* map "this frame" is dead by default on CRCD.

### 2.1 `first_frame_mapping` — `ddsslam.py:284-367`

Fixes `est_c2w[0]=GT` (`:299-300`). Loops `first_iters` (`:305`): zero `map_optimizer`, random-sample pixels of frame 0, gather rays (rgb/edge/depth, + optional dino/seg), build world rays from the *fixed* `c2w`, append `timestamp=0` if `dynamic`, `forward(notFirstMap=False)`, `get_loss_from_ret`, backward, `map_optimizer.step` (`:306-330`). **No pose optimization.** Frame 0 is forced into the KF DB (`:333`). `time_net` is in `map_optimizer`'s decoder group, but at `t=0` the field is anchored to Δx=0 (`scene_rep.py:217-218`), so it receives ~no gradient on frame 0.

### 2.2 `current_frame_mapping` — `ddsslam.py:369-446` (usually a DEAD no-op)

**EARLY-RETURN if `cur_frame_iters <= 0`** (`:382-383`) — so on the canonical/base CRCD config this is a **dead no-op**. When `>0`: takes the est pose for the current frame, builds a pose optimizer via `get_pose_param_optim(mapping=True)` (`:389`) and uses `self.cur_map_optimizer`. Loops `cur_frame_iters` (`:391`): samples current-frame pixels (`:395-411`, incl. optional baked `deform_dx`/trust targets), builds rays at the current pose, then either trains the field alone on `deform_teacher_loss` (teacher-only) or runs the render forward + render loss (optionally adding the teacher term), then `cur_map_optimizer.step` (`:444-445`). **Note:** the constructed pose optimizer (`:389`, zeroed `:392`) **never has `.step()` called** — current-frame pose is not updated by this path. `cur_map_optimizer` membership (`ddsslam.py:886-894`): decoder (incl. `time_net` **unless** `cur_frame_map_only`) + `embed_fn` (+`embed_fn_color` if two grids).

### 2.3 `global_BA` — `ddsslam.py:483-618` (THE dominant mapper)

**Pose-set assembly (`:493-518`):** `poses` = all KF poses 0,5,10,… (`:496`). If `<2` KFs, all KF poses fixed + current appended (`:501-504`). Else **KF[0] is FIXED** (gauge anchor, `:507`); KF[1:] are made optimizable (`:511/:516`); the current pose is either folded in (`optim_cur`, `:510-513`) or appended fixed (`:515-518`).

**Ray sampling per BA iter (loop `mapping.iters`, `:530`):** `sample_global_rays(sample)` draws rays **UNIFORMLY over ALL stored KF rays** (`:535`), returning rays + frame_ids; PLUS a current-frame injection `idx_cur = max(sample/len(KFs), min_pixels_cur)` random current-frame pixels, concatenated and **tagged `id=-1`** (`:538-541`). Rays → world via `poses_all[ids_all]` (`:553-554`). Timestamp channel = cat(KF ids, current id) (`:557-558`); `global_ba_time_fix` optionally normalizes (`:567-568`).

**Updates:** `forward` (`:571`), `loss = get_loss_from_ret(..., smooth=True)` (`:573`, adds SDF-TV smoothness), backward retain_graph (`:575`). `map_optimizer.step` on `map_accum_step` cadence after `map_wait_step` (`:577-583`); `pose_optimizer.step` on `pose_accum_step` cadence (`:585-602`). **WHICH params update:** (1) the **MAP** via `self.map_optimizer` = hash `embed_fn` (+`embed_fn_color`) + decoder main group + `time_net` group (`ddsslam.py:880-883`, `_dec_groups :862-871`) — so **the deformation field `time_net` receives global_BA render gradients by default**; (2) **KF poses KF[1:]** and optionally current, written back to `est_c2w_data` (`:604-610`); (3) `flow_track.freeze_ba` re-applies gate-frozen KF poses post-BA (`:614-618`). **KF[0] is never moved.**

### 2.4 The keyframe DB — `model/keyframe.py` (co-visibility windowing is DEAD CODE)

Storage: `rays` buffer `[num_kf, num_rays_to_save, 8]` where width-8 = `[dir3, rgb3, depth1, edge_semantic1]` (`keyframe.py:14-16,62`). **No DINO/seg/deform stored** (v2 samples DINO on-demand per frame). `add_keyframe` (`:56-75`): packs the 8 channels, sub-samples (random, or `filter_depth` keeps depth in `(0, depth_trunc]`), attaches `frame_id`. `sample_global_rays` (`:77-87`): the **only** sampler used by `global_BA` — flattens rays over ALL KFs and samples uniformly (no recency/co-visibility weighting). **DEAD CODE:** `sample_global_keyframe(window_size, n_fixed)` (`:89-109`) and `sample_overlap_keyframe` (NICE-SLAM co-visibility frustum overlap, `:111-181`) are **defined but never called** from `ddsslam.py` (confirms the FM3 "co-vis windowing is dead" memory; `mapping_window_size` in config is read nowhere). **Effective behavior:** an ever-growing flat pool, uniformly sampled — older frames are never down-weighted.

### 2.5 The mapping losses — `scene_rep.py:forward` (:493-711) + `model/utils.py`

Assembled in `get_loss_from_ret` (`ddsslam.py:227-282`):
`loss = rgb_weight*rgb + depth_weight*depth + sdf_weight*sdf + fs_weight*fs + 0.5*rgb_weight*edge + 0.1*rgb_weight*edge_semantic` (+ optional `def_reg`/`nll`/`whatkind` behind default-0 weights; + SDF-TV `smooth` when `smooth=True`, `:278-281`).
- **Depth-anchored sampling:** `render_rays` (`:373-396`) places `n_range_d` samples in `target_d ± range_d`; invalid-depth rays fall back to `near..far`; perturb jitter (`:399-403`).
- **rgb/depth:** `compute_loss` (`utils.py:110-155`) MSE over valid-depth pixels, with an `rgb_missing` weight for no-depth pixels (`scene_rep.py:555-557,587-589`).
- **sdf+fs:** `get_sdf_loss`/`get_masks` (`utils.py:80-198`): front_mask (free space → +tr), sdf_mask (`|z-d|<=tr` → signed target), inverse-class-frequency weights; `truncation = trunc * sc_factor`.
- **Inc-2 σ² down-weight (`:570-575`) and flow `track_ray_w` (`:582-585`) apply ONLY when `tracking=True`** → they **do not touch any mapping path** (mapping/BA/current-frame forwards pass `tracking=False`). *This asymmetry is exactly what §4 inverts.*

### 2.6 The deformation field's role in mapping — `scene_rep.py:194-231` (verified)

**WHERE it warps:** `run_network` — for EVERY forward (mapping, BA, current-frame, tracking, render) when `config['dynamic']`: split `pts`/`frame_time`, `embed_time(t) ⊕ embed_fre_pos(x)` → `time_net` → `vox_motion` Δx (`:205-208`), optional hardbound tanh (`:212-214`), `t=0` anchor zeros Δx (`:217-218`), then `inputs_flat = pts + vox_motion` (`:230`, verified). `def_reg = (vox_motion**2).mean()` (`:231`). `deformation_off` hard-zeros Δx (`:202-203`). **So the field warps sample positions BEFORE the hash/SDF/color query, in global_BA and current_frame_mapping identically to render.** `time_net` is a bias-free MLP, `embed_time(L=12) ⊕ embed_fre_pos(L=12) → hidden_dim_time → 3` (`decoder.py:6-53`).

**WHEN field is trained vs map:** By default the field is trained by the **same render gradient as the map**, because `time_net` sits in `map_optimizer`'s decoder group (`ddsslam.py:866-870`, included in `map_optimizer :880` and `cur_map_optimizer :891`). **This is the path that collapses Δx→0** (the field "explains" nothing because the static map already minimizes the render loss).

**The teacher/replay (default-off):** `deform_teacher_loss` (`scene_rep.py:253-271`) replicates the field forward and regresses `D(Xk,t)` toward baked Δx* targets, trust-weighted. Invoked in `current_frame_mapping` (`:419-438`) and `_deform_replay_step` (`:935-955`), which trains the field on a *causal* replay buffer via a dedicated `self.field_optimizer`. `deform_field_teacher_only` (`:873-877`) removes `time_net` from `map_optimizer` so render can never collapse it; `field_optimizer` (`:900-906`) trains it independently. All gated (`deformation_sup_weight>0`, `deform_replay_iters>0`), default OFF → base is bit-identical and the field is render-trained-only.

### 2.7 Optimizer layout — `create_optimizer ddsslam.py:853-906`

- `map_optimizer` (`:880-883`): decoder-main + `time_net` group (unless `deform_field_teacher_only`) + `embed_fn` (+`embed_fn_color`). Drives `first_frame_mapping` AND `global_BA`.
- `cur_map_optimizer` (`:886-894`, only if `cur_frame_iters>0`): same, `time_net` excluded iff `cur_frame_map_only`. Drives `current_frame_mapping`.
- `field_optimizer` (`:900-906`, only if `deformation_sup_weight>0`): Adam over `time_net` alone. Drives the replay teacher.
- pose optimizers are **ephemeral**, built per-call (`get_pose_param_optim :474-481`); poses are **never** in the map optimizers.

---

## 3. Why DDS-SLAM fails on CRCD (~22 PSNR, blurry on deformation) — ranked causes

CRCD render failure is ~half a real under-training deficit, ~half structural, plus a measurement artifact. Ranked by attributable PSNR/structural impact.

**RANK 1 — MAP UNDER-TRAINING (biggest controllable; a copied-from-StereoMIS sparse schedule).** `configs/CRCD/crcd.yaml` sets `cur_frame_iters:0` (`:25`), `iters:20` (`:24`), `keyframe_every:5` (`:30`), `map_every:5` (`:31`). `configs/Super/Super.yaml` sets the dense `cur_frame_iters:100` (`:27`), `iters:200` (`:15`), `keyframe_every:1` (`:20`), `map_every:1` (`:21`). Three compounding deficits: (a) `current_frame_mapping` early-returns (`ddsslam.py:382-383`) → the just-tracked frame is **never directly mapped** (the single biggest CRCD lever); (b) 20 BA iters/keyframe vs 200; (c) `map_every:5` × `keyframe_every:5` → ~80% of frames contribute *zero* direct map supervision. Net ~order-of-magnitude fewer optimizer steps/region than SemSup. Provenance: the `0` is an upstream StereoMIS default hand-copied in (commit `096d253`), not ours; the authors used `100` on their own data. (Evidence: configs cited; `ddsslam.py:382-383`; memory `project_mapping_free_wins_audit_20260621`.)

**RANK 2 — UN-ROUTED DEFORMATION-FIELD SMEAR (the prime deformation-blur driver).** The now-ALIVE field is applied to *every* sampled point with **no per-region gate**: `vox_motion = self.time_net(h)` then `inputs_flat = pts + vox_motion` (`scene_rep.py:208,230`), with the routers `oracle_w`/`surf_w` both `None` on the canon path. A field tuned to explain moving tissue therefore *also drags rigid bg/tool*, blurring them; the map co-adapts to a warp it cannot localize. Memory pins the un-routed cost at **~4 PSNR** (22→28 ceiling), of which `cur_frame_map_only` already recovered ~2 (22.2→24.1) by isolating the field optimizer. (Evidence: `scene_rep.py:204-231`; memory `project_combine_routing_model_20260619`, `project_mapping_free_wins_audit_20260621`.)

**RANK 3 — DEAD KEYFRAME WINDOWING → tail frames drowned (the forgetting mechanism).** `global_BA` draws rays *only* via `sample_global_rays` (`ddsslam.py:535`), uniform over ALL keyframes (`keyframe.py:77-87`); the recency/co-visibility selectors (`:89`, `:112`) are never called. As the DB grows (c1_001 = 360 frames / `keyframe_every` 5 ≈ 72 KFs), recent KFs become an ever-smaller fraction of sampled rays → newly-revealed/deforming geometry is under-fit → tail blur. Compounded by a single incremental pass with one final ckpt (`ddsslam.py:1012-1014`) and **no final BA sweep / post-hoc re-render**. (Evidence: cited; memory FM3.)

**RANK 4 — DEPTH-ANCHORED LIVE-JPEG ILLUSION (metric artifact, not the cause — but CRCD's ~22 is already near-honest).** All forwards sample in a band centered on the MoGe GT depth (`scene_rep.py:385-386`), and the `.jpg` is written live right after mapping that frame (`ddsslam.py:995-996`). On SemSup (`cur_frame_iters:100`) this overfits-fresh → inflated in-training ~30 vs honest post-hoc ~21.7 (the 7 dB gap). On CRCD `cur_frame_iters:0` means *no* fresh overfit → the live ~22 is already near honest. **Implication:** the CRCD number is real; but the methodology fix (headline POST-HOC re-render from final ckpt) is still required so a working field/routing can be *credited*. (Evidence: cited; memory `project_mapping_free_wins_audit_20260621`.)

**RANK 5 — STRUCTURAL CRCD FLOOR (specular / tool-occlusion / sparse-texture).** Both literal upstream configs plateau at LPIPS ~0.52 regardless of knobs — specular liver, tool occlusion, low texture that the SDF+volume renderer cannot represent. A hard floor: PSNR/SSIM improve with iters/hash, but the structural LPIPS component needs a representation change (Gaussian Splatting), not config tuning. DON'T-CHASE for knob work. (Evidence: memory `project_mapping_free_wins_audit_20260621`, `project_crcd_c1_001_T0_results_20260604`.)

### 3.1 The mechanism of "BLURRY ON DEFORMATION"

Two mechanisms, in contribution order:

**(A) STATIC-MAP-AVERAGES-MOTION (dominant).** `global_BA` fits ONE SDF+color field to rays pooled *uniformly* from many keyframes captured at *different deformation states* (`keyframe.py:77-87`). For a surface point whose true position/appearance changes over time, the least-squares fit settles on the **time-average** → a smeared, low-frequency reconstruction *exactly where tissue deforms*. The deformation field is *supposed* to absorb this time variation, but it is excluded from / under-trained in the map path (`deform_field_teacher_only` keeps it out of BA; `cur_frame_iters:0` removes the per-frame sharpening that would let the map co-adapt) AND BA's uniform sampling gives no recency weight → the static field wins and blurs.

**(B) FIELD-WARPS-EVERYTHING (secondary).** Where the field IS active it applies one un-routed Δx to bg+tool+tissue (`scene_rep.py:230`) — it cannot model tissue motion without also displacing rigid content → global low-frequency blur + tool ghosting.

**The fix for both** = route the field (tissue→field/warped, bg→camera/sharp, tool→own-SE(3)/excluded) using the per-region disagree-map from the flow-agreement gate, validated with `field_warped_pin_epe.py` (render/ATE are field-blind). This is §4.

---

## 4. THE BRIDGE — the tracking flow-signal INVERTED for mapping (centerpiece)

### 4.0 The inversion principle (one sentence, then the wiring)

> Tracking uses the per-region flow-agreement signal to **decide trust in pose** (regions that agree with one rigid camera motion are TRACKED; still/scene/tool regions are FIXED/down-weighted via `track_ray_w`, `scene_rep.py:582-585`). **Mapping makes the SAME measurement and takes the OPPOSITE action:** the same per-region disagree-map becomes the **WHAT-MOVES router** for the map.

Each map ray is classified into one of THREE handlers, the same flow signal selecting the handler:

- **AGREES with rigid camera (parallax-consistent) → STATIC** → map normally, **field EXCLUDED** (sharp, no warp). *Where tracking TRACKS, mapping maps-as-usual.*
- **DISAGREES (Sampson-high) ∩ what-kind == tissue → DEFORMING TISSUE** → route to the **alive field** (Δx ON for these rays), map carefully. *Where tracking DOWN-WEIGHTS pose, mapping UP-WEIGHTS the field's modelling of that ray.* — the user's "down-weight pose → up-weight mapping" / "scene deforming → map carefully."
- **DISAGREES ∩ what-kind == tool → TOOL** → **exclude from the static SDF map** (v0) or give own SE(3) (v1); never let the tool corrupt static geometry. — the user's "tool moving → map it in its new position, not smeared into the liver."

Router output = a single per-ray vector `route_w` ∈ {static, tissue, tool} (hard {0,1}³ or soft summing to 1). **The seam that already exists** — `run_network`'s `oracle_w` (`scene_rep.py:223-225`, verified: `vox_motion = vox_motion * ow`) — is the *exact* insertion point: `route_w_tissue` feeds `oracle_w`; the SDF/rgb losses are masked by `route_w_static` (+tissue); `route_w_tool` excludes tool rays from static-map updates.

### 4.1 The signal — compute once per mapped frame, reuse the tracking probe

Reuse `Addons/motion/flow_track.py` **unchanged** for measurement. `agreement_gate(ref, cur, dino_g, …)` returns `(cam_mag, disagree_frac)` frame-level (`flow_track.py:89-125`); the per-pixel Sampson residual + per-region (KMeans-on-DINO) labels are computed *inside* it (`:111-116`) but currently **collapsed to a scalar**. Mapping needs the per-pixel map.

**ACTION (new, small ~20 lines):** add `flow_track.region_route(ref_bgr, cur_bgr, dino_g, raft, tf, device, seg=None, deadband, n_groups)` returning a per-pixel `[H,W]` integer route map `{0=static, 1=tissue, 2=tool}` (or soft `[H,W,3]`): fit F to flow (existing `_sampson` path), per-DINO-region `median(Sampson) > deadband` ⇒ moving; intersect the moving-mask with the what-kind label (seg if supplied, else DINO-KMeans cluster identity mapped via the `dino_separability_probe` head). It reuses the exact code in `agreement_gate:111-125` and simply *stops collapsing to a fraction*.

### 4.2 Concrete wiring — insertion points (file:line, what receives the weight)

Mirror the tracking flow plumbing (init in `torch.random.fork_rng`, gated default-off, causal ref buffer) so parity stays clean.

**(a) State + signal capture.**
- `ddsslam.py:80-96` (`__init__`): already loads RAFT+DINO when `flow_track.enable`. Add `self.map_route_on = config.get('map_route',{}).get('enable',False)` reusing `self._raft`/`self._dino` (load them if the flow gate is off but map_route on). Add `self._map_route_buf` (deque, same `ref_stride` pattern as `self._flow_buf`) and `self._route_map = None`.
- In `run()` at `ddsslam.py:990`, **before** `self.global_BA(batch, i)`:
  ```python
  if self.map_route_on:
      self._route_map = self._compute_route_map(batch, i)   # [H,W] or [H,W,3], CPU
  ```
  New `_compute_route_map` mirrors `tracking_render:674-718` (causal ref from buffer, `_rgb_to_bgr_u8 :636`, `dino_grid`) but calls `flow_track.region_route` and returns the dense map. Store on `self` so global_BA / current_frame_mapping / replay all read the SAME map for frame `i`.

**(b) `global_BA` — the dominant smear source (`ddsslam.py:483-618`).** Insertion at the ray-sampling block (`:535-571`). Because BA samples rays from MANY frames, a single current-frame route map does not align → two sub-options:
- **OPTION B1 (cheap, current-frame only):** the current frame's rays are appended at `:538-540`. Build `route_w_cur` from `self._route_map` at those pixels and pass it as `oracle_w` for the current-frame sub-batch only; keyframe rays stay un-routed (their warp is mostly anchored). Minimal change; directly addresses "the current frame's tool/deformation must not smear the map."
- **OPTION B2 (full fix):** store a per-keyframe route map at `add_keyframe` time. Extend `keyframe.py:14` ray width 8→9 (or a parallel `self.route` buffer `[num_kf, num_rays, 1]`) capturing the route id per saved ray; `add_keyframe` (`:56-75`) packs it; `sample_global_rays` (`:77-87`) returns it alongside ids. Then build `route_w` for the whole batch and pass as `oracle_w`.
- **WIRING POINT:** `ddsslam.py:571` `ret = self.model.forward(rays_o, rays_d, ...)` → add `oracle_w=route_w` (forward already accepts the seam; §4.3). Δx for static/tool rays is zeroed (sharp); kept for tissue.
- **Tool exclusion from static geometry:** at the `get_loss_from_ret` call (`:573`), pass `weights = route_w_static` into `compute_loss` via the **same `weights=` kwarg the Inc-2 path uses** (`scene_rep.py:587-589`) so tool rays contribute ~0 to the static SDF/rgb loss.

**(c) `current_frame_mapping` — the cleanest insertion (route map == this exact frame, `ddsslam.py:369-446`).**
- At the sampling block (`:395-411`) you already have `indice_h`/`indice_w`. Gather `route_w = self._route_map[indice_h, indice_w]` (mirror the `target_dino` gather at `:403`).
- Pass `oracle_w = route_w_tissue` to `forward` at `:433` so the field warps ONLY tissue rays during the sharpen.
- `cur_frame_map_only` (`:425`, `:887-894`) already isolates the field from this optimizer (field stays alive via replay; MAP co-adapts). **Routing + `cur_frame_map_only` together = the render recovery:** static rays sharpen with no warp, tissue rays sharpen with the alive warp, tool rays excluded. Literal realization of `project_combine_routing_model:47`.
- Tool exclusion: multiply rgb/depth/sdf loss by `route_w_static` via the `compute_loss weights=` kwarg, as in (b).

**(d) The teacher REPLAY (`_deform_replay_step :935-955`, `_buffer_deform :908-927`).** The field's Δx* targets should only be buffered for **tissue** rays. At `_buffer_deform:923-924`, multiply trust `w` by `route_w_tissue` (or drop static/tool rays). One line: `w = w * route_w_tissue` before append. This keeps the alive field's supervision on-manifold (tissue only) and prevents the global re-collapse/blur seen in naive co-adapt (`project_deform_gauge_bug_20260620`).

**(e) Keyframe-ray sampling (FM3 free-win, orthogonal).** `sample_global_keyframe`/`sample_overlap_keyframe` (`keyframe.py:89-181`) are dead code; wire `n_fixed` recency as a *separate* A/B. Compounds the tail-blur fix but is independent of the router.

### 4.3 The `oracle_w` seam — already half-built (the key reuse)

`run_network` ALREADY accepts `oracle_w` and multiplies Δx per-ray (`scene_rep.py:223-225`, **verified**: `vox_motion = vox_motion * ow`); `render_rays` threads it (`:373`, `:427`); `forward` computes it from `target_edge_semantic` only when `oracle_routing` and not `render_only` (`:511`).

**ACTION:** generalize `forward` to accept an explicit `route_w` (default `None`), preferred over the edge-prior:
```python
oracle_w = route_w if route_w is not None else (target_edge_semantic if oracle_routing ... else None)
```
The whole field-routing machinery is then **ONE new kwarg** threaded from the 3 mapping callers into the existing, tested `oracle_w` path. No new forward logic. `route_w=None` ⇒ byte-identical base (the off path is already proven by the Inc-0 gate `Addons/regression/test_inc0_bitidentical.py`).

### 4.4 Reconcile with the COMBINE field-routing, the TriGauge what-kind axis, and the pin-EPE judge

- **Combine consistency.** `project_combine_routing_model_20260619:47` says "route the field — static→camera, tool→SE(3), tissue→field" and "the flow agreement-gate's per-region disagree-map IS the what-moves router — same signal as the tracking gate, now applied to the FIELD's per-ray Δx in mapping." This design is its literal implementation: disagree-map → `route_w` → `oracle_w` → per-ray Δx in current_frame_mapping/global_BA. **CONFIRMED consistent.**
- **WHAT-MOVES vs WHAT-KIND (the two TriGauge axes).** Flow disagree-map = WHAT-MOVES (binary moving/still, parallax-aware via Sampson). The DINO/seg head = WHAT-KIND (tissue/tool/bg). The router needs BOTH: moving∩tissue→field; moving∩tool→exclude/SE(3); still(any)→static. Per the locked model, what-kind must be **LEARNED DINO** (`dino_separability_probe` head: kmeans 0.475, supervised-LR 0.875 held-out), **never** hard `if class==tissue` rules; seg is a droppable training prior.
- **Tool handler.** v0 = **exclude-from-static-map** (mask the loss) — stops smear cheaply. v1 = own SE(3) (re-render the tool in its new pose) — defer.
- **Validation = the field-sensitive judge, NOT render/ATE** (both proven field-blind: a dead field renders fine). Validate routing with `Addons/eval/field_warped_pin_epe.py` (`--deform_dir`): **pin-EPE must stay ALIVE** (reduction ≫ shuffled-time control) **WHILE post-hoc PSNR rises toward 28.** *Over-gate* (route too few rays to the field) ⇒ field dies (pin-EPE collapses). *Under-gate* (route everything) ⇒ render stays smeared (~22-24). The gap decomposition (`project_deform_gauge_bug:0`): ~2 PSNR was disabled-sharpening (already recovered by `cur_frame_map_only`); the remaining ~4 is un-routed smear — **this routing recovers that ~4.**

### 4.5 The collapse caveat (do not re-introduce)

`global_BA`'s render gradient collapses the field to |Δx|=0 (already handled via separate `field_optimizer` + `cur_frame_map_only`). When the mapping-half routing makes `current_frame_mapping` co-adapt the MAP to the field, **confirm the field is NOT re-added to a render-gradient optimizer** (keep `deform_field_teacher_only`/`cur_frame_map_only` semantics). Tissue-only teacher buffering (4.2d) is the second guard against global re-collapse.

---

## 5. External techniques we can adopt (with citations)

**KEY FINDING:** "mapping = inverse of tracking" is RIGHT, but the literature mostly does the *opposite* — that is the white-space. **NRGS-SLAM** (arXiv:2602.17182, *read in full*) and **WildGS-SLAM** (CVPR 2025) DOWN-WEIGHT deforming/uncertain regions in the map loss — but **only in the POSE/BA sub-loss**; a deformation FIELD (per-Gaussian basis / canonical+deform MLP) renders those regions at *full weight* in the field-update step. So the **up-weight intuition is correct for the FIELD/map-content sub-loss** (our Arm-2 replay), **NOT** the pose/BA sub-loss. NRGS tracking: `min_T Σ(1-M_def)L_ph`; mapping-BA: `min_T Σ_k Σ(1-M_def)L_ph`. WildGS mapping: `L_render = (λ·L_color + λ·L_depth)/β²`.

**(A) Tool-mask-guided training — SurgicalGaussian (highest-EV, cheapest win).** `L_color = ||(I-Î)(1-M)||₁`, `L_depth = ||(D-D̂)(1-M)||₁`, `M=1` on tool. We have CRCD label `3=Tool` (all 3 datasets have masks). Apply `(1-M_tool)` to mapping rgb+depth in `current_frame_mapping`/`global_BA` (= §4.2 tool exclusion). Depth-init collects occluded-then-revealed pixels `P* = ∩(1-M)` → fills holes behind tools = "tool moves → map revealed tissue." Local-rigidity reg `L_pos`/`L_cov` on K=5 NN ≈ near-free `time_net` smoothness (and fixes the noted `smoothness()`→static-SDF mis-wiring). *Validate POST-HOC render + pin-EPE.* (arXiv:2407.05023, MICCAI 2024.)

**(B) σ²-routed mapping — WildGS-SLAM (exact Inc-1-on-map template).** `L_render = (λ·L_color + λ·L_depth)/β² + λ·L_iso`, with a **log-β floor** and **DETACHED gradients** between map and uncertainty-MLP (the load-bearing stop-grad). Uncertainty MLP trained on-the-fly on 3D-aware DINOv2, supervised by SSIM + Metric3D-depth (we use MoGe-2). **BUT σ²-routed mapping is deformation-BLIND** (appearance-preserving motion → low residual → low σ²) → must route by **MOTION** (flow/field), not photometric σ² — the *strongest external corroboration of our spine.* Pair with (C) for deformation. (arXiv:2504.03886; NeRF-on-the-go CVPR2024 cited within.)

**(C) Motion-routed mapping CAPACITY — the contribution, literature-backed.** Motion-aware-densification work explains BLURRY-ON-DEFORMATION: dynamic regions get far fewer effective map updates → blur; fix = up-weight capacity/iters/samples on scene-moving regions. DDS analogue: in `current_frame_mapping`/`global_BA`, up-weight `cur_frame_iters`/`render_rays` samples/loss on flow-agreement scene-moving regions; exclude tools; route tissue→`time_net`. Static/dynamic capacity decoupling: full time-conditioned capacity ONLY in dynamic regions (gate `time_net` per-region — exactly §4's `route_w_tissue`). (SDD-4DGS arXiv:2503.09332; Hybrid-3D-4D-GS arXiv:2505.13215; SharpTimeGS arXiv:2602.02989.)

**(D) One unified deformation-probability signal for BOTH loops — NRGS Boltzmann posterior.** `w_d = sigmoid(log(π_d/π_r) + β(E_R − E_D))`, β=200, BCE-supervised; soft-gates the map `A_i(t) = A_canonical + w_d·δA_i(t)`. Field residuals computed only for `w_d>ε` (rigid frozen) = **matches `cur_frame_map_only`.** Compute `E_R − E_D` once via the `deformation_off` pass (our E_rigid−E_deform gate). Use it to feed BOTH tracking pose down-weight (have) AND `time_net`/map gating (new). **Adopt `sigmoid(β(E_R−E_D))` as the σ²-teacher target, NOT raw ReLU.** Gauge-correct and scale-robust (critical for up-to-scale MoGe depth). NRGS has **no tool handling** (a weakness to beat). (arXiv:2602.17182, read in full.)

**(E) Dynamic-NeRF-SLAM masks + surgical flow-mapping precedents.** RoDyn-SLAM fuses flow + semantic motion masks to exclude dynamics; DDN/NID/DNIV/TivNe exclude dynamic rays and fill voids. **Surgical twist:** exclude TOOLS but *route TISSUE-deformation to the field* (two handlers for two dynamics — unlike generic dynamic-SLAM which removes all motion). **EndoFlow-SLAM** (MICCAI 2025, arXiv:2506.21420) is the *published* surgical flow-constrained-mapping precedent — optical-flow loss as a geometric mapping constraint, robust to breathing, StereoMIS-tested + depth reg — **read before finalizing the mapping section.** **EndoSurf** (MICCAI 2023, arXiv:2401.11535) = our direct architectural twin (canonical SDF + deform field), with *zero uncertainty / no SLAM* = our white-space. **Deform3DGS** (arXiv:2405.17835, 1D-Gaussian basis, 1-min train) = fallback if `time_net` proves weak. (RoDyn arXiv:2407.01303; NID arXiv:2401.01189; DDN arXiv:2401.01545.)

**Same-lab cite — SNI-SLAM** (CVPR 2024, IRMVLab, arXiv:2311.11016): cross-attention rgb+geo+semantic fusion = the principled form of the deferred DINO-query fusion for the TriGauge what-kind head (we already falsified flat concat; gate cross-attention on the combine showing signal first).

**Our differentiation vs the closest prior art (NRGS):** SDF vs GS; continuous NLL σ² in *both* loops vs binary BCE `w_d`; **3-way TriGauge + explicit tool handling vs 2-way (NRGS treats tools as deformable)**; honest field-sensitive pin-EPE judge + post-hoc render methodology.

---

## 6. PROPOSED PLAN — smallest first mapping experiment

**Goal:** test the **inverted-flow mapping-routing** at minimal blast radius, flag-gated default-off, judged by the **field-sensitive pin-EPE judge + honest POST-HOC PSNR** — composed with the already-validated tracking gate and the alive field.

### 6.1 The smallest experiment (E0) — `current_frame_mapping` routing only

`current_frame_mapping` is the cleanest seam (route map == this exact frame; §4.2c) and the cheapest viable test of the whole thesis.

**Build (all default-off, parity-safe):**
1. `flow_track.region_route()` — the ~20-line per-pixel `{static,tissue,tool}` map (§4.1), reusing `agreement_gate:111-125`.
2. Generalize `forward` to accept `route_w` preferring it over the edge-prior (§4.3) — `route_w=None` ⇒ byte-identical base.
3. `__init__` guard `self.map_route_on` + causal `self._map_route_buf` reusing RAFT/DINO (§4.2a); compute `self._route_map` in `run()` before the map step (`ddsslam.py:990`).
4. In `current_frame_mapping` (`:395-433`): gather `route_w` at `indice_h/w`; pass `oracle_w=route_w_tissue`; mask tool rays from rgb/depth/sdf via the `compute_loss weights=` kwarg (§4.2c). Turn on `cur_frame_iters>0` + `cur_frame_map_only` for the routed arm (CRCD base has it at 0; SemSup at 100).
5. Tissue-only teacher buffering: `w = w * route_w_tissue` at `_buffer_deform:923-924` (§4.2d).

**Parity gate FIRST:** `Addons/regression/test_inc0_bitidentical.py` must pass with `map_route.enable=false` before any run (RNG-after-build + param count + state-dict keys).

### 6.2 A/B ladder (each FROM base, default-off flag `map_route.enable`)

```
base → +route(current_frame, cur_frame_map_only)  [E0]
     → +route(global_BA B1: current-frame sub-batch only)
     → +route(global_BA B2: per-keyframe route storage)
     → +tool-exclude(static-loss mask)
     → +tissue-only-teacher-buffer
```
Each rung is one isolated change, committed separately (personality rule). **B1 before B2:** if E0 + B1 already recover the ~4 PSNR, B2's per-keyframe storage may be unnecessary.

### 6.3 Judging (the arbiter — both diagnostic sets, by default)

**NUMERICAL:**
- **Field-sensitive pin-EPE** (`Addons/eval/field_warped_pin_epe.py`, `--deform_dir`) — the *primary* gate. Pin-EPE must stay **ALIVE** (reduction ≫ shuffled-time control) for the routed arm. Over-gate kills it; this catches it.
- **Honest POST-HOC PSNR/SSIM/LPIPS** (re-render from final ckpt, **never live JPEGs**) via `Addons/eval/eval_rendering.py`. Target: routed arm PSNR rises toward 28 while pin-EPE stays alive. *(Methodology note: a post-loop re-render path does not currently exist in `ddsslam.py` — it must land before any iters/cur_frame_iters A/B is judged, since raising them reintroduces the live-JPEG illusion. This is a prerequisite, tracked in §6.5.)*

**VISUAL (auto-ship to Drive, co-located):** the canonical 6-panel inline video (`Addons/viz/generate_video.py`) + σ²/depth panels + a route-map overlay panel (static/tissue/tool) so over/under-gating is visible.

**Dataset routing:**
- **Render arbiter = SemSup** (PSNR/SSIM/LPIPS; ATE fictional) with the green-pin GT + MoGe-2 corpus (`MoGe2_trail3_20260608`) — this is where pin-EPE + render are both valid.
- **CRCD** for the route-DECISION sanity (the gate is ~91% correct vs GT) but **NOT the render arbiter here** (CRCD GT 55% held, deformation ⊥ awake-GT — the measurability confound). Use `Addons/eval/sim3_ate.py` (Sim3 + path-ratio + Pearson) only as the up-to-scale tracking sanity, never the rigid `output.txt`.
- **n=3 seeds always** (seed coin-flip is real; a win must clear base seed-std).

### 6.4 How it composes with the validated tracking gate + the alive field

- **Same sensor, opposite polarity, ZERO conflict.** The tracking gate already consumes the disagree-map to FIX pose on still/scene frames (`flow_track.freeze_ba`); E0 consumes the *same* map (computed once per mapped frame, stored on `self`) to *route the field's Δx* in mapping. Tracking touches only `est_c2w`; mapping touches only the map/field — the decoupling (§1.5) means they never fight over a parameter.
- **The field stays alive by construction.** `cur_frame_map_only` keeps `time_net` out of the render-gradient optimizer (field trained by `field_optimizer`/replay); routing only changes *which rays* the field is allowed to warp. So E0 cannot re-trigger the |Δx|=0 collapse.
- **Combine readiness.** E0 is the literal first brick of the combine field-routing (`project_combine_routing_model:47`). It is judged by the field judge that already decided the field is alive (`project_deform_gauge_bug_20260620`), so the result plugs directly into the COMBINE step (does fixing add value on top of improving).

### 6.5 Risks / prerequisites / kill-criteria

- **PREREQ:** post-hoc re-render path (currently nonexistent) before any iters/cur_frame_iters A/B — else PSNR is the live-JPEG illusion.
- **KILL E0 if:** parity gate fails with the flag off (non-negotiable); OR routed pin-EPE collapses below the shuffled-time control (over-gate, field dead) AND no `deadband` setting recovers both pin-EPE *and* render; OR routed post-hoc PSNR does not clear base seed-std (n=3).
- **Soft vs hard routing** (`{0,1}` handler assignment vs soft `(w_static,w_tissue,w_tool)`) is a follow-on A/B (soft avoids edge artifacts but may re-introduce mild smear) — start hard for interpretability.
- **What-kind at inference:** deploy the **in-domain DINO head (0.93)** per the ARM4 open decision, with DINO-kmeans cluster identity as the fallback when seg is absent; seg-prior is train-time-only and droppable.

---

### Open questions carried forward
1. **B1 vs B2 for `global_BA`** — does E0 + B1 recover the ~4 PSNR, or is per-keyframe route storage (B2) needed because the keyframe-ray smear is large?
2. **Tool v0 (exclude) vs v1 (own SE(3))** — is exclusion sufficient for the thesis contribution given CRCD/SemSup tool prevalence?
3. **Replay-buffer windowing as a shared fix** — does wiring `sample_global_keyframe(n_fixed)`/recency simultaneously fix tail blur AND give the field a windowed replay ("Learn to Memorize and to Forget" suggests the motion signal should also control what the buffer FORGETS)? Currently deferred (`project_dds_fundamentals_deferred_20260620`).
4. **SDF vs Gaussian-Splatting substrate** for the CRCD LPIPS 0.52 structural ceiling — "needs architecture not knobs," out of current scope (EndoGaussians/SAGS line).
5. **Scope the up-weight to the FIELD sub-loss only** — verify in `current_frame_mapping`/`global_BA` exactly which loss term each deforming ray contributes to (pose-coupled vs field-only) *before* wiring the up-weight, to avoid the NRGS/WildGS failure mode of deformation corrupting pose.

**Files this plan creates/touches:** `Addons/motion/flow_track.py` (+`region_route`), `model/scene_rep.py` (`forward` `route_w` kwarg; `oracle_w` path unchanged), `ddsslam.py` (`__init__` guard + `_compute_route_map` + `current_frame_mapping`/`global_BA`/`_buffer_deform` wiring), `model/keyframe.py` (B2 only: route buffer), a post-hoc re-render path (new), config block `map_route:` (default `enable:false`). Validation: `Addons/eval/field_warped_pin_epe.py`, `Addons/eval/eval_rendering.py`, `Addons/eval/sim3_ate.py`, `Addons/viz/generate_video.py`. Parity: `Addons/regression/test_inc0_bitidentical.py`.