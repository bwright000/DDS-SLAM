# GS C2 (flow-as-sensor) — Build-Ready Implementation Spec

Date: 2026-06-23. Target pipeline: **EndoGSLAM** (Gaussian-Splatting, NOT the NeRF-SDF DDS-SLAM).
All line numbers below are **verified** against the live source on 2026-06-23:
`c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/EndoGSLAM/scripts/main.py`
and `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/motion/flow_track.py` (200 lines).

This spec **folds in every build-breaking fix** from the three adversarial code/parity/test-validity passes.
Where a flaw could not be fully resolved at spec time it is marked **OPEN** in §10, not papered over.

---

## (0) Scope, novelty position, NRGS precondition

**Scope.** C2 = *flow-as-sensor* tracking gate. Per frame: dense RAFT flow vs a **causal** past frame →
fit ONE camera-consensus model (fundamental + Sampson) → pool the residual into DINOv2 patch regions →
**region-level rigid-motion AGREEMENT test**. Two consumables wired into EndoGSLAM tracking:
- **hard FIX**: skip the pose optimizer for this frame (keep the const-velocity init pose), *still map*, when the camera is STILL **or** the scene is DEFORMING.
- **soft per-pixel weight** `w[H,W]`: down-weight deforming pixels in the tracking loss.

This spec touches **only** the tracking path. It does not edit the GS render/rasterizer math.

**Novelty position (honest).** The **headline contribution is the region-consensus (agreement) hard-FIX arm only**.
Soft-weight-only and the DINO-free homography "pixel" fallback are **not novel** (Flow4DGS / WildGS-SLAM occupy them) — they ship as ablation/control arms, not as the claim.

**NRGS precondition (LOCKED, blocking).** Read **NRGS-SLAM (arXiv:2602.17182) IN FULL** and complete the
Analysis-E dry-run before writing any novelty claim in the thesis. NRGS does rigid/deform routing for GS-SLAM;
the agreement-gate's delta vs NRGS must be stated explicitly before publication. (Ref: memory
`project_arm2_validity_gate_contrast_20260618`, `project_combine_routing_model_20260619`.) Building/measuring C2 is
allowed before the read; **claiming novelty is not**.

---

## (1) Verified hooks (exact file:line + code)

`flow_track.py` — **free functions only, NO class** (verified, 200 lines). Signatures that the adapter must call:
```
L15  def load_raft(device, small=False):              ... L20  return m, w.transforms()     # 2-TUPLE
L46  def flow_residual(ref_bgr, cur_bgr, model, tf, device, ransac_thresh=1.0, max_fit=4000):
                                                       ... L58  return _sampson(...).reshape(H,W)   # numpy [H,W]
L61  def camera_motion(ref_bgr, cur_bgr, model, tf, device):  ... L67 return float(...)
L70  def load_dino(device, backbone='dinov2_vits14_reg'):     ... L73 return hub.load(...).eval()
L76  def dino_grid(rgb_bgr, dino_model, device):              ... L86 return tok.reshape(gh//14, gw//14, -1)  # numpy
L89  def agreement_gate(ref_bgr, cur_bgr, dino_g, raft_model, raft_tf, device,
                        n_groups=..., ransac_thresh=..., deadband=..., min_px=..., seed=...):
                                                       ... L125 return cam_mag, (dis / max(tot,1))   # (float, float) TUPLE — NO bool
L128 def region_route(...):  ... L190 return route    # per-pixel mask — NOT used by C2
L193 def residual_to_weight(resid, alpha=0.5, w_min=0.1, w_max=1.0, deadband=0.0):
                                                       ... L200 return np.clip(...).astype(np.float32)  # numpy [H,W]
```
**Consequences folded in:** `load_raft` → unpack `(model, tf)`. `agreement_gate` returns a **tuple**, the AND
decision is the caller's job. `residual_to_weight`/`flow_residual`/`dino_grid` return **numpy** → a torch
conversion at the boundary is mandatory (see §2). There is **no `GSFlowGate`, no `step`, no `to_bgr_u8`** in
`flow_track.py` — those are net-new (§3, §5).

`EndoGSLAM/scripts/main.py` — verified hooks:
```
L8-10   _BASE_DIR = dirname(dirname(abspath(__file__))); sys.path.insert(0, _BASE_DIR)   # only EndoGSLAM root on path
L198-200 def get_loss(..., tracking=False, mapping=False, do_ba=False, plot_dir=None,
                       visualize_tracking_loss=False, tracking_iteration=None):          # signature to extend
L246-248 depth_sq=...; uncertainty = depth_sq - depth**2; uncertainty=uncertainty.detach()
L258/259 mask = (curr_data['depth'] > 0); mask = mask & nan_mask & bg_mask              # mask is [1,H,W] (depth is [1,H,W])
L266     mask = mask.detach()
L268     losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()            # TRACKING depth residual
L270     losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()           # MAPPING — untouched
L272-275 color_mask = torch.tile(mask,(3,1,1)).detach(); losses['im'] = abs(im_res)[color_mask].sum()  # TRACKING rgb (LIVE path)
L277     losses['im'] = torch.abs(curr_data['im'] - im).sum()                            # tracking, no-sil branch (DEAD in our cfg)
L279     ... 0.8*l1 + 0.2*(1-ssim)                                                        # MAPPING rgb — untouched
L281-282 weighted_losses = {k:v*loss_weights[k]}; loss = sum(...)
L399-401 if "use_depth_loss_thres" not in cfg['tracking']: cfg['tracking']['use_depth_loss_thres']=False; depth_loss_thres=100000
L609     for time_idx in tqdm(range(checkpoint_time_idx, num_frames)):
L615     color, depth, _, gt_pose = dataset[time_idx]                                     # color is [H,W,3], 0-255 here
L619     color = color.permute(2,0,1) / 255                                               # <-- capture BGR BEFORE this
L630-632 if seperate_tracking_res: tracking_color = ...; tracking_color = tracking_color.permute(2,0,1)/255
L643-644 if time_idx>0: params = initialize_camera_pose(params, time_idx, forward_prop=cfg['tracking']['forward_prop'])
L650     if time_idx > 0 and not config['tracking']['use_gt_poses']:                      # tracking block head — wrap this
L652     optimizer = initialize_optimizer(...)
L656     current_min_loss = float(1e20)
L662     while True:
L665-669 loss, variables, losses = get_loss(..., tracking=True, ..., tracking_iteration=iter)   # call site — thread weight here
L677-680 if loss < current_min_loss: save candidate                                       # per-frame argmin — constant-rescale SAFE
L692-693 if iter==num_iters_tracking: if losses['depth'] < depth_loss_thres and use_depth_loss_thres: break  # DEAD (thres off)
L704-706 params['cam_unnorm_rots'/'cam_trans'][...,time_idx] = candidate                   # copy-best — inside the block
L707     elif time_idx > 0 and config['tracking']['use_gt_poses']:                         # GT-pose branch — unaffected
```
`utils/common_utils.py:7-19` `seed_everything` sets `random/np/torch.manual_seed` **but NOT**
`cudnn.deterministic` / `use_deterministic_algorithms` (the docstring's claim is false) → GPU runs are **not**
bit-reproducible run-to-run (governs §7 parity test design).
`configs/crcd/crcd_base.py:69` ships `forward_prop=True`; there is **no** `flow_track` block and **no**
`FWD_PROP/LR_TRANS` env knobs (the "mirror existing _FWD/_LRT pattern" premise from a draft is unfounded — they are net-new, §6).

---

## (2) Loss-weight insertion + mean-preserving math (literal diff)

The weight is computed once per frame (frame-constant, reused across tracking iters). It enters `get_loss` as a
new default-`None` kwarg and multiplies **both** tracking residuals **before** the boolean index `[mask]`/`[color_mask]`.

**FIX folded in (type + resolution).** `track_weight` arrives as **numpy `[H,W]`** at *native* dataset resolution.
The tracking residual is at *rendered/tracking* resolution `[1,Hr,Wr]`. Both the numpy→torch conversion **and** the
`cv2.resize` to `(Wr,Hr)` happen **caller-side inside `get_loss`** (only `get_loss` holds `depth.device` and `mask`).
The adapter (§3) returns raw numpy `[H,W]`; it never touches torch.

### 2a. Signature (L198-200)
```python
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False,
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False,
             tracking_iteration=None, track_weight=None):
```
`track_weight` is `None` on every mapping call and on every base/flag-off tracking call.

### 2b. Build `w_norm` ONCE, immediately after `mask = mask.detach()` (insert after L266)
```python
    use_tw = (tracking and track_weight is not None)
    if use_tw:
        import cv2  # cheap; only on the enabled tracking path
        Hr, Wr = depth.shape[-2:]                                   # rendered residual spatial shape
        w_np = track_weight                                         # numpy [H,W] float32 from GSFlowGate.step
        if w_np.shape != (Hr, Wr):
            w_np = cv2.resize(w_np, (Wr, Hr), interpolation=cv2.INTER_LINEAR)   # cv2 takes (W,H)
        wt = torch.from_numpy(w_np).to(depth.device).float().unsqueeze(0)       # -> torch [1,Hr,Wr]
        wt = wt.detach()                                            # sensor: zero grad into pose
        # --- MEAN-PRESERVING over the SAME mask that selects the residual ---
        m_cnt  = mask.sum().clamp_min(1)                            # mask is [1,Hr,Wr] bool
        w_mean = (wt * mask).sum() / m_cnt                          # masked mean over active px
        w_norm = wt / w_mean.clamp_min(1e-8)                        # masked-mean(w_norm) == 1 exactly
```
Derivation. Let `M={i: mask[i]=1}`, `|M|=m_cnt`, `w_mean = (1/|M|) Σ_M wt[i]`, `w_norm[i] = wt[i]/w_mean`. Then
`(1/|M|) Σ_M w_norm[i] = w_mean/w_mean = 1`. ∎ The RGB residual is selected by `color_mask = tile(mask,(3,1,1))`;
`w_norm` (`[1,Hr,Wr]`) broadcasts identically across the 3 channels, so its masked mean over `color_mask` is **also
exactly 1** — one scalar `w_mean` serves both residuals.

### 2c. Multiply into BOTH tracking residuals (BEFORE the index)
Depth (replace L268, leave L270 mapping untouched):
```python
        if tracking:
            depth_res = torch.abs(curr_data['depth'] - depth)      # [1,Hr,Wr]
            if use_tw:
                depth_res = depth_res * w_norm                     # elementwise, BEFORE [mask]
            losses['depth'] = depth_res[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
```
RGB (replace L272-277; touch the live sil branch and the dead no-sil branch for totality):
```python
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1)).detach()
        im_res = torch.abs(curr_data['im'] - im)                   # [3,Hr,Wr]
        if use_tw:
            im_res = im_res * w_norm                               # [1,Hr,Wr] broadcast -> [3,Hr,Wr]
        losses['im'] = im_res[color_mask].sum()
    elif tracking:
        im_res = torch.abs(curr_data['im'] - im)                   # full-frame, no mask (DEAD in our cfg)
        if use_tw:
            im_res = im_res * w_norm                               # NB: not full-frame-normalized; see §10 OPEN-D
        losses['im'] = im_res.sum()
```

### 2d. Why mean-preserving (corrected justification)
The early-stop **L692-693** reads tracking-depth-loss *magnitude* against `depth_loss_thres`. **In the operating
config that branch is DEAD** (`use_depth_loss_thres` defaults `False` at L399-401), so L693 is **not** the live
reason. The **live** reasons mean-preserving is still required:
1. **L677 argmin candidate-pick** is constant-rescale-safe *within a frame* (same weighting every iter) — but
   `residual_to_weight` returns `w = clip(1/(1+α·excess), w_min, w_max)`, whose masked mean is data-dependent and
   `< 1`, so an **un-normalized** `w` rescales total tracking-loss **frame-to-frame** and **arm-to-arm**, corrupting
   A/B loss comparability and any reported `losses`. Mean-1 normalization removes that drift.
2. **Robustness**: if a future config flips `use_depth_loss_thres=True`, the un-normalized weight would silently
   shift every frame's stop point. Mean-preserving makes the gate safe under that flip too.
Gradient effect: `∂loss/∂pose = Σ_M w_norm[i]·∂r[i]/∂pose`; mean-1 fixes total weight mass at `|M|` (same as uniform)
but redistributes it onto the rigid majority — "preserve magnitude, re-route gradient."

---

## (3) The `GSFlowGate` adapter (net-new; class + buffer + capture + resolution + modes)

**This is the ONLY net-new numerics-free orchestration code.** It is NOT in `flow_track.py` and must NOT be imported
from it. New file: `EndoGSLAM/Addons/motion/gs_flow_gate.py` (scaffolding in §5). It delegates **all** math to the
lifted `flow_track.py` free functions.

```python
# EndoGSLAM/Addons/motion/gs_flow_gate.py
import cv2, numpy as np
from collections import OrderedDict

class GSFlowGate:
    """Per-frame causal flow gate for EndoGSLAM tracking (flow-as-sensor).
    NET-NEW adapter; ALL numerics delegated to Addons/motion/flow_track.py free functions.
    Built once in rgbd_slam init IFF cfg['flow_track']['enable']."""

    def __init__(self, cfg_ft, device):
        import Addons.motion.flow_track as ft          # lazy: only under the enable branch
        self.ft, self.dev, self.cfg = ft, device, cfg_ft
        self.gate      = bool(cfg_ft.get('gate', False))      # 0 soft / 1 hard-FIX
        self.agreement = bool(cfg_ft.get('agreement', True))  # region-DINO consensus
        if   not self.gate:      self.mode = 'soft'
        elif self.agreement:     self.mode = 'agreement'      # HEADLINE
        else:                    self.mode = 'freeze'         # neg-control (do_track always False)
        # models: RAFT always; DINO only for agreement (saves hub dl + VRAM, keeps controls clean)
        self.raft, self.raft_tf = ft.load_raft(device, small=cfg_ft.get('raft_small', False))  # UNPACK 2-tuple
        self.dino = ft.load_dino(device, cfg_ft.get('dino_backbone', 'dinov2_vits14_reg')) \
                    if self.mode == 'agreement' else None
        self.ref_stride      = int(cfg_ft.get('ref_stride', 8))
        self.cam_thresh      = float(cfg_ft.get('cam_thresh', 2.0))
        self.disagree_thresh = float(cfg_ft.get('disagree_thresh', 0.2))
        self.n_groups        = int(cfg_ft.get('n_groups', 12))
        self.ransac_thresh   = float(cfg_ft.get('ransac_thresh', 1.0))
        self.region_deadband = float(cfg_ft.get('deadband', 3.0))
        self.min_px          = int(cfg_ft.get('min_px', 50))
        self.alpha           = float(cfg_ft.get('alpha', 0.5))
        self.w_min           = float(cfg_ft.get('w_min', 0.1))
        self.w_max           = float(cfg_ft.get('w_max', 1.0))
        self.w_deadband      = float(cfg_ft.get('w_deadband', 0.0))
        self.seed            = int(cfg_ft.get('seed', 0))     # PINS KMeans random_state -> determinism
        self._buf            = OrderedDict()                  # time_idx -> [H,W,3] uint8 BGR

    @staticmethod
    def to_bgr_u8(color_chw_norm):
        """color is [C,H,W] float in [0,1] (post-/255). Reconstruct [H,W,3] uint8 BGR for RAFT/opencv."""
        rgb = (color_chw_norm.detach().permute(1, 2, 0).clamp(0, 1) * 255.0
               ).round().to('cpu').numpy().astype(np.uint8)        # [H,W,3] RGB
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _push(self, time_idx, bgr):
        self._buf[time_idx] = bgr
        while len(self._buf) > self.ref_stride + 1:
            self._buf.popitem(last=False)                          # drop oldest

    def step(self, cur_bgr, time_idx):
        """Returns (do_track: bool, w: np.ndarray[H,W] float32 | None).
        do_track=False -> caller SKIPS the optimizer (keeps const-velocity init pose), still maps.
        w (numpy HxW)  -> per-pixel tracking-loss down-weight; None = uniform/NOP."""
        ref_idx = time_idx - self.ref_stride
        ref_bgr = self._buf.get(ref_idx)
        if ref_bgr is None:                                        # WARM-UP: causal ref not yet buffered
            self._push(time_idx, cur_bgr); return True, None       # neutral NOP, identical to base start
        if self.mode == 'freeze':
            self._push(time_idx, cur_bgr); return False, None
        # soft per-pixel weight from camera-vs-scene Sampson residual (numpy [H,W])
        resid = self.ft.flow_residual(ref_bgr, cur_bgr, self.raft, self.raft_tf, self.dev,
                                      ransac_thresh=self.ransac_thresh)        # zeros on F-fit fail
        w = self.ft.residual_to_weight(resid, alpha=self.alpha, w_min=self.w_min,
                                       w_max=self.w_max, deadband=self.w_deadband)
        do_track = True
        if self.mode == 'agreement':
            dino_g = self.ft.dino_grid(cur_bgr, self.dino, self.dev)
            cam_mag, disagree_frac = self.ft.agreement_gate(
                ref_bgr, cur_bgr, dino_g, self.raft, self.raft_tf, self.dev,
                n_groups=self.n_groups, ransac_thresh=self.ransac_thresh,
                deadband=self.region_deadband, min_px=self.min_px, seed=self.seed)
            # WON DDS rule: track iff camera MOVING and regions AGREE on one rigid motion
            do_track = (cam_mag > self.cam_thresh) and (disagree_frac <= self.disagree_thresh)
        self._push(time_idx, cur_bgr)
        return do_track, (w if self.mode != 'freeze' else None)
```

**Causal buffer.** `ref_idx = time_idx - ref_stride < time_idx` always; the current frame is pushed **after** the
lookup so `step` never references itself. Buffer holds `ref_stride+1` frames so `_buf[time_idx-ref_stride]` is present
once warmed.

**Frame capture (FIX folded in).** RAFT/DINO need `[H,W,3] uint8 BGR`. `main.py` has already `/255`-permuted `color`
at L619 by the time the gate runs. `to_bgr_u8` reconstructs from the post-`/255` `[C,H,W]` tensor (`*255 → round →
RGB→BGR → uint8`). The `round()` makes it the exact inverse of integer `/255` (the dataset value was uint8 → float)
— verify with the §9 round-trip assert that `to_bgr_u8` reproduces `dataset[time_idx]` raw bytes. If `seperate_tracking_res`
is on, pass the **tracking-resolution** frame (`tracking_color`, L632) for geometric consistency with the optimizer's
residual; else pass `color`.

**Resolution contract.** Adapter works only in native `[H,W]`; it returns numpy `[H,W]`. The **caller (`get_loss`,
§2b)** resizes to `[Hr,Wr]` and converts to torch. Single responsibility: only the consumer knows the live residual
shape.

**Three modes** (`step` return / caller behavior):
| Mode | cfg | DINO | `step` returns | Caller |
|---|---|---|---|---|
| soft | `gate=0` | no | `(True, w)` always | always optimizes; multiplies normalized `w` into residual |
| agreement (HEADLINE) | `gate=1, agreement=1` | yes | `(do_track, w)` | if `do_track`: optimize + apply `w`; else skip optimizer |
| freeze (neg-control) | `gate=1, agreement=0` | no | `(False, None)` | always skip optimizer (degenerate baseline C2 must beat) |
Warm-up returns `(True, None)` in **all** modes → identical to base at sequence start.

---

## (4) `rgbd_slam` loop wiring (`if do_track:` FIX) + delete freeze_ba

**No top-level import** of the adapter (parity, §7). Build it lazily under the enable flag.

**(4a) Build once — insert between L644 region setup and the tracking block, actually right after the frame loop opens
(after L609 init of per-run state). Place the construction before the loop, once:**
```python
    # before the for-loop (one-time): default-off, flag-gated
    flow_gate = None
    flow_cfg = config.get('flow_track', {})
    if flow_cfg.get('enable', False):
        from Addons.motion.gs_flow_gate import GSFlowGate     # lazy: base never imports it
        flow_gate = GSFlowGate(flow_cfg, device)
    fix_count = 0
    track_eligible = 0
```

**(4b) Per-frame decision — insert between L646 and L648 (after init_camera_pose L644, before the tracking block L650):**
```python
        do_track = True
        track_weight = None
        if flow_gate is not None and time_idx > 0 and not config['tracking']['use_gt_poses']:
            track_eligible += 1
            cur_src = tracking_color if seperate_tracking_res else color   # [C,H,W] post-/255
            cur_bgr = flow_gate.to_bgr_u8(cur_src)
            do_track, track_weight = flow_gate.step(cur_bgr, time_idx)     # numpy w or None
            if not do_track:
                fix_count += 1
```

**(4c) The hard FIX — wrap the tracking block. Change L650:**
```python
        if time_idx > 0 and not config['tracking']['use_gt_poses']:          # BEFORE
        if do_track and time_idx > 0 and not config['tracking']['use_gt_poses']:   # AFTER
```
`do_track` is initialized `True` and only set `False` inside the `flow_gate is not None` branch (which is itself
guarded by `not use_gt_poses`), so the GT-pose `elif` at L707 is unaffected. When `do_track is False` the pose for
`time_idx` is whatever `initialize_camera_pose` (L644) wrote — with `forward_prop=True` that is the constant-velocity
predicted pose; with the nofwd baseline (§6) it is the previous frame's pose verbatim. Skipping L650-706 (incl.
copy-best L704-706) leaves that init pose untouched. **No "copy previous pose" statement is needed.**

**(4d) Thread the soft weight — extend the call at L665-669:**
```python
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx,
                    config['tracking']['loss_weights'], config['tracking']['use_sil_for_loss'],
                    config['tracking']['sil_thres'], config['tracking']['use_l1'],
                    config['tracking']['ignore_outlier_depth_loss'], tracking=True,
                    plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                    tracking_iteration=iter, track_weight=track_weight)
```
`track_weight` is frame-constant (computed once in 4b, reused every iter) → L677 argmin stays monotone-safe.

**(4e) Delete `freeze_ba`.** `get_loss` is **never** called with `do_ba=True` (verified: tracking call L665 and the
mapping calls all leave `do_ba` default-False). The BA-freeze path is **dead**. Do **not** add any `freeze_ba` /
`do_ba` toggle / "freeze BA on FIX frames" logic — the hard FIX (4c, skip the optimizer) is the only real freeze.

**(4f) FIX-rate accounting — after the frame loop (before final eval/save):**
```python
    if flow_gate is not None and track_eligible > 0:
        fix_rate = fix_count / track_eligible
        print(f"[FT] fix_rate={fix_rate:.4f} ({fix_count}/{track_eligible})")
```
`gs_eval` greps `[FT] fix_rate=` into `metrics.txt` (default `0.0` for base/disabled so the column always exists).

---

## (5) `flow_track.py` lift map (byte-for-byte vs net-new)

| Item | Status | Action |
|---|---|---|
| `flow_track.py` (200 lines: `load_raft`,`_raft_flow`,`_sampson`,`flow_residual`,`camera_motion`,`load_dino`,`dino_grid`,`agreement_gate`,`region_route`,`residual_to_weight`) | **byte-for-byte LIFT** | copy verbatim to `EndoGSLAM/Addons/motion/flow_track.py` |
| `EndoGSLAM/Addons/__init__.py`, `EndoGSLAM/Addons/motion/__init__.py` | **net-new (empty)** | create so `Addons.motion...` resolves on the `_BASE_DIR` path (L8-10) |
| `GSFlowGate` (class), `to_bgr_u8`, `_push`, `step` | **net-new** (§3) | author in `EndoGSLAM/Addons/motion/gs_flow_gate.py` — NOT lifted |
| `get_loss` weight insertion, loop wiring, config block, parity test | **net-new** | §2, §4, §6, §7 |

The AND decision and numpy→torch conversion are net-new caller logic; every flow/F-fit/Sampson/KMeans op stays inside
the unmodified lifted file. **Do not edit `flow_track.py`** (keeps the validated numerics + the eventual single-flow
micro-opt out of scope — see §10 OPEN-G).

---

## (6) Config block (env-driven, default-off == base)

Add a net-new `flow_track` dict to `EndoGSLAM/configs/crcd/crcd_base.py` (the `_FWD/_LRT` "existing pattern" claim is
unfounded — these are new). **Every default == neutral/base.** Read flags everywhere with `.get(..., default)` so the
pristine base (no `flow_track` key) never `KeyError`s.
```python
import os
def _eb(k, d): return bool(int(os.environ.get(k, d)))
def _ef(k, d): return float(os.environ.get(k, d))
def _ei(k, d): return int(os.environ.get(k, d))

config = dict(
    ...,
    flow_track=dict(
        enable          = _eb("FT_ENABLE", 0),          # master OFF => adapter never imported/built
        gate            = _eb("FT_GATE", 0),            # 0 soft / 1 hard-FIX
        agreement       = _eb("FT_AGREE", 1),           # region-DINO consensus (HEADLINE when gate=1)
        ref_stride      = _ei("FT_REF_STRIDE", 8),
        cam_thresh      = _ef("FT_CAM_THRESH", 2.0),
        disagree_thresh = _ef("FT_DISAGREE_THRESH", 0.2),
        n_groups        = _ei("FT_N_GROUPS", 12),
        deadband        = _ef("FT_DEADBAND", 3.0),
        min_px          = _ei("FT_MIN_PX", 50),
        alpha           = _ef("FT_ALPHA", 0.5),
        w_min           = _ef("FT_W_MIN", 0.1),
        w_max           = _ef("FT_W_MAX", 1.0),
        w_deadband      = _ef("FT_W_DEADBAND", 0.0),
        ransac_thresh   = _ef("FT_RANSAC_THRESH", 1.0),
        raft_small      = _eb("FT_RAFT_SMALL", 0),
        dino_backbone   = os.environ.get("FT_DINO_BACKBONE", "dinov2_vits14_reg"),
        seed            = _ei("FT_SEED", 0),            # KMeans random_state determinism
        max_fix_rate    = _ef("FT_MAX_FIX_RATE", 0.30), # win-gate cap (§8) — tightened from 0.60
    ),
)
```
Every read site uses `config.get('flow_track', {}).get('enable', False)` etc. (do NOT hard-subscript
`config['flow_track']['enable']` — the pristine base has no such key → KeyError = parity break).

---

## (7) Default-off PARITY proof + the Inc-0 test (adapted for GS)

**Off-path proof.** With `FT_ENABLE=0` (or the block absent):
1. §4a `flow_cfg.get('enable', False)` is `False` ⇒ `GSFlowGate` **never constructed**, `from Addons.motion.gs_flow_gate import GSFlowGate` **never executes** ⇒ no RAFT/DINO load, **zero extra RNG draws**.
2. §4b gate block skipped (`flow_gate is None`) ⇒ `do_track=True`, `track_weight=None` for every frame.
3. §4c L650 reads `if do_track and time_idx>0 and not use_gt_poses` with `do_track==True` ⇒ logically identical to base; L650-706 runs for the same frames; copy-best identical.
4. §2 `use_tw = (tracking and None is not None) = False` ⇒ the L266-insert block is skipped; `depth_res`/`im_res` are *named* locals computing `torch.abs(...)` then `[mask].sum()` — **same ops, same values** as base.
5. No new `losses` key ⇒ `weighted_losses`/`loss`/`loss_weights` unchanged.

**The Inc-0 test CANNOT be `test_inc0_bitidentical.py`** — that builds `model.scene_rep.JointEncoding` and snapshots
RNG-after-build; EndoGSLAM has **no `JointEncoding`** (GS rasterizer pipeline). New test
`EndoGSLAM/Addons/regression/test_ft_parity.py`, **two layers**:

- **Layer 1 — import isolation (decisive, GPU-safe).** Run `main.py` 2 frames with `FT_ENABLE=0` in a subprocess;
  assert `'Addons.motion.gs_flow_gate' not in sys.modules` **and**
  `'torchvision.models.optical_flow' not in sys.modules` afterward. Proves the off-path cannot draw RNG or load models.
- **Layer 2 — execution-trace identity (GPU-safe), NOT float byte-equality.** `seed_everything` does **not** set
  deterministic algos and the GS rasterizer backward uses gradient **atomic-adds** → a 2-frame pose byte-cmp is
  **flaky base-vs-base on GPU** and would false-fail. So Layer-2 asserts the *off path executes the base statements
  unchanged*: instrument (via a coverage trace or an injected counter) that with `FT_ENABLE=0`, (i) the `use_tw`
  branch is never taken, (ii) `track_weight is None` at the L665 call, (iii) lines 268/275 run their original
  expression, (iv) the `if do_track` wrap evaluates `do_track==True` every eligible frame. This is the robust GPU
  analogue of the RNG-after-build check.
- **OPTIONAL Layer 3 — numerical-NOP tolerance (CPU only, if rasterizer has a CPU path).** `FT_ENABLE=1, gate=0,
  w_min=w_max=1.0` makes `residual_to_weight` clip to exactly 1.0; assert poses match base **within float tol**
  (NOT byte-identity — multiply-by-ones changes reduction order). Run under `torch.use_deterministic_algorithms(True)`.
  This is a *separate* sanity check, **not** the mandatory gate. The mandatory gate is Layers 1+2 at `FT_ENABLE=0`.

Gate is **mandatory**: a non-isolated off-path is a leak → abort before any enabled A/B.

---

## (8) A/B sweep order + hardened win-gate + freeze-control + held-out + FIX-rate cap

**Reference baseline.** State the parity reference explicitly: pristine `crcd_base` ships `forward_prop=True`. The
*operating* A/B reference is the `nofwd_lr` baseline (Sim3-ATE 3.24mm, path-ratio 31.8, |Pearson|dom 0.888, PSNR
20.4) — **which is itself net-new env-knobs** (`FWD_PROP`, `LR_TRANS_MULT`, `LR_ROT_MULT`) that must be added first;
they do **not** exist in-tree today. C2 default-off on top of nofwd_lr must equal nofwd_lr (two-layer parity).

**PREREQUISITE (blocking — the gate is uncomputable without it).** `gs_eval.py` today emits only
`frames/PSNR/SSIM/LPIPS/L1depth_mm/ATE_mm`. It does **NOT** emit `path_ratio`, `|Pearson|dom`, or `fix_rate`. Before
any A/B: (a) extend `gs_eval.py` to call `sim3_ate.evaluate` and write `path_ratio`, `pearson_dom`, and the
**dominant-axis est std** to `metrics.txt`; (b) grep `[FT] fix_rate=` from the SLAM log into `metrics.txt` (0.0 for
base); (c) add a **C2-liveness** column: fraction of frames where the F/H fit SUCCEEDED and `std(w) > eps` (i.e. `w`
departed from uniform) + mean `(1-w)` mass. Gate the experiment on these columns being present and non-NaN.

**Sweep order** (each arm n=3, on the nofwd_lr base; negative control adjacent to headline):
```
0) nofwd_lr                                   (baseline + seed-std = noise floor)        seeds 0,1,2
A) ft_soft   FT_ENABLE=1 FT_GATE=0 FT_AGREE=1 (soft weight only, NOT novel)              seeds 0,1,2
B) ft_fix    FT_ENABLE=1 FT_GATE=1 FT_AGREE=1 (agreement hard-FIX, HEADLINE)             seeds 0,1,2
C) ft_freeze FT_ENABLE=1 FT_GATE=1 FT_AGREE=1 FT_CAM_THRESH=1e9 (freeze-all neg-control) seeds 0,1,2
D) ft_fix_raw FT_ENABLE=1 FT_GATE=1 FT_AGREE=0 (raw-F, no DINO ablation)                 seeds 0,1,2
```

**Hardened win gate.** `base=nofwd_lr`, `B=ft_fix`, `freeze=ft_freeze` (all n=3 mean ± seed-std). **B kept only if
ALL hold** (any failure ⇒ reject + log negative):
1. **Anti-collapse precondition (checked FIRST).** Reject any run (incl. freeze) whose dominant-axis est std `< 1e-4 m`
   OR whose `pearson_dom` is NaN — treat as *degenerate-frozen* → **auto-FAIL**, never auto-pass. (Fix folded:
   `np.corrcoef` on a near-constant frozen trajectory returns NaN; `NaN > x` is always `False` in Python, so an
   un-guarded comparison silently mis-handles the control. The freeze control is defined as *degenerate*, not a number.)
2. **path-ratio in a TARGET BAND**, not merely "down": `B.path_ratio ∈ [floor≈8, base≈31.8]`. Below the up-to-scale
   floor = over-collapse (Sim3 scale `s` inflates as `pathlen(est)→0`, so `path_ratio=pathlen·s` is **not** monotone-
   good) → reject. Must clear seed-std vs base on the *down* side.
3. **|Pearson|dom UP** clearing seed-std (base≈0.888) AND **B is non-degenerate while freeze is degenerate-or-lower**
   (the selectivity falsifier).
4. **ATE flat-or-down**: `B.ate ≤ base.ate + base.std` (Sim3 mm, base 3.24).
5. **render not regressed**: `B.PSNR ≥ base.PSNR − base.std` (base 20.4); SSIM/LPIPS not materially worse; AND
   `B.PSNR > freeze.PSNR` (beats the degenerate on render).
6. **FIX-rate ≤ `max_fix_rate`=0.30** (tightened from 0.60; 0.60 freezes near-collapse on static-heavy CRCD). Make it
   a per-snippet data-derived cap if the static fraction is known.
7. **Movement band**: median per-frame est translation ∈ `[k_low, k_high]×` nofwd_lr median — rejects both freeze
   (too small) and jitter (too large).
8. **C2-liveness floor**: the soft `w` actually departed from uniform on ≥ a floor fraction of frames; a ~uniform `w`
   run is reported **INERT** (correct negative), not a pass.

**Held-out validation (corrected — not one snippet).** Tune `(cam_thresh, disagree_thresh)` on a 3×3 grid
(`{1,2,4}×{0.1,0.2,0.3}`, deadband fixed 3.0) on **c1_001** (n=3); lock the single best by §8 margin; **report the
full grid** to expose overfitting. Then validate the **locked, unchanged** thresholds on a **held-out SET of ≥3 CRCD
snippets** (NOT touched in tuning; same MoGe-2 depth + 360-GT contract), each n=3, with B/soft/freeze/raw-F arms run
**on the same held-out snippets**. Win declared only if §8 passes on the **majority** with effect `>` pooled seed-std.
A c1_001-only pass is a tuning artifact (CRCD is sub-SNR; a single held-out snippet at n=3 is within seed-std =
underpowered). Same `gs_eval` + 6-panel video on every arm (two-diagnostic-sets standing rule).

---

## (9) Implementation CHECKLIST (ordered, each independently testable)

1. **Scaffold** `EndoGSLAM/Addons/{__init__.py, motion/__init__.py, regression/__init__.py}` (empty). **Test:**
   `python -c "import Addons.motion"` from `EndoGSLAM/` succeeds (path on `_BASE_DIR`).
2. **Lift** `flow_track.py` byte-for-byte → `EndoGSLAM/Addons/motion/flow_track.py`. **Test:** `import Addons.motion.flow_track as ft; ft.load_raft, ft.agreement_gate` resolve; diff against the DDS-SLAM original is empty.
3. **Author** `gs_flow_gate.py` (§3). **Test (offline, no SLAM):** feed two real CRCD frames; assert `step` returns
   `(bool, ndarray[H,W] float32)` for soft and `(bool, None)` for freeze; warm-up returns `(True, None)`; KMeans
   labels reproduce across two calls with same `seed`.
4. **Frame-capture round-trip.** Assert `to_bgr_u8(color_post_/255)` reproduces `dataset[time_idx]` raw uint8 (RGB→BGR)
   exactly (the `round()` inverse of integer `/255`).
5. **Config block** (§6) into `crcd_base.py`; all `.get()`-guarded reads. **Test:** pristine base (no env) builds; `config['flow_track']['enable'] == False`.
6. **`get_loss` weight insertion** (§2): signature kwarg, `use_tw` guard, resize+convert+mean-preserving, multiply
   both residuals. **Test:** unit — synthetic `[1,Hr,Wr]` residual + numpy `[H,W]` weight at a *different* resolution;
   assert no broadcast error, `masked_mean(w_norm)==1±1e-6`, and `track_weight=None` reproduces the base scalar exactly.
7. **Loop wiring** (§4a-4f): lazy build, per-frame `step`, `if do_track` wrap, call-site kwarg, fix_count accounting,
   delete any `freeze_ba`. **Test:** `FT_ENABLE=1 FT_GATE=1` 5-frame run prints `[FT] fix_rate=`; no crash.
8. **Parity test** `test_ft_parity.py` (§7 Layers 1+2). **Test:** Layer-1 import-isolation passes; Layer-2 trace shows
   off-path takes zero new branches. **MANDATORY GREEN before any enabled A/B.**
9. **`gs_eval.py` extension** (§8 prereq): path_ratio, pearson_dom, dom-axis-std, fix_rate, C2-liveness into
   `metrics.txt`; NaN-Pearson → "degenerate". **Test:** run on the freeze control → flagged degenerate, not a number.
10. **nofwd_lr baseline knobs** (`FWD_PROP/LR_TRANS_MULT/LR_ROT_MULT`) net-new + n=3 baseline. **Test:** reproduces
    3.24mm / 31.8 / 0.888 / 20.4 within seed-std.
11. **A/B sweep** (§8 order) via the overnight harness; `.DONE`/`.FAILED` resumable, not `set -e`.
12. **Tune-on-c1_001 → validate-on-held-out-set (≥3 snippets)** → declare via the hardened gate (§8).

---

## (10) Risks + open decisions

- **OPEN-A (F-degenerate forward/zoom).** Endoscope forward/zoom motion makes the fundamental matrix degenerate
  (`flow_track.py` warns at L136-137) → `flow_residual` returns zeros → `w≈1` → soft arm is **inert** (NOT a bug, but
  it must be *reported* inert via the §8 C2-liveness column, not mistaken for "safe"). The intended fallback is the
  homography "pixel" mode, which is **not wired** in any draft. **Decision needed:** either wire the homography branch
  for zoom-dominated snippets or document C2-soft as invalid there. Log the F-fit failure rate per run.
- **OPEN-B (over-gating on deformation-pervasive CRCD).** If most of the field deforms, the agreement gate may FIX
  most frames → collapse toward freeze. The `max_fix_rate=0.30` cap + movement band + Pearson-vs-freeze falsifier
  guard this, but the cap should become per-snippet/data-derived. **Decision:** derive the cap from the static-pixel
  fraction once measured.
- **OPEN-C (depth_loss_thres interaction).** L693 is **dead** in the operating config (`use_depth_loss_thres=False`),
  so mean-preserving is justified by L677/A-B comparability, **not** L693. If a config ever flips it on, mean-
  preserving already covers it — but add an assert `use_sil_for_loss is True` when soft mode is active (the dead L277
  branch normalizes over the mask, not full-frame; see OPEN-D).
- **OPEN-D (dead L277 normalizer).** The no-sil tracking RGB branch (L277, dead under `use_sil_for_loss=True`)
  multiplies by the *mask*-normalized `w_norm`, which is not full-frame-mean-1. Harmless while dead. If ever live,
  compute a separate full-frame `w_mean_full = wt.mean()`. **Decision:** assert `use_sil_for_loss` in soft mode rather
  than implement the second normalizer now.
- **OPEN-E (KMeans determinism).** `agreement_gate(..., seed=self.seed)` pins `random_state` → reproducible region
  labels. Verify the lifted `agreement_gate` actually threads `seed` into `KMeans(random_state=seed)` (it should, per
  its signature). If not, that is a one-line fix **in the lifted file** — flag before lifting.
- **OPEN-F (frame BGR fidelity).** `to_bgr_u8` reconstructs from the post-`/255` tensor with `round()`. Validated by
  the §9.4 round-trip assert; if the dataset applies any pre-scaling (gamma/normalization) before L619, capture from
  `dataset[time_idx]` raw instead. **Decision:** confirm the round-trip assert is exact before trusting reconstruction.
- **OPEN-G (two RAFT passes/frame).** Agreement mode calls `flow_residual` (soft `w`) AND `agreement_gate`
  (its own internal RAFT flow) → **two RAFT forwards/frame**. A combined entry point returning
  `(cam_mag, disagree_frac, resid)` from one flow would halve it, but that **edits the lifted file** → deliberately
  **out of scope** (keep `flow_track.py` byte-for-byte; eat the second pass). **Decision for the user.**
- **OPEN-H (parity test on GPU).** Byte-equal pose cmp is **not** GPU-reproducible (no deterministic algos +
  rasterizer atomic-adds). The mandatory gate is therefore the import-isolation + execution-trace (§7 Layers 1+2), not
  float byte-equality. The CPU NOP-tolerance check (Layer 3) is optional and tolerance-based only.

**NRGS-SLAM (arXiv:2602.17182) must be read IN FULL before any novelty claim** (§0).

---

### Files (absolute)
- New adapter: `EndoGSLAM/Addons/motion/gs_flow_gate.py` (net-new)
- Lifted verbatim: `EndoGSLAM/Addons/motion/flow_track.py` ← `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/motion/flow_track.py`
- Wiring target: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/EndoGSLAM/scripts/main.py` (L198-200, 266-277, 609, 643-706)
- Config: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/EndoGSLAM/configs/crcd/crcd_base.py`
- Parity test: `EndoGSLAM/Addons/regression/test_ft_parity.py` (net-new)
- Eval to extend: `EndoGSLAM/.../gs_eval.py` + canonical metric `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/eval/sim3_ate.py`
- Prep source: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/docs/GS_CONTRIB_PREP_20260622.md`
