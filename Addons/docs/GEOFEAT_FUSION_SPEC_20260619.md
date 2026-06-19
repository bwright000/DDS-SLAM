# geo_feat SNI-FUSION BUILD SPEC (2026-06-19, wf wgdm1ply1)

Fuse geo_feat (geometry/jitter-killer) WITH DINO (semantic/spike-bounder) as the sigma^2 input.
Parity STRUCTURAL (fuse branch inside the enable block the golden build never enters). geo_feat DETACHED.
v1 concat (fuse=geo / geo_rgbd) build-now; v2 cross-attention later. Gate = 2D win (geo path-ratio AND dino ATE_max).
NOTE: like all Arm-1 work this feeds TRACKING (Inc-2); the sigma^2 NLL is the only (small) mapping footprint.

---

I have everything. The build spec follows.

---

# geo_feat SNI-FUSION — BUILD SPEC (LEAD)

The two readings are code-verified correct. One sharpening I can now assert with certainty: **the Inc-0 gate forces `enable=False` (test line 50-51), so the entire `fuse` flag — whatever its value — lives inside the unreachable `enable && mode=='dino'` branch (decoder.py:488).** Parity for v1 is therefore not "fragile, mind the RNG" — it is *structurally* guaranteed, because the golden build never enters the branch at all. This is the load-bearing fact for everything below.

---

## 1. PER-RAY geo_feat — the exact exposure

**The crux is real and the machinery already exists.** `geo_feat` is per-SAMPLE `[N_rays, N_samples, 15]` (decoder.py:510), consumed but discarded. The σ² head fires per-RAY. The volume-render that bridges sample→ray already runs for the scalar σ² at **`scene_rep.py:141`**: `sigma2_map = torch.sum(weights[...,None] * sigma2, -2)`, using `weights = self.sdf2weights(...)` at **`scene_rep.py:129`** — the identical weights as rgb (`:130`), edge (`:131`), depth (`:148`). We replicate that one line on the 15-dim feature.

**Path A (the only correct route — Path B is dead because `raw` carries only `[rgb,sdf]`, decoder.py:534).** Stop discarding geo_feat; volume-render it with the same weights; drop it in `rend_dict`.

Four plumb edits, every one `None`-guarded so the off-path tuple shapes are byte-identical:

| # | file:line | change |
|---|---|---|
| A1 | `decoder.py:534` | return a gated 4th element: `..., sigma2_raw, (geo_feat if self._surface_geo else None)`. Set `self._surface_geo = (mode=='dino' and fuse in ('geo','geo_rgbd'))` in `__init__`. |
| A2 | `scene_rep.py:240` | unpack 4-tuple: `outputs_flat, edge_semantic, sigma2_flat, geo_feat_flat = batchify(...)`; reshape `geo_feat_flat` like sigma2 (`:244-246`, `None` when off); add to `run_network` return (`:247`). |
| A3 | `scene_rep.py:~141` | beside the scalar blend: `geo_feat_map = torch.sum(weights[...,None] * geo_feat, -2)` (only `if geo_feat is not None`); return it guarded, exactly like `sigma2_map` at `:156-158`. |
| A4 | `scene_rep.py:462` | `if geo_feat_map is not None: rend_dict['geo_feat'] = geo_feat_map` (mirrors the `sigma2_map` guard). |

**`batchify` arity note (must-check during build):** A2 changes the unpacked tuple width from `query_color_sdf`. `batchify` must pass through whatever arity the decoder returns. Since A1 makes the decoder *always* return a 4-tuple (4th = `None` when off), this is uniform — but confirm `batchify` doesn't hardcode 3. If it does, that's a one-line fix and still parity-inert (the 4th is `None` on the golden path; and the golden path never has `enable` anyway).

**DETACH — decisive, detach.** `rend_dict['geo_feat'].detach()` into the σ² head. Three reasons, in priority order:
1. σ² is a **mapping-only aleatoric hedge** (Inc-1) feeding the **already-`.detach()`'d** Inc-2 tracking down-weight (scene_rep.py:533 comment). The whole pathway is *read-only* on geometry by design.
2. Live gradients let the NLL `1/(2σ²)‖err‖² + ½logσ²` **lower its own loss by corrupting the SDF geo_feat to inflate σ²** — the σ∝‖error‖ degeneracy the user's own ARM-1 lit-positioning memo names as the exact failure the field abandoned. A 15-D gradient highway into the SDF net is the *strongest* version of that risk.
3. The existing fuse already detaches rgb/depth (`scene_rep.py:514-515`). geo_feat is a stronger signal → detach is *more* important, not optional.

"Let σ² teach geometry where it's uncertain" is a joint objective the thesis forbids (metrics-first, σ² is a hedge). If ever wanted, it's a separate named A/B — **not** the default.

---

## 2. v1 CONCAT — build NOW

**Config flags** (extend the existing `fuse` enum; mirror `c1_001_canon_uncert_dino_reg_fused.yaml`):
- `fuse: 'geo'` → `[DINO(384) ; geo_feat(15)]` = **399**. *The headline arm* — pure geometry+semantic fusion, no rgb/depth proxy noise.
- `fuse: 'geo_rgbd'` → `[DINO ; geo_feat(15) ; rgb(3) ; depth(1)]` = **403**. The full SNI-modality stack.
- `fuse: 'rgbd'` (existing) → 388, byte-unchanged.

**decoder.py:495-501 — in_dim arithmetic:**
```python
_fuse = config.get('uncertainty', {}).get('fuse', '')
_Cgeo = int(config['decoder']['geo_feat_dim'])     # 15
_extra = 0
if _fuse in ('geo', 'geo_rgbd'):  _extra += _Cgeo
if _fuse in ('rgbd', 'geo_rgbd'): _extra += 4
self.dino_unc_net = UncertaintyDINONet(in_dim=int(config['uncertainty']['dino_dim']) + _extra, ...)
```

**scene_rep.py:510-515 — the concat** (replace the single `rgbd` branch):
```python
_fuse = self.config.get('uncertainty', {}).get('fuse', '')
_parts = [target_dino]
if _fuse in ('geo', 'geo_rgbd') and 'geo_feat' in rend_dict:
    _parts.append(rend_dict['geo_feat'].detach().reshape(target_dino.shape[0], -1))   # [N,15]
if _fuse in ('rgbd', 'geo_rgbd'):
    _parts.append(rend_dict['rgb'].detach().reshape(target_dino.shape[0], 3))
    _parts.append(rend_dict['depth'].detach().reshape(target_dino.shape[0], 1))
_feat = torch.cat(_parts, dim=-1) if len(_parts) > 1 else target_dino
```
The only new module touched is `UncertaintyDINONet`'s first `Linear`, which grows with `in_dim` — already `in_dim`-flexible (decoder.py:341).

**The direct test this runs:** geo jitter-suppression + dino structure in ONE σ². The σ² head now reads the *actual* geometry feature (the geo arm's jitter-killer) AND the DINO semantic structure (the dino_reg spike-bounder) simultaneously. This is the literal hypothesis: "smooth AND bounded."

---

## 3. v2 CROSS-ATTENTION — later, gated behind v1 showing signal

**Module** `UncertaintyCrossAttnNet(nn.Module)` (next to `UncertaintyDINONet`, decoder.py:307), DINO-as-query:
- `q_proj: Linear(384, D)`, `k_proj/v_proj: Linear(D, D)` over per-source-projected tokens, single-head scaled-dot-product over T≈2-3 modality tokens (geo, appearance), `out_head: Linear(D, 1)`. D≈64. ~few×10⁴ params, all on the decoder → swept into `map_optimizer.main` by `_dec_groups()` (ddsslam.py:742, enumerates `decoder.named_parameters()`).
- Per-ray-sampling compatible: every token is **already** per-ray (geo_feat_map from v1's volume-render, rgb/depth maps, per-pixel DINO). Attention is over the tiny modality set `[N, T, D]`, never over samples. **v1's per-ray geo_feat is the shared prerequisite** — v2 just swaps concat-MLP for attention over the same vectors.
- Same DETACH rule (geo_feat_map/rgb/depth into KV; DINO query is a frozen `.npy`, no graph).

**Construction (parity):** built **only** in a new sub-branch `if _fuse == 'geo_attn'` inside the existing `elif enable and mode=='dino'` (decoder.py:488), built **last**. Off-build never enters `enable` → never constructs it → zero RNG/params/keys.

**Gate:** do not write v2 until v1 (`fuse:'geo'`) clears the §4 metric gate. If concat already wins the 2D test, v2 is a refinement A/B, not a necessity.

---

## 4. METRIC GATE (n=1, effect-size + coherence)

Reference points from the fresh finding (n=1 raw-left): geo path-ratio **4.20**, ATE_max **8.74**; dino_reg_rd path-ratio **6.41**, ATE_max **5.81**. The complementarity is the whole premise: geo=smooth/spiky, dino=wobbly/bounded.

**`fuse:'geo'` (and `geo_rgbd`) must hit a 2D win — beat BOTH parents at once:**

| Axis | Source-of-truth | PASS condition |
|---|---|---|
| Jitter (smooth) | `sim3_ate.py` path-ratio | ≤ ~4.2 (≈geo; decisively < dino_reg_rd 6.41) |
| Worst-case (bounded) | `sim3_ate.py` ATE_max | ≤ ~5.8 (≈dino_reg_rd; decisively < geo 8.74) |
| σ² validity (cross-check) | `sigma2_quality.py` frac_oracle + motion-corr | ≥ both parents, or at least holds (not degraded) |
| Mean ATE (sanity) | `sim3_ate.py` ATE_mean + Pearson_dom | not worse than the better parent |

**PASS = a single config that is simultaneously ≤geo on path-ratio AND ≤dino_reg_rd on ATE_max, with frac_oracle held.** That is the "smooth AND bounded" claim made falsifiable. A config that only matches one parent is a FAIL (fusion bought nothing over picking the right single feature). Judge by effect-size + the σ²-quality cross-check (does the fused σ² actually carry both geo's low-jitter structure and dino's semantic boundedness), **not** seeds — n=1 by standing rule.

---

## 5. PARITY + BUILD ORDER

**Parity verdict:**
- **v1:** no new module — only `UncertaintyDINONet` `in_dim` widening + `None`-guarded tensor plumbing. The Inc-0 gate forces `enable=False`, so the `fuse` branch is structurally unreachable on the golden build → RNG/params/keys bit-identical → **PASS by construction**. The `geo_feat` 4th-return / `geo_feat_map` blend add zero params and are `None`-guarded → no new op on the off-path. The one diligence item: the decoder→`run_network` tuple-width change must thread `None` cleanly so `'geo_feat' in rend_dict` is False off-path (it is, because `_surface_geo` is False unless `mode=='dino'`).
- **v2:** new `UncertaintyCrossAttnNet` constructed **only** under `enable && mode=='dino' && fuse=='geo_attn'`, built last → absent off-build → **PASS**; built-last → clean single-variable A/B vs base.

**Run the gate after each:** `python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml` must print PASS.

**Build order (rectified pipeline going forward; every run ships metrics + 6-panel video + sigma2_quality):**
1. Plumb-out (A1-A4) + extend in_dim arithmetic + extend concat. Run Inc-0 gate → PASS.
2. New configs `c1_001_canon_uncert_dino_reg_geo.yaml` (`fuse:'geo'`) and `..._geo_rgbd.yaml`, inheriting `c1_001_canon_uncert_dino_reg.yaml` (mirror the existing `_fused.yaml`).
3. Run via `run_cell.sh` (rectified), n=1: base / geo / dino_reg_rd / **geo-fused** — same harness, same GT (CRCD-Published 360). Ship Sim3 ATE table + path-ratio + ATE_max + sigma2_quality + 6-panel video for each.
4. Apply §4 gate. Only if `fuse:'geo'` clears it → write v2.

---

## 6. HONEST VERDICT vs the readings + OPEN DECISIONS

**Which first:** build **`fuse:'geo'` (DINO+geo_feat, no rgbd)** as the headline, with `geo_rgbd` as a same-cost companion arm. Rationale grounded in the jitter finding: geo's jitter-suppression comes from the *actual geometry feature*, and rgb/depth are explicitly "a COARSE geometry proxy" (the existing `rgbd` fuse, which did **not** reproduce geo's path-ratio 4.20 — that's *why* this build exists). Adding rgb/depth on top of the real geo_feat risks diluting the clean geometry signal with proxy noise. So `fuse:'geo'` isolates the actual hypothesis; `geo_rgbd` only earns its place if it beats `geo`.

**Is the fusion a tracking-IMPROVER or also ceiling-limited?** Genuinely an **improver**, and this is the key difference from a render-ceiling problem. The σ² → Inc-2 tracking down-weight feeds **directly into the pose loss weighting** (`compute_loss weights=`), and the n=1 finding already shows the σ² *feature source* moves the two tracking metrics that matter (path-ratio and ATE_max) in *opposite* directions. Fusion is not chasing a saturated render PSNR — it is combining two demonstrated, complementary *tracking* effects. The mechanism is causal (better σ² → better pose down-weight → better trajectory), not cosmetic. The risk is not a ceiling; it is that **the two effects don't compose** — that the head learns to lean on one feature and ignore the other, or that the combined σ² is a muddy average (path-ratio ~5.3, ATE_max ~7) that beats neither parent. That is exactly what the §4 2D gate is built to catch.

**OPEN DECISIONS for the user:**

1. **σ² floor / softplus constant on the fused head — match geo or dino?** The geo head applies softplus+floor at `raw2outputs:140-145` (per-sample then per-ray); the dino head applies it at the `scene_rep` call site (`:516-517`). The geo-fused head runs through the **dino** path (it IS the dino head with a wider in_dim), so it inherits the dino floor. Is that the intended normalisation, or should the fused arm get geo's per-sample-then-render floor treatment to faithfully inherit geo's smoothing? **My recommendation: keep the dino-site floor** (single-variable change = feature source only, floor held constant vs dino_reg_rd) — but flag it because the floor is part of *why* geo is smooth, and you may want a floor A/B as a follow-up.

2. **`geo` vs `geo_rgbd` as the primary arm — confirm.** I recommend `fuse:'geo'` headline + `geo_rgbd` companion (reasoning above). If you instead want the maximal-SNI single shot, we lead with `geo_rgbd`. Your call drives which config is the §4 headline.

(One process note, not a blocker: confirm `batchify` passes through the 4th return element without a hardcoded 3-tuple unpack — I flagged it in §1; it's a trivial parity-inert fix if needed, verified during the build.)