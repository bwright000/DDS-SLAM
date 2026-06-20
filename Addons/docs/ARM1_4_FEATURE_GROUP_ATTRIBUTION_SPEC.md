# ARM-1 FEATURE-GROUP ATTRIBUTION ENGINE — IMPLEMENTATION SPEC (2026-06-20, wf wp8nep3zm)

The 2nd engine: slot-attention over the scattered DINO-feature batch -> region-coherent sigma^2 + the LEARNED what-kind {tissue,tool,bg}. The feature side of the pixel/feature asymmetry. Code-grounded, parity-first.
DE-RISK FIRST: the offline DINO-separability triad probe (frozen DINO vs true/shuffled/k-means seg, held-out mIoU) BEFORE building the module.

---

Everything is confirmed and grounded. The judge metrics are exactly `frac_oracle`, `Pearson(σ², inter-frame motion)`, and the speckle index (in `sigma2_diagnostics.py`). The #3 spec is a pure-loss term as described. I now have full code-grounding to write the build spec. One correction to fold in: `dino_grid` is NOT currently passed into `scene_rep.forward` — only the gathered `target_dino` is — so Option B requires real new plumbing while Option A is the drop-in. I'll be decisive on Option A for v1.

Here is the build spec.

---

# FEATURE-GROUP ATTRIBUTION-ENGINE — IMPLEMENTATION DESIGN

`Addons/docs/ARM1_4_FEATURE_GROUP_ATTRIBUTION_SPEC.md` — the build spec for the second engine (FEATURE-ATTRIBUTION) that subsumes the per-pixel σ² head's speckle and adds the learned what-kind axis of TriGauge. Code-grounded against live `decoder.py` / `scene_rep.py` / `ddsslam.py` / `dataset.py` (2026-06-20). Parity-first, metric-gated, honest about Arm-2 gating.

---

## 0. The decision in one paragraph

The per-pixel σ² head is speckly (0.496) and attribution-blind because it runs on the **post-gather** per-ray tensor `target_dino [N,384]` (`scene_rep.py:521`) — by then the 2048 rays are a random scatter (`ddsslam.py:271-279`) with no spatial neighbours. The new engine, **`FeatureGroupNet`**, is a slot-attention module (Locatello 2020) that operates **WITHIN the scattered 2048-ray batch in DINO-feature space** (K=3 slots compete via softmax-over-slots, T=3 GRU refinement, `O(N·K·T)` ≪ N²), producing **both** a region-coherent σ² **and** a `[N,3]` learned what-kind `{tissue,tool,bg}` from one pooled representation. It is **scatter-proof by construction** (slot binding is permutation-invariant over the input set and feature-driven, not spatial — the Caron-2021 enabler: frozen DINO already clusters by object without labels). It is built **only** inside the existing `enable ∧ mode=='dino'` branch behind a new `uncertainty.group` sub-flag, constructed LAST → off-build draws no RNG, adds no params, adds no state-dict keys → `test_inc0_bitidentical` green with no golden re-write.

**Why slot-attention over the alternatives** (cross-attn / pooling / non-local): it is the **only** candidate whose native output IS the attribution axis the program is missing. Cross-attn (N²) and non-local give stabilised magnitude but bolt attribution on afterwards; static pooling gives a weak k-means version. Slot-attention delivers stabilised-σ² **and** the `[N,3]` what-kind in one pass, at the lowest cost of all four — exactly the "look at the whole feature, do both" the brief asks for.

---

## 1. THE MECHANISM (decided)

### 1.1 Operating point: WITHIN the scattered batch (Option A), not the full-2D map (Option B)

The full DINO grid `batch['dino_grid'] [gh,gw,384]` exists per frame (`dataset.py:117`) but is consumed by `sample_dino_grid` **before** `model.forward` is called (`ddsslam.py:279`) — **`dino_grid` is NOT threaded into `scene_rep.forward`** (only the gathered `target_dino` is, `ddsslam.py:288`). So:

- **Option A (chosen for v1):** group over the `[N=2048, 384]` scattered batch at the fuse site. **No plumbing change** — drops in exactly where `dino_unc_net(_feat)` is called (`scene_rep.py:535`). Grouping is by DINO-feature affinity, which IS the object adjacency (Caron 2021). Scatter is irrelevant because slot binding is permutation-invariant over the set.
- **Option B (deferred to v2):** run the module on the dense grid in `ddsslam.py` right after `_attach_dino`, then gather two output maps with `sample_dino_grid`. Recovers true spatial neighbourhoods but pays a dense pass per mapping iter (re-introducing the upsample cost `sample_dino_grid` was built to avoid — `ddsslam.py:40`, "13ms vs 944ms/472MB") and needs a new forward kwarg + edits at 4 gather sites (`ddsslam.py:279,364,635,895`). **Only build B if v1 slots are spatially incoherent on the σ² panel.**

### 1.2 Exact tensors and cost (Option A, fuse site `scene_rep.py:521-536`)

```
target_dino                       [N=2048, C=384]   # gathered per-ray DINO (mapping batch); tracking N=1024
k = Wk · target_dino              [N, D=64]
v = Wv · target_dino              [N, D=64]
slots (learned init)              [K=3, D=64]
for t in 1..T(=3):
    q    = Wq · slots             [K, D]
    attn = softmax_over_K( k @ qᵀ / √D )    [N, K]   # ← competition across the 3 slots = WHAT-KIND
    upd  = (attn / attn.sum(0, keepdim=True)).t() @ v   [K, D]   # weighted mean over N
    slots = GRU(slots, upd) + MLP(slots)    [K, D]
context  = attn @ slots           [N, D]             # region-coherent per-ray feature
# ---- two heads from the shared pooled feature ----
sigma2_raw   = head_sigma( cat[ target_dino , context.detach() ] )   [N, 1]   # (A) magnitude
whatkind_log = head_kind(  context )                                  [N, 3]   # (B) attribution
```

- **Cost:** `O(N·K·T) = 2048·3·3 ≈ 18k` slot-update ops — tiny, far below the `2048² = 4.2M` of cross-attn, and below the #3 spec's already-blessed `[N,N]` budget (~34 MB / a few ms). No chunking; no T4 OOM concern.
- **Two heads, one trunk:** `context` is the expensive shared "what is this region" summary; σ² and what-kind are read-outs of it. Sharing couples them — the what-kind gradient shapes the σ² feature space so σ² respects object boundaries (anisotropic, along feature edges), which is precisely what the per-pixel head and dino_reg's isotropic blur could not do.
- **`context.detach()` into `head_sigma`** mirrors the established stop-grad fusion discipline (`scene_rep.py:524`, "extra modalities DETACHED so σ² READS them but doesn't drive geometry"). σ² floored `softplus(·)+1e-6` at the call site, exactly as today (`scene_rep.py:535`).

### 1.3 Honest ceiling (state, don't oversell)

Slot-attention reshapes σ² in feature space and supplies the *what-kind*; it does **not** inject motion. DINO gives WHAT-KIND, not HOW-MUCH — the same tissue feature whether moving or still. It can **group-complete** a deformation cue present on *some* tissue rays to the rest, but **cannot manufacture** one absent from the residual (deformation is appearance-preserving → low residual → invisible to σ² alone). Deformation-awareness is **COMBINE territory** (Arm-2 motion × slot attribution). Frame v1 as *"learned 3-way feature-group attribution + region-coherent σ²"*, NOT *"deformation-aware σ²."*

---

## 2. THE PLUG-IN (code-grounded, parity-safe)

### 2.1 Construction — `decoder.py`, inside the existing `elif … mode=='dino'` branch (after line 509)

The module is **born in the same branch that builds `dino_unc_net`** (the only branch entered when `enable ∧ mode=='dino'`), gated on a new sub-flag, constructed LAST. It lives on the **decoder** (not scene_rep) — load-bearing: `map_optimizer` enumerates `decoder.named_parameters()` (`ddsslam.py:703`); a module on scene_rep gets ZERO gradient (`decoder.py:481-484`).

```python
# decoder.py, ColorSDFNet_v2.__init__, appended INSIDE the existing  elif enable and mode=='dino':  block
# (after self.dino_unc_net is built at line 509, so it is the LAST module constructed)
_group = config.get('uncertainty', {}).get('group', '')          # '' = per-pixel (default, untouched)
if _group == 'slot':
    self.dino_group_net = FeatureGroupNet(
        in_dim   = int(config['uncertainty']['dino_dim']),       # 384
        slot_dim = config.get('uncertainty', {}).get('group_dim', 64),
        n_slots  = config.get('uncertainty', {}).get('group_slots', 3),    # K=3 TriGauge
        n_iter   = config.get('uncertainty', {}).get('group_iters', 3))    # T=3
# else: dino_unc_net stays the active head -> per-pixel path byte-identical to today's dino arm
```

`FeatureGroupNet` (new class in `decoder.py`, alongside `UncertaintyDINONet` @307) owns: learned slot init `[K,D]`, `Wk/Wq/Wv [384→D]`, a `GRUCell(D,D)`, a slot `MLP`, and the two read-out heads `head_sigma (384+D→1)` and `head_kind (D→3)`. `forward(target_dino) -> (sigma2_raw [N,1], whatkind_logits [N,3])`.

### 2.2 Run — `scene_rep.py`, the fuse site (replace/wrap the call at line 535)

```python
# scene_rep.py, replacing  _sig = softplus(self.decoder.dino_unc_net(_feat)...)  at line 535
if hasattr(self.decoder, 'dino_group_net'):
    _sig_raw, _whatkind = self.decoder.dino_group_net(target_dino)   # group over the [N,384] batch
    rend_dict['whatkind'] = _whatkind                                # [N,3] NEW key, gated on module existence
    _sig = torch.nn.functional.softplus(_sig_raw.float()) + 1e-6
else:
    _sig = torch.nn.functional.softplus(self.decoder.dino_unc_net(_feat).float()) + 1e-6   # unchanged per-pixel path
rend_dict['sigma2'] = torch.clamp_min(_sig, 1e-6)
```

Everything downstream is **already wired to receive `rend_dict['sigma2']` unchanged**: Inc-2 down-weight (`scene_rep.py:557-562`), NLL teacher (`scene_rep.py:599-631`), surfaced `ret['sigma2']` (`scene_rep.py:653`). The **stabilised-magnitude half ships with zero downstream edits** — the only requirement is that `dino_group_net` returns a region-coherent `_sig_raw`. The `slot` path is **mutually exclusive** with `dino_unc_net` in dino mode; keep `dino_unc_net` as the legacy v1 σ²-head A/B behind `group=''`.

### 2.3 The flag and config block (all no-op defaults)

```yaml
uncertainty:
  group: ''                 # '' = per-pixel head (default, base path). 'slot' = FeatureGroupNet.
  group_dim: 64
  group_slots: 3            # K = TriGauge axes
  group_iters: 3            # T slot-attention iterations
  whatkind_weight: 0.0      # CE seg-prior weight. 0 -> loss never computed -> base inert (§3)
  whatkind_label_smooth: 0.1
  whatkind_source: gt       # gt | arm4  (distillation, §3.4)
  whatkind_shuffle: false   # the shuffled-label negative-control arm (§3.3)
```

### 2.4 PARITY — exactly how to stay Inc-0-green

The gate (`test_inc0_bitidentical.py:62-68`) asserts, on the **flags-off build**: torch+cuda **RNG-state after `JointEncoding` construction**, **param count + tensor count**, and **state_dict key set**. A new module adds all three. The rule is mechanical and matches the existing `dino_unc_net` discipline exactly:

1. **Build `dino_group_net` ONLY inside `enable ∧ mode=='dino' ∧ group=='slot'`.** Default (no `uncertainty` block, `enable:false`, `mode:'geo'`, or `group:''`) → the inner guard is false → **no module → no RNG draw, no params, no keys → gate PASS.** The gate forces `enable=False` (`test:50-51`), so even a config naming `group:'slot'` is neutralised in the golden comparison — correct.
2. **Construct LAST** (after every base module AND after `dino_unc_net`) so the ON run's base backbone draws identical RNG → clean single-variable A/B (the `decoder.py:467-471` rationale).
3. **Gate the new `rend_dict['whatkind']` key on `hasattr(decoder,'dino_group_net')`** so base/geo/per-pixel-dino paths never create it. The parity gate checks module construction, not dict keys, but a stray key would break the per-pixel dino arm's A/B — so keep it strictly behind the module's existence.
4. **`whatkind_weight=0 ∧ group=''` → no module, no key, no loss → base bit-identical.** Even `group='slot'` with `whatkind_weight=0` adds zero loss ops (the CE term is computed-never), mirroring `def_reg`/`nll`/`consistency_weight`.

---

## 3. LEARNED-NOT-TOLD: what-kind as a seg TRAINING PRIOR

### 3.1 The principle

The what-kind head is **supervised by GT seg as a soft prior during TRAINING; it reads DINO ONLY at inference.** No GT at test. The seg prior does **not teach the head what a tool is** — it teaches the head **which DINO cluster the dataset names "tool"**, aligning the 3 slots to the `{tissue,tool,bg}` naming. At inference the head reads the frozen-DINO clusters (still present on any dataset — DINO is a frozen foundation backbone) and emits the learned naming → the naming transfers. This is the explicit rejection of a hard `if class==tool` rule (which is one-dataset and not learned).

### 3.2 The loss (mapping-only, behind a default-0 weight → parity)

```
L_kind = whatkind_weight · CE_labelsmooth( whatkind_logits[valid] , seg_label[valid] )
```
Aggregated in `get_loss_from_ret` behind `whatkind_weight` (mirror the `nll_weight`/`def_reg` pattern), gated `not tracking` like the NLL (`scene_rep.py:599`) — mapping-only.

### 3.3 The seg label — MUST be surfaced fresh (the load-bearing caveat)

The per-ray seg signal already in `forward`, **`target_edge_semantic`** (`scene_rep.py:493`, gathered `ddsslam.py:276`), is **NOT a class label** — `compute_edge_semantic` (`dataset.py:36-53`, confirmed) runs `cv2.Canny` then `exp(-distanceTransform/10)`, collapsing the mask to a soft **edge-proximity scalar**. **It is unusable as a 3-way target.** You must surface the RAW per-class mask:

- Add a `_attach_seg_label(ret, index)` to the dataset, **modelled exactly on `_attach_deform` (`dataset.py:120`)** — store the raw mask `batch['seg_label']` (canonicalised per-dataset: CRCD `{0bg,1Liver,2Gallbladder,3Tool}` → `{bg, tissue, tool}`, Liver+Gallbladder→tissue, per the seg-policy memory). `None` ⇒ no key ⇒ base bit-identical.
- Gather it per-ray at the same `[indice_h, indice_w]` as `target_d`/`target_dino` (`ddsslam.py:276-279`), pass as a new `forward` kwarg `target_seg=`. **Only read it when `whatkind_weight>0`** so off-path adds no op.

### 3.4 The defensibility triad (the thesis-grade discriminator — one flag, three arms)

| Arm | `whatkind_weight` / flag | Label | Isolates | Proves |
|---|---|---|---|---|
| **seg-supervised** | `>0`, `shuffle:false` | true GT seg | full method | upper bound: does named attribution help routing |
| **pure-DINO grouping** | `0` | none | the *unsupervised* grouping | grouping is **DINO-intrinsic** — if σ²-coherence + routing benefit survive with NO label, attribution is foundation-driven, not a CRCD-fit. **The generalisation evidence.** Slot-attention is *designed* for this arm (object discovery without labels). |
| **shuffled-label** | `>0`, `shuffle:true` | seg ids permuted across classes | seg supervision *informative* not just *capacity* | if shuffled ≈ no-seg `<` true-seg → the win came from CORRECT labels. If shuffled ≈ true-seg → attribution is FAKE (gain was capacity/smoothing). **The falsifier.** |

`whatkind_shuffle:true` permutes the seg class ids before CE — one A/B arm, no code-path divergence. **Held-out judging:** train the prior on train snippets, measure what-kind **mIoU on a held-out snippet** (leave-one-snippet-out, the Arm-4 LOSO protocol). True-seg must beat shuffled on held-out mIoU.

### 3.5 Relationship to the Arm-4 DINOv2 seg-head (`DINO2SEG`, `train_dinov2_crcd.py:141`)

The what-kind head is a **fresh, lighter head sharing the in-model pooled feature — NOT the Arm-4 head transplanted.** Arm-4 runs DINOv2 ViT-B/14 (768-dim) + conv-upsample on the **raw RGB at train time**; in-model we have the **baked vits14 384-dim frozen grid** — different backbone, dim, and input. The `.pth` does not load, and Arm-4 is in-domain-limited (0.34 held-out) precisely because it fine-tunes 8 backbone blocks. The frozen baked grid **cannot overfit the backbone** → strictly more transferable.

**Right relationship — Arm-4 is the offline teacher, the in-model head is the online student:**
- **(a) GT-soft-label (`whatkind_source:gt`, default):** supervise directly by GT seg (§3.2). Simplest; triad runs as-is. **Build this first.**
- **(b) Arm-4-distilled (`whatkind_source:arm4`):** supervise by Arm-4's predicted soft seg logits → a prior on datasets with NO GT seg (the path to dataset-agnostic deployment). **Gated behind (a) showing signal** — distilling a 0.34-held-out teacher could inject noise.

Either way the **trunk** (not the backbone) is what is shared, so σ² and what-kind co-shape. Arm-4 moves in-model as a *supervisory signal*, not a module.

---

## 4. STAGED BUILD (each metric-gated)

| Stage | What | New module? | Parity | Arm-2? | Status |
|---|---|---|---|---|---|
| **v0** | #3 cosine-margin σ²-consistency reg (the loss-only floor) | NO (pure loss) | structurally untouched | NO | **specced, ready** (`ARM1_3` spec) — ship first |
| **v1a** | `FeatureGroupNet` (slot, K=3) → region-coherent σ² ONLY (`group:slot`, `whatkind_weight:0`) | YES | gated branch, build-last | NO | **buildable NOW** |
| **v1b** | + what-kind head + GT seg prior + triad (`whatkind_weight>0`, `seg_label`) | (same module) | new key gated on module | NO | **buildable NOW** |
| **v2** | Option-B full-2D-map pool-then-gather (true spatial coherence) | (relocated) | new forward kwarg | NO | gated on v1 scatter-incoherence |
| **combine** | what-kind `p` gates Arm-2 motion→σ² teacher into 3-way TriGauge route | NO (router) | reuses Inc-2 actuator | **YES** | **gated on Arm-2 revival** |

**Buildable NOW (Arm-1 side — needs only DINO + a seg prior, both in hand):** v0, v1a, v1b. This delivers (1) region-coherent σ² along object boundaries, (2) the learned what-kind attribution = the missing HALF, (3) the router's feature-group half wired + triad-validated, (4) tool-as-a-unit down-weighting → ATE_max relief from attribution alone — all **Arm-2-independent**.

**Gated on Arm-2 (the COMBINE):** the **how-much-deformation** temporal signal. What-kind says *which destination*; Arm-2's `E_R−E_D`-contrast→σ² teacher says *how strongly*. **The product is the method.** The combine swaps σ²'s payload (photometric → deformation-aware) and gates each destination by `p = softmax(whatkind)`:
```
bg     → CAMERA ANCHOR : tracking trust ∝ p_bg · clamp(1/σ²)   # Inc-2 actuator (scene_rep.py:560) × p_bg
tissue → DEFORM FIELD  : field teacher gated by p_tissue        # Arm-1 gates Arm-2's per-target trust
tool   → own SE(3)     : excluded from BOTH pose + field (v0: weights→0)  # our 3-way vs NRGS 2-way
```
This reuses the existing detached-weight actuator — no new routing thread. **v0 of the combine simply excludes tool from both** (tracking weight + field-teacher weight → 0) — already a win over the binary rigid/deformable split.

---

## 5. THE METRIC GATE per stage

Judges: `Addons/eval/sigma2_quality.py` (`frac_oracle`, `Pearson(σ², inter-frame motion)`), `sigma2_diagnostics.py` (speckle index), `sim3_ate.py` (ATE_max + path-ratio + Pearson), held-out mIoU (Arm-4 LOSO).

**v0 (#3 reg) — does feature-affinity → coherent σ² at all:** speckle ↓ vs per-pixel `0.496`; `frac_oracle` HELD ≥ geo_rd within seed-std (anti-collapse). Validates the affinity premise before spending a module.

**v1a (σ² head) — STABILISE without breaking calibration:** the discriminator is the **conjunction** dino_reg could NOT do (it dropped frac_oracle 0.773→0.476 to hit speckle 0.018):
- **speckle: 0.496 → < 0.1** (region-coherent, anisotropic — smooths along feature boundaries),
- **frac_oracle HELD ≥ ~0.7** (geo_rd level, within seed-std) — collapse → over-smoothed → revert,
- **motion-corr: 0.122/0.249 → HOLD** (a small rise from group-completion is a bonus, NOT a headline — manufacturing motion-corr is Arm-2 territory).

**v1b (what-kind) — beat the shuffled control AND improve routing:**
- **held-out what-kind mIoU: true-seg `>` shuffled** non-trivially (LOSO). True≈shuffled → attribution is FAKE → strip the head, keep σ²-stabilisation only ("anisotropic stabiliser", not "attribution"). The gate is the *gap over shuffled*, not absolute mIoU (frozen-DINO will be lower-absolute than Arm-4's fine-tuned 0.34 but generalise better).
- **ATE_max ↓ vs the σ²-only (v1a) cell** (n=1-rigor: effect-size vs the ~0.2–0.3 mm seed-std floor, coherence across path-ratio + Pearson + the σ² judge). Headline **ATE_max + held-out mIoU gap**; ATE_mean is at-the-floor.

**Decision rule:** PASS = (v1a: speckle↓ ∧ frac_oracle held ∧ motion-corr held) ∧ (v1b: held-out mIoU beats shuffled ∧ ATE_max ↓ vs σ²-only). FAIL on frac_oracle collapse → revert (broke Inc-1's mapping job). what-kind ≈ shuffled → strip what-kind, keep σ²-stabilisation.

---

## 6. OPEN DECISIONS, BIGGEST RISK, CHEAPEST DE-RISK

### Open decisions for the user (2–3)

1. **Within-batch slot binding (Option A) vs full-2D pre-render (Option B) for v1.** Recommendation: **A** (no plumbing, parity-trivial, slot binding is feature-driven so scatter is irrelevant); defer B to a v2 A/B only if v1 slots are spatially incoherent on the σ² panel. **Confirm A for v1, or insist on B's spatial fidelity up-front?**
2. **Seg-prior source for v1b: GT-soft-label (a) vs Arm-4-distilled (b).** Recommendation: **(a) first** (clean triad, no teacher-noise); (b) only after Arm-4 held-out improves, for dataset-agnostic deployment. **Confirm (a)-first.**
3. **Does v1a (σ²-stabilisation) ship independently if v1b's what-kind ≈ shuffled?** Recommendation: **yes** — keep `FeatureGroupNet` as an anisotropic stabiliser (a real win over the per-pixel head's speckle), drop the what-kind head. **Confirm this fallback is acceptable as a standalone contribution.**

### The single biggest risk

**The what-kind head collapses to the shuffled control** — i.e. the slot grouping is real but the *gain* is capacity/smoothing, not correct semantic attribution (shuffled ≈ true-seg). That would mean the attribution "engine" is decoration and only the σ²-stabilisation survives — gutting the TriGauge identity-axis claim that is the program's missing half. (Secondary risk: frac_oracle collapse from over-smoothing — but that has a clean revert.)

### The cheapest de-risk

**Run the triad as an offline probe on the FROZEN baked DINO grid BEFORE building `FeatureGroupNet`** — i.e. fit a tiny 3-way linear/MLP head on `dino_grid` features against (true seg) vs (shuffled seg) vs (k-means-only) and measure held-out mIoU. This is a pure-numpy/sklearn bake (no module, no parity exposure, no GPU run) that directly answers "do frozen DINO features carry separable tissue/tool/bg with a real gap over shuffled?" If the gap is absent on the raw features, slot-attention cannot conjure it → skip v1b, ship v1a only. If the gap is strong, v1b is de-risked before a line of model code. This is the Analysis-E-style dry-run the program already uses to gate module spends, and it reuses the existing Arm-4 LOSO split.

---

## Files / anchors (absolute, all verified 2026-06-20)

- Build site: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/model/decoder.py:488-509` (dino branch, build-last) + new `FeatureGroupNet` alongside `UncertaintyDINONet` @307.
- Fuse site: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/model/scene_rep.py:521-536` (replace the `dino_unc_net` call); Inc-2 actuator `:557-562`; NLL `:599-631`; surfaced `ret` `:653`.
- Gather / scatter: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/ddsslam.py:37-50` (`sample_dino_grid`), `:271-288` (mapping gather + forward call; N=`mapping.sample`=2048). **`dino_grid` NOT passed into forward — only `target_dino`** (the fact that picks Option A).
- Seg label: NEW `_attach_seg_label` modelled on `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/datasets/dataset.py:120` (`_attach_deform`). Edge-field (unusable as label) confirmed `:36-53`. DINO grid `:107-118`.
- Arm-4 teacher: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/seg/train_dinov2_crcd.py:141` (`DINO2SEG` — offline teacher/distill source, NOT transplanted).
- Parity gate: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/regression/test_inc0_bitidentical.py` (RNG-after-build + params + keys; forces flags-off).
- σ² judge: `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/eval/sigma2_quality.py` (`frac_oracle`, motion-corr) + `sigma2_diagnostics.py` (speckle).
- v0 floor (the loss-only primitive `FeatureGroupNet` subsumes): `c:/Users/benli/OneDrive/Documents/GitHub/DDS-SLAM/DDS-SLAM/Addons/docs/ARM1_3_FEATURE_SIGMA_CONSISTENCY_SPEC_20260619.md`.

Prior art: Locatello et al. *Object-Centric Learning with Slot Attention* NeurIPS 2020 (the chosen core); Zhu et al. *SNI-SLAM* CVPR 2024 (attention feature-collaboration, IRMVLab lineage); Caron et al. ICCV 2021 (DINO emergent object clustering — the unsupervised-arm enabler); Wang et al. *Non-local Networks* CVPR 2018 (the N² primitive we reject for slot-attention); NRGS-SLAM (binary 2-way deformation posterior — our continuous 3-way slot attribution is the differentiation).

**Note:** I did not write this to a file (per instruction not to create .md report files). If you want it committed as `Addons/docs/ARM1_4_FEATURE_GROUP_ATTRIBUTION_SPEC.md`, say so and I'll write + stage it.