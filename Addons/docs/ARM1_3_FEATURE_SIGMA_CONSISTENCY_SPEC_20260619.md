# Arm-1 #3 BUILD SPEC — DINO-feature-space sigma^2-consistency (2026-06-19, wf w9c2d50k8)

Ready-to-implement. Pure loss term (no module/param/RNG) -> parity gate structurally untouched, default-off.
Decisions LOCKED (user 06-19): D1 = cosine-margin headline (RBF secondary), D2 = log-sigma^2 space.
BUILD-GATING NOTE: honest verdict = a TriGauge ROUTER-COMPONENT (feature-group kernel), NOT a ceiling-breaker;
dino-ONLY so it does NOT bring geo jitter-suppression in -> the geo_feat FUSION is the tracking-aligned build.

---

Critical confirmation from the parity gate: it checks **RNG-state-after-construction + param count + state-dict keys**. Since #3 is a **pure loss term** (no new `nn.Module`, no new parameter, no RNG draw at construction), the gate is structurally untouched regardless of the config flag — exactly as the readings assert. The gate only inspects model construction, not the loss path. The `consistency_weight: 0` default keeps the loss path inert too.

I have everything verified against live code. Producing the build spec.

---

# Arm-1 #3 BUILD SPEC — DINO-feature-space σ²-consistency

**Status: ready to implement on GO. All line anchors verified against live `scene_rep.py` / `ddsslam.py` / config family (2026-06-19).** Pure loss term — no new module, no new param, no RNG draw → parity gate (`test_inc0_bitidentical.py`) is structurally untouched, default-off keeps it green without re-running golden.

---

## 1. THE EXACT CHANGE

### 1a. Config flags (all under `uncertainty:`, all no-op defaults)
Add to `configs/CRCD/c1_001_canon_uncert_dino_reg.yaml` (the dino_reg arm is the #3 lineage — it is the current best and #3 *refines* it). Mirror into the `Super/` config family.

```yaml
uncertainty:
  consistency_weight: 0.0      # the ON switch. default 0 -> term computed-never, base inert
  consistency_margin: 0.5      # cosine floor; dissimilar pairs (cos < m) -> zero affinity
  consistency_pow: 2           # sharpen near-threshold affinity
  consistency_kernel: cosine   # 'cosine' (default/portable) | 'rbf' (On-the-go-faithful A/B)
  consistency_tau: median      # rbf bandwidth; 'median' = per-step median-heuristic, never hardcoded
  consistency_anchor: 0.0      # belt-and-braces global-mean anchor; OFF by default (§2d)
```

### 1b. Plug-in point — the loss term itself
**`model/scene_rep.py`, inside `forward()`, after line 612** (`nll_loss = _nll[valid_depth_mask].mean()`), still inside the `if ('sigma2' in rend_dict) and (not tracking):` guard (mapping-only, line 580), before the `else:` at 614:

```python
                # --- ARM-1 #3: DINO-feature-space σ²-consistency. Penalises log-σ²
                # disagreement between rays whose DINO features are SIMILAR (cosine).
                # Generalises dino_reg's SPATIAL smoothing to FEATURE-space neighbours
                # (semantically-alike scattered rays). MAPPING-only, dino-only (target_dino
                # required). Stashed in ret['sigma2_consistency']; ADDED only behind
                # consistency_weight (default 0) in get_loss_from_ret -> base inert.
                # Pure loss op: no module/param/RNG -> Inc-0 parity gate untouched.
                _cw = self.config.get('uncertainty', {}).get('consistency_weight', 0)
                if _cw > 0 and target_dino is not None:
                    _m   = self.config.get('uncertainty', {}).get('consistency_margin', 0.5)
                    _p   = self.config.get('uncertainty', {}).get('consistency_pow', 2)
                    _ker = self.config.get('uncertainty', {}).get('consistency_kernel', 'cosine')
                    _s   = torch.log(_s2).squeeze(-1)[valid_depth_mask]          # [M] log σ², same support as NLL
                    _f   = target_dino[valid_depth_mask].float()                 # [M,C] baked data, no grad
                    if _s.shape[0] >= 2:
                        if _ker == 'cosine':
                            _fn  = torch.nn.functional.normalize(_f, dim=-1)
                            _sim = _fn @ _fn.t()                                 # [M,M] in [-1,1]
                            _W   = torch.relu(_sim - _m).pow(_p)
                        else:  # rbf, median-heuristic bandwidth (Gretton)
                            _d2f = torch.cdist(_f, _f).pow(2)                    # [M,M]
                            _tau = self.config.get('uncertainty', {}).get('consistency_tau', 'median')
                            _bw  = _d2f.detach().median() if _tau == 'median' else float(_tau)
                            _W   = torch.exp(-_d2f / (_bw + 1e-8))
                        _W.fill_diagonal_(0.0)                                   # no self-pairs
                        _dlog = (_s[:, None] - _s[None, :].detach()) ** 2        # [M,M] mean-teacher detach (§2)
                        consistency_loss = (_W * _dlog).sum() / (_W.sum() + 1e-8)
                        # optional global-mean anchor (default off): pin only batch-mean level
                        _aw = self.config.get('uncertainty', {}).get('consistency_anchor', 0)
                        if _aw > 0:
                            consistency_loss = consistency_loss + _aw * (_s.mean() - _s.mean().detach()) ** 2
```

(`_s2`, `valid_depth_mask`, `target_dino` all in scope at this site — verified lines 581, 523, 481. `target_dino` is a baked `.npy` leaf, `requires_grad=False` → `_f`/`_W`/`_sim` carry no gradient; only `_s` is live. No `.detach()` on `_f` needed.)

### 1c. Surface the key
**`model/scene_rep.py:636-637`**, next to `ret['nll']`, same `(not render_only) and (not tracking)` guard:

```python
            if (not render_only) and (not tracking):   # NLL is a MAPPING-only training signal, never on pose
                ret['nll'] = nll_loss
                if (self.config.get('uncertainty', {}).get('consistency_weight', 0) > 0
                        and target_dino is not None):
                    ret['sigma2_consistency'] = consistency_loss
```

### 1d. Aggregate behind the weight — base stays bit-identical
**`ddsslam.py`, in `get_loss_from_ret`, after line 240** (the NLL block), mirroring the `_nll_w` / `def_reg` optional-loss-behind-a-weight pattern verified at 231-240:

```python
        _dc_w = self.config.get('uncertainty', {}).get('consistency_weight', 0)
        if _dc_w > 0 and ret.get('sigma2_consistency') is not None:
            loss += _dc_w * ret['sigma2_consistency']
```

**Parity argument (covered):** `consistency_weight` absent/0 → key never stashed (1c guard) AND never added (1d guard) → zero tensors, zero ops on the off-path. No `nn.Module`, no parameter, no RNG draw at construction → `test_inc0_bitidentical.py` (RNG-after-build + param-count + state-dict-keys) is **structurally** unable to see this change. Same guarantee already proven for `nll` and `def_reg`.

---

## 2. THE FEASIBLE FORM

| Decision | Choice | Why (load-bearing) |
|---|---|---|
| **N×N vs sampled** | **Full dense pairwise**, no chunk | N=mapping.sample=2048 (tracking excluded by `not tracking`; global_BA has `target_dino=None`). 2048² f32 ≈ 16.8 MB sim + 16.8 MB diff ≈ 34 MB transient, single `[M,C]@[C,M]` matmul — few ms on A100/T4. Cap with `idx=randperm(M)[:1024]` ONLY if `mapping.sample` later pushed ≥5k. |
| **Space** | **log σ²** (`torch.log(_s2)`) | σ² spans >25× (speckle 0.018 vs 0.496); a raw-σ² penalty is dominated by tool/specular pairs and ignores tissue interior. log penalises RATIOS (scale-equivariant — correct prior for a multiplicative quantity) and matches the NLL's own `log σ²` coordinate → commensurate gradients, no 1/σ⁴ blow-up. σ² floored ≥1e-6 upstream → log finite. |
| **Similarity** | **cosine + margin** default; **RBF-median** as secondary A/B | DINO semantics live in DIRECTION; cosine is canonical (your own PCA pipeline). `m=0.5` gives a dataset-independent threshold (RBF τ must retune per backbone/dataset). RBF variant uses per-step median bandwidth (Gretton), never a constant. |
| **Stop-gradient** | **Detach the target side**: `(_s[:,None] - _s[None,:].detach())**2` | Both-live coupling = a graph-Laplacian spring `sᵀLs` minimised by collapse-to-constant. Detaching one side = mean-teacher regression of live `s_i` onto fixed pseudo-target (Tarvainen&Valpola 2017; On-the-go's stop-grad). Consistent with stack discipline (Inc-2 detach @541, β-NLL). In-batch detach only — **no EMA network, no state-dict buffer** → parity-clean. Symmetric matrix means every ray is still live in its own row → full coverage. |
| **Anti-collapse** | **margin (primary) + NLL anchor (secondary) + junior weight (operational)** | (a) `m=0.5` zeroes cross-group affinity → reg is a *within-group* smoother; minimiser is "constant per connected component," NOT one global constant (tool σ² and tissue σ² never pulled together). (b) NLL `0.5(err²/σ²+log σ²)` stationary at σ²*=err² pins each group's LEVEL → a global constant has large NLL → not a joint minimiser as long as `nll_weight>0`. (c) keep `consistency_weight/nll_weight ≈ 0.1–0.3`; sweep up only until the judge degrades. |
| **Anchor** | **off by default** (`consistency_anchor=0`) | belt-and-braces global-mean pin; only reach for it if the judge shows global drift. |

**Hyperparameter start:** `consistency_weight=0.03`, `margin=0.5`, `pow=2`, `kernel=cosine`. Sweep `{0.01, 0.03, 0.1}`.

---

## 3. THE METRIC GATE (the arbiter — rectified, n=3, FROM base)

**Control arm = `dino_reg_rd` re-established rectified n=3** (the n=1 raw-left ATE 2.03 / motion-corr 0.249 is NOT the gate — re-run it as the rectified control). #3 is a *paired contrast vs dino_reg*, not vs geo. Per cell: `sigma2_quality.py` (frac_oracle + motion-corr) AND `sigma2_diagnostics.py` (speckle) AND `sim3_ate.py` (rectified) AND 6-panel video.

**PASS requires all three, n=3 non-overlapping:**
1. **Calibration HELD** — `frac_oracle` ≥ dino_reg_rd within seed-std. (FAIL if it drops > one seed-std below — #3 must not trade away render-error calibration. This is the over-smoothing / collapse signature.)
2. **Deformation-awareness ROSE** — `motion-corr = Pearson(σ², inter-frame motion) > 0.249` (rectified dino_reg_rd), non-overlapping. This is #3's reason to exist — feature-grouping lifts σ² on coherent moving-tissue *groups* where dino_reg's spatial reg leaks across the tissue/tool boundary. If flat, #3 added nothing the spatial reg didn't.
3. **Sim3 ATE confirms** — `ATE_mean < 2.03 mm` (beat dino_reg_rd), path-ratio + dominant-axis |Pearson| not worse, n=3 non-overlapping; clearly beats geo. Quote ATE WITH path-ratio + dom-Pearson (sub-SNR rule), never headline alone.

**Decision rule:** PASS = (1)∧(2)∧(3). motion-corr up + flat ATE = WEAK PASS (mechanism works, sub-SNR ATE blind) — keep iff calibration held. ATE down + flat motion-corr = SUSPICIOUS (lucky tracker, mechanism didn't fire) — do NOT credit #3, investigate. **FAIL on calibration = revert regardless of ATE** (broke Inc-1's mapping job). The judge IS the anti-collapse gate: `frac_oracle→0` while speckle keeps dropping = over-weighted → back off `consistency_weight`. Trustworthy because frac_oracle and motion-corr are *designed to pull apart* — passing BOTH (not trading) is strong evidence the grouping is real.

---

## 4. BUILD ORDER + GATING

1. **dino-only by construction** — `target_dino` is gathered ONLY when `'dino_grid' in batch`; geo/off have no per-ray feature → the `_cw>0 and target_dino is not None` gate is naturally inert on geo. **The same config family is therefore safe on geo as a built-in negative control** (geo can't express feature-grouping — this is the structural geo-vs-dino separator).
2. **Fires AFTER** the n=3 rectified confirm of `dino_reg_rd` (re-establishing the control as rectified, since canon −23% was rectified but the dino_reg_rd evidence is raw-left n=1). Do not build against the stale raw-left number.
3. **Default-off, parity-gated** — `test_inc0_bitidentical.py` green with no golden re-write needed (pure loss, no construction change).
4. **Ships both diagnostic sets** (standing rule) — NUMERICAL: `sigma2_quality.py` + `sigma2_diagnostics.py` + `sim3_ate.py` table {Sim3 ATE, frac_oracle, motion-corr, speckle, PSNR/SSIM/LPIPS}. VISUAL: 6-panel video + σ² panel, auto-shipped to `manual_cells/<name>/figs/`. Cells `dino3_rd_s{0,1,2}`.

---

## 5. THE TRIGAUGE DOWN-PAYMENT (one sentence)

#3 leaves the router's **feature-group half fully wired**: the DINO-affinity kernel `K(f_i,f_j)` and the per-ray DINO read are built and validated, and the Inc-2 actuator `clamp(1/σ²).detach()` (scene_rep.py:541) is already the routing weight — so the COMBINE swaps only the *aggregated payload* (σ² → Arm-2 motion) and gates `K` on the live field, adding **no new routing plumbing**.

---

## 6. HONEST VERDICT

**BUILD #3 — yes, decisively, but as a router-component build, not a ceiling-breaker.** Ranked reasons:
1. **The only Arm-1-alone increment that produces a reusable TriGauge component** (kernel `K` + DINO-as-affinity premise validated *before* you debug it entangled with a live field — every other Arm-1 knob is thrown away at combine). De-risking the combine justifies the build alone.
2. **The structural geo-vs-dino separator** — geo_feat has no semantic grouping and literally cannot express "similar-feature → similar σ²." A clean PASS is the first result saying dino beats geo *because of semantics*, not incidental smoothing. Thesis-grade discriminator.
3. **Cheap + parity-safe** — one reg term on the existing per-ray DINO batch.

**The deformation-blind CEILING (state it, don't oversell):** deformation is appearance-preserving → low photometric residual → low σ² on exactly the moving pixels you want flagged. #3 reshapes σ² in feature space; it does **not** inject motion. It can *propagate* a deformation cue that exists on *some* rays of a group to the *rest* of the group (group-completion), but it **cannot manufacture** a cue absent from the residual. Breaking the deformation-blind ceiling is **combine territory** (Arm-2 motion → σ²), full stop.

**Realistic best-case win (metric-first, specific):**
- σ² respecting tissue/tool/bg feature boundaries (clean per-region σ², not dino_reg's isotropic blur that bleeds across them) — visible in the σ² panel.
- `motion-corr` nudging **0.249 → ~0.30** (group-completion, bounded by residual already present — NOT the combine's target levels).
- `frac_oracle` HELD at/above dino_reg_rd (no calibration trade).
- modest `ATE_mean < 2.03 mm` over dino_reg_rd from group-coherent down-weighting (tool down-weighted as a *unit*). **That is a success, not a disappointment.**

**Frame honestly as** "sharper feature-grouped σ² + cleaner tracking down-weight + router affinity kernel pre-validated," NOT "deformation-aware σ²" — reserve that for the combine, or you mis-attribute a combine result to Arm-1 and corrupt the COMBINE-LAST narrative.

**OPEN DECISIONS for you (2):**
- **D1 — kernel default:** I recommend **cosine-margin (m=0.5)** as the headline (portable, DINO-canonical, dataset-independent threshold) with **RBF-median as the secondary A/B**. Flip ONLY if you want strict On-the-go faithfulness as the headline. Your call.
- **D2 — gate space for σ²-difference:** I specced **log σ²** (scale-equivariant, NLL-commensurate). Confirm, or do you want a linear-σ² A/B arm as a control? (I'd default log; one extra arm if you want the contrast on the record.)

I did **not** write any code — this is the spec, ready to implement the instant the n=3 rectified `dino_reg_rd` confirm says go.