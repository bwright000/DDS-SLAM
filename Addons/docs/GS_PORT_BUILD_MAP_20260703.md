# GS PORT & BUILD MAP — "Corrected DDS-SLAM" on Gaussian Splatting (2026-07-03)

> **⚠️ REVISION 2026-07-04 — read `GS_SOTA_REVIEW_20260704.md` WITH this doc.** A 9-agent SoTA
> sweep + local code audits changed material parts of this plan: (1) P2 streaming-time is now
> keyframe-anchored absolute-time bases + partial activation + replay (Free-DyGS/NRGS recipe),
> NOT sliding-window local time; (2) NRGS-SLAM (Feb 2026, co-authored by the DDS-SLAM lab's
> Hesheng Wang — IRMVLab has pivoted to GS) occupies the P1+P2 square on paper, code unreleased →
> our claim = first RELEASED artifact + teacher supervision + tool-as-object + protocol factorial;
> (3) P3 repositioned: T2GS (MICCAI 25) models the tool OFFLINE → our claim is the ONLINE
> tool-object with lifecycle; (4) 🚨 deform hooks must cover BOTH transform_to_frame AND
> transform_to_frame_eval or held-state eval silently violates the BUILD LAW; (5) 🚨 depth-L1 vs
> MoGe-2 is circular — add C3VD + independent stereo depth for any reconstruction claim;
> (6) path corrections below (teacher baker = Addons/deform/…; no Addons/tracking/ — extract
> functions from flow_track.py, never import it).

> **Decision (user, 2026-07-03): GS is THE platform.** Port a *corrected DDS-SLAM* onto the
> EndoGSLAM base, then port our improvements on top. **Budget: ~3 months to paper submission**
> (≈10 build weeks + ~2 weeks freeze/write). This doc supersedes the phasing in
> `GS_MIGRATION_HANDOVER_20260622.md` §5–6 where they conflict; Part II of that doc (code traces,
> hook map, D1–D7) remains the engineering substrate and is referenced, not repeated.

---

## 0. What changed since the 06-22 handover (re-based premises)

Three findings from the 2026-07-03 oracle campaign (memory: `project_field_autopsy_20260703`)
rewrite the plan's assumptions:

1. **Deformation IS a render/reconstruction win — the handover's §8 "don't revive
   deformation-as-render-win" is OVERTURNED.** Held-state eval (re-render all frames from the
   FINAL map): a correct field = **+1.25 dB PSNR, SSIM +0.065, LPIPS −0.092, depth-L1 −33%**
   (SemSup trail_3, paired, 140/151 frames). The 2×2 factorial decomposes it exactly:
   **fusion +0.56 dB** (warping training data to canonical → cleaner map *everywhere*, even
   rendered unwarped) ⊕ **gauge-consistent render-warp +0.69 dB** (150/151 frames — but ≈0 on a
   map trained unwarped). Phase-2 deformation is therefore **evidenced, not a gamble** — with a
   measured value ceiling and a build law.

2. **THE BUILD LAW (locked by the factorial): apply deformation in BOTH fusion and render, in the
   SAME gauge, and judge HELD-STATE.** A warp bolted onto a mismatched map gains nothing for
   appearance (−0.07 dB) though it still fixes geometry (depth-L1 −23%). For GS: warp the
   Gaussians (or equivalently the incoming observations) to canonical during MAPPING, and apply
   the same per-frame deformation at RENDER. Per-blob deformation makes this native.

3. **Protocol re-base — the "6 dB GS render gap" was protocol-confounded.** EndoGSLAM's
   `gs_eval.py` renders offline from the FINAL `params.npz` = **held-state**. The DDS 26.5 PSNR
   yardstick is the **recency** protocol (each frame rendered right after being fit). These are
   not comparable: on SemSup the same DDS run scores 29.0 recency vs **18.1 held-state** (~11 dB
   protocol gap). ⇒ **Never compare across protocols again.** Early task T0.3 establishes the
   matched-protocol yardstick on CRCD. Corollary: GS's 20.4 held-state may already be *ahead* of
   the NeRF map as a reconstruction; and held-state is the honest protocol for a SLAM *map* —
   we adopt it as the paper's reconstruction metric (recency reported alongside for
   comparability with prior work).

Also carried in from the 06-27→07-02 NeRF sessions (all are port-relevant corrections):
- **σ² uncertainty confound (`uncfix`)**: the Inc-1/2 port must use the CORRECTED head (empty-ray
  inversion fix + rebalance; `project_model_review_20260702`) — do not port the pre-review code.
- **Flow gate is condemned** (inverts on tools, freeze-luck; `project_flow_gate_inverts_tool_e3005`).
  Its replacement lineage: **zero-motion prior in the tracking loop** (`project_leancore_...`) and
  the **region-VOTE detector** (`project_vote_detector_20260702`, label-free tool exclusion) —
  whichever wins the pending vote_scan bench is the gate we port. Do NOT port `flow_track.py`.
- **Pose-save semantics lesson** (`project_kf_composition_artifact_20260702`): whatever the GS
  loop saves as `est_c2w`, save ONE consistent basis (+ a `_raw`), or the eval sees sawtooth.
- **Time-MLP is wrong-slot** (autopsy): no global time-conditioned MLP anywhere in the GS build.
  Deformation state must be per-Gaussian and temporally local (RBF basis / per-KF states).

## 1. Target architecture — what "corrected DDS-SLAM on GS" means

DDS-SLAM's paper promised: surgical RGB-D SLAM + deformation modelling + edge-semantic guidance.
What shipped: a Co-SLAM fork whose deformation field is dead (denormal-collapsed), whose seg
input is collapsed to Canny edges, and whose eval protocol cannot detect either. The GS build is
that promise, built correctly:

| Subsystem | DDS-SLAM (as shipped) | Corrected-on-GS |
|---|---|---|
| Map | canonical SDF hash-grid (one value/location, no time axis) | 3DGS param dict; silhouette/depth-mismatch spawn + prune (topology-capable) |
| Tracking | SDF gradient-descent pose opt (jittery, sub-SNR) | render-and-compare photometric+depth (EndoGSLAM; observed cleaner) + **zero-motion prior / vote-gate** port |
| Deformation | global time-MLP, render-loss-only ⇒ starved ⇒ dead | **per-Gaussian RBF basis** (Deform3DGS lift, ~10 lines pre-rasterizer), **taught by the Δx\* teacher**, applied fusion+render same gauge (BUILD LAW) |
| Uncertainty | none (our Inc-1/2 was the add) | port Inc-1 σ² (image-space DINO head, `uncfix` corrected) + Inc-2 1/σ² tracking down-weight; free ablation baseline = EndoGSLAM's rendered depth-variance |
| Semantics | seg collapsed to Canny edge weight | seg = **required input** ({tool,tissue,bg}); tool exclusion label-free (vote detector) or seg-gated |
| Eval | recency render + rigid-Horn ATE (both blind) | **held-state render** (default in gs_eval) + Sim3-ATE + path-ratio/Pearson + pin-EPE + 6-panel video |

## 2. Port inventory (component → source → target hook → status)

**Already DONE (in `Addons/gs/` + workspace clones — more than the handover records):**
| Asset | State |
|---|---|
| EndoGSLAM clone + pin (6338c643) + `setup_endogslam_crcd.sh` bootstrap + `apply_patches.py` | DONE |
| CRCD loader + config (`overlay/datasets/.../crcd.py`, `overlay/configs/`) | DONE, ran |
| `gs_eval.py` (held-state render + 5 metrics + 6-panel + Drive ship) | DONE |
| `eval_sim3_crcd.py` (Sim3 repoint) | DONE |
| Phase-0 seed-0 run: Sim3 ATE 4.21 mm (vs SGS 3.31 / NeRF 3.15), \|Pearson\| 0.85, path-ratio 93× | RAN (n=1) |
| Render characterization: base ~20.4 held-state PSNR; full-SH WORSE (drop it); SGS peer 22.6 | RAN |
| v1 flow-map A/B machinery (`apply_patches_flowmap.py`, runbook, inc0 regression test) | BUILT, unrun/parked |
| Attribution panel (`attribution_panel.py`, 5-snip runbook) | BUILT |
| SGS-SLAM 5-snippet peer baseline renders | ON DISK (F:) |

**To PORT (from DDS-SLAM / siblings — sources verified):**
| Component | Source | Target hook (handover §10 map) | Effort |
|---|---|---|---|
| Inc-1 σ² head (image-space, DINO, `uncfix`-corrected) | `DDS-SLAM/model/scene_rep.py` σ² path + `WildGS-SLAM/src/utils/dyn_uncertainty/` | post-render in `get_loss` (mapping NLL) | ~3 d |
| Inc-2 tracking down-weight (1/σ², clipped, detached) | `DDS-SLAM` Inc-2 + normalisation lesson | `get_loss` tracking residual reduction (~262–274) | ~1 d (after Inc-1) |
| Tracking gate: zero-motion prior OR vote detector | EXTRACT functions from `Addons/motion/flow_track.py` — `zero_motion_prior` (:427-444), `region_vote`/`_vote_fit` (:319-424); NEVER import the module (rest of file = condemned flow gate). Also port `_still_gate_decide` (ddsslam.py:822-868, map-anchored still test — near-verbatim in GS) | tracking loss / pre-solve weight | ~3 d (pick by vote_scan) |
| Δx\* teacher bake + regauge | `Addons/deform/generate_deform_targets.py`, `Addons/deform/regauge_deform_targets.py` (path corrected 07-04; CRCD bakes need --est_c2w + CRCD intrinsics + OpenCV rays) | offline bake (env-agnostic .npz) — reuse as-is | ~1 d |
| Teacher supervision → ψ_g(t) | `deform_teacher_loss` concept (D7: Δx\* supervises nearest-GAUSSIAN trajectory — a Gaussian IS a surface point) | new loss on `_coefs` during mapping | in P2 |
| Held-state A/B eval | `Addons/experiments/heldstate_eval.py` (protocol + factorial compare) | gs_eval already held-state; add paired compare + masked split + factorial mode | ~1 d |
| pin-EPE judge | `Addons/eval/field_warped_pin_epe.py` (judge via ψ_g instead of time_net) | offline judge | ~1 d |
| Sim3/video/DINO baker | `Addons/eval/sim3_ate.py`, `Addons/viz/generate_video.py`, `Addons/dino/` | already repointed / reuse | done/trivial |
| SemSup trail_3 loader | pattern of `crcd.py` + `DDS-SLAM/datasets/dataset.py` | new gradslam loader (render+pins bench; NEVER SemSup ATE) | ~1 d |

**To LIFT (from siblings):**
| Component | Source | Note |
|---|---|---|
| Per-Gaussian RBF motion basis (`_coefs[N,10,3,17]`, ψ(t), deform table) | `Deform3DGS/scene/flexible_deform_model.py:498–544` | ~10 lines pure Python pre-rasterizer (handover §11) — NO CUDA; drop its batch loop/known-pose/global-time |
| SemGauss rasterizer (per-Gaussian sem/routing channel) | `SemGauss-SLAM/` | **Phase-3 ONLY if needed** (D2); stock rasterizer through P2 |

**To BUILD (net-new research):**
| Item | The research | Phase |
|---|---|---|
| Streaming-time basis (D4) | RBF μ_j needs global clip-time; online has none. First cut: sliding-window local time; fallback: learned time-encoder | P2 (the crux) |
| Gauge-consistent online deform loop | BUILD LAW in a streaming loop: deform at fusion (`transform_to_frame` before w2c) AND at render, same ψ state | P2 |
| Tool-as-object | seg-gated attribution of `add_new_gaussians` spawns → per-object index set + SE(3)/frame + enter/exit lifetime + composite | P3 |

## 3. Phase plan (10 build weeks; every phase gate = ship-or-descope, nothing blocks the paper)

**P0-finish (wk 1) — close the Phase-0 gate properly.**
- T0.1 n=3 CRCD c1_001 (seed-0 said 4.21 mm; need seed-std + path-ratio story — 93× jitter is
  the Inc-2 target, document it as the baseline defect).
- T0.2 MoGe per-frame scale-flicker audit (the real B1; handover §12).
- T0.3 **Matched-protocol yardstick**: DDS-SLAM held-state on CRCD c1_001 (adapt
  `heldstate_eval.py` frame source; its ckpts exist) vs GS held-state 20.4. This replaces the
  confounded 26.5-vs-20.4 comparison in all narrative.
- T0.4 SemSup trail_3 loader (unblocks teacher/pins later; render-only).
- GATE G0: GS base characterized n=3 with honest yardstick → P1. (Base already ran; low risk.)

**P1 (wk 2–3) — port the proven wins = the guaranteed paper floor.**
- T1.1 Inc-1 σ² head (image-space; `uncfix`-corrected; ablate vs EndoGSLAM's free depth-variance).
- T1.2 Inc-2 tracking down-weight (one-liner at the reduction; mind loss normalisation).
- T1.3 Tracking gate: whichever of {zero-motion prior, vote detector} the vote_scan bench favors.
  (If vote_scan hasn't run by wk 2, run it FIRST on the NeRF side — it's built.)
- GATE G1: n=3, flags-off parity vs P0 base; target = replicate the −23%-class ATE win (and/or
  path-ratio 93×→O(10)). **Whatever passes here is publishable regardless of P2/P3.**

**P2 (wk 4–7) — deformation, now EVIDENCED (the +1.25 dB ceiling).**
- T2.1 RBF basis lift into `transform_to_frame` (rigid parity at `_coefs=0` — regression-gated).
- T2.2 Streaming time (D4): sliding-window local time first; measure basis-phase stability.
- T2.3 Teacher: bake Δx\* (SemSup first — pins exist), supervise nearest-Gaussian ψ_g(t) during
  mapping (isolated optimizer group — the NeRF lesson: a dominant loss must not own the
  deformation params).
- T2.4 **BUILD-LAW wiring**: same ψ applied at fusion AND at render; held-state judge + pin-EPE
  + the 2×2 factorial (our new standard ablation: {deform-fusion on/off} × {deform-render on/off}).
- GATE G2 (wk 7 hard stop): held-state render/depth-L1 beats P1 base (n=3) AND pin-EPE >> shuffled.
  Kill-criterion: if streaming-time can't stabilize by wk 6, descope to per-KF piecewise-constant
  deform states (still satisfies the BUILD LAW, weaker basis) — ship that.
- CRCD transfer check only (sub-SNR; never headline CRCD deformation).

**Tool-entry PROBE (wk 7, 2 days, GO/NO-GO for P3).**
- Mid-sequence tool entry clip (CRCD w/ seg or STIR): does seg-gated spawn render the tool sharply
  vs the SDF ghost? Metric: tool-region PSNR/LPIPS over the entry window + side-by-side panel.

**P3 (wk 8–10) — tool-as-object (the headline if the probe passes).**
- T3.1 Attribute spawns: seg-gated tool index set at `add_new_gaussians`.
- T3.2 Per-object SE(3) per frame (tracked like a second camera) + enter/exit lifetime.
- T3.3 Composite render; rigid-vs-deform routing at the contact boundary ONLY if time allows
  (NRGS overlap — do not headline routing).
- SemGauss rasterizer swap ONLY if per-Gaussian channels become necessary (D2 says defer).
- Probe fails → P3 becomes "honest negative on topology" + the paper ships P1+P2.

**Freeze (wk 10) → write-up.** All results n=3, two diagnostic sets, Sim3+path-ratio+Pearson.

## 4. Paper skeleton (what each phase buys)

1. **Motivation/diagnosis (banked, NeRF side):** DDS-SLAM's field is dead (denormal autopsy);
   its protocol cannot detect it (oracle online −0.24 dB); the DOF is real (+1.25 dB held-state,
   factorial decomposition + BUILD LAW). ⇒ *why a per-primitive, fusion-integrated deformation
   representation on an editable substrate.*
2. **Method:** corrected DDS-SLAM on GS (§1 table) = EndoGSLAM loop + σ²/gate ports + RBF deform
   under the BUILD LAW + (P3) tool-as-object.
3. **Eval contribution:** held-state protocol for deformable SLAM maps (+ the 2×2 factorial
   ablation), pin-EPE, Sim3/path-ratio/Pearson for sub-SNR.
4. **Results:** G1 tracking wins; G2 held-state deform wins; probe/P3 tool renders.
   Fallback story at every gate is a complete paper (P1-only = "corrected surgical GS-SLAM with
   uncertainty"; +P2 = "+online deformation"; +P3 = the full factorized-scene claim).

## 5. Risks & standing decisions

- **D4 streaming-time = the research risk** (4 wks budgeted, kill-criterion wk 6 → per-KF states).
- **Compute:** A100 via tunnel; sm_80 rasterizer rebuild each box (bootstrap handles); n=3 always.
- **License:** EndoGSLAM has no LICENSE file — contact authors before submission; rasterizer dep
  is licensed (Luiten). Track.
- **NRGS (2602.17182):** read in full before novelty claims; don't headline routing.
- **CRCD sub-SNR:** tracking wins need path-ratio+Pearson framing; deformation evidence lives on
  SemSup (pins) + STIR if ported — CRCD is transfer-check only.
- **Do-NOTs carried over:** no rigid-Horn ATE, no stale 271-row GT, no full-SH (measured worse),
  no `flow_track.py` port, no global time-MLP, no Deform3DGS batch loop/rasterizer.

## 6. Immediate next actions (this week)

1. Run **vote_scan** bench (NeRF side, already built) — decides the P1.3 gate port.
2. **T0.1** n=3 Phase-0 CRCD runbook (extend `overnight_crcd.sh`).
3. **T0.3** matched-protocol yardstick (adapt `heldstate_eval.py` to CRCD ckpts).
4. **T0.4** SemSup gradslam loader.
