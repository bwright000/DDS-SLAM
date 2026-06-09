# DDS-SLAM Diagnosis + Improvement — live session plan (2026-06-11)

Goal: **Part A** — decisively separate the *cause* of the dead deformation field (the headline open question in `failure_mode_audit.md`), then **Part B** — begin the revival demo. Run together, debug live.

Maps our audit's prioritized interventions (#1/#2/#5) onto the D1–D5 battery. Prereqs: a **dead-field SemSup checkpoint** (paperfaith, from the overnight) + DDS env on Colab. SemSup-first (near-static camera ⇒ all motion is deformation ⇒ cleanest test).

---

## Part A — cause-separation battery (run in order; each step gates the next)

### Step 0 — weights vs output `BUILT tonight, zero-risk`
`python diagnosis/infra/timenet_weight_audit.py --ckpt <semsup_paperfaith.pt> --json diagnosis/report/timenet_weights.json`
+ re-run `dx_hook` retaining `summary.json` + raw NPZ (commit the reproducibility link — audit #1).
- **DEAD-WEIGHTS** (TimeNet max|.| < 1e-6, backbone live) ⇒ cause is weight collapse → prioritise A-D5 (decay) + A1 (encoding).
- **TRAINED-SILENT** (live weights, dx=0) ⇒ gauge / no-reward → prioritise A2 + A3 + A4.

### A1 — frame_time encoding fix (audit #2a, our D3) `build live`
Integer `frame_time` → tinycudann Frequency gives only frame **parity** (`sin(kπ·int)=0`). Re-enable `/n_imgs` normalisation (`ddsslam.py:560` + :246/:329/:445). Short SemSup retrain, re-measure dx.
- dx>0 ⇒ impoverished time-code was a killer. dx=0 ⇒ not the encoding.

### A2 — reward + gauge-break (audit #2b) `build live`
Wire the **dead** `time_smoothness_weight` into the loss (`ddsslam.py:209`): `‖vox_motion‖²` + temporal/spatial coherence, in **all phases** (not just BA). AND/OR subtract per-frame mean of `vox_motion` before adding to points (`scene_rep.py:180`) to break the rigid gauge.
- dx>0 ⇒ "no reward + two-sided gauge" confirmed. dx=0 ⇒ deeper.

### A3 — frozen-pose + grid-freeze (Fable's frozen-GT-pose, de-confounded) `build live`
Freeze pose (identity for near-static SemSup; GT for StereoMIS/CRCD) **and freeze the canonical hash grid**, train **only** TimeNet. (Frozen pose alone is NOT decisive — the grid is a co-absorber; must freeze both.)
- dx>0 ⇒ field is *capable*, was out-raced (gauge/grid starvation). dx=0 ⇒ genuinely dormant (weights/encoding) → back to Step 0/A1.

### A4 — synthetic controlled-motion sentinel (audit #5, our D4) `build live — THE decisive test`
New `diagnosis/phase2/`: (i) **pure deformation + fixed camera** with a *known injected* warp; (ii) **pure camera + zero deformation**. Run dx_hook + test1 on each.
- pure-deform, dx=0 while pose moves ⇒ **gauge absorption PROVEN** (the headline confound — sub-SNR jitter vs gauge — finally separated).
- pure-deform, dx tracks the injected warp ⇒ field works; the real-data failure is data/SNR, not architecture.

### A-D5 — weight_decay ablation `build live`
`weight_decay=1e-6` on `decoder.parameters()` (`ddsslam.py:661,671`) → toggle to 0 for the TimeNet group; retrain; does the collapse soften? (Tests the multiplicative-shrinkage story behind the 1e-23/1e-39 weights.)

**Output of Part A:** a causal decomposition (gauge / starvation / decay / encoding — which are binding), with figures. *This is the thesis's novel core.*

---

## Part B — revival demo (begin live, once cause is known)

Harden the gate spec against the verified dead-fixed-point, then build inc 1→5 test-as-you-go:
- **Step-0 bootstrap = stereo-ICP residual** (pose-independent deformation evidence, alive at dx=0 — resolves the posterior degeneracy; all 3 datasets have stereo).
- live-init / anneal `m`; **`detach(M_def)`** in the pose loss; **concentration prior** (NeRF-W L1 / D²NeRF skewed entropy); decoupled forced-live E^D render.
- **Success:** ‖Δx‖>0, the 30↔22 render gap closes, Sim3/tracking improves.
- **Honest baselines to beat:** dilated tool mask + Huber pose; Hayoz-style learned per-pixel weight with NO deformation field; static mask.

---

## Build-live checklist (gaps; the rest is run+interpret)
- [ ] A1 frame_time `/n_imgs` toggle (flag-gated)
- [ ] A2 `time_smoothness_weight` wiring + per-frame-mean gauge-break (flag-gated)
- [ ] A3 freeze-pose + freeze-grid flags in the optimiser setup
- [ ] A4 synthetic deformation injector + `phase2/` sentinel (THE decisive test)
- [ ] A-D5 weight_decay-per-group toggle
- [x] Step-0 TimeNet weight audit (`diagnosis/infra/timenet_weight_audit.py`)

All flag-gated, default OFF, so the normal pipeline is unchanged. Cite NRGS / Hayoz / D²NeRF / NeRF-W as pre-empting baselines in the write-up.
