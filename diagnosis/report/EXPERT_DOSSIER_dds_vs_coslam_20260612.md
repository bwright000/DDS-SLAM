# Expert Dossier — DDS-SLAM vs Co-SLAM: innovations, reality, and whether it actually wins

Source: workflow wlzja28ry (7 agents, adversarially cross-checked, all claims file:line / paper / result verified), 2026-06-12.
Evidence discipline: PAPER CLAIM vs REPRODUCED REALITY; where they conflict, reality wins.

## Essence
DDS-SLAM (IROS 2024) is a **Co-SLAM fork** for deformable endoscopy. Map + renderer + tracker are byte-for-byte
Co-SLAM (3D hash-grid SDF, sdf2weights, const-velocity tracking, global BA). It adds TWO authors'-original
components absent from Co-SLAM (Co-SLAM source has ZERO edge/time/semantic code; both heads are byte-identical in
pristine DDS-SLAM-Base → genuinely authors'): (1) a **4D deformation field** (TimeNet, x'=x+D(x,t)) — the headline;
(2) a **semantic distance-field loss** (paper's "SDL"; code `EdgeNet_Semantic` / `edge_semantic`). In our reproduced
reality the headline field is **DEAD** (Δx≡0.0 bit-exact, all 311 frames) → DDS-SLAM as-run ≈ static Co-SLAM + a
low-weight semantic-distance head.

## Material corrections (to my own prior claims — IMPORTANT)
1. **Paper StereoMIS headline is 14.833 mm / 2.52× (bold primary row), NOT "8.3 mm / 4.5×".** 8.261 is a stacked
   sub-row. We've been chasing the sub-row. (Table II: Co-SLAM 37.365 vs DDS bold 14.833; sub-rows 19.412/8.261.)
   → updates [[project_stereomis_dds_audit_20260610]] (target is 14.8mm headline, 8.3 a best-case sub-row).
2. **"Semantic-edge head" was my mislabel** → it's the paper's **semantic distance field / SDL** (contribution #3).
3. **The semantic head is DATASET-DEPENDENT and near-inert where the paper claims the win:** loss share ~7-8.5%
   on SemSup but **0.0013% on StereoMIS** (target 2312× smaller — StereoMIS masks are sparse instrument-only). So it
   CANNOT mechanistically explain a StereoMIS tracking edge. Alive mainly on SemSup (which has NO real pose GT —
   load_poses returns identity). Refines the blanket "alive semantic head carries the win".
4. Drop the weight magnitudes (7e-39/4.5e-23) — print-rounding artifacts; the solid fact is bit-exact Δx=0.0.

## Innovation map (delta over Co-SLAM)
| Innovation | Status | Note |
|---|---|---|
| 4D deformation field (TimeNet) — paper #1 headline | **DEAD** | inert; time enters only via TimeNet so Δx=0 ⇒ static == Co-SLAM |
| Semantic distance-field head (EdgeNet_Semantic) — paper #3 (SDL) | **ALIVE** | the ONE alive arch delta; backprops into geometry (shared geo_feat→sdf→grid) AND pose (tracking loss) |
| edge_semantic_loss (5th loss, weight 0.5 hardcoded) | ALIVE | vs sdf=1000; a token regularizer |
| semantic distance target (Canny→distanceTransform→exp(−d/10)) | ALIVE | distance-to-semantic-boundary field |
| robust trimmed edge loss (keep-best-80% + border-exclude) | ALIVE but ATTENUATING | down-weights the already-tiny edge signal |
| endoscopic-scale config (hash_size, n_range_d=16) | ALIVE — but NOT a DDS innovation | generic Co-SLAM knobs; carries MOST of our render recovery |
| deformation regularizer knobs (time_smoothness/eikonal/plane_tv) | **DEAD** | defined in yaml, Grep over *.py = ZERO matches → never wired (primary reason the field dies) |
| integer time encoding (`/n_imgs` commented out) | DEAD pathway | frame_time = raw integer → Frequency enc collapses to parity |
| frame-0 anchor (Δx≡0 at t=0) | inert pin | removes the frame where nonzero Δx is unambiguously rewarded |
| SDF/color/grid/encodings/tracking/BA | IDENTICAL to Co-SLAM | byte-identical across all 3 repos |

## THE ANSWER — "why is DDS-SLAM better than Co-SLAM despite the dead field?"
**Honest: in our reproduced reality it is NOT robustly better.** The apparent edge is thin, ATE-only, single-seed,
alignment-sensitive, confound-dominated.
- **Paper's 2.52× does NOT reproduce** — we never reach paper ATE on either method.
- **One clean matched-config head-to-head** (apr21, fx560/scale10000/lr=1e-3): DDS ~49.94mm vs Co-SLAM 60.82mm =
  ~18% on ATE ONLY, **render quality TIED** (PSNR ~23.9 both — contradicts paper's "superior rendering").
- **Even the ~18% is unsafe:** the canonical "best DDS 49.94 vs best Co-SLAM 43.12" is apples-to-oranges — Co-SLAM-BC0
  was lr-tuned (1e-4), DDS was not. **No matched-lr DDS run exists; equal tuning plausibly inverts the ranking.**
  Plus n=1 seed, alignment-block cherry-pick, and the semantic head ~0% on StereoMIS.
- **Strongest real lever = generic config/encoding capacity (hash_size 16→19, n_range_d 11→16), which is NOT a DDS
  contribution.** It drove our SemSup PSNR to 30.148 (exceeds paper). The semantic head is a minor dataset-dependent
  regularizer. The deformation field is dead.

**Bottom line: DDS-SLAM's reproducible advantage over Co-SLAM is not demonstrated at matched tuning.** Any genuine
edge is config + a small semantic regularizer — NOT the named deformable/dynamic contribution.

## What the dead field costs
The field is the entire basis for claiming superiority over rigid neural SLAM on deformable scenes. Dead ⇒ DDS-SLAM
has nothing to model breathing/tool motion; fails identically to Co-SLAM on StereoMIS. The paper's only direct
evidence the field helps (Table III w/o-HNDSR 9.834 vs Full 8.261) is unreconstructable — our "full" already behaves
like "w/o HNDSR". Five file:line-cited death causes: no deformation regularizer wired; dead reg knobs; frame-0 anchor;
integer-time→parity encoding; weight_decay vs a rich 3D grid that out-competes the field (pose↔field gauge the pose wins).

## THE DECISIVE EXPERIMENT (not yet run) — what the whole DDS>Co claim rests on
StereoMIS P2_1 back-4000, **matched config + matched lr_trans=1e-4 + ≥3 seeds**, 4-cell:
- A = Co-SLAM (parent) · B = DDS full · C = DDS `edge_semantic=False` (isolate the head) · D = DDS `deformation_off=True` (field control)
Read: B≈C≈A ⇒ DDS has NO reproducible edge (all config). B<C≈A ⇒ the semantic head IS the differentiator. B≈D ⇒ field inert (expected).
Add SemSup as a 2nd cell (only place the head carries real weight).

## Caveats
Dead verified only in OUR checkpoints (paper's own released checkpoints never run — can't assert universal deadness).
The "50.17/60.49" pair quoted elsewhere appears only in a derived doc; real provenance is apr21 49.94/60.82.

## Thesis implication
This NUKES the "DDS-SLAM > Co-SLAM" premise as a settled fact → strengthens the **redesign** direction
([[project_hypconfirm_semsup_20260611]]): a redesign that makes the deformation field actually work + makes the
semantic supervision genuinely load-bearing (dense / tool-exclusion, not a 0.0013%-weight scalar) is a well-motivated
constructive contribution. Full evidence + source ledger in workflow wlzja28ry output.
