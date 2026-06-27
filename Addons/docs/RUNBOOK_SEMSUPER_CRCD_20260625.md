# RUNBOOK 2 — Semantic-SuPer CRCD revival (matplotlib OOM fixed)

**Goal.** Re-run the Semantic-SuPer CRCD onboarding that we earlier shelved. The two blockers are now
fixed in `Addons/colab/run_semanticsuper.sh`:
- **matplotlib OOM (the ~195-frame kill):** `plot_pcd()` (`Python-SuPer/utils/utils.py:327`) opened a
  figure every viz frame and never closed it — both the `filename is None` branch (returns the image
  @353 then `return`) and the `savefig` branch (@358). At `--save_sample_freq 1` the figures accumulated
  → system-RAM OOM. **Fix:** two new idempotent `ss_patch_crcd` patches `[DDS-crcd-pltclose-a/-b]` insert
  `plt.close(fig)` on **both** paths (a: after the image is extracted, before `return`; b: after `savefig`),
  and `utils/utils.py` was added to the clean-slate `git checkout` list so the patch re-applies on re-run.
- **no-green-pins crashes:** already guarded — `[DDS-crcd-noeval]` early-returns `evaluate()` when
  `evaluate_tracking=False` (CRCD omits `--tracking_gt_file`).

`--save_sample_freq 1` is **kept** (the per-frame render-save feeds dense PSNR; the leak is gone now).

**Caveat (unchanged from the drop rationale):** Semantic-SuPer is a deformable-surface tracker, not a
camera-SLAM method; its native metric (green-pin reproj) is unmeasurable on CRCD. This revival gives
**render PSNR/SSIM/LPIPS + video** only. Sim3-ATE/Depth-L1 (Step-2: needs `T_g = deform_verts[-1]`
export + rendered-depth export) stay **deferred** — out of scope tomorrow.

---

## Prerequisites
- **Data (Drive):** `Datasets/CRCD-Published/<EP>/snippet_<SID>/{video_frames or rgb+rgbright, semantic_class/semantic_instance, groundtruth.txt}` + `Datasets/CRCD-Published-MoGe-2/...` + calib `cam_calib/ECM_STEREO_1280x720_L2R_calib_data_opencv.pkl`.
- **Env (ENV-SS):** isolated py3.8/torch1.11+cu113/pytorch3d-0.6.2 (Pulsar). Built by `run_semanticsuper.sh env`.
- **GPU: T4** (sm_75). **NOT A100** — cu113 predates sm_80; the Pulsar build targets T4.
- **Repos:** DDS-SLAM (diagnosis-live) + Python-SuPer (`@ be244fa`) cloned.

---

## Run
```bash
cd /content/DDS-SLAM && git pull
# one-time: build ENV-SS + the Pulsar smoke (skip if the env already exists)
bash Addons/colab/run_semanticsuper.sh env

# full 5-snippet CRCD set (c1_001 c2_001 e3_005 c3_001 g3_001):
nohup bash Addons/colab/run_semanticsuper.sh crcd bench5 > /content/ss_crcd.log 2>&1 &
tail -f /content/ss_crcd.log | grep -a --line-buffered -E "PHASE|patch-crcd|pltclose|Killed|FATAL|Traceback|Tracking|PSNR|SSIM|DONE"
# single snippet: bash Addons/colab/run_semanticsuper.sh crcd c1_001
```

**Watch for:**
- Phase patch step prints `[patch-crcd] utils/utils.py: patched ([DDS-crcd-pltclose-a])` and `(-b)`.
- **It should now pass frame 195 without `Killed`** (the OOM symptom). It runs the full 360 (c1) etc.
- End: `[rename] N render/gt pairs` → `Results: PSNR/SSIM/LPIPS` → `video.mp4` → `.DONE`.

**Output:** `MyDrive/Outputs/SemanticSuPer_crcd_<DATE>/<UP>/` — renders, `render_eval.csv`/`.txt`
(PSNR/SSIM/LPIPS), `video.mp4`. Defaults: 640×360, `CRCD_SEG=GT`, `mesh_step=32`.

---

## OPEN ITEMS
1. **Step-2 metrics deferred.** Sim3-ATE (export `T_g=deform_verts[-1]`) + Depth-L1 (rendered-depth export) are NOT in this revival; render-only tomorrow. `groundtruth.txt` is already staged for a future Sim3.
2. **T4 only.** If only an A100 is available, the Pulsar/cu113 stack won't build — needs a T4 runtime.
3. **If OOM still recurs ~195f:** the `plt.close` patch didn't apply — check the `[patch-crcd] ... pltclose` lines printed; if "anchor missing", the upstream `utils.py` moved (re-confirm lines 353/358).
4. **Seg input = GT** (`CRCD_SEG=GT`, baked to `seg/GT/*.npy`). Feeding the Runbook-1 DINOv2 heads into SuPer is a separate future step.
