#!/usr/bin/env python3
"""Attribution panel (GS v1 de-risk) — does the consensus flow residual carry real independent-motion
signal, or is it noise?  THE cheap GO/NO-GO before building v1 (residual -> mapping catch-up).

The sensor (Addons/motion/flow_track.flow_residual): RAFT dense flow -> ONE fundamental-matrix fit ->
per-pixel Sampson distance.  Camera-explained motion (incl. parallax) ~0; motion the camera CAN'T
explain (deforming/moving tissue, the TOOL) is high.  We can't label deformation in CRCD, but we CAN
label the TOOL (seg class 3) — an unambiguous independently-moving object.  So the test:

  Does the residual sit on the TOOL more than on static tissue, beyond two NULLS?
    NULL-A (label-shuffle):     can R classify tool-vs-tissue pixels above chance?  -> AUC vs 0.5
    NULL-B (spatial misalign):  pair frame i's residual with a DIFFERENT frame's tool mask
                                -> if R genuinely sits on the real tool, the ratio collapses to ~1.

GO  => the residual localizes independent motion -> build v1 confident.
NO-GO => the residual is noise wrt real motion -> v1 is mis-aimed (the fix is depth-scale/LR, not a gate).

Ships results.txt (numbers + verdict) + a 4-col visual panel to Drive (two-diagnostic-sets rule).
Reuses flow_track byte-for-byte.  Readable logging: stage banners + heartbeat + final summary.

  python Addons/gs/attribution_panel.py --scene /content/EndoGSLAM/data/CRCD/C1_001
"""
import argparse
import glob
import os
import shutil
import sys
import time
from collections import deque

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'motion'))
import flow_track as ft   # noqa: E402

TOOL_IDS = (3,)            # CRCD seg legend: 0 bg, 1 Liver, 2 Gallbladder, 3 Tool
TISSUE_IDS = (1, 2)
MIN_PX = 400              # need at least this many tool & tissue px in a frame to use it


def log(m):
    print(m, flush=True)


def _natkey(p):
    b = os.path.splitext(os.path.basename(p))[0]
    d = ''.join(ch for ch in b if ch.isdigit())
    return int(d) if d else b


def _ls(scene, sub, exts):
    for e in exts:
        f = glob.glob(os.path.join(scene, sub, f'*.{e}'))
        if f:
            return sorted(f, key=_natkey)
    return []


def _seg(path, H, W):
    s = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if s is None:
        return None
    if s.ndim == 3:
        s = s[..., 0]
    if s.shape != (H, W):
        s = cv2.resize(s, (W, H), interpolation=cv2.INTER_NEAREST)
    return s


def _masks(seg):
    tool = np.isin(seg, TOOL_IDS)
    tissue = np.isin(seg, TISSUE_IDS)
    return tool, tissue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='/content/EndoGSLAM/data/CRCD/C1_001',
                    help='assembled CRCD scene: frames/ depths/ semantic_ids/')
    ap.add_argument('--out', default='/content/EndoGSLAM/experiments/attribution/C1_001')
    ap.add_argument('--drive', default='/content/drive/MyDrive/Outputs/GS_attribution')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--small', action='store_true', help='raft_small (faster, less accurate)')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--shift', type=int, default=30, help='NULL-B spatial-misalign offset (frames)')
    ap.add_argument('--n_viz', type=int, default=6)
    a = ap.parse_args()

    import torch
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    figs = os.path.join(a.out, 'figs')
    os.makedirs(figs, exist_ok=True)

    # auto-detect layout: EndoGSLAM (frames/ + semantic_ids/) OR rect-bench (video_frames/*l.png + semantic_class/)
    frames = _ls(a.scene, 'frames', ('jpg', 'png')) \
        or sorted(glob.glob(os.path.join(a.scene, 'video_frames', '*l.png')), key=_natkey)
    segs = _ls(a.scene, 'semantic_ids', ('png',)) or _ls(a.scene, 'semantic_class', ('png',))
    depths = _ls(a.scene, 'depths', ('png',)) or _ls(a.scene, 'depth', ('png',))
    layout = 'EndoGSLAM' if _ls(a.scene, 'frames', ('jpg', 'png')) else 'rect-bench'
    n = len(frames)
    assert n > 1 and len(segs) >= n, (
        f"need frames+class-seg in {a.scene} (got {n} frames, {len(segs)} segs). "
        f"Expected frames/+semantic_ids/ (EndoGSLAM) or video_frames/*l.png+semantic_class/ (rect-bench).")
    if a.max_frames:
        n = min(n, a.max_frames)
    H, W = cv2.imread(frames[0]).shape[:2]

    log("=" * 64)
    log(f">>> [attribution] scene={a.scene}")
    log(f">>> [attribution] {n} frames @ {W}x{H} · layout={layout} · stride={a.stride} · RAFT={'small' if a.small else 'large'} · dev={dev}")
    log("=" * 64)
    log(">>> [attribution] loading RAFT ...")
    model, tf = ft.load_raft(dev, small=a.small)

    # accumulators
    rt, rti, rbg = [], [], []          # per-frame median residual on tool / tissue / (real)
    ratio_real, ratio_null = [], []    # per-frame med_tool/med_tissue  (real & spatially-misaligned)
    auc_tool_vals, auc_labels = [], []  # pooled pixel residuals + labels (1 tool, 0 tissue) for AUC
    n_used = n_fail = n_notool = 0
    seg_buf = deque(maxlen=a.shift + 1)  # ring buffer of (tool,tissue) masks for NULL-B
    viz_pool = []                        # (tool_count, ref_idx) -> pick top n_viz

    log(f">>> [attribution] scanning { (n-a.stride)//1 } pairs ...")
    t0 = time.time()
    for i in range(a.stride, n):
        ref, cur = i - a.stride, i
        R = ft.flow_residual(cv2.imread(frames[ref]), cv2.imread(frames[cur]), model, tf, dev)
        seg = _seg(segs[ref], H, W)        # residual is in the REF pixel grid -> align with ref seg
        if seg is None:
            continue
        tool, tissue = _masks(seg)
        seg_buf.append((tool, tissue))

        if R.max() <= 0:                   # F-fit failed -> residual all-zero (degenerate forward/zoom)
            n_fail += 1
        if tool.sum() < MIN_PX or tissue.sum() < MIN_PX:
            n_notool += 1
        else:
            n_used += 1
            mt, mi, mb = np.median(R[tool]), np.median(R[tissue]), np.median(R[seg == 0]) if (seg == 0).any() else 0.0
            rt.append(mt); rti.append(mi); rbg.append(float(mb))
            ratio_real.append(mt / (mi + 1e-6))
            # NULL-B: this frame's residual vs the tool mask from `shift` frames ago
            if len(seg_buf) == seg_buf.maxlen:
                t_old, i_old = seg_buf[0]
                if t_old.sum() >= MIN_PX and i_old.sum() >= MIN_PX:
                    ratio_null.append(np.median(R[t_old]) / (np.median(R[i_old]) + 1e-6))
            # pooled AUC sample (cap per frame so it stays bounded)
            k = 1500
            ti = np.flatnonzero(tool.ravel()); si = np.flatnonzero(tissue.ravel())
            ti = ti[np.linspace(0, len(ti) - 1, min(k, len(ti))).astype(int)]
            si = si[np.linspace(0, len(si) - 1, min(k, len(si))).astype(int)]
            Rr = R.ravel()
            auc_tool_vals.append(Rr[ti]); auc_labels.append(np.ones(len(ti)))
            auc_tool_vals.append(Rr[si]); auc_labels.append(np.zeros(len(si)))
            viz_pool.append((int(tool.sum()), ref))

        if (i % 30 == 0) or (i == n - 1):
            el = time.time() - t0
            eta = el / max(i - a.stride + 1, 1) * (n - 1 - i)
            log(f"    pair {i}/{n-1} · used {n_used} · no-tool {n_notool} · F-fail {n_fail} · {el:.0f}s · eta {eta:.0f}s")

    # ---- stats ----
    fitfail_rate = n_fail / max(n_used + n_notool + n_fail, 1)
    if n_used < 5:
        # Not enough tool-visible frames to use the tool as the independent-motion proxy.
        rr = rn = frac = auc_real = auc_null = float('nan')
        verdict = "UNTESTABLE"
        log(f">>> [attribution] only {n_used} frames had both tool & tissue (>= {MIN_PX}px) -> tool-proxy test UNTESTABLE.")
        log(">>> [attribution] will still ship a residual panel for visual inspection.")
        rt = rt or [float('nan')]; rti = rti or [float('nan')]; rbg = rbg or [float('nan')]
    else:
        from sklearn.metrics import roc_auc_score
        vals = np.concatenate(auc_tool_vals); labs = np.concatenate(auc_labels)
        auc_real = roc_auc_score(labs, vals)
        rng = np.random.default_rng(0)
        auc_null = roc_auc_score(rng.permutation(labs), vals)   # NULL-A: shuffled labels -> ~0.5
        rr = float(np.median(ratio_real)) if ratio_real else float('nan')
        rn = float(np.median(ratio_null)) if ratio_null else float('nan')
        frac = float(np.mean(np.array(rt) > np.array(rti)))
        # ---- verdict ----
        go = (rr > 1.30) and (auc_real > 0.60) and (rr > 1.5 * (rn if rn == rn else 1.0)) and (auc_real - auc_null > 0.08)
        weak = (not go) and (rr > 1.12) and (auc_real > 0.55)
        verdict = "GO" if go else ("WEAK" if weak else "NO-GO")

    summary = [
        "=" * 64,
        ">>> [attribution] RESULT",
        "=" * 64,
        f"  frames used (tool present)   : {n_used}   (no-tool {n_notool}, F-fail {n_fail})",
        f"  F-fit failure rate           : {fitfail_rate:6.1%}   (high => forward/zoom degeneracy; sensor unreliable there)",
        f"  median residual  TOOL        : {np.median(rt):7.3f}",
        f"  median residual  TISSUE      : {np.median(rti):7.3f}",
        f"  median residual  BG          : {np.median(rbg):7.3f}",
        f"  ratio  tool/tissue  (REAL)   : {rr:6.2f}   <- effect size (>1 = residual sits on the tool)",
        f"  ratio  tool/tissue  (NULL-B) : {rn:6.2f}   <- spatial-misalign null (should collapse to ~1)",
        f"  frac frames  tool>tissue     : {frac:6.1%}",
        f"  AUC  R: tool-vs-tissue (REAL): {auc_real:6.3f}   <- can the residual classify the tool",
        f"  AUC  R: tool-vs-tissue (NULL): {auc_null:6.3f}   <- shuffled labels (~0.5)",
        "-" * 64,
        f"  VERDICT: {verdict}",
        {"GO":   "  -> residual localizes independent motion; build v1 on it.",
         "WEAK": "  -> WEAK signal; eyeball the panel before committing v1.",
         "NO-GO": "  -> residual does NOT track real motion; v1 mis-aimed (fix depth-scale/LR, not a gate).",
         "UNTESTABLE": "  -> too few tool frames; judge from the residual panel + rerun on a tool-rich snippet."}.get(verdict, ""),
        "=" * 64,
    ]
    for s in summary:
        log(s)
    with open(os.path.join(a.out, 'results.txt'), 'w') as f:
        f.write("\n".join(summary) + "\n")

    # ---- visual panel: top-n_viz frames by tool area ----
    log(">>> [attribution] building visual panel ...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pick = [r for _, r in sorted(viz_pool, reverse=True)[:a.n_viz]]
    rows = len(pick)
    if rows:
        fig, ax = plt.subplots(rows, 4, figsize=(16, 3.4 * rows))
        ax = np.atleast_2d(ax)
        for r, ref in enumerate(pick):
            cur = ref + a.stride
            R = ft.flow_residual(cv2.imread(frames[ref]), cv2.imread(frames[cur]), model, tf, dev)
            rgb = cv2.cvtColor(cv2.imread(frames[ref]), cv2.COLOR_BGR2RGB)
            seg = _seg(segs[ref], H, W)
            tool, tissue = _masks(seg)
            vmax = np.percentile(R[R > 0], 95) if (R > 0).any() else 1.0
            ax[r, 0].imshow(rgb); ax[r, 0].set_ylabel(f'frame {ref}', fontsize=9)
            ax[r, 1].imshow(np.clip(R / (vmax + 1e-6), 0, 1), cmap='turbo')
            ov = rgb.copy(); ov[tool] = (ov[tool] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
            ov[tissue] = (ov[tissue] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)
            ax[r, 2].imshow(ov)
            Rt = np.where(tissue, R, 0.0)      # residual ON TISSUE ONLY = candidate deformation
            ax[r, 3].imshow(np.clip(Rt / (vmax + 1e-6), 0, 1), cmap='turbo')
            for c in range(4):
                ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
        for c, t in enumerate(['input RGB', 'flow residual', 'seg (tool=R, tissue=G)', 'residual on TISSUE (cand. deform)']):
            ax[0, c].set_title(t, fontsize=11)
        fig.suptitle(f'attribution  c1_001  ·  tool/tissue ratio REAL {rr:.2f} vs NULL {rn:.2f}  ·  AUC {auc_real:.3f}  ·  {verdict}', fontsize=12)
        fig.tight_layout()
        pp = os.path.join(figs, 'attribution_panel.png')
        fig.savefig(pp, dpi=90, bbox_inches='tight'); plt.close(fig)
        log(f"    panel -> {pp}")

    # ---- ship (two-diagnostic-sets) ----
    if os.path.isdir('/content/drive/MyDrive'):
        dst = os.path.join(a.drive, os.path.basename(a.scene.rstrip('/')))
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(a.out, 'results.txt'), dst)
        for p in glob.glob(os.path.join(figs, '*.png')):
            shutil.copy2(p, dst)
        log(f">>> [attribution] shipped (results + panel) -> {dst}")
    else:
        log(">>> [attribution] /content/drive/MyDrive not mounted -> NOT shipped")
    log(f">>> [attribution] DONE · verdict {verdict}")


if __name__ == '__main__':
    main()
