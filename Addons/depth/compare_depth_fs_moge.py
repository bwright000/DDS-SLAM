#!/usr/bin/env python3
"""Inspect FoundationStereo (FS) depth against the current MoGe-2+stereo120 depth for one rectified CRCD snippet.

No GT depth exists for CRCD, so this reports three GT-free views:
  1. AGREEMENT   FS vs MoGe on common-valid pixels: scale ratio median(FS/MoGe), AbsRel + delta<1.25 both RAW (metric
                 agreement) and after per-frame median scaling (shape agreement).
  2. STEREO PHOTOMETRIC CONSISTENCY  warp the rectified RIGHT image into the left view with disparity = fx*B/Z of each
                 depth; mean |L - R_warp| (grey, 0-255) on the common-valid mask. Lower = the depth better explains the
                 actual stereo pair. A constant-depth plane (FS median) is scored too as an uninformative floor.
                 (FS is a stereo matcher, so this favours it by construction - read it as "physically consistent", and
                 read MoGe's number as how far its metric anchoring leaves it from the stereo geometry.)
  3. TEMPORAL    frame-to-frame relative change of the median depth (jitter) per method.
Optional per-class breakdown from <staged>/semantic_class/NNNNNN.png.

Outputs to --out: summary.json, per_frame.csv, montage.png, timeseries.png, hist.png, compare.mp4
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np


def read_fxb(path):
    kv = {}
    for line in open(path):
        m = re.match(r"\s*([A-Za-z_]+)\s*[: ]\s*([-+0-9.eE]+)", line)
        if m:
            kv[m.group(1)] = float(m.group(2))
    return kv["fx"] * kv.get("baseline_m", kv.get("baseline"))


def load_depth(path, scale):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        sys.exit(f"FATAL: cannot read {path}")
    return a.astype(np.float64) / scale


def photometric(Lg, Rg, D, fxb, mask):
    """mean |L - R(x - fxb/D)| on mask (and where the warp lands inside R)."""
    h, w = Lg.shape
    dsp = np.where(D > 0, fxb / np.maximum(D, 1e-9), 0).astype(np.float32)
    mx = (np.arange(w, dtype=np.float32)[None, :] - dsp)
    my = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
    Rw = cv2.remap(Rg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    m = mask & (mx >= 0) & (Rw >= 0)
    if not m.any():
        return np.nan, None
    err = np.abs(Lg - Rw)
    return float(err[m].mean()), np.where(m, err, np.nan)


def agree(fs, mo, m):
    a, b = fs[m], mo[m]
    if a.size < 100:
        return dict(ratio=np.nan, absrel_raw=np.nan, d1_raw=np.nan, absrel_aligned=np.nan, d1_aligned=np.nan)
    r = float(np.median(a / b))
    bs = b * r
    q_raw, q_al = np.maximum(a / b, b / a), np.maximum(a / bs, bs / a)
    return dict(ratio=r, absrel_raw=float(np.mean(np.abs(a - b) / a)), d1_raw=float(np.mean(q_raw < 1.25)),
                absrel_aligned=float(np.mean(np.abs(a - bs) / a)), d1_aligned=float(np.mean(q_al < 1.25)))


def colorize(D, lo, hi):
    x = np.clip((D - lo) / max(hi - lo, 1e-9), 0, 1)
    c = cv2.applyColorMap((255 * (1 - x)).astype(np.uint8), cv2.COLORMAP_TURBO)  # near = red/warm
    c[D <= 0] = 0
    return c


def logratio_img(fs, mo, m):
    lr = np.zeros_like(fs)
    lr[m] = np.log2(fs[m] / mo[m])
    x = np.clip((lr + 1) / 2, 0, 1)  # +-1 stop = 2x
    c = cv2.applyColorMap((255 * x).astype(np.uint8), cv2.COLORMAP_COOL)
    c[~m] = 0
    return c


def label(img, txt):
    cv2.putText(img, txt, (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True, help="rectified snippet dir (video_frames/, rectified_calib.txt)")
    ap.add_argument("--fs_dir", required=True)
    ap.add_argument("--moge_dir", required=True, help="current MoGe-2 + stereo120 metric depth/*.png")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth_scale", type=float, default=10000.0)
    ap.add_argument("--moge_scale", type=float, default=10000.0)
    ap.add_argument("--video_stride", type=int, default=2)
    ap.add_argument("--montage_n", type=int, default=6)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(os.path.normpath(args.staged))
    fxb = read_fxb(os.path.join(args.staged, "rectified_calib.txt"))
    lefts = sorted(glob.glob(os.path.join(args.staged, "video_frames", "*l.png")))
    stems = [os.path.basename(p)[:-5] for p in lefts]
    fs_ok = [s for s in stems if os.path.exists(os.path.join(args.fs_dir, s + ".png"))]
    mo_ok = [s for s in stems if os.path.exists(os.path.join(args.moge_dir, s + ".png"))]
    if len(fs_ok) != len(stems) or len(mo_ok) != len(stems):
        sys.exit(f"FATAL {name}: frames {len(stems)}  fs {len(fs_ok)}  moge {len(mo_ok)} - counts must match")
    sem_dir = os.path.join(args.staged, "semantic_class")
    has_sem = os.path.isdir(sem_dir)

    rows, cls_acc, hist_fs, hist_mo = [], {}, [], []
    montage_idx = set(np.linspace(0, len(stems) - 1, args.montage_n).round().astype(int).tolist())
    montage_rows, vw = [], None
    rng = np.random.default_rng(0)
    prev = {"fs": None, "mo": None}
    for i, s in enumerate(stems):
        L = cv2.imread(os.path.join(args.staged, "video_frames", s + "l.png"))
        R = cv2.imread(os.path.join(args.staged, "video_frames", s + "r.png"))
        fs = load_depth(os.path.join(args.fs_dir, s + ".png"), args.depth_scale)
        mo = load_depth(os.path.join(args.moge_dir, s + ".png"), args.moge_scale)
        if fs.shape != L.shape[:2] or mo.shape != L.shape[:2]:
            sys.exit(f"FATAL {name}/{s}: shapes rgb {L.shape[:2]} fs {fs.shape} moge {mo.shape}")
        vf, vm = fs > 0, mo > 0
        m = vf & vm
        row = dict(frame=s, valid_fs=float(vf.mean()), valid_moge=float(vm.mean()), valid_common=float(m.mean()),
                   med_fs=float(np.median(fs[vf])) if vf.any() else np.nan,
                   med_moge=float(np.median(mo[vm])) if vm.any() else np.nan)
        row.update(agree(fs, mo, m))
        for k, key in (("fs", "med_fs"), ("mo", "med_moge")):
            row[f"jitter_{k}"] = abs(row[key] - prev[k]) / row[key] if prev[k] else np.nan
            prev[k] = row[key]
        pe = {}
        if R is not None and m.any():
            Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY).astype(np.float32)
            Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY).astype(np.float32)
            row["photo_fs"], pe["fs"] = photometric(Lg, Rg, fs, fxb, m)
            row["photo_moge"], pe["mo"] = photometric(Lg, Rg, mo, fxb, m)
            row["photo_plane"], _ = photometric(Lg, Rg, np.where(m, row["med_fs"], 0.0), fxb, m)
        if has_sem:
            sp = os.path.join(sem_dir, s + ".png")
            if os.path.exists(sp):
                sem = cv2.imread(sp, cv2.IMREAD_UNCHANGED)
                for c in np.unique(sem):
                    mc = m & (sem == c)
                    if mc.sum() < 100:
                        continue
                    a = cls_acc.setdefault(int(c), dict(px=0, ratio=[], absrel_aligned=[], valid_fs=[], valid_moge=[]))
                    g = agree(fs, mo, mc)
                    a["px"] += int(mc.sum())
                    a["ratio"].append(g["ratio"])
                    a["absrel_aligned"].append(g["absrel_aligned"])
                    a["valid_fs"].append(float(vf[sem == c].mean()))
                    a["valid_moge"].append(float(vm[sem == c].mean()))
        rows.append(row)
        if vf.any():
            hist_fs.append(rng.choice(fs[vf], min(3000, int(vf.sum())), replace=False))
        if vm.any():
            hist_mo.append(rng.choice(mo[vm], min(3000, int(vm.sum())), replace=False))

        lo, hi = np.percentile(np.concatenate([fs[vf], mo[vm]]) if (vf.any() or vm.any()) else [0, 1], [2, 98])
        panels = [label(L.copy(), f"{name} {s}"), label(colorize(fs, lo, hi), "FoundationStereo"),
                  label(colorize(mo, lo, hi), "MoGe-2+stereo120"), label(logratio_img(fs, mo, m), "log2 FS/MoGe (+-1)")]
        if i % args.video_stride == 0:
            frame = np.concatenate([cv2.resize(p, (640, 360)) for p in panels], 1)
            if vw is None:
                vw = cv2.VideoWriter(os.path.join(args.out, "compare.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                     (frame.shape[1], frame.shape[0]))
            vw.write(frame)
        if i in montage_idx:
            pv = []
            for k, t in (("fs", "photo err FS"), ("mo", "photo err MoGe")):
                e = pe.get(k)
                img = np.zeros_like(L) if e is None else cv2.applyColorMap(
                    (255 * np.clip(np.nan_to_num(e, nan=0) / 40, 0, 1)).astype(np.uint8), cv2.COLORMAP_INFERNO)
                pv.append(label(img, t + " (0-40)"))
            montage_rows.append(np.concatenate([cv2.resize(p, (480, 270)) for p in panels + pv], 1))
        if i % 100 == 0:
            print(f"[cmp] {name} {i + 1}/{len(stems)}  ratio {row['ratio']:.3f}  photo fs/moge/plane "
                  f"{row.get('photo_fs', np.nan):.2f}/{row.get('photo_moge', np.nan):.2f}/{row.get('photo_plane', np.nan):.2f}",
                  flush=True)
    if vw is not None:
        vw.release()
    cv2.imwrite(os.path.join(args.out, "montage.png"), np.concatenate(montage_rows, 0))

    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(args.out, "per_frame.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    col = lambda k: np.array([r.get(k, np.nan) for r in rows], dtype=float)
    nm = lambda a: float(np.nanmean(a))
    hf, hm = np.concatenate(hist_fs), np.concatenate(hist_mo)
    pc = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (1, 5, 50, 95, 99)}
    ratio = col("ratio")
    summary = {
        "snippet": name, "frames": len(rows), "fx_times_B": fxb,
        "valid_pct": {"fs": 100 * nm(col("valid_fs")), "moge": 100 * nm(col("valid_moge")),
                      "common": 100 * nm(col("valid_common"))},
        "depth_m_percentiles": {"fs": pc(hf), "moge": pc(hm)},
        "scale_ratio_fs_over_moge": {"median": float(np.nanmedian(ratio)), "p5": float(np.nanpercentile(ratio, 5)),
                                     "p95": float(np.nanpercentile(ratio, 95)),
                                     "cv": float(np.nanstd(ratio) / np.nanmean(ratio))},
        "agreement_raw": {"absrel": nm(col("absrel_raw")), "delta1": nm(col("d1_raw"))},
        "agreement_scale_aligned": {"absrel": nm(col("absrel_aligned")), "delta1": nm(col("d1_aligned"))},
        "stereo_photometric_L1": {"fs": nm(col("photo_fs")), "moge": nm(col("photo_moge")),
                                  "plane_floor": nm(col("photo_plane"))},
        "temporal_median_jitter": {"fs": nm(col("jitter_fs")), "moge": nm(col("jitter_mo"))},
        "unit_gate": {"fs_p1_m": float(np.percentile(hf, 1)),
                      "sgs_semgauss_k10_clears_0p2_cull": bool(np.percentile(hf, 1) * 10 > 0.2)},
        "per_class": {str(c): {"px": a["px"], "ratio_median": float(np.nanmedian(a["ratio"])),
                               "absrel_aligned": float(np.nanmean(a["absrel_aligned"])),
                               "valid_fs_pct": 100 * float(np.mean(a["valid_fs"])),
                               "valid_moge_pct": 100 * float(np.mean(a["valid_moge"]))}
                      for c, a in sorted(cls_acc.items())},
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)

    t = np.arange(len(rows))
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(t, col("med_fs"), label="FS"); ax[0].plot(t, col("med_moge"), label="MoGe+stereo120")
    ax[0].set_ylabel("median depth (m)"); ax[0].legend()
    ax[1].plot(t, ratio, c="k"); ax[1].axhline(1, ls="--", c="grey"); ax[1].set_ylabel("median FS/MoGe")
    for x in range(0, len(rows), 120):
        ax[1].axvline(x, c="orange", lw=0.5)  # stereo120 anchor frames
    ax[2].plot(t, col("photo_fs"), label="FS"); ax[2].plot(t, col("photo_moge"), label="MoGe")
    ax[2].plot(t, col("photo_plane"), label="plane floor", c="grey", lw=0.8)
    ax[2].set_ylabel("stereo photo L1"); ax[2].legend()
    ax[3].plot(t, 100 * col("valid_fs"), label="FS"); ax[3].plot(t, 100 * col("valid_moge"), label="MoGe")
    ax[3].set_ylabel("valid %"); ax[3].set_xlabel("frame"); ax[3].legend()
    fig.suptitle(f"{name}: FoundationStereo vs MoGe-2+stereo120 (orange = anchor frames)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "timeseries.png"), dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hi = np.percentile(np.concatenate([hf, hm]), 99.5)
    ax.hist(hf, bins=150, range=(0, hi), alpha=0.6, label="FS", density=True)
    ax.hist(hm, bins=150, range=(0, hi), alpha=0.6, label="MoGe+stereo120", density=True)
    ax.axvline(0.02, c="r", ls="--", lw=0.8, label="0.02 m (x10 = 0.2 cull)")
    ax.set_xlabel("depth (m)"); ax.legend(); ax.set_title(name)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "hist.png"), dpi=110); plt.close(fig)

    s = summary
    print(f"[cmp] DONE {name}: ratio FS/MoGe {s['scale_ratio_fs_over_moge']['median']:.3f} "
          f"(cv {s['scale_ratio_fs_over_moge']['cv']:.3f})  aligned AbsRel {s['agreement_scale_aligned']['absrel']:.3f}  "
          f"photo fs/moge/plane {s['stereo_photometric_L1']['fs']:.2f}/{s['stereo_photometric_L1']['moge']:.2f}/"
          f"{s['stereo_photometric_L1']['plane_floor']:.2f}  valid fs/moge {s['valid_pct']['fs']:.1f}/{s['valid_pct']['moge']:.1f}%",
          flush=True)


if __name__ == "__main__":
    main()
