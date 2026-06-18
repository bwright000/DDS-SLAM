#!/usr/bin/env python3
"""Audit CRCD raw-left metric depth on Drive: do the depth-frame counts match the GT for every snippet?

For each snippet under <published_root>/<EP>/snippet_<SID>/ it takes the CANONICAL frame count =
groundtruth.txt rows (the same 360-row CRCD-Published GT the SLAM evals against), then compares the
RAW MoGe intermediate and the final METRIC depth under <depth_root>/<EP>/snippet_<SID>/. The raw-vs-
metric split localises failures: no raw  => MoGe step failed; raw but no/short metric => stereo120
scaling failed; short both => partial run. Prints a table + a copy-pasteable TO-FIX list.

  python Addons/depth/audit_crcd_depth.py \
    --published_root /content/drive/MyDrive/Datasets/CRCD-Published \
    --depth_root     /content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2
"""
import os, glob, argparse


def count_lines(p):
    if not os.path.isfile(p): return -1
    n = 0
    for ln in open(p, errors='ignore'):
        s = ln.strip()
        if s and not s.startswith('#'): n += 1
    return n


def count_glob(d, g):
    return len(glob.glob(os.path.join(d, g))) if os.path.isdir(d) else -1


def find_rgb(sn):                                          # best-effort source-frame count for cross-check
    for sub, g in [('rgb', '*.png'), ('rgb', '*.jpg'), ('video_frames', '*l.png'),
                   ('video_frames', '*.png'), ('left', '*.png'), ('', '*left.png'), ('', '*l.png')]:
        n = count_glob(os.path.join(sn, sub) if sub else sn, g)
        if n > 0: return n, (sub + '/' + g if sub else g)
    return -1, '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published_root', default='/content/drive/MyDrive/Datasets/CRCD-Published')
    ap.add_argument('--depth_root', default='/content/drive/MyDrive/Datasets/CRCD-Published-MoGe-2')
    ap.add_argument('--gt_name', default='groundtruth.txt')
    ap.add_argument('--raw_subdir', default='depth_rawleft')
    ap.add_argument('--metric_subdir', default='depth_rawleft_metric')
    ap.add_argument('--depth_glob', default='*.png')
    args = ap.parse_args()

    assert os.path.isdir(args.published_root), f'no published_root {args.published_root}'
    eps = sorted(d for d in os.listdir(args.published_root) if os.path.isdir(os.path.join(args.published_root, d)))
    rows, tofix = [], []
    print(f"published: {args.published_root}\ndepth:     {args.depth_root}\n")
    hdr = f"{'EP/snippet':22s} {'GT':>5s} {'rgb':>5s} {'rawMoGe':>8s} {'metric':>7s}  status"
    print(hdr); print('-' * len(hdr))
    for ep in eps:
        for sn in sorted(glob.glob(os.path.join(args.published_root, ep, 'snippet_*'))):
            sid = os.path.basename(sn)
            gt = count_lines(os.path.join(sn, args.gt_name))
            rgb, _ = find_rgb(sn)
            ddir = os.path.join(args.depth_root, ep, sid)
            raw = count_glob(os.path.join(ddir, args.raw_subdir), args.depth_glob)
            met = count_glob(os.path.join(ddir, args.metric_subdir), args.depth_glob)
            exp = gt if gt > 0 else rgb                         # GT is the anchor; rgb fallback
            if met < 0 and raw < 0:
                st = 'MISSING (no depth dir / MoGe never ran)'
            elif met < 0:
                st = f'NO_METRIC (raw={raw}) -> stereo120 failed' if raw > 0 else 'NO_METRIC & NO_RAW -> MoGe failed'
            elif exp > 0 and met == exp:
                st = 'OK'
            elif exp > 0 and met < exp:
                st = f'PARTIAL {met}/{exp}' + ('' if raw == exp or raw < 0 else f' (raw {raw})')
            elif exp > 0 and met > exp:
                st = f'EXTRA {met}/{exp} (?!)'
            else:
                st = f'metric={met} (no GT to check)'
            rows.append((ep, sid, gt, rgb, raw, met, st))
            print(f"{ep+'/'+sid:22s} {gt:>5d} {rgb:>5d} {raw:>8d} {met:>7d}  {st}")
            if st != 'OK' and not st.startswith('metric='):
                tofix.append(f"{ep}/{sid}")

    ok = sum(1 for r in rows if r[6] == 'OK')
    print(f"\n=== {ok}/{len(rows)} snippets OK ===")
    if tofix:
        print(f"TO FIX ({len(tofix)}): " + ' '.join(tofix))
        print("  NO_RAW/MoGe-failed  -> re-run MoGe for these (diagnose if it re-fails: corrupt frame / OOM).")
        print("  NO_METRIC/raw-ok    -> re-run stereo120 scaling only (raw MoGe is already there).")
        print("  PARTIAL             -> re-run the snippet end-to-end (.DONE gate should have caught it).")
    else:
        print("ALL SNIPPETS COMPLETE.")


if __name__ == '__main__':
    main()
