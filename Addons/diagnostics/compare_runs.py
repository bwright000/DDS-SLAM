"""
Per-frame comparison of multiple DDS-SLAM debug_log.csv runs.

For each metric, plots the three runs on the same time axis so we can see
where the correct-depth fix helps vs hurts frame-by-frame. Aggregate stats
in a side-by-side table for immediate decision-making.

Usage:
  python Addons/diagnostics/compare_runs.py \\
      --runs "label1=<dir1>" "label2=<dir2>" "label3=<dir3>" \\
      --output compare.html
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


METRICS = [
    # (column, label, unit, scale_factor, log_y)
    ('trans_err_aligned_m', 'Aligned ATE per frame', 'mm', 1000.0, False),
    ('rpe_trans_m',         'RPE trans per frame',   'mm', 1000.0, False),
    ('rpe_rot_rad',         'RPE rot per frame',     'deg', 180.0/np.pi, False),
    ('trans_delta_m',       'est |Δt| per frame',    'mm', 1000.0, False),
    ('loss_rgb',            'L_rgb (raw)',           '',   1.0,    True),
    ('loss_depth',          'L_depth (raw)',         '',   1.0,    True),
    ('loss_sdf',            'L_tr (raw)',            '',   1.0,    True),
    ('loss_fs',             'L_fs (raw)',            '',   1.0,    True),
    ('loss_edge_semantic',  'L_m (raw)',             '',   1.0,    True),
    ('n_fs_samples',        'n_fs_samples',          '',   1.0,    False),
    ('n_sdf_samples',       'n_sdf_samples',         '',   1.0,    False),
    ('depth_mean',          'depth_mean (input)',    'mm', 1000.0, False),
    ('psnr',                'PSNR',                  'dB', 1.0,    False),
    ('init_trans_err_m',    'init trans err (raw)',  'mm', 1000.0, False),
    ('best_loss',           'best_loss',             '',   1.0,    True),
]

COLORS = ['#f85149', '#58a6ff', '#56d364', '#f0883e', '#d2a8ff']


def load(debug_dir, label):
    p = os.path.join(debug_dir, 'debug_log.csv')
    df = pd.read_csv(p)
    df['_label'] = label
    return df


def summary_row(label, df):
    def rmse(x):
        x = pd.Series(x).dropna()
        return float(np.sqrt((x ** 2).mean())) if len(x) else float('nan')

    s = {
        'run': label,
        'N': len(df),
        'ATE(R,t) mm':       rmse(df.get('trans_err_aligned_m', pd.Series([]))) * 1000,
        'ATE(R,t,s) mm':     rmse(df.get('trans_err_sim_m', pd.Series([]))) * 1000,
        'RPE trans mm/f':    rmse(df.get('rpe_trans_m', pd.Series([]))) * 1000,
        'RPE rot deg/f':     np.degrees(rmse(df.get('rpe_rot_rad', pd.Series([])))),
        'depth_mean mm':     df['depth_mean'].median() * 1000 if 'depth_mean' in df else float('nan'),
        'n_fs_mean':         df['n_fs_samples'].mean() if 'n_fs_samples' in df else float('nan'),
        'n_sdf_mean':        df['n_sdf_samples'].mean() if 'n_sdf_samples' in df else float('nan'),
        'frames n_fs=0':     int((df['n_fs_samples'] == 0).sum()) if 'n_fs_samples' in df else -1,
        'loss_rgb share %':  (5.0 * df['loss_rgb'] / df['best_loss']).median() * 100,
        'loss_sdf share %':  (1000.0 * df['loss_sdf'] / df['best_loss']).median() * 100,
        'PSNR median':       df['psnr'].median() if 'psnr' in df else float('nan'),
    }
    return s


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8"><title>Run comparison</title>
<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<style>
 body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;margin:0;padding:20px;}
 h1{color:#c9d1d9;}
 table{border-collapse:collapse;margin:10px 0 30px 0;}
 th,td{border:1px solid #30363d;padding:6px 10px;font-size:13px;}
 th{background:#161b22;text-align:left;color:#ffa657;}
 .chart{margin:20px 0;}
 code{background:#161b22;padding:2px 5px;}
</style></head><body>
<h1>Run comparison — per-frame metrics</h1>
"""


def build_html(runs, out_path):
    # Summary table
    rows = [summary_row(label, df) for label, df in runs]
    tbl = '<table><tr>' + ''.join(f'<th>{k}</th>' for k in rows[0].keys()) + '</tr>'
    for r in rows:
        tbl += '<tr>' + ''.join(
            f'<td>{v:.3f}</td>' if isinstance(v, float) else f'<td>{v}</td>'
            for v in r.values()) + '</tr>'
    tbl += '</table>'

    html = HTML_HEAD + tbl

    for idx, (col, label, unit, scale, log_y) in enumerate(METRICS):
        div_id = f'chart_{idx}'
        html += f'<div class="chart"><div id="{div_id}" style="width:100%;height:340px;"></div></div>'

    # Build plotly traces
    script = ['<script>']
    DARK = {
        'plot_bgcolor': '#0d1117', 'paper_bgcolor': '#0d1117',
        'font': {'color': '#c9d1d9'},
        'xaxis': {'gridcolor': '#21262d', 'color': '#c9d1d9'},
        'yaxis': {'gridcolor': '#21262d', 'color': '#c9d1d9'},
        'legend': {'bgcolor': '#161b22'},
    }

    for idx, (col, label, unit, scale, log_y) in enumerate(METRICS):
        traces = []
        for (run_label, df), color in zip(runs, COLORS):
            if col not in df.columns:
                continue
            y = (df[col] * scale).tolist()
            traces.append({
                'x': df['frame_id'].tolist(),
                'y': y,
                'type': 'scatter', 'mode': 'lines', 'name': run_label,
                'line': {'color': color, 'width': 1},
            })
        layout = dict(DARK)
        layout['title'] = f'{label}' + (f' ({unit})' if unit else '')
        layout['xaxis'] = {**layout['xaxis'], 'title': 'frame'}
        layout['yaxis'] = {**layout['yaxis'], 'title': unit}
        if log_y:
            layout['yaxis']['type'] = 'log'

        script.append(
            f"Plotly.newPlot('chart_{idx}', "
            f"{json.dumps(traces)}, "
            f"{json.dumps(layout)});"
        )

    script.append('</script>')
    html += '\n'.join(script) + '</body></html>'

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f'wrote {out_path} ({size_mb:.2f} MB)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='list of label=/path/to/debug or label=/path/to/debug_log.csv')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    runs = []
    for spec in args.runs:
        if '=' not in spec:
            print(f'skip {spec} (missing label=)'); continue
        label, path = spec.split('=', 1)
        if path.endswith('.csv'):
            path = os.path.dirname(path)
        if not os.path.isdir(path):
            print(f'skip {label} ({path} is not a dir)'); continue
        df = load(path, label)
        print(f'loaded {label}: {len(df)} rows')
        runs.append((label, df))

    # Print summary table
    rows = [summary_row(label, df) for label, df in runs]
    df_summary = pd.DataFrame(rows)
    print()
    print(df_summary.to_string(index=False))
    print()

    build_html(runs, args.output)


if __name__ == '__main__':
    main()
