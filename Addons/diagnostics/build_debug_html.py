#!/usr/bin/env python
"""
build_debug_html.py - Generate a self-contained HTML dashboard from a DDS-SLAM
debug run (debug_log.csv + pose_snapshots/).

Usage:
    python Addons/diagnostics/build_debug_html.py \
        --debug_dir output/StereoMIS/P2_1/demo/debug \
        --output debug_dashboard.html

Open the output HTML in any modern browser (no server needed). Plotly CDN loads
once; data is embedded inline. Tabs cover:
  * Overview (key stats)
  * Trajectory (3D + xy projection)
  * Per-axis position + errors over time
  * Per-frame motion delta + distribution
  * Loss components + PSNR
  * Tracker convergence
  * Input signals
  * Keyframe pose evolution across snapshots
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def umeyama(src, dst, with_scale=False):
    """Umeyama (1991) least-squares similarity/rigid alignment.

    src, dst: (N, 3). Returns (R, t, s) such that  s*R @ src.T + t ≈ dst.T.
    When with_scale=False, s is fixed to 1 (pure rigid SE(3) alignment).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / len(src)
        s = (S * np.diag(D)).sum() / var_s if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def apply_sim3(src, R, t, s):
    return (s * (R @ np.asarray(src).T)).T + t


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DDS-SLAM Debug Dashboard - __SOURCE__</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 0; background: #0d1117; color: #e6edf3; }
  header { padding: 16px 24px; background: #161b22; border-bottom: 1px solid #30363d; }
  header h1 { margin: 0; font-size: 20px; }
  header .src { font-size: 12px; color: #8b949e; margin-top: 4px; font-family: monospace; }
  nav { background: #161b22; padding: 8px 24px; border-bottom: 1px solid #30363d;
        position: sticky; top: 0; z-index: 100; display: flex; gap: 8px; flex-wrap: wrap; }
  nav button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
               padding: 6px 12px; cursor: pointer; border-radius: 4px; font-size: 13px; }
  nav button:hover { background: #30363d; }
  nav button.active { background: #1f6feb; color: white; border-color: #1f6feb; }
  .section { display: none; padding: 16px 24px; max-width: 1600px; }
  .section.active { display: block; }
  .section h2 { margin-top: 0; color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 8px; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 12px; margin-bottom: 16px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
               padding: 12px; }
  .stat-card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .value { font-size: 22px; font-weight: 600; margin-top: 4px; color: #f0f6fc; }
  .stat-card .subvalue { font-size: 12px; color: #8b949e; margin-top: 2px; }
  .chart { background: #161b22; border: 1px solid #30363d; border-radius: 6px;
           padding: 8px; margin-bottom: 16px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 1200px) { .two-col { grid-template-columns: 1fr; } }
  select, input { background: #0d1117; color: #e6edf3; border: 1px solid #30363d;
                  padding: 4px 8px; border-radius: 4px; font-family: monospace; }
  label { color: #8b949e; font-size: 12px; margin-right: 8px; }
  .note { color: #8b949e; font-size: 13px; margin: 4px 0 12px 0; }
  code { background: #161b22; padding: 2px 4px; border-radius: 3px; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>DDS-SLAM Debug Dashboard</h1>
  <div class="src">source: __SOURCE__</div>
</header>
<nav id="nav"></nav>

<section class="section active" id="sec-overview">
  <h2>Overview</h2>
  <div class="stats-grid" id="stats-grid"></div>
  <div class="chart"><div id="chart-motion-inflation"></div></div>
</section>

<section class="section" id="sec-trajectory">
  <h2>Trajectory</h2>
  <div class="note" style="background:#161b22;border:1px solid #30363d;border-radius:6px;padding:12px;line-height:1.5;">
    <b>How SLAM trajectories are compared (TUM / Sturm convention):</b>
    <ol style="margin:6px 0 0 20px;padding:0;">
      <li>SLAM outputs trajectory in its own arbitrary frame (first pose = identity).</li>
      <li>Evaluation script runs <b>Horn / Umeyama alignment</b> against GT before computing ATE.</li>
      <li>Alignment finds the best rigid <code>(R, t)</code> (for SE(3) ATE) or <code>(R, t, s)</code> (for Sim(3) ATE) that maps est &rarr; GT.</li>
      <li>ATE is computed on the <b>aligned</b> trajectory.</li>
    </ol>
    <div style="margin-top:8px;">All plots show: <span style="color:#56d364">GT</span>, <span style="color:#8b949e">raw est (unaligned)</span>, <span style="color:#f85149">SE(3)-aligned est (R+t)</span>, <span style="color:#ffa657">Sim(3)-aligned est (R+t+s)</span>. The ATE in the Overview tab is computed on the SE(3)-aligned curve. Note: raw est is ~3&times; larger in extent than GT, so the aligned curves may appear compressed in the 3D view.</div>
    <div id="align-stats" style="margin-top:8px;color:#8b949e;"></div>
  </div>
  <div class="chart"><div id="chart-traj-3d" style="height: 600px;"></div></div>
  <div class="two-col">
    <div class="chart"><div id="chart-traj-xy"></div></div>
    <div class="chart"><div id="chart-traj-xz"></div></div>
  </div>
</section>

<section class="section" id="sec-errors">
  <h2>Pose error over time</h2>
  <p class="note">Translation (mm) and rotation (deg) error at each frame. Dashed line = const-velocity init error before tracking.</p>
  <div class="chart"><div id="chart-trans-err"></div></div>
  <div class="chart"><div id="chart-rot-err"></div></div>
  <h2>Per-axis positions</h2>
  <p class="note">Est vs GT trajectory projected onto each axis. Diverging ranges indicate directional drift.</p>
  <div class="chart"><div id="chart-axis-x"></div></div>
  <div class="chart"><div id="chart-axis-y"></div></div>
  <div class="chart"><div id="chart-axis-z"></div></div>
</section>

<section class="section" id="sec-motion">
  <h2>Per-frame motion delta</h2>
  <p class="note">Translation magnitude between consecutive est poses vs. the equivalent GT motion. 5x inflation is the signature of our current failure mode.</p>
  <div class="chart"><div id="chart-delta-time"></div></div>
  <div class="two-col">
    <div class="chart"><div id="chart-delta-hist"></div></div>
    <div class="chart"><div id="chart-rot-delta"></div></div>
  </div>
</section>

<section class="section" id="sec-losses">
  <h2>Loss components</h2>
  <p class="note">Per-frame loss at the best tracker iteration (un-weighted). Log scale.</p>
  <div class="chart"><div id="chart-losses"></div></div>
  <div class="two-col">
    <div class="chart"><div id="chart-best-vs-last"></div></div>
    <div class="chart"><div id="chart-psnr"></div></div>
  </div>
</section>

<section class="section" id="sec-tracker">
  <h2>Tracker convergence</h2>
  <p class="note">How many iters the tracker actually ran, and whether init or final was closer to GT.</p>
  <div class="two-col">
    <div class="chart"><div id="chart-iters-used"></div></div>
    <div class="chart"><div id="chart-init-vs-final"></div></div>
  </div>
</section>

<section class="section" id="sec-input">
  <h2>Input signals</h2>
  <p class="note">Depth validity, mean depth, mean RGB - is the input quality stable across the run?</p>
  <div class="chart"><div id="chart-depth-frac"></div></div>
  <div class="chart"><div id="chart-depth-mean"></div></div>
  <div class="chart"><div id="chart-rgb-mean"></div></div>
</section>

<section class="section" id="sec-snapshots">
  <h2>Keyframe pose evolution (from .npz snapshots)</h2>
  <p class="note">How does a past keyframe's pose change as BA rewrites it across the run?</p>
  <div>
    <label>Keyframe:</label>
    <select id="snap-frame-select"></select>
    <span class="note" id="snap-available"></span>
  </div>
  <div class="chart"><div id="chart-snap-trans"></div></div>
  <div class="chart"><div id="chart-snap-rot"></div></div>
  <h2>BA rewrite magnitude per interval</h2>
  <div class="chart"><div id="chart-ba-intervals"></div></div>
</section>

<script>
const DATA = __DATA__;
const DARK = {
  paper_bgcolor: '#161b22',
  plot_bgcolor: '#0d1117',
  font: { color: '#c9d1d9' },
  xaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' },
  yaxis: { gridcolor: '#30363d', zerolinecolor: '#30363d' }
};

// Navigation
const sections = ['overview', 'trajectory', 'errors', 'motion', 'losses', 'tracker', 'input', 'snapshots'];
const nav = document.getElementById('nav');
sections.forEach((s, i) => {
  const btn = document.createElement('button');
  btn.textContent = s.charAt(0).toUpperCase() + s.slice(1);
  btn.onclick = () => {
    document.querySelectorAll('.section').forEach(e => e.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(e => e.classList.remove('active'));
    document.getElementById('sec-' + s).classList.add('active');
    btn.classList.add('active');
    // Plotly charts need re-render when revealed
    window.dispatchEvent(new Event('resize'));
  };
  if (i === 0) btn.classList.add('active');
  nav.appendChild(btn);
});

const df = DATA.df;
const n = df.frame_id.length;

// ========= OVERVIEW STATS =========
function mean(arr) { return arr.reduce((a,b)=>a+b, 0) / arr.length; }
function median(arr) { const s = [...arr].sort((a,b)=>a-b); return s[Math.floor(s.length/2)]; }
function p95(arr) { const s = [...arr].sort((a,b)=>a-b); return s[Math.floor(s.length*0.95)]; }

// Per-frame deltas
const trans_delta_mm = df.trans_delta_m.filter(v => v !== null && v !== '').map(v => v * 1000);
const gt_delta_mm = [];
for (let i = 1; i < n; i++) {
  const dx = df.gt_tx[i] - df.gt_tx[i-1];
  const dy = df.gt_ty[i] - df.gt_ty[i-1];
  const dz = df.gt_tz[i] - df.gt_tz[i-1];
  gt_delta_mm.push(Math.sqrt(dx*dx + dy*dy + dz*dz) * 1000);
}

const final_trans_err_mm = df.trans_err_m[n-1] * 1000;
const mean_est_delta = mean(trans_delta_mm);
const mean_gt_delta = mean(gt_delta_mm);

// Aligned metrics (present only after reanalyse_csv has run — logger finalises on close)
const has_aligned = df.trans_err_aligned_m !== undefined;
function rmse(arr) {
  const valid = arr.filter(v => v !== null && v !== '' && !isNaN(v));
  if (!valid.length) return NaN;
  return Math.sqrt(valid.reduce((a, v) => a + v * v, 0) / valid.length);
}
const ate_rt_mm = has_aligned ? rmse(df.trans_err_aligned_m.map(v => v * 1000)) : NaN;
const ate_sim_mm = has_aligned ? rmse(df.trans_err_sim_m.map(v => v * 1000)) : NaN;
const rpe_trans_mm = has_aligned ? rmse(df.rpe_trans_m.map(v => v === null || v === '' ? null : v * 1000).filter(v => v !== null)) : NaN;
const rpe_rot_deg = has_aligned ? rmse(df.rpe_rot_rad.filter(v => v !== null && v !== '')) * 180 / Math.PI : NaN;

const stats = [
  { label: 'Frames', value: n, subvalue: DATA.meta.kf_count + ' keyframes' },
  { label: 'Run duration', value: (df.wall_s[n-1] / 60).toFixed(1) + ' min' },
  has_aligned
    ? { label: 'ATE (R,t) — paper', value: ate_rt_mm.toFixed(1) + ' mm', subvalue: 'Umeyama-aligned RMSE' }
    : { label: 'Final raw trans err', value: final_trans_err_mm.toFixed(1) + ' mm', subvalue: 'unaligned — run reanalyse_csv' },
  has_aligned
    ? { label: 'ATE (R,t,s) — scaled', value: ate_sim_mm.toFixed(1) + ' mm', subvalue: 'with scale correction' }
    : { label: 'Per-frame motion (est)', value: median(trans_delta_mm).toFixed(3) + ' mm', subvalue: 'median, ' + p95(trans_delta_mm).toFixed(3) + ' mm p95' },
  has_aligned
    ? { label: 'RPE trans / frame', value: rpe_trans_mm.toFixed(3) + ' mm', subvalue: 'alignment-invariant' }
    : { label: 'Per-frame motion (GT)', value: median(gt_delta_mm).toFixed(3) + ' mm', subvalue: 'median, ' + p95(gt_delta_mm).toFixed(3) + ' mm p95' },
  { label: 'Motion inflation', value: (mean_est_delta / mean_gt_delta).toFixed(2) + 'x', subvalue: 'est_mean / gt_mean' },
  { label: 'Final PSNR', value: df.psnr[n-1] ? df.psnr[n-1].toFixed(2) : 'n/a', subvalue: 'mean: ' + mean(df.psnr.filter(v => v !== null && v !== '')).toFixed(2) },
  { label: 'Total time', value: df.wall_s[n-1].toFixed(0) + ' s' }
];
const grid = document.getElementById('stats-grid');
stats.forEach(s => {
  const d = document.createElement('div');
  d.className = 'stat-card';
  d.innerHTML = `<div class="label">${s.label}</div><div class="value">${s.value}</div>${s.subvalue ? `<div class="subvalue">${s.subvalue}</div>` : ''}`;
  grid.appendChild(d);
});

// Motion inflation chart (on overview)
const frame_ids = df.frame_id;
Plotly.newPlot('chart-motion-inflation', [
  { x: frame_ids.slice(1), y: trans_delta_mm, type: 'scatter', mode: 'lines', name: 'est', line: { color: '#f85149', width: 1 } },
  { x: frame_ids.slice(1), y: gt_delta_mm, type: 'scatter', mode: 'lines', name: 'GT', line: { color: '#56d364', width: 1 } }
], { ...DARK, title: 'Per-frame motion magnitude (est vs GT)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'mm' }, height: 350 });

// ========= TRAJECTORY =========
const aligned = DATA.aligned;   // { est_rt: {x,y,z}, est_rts: {x,y,z}, stats: {...} } or null
const hasAlign = aligned && aligned.est_rt;

const traj3d = [
  { x: df.gt_tx,  y: df.gt_ty,  z: df.gt_tz,  type: 'scatter3d', mode: 'lines',
    name: 'GT',                     line: { color: '#56d364', width: 4, dash: 'solid' } },
  { x: df.est_tx, y: df.est_ty, z: df.est_tz, type: 'scatter3d', mode: 'lines',
    name: 'est (raw)',              line: { color: '#8b949e', width: 2, dash: 'solid' } },
];
if (hasAlign) {
  traj3d.push({ x: aligned.est_rt.x,  y: aligned.est_rt.y,  z: aligned.est_rt.z,
    type: 'scatter3d', mode: 'lines', name: 'est (R,t aligned)',
    line: { color: '#f85149', width: 3, dash: 'solid' } });
  traj3d.push({ x: aligned.est_rts.x, y: aligned.est_rts.y, z: aligned.est_rts.z,
    type: 'scatter3d', mode: 'lines', name: 'est (R,t,s aligned)',
    line: { color: '#ffa657', width: 3, dash: 'solid' } });
}
Plotly.newPlot('chart-traj-3d', traj3d,
  { ...DARK, title: 'Trajectory 3D',
    scene: { xaxis: { color: '#c9d1d9' }, yaxis: { color: '#c9d1d9' }, zaxis: { color: '#c9d1d9' }, bgcolor: '#0d1117' },
    height: 580 });

function projTraces(ax1, ax2) {
  const out = [
    { x: df['gt_t' + ax1],  y: df['gt_t' + ax2],  type: 'scatter', mode: 'lines',
      name: 'GT',        line: { color: '#56d364', width: 3, dash: 'solid' } },
    { x: df['est_t' + ax1], y: df['est_t' + ax2], type: 'scatter', mode: 'lines',
      name: 'est (raw)', line: { color: '#8b949e' } },
  ];
  if (hasAlign) {
    out.push({ x: aligned.est_rt[ax1],  y: aligned.est_rt[ax2],  type: 'scatter', mode: 'lines',
               name: 'est (R,t)',   line: { color: '#f85149' } });
    out.push({ x: aligned.est_rts[ax1], y: aligned.est_rts[ax2], type: 'scatter', mode: 'lines',
               name: 'est (R,t,s)', line: { color: '#ffa657' } });
  }
  return out;
}
Plotly.newPlot('chart-traj-xy', projTraces('x', 'y'),
  { ...DARK, title: 'XY projection', xaxis: { ...DARK.xaxis, title: 'x (m)', scaleanchor: 'y', scaleratio: 1 },
    yaxis: { ...DARK.yaxis, title: 'y (m)' }, height: 400 });
Plotly.newPlot('chart-traj-xz', projTraces('x', 'z'),
  { ...DARK, title: 'XZ projection', xaxis: { ...DARK.xaxis, title: 'x (m)' },
    yaxis: { ...DARK.yaxis, title: 'z (m)' }, height: 400 });

if (hasAlign) {
  const st = aligned.stats;
  document.getElementById('align-stats').innerHTML =
    '<b>Alignment fit:</b> ' +
    'Sim(3) scale s = ' + st.scale.toFixed(4) +
    '  (&rarr; est trajectory is ' + (1 / st.scale).toFixed(2) + '&times; too large)' +
    '  &bull; ATE(R,t) = '   + (st.ate_rt_m  * 1000).toFixed(2) + ' mm' +
    '  &bull; ATE(R,t,s) = ' + (st.ate_rts_m * 1000).toFixed(2) + ' mm';
}

// ========= ERRORS =========
const trans_err_mm = df.trans_err_m.map(v => v * 1000);
const init_trans_err_mm = df.init_trans_err_m.map(v => (v === null || v === '') ? null : v * 1000);
const rot_err_deg = df.rot_err_rad.map(v => v * 180 / Math.PI);
const init_rot_err_deg = df.init_rot_err_rad.map(v => (v === null || v === '') ? null : v * 180 / Math.PI);

// Prefer aligned columns if present (logger finalises CSV on close via reanalyse_csv)
const trans_err_aligned_mm = has_aligned ? df.trans_err_aligned_m.map(v => v * 1000) : null;
const rot_err_aligned_deg  = has_aligned ? df.rot_err_aligned_rad.map(v => v * 180 / Math.PI) : null;
const rpe_trans_series_mm  = has_aligned ? df.rpe_trans_m.map(v => (v === null || v === '') ? null : v * 1000) : null;
const rpe_rot_series_deg   = has_aligned ? df.rpe_rot_rad.map(v => (v === null || v === '') ? null : v * 180 / Math.PI) : null;

const trans_traces = has_aligned
  ? [
      { x: frame_ids, y: trans_err_aligned_mm, type: 'scatter', mode: 'lines', name: 'aligned (Umeyama R,t)', line: { color: '#f85149' } },
      { x: frame_ids, y: trans_err_mm,        type: 'scatter', mode: 'lines', name: 'raw (unaligned)',      line: { color: '#8b949e', dash: 'dot' } },
      { x: frame_ids, y: rpe_trans_series_mm, type: 'scatter', mode: 'lines', name: 'RPE (body-frame)',     line: { color: '#58a6ff' }, yaxis: 'y2' },
    ]
  : [
      { x: frame_ids, y: trans_err_mm,        type: 'scatter', mode: 'lines', name: 'final (after tracker)', line: { color: '#f85149' } },
      { x: frame_ids, y: init_trans_err_mm,   type: 'scatter', mode: 'lines', name: 'init (before tracker)', line: { color: '#8b949e', dash: 'dash' } },
    ];
const trans_title = has_aligned
  ? 'Translation error — aligned (red) + raw (grey) + RPE per frame (blue, right axis)'
  : 'Translation error (raw, unaligned — run reanalyse_csv for aligned view)';
const trans_layout = { ...DARK, title: trans_title, xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'mm' }, height: 350 };
if (has_aligned) {
  trans_layout.yaxis2 = { title: 'RPE mm/frame', overlaying: 'y', side: 'right', color: '#58a6ff', gridcolor: '#21262d' };
}
Plotly.newPlot('chart-trans-err', trans_traces, trans_layout);

const rot_traces = has_aligned
  ? [
      { x: frame_ids, y: rot_err_aligned_deg, type: 'scatter', mode: 'lines', name: 'aligned (Umeyama R,t)', line: { color: '#f85149' } },
      { x: frame_ids, y: rot_err_deg,         type: 'scatter', mode: 'lines', name: 'raw (unaligned)',      line: { color: '#8b949e', dash: 'dot' } },
      { x: frame_ids, y: rpe_rot_series_deg,  type: 'scatter', mode: 'lines', name: 'RPE (body-frame)',     line: { color: '#58a6ff' }, yaxis: 'y2' },
    ]
  : [
      { x: frame_ids, y: rot_err_deg,         type: 'scatter', mode: 'lines', name: 'final', line: { color: '#f85149' } },
      { x: frame_ids, y: init_rot_err_deg,    type: 'scatter', mode: 'lines', name: 'init',  line: { color: '#8b949e', dash: 'dash' } },
    ];
const rot_title = has_aligned
  ? 'Rotation error — aligned (red) + raw (grey) + RPE per frame (blue, right axis)'
  : 'Rotation error (raw, unaligned — run reanalyse_csv for aligned view)';
const rot_layout = { ...DARK, title: rot_title, xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'deg' }, height: 350 };
if (has_aligned) {
  rot_layout.yaxis2 = { title: 'RPE deg/frame', overlaying: 'y', side: 'right', color: '#58a6ff', gridcolor: '#21262d' };
}
Plotly.newPlot('chart-rot-err', rot_traces, rot_layout);

// Per-axis (delta from frame 0, to remove coord baseline)
const est_x0 = df.est_tx[0], est_y0 = df.est_ty[0], est_z0 = df.est_tz[0];
const gt_x0 = df.gt_tx[0], gt_y0 = df.gt_ty[0], gt_z0 = df.gt_tz[0];
['x', 'y', 'z'].forEach(ax => {
  const est = df['est_t' + ax].map((v, i) => (v - df['est_t' + ax][0]) * 1000);
  const gt  = df['gt_t' + ax].map((v, i) => (v - df['gt_t' + ax][0]) * 1000);
  Plotly.newPlot('chart-axis-' + ax, [
    { x: frame_ids, y: est, type: 'scatter', mode: 'lines', name: 'est', line: { color: '#f85149' } },
    { x: frame_ids, y: gt,  type: 'scatter', mode: 'lines', name: 'GT',  line: { color: '#56d364' } }
  ], { ...DARK, title: ax + ' position (delta from frame 0)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'mm' }, height: 300 });
});

// ========= MOTION =========
Plotly.newPlot('chart-delta-time', [
  { x: frame_ids.slice(1), y: trans_delta_mm, type: 'scatter', mode: 'lines', name: 'est delta', line: { color: '#f85149', width: 1 } },
  { x: frame_ids.slice(1), y: gt_delta_mm,    type: 'scatter', mode: 'lines', name: 'GT delta',  line: { color: '#56d364', width: 1 } }
], { ...DARK, title: 'Per-frame translation magnitude', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'mm / frame' }, height: 400 });

Plotly.newPlot('chart-delta-hist', [
  { x: trans_delta_mm, type: 'histogram', name: 'est', marker: { color: '#f85149' }, opacity: 0.7, nbinsx: 50 },
  { x: gt_delta_mm,    type: 'histogram', name: 'GT',  marker: { color: '#56d364' }, opacity: 0.7, nbinsx: 50 }
], { ...DARK, title: 'Per-frame motion distribution', xaxis: { ...DARK.xaxis, title: 'mm' }, yaxis: { ...DARK.yaxis, title: 'count' }, barmode: 'overlay', height: 350 });

const rot_delta_deg = df.rot_delta_rad.map(v => (v === null || v === '') ? null : v * 180 / Math.PI);
Plotly.newPlot('chart-rot-delta', [
  { x: frame_ids.slice(1), y: rot_delta_deg.slice(1), type: 'scatter', mode: 'lines', name: 'rot delta', line: { color: '#a371f7', width: 1 } }
], { ...DARK, title: 'Per-frame rotation delta (deg)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'deg / frame' }, height: 350 });

// ========= LOSSES =========
const lossCols = ['loss_rgb', 'loss_depth', 'loss_sdf', 'loss_fs', 'loss_edge_semantic'];
const lossColors = ['#f85149', '#56d364', '#58a6ff', '#e3b341', '#a371f7'];
Plotly.newPlot('chart-losses', lossCols.map((c, i) => ({
  x: frame_ids, y: df[c], type: 'scatter', mode: 'lines', name: c.replace('loss_', ''),
  line: { color: lossColors[i], width: 1 }
})), { ...DARK, title: 'Per-component losses (log scale)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'loss', type: 'log' }, height: 400 });

Plotly.newPlot('chart-best-vs-last', [
  { x: frame_ids, y: df.best_loss, type: 'scatter', mode: 'lines', name: 'best', line: { color: '#56d364', width: 1 } },
  { x: frame_ids, y: df.last_loss, type: 'scatter', mode: 'lines', name: 'last', line: { color: '#f85149', width: 1 } }
], { ...DARK, title: 'Tracker: best vs last iter loss', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'loss' }, height: 350 });

Plotly.newPlot('chart-psnr', [
  { x: frame_ids, y: df.psnr, type: 'scatter', mode: 'lines', name: 'PSNR', line: { color: '#58a6ff' } }
], { ...DARK, title: 'Rendering PSNR', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'dB' }, height: 350 });

// ========= TRACKER =========
Plotly.newPlot('chart-iters-used', [
  { x: df.tracking_iters_used, type: 'histogram', marker: { color: '#58a6ff' }, name: 'iters_used', nbinsx: 15 }
], { ...DARK, title: 'Tracker iters used per frame (max: ' + df.tracking_iters_config[1] + ')', xaxis: { ...DARK.xaxis, title: 'iters' }, yaxis: { ...DARK.yaxis, title: 'frame count' }, height: 350 });

// init vs final scatter
const init_final_gap = trans_err_mm.map((v, i) => v - (init_trans_err_mm[i] || v));
Plotly.newPlot('chart-init-vs-final', [
  { x: init_trans_err_mm, y: trans_err_mm, type: 'scatter', mode: 'markers',
    marker: { color: '#a371f7', size: 3, opacity: 0.5 }, name: 'frame' }
], { ...DARK, title: 'Init vs final trans err (points below y=x = tracker helped)',
    xaxis: { ...DARK.xaxis, title: 'init err (mm)' },
    yaxis: { ...DARK.yaxis, title: 'final err (mm)', scaleanchor: 'x', scaleratio: 1 },
    shapes: [{ type: 'line', x0: 0, y0: 0, x1: Math.max(...init_trans_err_mm.filter(v=>v!==null))||200, y1: Math.max(...init_trans_err_mm.filter(v=>v!==null))||200, line: { color: '#8b949e', dash: 'dash', width: 1 } }],
    height: 400 });

// ========= INPUT =========
Plotly.newPlot('chart-depth-frac', [
  { x: frame_ids, y: df.depth_valid_frac, type: 'scatter', mode: 'lines', line: { color: '#56d364' } }
], { ...DARK, title: 'Depth valid fraction (pixels with 0 < d < depth_trunc)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'fraction', range: [0, 1.05] }, height: 300 });
Plotly.newPlot('chart-depth-mean', [
  { x: frame_ids, y: df.depth_mean, type: 'scatter', mode: 'lines', line: { color: '#58a6ff' } }
], { ...DARK, title: 'Mean depth (meters, nonzero pixels)', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'm' }, height: 300 });
Plotly.newPlot('chart-rgb-mean', [
  { x: frame_ids, y: df.rgb_mean, type: 'scatter', mode: 'lines', line: { color: '#e3b341' } }
], { ...DARK, title: 'Mean RGB intensity', xaxis: { ...DARK.xaxis, title: 'frame' }, yaxis: { ...DARK.yaxis, title: 'value' }, height: 300 });

// ========= SNAPSHOTS =========
const snaps = DATA.snapshots;
const snapTimes = Object.keys(snaps).map(Number).sort((a,b) => a-b);
const snapSelect = document.getElementById('snap-frame-select');
const availDiv = document.getElementById('snap-available');

if (snapTimes.length === 0) {
  availDiv.textContent = 'No pose snapshots found.';
} else {
  // Populate dropdown with keyframes available in first snapshot (every 5)
  const firstSnap = snaps[snapTimes[0]];
  const firstIds = firstSnap.ids.filter(i => i % 5 === 0).slice(0, 400);
  firstIds.forEach(fid => {
    const opt = document.createElement('option');
    opt.value = fid; opt.textContent = 'frame ' + fid;
    snapSelect.appendChild(opt);
  });
  availDiv.textContent = '(snapshots at frames: ' + snapTimes.join(', ') + ')';

  function pose_delta(A, B) {
    const dx = A[0][3] - B[0][3], dy = A[1][3] - B[1][3], dz = A[2][3] - B[2][3];
    const trans = Math.sqrt(dx*dx + dy*dy + dz*dz);
    let tr = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) tr += A[i][j] * B[j][i];
    const ct = Math.max(-1, Math.min(1, (tr - 1) / 2));
    const rot = Math.acos(ct) * 180 / Math.PI;
    return [trans * 1000, rot];
  }

  function updateSnapCharts() {
    const fid = parseInt(snapSelect.value);
    const xs = [], ys_trans = [], ys_rot = [];
    let ref = null;
    for (const t of snapTimes) {
      const snap = snaps[t];
      const idx = snap.ids.indexOf(fid);
      if (idx === -1) continue;
      const p = snap.poses[idx];
      if (ref === null) ref = p;
      const [dt, dr] = pose_delta(p, ref);
      xs.push(t); ys_trans.push(dt); ys_rot.push(dr);
    }
    Plotly.newPlot('chart-snap-trans', [{
      x: xs, y: ys_trans, type: 'scatter', mode: 'lines+markers', line: { color: '#f85149' }, name: 'd_trans'
    }], { ...DARK, title: 'Frame ' + fid + ': translation drift across snapshots', xaxis: { ...DARK.xaxis, title: 'snapshot @ frame' }, yaxis: { ...DARK.yaxis, title: 'd_trans vs first-seen (mm)' }, height: 320 });
    Plotly.newPlot('chart-snap-rot', [{
      x: xs, y: ys_rot, type: 'scatter', mode: 'lines+markers', line: { color: '#a371f7' }, name: 'd_rot'
    }], { ...DARK, title: 'Frame ' + fid + ': rotation drift across snapshots', xaxis: { ...DARK.xaxis, title: 'snapshot @ frame' }, yaxis: { ...DARK.yaxis, title: 'd_rot vs first-seen (deg)' }, height: 320 });
  }
  snapSelect.onchange = updateSnapCharts;
  updateSnapCharts();

  // BA rewrite magnitudes per interval
  const intervalLabels = [], intervalMeans = [], intervalP95s = [], intervalMaxs = [];
  for (let i = 0; i < snapTimes.length - 1; i++) {
    const a = snapTimes[i], b = snapTimes[i+1];
    const snapA = snaps[a], snapB = snaps[b];
    const mapA = {}; snapA.ids.forEach((id, k) => { mapA[id] = snapA.poses[k]; });
    const mapB = {}; snapB.ids.forEach((id, k) => { mapB[id] = snapB.poses[k]; });
    const deltas = [];
    Object.keys(mapA).forEach(id => {
      if (mapB[id]) {
        const [dt, _] = pose_delta(mapB[id], mapA[id]);
        deltas.push(dt);
      }
    });
    deltas.sort((x, y) => x - y);
    intervalLabels.push(a + ' -> ' + b);
    intervalMeans.push(deltas.reduce((x, y) => x + y, 0) / deltas.length);
    intervalP95s.push(deltas[Math.floor(deltas.length * 0.95)]);
    intervalMaxs.push(deltas[deltas.length - 1]);
  }
  Plotly.newPlot('chart-ba-intervals', [
    { x: intervalLabels, y: intervalMeans, type: 'bar', name: 'mean', marker: { color: '#56d364' } },
    { x: intervalLabels, y: intervalP95s,  type: 'bar', name: 'p95',  marker: { color: '#e3b341' } },
    { x: intervalLabels, y: intervalMaxs,  type: 'bar', name: 'max',  marker: { color: '#f85149' } }
  ], { ...DARK, title: 'BA rewrite magnitudes (d_trans mm) across common frames per interval',
       xaxis: { ...DARK.xaxis, title: 'snapshot interval' }, yaxis: { ...DARK.yaxis, title: 'mm' }, height: 400 });
}
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--debug_dir', required=True,
                    help='Path to the debug/ folder (containing debug_log.csv + pose_snapshots/)')
    ap.add_argument('--output', default=None,
                    help='Output HTML path (default: <debug_dir>/debug_dashboard.html)')
    args = ap.parse_args()

    debug_dir = Path(args.debug_dir)
    if not debug_dir.exists():
        raise SystemExit(f'debug_dir not found: {debug_dir}')
    csv_path = debug_dir / 'debug_log.csv'
    if not csv_path.exists():
        raise SystemExit(f'debug_log.csv not found at {csv_path}')

    df = pd.read_csv(csv_path)
    # Replace NaN with None for valid JSON
    df = df.where(pd.notnull(df), None)

    # Load pose snapshots if present
    snaps = {}
    snap_dir = debug_dir / 'pose_snapshots'
    if snap_dir.exists():
        for f in sorted(snap_dir.glob('poses_frame_*.npz')):
            # Strip 'poses_frame_' prefix; Co-SLAM writes 'poses_frame_NNN_final.npz'
            # for its last snapshot, so keep only the numeric head.
            stem = f.stem.replace('poses_frame_', '')
            digits = ''.join(c for c in stem.split('_')[0] if c.isdigit())
            if not digits:
                continue
            tag = int(digits)
            d = np.load(f)
            snaps[tag] = {
                'ids': d['ids'].astype(int).tolist(),
                'poses': d['poses'].astype(float).tolist(),
            }
        print(f'loaded {len(snaps)} pose snapshots')
    else:
        print(f'no pose_snapshots/ dir — snapshots section will be empty')

    # --- Umeyama alignments (SE(3) and Sim(3)) for the trajectory plots ---
    aligned_payload = None
    try:
        est = df[['est_tx', 'est_ty', 'est_tz']].to_numpy(dtype=np.float64)
        gt  = df[['gt_tx',  'gt_ty',  'gt_tz' ]].to_numpy(dtype=np.float64)
        valid = np.isfinite(est).all(1) & np.isfinite(gt).all(1)
        if valid.sum() >= 3:
            est_v, gt_v = est[valid], gt[valid]
            R_rt, t_rt, _      = umeyama(est_v, gt_v, with_scale=False)
            R_rts, t_rts, s_rts = umeyama(est_v, gt_v, with_scale=True)

            est_rt  = apply_sim3(est, R_rt,  t_rt,  1.0)
            est_rts = apply_sim3(est, R_rts, t_rts, s_rts)

            resid_rt  = np.linalg.norm(est_rt[valid]  - gt_v, axis=1)
            resid_rts = np.linalg.norm(est_rts[valid] - gt_v, axis=1)
            ate_rt  = float(np.sqrt((resid_rt  ** 2).mean()))
            ate_rts = float(np.sqrt((resid_rts ** 2).mean()))

            aligned_payload = {
                'est_rt':  {'x': est_rt[:, 0].tolist(),  'y': est_rt[:, 1].tolist(),  'z': est_rt[:, 2].tolist()},
                'est_rts': {'x': est_rts[:, 0].tolist(), 'y': est_rts[:, 1].tolist(), 'z': est_rts[:, 2].tolist()},
                'stats': {'scale': float(s_rts), 'ate_rt_m': ate_rt, 'ate_rts_m': ate_rts},
            }
            print(f'alignment: scale={s_rts:.4f}  ATE(R,t)={ate_rt*1000:.2f}mm  ATE(R,t,s)={ate_rts*1000:.2f}mm')
    except Exception as e:
        print(f'alignment failed: {e}')

    data = {
        'df': df.to_dict(orient='list'),
        'snapshots': snaps,
        'aligned': aligned_payload,
        'meta': {
            'n_frames': int(len(df)),
            'kf_count': int(df['is_keyframe'].sum()),
            'run_duration_s': float(df['wall_s'].iloc[-1]) if len(df) else 0,
        },
    }

    source_str = str(debug_dir.resolve()).replace('\\', '/')
    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(data, separators=(',', ':')))
    html = html.replace('__SOURCE__', source_str)

    out = Path(args.output) if args.output else debug_dir / 'debug_dashboard.html'
    out.write_text(html, encoding='utf-8')
    mb = out.stat().st_size / 1e6
    print(f'wrote {out} ({mb:.2f} MB)')
    print(f'open in browser: file://{out.resolve().as_posix()}')


if __name__ == '__main__':
    main()
