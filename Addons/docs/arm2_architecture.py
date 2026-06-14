"""
Render the Arm-2 stack architecture diagram (code-grounded + adversarially verified).

This is the CORRECTED dataflow: the 6 errors the verification pass found in the first
draft are fixed here (per-ray vs per-sample shapes, scalar-loss residual, def_reg mask
site, seg-signal caveat, tracking-only weighting, weighted-sum losses).

Run:  python Addons/docs/arm2_architecture.py   ->  Addons/docs/arm2_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C_BASE  = '#e9e9e9'   # base DDS-SLAM path (untouched)
C_IN    = '#fff2cc'   # inputs
C_WILD  = '#cfe2f3'   # WildGS  (Inc-1/2/3) learnt core
C_NRGS  = '#fce5cd'   # NRGS    (Inc-4) gated refinement
C_SNI   = '#d9ead3'   # SNI     (Inc-5) optional
C_LOSS  = '#f4cccc'   # losses / optimisers


def box(ax, cx, cy, w, h, text, fc, fs=8.5, weight='normal', ec='#333333'):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle='round,pad=0.02,rounding_size=0.10', fc=fc, ec=ec, lw=1.2, zorder=2))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs, zorder=3, weight=weight)


def arrow(ax, p0, p1, color='#333333', lw=1.4, rad=0.0, ls='-', style='-|>'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                 color=color, lw=lw, connectionstyle=f'arc3,rad={rad}', zorder=1, linestyle=ls))


fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20); ax.set_ylim(0, 12.6); ax.axis('off')

ax.text(10, 12.2, 'Arm-2 stack on DDS-SLAM-Base  —  WildGS + NRGS + SNI as config-gated increments',
        ha='center', fontsize=15, weight='bold')
ax.text(10, 11.78, 'grey = base (untouched)   blue = WildGS learnt core (Inc-1/2/3)   '
        'orange = NRGS gated (Inc-4)   green = SNI optional (Inc-5)', ha='center', fontsize=9.5, color='#444')
ax.text(10, 11.5, 'σ² has THREE uses: (a) down-weight POSE   (b) ROUTE the field   (c) THROTTLE the MAP  '
        '(c = Battery-7 refinement, pending probe)', ha='center', fontsize=9, color='#1f5fa8')

# ---------------- BASE LOOP (centre) ----------------
box(ax, 10, 11.0, 5.4, 0.7, 'rays (o, d, t)  +  target rgb / depth', C_IN, weight='bold')
box(ax, 10, 9.85, 5.4, 0.9, 'TimeNet (MLP)  ->  Δx = vox_motion  [per-SAMPLE]\ndecoder.py:407  /  scene_rep.py:183', C_BASE)
box(ax, 10, 8.55, 6.2, 1.0,
    '× gates in run_network:  hardbound -> anchor -> [oracle_w] -> [surf_w] -> pts+Δx\n'
    'def_reg = (Δx²).mean  POST-gate   scene_rep.py:187-206', C_BASE)
box(ax, 10, 7.25, 6.2, 0.9, 'SDFNet -> sdf, geo_feat(15) -> color / edge heads\ndecoder.py:430', C_BASE)
box(ax, 10, 5.95, 6.2, 0.9, 'raw2outputs  (volume render, weights)\n-> rgb_map, depth, edge_map  [per-RAY]   scene_rep.py:104-128', C_BASE)
box(ax, 7.0, 4.2, 4.4, 1.05,
    'TRACKING loss (pose only)\nrgb/depth/edge  scene_rep.py:448-464\noptimise POSE  ddsslam.py:575', C_LOSS)
box(ax, 13.0, 4.2, 4.8, 1.15,
    'MAPPING / BA loss (map + KF pose)\n+ def_reg  ddsslam.py:214 · forward :460\n'
    '(c) σ² THROTTLES map grad here [pending probe]', C_LOSS)
arrow(ax, (10, 10.65), (10, 10.32)); arrow(ax, (10, 9.4), (10, 9.06))
arrow(ax, (10, 8.05), (10, 7.71)); arrow(ax, (10, 6.8), (10, 6.41))
arrow(ax, (8.7, 5.55), (7.6, 4.75)); arrow(ax, (11.3, 5.55), (12.4, 4.75))

# ---------------- WildGS (left, blue) ----------------
box(ax, 3.0, 7.25, 5.0, 0.95, 'WildGS  UncertaintyNet(geo_feat)\n-> σ²  [per-SAMPLE]   [NEW · Inc-1]', C_WILD, weight='bold')
box(ax, 3.0, 5.95, 5.0, 0.9, 'volume-render σ² (raw2outputs weights)\n-> σ²  [per-RAY]', C_WILD)
box(ax, 3.0, 8.7, 5.2, 1.0,
    '(b) ROUTE deform  [Inc-3]\na_ray = σ²/(σ²+c);  eff_gate = max(a_ray, λ·seg)\n-> oracle_w  (PER-RAY only)', C_WILD)
box(ax, 3.0, 4.1, 5.2, 1.05,
    '(a) POSE down-weight  [Inc-2 · TRACKING only]\nweights = (rgb_mask · clip(1/σ², w_min, w_max)).detach()\nvia compute_loss(weights=) — linear, NOT the ×-trick', C_WILD)
# geo_feat -> uncertainty
arrow(ax, (7.0, 7.25), (5.5, 7.25), color='#1f5fa8')
ax.text(6.25, 7.45, 'geo_feat', fontsize=7.5, color='#1f5fa8', ha='center')
arrow(ax, (3.0, 6.78), (3.0, 6.4), color='#1f5fa8')
# route branch up to the gate
arrow(ax, (3.0, 7.72), (3.0, 8.2), color='#1f5fa8')
arrow(ax, (5.6, 8.7), (6.9, 8.62), color='#1f5fa8', rad=-0.15)
ax.text(6.0, 9.15, 'oracle_w slot', fontsize=7.5, color='#1f5fa8', ha='center')
# pose branch down to tracking loss
arrow(ax, (3.0, 5.5), (3.0, 4.63), color='#1f5fa8')
arrow(ax, (4.6, 3.9), (5.0, 4.0), color='#1f5fa8', rad=-0.2)
ax.text(4.9, 3.55, 'weights=', fontsize=7.5, color='#1f5fa8', ha='center')

# ---------------- NRGS (right, orange) ----------------
box(ax, 17.0, 7.25, 5.4, 1.0,
    'NRGS force-rigid render (deformation_off=True)\n-> E_rigid [per-RAY]   (restore in finally!)\n[Inc-4 · MAPPING only · 2× render]', C_NRGS, weight='bold')
box(ax, 17.0, 5.95, 5.4, 0.8, 'normal render -> E_deform [per-RAY]\nfrom rgb_map BEFORE reduction', C_NRGS)
box(ax, 17.0, 4.75, 5.4, 0.8, 'w* = σ( α·(2·seg−1) + β·(E_rigid − E_deform) )\n⚠ needs BINARY tool seg (see note)', C_NRGS)
box(ax, 17.0, 3.25, 5.4, 1.0,
    'supervise:  BCE(edge gate, w*)\n+ field  w*·E_deform\n+ def_reg × (1−w*)   [ONE mask site]', C_NRGS)
arrow(ax, (14.1, 6.9), (15.0, 7.0), color='#b06a17', rad=0.1)
ax.text(14.6, 7.45, 'force-rigid', fontsize=7.5, color='#b06a17', ha='center')
arrow(ax, (17.0, 6.75), (17.0, 6.36), color='#b06a17')
arrow(ax, (17.0, 5.55), (17.0, 5.16), color='#b06a17')
arrow(ax, (17.0, 4.34), (17.0, 3.76), color='#b06a17')
arrow(ax, (15.0, 3.0), (14.2, 3.9), color='#b06a17', rad=0.15)

# ---------------- SNI (bottom, green) ----------------
box(ax, 10, 2.15, 7.6, 0.85,
    'SNI  FiLM conditions geo_feat before the heads   [Inc-5 · optional]\n'
    'mode=film (default)  |  xattn = larger-data / CRCD cell', C_SNI, weight='bold')
arrow(ax, (10, 2.58), (10, 6.78), color='#3a7d2c', rad=0.0, ls=(0, (5, 3)))
ax.text(10.35, 4.7, 'FiLM (optional)', fontsize=7.5, color='#3a7d2c', ha='left', rotation=90)

# ---------------- seg input (resolved decision) ----------------
box(ax, 10, 0.95, 10.4, 0.95,
    'seg = REQUIRED canonicalized input (thesis aim).  Masks EXIST for all 3 datasets but are currently COLLAPSED to a Canny edge field (dataset.py:167/338) — surface the RAW mask.\n'
    'Formats: StereoMIS binary tool(0/255) · CRCD {0=bg,1=Liver,2=Gallbladder,3=Tool} · SemSup {0,1,2}.  Canonicalize -> {tool,tissue,bg} (TriGauge), plumb PER-RAY into route + NRGS.',
    '#e8f0e0', fs=8.0)

# ---------------- build/flags note ----------------
box(ax, 3.0, 1.35, 5.4, 1.6,
    'Inc-0 (this commit): config blocks\nuncertainty / nrgs / sni  (all enable:false)\n'
    '+ regression harness.\nFlags off => modules NOT built\n=> 0 torch-RNG => bit-identical base.',
    '#eeeeee', fs=8.0)
box(ax, 17.0, 1.35, 5.4, 1.6,
    'Build order:\nInc-0 plumbing -> Inc-1 σ² head ->\nInc-2 pose -> Inc-3 route ->\nInc-0.5 probe -> [Inc-4 NRGS] ->\n[Inc-5 SNI].  Scored: SemSup + CRCD vs mogev2',
    '#eeeeee', fs=8.0)

plt.tight_layout()
out = __file__.replace('arm2_architecture.py', 'arm2_architecture.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('wrote', out)
