#!/usr/bin/env python3
"""FIELD AUTOPSY -- independent, code-grounded re-test of the deformation field's dead-verdict,
runnable on CPU from checkpoints alone (no tcnn/GPU/dataset).

Re-derives everything from the CURRENT code rather than trusting recorded conclusions:
  forward path (scene_rep.run_network): h = [freq(t) | freq(xyz)] -> time_net (Linear bias=False +
  ReLU, 96->128->128->128->3) -> optional hardbound tanh -> t==0 anchor -> pts + dx.
  CRCD chain verified: dynamic=True, no deformation_off, raw INT frame times, hardbound unset.

Tests per checkpoint:
  A WEIGHT AUDIT  (encoder-independent): per-layer ||W||_F + max|w| vs a fresh Kaiming init.
    Output-layer ~0 => dx==0 for ANY input (structural death, no encoder assumptions needed).
  B FUNCTIONAL AUDIT: |dx| over a grid of in-bound points x all frame times, through a local
    replication of tcnn's parameter-free Frequency encoding. Layout uncertainty hedged by running
    BOTH plausible layouts (dim-major sin/cos interleave vs freq-major); magnitudes are layout-
    robust (bounded sin/cos inputs into a weight-normed MLP). Random-init control net = the scale
    a NON-collapsed net of this architecture produces (the memory's field_off control ~0.028).
  C MECHANISM (fresh net, no checkpoint): (i) gradient-flow -- can this exact forward learn a
    synthetic dx* (sanity that the architecture is trainable); (ii) starvation -- Adam with the
    fork's timenet weight_decay default (1e-6) + zero-mean noise gradients (render-loss proxy):
    does the function decay toward 0?

Usage:  python Addons/eval/field_autopsy.py            (probes the default local checkpoint set)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

CKPTS = {
    'E3_base_jun26':  ("F:/Datasets/DDS-SLAM Results/results/benslam/best/E_3-005/E3_005_base_s0/demo/checkpoint264.pt", 265,
                       [[-0.069, 0.043], [-0.039, 0.0386], [0.0112, 0.1128]]),
    'E3_best_jun26':  ("F:/Datasets/DDS-SLAM Results/results/benslam/best/E_3-005/E3_005_best_s0/demo/checkpoint264.pt", 265,
                       [[-0.069, 0.043], [-0.039, 0.0386], [0.0112, 0.1128]]),
    'C1_jun04':       ("F:/Datasets/DDS-SLAM Results/results/ddsslam/crcd/dds_crcd_4snippets_20260604-20260605T052329Z-3-001/dds_crcd_4snippets_20260604/C1_001/_extracted/demo/checkpoint359.pt", 360,
                       [[-0.5, 0.5], [-0.5, 0.5], [0.3, 1.0]]),
    'SemSup_paperf':  ("F:/Datasets/DDS-SLAM Results/results/ddsslam/crcd/_extracted/paper_faithful/checkpoint1286.pt", 151,
                       [[-0.5, 0.5], [-0.5, 0.5], [0.3, 1.0]]),
    'SemSup_v2':      ("F:/Datasets/DDS-SLAM Results/results/ddsslam/crcd/_extracted/semsup_v2/checkpoint150.pt", 151,
                       [[-0.5, 0.5], [-0.5, 0.5], [0.3, 1.0]]),
}


def freq_encode(x, F=12, layout='interleave'):
    """tcnn 'Frequency' replication: per dim, frequencies 2^0..2^(F-1) (x scaled by pi), sin & cos.
    layout 'interleave' = per-dim blocks [sin f0, cos f0, sin f1, ...]; 'freqmajor' = per-freq blocks."""
    outs = []
    for d in range(x.shape[-1]):
        xd = x[..., d:d + 1]
        per = []
        for j in range(F):
            arg = xd * (2.0 ** j) * np.pi
            per += [torch.sin(arg), torch.cos(arg)]
        outs.append(torch.cat(per, dim=-1))
    if layout == 'interleave':
        return torch.cat(outs, dim=-1)
    blocks = []
    for j in range(2 * F):
        blocks += [o[..., j:j + 1] for o in outs]
    return torch.cat(blocks, dim=-1)


def build_time_net(shapes=(96, 128, 128, 128, 3)):
    layers = []
    for i in range(len(shapes) - 1):
        layers.append(nn.Linear(shapes[i], shapes[i + 1], bias=False))
        if i < len(shapes) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def load_time_net(sd):
    keys = sorted([k for k in sd if k.startswith('decoder.time_net.model.')],
                  key=lambda k: int(k.split('.')[3]))
    net = build_time_net(tuple([sd[keys[0]].shape[1]] + [sd[k].shape[0] for k in keys]))
    lin = [m for m in net if isinstance(m, nn.Linear)]
    for m, k in zip(lin, keys):
        m.weight.data.copy_(sd[k])
    return net, keys


FREQ_BY_INDIM = {96: (12, 12), 68: (4, 10)}   # in_dim -> (freq_n_t, freq_n_xyz): 12/12 default, 4/10 paper-faithful


def dx_stats(net, bound, n_frames, layout, n_pts=2000, n_t=24, seed=0, freqs=(12, 12)):
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor([b[0] for b in bound]); hi = torch.tensor([b[1] for b in bound])
    pts = lo + (hi - lo) * torch.rand(n_pts, 3, generator=g)
    ts = torch.linspace(0, n_frames - 1, n_t).round()          # RAW int frame times (CRCD convention)
    Ft, Fx = freqs
    mags = []
    with torch.no_grad():
        for t in ts:
            h = torch.cat([freq_encode(torch.full((n_pts, 1), float(t)), F=Ft, layout=layout),
                           freq_encode(pts, F=Fx, layout=layout)], dim=-1)
            dx = net(h)
            if float(t) == 0.0:
                dx = torch.zeros_like(dx)                       # the t==0 anchor (scene_rep:218)
            mags.append(dx.norm(dim=-1))
    m = torch.stack(mags)                                       # [n_t, n_pts]
    return float(m.mean()), float(m.max()), float(m[1:].mean())  # incl/excl the anchored t=0


def main():
    torch.manual_seed(0)
    print("=" * 100)
    print(f"{'ckpt':16s} {'L0..L3 ||W||_F (fresh-init ref in brackets)':52s} {'|dx| mean':>10s} {'max':>8s} {'ctrl-mean':>9s}")
    for name, (path, n_frames, bound) in CKPTS.items():
        if not os.path.exists(path):
            print(f"{name:16s} MISSING {path}"); continue
        sd = torch.load(path, map_location='cpu', weights_only=False)['model']
        net, keys = load_time_net(sd)
        in_dim = sd[keys[0]].shape[1]
        freqs = FREQ_BY_INDIM.get(in_dim, (12, 12))
        with torch.no_grad():
            lin = [m for m in net if isinstance(m, nn.Linear)]
            fresh = build_time_net(tuple([in_dim] + [m.weight.shape[0] for m in lin]))
            fresh_norms = [float(m.weight.norm()) for m in fresh if isinstance(m, nn.Linear)]
            norms = [float(m.weight.norm()) for m in lin]
        nstr = " ".join(f"{n:7.3f}[{f:5.1f}]" for n, f in zip(norms, fresh_norms))
        mean_i, max_i, mean_x = dx_stats(net, bound, n_frames, 'interleave', freqs=freqs)
        mean_f, _, _ = dx_stats(net, bound, n_frames, 'freqmajor', freqs=freqs)
        ctrl, _, _ = dx_stats(fresh, bound, n_frames, 'interleave', freqs=freqs)
        # forensic: HOW dead. |w| all at one tiny magnitude (std/mean << fresh) + denormal range
        # (< 1.18e-38) = Adam wd sign-decay (equal-rate shrink) -> eps-knee multiplicative collapse
        # (step ~ W*(1 - lr*wd/eps) once sqrt(v) << eps) -> float32 denormal annihilation.
        a = torch.cat([m.weight.detach().flatten().abs() for m in lin])
        forensic = (f"|w| med={float(a.median()):.1e} spread(std/mean)={float(a.std() / a.mean()):.2f} "
                    f"{'DENORMAL-ANNIHILATED' if float(a.max()) < 1.18e-38 else ('~fresh-scale' if float(a.median()) > 1e-3 else 'small')}")
        print(f"{name:16s} {nstr:52s} {mean_x:10.6f} {max_i:8.4f} {ctrl:9.4f}   (freqs {freqs}, layoutB mean {mean_f:.6f}; {forensic})")
    print("=" * 100)
    # C MECHANISM (fresh nets)
    net = build_time_net(); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    tgt_fn = lambda pts: 0.01 * torch.sin(pts * 20)             # synthetic smooth dx*
    pts = torch.rand(512, 3) * 0.1; t = torch.full((512, 1), 5.0)
    h = torch.cat([freq_encode(t), freq_encode(pts)], dim=-1)
    l0 = None
    for i in range(300):
        opt.zero_grad(); loss = ((net(h) - tgt_fn(pts)) ** 2).mean(); loss.backward(); opt.step()
        if i == 0: l0 = float(loss)
    print(f"C-i  gradient-flow : synthetic-dx* loss {l0:.2e} -> {float(loss):.2e} "
          f"({'LEARNS' if float(loss) < 0.1 * l0 else 'FAILS'})")
    net2 = build_time_net(); opt2 = torch.optim.Adam(net2.parameters(), lr=1e-3, weight_decay=1e-6)
    f0 = float(net2(h).norm(dim=-1).mean())
    for i in range(3000):                                       # zero-mean noise grads = render-loss proxy
        opt2.zero_grad()
        out = net2(h); (out * torch.randn_like(out) * 1e-3).sum().backward(); opt2.step()
    f1 = float(net2(h).norm(dim=-1).mean())
    print(f"C-ii starvation    : |dx| {f0:.4f} -> {f1:.4f} under wd=1e-6 + zero-mean grads "
          f"({'DECAYS' if f1 < 0.5 * f0 else 'PERSISTS'} after 3k steps)")


if __name__ == '__main__':
    main()
