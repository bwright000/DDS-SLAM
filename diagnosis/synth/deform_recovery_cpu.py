"""
CPU deformation-recovery benchmark — "does the redesigned field WORK given a signal?"

The full-SLAM failure is UNDERDETERMINATION (gauge ambiguity): photometric loss alone can't pin Δx,
so the field dies / coin-flips. This isolates the OTHER question — independent of the gauge problem:
can the deformation field (DDS-SLAM's TimeNet: bias-free-ReLU MLP, freq-encoded x,t) REPRESENT and
RECOVER a realistic tissue deformation, AND do our redesign constraints (hard-bound + function-space
leash + fixed seed + frame-0 anchor) PRESERVE that recovery rather than break it (battery-6 showed a
hard cap ALONE saturates)?

This is the foundation: if the field can't recover a known deformation even WITH a direct signal, no
amount of SLAM-side information will help. If it CAN (and the constraints don't break it), then the
remaining problem is purely supplying the signal in-SLAM (tracks / dual-hypothesis / synthetic GT).

Runs fully on CPU (TimeNet is tiny; freq encoders are pure torch). No TCNN, no GPU, no dataset.
"""
import argparse, math, json
import numpy as np
import torch
import torch.nn as nn


def freq_encode(p, n_freq):
    # NeRF-style: [p, sin(2^k pi p), cos(2^k pi p) for k in 0..n_freq-1]
    out = [p]
    for k in range(n_freq):
        f = (2.0 ** k) * math.pi
        out.append(torch.sin(f * p)); out.append(torch.cos(f * p))
    return torch.cat(out, dim=-1)


class TimeNet(nn.Module):
    # matches DDS-SLAM decoder.py: bias-free Linear + ReLU, hidden 64, 3 layers, out 3
    def __init__(self, in_ch, hidden=64, layers=3):
        super().__init__()
        m = []
        for l in range(layers):
            i = in_ch if l == 0 else hidden
            o = 3 if l == layers - 1 else hidden
            m.append(nn.Linear(i, o, bias=False))
            if l != layers - 1:
                m.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*m)
    def forward(self, x):
        return self.model(x)


def D_gt(x, t):
    """Known physical tissue deformation: TWO localised bumps that grow over time, pushing
    TOWARD the camera (+z here), a couple mm in model units. Zero at t=0 (anchor-consistent).
    x: (...,3) canonical points (scene z~[0.7,1.2], xy~[-0.5,0.5]); t: (...,1) in [0,1]."""
    c1 = torch.tensor([0.20, 0.10, 0.95]); c2 = torch.tensor([-0.25, -0.15, 1.00])
    s1, s2 = 0.12, 0.10
    amp = 0.03  # ~ a couple mm in model units
    b1 = torch.exp(-((x - c1) ** 2).sum(-1, keepdim=True) / (2 * s1 ** 2))
    b2 = torch.exp(-((x - c2) ** 2).sum(-1, keepdim=True) / (2 * s2 ** 2))
    dir1 = torch.tensor([0.1, 0.0, 1.0]); dir1 = dir1 / dir1.norm()
    dir2 = torch.tensor([-0.2, 0.3, 0.9]); dir2 = dir2 / dir2.norm()
    return t * amp * (b1 * dir1 + 1.2 * b2 * dir2)


def sample(n, device):
    x = torch.rand(n, 3, device=device)
    x[:, 0] = x[:, 0] - 0.5; x[:, 1] = x[:, 1] - 0.5; x[:, 2] = 0.7 + 0.5 * x[:, 2]  # scene box
    t = torch.rand(n, 1, device=device)
    return x, t


def run(hardbound, reg, seed=0, iters=3000, n=8192, Lx=10, Lt=4, lr=1e-3, lr_mult=1.0):
    torch.manual_seed(seed); np.random.seed(seed)  # the seed_everything fix, applied
    dev = torch.device('cpu')
    in_ch = (3 + 3 * 2 * Lx) + (1 + 1 * 2 * Lt)
    net = TimeNet(in_ch).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr * lr_mult)
    for it in range(iters):
        x, t = sample(n, dev)
        gt = D_gt(x, t)
        h = torch.cat([freq_encode(t, Lt), freq_encode(x, Lx)], -1)
        raw = net(h)
        if hardbound and hardbound > 0:
            dx = hardbound * torch.tanh(raw / hardbound)
        else:
            dx = raw
        # frame-0 anchor (matches scene_rep): Δx=0 at t=0
        dx = torch.where(t == 0, torch.zeros_like(dx), dx)
        loss = ((dx - gt) ** 2).mean() + reg * (dx ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    # eval on a fresh held-out set
    with torch.no_grad():
        x, t = sample(16384, dev); gt = D_gt(x, t)
        h = torch.cat([freq_encode(t, Lt), freq_encode(x, Lx)], -1)
        raw = net(h)
        dx = hardbound * torch.tanh(raw / hardbound) if (hardbound and hardbound > 0) else raw
        dx = torch.where(t == 0, torch.zeros_like(dx), dx)
        epe = (dx - gt).norm(dim=-1)
        gtn = gt.norm(dim=-1)
        rel = (epe.mean() / (gtn.mean() + 1e-9)).item()
        # structure: correlation of predicted vs true magnitude (is the bump in the right place?)
        pm, gm = dx.norm(dim=-1).cpu().numpy(), gtn.cpu().numpy()
        corr = float(np.corrcoef(pm, gm)[0, 1]) if pm.std() > 1e-9 and gm.std() > 1e-9 else 0.0
        return dict(epe_mm_units=round(epe.mean().item(), 5), rel_epe=round(rel, 4),
                    max_pred=round(dx.norm(dim=-1).max().item(), 4),
                    mean_gt=round(gtn.mean().item(), 5), struct_corr=round(corr, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    # The redesign + an ablation proving each constraint preserves recovery (not break it like b6 saturation)
    configs = [
        ('redesign (bound 0.04 + reg 1e-4 + seed)', dict(hardbound=0.04, reg=1e-4)),
        ('no-bound (reg only)',                     dict(hardbound=0.0,  reg=1e-4)),
        ('bound only (no reg)',                     dict(hardbound=0.04, reg=0.0)),
        ('bound TOO TIGHT 0.005 (saturation test)', dict(hardbound=0.005, reg=1e-4)),
        ('reg TOO STRONG 1e-1 (over-damp test)',    dict(hardbound=0.04, reg=1e-1)),
    ]
    results = {}
    print(f"{'config':<42}{'rel_epe':>9}{'epe':>9}{'max':>8}{'struct':>8}  verdict")
    for name, kw in configs:
        r = run(**kw)
        # verdict: recovered if rel_epe small + structure high
        ok = r['rel_epe'] < 0.4 and r['struct_corr'] > 0.9   # struct-primary: deformation in the RIGHT place
        r['recovered'] = ok
        results[name] = r
        print(f"{name:<42}{r['rel_epe']:>9}{r['epe_mm_units']:>9}{r['max_pred']:>8}{r['struct_corr']:>8}  {'RECOVERED' if ok else 'no'}")
    # reproducibility check: same config, two seeds -> should match (vs the SLAM coin-flip)
    a = run(hardbound=0.04, reg=1e-4, seed=0); b = run(hardbound=0.04, reg=1e-4, seed=1)
    repro = abs(a['rel_epe'] - b['rel_epe'])
    print(f"\nreproducibility (rel_epe seed0 vs seed1): {a['rel_epe']} vs {b['rel_epe']}  (abs_diff={repro:.4f})")
    results['_reproducibility'] = dict(seed0=a['rel_epe'], seed1=b['rel_epe'], abs_diff=round(repro, 4))
    if args.json:
        json.dump(results, open(args.json, 'w'), indent=2)
    print("\nREAD: 'redesign' should RECOVER (rel_epe small, struct~1). 'TOO TIGHT'/'TOO STRONG' should FAIL")
    print("(shows the constraints must be set right, like battery-6's saturation). Reproducible across seeds")
    print("=> the FIELD works given a signal; the SLAM failure is the GAUGE/signal problem, not the field.")


if __name__ == '__main__':
    main()
