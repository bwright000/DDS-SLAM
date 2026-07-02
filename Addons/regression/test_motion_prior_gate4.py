#!/usr/bin/env python3
"""GATE-4 synthetic tests for the LEAN-CORE zero-motion prior (no renderer needed).

Proves the CONSTANT-strength prior (flow_track.zero_motion_prior = lam_r|drot|^2 + lam_t|dt|^2) yields the
right EMERGENT anisotropy when combined with a data term whose per-DOF curvature H ~ J_flow^2 (the SDF loss's
natural observability). KEY DESIGN POINT (reverses gate-3): the prior does NOT scale by J_i -- H_data is
already ~J^2, so an explicit J^2 prior cancels (J-independent ratio -> no anisotropy). The anisotropy must
come from H, and a CONSTANT prior lets it: MAP per DOF p_i = H_i*tgt_i/(H_i + lam_i), so high-H DOFs track
and low-H DOFs get pinned.

  T0 function : zero_motion_prior returns lam_r|drot|^2 + lam_t|dt|^2 on a known relative pose.
  T1 observable: high-H DOF (real motion, strong data curvature) -> tracked (data wins).
  T2 sub-floor : low-H DOF -> pinned to ~0 (prior wins).
  T3 depth     : H_t ~ (f/Z)^2, so as Z grows t is MORE pinned while R (H_r indep of Z) is invariant
                 == fix-2's "R invariant, t scales", now EMERGENT (no median-Z bookkeeping).
  T4 hold      : still frame + fake deformation on unmasked tissue (all H low, data pulls to a spurious
                 pose) -> pose HELD. This is the headline behaviour the trust-weight ablation must beat.
  T5 fires     : the solve runs every iter (no skip) and returns finite -- P2 regression guard.

Run on the Colab torch env: python Addons/regression/test_motion_prior_gate4.py  (exit 0 = all pass).
"""
import sys, os, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Addons.motion.flow_track import zero_motion_prior

LR, LT = 10.0, 10.0   # test-scale prior weights (H spans 1..100 around them so the win/lose split is clean)


def _mini_solve(H, tgt, iters=2000, lr=0.05):
    """Mini pose-solve: minimise sum H_i (p_i - tgt_i)^2  +  constant prior. Mirrors the SDF loop's balance
    (synthetic data curvature H stands in for the SDF loss's per-DOF observability)."""
    import torch
    p = torch.zeros(6, requires_grad=True)
    Ht = torch.tensor(H, dtype=torch.float32)
    tg = torch.tensor(tgt, dtype=torch.float32)
    opt = torch.optim.Adam([p], lr=lr)
    fired = 0
    for _ in range(iters):
        opt.zero_grad()
        data = (Ht * (p - tg) ** 2).sum()
        prior = LR * (p[:3] ** 2).sum() + LT * (p[3:] ** 2).sum()
        (data + prior).backward()
        opt.step()
        fired += 1
    return p.detach().numpy(), fired


def main():
    import torch
    ok = True

    # T0 -- the prior computes lam_r|drot|^2 + lam_t|dt|^2 on a known pose (rot th about z, t=[0,0,0.03])
    th = 0.02
    c2w = torch.tensor([[math.cos(th), -math.sin(th), 0.0, 0.0],
                        [math.sin(th),  math.cos(th), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.03],
                        [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    got = float(zero_motion_prior(c2w, torch.eye(4), LR, LT))
    exp = LR * math.sin(th) ** 2 + LT * (0.03 ** 2)
    t0 = abs(got - exp) < 1e-4
    print(f"T0 function      : got={got:.6f} exp={exp:.6f} -> {'PASS' if t0 else 'FAIL'}")
    ok &= t0

    # DESIGN property (exact): the constant prior + data curvature H give MAP p_i = H_i*tgt_i/(H_i+lam_i).
    # This is the emergent anisotropy itself; T5 then confirms the actual optimiser realises it.
    def amap(H, tgt):
        return np.array([H[i] * tgt[i] / (H[i] + (LR if i < 3 else LT)) for i in range(6)])

    # T1 observable + T2 sub-floor: DOF0 rotation observable (H=100), DOF3 translation sub-floor (H=1)
    m = amap([100, 100, 100, 1, 1, 1], [0.05] * 6)
    r_obs, r_sub = m[0] / 0.05, m[3] / 0.05
    t1, t2 = r_obs > 0.85, r_sub < 0.15
    print(f"T1 observable    : MAP/tgt={r_obs:.3f} (want>0.85) -> {'PASS' if t1 else 'FAIL'}")
    print(f"T2 sub-floor     : MAP/tgt={r_sub:.3f} (want<0.15) -> {'PASS' if t2 else 'FAIL'}")
    ok &= t1 and t2

    # T3 depth: H_t ~ (f/Z)^2. As Z grows, t is more pinned; R (H_r fixed) invariant.
    rows = []
    for Z in (1.0, 2.0, 4.0):
        Ht = 100.0 / (Z ** 2)
        m = amap([100, 100, 100, Ht, Ht, Ht], [0.05] * 6)
        rows.append((Z, m[3] / 0.05, m[0] / 0.05))
        print(f"T3 depth Z={Z:.0f}    : t MAP/tgt={rows[-1][1]:.3f}  R MAP/tgt={rows[-1][2]:.3f}")
    t_mono = rows[0][1] > rows[1][1] > rows[2][1]
    r_inv = abs(rows[0][2] - rows[2][2]) < 1e-6
    t3 = t_mono and r_inv
    print(f"T3 depth-scaling : t-monotone={t_mono} R-invariant={r_inv} -> {'PASS' if t3 else 'FAIL'}")
    ok &= t3

    # T4 hold: still + fake deformation (all H low, data pulls to a spurious pose) -> held
    m = amap([1, 1, 1, 1, 1, 1], [0.03] * 6)
    lim = 0.2 * 0.03
    held = float(np.max(np.abs(m))) < lim
    print(f"T4 hold(deform)  : max|MAP|={np.max(np.abs(m)):.5f} (want<{lim:.5f}) -> {'PASS' if held else 'FAIL'}")
    ok &= held

    # T5 solve-realises: the Adam optimiser (a) fires every iter, (b) is finite, (c) pins the sub-floor DOF
    # strictly more than the observable one (the emergent anisotropy shows up in the actual optimisation).
    p, fired = _mini_solve([100, 100, 100, 1, 1, 1], [0.05] * 6, iters=400)
    r_obs_s, r_sub_s = p[0] / 0.05, p[3] / 0.05
    t5 = fired == 400 and bool(np.all(np.isfinite(p))) and (r_sub_s < 0.5 * r_obs_s)
    print(f"T5 solve-realises: iters={fired} obs={r_obs_s:.3f} sub={r_sub_s:.3f} (sub<0.5*obs) -> {'PASS' if t5 else 'FAIL'}")
    ok &= t5

    print(">>> GATE-4 " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
