#!/usr/bin/env python3
"""Field-WARPED green-pin EPE — the ONLY field-SENSITIVE metric (Arm-2 decisive verdict).

THE QUESTION: does DDS-SLAM's deformation field D model the tissue deformation? Render + Sim3 ATE are
both proven FIELD-BLIND (field_off ~= field_on), so they can't answer it. The green pins ARE the
deformation GT. This warps them through the field and asks whether D reduces the pin error.

CONVENTION (the load-bearing correctness, grounded in code, NO inverse so nothing to get backwards):
  * field forward = scene_rep.py:205-218 exactly, via the LOADED model's own embed_time/embed_fre_pos/
    time_net (+ deform_hardbound + the where(t==0,0,·) anchor). D maps observed->canonical: pts+D = canonical.
  * pins back-projected in the field's OWN world frame (OpenGL rays, datasets/utils.py:50):
    X = c2w_t + d * (c2w_R @ [(u-cx)/fx, -(v-cy)/fy, -1]).
  * time t_k = k/num_frames if time_normalize else k (ddsslam.py:285), read from the config.

THE METRIC: a pin observed at frame k is X_k; its canonical (frame-0) anchor is X0. If the field is
right, X_k + D(X_k, t_k) == X0 (same physical point -> same canonical). So:
  rigid_epe = mean||X_k - X0||           (no field; = the raw deformation)
  field_epe = mean||(X_k + D) - X0||      (field-corrected)
  reduction = (rigid - field)/rigid      ->  >0: field explains deformation;  ~0: field INERT.
field_off (D=0) is the control (reduction must be ~0). THREE guards: |Δx| activity (convention-robust:
is the field even non-zero?), anchor check (D@t=0 must be ~0), shuffled-time control (if scrambled t
gives the same reduction, it's overfitting/gaming, not real motion).

  python Addons/eval/field_warped_pin_epe.py --config configs/Super/trail3_field_on.yaml \
    --checkpoint <field_on>/checkpoint.pt --est_c2w <field_on>/est_c2w_data.txt \
    --pts Addons/eval/gt_pins/trial_3_l_pts.npy \
    --depth_dir data/Super/trail_3/depth/moge2 --depth_glob '*left_depth.npy'
"""
import os, sys, glob, argparse, numpy as np, torch
sys.path.insert(0, os.getcwd())


def load_config(p):
    import yaml
    d = yaml.safe_load(open(p)); b = load_config(d['inherit_from']) if 'inherit_from' in d else {}
    def m(a, b):
        o = dict(a)
        for k, v in b.items(): o[k] = m(o[k], v) if isinstance(v, dict) and isinstance(o.get(k), dict) else v
        return o
    return m(b, d)


def load_c2w(p):
    P = []
    for ln in open(p):
        v = ln.split()
        if len(v) >= 12 and not v[0].startswith('#'):
            T = np.eye(4); T[:3, :4] = np.array(list(map(float, v[:12]))).reshape(3, 4); P.append(T)
    return P


def load_pts(p):
    d = np.load(p, allow_pickle=True); d = d.item() if hasattr(d, 'item') else d
    gt = d['gt'] if isinstance(d, dict) and 'gt' in d else d
    return {int(k): np.asarray(v, np.float64) for k, v in gt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--est_c2w', required=True); ap.add_argument('--pts', required=True)
    ap.add_argument('--depth_dir', required=True); ap.add_argument('--depth_glob', default='*left_depth.npy')
    ap.add_argument('--fx', type=float, default=768.98551924); ap.add_argument('--fy', type=float, default=768.98551924)
    ap.add_argument('--cx', type=float, default=292.8861567); ap.add_argument('--cy', type=float, default=291.61479526)
    ap.add_argument('--ray', default='OpenGL', choices=['OpenGL', 'OpenCV'])
    ap.add_argument('--fig_dir', default='', help='where to write the diagnostic PNG; default = the est_c2w run dir (ships to Drive alongside the payload, per the standing visuals rule)')
    ap.add_argument('--tag', default='', help='figure label/filename stem; default = the run-dir name')
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.config)
    pds = cfg['cam']['png_depth_scale']
    tnorm = cfg['training'].get('time_normalize', False)
    hb = cfg.get('deform_hardbound', 0); anchor_off = cfg.get('deformation_anchor_off', False)

    from model.scene_rep import JointEncoding
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32).to(dev)
    model = JointEncoding(cfg, bound).to(dev).eval()
    ck = torch.load(args.checkpoint, map_location=dev)
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)

    @torch.no_grad()
    def field_D(pts, t):                      # pts [N,3] world (field frame), t scalar
        ft = torch.full((pts.shape[0], 1), float(t), device=dev)
        h = torch.cat([model.embed_time(ft), model.embed_fre_pos(pts)], -1)
        vox = model.time_net(h).float()
        if hb and hb > 0: vox = hb * torch.tanh(vox / hb)
        if not anchor_off: vox = torch.where(ft == 0, torch.zeros_like(vox), vox)
        return vox                            # D = observed->canonical displacement

    poses = load_c2w(args.est_c2w)
    pins = load_pts(args.pts)
    deps = sorted(glob.glob(os.path.join(args.depth_dir, args.depth_glob)))
    N = min(len(poses), len(deps)); nf = max(pins) + 1
    print(f"poses {len(poses)} depth {len(deps)} pin-frames {len(pins)} | time_normalize={tnorm} pds={pds} ray={args.ray} hardbound={hb}")

    def backproj(uv, d, T):                    # uv [P,2] pixels, d HxW depth(m), T 4x4 c2w -> world [P,3]
        u, v = uv[:, 0], uv[:, 1]
        z = d[np.clip(v.astype(int), 0, d.shape[0]-1), np.clip(u.astype(int), 0, d.shape[1]-1)]
        sy = -1.0 if args.ray == 'OpenGL' else 1.0; sz = -1.0 if args.ray == 'OpenGL' else 1.0
        cam = np.stack([(u-args.cx)/args.fx, sy*(v-args.cy)/args.fy, sz*np.ones_like(u)], -1) * z[:, None]
        return (T[:3, :3] @ cam.T).T + T[:3, 3], z

    g0 = pins[0]; d0 = np.load(deps[0]).astype(np.float64).squeeze()/pds
    X0, z0 = backproj(g0[:, :2], d0, poses[0])

    rigid, field, dxm, cosd = [], [], [], []; shuf_field = []; fids = []
    rng_t = [k/nf if tnorm else k for k in sorted(pins) if k != 0]
    perm = np.random.permutation(rng_t)              # shuffled-time control
    for idx, k in enumerate([k for k in sorted(pins) if k != 0]):
        if k >= N: continue
        gk = pins[k]; dk = np.load(deps[k]).astype(np.float64).squeeze()/pds
        Xk, zk = backproj(gk[:, :2], dk, poses[k])
        val = (g0[:, 2] == 1) & (gk[:, 2] == 1) & (z0 > 1e-3) & (zk > 1e-3)
        if val.sum() == 0: continue
        Xk_t = torch.tensor(Xk[val], dtype=torch.float32, device=dev)
        tk = k/nf if tnorm else k
        D = field_D(Xk_t, tk).cpu().numpy()
        Ds = field_D(Xk_t, float(perm[idx])).cpu().numpy()
        r = np.linalg.norm(Xk[val]-X0[val], axis=1)
        gt_dir = X0[val] - Xk[val]                    # the displacement the field SHOULD undo
        cosd += list((D*gt_dir).sum(1) / (np.linalg.norm(D,axis=1)*np.linalg.norm(gt_dir,axis=1) + 1e-12))
        rigid += list(r); field += list(np.linalg.norm(Xk[val]+D-X0[val], axis=1))
        shuf_field += list(np.linalg.norm(Xk[val]+Ds-X0[val], axis=1)); dxm += list(np.linalg.norm(D, axis=1))
        fids += [k] * int(val.sum())
    rigid, field, shuf_field, dxm, cosd, fids = map(np.array, (rigid, field, shuf_field, dxm, cosd, fids))
    # anchor check: D at t=0 on the frame-0 pins must be ~0
    anc = float(np.linalg.norm(field_D(torch.tensor(X0[g0[:,2]==1], dtype=torch.float32, device=dev), 0.0).cpu().numpy(), axis=1).max())

    print(f"\n=== FIELD-WARPED PIN EPE (n={len(rigid)} pin-observations, world units) ===")
    print(f"  rigid (no field)   : {rigid.mean():.5f}")
    print(f"  field-warped       : {field.mean():.5f}   reduction = {100*(rigid.mean()-field.mean())/max(rigid.mean(),1e-9):+.1f}%")
    print(f"  shuffled-time ctrl : {shuf_field.mean():.5f}   reduction = {100*(rigid.mean()-shuf_field.mean())/max(rigid.mean(),1e-9):+.1f}%")
    print(f"  |Δx| field activity: mean={dxm.mean():.5f} max={dxm.max():.5f}  (0 => dead field)")
    cm = float(np.nanmean(cosd)) if len(cosd) else float('nan')
    print(f"  cos(D, X0-Xk) dir  : {cm:+.3f}  (>0 toward canonical; <=0 wrong-way/HOLLOW even if reduction looks ok)")
    print(f"  anchor check D@t=0 : {anc:.2e}  (must be ~0)")
    print("\nVERDICT: reduction >> shuffled AND |Δx|>0 AND cos>0  -> field models deformation (ALIVE).")
    print("         reduction ~= shuffled, or ~0, or |Δx|~0, or cos<=0  -> field INERT/HOLLOW (motion-teacher needed).")

    # ---- progress visualisation (standing rule 2026-06-18: GPU/judge runs ship visuals to Drive, co-located) ----
    fig_dir = args.fig_dir or os.path.dirname(os.path.abspath(args.est_c2w))
    tag = args.tag or (os.path.basename(os.path.dirname(os.path.abspath(args.est_c2w))) or 'field_pin_epe')
    if len(rigid) == 0:
        print("[viz] SKIPPED: no valid pin observations")
    else:
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            os.makedirs(fig_dir, exist_ok=True)
            red = 100*(rigid.mean()-field.mean())/max(rigid.mean(), 1e-9)
            sred = 100*(rigid.mean()-shuf_field.mean())/max(rigid.mean(), 1e-9)
            verdict = 'ALIVE' if (red > sred + 5 and dxm.mean() > 0 and cm > 0) else ('HOLLOW' if (dxm.mean() > 0 and cm <= 0) else 'INERT')
            fig, ax = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle(f'Field-warped pin EPE  —  {tag}   [{verdict}]', fontsize=14, weight='bold')
            a = ax[0, 0]; vals = [rigid.mean(), field.mean(), shuf_field.mean()]
            a.bar(['rigid', 'field', 'shuffled'], vals, color=['#888', '#2a8', '#c66'])
            a.set_title(f'mean pin-EPE (world u)\nfield {red:+.1f}%  vs shuffled {sred:+.1f}%'); a.set_ylabel('EPE')
            for i, v in enumerate(vals): a.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
            a = ax[0, 1]; a.hist(dxm, bins=40, color='#48a'); a.axvline(dxm.mean(), color='k', ls='--', label=f'mean {dxm.mean():.4f}')
            a.set_title(f'|Δx| field activity (max {dxm.max():.4f})\n0 => DEAD field'); a.set_xlabel('|Δx| world u'); a.legend()
            a = ax[0, 2]; a.hist(cosd, bins=40, range=(-1, 1), color='#8a4'); a.axvline(0, color='r'); a.axvline(cm, color='k', ls='--', label=f'mean {cm:+.3f}')
            a.set_title('cos(D, X0-Xk)  >0 correct\n<=0 => HOLLOW'); a.set_xlabel('cos'); a.legend()
            a = ax[1, 0]; mx = float(max(rigid.max(), field.max())) * 1.05; imp = field < rigid
            a.scatter(rigid[imp], field[imp], s=5, alpha=0.3, color='#2a8', label=f'improved {imp.mean()*100:.0f}%')
            a.scatter(rigid[~imp], field[~imp], s=5, alpha=0.3, color='#c66')
            a.plot([0, mx], [0, mx], 'k--', lw=1); a.set_xlim(0, mx); a.set_ylim(0, mx)
            a.set_title('per-pin: field vs rigid EPE\nbelow y=x = improved'); a.set_xlabel('rigid'); a.set_ylabel('field'); a.legend()
            a = ax[1, 1]; ks = np.unique(fids)
            pf = [100*(rigid[fids == kk].mean()-field[fids == kk].mean())/max(rigid[fids == kk].mean(), 1e-9) for kk in ks]
            a.plot(ks, pf, '-o', ms=3, color='#2a8'); a.axhline(0, color='r'); a.set_title('per-frame field reduction %'); a.set_xlabel('frame k'); a.set_ylabel('reduction %')
            a = ax[1, 2]; a.axis('off')
            txt = (f'n obs        : {len(rigid)}\nrigid EPE    : {rigid.mean():.5f}\nfield EPE    : {field.mean():.5f}\n'
                   f'reduction    : {red:+.1f}%\nshuffled red : {sred:+.1f}%\n|Δx| mean/max: {dxm.mean():.5f}/{dxm.max():.5f}\n'
                   f'cos mean     : {cm:+.3f}\nanchor D@t=0 : {anc:.2e}\n\nVERDICT: {verdict}\n(ALIVE: red>>shuf & |Δx|>0 & cos>0)')
            a.text(0.02, 0.98, txt, va='top', ha='left', family='monospace', fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out = os.path.join(fig_dir, f'{tag}_field_pin_epe.png'); fig.savefig(out, dpi=110); plt.close(fig)
            print(f"\n[viz] saved {out}")
        except Exception as e:
            print(f"\n[viz] SKIPPED ({type(e).__name__}: {e}) — metrics above are unaffected")


if __name__ == '__main__':
    main()
