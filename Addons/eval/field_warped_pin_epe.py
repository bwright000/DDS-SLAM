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

    rigid, field, dxm = [], [], []; shuf_field = []
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
        rigid += list(r); field += list(np.linalg.norm(Xk[val]+D-X0[val], axis=1))
        shuf_field += list(np.linalg.norm(Xk[val]+Ds-X0[val], axis=1)); dxm += list(np.linalg.norm(D, axis=1))
    rigid, field, shuf_field, dxm = map(np.array, (rigid, field, shuf_field, dxm))
    # anchor check: D at t=0 on the frame-0 pins must be ~0
    anc = float(np.linalg.norm(field_D(torch.tensor(X0[g0[:,2]==1], dtype=torch.float32, device=dev), 0.0).cpu().numpy(), axis=1).max())

    print(f"\n=== FIELD-WARPED PIN EPE (n={len(rigid)} pin-observations, world units) ===")
    print(f"  rigid (no field)   : {rigid.mean():.5f}")
    print(f"  field-warped       : {field.mean():.5f}   reduction = {100*(rigid.mean()-field.mean())/max(rigid.mean(),1e-9):+.1f}%")
    print(f"  shuffled-time ctrl : {shuf_field.mean():.5f}   reduction = {100*(rigid.mean()-shuf_field.mean())/max(rigid.mean(),1e-9):+.1f}%")
    print(f"  |Δx| field activity: mean={dxm.mean():.5f} max={dxm.max():.5f}  (0 => dead field)")
    print(f"  anchor check D@t=0 : {anc:.2e}  (must be ~0)")
    print("\nVERDICT: field-warped reduction >> shuffled AND |Δx|>0  -> field models deformation (ALIVE).")
    print("         reduction ~= shuffled, or ~0, or |Δx|~0          -> field INERT (motion-teacher needed).")


if __name__ == '__main__':
    main()
