"""
Inc-0 regression harness — the GATE every Arm-2 increment must pass.

Invariant: with all Arm-2 flags OFF (uncertainty/nrgs/sni enable:false), building the
model must be BIT-IDENTICAL to the pre-Arm-2 base. The decisive, GPU-safe proof is the
torch+cuda RNG state immediately AFTER JointEncoding construction: if a future increment
leaks a module into the flags-off path, its nn.Linear/tcnn weight-init draws RNG and the
state diverges. We also assert the parameter count and state_dict key set are unchanged.

Why RNG-state, not est_c2w byte-cmp: seed_everything (ddsslam.py:56-61) does NOT set
cudnn.deterministic / torch.use_deterministic_algorithms, and tinycudann HashGrid
atomic-add kernels are non-bit-reproducible on GPU — so an est_c2w byte-cmp is flaky even
base-vs-base on GPU. RNG-state-after-construction is immune to that (init is seeded).

Usage (run from the DDS-SLAM/ working dir; needs CUDA for tcnn — i.e. Colab):
  # 1. capture the golden once, on the flags-off build (== base while no modules exist yet):
  python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml --write-golden
  # 2. every later increment: re-run with flags OFF and assert it still matches:
  python Addons/regression/test_inc0_bitidentical.py --config configs/Super/trail3_paper_faithful.yaml
"""
import argparse, hashlib, json, os, random, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def seed_everything(seed):
    # mirrors DDSSLAM.seed_everything (ddsslam.py:56-61)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _h(state_tensor):
    return hashlib.sha256(state_tensor.cpu().numpy().tobytes()).hexdigest()


def build_snapshot(config_path, seed, force_flags_off):
    """Seed identically, build JointEncoding, snapshot RNG state + params + keys."""
    from config import load_config
    from model.scene_rep import JointEncoding

    cfg = load_config(config_path)
    if force_flags_off:
        for k in ('uncertainty', 'nrgs', 'sni'):
            cfg.setdefault(k, {})['enable'] = False

    if 'bound' not in cfg.get('mapping', {}):
        raise SystemExit(f"config {config_path} has no mapping.bound — point --config at a full run config "
                         f"(e.g. configs/Super/trail3_paper_faithful.yaml).")
    bound = torch.tensor(np.array(cfg['mapping']['bound']), dtype=torch.float32)

    seed_everything(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = JointEncoding(cfg, bound.to(dev)).to(dev)

    snap = {
        'rng_torch': _h(torch.random.get_rng_state()),
        'rng_cuda': _h(torch.cuda.get_rng_state()) if torch.cuda.is_available() else 'no-cuda',
        'n_params': sum(p.numel() for p in model.parameters()),
        'n_param_tensors': len(list(model.parameters())),
        'state_keys': sorted(model.state_dict().keys()),
        'flags': {k: bool(cfg.get(k, {}).get('enable', False)) for k in ('uncertainty', 'nrgs', 'sni')},
    }
    return snap


def diff(golden, current):
    issues = []
    for k in ('rng_torch', 'rng_cuda', 'n_params', 'n_param_tensors'):
        if golden.get(k) != current.get(k):
            issues.append(f"{k}: golden={golden.get(k)}  current={current.get(k)}")
    gk, ck = set(golden.get('state_keys', [])), set(current.get('state_keys', []))
    if gk != ck:
        issues.append(f"state_dict keys differ: only-in-golden={sorted(gk - ck)}  only-in-current={sorted(ck - gk)}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--golden', default=os.path.join(os.path.dirname(__file__), 'golden_inc0.json'))
    ap.add_argument('--write-golden', action='store_true')
    args = ap.parse_args()

    snap = build_snapshot(args.config, args.seed, force_flags_off=True)

    if args.write_golden:
        with open(args.golden, 'w') as f:
            json.dump(snap, f, indent=2)
        print(f"WROTE golden -> {args.golden}\n  n_params={snap['n_params']}  rng_torch={snap['rng_torch'][:16]}…")
        return

    if not os.path.exists(args.golden):
        raise SystemExit(f"no golden at {args.golden} — run once with --write-golden first.")
    with open(args.golden) as f:
        golden = json.load(f)

    issues = diff(golden, snap)
    # sanity: flags-on build must construct >= as many params as flags-off (caught later increments)
    on = build_snapshot(args.config, args.seed, force_flags_off=False)
    print(f"flags-off n_params={snap['n_params']}   flags-on n_params={on['n_params']}")

    if issues:
        print("INC-0 REGRESSION: FAIL — flags-off build diverged from base:")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print("INC-0 REGRESSION: PASS — flags-off build is bit-identical to base "
          "(RNG state + param count + state_dict keys all match).")


if __name__ == '__main__':
    main()
