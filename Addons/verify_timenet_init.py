"""Probe TimeNet's output at initialization for representative frame indices.

What this shows
---------------
- Builds a fresh TimeNet identical to DDS-SLAM's (4 layers, 128 hidden, bias=False).
- Feeds it the same (time-encoding, position-encoding) tensor produced by
  manual Fourier encoding (matching tcnn's formula: sin/cos(2^L * pi * x)).
- Reports the magnitude of vox_motion at different frame indices, both for
  raw-integer t (current code) and normalised t/n_imgs (proposed fix).

Run:
    python Addons/verify_timenet_init.py
Pure PyTorch CPU, no tcnn or CUDA required.
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)

N_FREQ = 12
HIDDEN = 128
NUM_LAYERS = 4


def manual_freq_encode(x: torch.Tensor, n_frequencies: int = N_FREQ) -> torch.Tensor:
    """Match tcnn Frequency: out[2L]=sin(2^L pi x), out[2L+1]=cos(2^L pi x)."""
    assert x.dim() == 2
    D = x.shape[1]
    out = torch.zeros(x.shape[0], 2 * n_frequencies * D)
    for d in range(D):
        for L in range(n_frequencies):
            freq = (2.0 ** L) * np.pi
            out[:, 2 * (L + d * n_frequencies)] = torch.sin(freq * x[:, d])
            out[:, 2 * (L + d * n_frequencies) + 1] = torch.cos(freq * x[:, d])
    return out


def make_timenet(in_dim: int) -> nn.Sequential:
    layers = []
    dims = [in_dim] + [HIDDEN] * (NUM_LAYERS - 1) + [3]
    for i in range(NUM_LAYERS):
        layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i != NUM_LAYERS - 1:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def probe(label: str, times: torch.Tensor, n_pts: int = 100) -> None:
    """
    For each time value, sample n_pts random 3D positions in a [-1,1] box and
    compute TimeNet's vox_motion. Report per-time mean magnitude of the offset.
    """
    torch.manual_seed(42)  # same positions across runs
    positions = torch.rand(n_pts, 3) * 2.0 - 1.0  # [-1,1]^3

    # Fresh TimeNet (re-seeded so both conditions see the same init weights)
    torch.manual_seed(0)
    time_input_dim = 2 * N_FREQ  # 24
    pos_input_dim = 2 * N_FREQ * 3  # 72
    net = make_timenet(time_input_dim + pos_input_dim)
    net.eval()

    print(f"\n--- {label} ---")
    print(f"{'t_value':>10s} | mean |vox_motion| (m) | max |vox_motion| (m)")
    print("-" * 60)

    embed_pos = manual_freq_encode(positions)  # (n_pts, 72)

    with torch.no_grad():
        for t in times:
            t_col = t.expand(n_pts, 1)
            embed_time = manual_freq_encode(t_col)  # (n_pts, 24)
            h = torch.cat([embed_time, embed_pos], dim=-1)  # (n_pts, 96)
            vox_motion = net(h)  # (n_pts, 3)
            mag = torch.linalg.norm(vox_motion, dim=-1)  # (n_pts,)
            print(f"{float(t):>10.4f} | {mag.mean().item():>20.4f} | {mag.max().item():>20.4f}")


def main() -> None:
    print("Scene bounds in P2_1 config are roughly ±3 metres on any axis.")
    print("For reference, paper ATE is 8.3 mm. Any vox_motion > 10 cm is")
    print("larger than the target tracking precision.\n")

    # 1) Raw integer frame indices (current code).
    raw = torch.tensor([[0.0], [1.0], [2.0], [100.0], [1000.0], [3999.0]])
    probe("RAW integer t (current DDS-SLAM code)", raw)

    # 2) Normalised t (proposed fix).
    norm = raw / 4000.0
    probe("NORMALISED t / 4000 (proposed fix)", norm)

    print("\n\n=== Interpretation ===")
    print("If raw-integer rows show similar vox_motion magnitudes across very")
    print("different t's, that confirms the degenerate-encoding mechanism:")
    print("TimeNet cannot distinguish frames well, so its init output is almost")
    print("invariant to t (except for parity). Compare against the normalised")
    print("rows which should show smooth variation with t.")


if __name__ == "__main__":
    main()
