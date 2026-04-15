"""Verify the frequency-encoding pathology for integer frame indices.

Background
----------
DDS-SLAM passes the raw integer frame index into a TCNN Frequency encoder
(`self.embed_time` in model/scene_rep.py). The TCNN Frequency encoding, per
its source (dependencies/tiny-cuda-nn/include/tiny-cuda-nn/encodings/frequency.h),
computes for each input dim x and each level L in [0, n_frequencies):

    out[2L    ] = sin(2^L * pi * x)
    out[2L + 1] = cos(2^L * pi * x)

With n_frequencies=12 and input_dim=1 this gives 24 output dims.

Hypothesis
----------
For *integer* x (as DDS-SLAM passes in):
  - sin(2^L * pi * integer) = 0  for all L >= 0      (degenerate)
  - cos(pi * integer)       = (-1)^integer           (alternates by parity)
  - cos(2^L * pi * integer) = 1  for all L >= 1      (constant!)

=> Only 1 of 24 channels carries information, and it only distinguishes
   parity (odd vs even frame). The other 23 channels are constants.

Expected consequence: TimeNet's time input is effectively a 1-bit parity
signal + padding, not a rich temporal encoding.

Run:
    python Addons/verify_time_encoding.py
No CUDA / no tcnn required for the manual check. The optional tcnn block
at the bottom runs the real encoder if you're on Colab with tcnn installed.
"""

import numpy as np

N_FREQ = 12
OUT_DIM = 2 * N_FREQ  # for input_dim = 1


def manual_freq_encode(x: np.ndarray, n_frequencies: int = N_FREQ) -> np.ndarray:
    """Replicates tcnn's Frequency encoding formula.

    x: shape (N, 1)
    returns: shape (N, 2 * n_frequencies)
    """
    assert x.ndim == 2 and x.shape[1] == 1, "input must be (N,1)"
    out = np.zeros((x.shape[0], 2 * n_frequencies), dtype=np.float64)
    for L in range(n_frequencies):
        freq = (2.0 ** L) * np.pi
        out[:, 2 * L] = np.sin(freq * x[:, 0])
        out[:, 2 * L + 1] = np.cos(freq * x[:, 0])
    return out


def summarise(label: str, x: np.ndarray) -> None:
    print(f"\n--- {label} ---")
    enc = manual_freq_encode(x)
    print(f"input values: {x.ravel().tolist()}")
    print(f"encoded shape: {enc.shape}")
    # Per-channel variance across the inputs — zero variance => dead channel
    var = enc.var(axis=0)
    print("per-channel std across inputs:")
    for L in range(N_FREQ):
        s_std = enc[:, 2 * L].std()
        c_std = enc[:, 2 * L + 1].std()
        print(f"  L={L:2d}  sin std={s_std:.3e}  cos std={c_std:.3e}")
    dead = int((var < 1e-12).sum())
    print(f"dead channels (std < 1e-12): {dead} / {OUT_DIM}")

    # Pairwise L2 distance between frame embeddings — if everything collapses,
    # distances are near-zero.
    if x.shape[0] >= 2:
        diffs = []
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                diffs.append(np.linalg.norm(enc[i] - enc[j]))
        print(f"pairwise ||enc_i - enc_j|| min/mean/max: "
              f"{min(diffs):.3e} / {np.mean(diffs):.3e} / {max(diffs):.3e}")


def main() -> None:
    # 1) Raw integer frame indices: what DDS-SLAM actually feeds today
    integer_frames = np.array([[0.0], [1.0], [2.0], [100.0], [1000.0], [3999.0]])
    summarise("RAW integer frame indices (as DDS-SLAM passes)", integer_frames)

    # 2) Normalised frame indices: what it would be with t / num_frames
    normalised = integer_frames / 4000.0
    summarise("NORMALISED t / num_frames (proposed fix)", normalised)

    print("\n\n=== Interpretation ===")
    print("If the RAW run shows many dead channels and tiny pairwise distances,")
    print("the hypothesis is confirmed: TimeNet receives a near-degenerate")
    print("temporal signal and can only learn parity-like dependence on t.")

    # 3) Optional: run the actual tcnn encoder to confirm the manual match.
    try:
        import torch
        import tinycudann as tcnn
    except ImportError:
        print("\n(tcnn not installed here — skip real-encoder sanity check.")
        print(" Re-run on Colab where the DDS-SLAM env has tcnn to confirm.)")
        return

    if not torch.cuda.is_available():
        print("\n(CUDA not available — skip real-encoder sanity check.)")
        return

    enc = tcnn.Encoding(
        n_input_dims=1,
        encoding_config={"otype": "Frequency", "n_frequencies": N_FREQ},
        dtype=torch.float,
    )
    x = torch.tensor(integer_frames, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        y = enc(x).cpu().numpy()
    print("\n--- TCNN actual output, raw integer frames ---")
    print("shape:", y.shape)
    print("per-channel std:", y.std(axis=0))
    manual = manual_freq_encode(integer_frames)
    max_abs_diff = np.max(np.abs(y - manual))
    print(f"max |tcnn - manual| = {max_abs_diff:.3e}")
    if max_abs_diff < 1e-3:
        print("=> tcnn matches the manual formula. Pathology is real.")
    else:
        print("=> tcnn does NOT match the assumed formula (different scale/base).")
        print("   Re-derive from tcnn/frequency.h before concluding.")


if __name__ == "__main__":
    main()
