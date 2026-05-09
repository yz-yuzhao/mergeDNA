"""Configuration dataclasses for MergeDNA.

Two configs:

  - ``MergeDNAConfig``  — model architecture (paper-faithful defaults).
  - ``TrainConfig``     — training-loop hyperparameters.

A ``tiny_config()`` factory returns a CPU-friendly pair used by the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MergeDNAConfig:
    """Architecture configuration.

    Defaults reproduce the paper's 380M parameter pretraining model:
    D=1024, 4/20/4/2 layers (E_φ / E_ψ / E_ω / E_ζ), window size 16.
    """

    # Vocabulary: {A, T, C, G, N, [MASK], [PAD]} = 7 ids; see mergedna.data.vocab.
    vocab_size: int = 7

    # Embedding dimension shared across all transformer modules.
    d_model: int = 1024

    # Per-module block counts.
    n_local_enc_layers: int = 4
    n_latent_enc_layers: int = 20
    n_latent_dec_layers: int = 4
    n_local_dec_layers: int = 2

    n_heads: int = 16
    mlp_ratio: float = 4.0

    # Local-window size, used by both the local encoder and local decoder.
    window_size: int = 16

    dropout: float = 0.0

    # Maximum nucleotide-resolution sequence length supported by positional embeddings.
    max_seq_len: int = 4096

    # ----- merge schedule -----
    # Local encoder: target compression ratio L / N.
    target_local_compression: float = 0.5
    # Stochastic schedule: σ as a fraction of N. Set to 0 to disable jitter.
    local_compression_jitter: float = 0.05

    # Latent encoder: target compression K / L when global ToMe is fired (pass 2 only).
    target_global_compression: float = 0.5

    # The latent-encoder layer index (0-based) after which the one-shot global ToMe fires.
    # Default = mid-stack.
    global_merge_layer: int = 10

    # ----- merge-step internals -----
    # Dimension of the lightweight DTEM-style grouping embedding used as similarity metric.
    # ``None`` -> defaults to d_model // 4 inside GroupingHead.
    grouping_dim: int | None = None

    metric_source: Literal["grouping_head", "key"] = "grouping_head"

    # Whether the latent encoder/decoder use proportional attention (+log size bias).
    # The local encoder/decoder never use it (size variance is small there).
    use_proportional_attention: bool = True


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop hyperparameters."""

    batch_size: int = 4
    seq_len: int = 1024
    lr: float = 3e-4
    weight_decay: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    steps: int = 200

    # Loss weight on the latent-only reconstruction term (paper: 0.25).
    lambda_latent: float = 0.25

    # Number of local tokens to mask per example for the L_AMTM term. Defaults to K
    # (the number of latent tokens after global merge). Set explicitly to override.
    n_mask: int | None = None

    # Optional RNG seed for reproducibility.
    seed: int = 0


def tiny_config() -> tuple[MergeDNAConfig, TrainConfig]:
    """Return a small (model, train) pair used by the test suite.

    Sized so that an end-to-end forward + backward runs in well under a second on CPU.
    """
    model = MergeDNAConfig(
        vocab_size=7,
        d_model=32,
        n_local_enc_layers=2,
        n_latent_enc_layers=2,
        n_latent_dec_layers=1,
        n_local_dec_layers=1,
        n_heads=4,
        mlp_ratio=2.0,
        window_size=4,
        max_seq_len=64,
        target_local_compression=0.5,
        local_compression_jitter=0.0,
        target_global_compression=0.5,
        global_merge_layer=0,
        grouping_dim=8,
    )
    train = TrainConfig(
        batch_size=2,
        seq_len=32,
        lr=3e-4,
        steps=4,
    )
    return model, train
