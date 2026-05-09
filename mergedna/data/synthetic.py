"""Synthetic random-DNA dataset.

Used by the test suite and the test run. Each example is a uniform
random sequence over ``{A, T, C, G}``.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from mergedna.data.vocab import NUCLEOTIDE_IDS


class SyntheticDNA(Dataset):
    """Uniform i.i.d. random DNA sequences.

    Args:
        n_examples: number of examples in the dataset.
        seq_len:    nucleotide-resolution sequence length per example.
        seed:       RNG seed for reproducibility.

    Returns LongTensor of shape ``[seq_len]`` per example, with ids in
    ``{A, T, C, G}`` (0..3).
    """

    def __init__(self, n_examples: int, seq_len: int, seed: int = 0) -> None:
        self.n_examples = n_examples
        self.seq_len = seq_len
        gen = torch.Generator().manual_seed(seed)
        n_bases = len(NUCLEOTIDE_IDS)
        self._data: torch.Tensor = torch.randint(
            low=0, high=n_bases, size=(n_examples, seq_len), generator=gen, dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._data[idx]
