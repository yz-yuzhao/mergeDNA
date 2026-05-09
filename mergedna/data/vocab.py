"""Nucleotide vocabulary and string <-> id helpers.

The vocabulary has 7 ids:

  0  A      adenine
  1  T      thymine
  2  C      cytosine
  3  G      guanine
  4  N      any/unknown nucleotide
  5  [MASK] masking token (used by L_AMTM)
  6  [PAD]  padding token
"""

from __future__ import annotations

import torch

A: int = 0
T: int = 1
C: int = 2
G: int = 3
N_IDX: int = 4
MASK: int = 5
PAD: int = 6

TOKENS: tuple[str, ...] = ("A", "T", "C", "G", "N", "[MASK]", "[PAD]")
VOCAB_SIZE: int = len(TOKENS)

# Canonical bases used for synthetic data generation and reconstruction targets.
NUCLEOTIDE_IDS: tuple[int, ...] = (A, T, C, G)

_CHAR_TO_ID = {"A": A, "T": T, "C": C, "G": G, "N": N_IDX}


def encode(seq: str) -> torch.LongTensor:
    """Encode an uppercase DNA string into a 1-D LongTensor of ids.

    Unknown characters are mapped to ``N_IDX``.
    """
    ids = [_CHAR_TO_ID.get(ch, N_IDX) for ch in seq.upper()]
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    """Decode a 1-D LongTensor of ids into an uppercase DNA string."""
    if ids.dim() != 1:
        raise ValueError(f"decode expects a 1-D tensor, got shape {tuple(ids.shape)}")
    return "".join(TOKENS[int(i)] if int(i) < N_IDX + 1 else "N" for i in ids.tolist())
