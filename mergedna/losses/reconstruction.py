"""Cross-entropy reconstruction losses on nucleotide-resolution logits.

  - ``mtr_loss``        — Merged Token Reconstruction.
  - ``masked_mtr_loss`` — Adaptive Masked Token Modeling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mergedna.data.vocab import PAD


def mtr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = PAD,
) -> torch.Tensor:
    """Per-token cross entropy averaged over non-ignored positions.

    Args:
        logits:       ``[B, N, V]`` unnormalized scores.
        targets:      ``[B, N]`` LongTensor of ground-truth ids.
        ignore_index: id to ignore in the average (defaults to ``[PAD]``).
    """
    if logits.dim() != 3 or targets.dim() != 2:
        raise ValueError(
            f"shapes: logits {tuple(logits.shape)}, targets {tuple(targets.shape)}"
        )
    B, N, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * N, V),
        targets.reshape(B * N),
        ignore_index=ignore_index,
    )


def masked_mtr_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = PAD,
) -> torch.Tensor:
    """Cross entropy averaged over masked, non-ignored positions only.

    Args:
        logits:       ``[B, N, V]``.
        targets:      ``[B, N]``.
        mask:         ``[B, N]`` BoolTensor — ``True`` where the loss is taken.
        ignore_index: id to additionally exclude (e.g. ``[PAD]``).

    Returns:
        Scalar tensor. If no positions are masked, returns 0 (zero-grad).
    """
    if mask.dim() != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(mask.shape)}")
    keep = mask & (targets != ignore_index)
    if keep.sum() == 0:
        return logits.new_zeros(())
    # Per-position CE without reduction, then take a mean over kept positions.
    B, N, V = logits.shape
    nll = F.cross_entropy(
        logits.reshape(B * N, V),
        targets.reshape(B * N),
        reduction="none",
    ).reshape(B, N)
    return (nll * keep).sum() / keep.sum().clamp_min(1).to(nll.dtype)
