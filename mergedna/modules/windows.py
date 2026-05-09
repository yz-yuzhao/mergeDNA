"""Local-window helpers to reshape the tensors
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
    dim: int,
    value: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Pad ``x`` along ``dim`` so that ``x.shape[dim]`` is divisible by ``multiple``.

    Args:
        x:        the tensor to pad.
        multiple: target divisor (e.g. window size ``W``).
        dim:      axis to pad along.
        value:    fill value for the padding.

    Returns:
        ``(padded, pad_len)``. ``pad_len`` is 0 if no padding was needed.
    """
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    n = x.shape[dim]
    rem = n % multiple
    if rem == 0:
        return x, 0
    pad_len = multiple - rem
    # F.pad takes pads in *reverse* dim order, two values per dim (front, back).
    ndim = x.dim()
    pads: list[int] = [0] * (2 * ndim)
    # back-pad slot for ``dim`` is index 2*(ndim-1-dim) + 1
    pads[2 * (ndim - 1 - dim) + 1] = pad_len
    return F.pad(x, pads, value=value), pad_len


def to_windows(x: torch.Tensor, W: int) -> torch.Tensor:
    """Reshape ``[B, L, D] -> [B, L/W, W, D]``.
    """
    if x.dim() != 3:
        raise ValueError(f"to_windows expects 3-D input, got shape {tuple(x.shape)}")
    B, L, D = x.shape
    if L % W != 0:
        raise ValueError(f"sequence length {L} not divisible by window size {W}")
    return x.view(B, L // W, W, D)


def from_windows(x_w: torch.Tensor) -> torch.Tensor:
    """Inverse of ``to_windows``: ``[B, n_w, W, D] -> [B, n_w * W, D]``."""
    if x_w.dim() != 4:
        raise ValueError(f"from_windows expects 4-D input, got shape {tuple(x_w.shape)}")
    B, n_w, W, D = x_w.shape
    return x_w.reshape(B, n_w * W, D)
