"""Rotary Position Embedding (RoPE).

Replaces the absolute ``nn.Embedding(max_seq_len, d_model)`` previously added
at the input of the local encoder/decoder. RoPE rotates Q and K *inside*
attention by per-token positions, leaving V untouched.

Two notes specific to this codebase:

  - **Float positions are required.** After a merge step, a token's position is
    the size-weighted centroid of its source nucleotide indices, which is
    typically non-integer. RoPE works on arbitrary real-valued positions.
  - **Same rotation across all heads.** Standard RoPE; the cos/sin cache is
    broadcast over the head axis.

The rotation pairs adjacent dims ``(2i, 2i+1)`` and applies
``[[cos θ, -sin θ], [sin θ, cos θ]]`` with ``θ = pos · base^(-2i/d_head)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """Rotary position embedding for Q and K.

    Args:
        d_head: per-head feature dim. Must be even.
        base:   frequency base (paper default 10_000).
    """

    def __init__(self, d_head: int, base: float = 10000.0) -> None:
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE, got {d_head}")
        self.d_head = d_head
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
        )
        # Non-persistent: re-derived on load; depends only on (d_head, base).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate ``q`` and ``k`` by ``positions``.

        Args:
            q, k:      ``[B, H, L, d_head]``.
            positions: ``[B, L]`` (float; centroid coordinates from the merge
                       state — may be non-integer).

        Returns:
            ``(q_rot, k_rot)`` with the same shapes as the inputs.
        """
        # angles: [B, L, d_head/2] — fp32 for numerical stability of cos/sin.
        angles = positions.to(self.inv_freq.dtype).unsqueeze(-1) * self.inv_freq
        cos = angles.cos().unsqueeze(1).to(q.dtype)  # [B, 1, L, d_head/2]
        sin = angles.sin().unsqueeze(1).to(q.dtype)
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply 2-D rotation to adjacent dim pairs of ``x``.

    ``x`` is ``[B, H, L, d_head]``; ``cos``/``sin`` are ``[B, 1, L, d_head/2]``.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * cos - x_odd * sin
    y_odd = x_even * sin + x_odd * cos
    return torch.stack((y_even, y_odd), dim=-1).flatten(-2)
