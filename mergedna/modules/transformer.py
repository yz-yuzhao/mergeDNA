"""Transformer block with optional local-window attention.

If ``local_window`` is set, attention is performed within disjoint windows of
that size. This is for the local encoder/decoder mode.

Position information is supplied as a per-token ``positions`` tensor that the
attention module turns into RoPE rotations. In windowed mode, ``positions`` is
reshaped to ``[B, n_w, W]`` alongside ``x`` so that each token still rotates by
its absolute (whole-sequence) coordinate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mergedna.modules.attention import MultiHeadAttention
from mergedna.modules.windows import pad_to_multiple, to_windows, from_windows


class _MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """A pre-LN transformer block.

    Args:
        d_model:       feature dimension.
        n_heads:       attention heads.
        mlp_ratio:     hidden / d_model in the MLP.
        dropout:       residual / attention / MLP dropout.
        local_window:  if not None, attention is computed within disjoint
                       windows of this size (along the sequence axis). Used
                       by the local encoder and decoder.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        local_window: int | None = None,
    ) -> None:
        super().__init__()
        self.local_window = local_window
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = _MLP(d_model, mlp_ratio, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        size: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply attention then MLP.

        Args:
            x:         ``[B, L, D]``.
            attn_mask: optional ``[B, L]`` BoolTensor (``True`` = valid).
            size:      optional ``[B, L]`` token sizes (proportional attention).
                       Only meaningful for non-windowed attention; if
                       ``local_window`` is set, ``size`` is ignored.
            positions: optional ``[B, L]`` float positions for RoPE.

        Returns:
            ``[B, L, D]``.
        """
        h = self.norm1(x)
        if self.local_window is None:
            h = self.attn(h, attn_mask=attn_mask, size=size, positions=positions)
        else:
            h = self._apply_windowed_attention(h, attn_mask=attn_mask, positions=positions)
            # Note: size is intentionally ignored in windowed mode; see module
            # docstring.
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        return x + h

    def get_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Return K vectors from the attention sub-layer for ``x``.

        Applies the pre-attention LayerNorm then delegates to
        ``MultiHeadAttention.get_keys``. Used by ``LatentEncoder`` to extract
        a ToMe-style similarity metric after the merge-layer forward pass.

        Args:
            x: ``[B, L, D]`` block input (post-residual from this layer).

        Returns:
            ``[B, L, D]`` K vectors.
        """
        return self.attn.get_keys(self.norm1(x))

    def _apply_windowed_attention(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        positions: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run attention inside disjoint windows of ``self.local_window``."""
        W = self.local_window  # type: ignore[assignment]
        B, L, _ = x.shape

        # pad the input to make the sequence length divisible by window size
        x_p, pad = pad_to_multiple(x, W, dim=1, value=0.0)
        L_pad = x_p.shape[1]
        n_w = L_pad // W
        x_w = to_windows(x_p, W)                        # [B, n_w, W, D]

        if attn_mask is not None:
            mask_p, _ = pad_to_multiple(attn_mask, W, dim=1, value=False)
            mask_w = mask_p.view(B, n_w, W)
        else:
            mask_w = None

        if positions is not None:
            # Pad with 0.0 — padded slots have mask=False so the rotation value
            # there is irrelevant.
            pos_p, _ = pad_to_multiple(positions, W, dim=1, value=0.0)
            pos_w = pos_p.view(B, n_w, W)
        else:
            pos_w = None

        out_w = self.attn(x_w, attn_mask=mask_w, size=None, positions=pos_w)
        out = from_windows(out_w)                              # [B, L_pad, D]
        if pad > 0:
            out = out[:, :L, :]
        return out
