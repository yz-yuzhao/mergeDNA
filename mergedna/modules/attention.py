"""Multi-head self-attention with optional ToMe-style proportional attention
and RoPE.

Two deviations from a standard pre-LN MHA:

  - Proportional attention (optional ``size`` arg): ``log(size)`` is added
    to attention logits along the *key* axis before softmax — equivalent to
    treating a merged token of size ``g`` as ``g`` identical replicas. When
    ``size`` is supplied, the validity mask and ``log(size)`` are folded into a
    single additive float bias passed to SDPA.
  - RoPE (optional ``positions`` arg): rotary position embedding applied to
    Q and K. Under merging, a token's position is the size-weighted centroid
    of its source nucleotide indices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mergedna.modules.rope import RoPE


class MultiHeadAttention(nn.Module):
    """SDPA-backed multi-head self-attention with optional size-bias and RoPE.

    Args:
        d_model:    feature dimension.
        n_heads:    number of attention heads.
        dropout:    dropout on attention weights and output.
        rope_base:  RoPE frequency base (default 10_000).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.out_drop = nn.Dropout(dropout)
        self.rope = RoPE(self.d_head, base=rope_base)

    def get_keys(self, x: torch.Tensor) -> torch.Tensor:
        """Return K vectors without running full attention.

        Slices the K block from the fused QKV projection. Used by
        ``LatentEncoder`` to obtain a content-based similarity metric for
        ToMe-style bipartite matching (standard ToMe reuses K rather than a
        separate grouping head).

        Args:
            x: ``[B, L, D]`` (pre-attention LN should already be applied).

        Returns:
            ``[B, L, D]`` K vectors (all heads concatenated, pre-RoPE).
        """
        D = self.d_model
        return F.linear(x, self.qkv.weight[D : 2 * D], self.qkv.bias[D : 2 * D])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        size: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Self-attention.

        Args:
            x:         ``[..., L, D]`` input. Leading dims may be any shape;
                       attention runs over the second-to-last axis.
            attn_mask: optional ``[..., L]`` BoolTensor (``True`` = valid key).
            size:      optional ``[..., L]`` token sizes. When provided,
                       ``log(size)`` is added to attention logits along the
                       key axis (proportional attention).
            positions: optional ``[..., L]`` positions for RoPE.

        Returns:
            ``[..., L, D]`` tensor.
        """
        if x.dim() < 2:
            raise ValueError(f"x must have at least 2 dims, got shape {tuple(x.shape)}")
        leading = x.shape[:-2]
        L, D = x.shape[-2], x.shape[-1]
        if D != self.d_model:
            raise ValueError(f"feature dim {D} != d_model {self.d_model}")

        # Flatten leading dims to a single batch axis. Let the same module
        # handle [B, L, D] (latent stack) and [B, n_w, W, D] (local windowed).
        B = int(torch.tensor(leading).prod().item()) if len(leading) else 1
        xf = x.reshape(B, L, D)

        qkv = self.qkv(xf).reshape(B, L, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # [3, B, H, L, d_head]
        q, k, v = qkv.unbind(0)                     # each [B, H, L, d_head]

        if positions is not None:
            q, k = self.rope(q, k, positions.reshape(B, L))

        # Build the SDPA attention argument. Four cases:
        #   1. size only         → float bias  [B,1,1,L]
        #   2. attn_mask only    → bool mask   [B,1,1,L]
        #   3. both              → float bias with -inf where invalid
        #   4. neither           → None
        attn_arg: torch.Tensor | None
        if size is not None:
            size_f = size.reshape(B, L).to(q.dtype)
            log_size = torch.log(size_f.clamp_min(1.0))     # [B, L]
            bias = log_size[:, None, None, :]               # [B, 1, 1, L]
            if attn_mask is not None:
                mask = attn_mask.reshape(B, L)
                bias = bias.masked_fill(~mask[:, None, None, :], float("-inf"))
            attn_arg = bias
        elif attn_mask is not None:
            mask = attn_mask.reshape(B, L)
            attn_arg = mask[:, None, None, :]
        else:
            attn_arg = None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_arg,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )                                            # [B, H, L, d_head]

        # Rows whose entire key axis was masked produce NaNs from softmax of
        # all -inf. Replace with 0.
        out = torch.nan_to_num(out, nan=0.0)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_drop(self.out(out))
        return out.reshape(*leading, L, D)
