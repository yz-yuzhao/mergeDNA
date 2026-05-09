"""Latent decoder

A stack of full-attention blocks symmetric to the latent encoder.
Proportional attention is used.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from mergedna.config import MergeDNAConfig
from mergedna.modules.transformer import TransformerBlock


class LatentDecoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    local_window=None,
                )
                for _ in range(cfg.n_latent_dec_layers)
            ]
        )

    def forward(
        self,
        z: torch.Tensor,
        size: torch.Tensor,
        token_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """``[B, L_or_K, D] -> [B, L_or_K, D]``.

        Args:
            z:           feature tensor.
            size:        token sizes (used for proportional attention).
            token_mask:  validity mask.
            positions:   ``[B, L_or_K]`` float positions for RoPE.
        """
        for layer in self.layers:
            size_arg = size.to(z.dtype) if self.cfg.use_proportional_attention else None
            z = layer(z, attn_mask=token_mask, size=size_arg, positions=positions)
        return z
