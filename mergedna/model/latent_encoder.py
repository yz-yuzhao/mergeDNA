"""Latent encoder

A stack of full-attention transformer blocks, optionally interleaved with a
single ToMe-style global merge. Proportional attention is used.

Positions (for RoPE) are inherited from the local encoder — each latent token
carries a centroid coordinate in the original nucleotide space — and are
updated through the global merge step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mergedna.config import MergeDNAConfig
from mergedna.modules.merge import (
    MergePlan,
    apply_merge_plan,
    bipartite_match_global,
    update_positions,
)
from mergedna.modules.transformer import TransformerBlock


class LatentEncoder(nn.Module):
    """``cfg.n_latent_enc_layers`` full-attention blocks with optional global ToMe."""

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
                for _ in range(cfg.n_latent_enc_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        size: torch.Tensor,
        token_mask: torch.Tensor,
        positions: torch.Tensor,
        do_global_merge: bool = False,
        r_global: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, MergePlan | None]:
        """Run the latent stack.

        Args:
            h:                ``[B, L, D]`` features (output of local encoder).
            size:             ``[B, L]`` token sizes.
            token_mask:       ``[B, L]`` validity mask.
            positions:        ``[B, L]`` float positions (for RoPE).
            do_global_merge:  when True, fire one global ToMe merge after layer
                              ``cfg.global_merge_layer``.
            r_global:         number of A-tokens to merge in the global step.
                              Ignored when ``do_global_merge`` is False.

        Returns:
            ``(z, size_out, token_mask_out, positions_out, global_plan)``.
            If no global merge fired, ``size_out``/``token_mask_out``/
            ``positions_out`` are unchanged from the inputs and ``global_plan``
            is None.
        """
        global_plan: MergePlan | None = None
        merge_layer = self.cfg.global_merge_layer

        for i, layer in enumerate(self.layers):
            size_arg = size.to(h.dtype) if self.cfg.use_proportional_attention else None
            h = layer(h, attn_mask=token_mask, size=size_arg, positions=positions)

            if do_global_merge and i == merge_layer and global_plan is None:
                if r_global is None or r_global <= 0:
                    # No merge requested at this layer; behave as a no-op.
                    continue
                metric = layer.get_keys(h)
                plan = bipartite_match_global(metric, size, r=r_global, valid=token_mask)
                positions = update_positions(positions, size, plan)
                h, size = apply_merge_plan(h, size, plan)
                # Recompute token_mask at new resolution.
                K_max = plan.L_new_max
                token_mask = (
                    torch.arange(K_max, device=h.device).unsqueeze(0)
                    < plan.L_new.unsqueeze(1)
                )
                global_plan = plan

        return h, size, token_mask, positions, global_plan
