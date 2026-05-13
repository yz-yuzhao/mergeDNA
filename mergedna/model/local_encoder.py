"""Local encoder 

Each layer works on local attention and token merging:

  1. Runs windowed self-attention.
  2. Computes a similarity metric via the Grouping Embeddings from the shared GroupingHead.
  3. Picks the top-r pairs per window by all-pairs matching.
  4. Combines features with size-weighted averaging and updates the SourceMap.

The compression ratio is sampled stochastically per training step (see
``sample_r_schedule``) and remains constant across the batch within a step.

Position information is RoPE-based.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn

from mergedna.config import MergeDNAConfig
from mergedna.modules.grouping_head import GroupingHead
from mergedna.modules.merge import (
    all_pairs_match_window,
    apply_merge_plan,
    soft_apply_merge_window,
    update_positions,
)
from mergedna.modules.source_map import SourceMap
from mergedna.modules.transformer import TransformerBlock


class LocalEncoder(nn.Module):
    """Token embedding + ``L_local`` rounds of (local-attn → merge).

    Returns:
        h:           ``[B, L_max, D]`` features after the final merge.
        source_map:  nucleotide ↔ local-token correspondence.
        size:        ``[B, L_max]`` — token sizes after the final merge.
        token_mask:  ``[B, L_max]`` — validity mask after the final merge.
    """

    def __init__(self, cfg: MergeDNAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    local_window=cfg.window_size,
                )
                for _ in range(cfg.n_local_enc_layers)
            ]
        )
        self.grouping_head = GroupingHead(cfg.d_model, cfg.grouping_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        r_schedule: List[int],
    ) -> tuple[torch.Tensor, SourceMap, torch.Tensor, torch.Tensor]:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be 2-D [B, N], got shape {tuple(token_ids.shape)}")
        if len(r_schedule) != len(self.layers):
            raise ValueError(
                f"r_schedule must have one entry per layer "
                f"(got {len(r_schedule)} entries for {len(self.layers)} layers)"
            )
        B, N = token_ids.shape
        device = token_ids.device

        x = self.token_embedding(token_ids)

        source_map = SourceMap.identity(B, N, device=device)
        size = torch.ones(B, N, dtype=torch.long, device=device)
        token_mask = torch.ones(B, N, dtype=torch.bool, device=device)
        positions = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, N).contiguous()

        for i, layer in enumerate(self.layers):
            x = layer(x, attn_mask=token_mask, size=None, positions=positions)
            metric = self.grouping_head(x)
            plan = all_pairs_match_window(
                metric=metric,
                size=size,
                r=r_schedule[i],
                W=self.cfg.window_size,
                valid=token_mask,
            )
            # Update positions using *old* sizes before reassigning.
            positions = update_positions(positions, size, plan)
            x_hard, size_new = apply_merge_plan(x, size, plan)
            # STE bridge — gives the grouping head a gradient path. Forward
            # value is unchanged (x_soft - x_soft.detach() == 0); backward
            # routes through x_soft, which depends continuously on `metric`.
            if self.training and r_schedule[i] > 0:
                x_soft = soft_apply_merge_window(
                    x=x, size=size, metric=metric,
                    r=r_schedule[i], W=self.cfg.window_size,
                    valid=token_mask, plan=plan,
                )
                x = x_hard + (x_soft - x_soft.detach())
            else:
                x = x_hard
            size = size_new
            source_map = source_map.apply_merge(plan)
            token_mask = source_map.token_mask

        return x, source_map, size, token_mask


def sample_r_schedule(
    N: int,
    n_layers: int,
    target_compression: float = 0.5,
    jitter: float = 0.0,
    training: bool = True,
    generator: torch.Generator | None = None,
) -> List[int]:
    """Sample one target ``L`` for the whole step, then divide ``N − L`` evenly.

    Args:
        N:                  nucleotide-resolution input length.
        n_layers:           number of merge layers in the local encoder.
        target_compression: target ratio ``L / N``.
        jitter:             sigma as a fraction of ``N``. Set to 0 to disable
                            stochasticity (used at eval).
        training:           when False, ``jitter`` is ignored.
        generator:          optional torch RNG for determinism.

    Returns:
        A list of ``n_layers`` non-negative ints summing to ``N − L``.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}")
    if N <= 0:
        return [0] * n_layers

    target_L_float = N * target_compression
    if training and jitter > 0.0:
        sigma = jitter * N
        sample = torch.normal(
            mean=torch.tensor(target_L_float),
            std=torch.tensor(sigma),
            generator=generator,
        ).item()
        lo = math.ceil(0.4 * N)
        hi = math.floor(0.6 * N)
        L = int(round(sample))
        L = max(lo, min(hi, L))
    else:
        L = int(round(target_L_float))

    L = max(0, min(N, L))
    total_remove = N - L
    base, rem = divmod(total_remove, n_layers)
    return [base + (1 if i < rem else 0) for i in range(n_layers)]
