"""Adaptive masking for the L_AMTM loss term.

The paper masks local token with probability inversely proportional to the
square of their latent group size.

Sampling is done with the Gumbel-top-K trick: ``argmax_top_K(log P + Gumbel)``
gives an unbiased K-without-replacement sample.

The resulting local-token mask is then propagated to nucleotide resolution
through the local-encoder source map.
"""

from __future__ import annotations

import torch

from mergedna.modules.source_map import SourceMap


def sample_adaptive_mask(
    size_latent: torch.Tensor,
    parent_global: torch.Tensor,
    K_mask: int,
    token_mask_local: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``K_mask`` local-token positions with prob ∝ 1 / g_i².

    Args:
        size_latent:       ``[B, K]`` — sizes of latent tokens after global
                           merge.
        parent_global:     ``[B, L]`` — for each local token, the latent
                           token index it belongs to (= ``global_plan.old_to_new``).
        K_mask:            number of local tokens to mask per batch element.
        token_mask_local:  ``[B, L]`` BoolTensor — only valid local tokens
                           may be masked.
        generator:         optional torch RNG for determinism.

    Returns:
        ``[B, L]`` BoolTensor with exactly ``K_mask`` ``True`` entries per
        batch element (assuming at least ``K_mask`` valid tokens).
    """
    if parent_global.dim() != 2 or size_latent.dim() != 2:
        raise ValueError("parent_global and size_latent must be 2-D")

    B, L = parent_global.shape
    device = parent_global.device

    # g_i for each local token = size of its latent parent.
    g = size_latent.gather(dim=1, index=parent_global)              # [B, L]
    g_f = g.to(torch.float32).clamp_min(1.0)
    log_p = -2.0 * torch.log(g_f)                                   # ∝ 1 / g²

    # Mask invalid tokens out of the sampling distribution.
    log_p = log_p.masked_fill(~token_mask_local, float("-inf"))

    # Gumbel-top-K: argmax_top_K of (log_p + Gumbel(0, 1)).
    u = torch.rand(B, L, device=device, generator=generator).clamp(min=1e-20, max=1.0 - 1e-20)
    gumbel = -torch.log(-torch.log(u))
    keys = log_p + gumbel

    K_eff = min(K_mask, L)
    _, top_idx = keys.topk(K_eff, dim=-1)                           # [B, K_eff]
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    mask.scatter_(dim=1, index=top_idx, src=torch.ones_like(top_idx, dtype=torch.bool))

    # Belt-and-braces: ensure no invalid token slipped in (will only happen if
    # there are fewer valid tokens than K_mask, in which case top-k naturally
    # picks some -inf-key tokens).
    mask = mask & token_mask_local
    return mask


def propagate_mask_to_nucleotides(
    mask_local: torch.Tensor,
    source_map: SourceMap,
) -> torch.Tensor:
    """Broadcast a ``[B, L]`` local-token mask through ``source_map`` to ``[B, N]``.

    A nucleotide is masked iff its parent local-token is masked.
    """
    return source_map.propagate_mask(mask_local)
