"""Tests for adaptive masking."""

from __future__ import annotations

import torch

from mergedna.losses.adaptive_mask import (
    sample_adaptive_mask,
    propagate_mask_to_nucleotides,
)
from mergedna.losses.reconstruction import masked_mtr_loss
from mergedna.modules.source_map import SourceMap
from mergedna.modules.merge import MergePlan


def _pair_merge_plan(B: int, L: int) -> MergePlan:
    """Same helper as in test_source_map: merge tokens 0 and 1 into one."""
    L_new_v = L - 1
    old_to_new = torch.zeros(B, L, dtype=torch.long)
    old_to_new[:, 0] = 0
    old_to_new[:, 1] = 0
    old_to_new[:, 2:] = torch.arange(1, L_new_v).unsqueeze(0).expand(B, -1)
    new_size = torch.ones(B, L_new_v, dtype=torch.long)
    new_size[:, 0] = 2
    L_new = torch.full((B,), L_new_v, dtype=torch.long)
    return MergePlan(old_to_new=old_to_new, new_size=new_size, L_new=L_new, L_new_max=L_new_v)


def test_mask_count_correct():
    """Sample exactly ``K_mask`` Trues per batch element."""
    B, L = 2, 16
    size_local = torch.ones(B, L, dtype=torch.long)
    # Simulate a global merge where every group has size 2.
    K_global = 8
    size_latent = torch.full((B, K_global), 2, dtype=torch.long)
    parent_global = torch.arange(L).remainder(K_global).unsqueeze(0).expand(B, L).contiguous()
    token_mask = torch.ones(B, L, dtype=torch.bool)

    K_mask = 5
    gen = torch.Generator().manual_seed(0)
    mask = sample_adaptive_mask(
        
        size_latent=size_latent,
        parent_global=parent_global,
        K_mask=K_mask,
        token_mask_local=token_mask,
        generator=gen,
    )
    assert mask.shape == (B, L)
    for b in range(B):
        assert int(mask[b].sum()) == K_mask


def test_invalid_tokens_never_masked():
    B, L = 1, 8
    size_local = torch.ones(B, L, dtype=torch.long)
    size_latent = torch.ones(B, 4, dtype=torch.long)
    parent_global = torch.arange(L).remainder(4).unsqueeze(0)
    # Mark the second half as invalid.
    token_mask = torch.tensor([[True, True, True, True, False, False, False, False]])

    K_mask = 4
    gen = torch.Generator().manual_seed(1)
    mask = sample_adaptive_mask(
        
        size_latent=size_latent,
        parent_global=parent_global,
        K_mask=K_mask,
        token_mask_local=token_mask,
        generator=gen,
    )
    # All masked positions must be in the valid prefix.
    assert int(mask[0, :4].sum()) == K_mask
    assert int(mask[0, 4:].sum()) == 0


def test_inverse_square_probability_via_log_probs():
    """Check the underlying log-probabilities directly: ``-2 * log g``.

    We patch ``torch.rand`` to return constant 0.5 (so Gumbel = 0 across the
    board), making the Gumbel-top-K reduce to an ordinary top-K of ``log p``.
    """
    B, L = 1, 6
    size_local = torch.ones(B, L, dtype=torch.long)
    # Six local tokens partitioned into 3 latent groups of varying sizes:
    #   group 0 contains tokens [0, 1]   -> g=2
    #   group 1 contains tokens [2, 3]   -> g=2
    #   group 2 contains tokens [4, 5]   -> g=2
    # Make group sizes very different to give a clear ordering.
    size_latent = torch.tensor([[1, 4, 16]])
    parent_global = torch.tensor([[0, 0, 1, 1, 2, 2]])
    token_mask = torch.ones(B, L, dtype=torch.bool)

    # With deterministic Gumbel = 0, top-K returns the largest log_p positions.
    # log_p = -2 log g => most-likely positions are those in group 0 (g=1).
    # Sample a single position (K=1) and confirm it falls inside group 0.
    orig_rand = torch.rand
    try:
        torch.rand = lambda *args, **kw: torch.full(args[0] if isinstance(args[0], tuple) else args, 0.5, device=kw.get("device"))  # type: ignore[assignment]
        mask = sample_adaptive_mask(
            
            size_latent=size_latent,
            parent_global=parent_global,
            K_mask=1,
            token_mask_local=token_mask,
        )
    finally:
        torch.rand = orig_rand
    masked_pos = mask[0].nonzero().flatten().tolist()
    assert len(masked_pos) == 1 and masked_pos[0] in (0, 1), (
        f"expected mask in group 0 (tokens 0 or 1), got {masked_pos}"
    )


def test_propagation_via_source_map():
    sm = SourceMap.identity(B=1, N=4).apply_merge(_pair_merge_plan(B=1, L=4))
    # Token 0 covers nucleotides 0 and 1 (size 2); tokens 1 and 2 are size 1.
    mask_local = torch.tensor([[True, False, True]])
    mask_nuc = propagate_mask_to_nucleotides(mask_local, sm)
    assert mask_nuc.tolist() == [[True, True, False, True]]


def test_masked_loss_only_uses_masked_positions():
    """``masked_mtr_loss`` must produce zero when no positions are masked, and
    must equal a hand-computed average when only some positions are masked.
    """
    B, N, V = 1, 4, 7
    logits = torch.zeros(B, N, V)
    # logits = 0 everywhere => softmax is uniform => CE = log(V) per position.
    targets = torch.tensor([[0, 1, 2, 3]])

    none = torch.zeros(B, N, dtype=torch.bool)
    assert masked_mtr_loss(logits, targets, none).item() == 0.0

    only_first_two = torch.tensor([[True, True, False, False]])
    loss = masked_mtr_loss(logits, targets, only_first_two)
    import math
    assert torch.isclose(loss, torch.tensor(math.log(V))), (
        f"expected log(V) = {math.log(V):.4f}, got {loss.item():.4f}"
    )
