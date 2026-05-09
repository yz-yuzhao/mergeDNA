"""Tests for the RoPE module and per-token position bookkeeping.

The integration tests in test_modules_shapes.py exercise the full pipeline,
so this file only covers the new isolated pieces.
"""

from __future__ import annotations

import torch

from mergedna.modules.merge import (
    MergePlan,
    apply_merge_plan,
    update_positions,
)
from mergedna.modules.rope import RoPE
from mergedna.modules.source_map import SourceMap


def test_rope_position_zero_is_identity():
    """At position 0, every angle is 0 — Q and K must come back unchanged."""
    torch.manual_seed(0)
    B, H, L, d_head = 1, 2, 4, 8
    rope = RoPE(d_head)
    q = torch.randn(B, H, L, d_head)
    k = torch.randn(B, H, L, d_head)
    positions = torch.zeros(B, L)
    q_rot, k_rot = rope(q, k, positions)
    assert torch.allclose(q_rot, q, atol=1e-6)
    assert torch.allclose(k_rot, k, atol=1e-6)


def test_rope_preserves_pairwise_norms():
    """RoPE is a pure rotation per dim-pair — ||q|| and ||k|| must be preserved."""
    torch.manual_seed(1)
    B, H, L, d_head = 1, 2, 4, 8
    rope = RoPE(d_head)
    q = torch.randn(B, H, L, d_head)
    k = torch.randn(B, H, L, d_head)
    positions = torch.tensor([[0.0, 1.5, 7.0, 12.3]])
    q_rot, k_rot = rope(q, k, positions)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k.norm(dim=-1), k_rot.norm(dim=-1), atol=1e-5)


def test_update_positions_centroid_of_two_merged():
    """Merging tokens at positions 3 and 4 (each size 1) should yield 3.5."""
    # plan: old_to_new = [0, 0, 1] — merge token 0 into 0, token 1 into 0, token 2 stays as 1.
    # Wait — we need both tokens merged to map to the same new index.
    # Simpler: 2 tokens, both map to new index 0.
    plan = MergePlan(
        old_to_new=torch.tensor([[0, 0]], dtype=torch.long),
        new_size=torch.tensor([[2]], dtype=torch.long),
        L_new=torch.tensor([1], dtype=torch.long),
        L_new_max=1,
    )
    old_positions = torch.tensor([[3.0, 4.0]])
    old_size = torch.tensor([[1, 1]], dtype=torch.long)
    new_positions = update_positions(old_positions, old_size, plan)
    assert new_positions.shape == (1, 1)
    assert torch.allclose(new_positions, torch.tensor([[3.5]]))


def test_update_positions_size_weighted_centroid():
    """Merging a size-2 token at pos 3 with a size-1 token at pos 6 → centroid 4."""
    # weighted sum = 2*3 + 1*6 = 12; new_size = 3 → 12/3 = 4
    plan = MergePlan(
        old_to_new=torch.tensor([[0, 0]], dtype=torch.long),
        new_size=torch.tensor([[3]], dtype=torch.long),
        L_new=torch.tensor([1], dtype=torch.long),
        L_new_max=1,
    )
    old_positions = torch.tensor([[3.0, 6.0]])
    old_size = torch.tensor([[2, 1]], dtype=torch.long)
    new_positions = update_positions(old_positions, old_size, plan)
    assert torch.allclose(new_positions, torch.tensor([[4.0]]))


def test_source_map_token_positions_identity_is_arange():
    """Before any merge, every nucleotide is its own token — positions = arange."""
    sm = SourceMap.identity(B=2, N=8)
    pos = sm.token_positions()
    expected = torch.arange(8, dtype=torch.float32).unsqueeze(0).expand(2, 8)
    assert torch.allclose(pos, expected)


def test_token_positions_matches_update_positions_after_merge():
    """``SourceMap.token_positions`` and threaded ``update_positions`` must agree."""
    B, N = 1, 4
    sm = SourceMap.identity(B, N)
    # Merge tokens 1 and 2 into a single new token.
    # After: parent = [0, 1, 1, 2], sizes = [1, 2, 1].
    plan = MergePlan(
        old_to_new=torch.tensor([[0, 1, 1, 2]], dtype=torch.long),
        new_size=torch.tensor([[1, 2, 1]], dtype=torch.long),
        L_new=torch.tensor([3], dtype=torch.long),
        L_new_max=3,
    )
    sm_new = sm.apply_merge(plan)

    # Path 1: derive from source map.
    pos_from_sm = sm_new.token_positions()

    # Path 2: thread through update_positions.
    old_positions = torch.arange(N, dtype=torch.float32).unsqueeze(0)
    old_size = torch.ones(B, N, dtype=torch.long)
    pos_from_update = update_positions(old_positions, old_size, plan)

    assert torch.allclose(pos_from_sm, pos_from_update)
    # Sanity: token 1 = mean(1, 2) = 1.5
    assert torch.allclose(pos_from_sm, torch.tensor([[0.0, 1.5, 3.0]]))
