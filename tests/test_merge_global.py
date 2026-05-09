"""Tests for the latent-encoder bipartite global merge (standard ToMe)."""

from __future__ import annotations

import torch

from mergedna.modules.merge import (
    bipartite_match_global,
    apply_merge_plan,
)
from mergedna.modules.source_map import SourceMap


def test_r_zero_is_identity():
    B, L, D = 2, 8, 5
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = bipartite_match_global(metric, size, r=0, valid=valid)
    assert int(plan.L_new[0]) == L
    # Identity old_to_new.
    expected = torch.arange(L).unsqueeze(0).expand(B, L)
    assert torch.equal(plan.old_to_new, expected)


def test_only_even_indices_are_absorbed():
    """Bipartite invariant: every absorbed (non-keeper) token is an even index
    (a member of A). Multiple As may merge into the same B, so a merged group
    can contain several As but at most one B.
    """
    torch.manual_seed(13)
    B, L, D = 1, 10, 4
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    r = 3
    plan = bipartite_match_global(metric, size, r=r, valid=valid)

    o2n = plan.old_to_new[0].tolist()
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for k, new_id in enumerate(o2n):
        groups[new_id].append(k)

    merged_groups = [g for g in groups.values() if len(g) > 1]
    total_mergers = sum(len(g) - 1 for g in merged_groups)
    assert total_mergers == r, f"expected {r} merged tokens total, got {total_mergers}"

    for g in merged_groups:
        odds = [k for k in g if k % 2 == 1]
        evens = [k for k in g if k % 2 == 0]
        # Each merged group has at most one B (odd) — the keeper.
        assert len(odds) <= 1, f"group {g} has multiple odd-indexed tokens"
        # And at least one A (even) — the absorbed token(s).
        assert len(evens) >= 1, f"group {g} has no even-indexed tokens"


def test_global_match_can_cross_distant_indices():
    """Two highly similar tokens at far-apart positions should merge."""
    B, L, D = 1, 16, 4
    metric = torch.randn(B, L, D) * 0.01
    metric[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 13] = torch.tensor([1.0, 0.001, 0.0, 0.0])
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = bipartite_match_global(metric, size, r=1, valid=valid)
    # Token 0 (A) was merged into token 13 (B), so they share a new id.
    o2n = plan.old_to_new[0]
    assert int(o2n[0]) == int(o2n[13])


def test_size_sum_invariant():
    torch.manual_seed(2)
    B, L, D = 2, 12, 6
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = bipartite_match_global(metric, size, r=3, valid=valid)
    for b in range(B):
        L_b = int(plan.L_new[b])
        assert int(plan.new_size[b, :L_b].sum()) == L


def test_weighted_average_preserves_total_mass():
    torch.manual_seed(4)
    B, L, D = 2, 12, 5
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = bipartite_match_global(metric, size, r=3, valid=valid)
    x = torch.randn(B, L, D)
    x_new, size_new = apply_merge_plan(x, size, plan)

    old_mass = (x * size.unsqueeze(-1).float()).sum(dim=1)
    new_mass = (x_new * size_new.unsqueeze(-1).float()).sum(dim=1)
    assert torch.allclose(old_mass, new_mass, atol=1e-5)


def test_compose_with_source_map():
    torch.manual_seed(8)
    B, L, D = 2, 12, 4
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = bipartite_match_global(metric, size, r=3, valid=valid)
    sm = SourceMap.identity(B=B, N=L).apply_merge(plan)
    for b in range(B):
        assert int(sm.size[b, : int(sm.L[b])].sum()) == L
        assert (sm.size[b, : int(sm.L[b])] > 0).all()
