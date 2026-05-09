"""Tests for the local-encoder all-pairs greedy merge."""

from __future__ import annotations

import torch

from mergedna.modules.merge import (
    all_pairs_match_window,
    apply_merge_plan,
    MergePlan,
)
from mergedna.modules.source_map import SourceMap


# -------------------------------------------------------------------- #
# Within-window constraint                                             #
# -------------------------------------------------------------------- #


def test_within_window_constraint():
    """A token cannot be merged with one outside its window.

    We construct metrics where the most-similar pair is across a window
    boundary; the merge should *not* happen because matching is windowed.
    """
    B, L, W, D = 1, 8, 4, 4
    # Make tokens 3 and 4 (across the W=4 boundary) most similar to each other,
    # and all other pairs much less similar.
    metric = torch.randn(B, L, D) * 0.01
    # Two highly similar vectors at positions 3 and 4
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 3] = v
    metric[0, 4] = v + torch.tensor([0.0, 1e-3, 0.0, 0.0])
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=2, W=W, valid=valid)
    o2n = plan.old_to_new[0]

    # Tokens 3 and 4 must NOT have the same new id — they're in different windows.
    assert int(o2n[3]) != int(o2n[4]), (
        f"cross-window merge happened: token 3 and 4 both map to {int(o2n[3])}"
    )


# -------------------------------------------------------------------- #
# All-pairs (NOT bipartite) coverage                                   #
# -------------------------------------------------------------------- #


def test_all_pairs_can_merge_two_even_indices():
    """In W=4, force tokens at indices 0 and 2 (both 'even' under a bipartite
    split) to be the most similar. Verify they *do* merge — a bipartite
    even/odd scheme could never directly merge two same-parity tokens.
    """
    B, L, W, D = 1, 4, 4, 4
    metric = torch.zeros(B, L, D)
    # Tokens 0 and 2 identical and pointing one way; tokens 1 and 3 orthogonal.
    metric[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    metric[0, 3] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=1, W=W, valid=valid)
    o2n = plan.old_to_new[0]
    # 0 and 2 should land on the same new id; 1 and 3 should be on distinct ids.
    assert int(o2n[0]) == int(o2n[2])
    assert int(o2n[1]) != int(o2n[3])
    assert int(o2n[0]) != int(o2n[1])


# -------------------------------------------------------------------- #
# Each token participates in at most one merge per layer                #
# -------------------------------------------------------------------- #


def test_each_token_in_at_most_one_merge():
    """No accepted pair shares a token with another accepted pair."""
    torch.manual_seed(7)
    B, L, W, D = 2, 16, 4, 8
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=4, W=W, valid=valid)

    for b in range(B):
        # A token is "merged" if it shares an id with a *lower-indexed* token.
        # In our scheme the keeper has the lower index, so we count occurrences
        # of each new id and confirm no id has > 2 originals.
        new_ids = plan.old_to_new[b].tolist()
        from collections import Counter
        counts = Counter(new_ids)
        for new_id, count in counts.items():
            assert count <= 2, (
                f"batch {b}: new id {new_id} has {count} originals — a token "
                "participated in more than one merge"
            )


# -------------------------------------------------------------------- #
# Total compression budget honored                                      #
# -------------------------------------------------------------------- #


def test_total_merges_equals_r():
    B, L, W, D = 1, 16, 4, 4
    torch.manual_seed(11)
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=4, W=W, valid=valid)
    assert int(plan.L_new[0]) == L - 4


def test_r_zero_is_identity():
    B, L, W, D = 2, 12, 4, 4
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=0, W=W, valid=valid)
    assert int(plan.L_new[0]) == L
    # old_to_new must be 0..L-1 (identity).
    expected = torch.arange(L).unsqueeze(0).expand(B, L)
    assert torch.equal(plan.old_to_new, expected)


# -------------------------------------------------------------------- #
# Mass and size invariants under apply_merge_plan                       #
# -------------------------------------------------------------------- #


def test_size_sum_invariant():
    torch.manual_seed(3)
    B, L, W, D = 2, 16, 4, 6
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=4, W=W, valid=valid)

    for b in range(B):
        valid_size_sum = int(plan.new_size[b, : int(plan.L_new[b])].sum())
        assert valid_size_sum == L


def test_weighted_average_preserves_total_mass():
    """``Σ_l size_new[l] * x_new[l] == Σ_k size_old[k] * x_old[k]`` per dim."""
    torch.manual_seed(5)
    B, L, W, D = 2, 16, 4, 5
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=4, W=W, valid=valid)
    # Use a different feature tensor (not the metric) to be apply target.
    x = torch.randn(B, L, D)
    x_new, size_new = apply_merge_plan(x, size, plan)

    # Old total mass.
    old_mass = (x * size.unsqueeze(-1).float()).sum(dim=1)            # [B, D]
    new_mass = (x_new * size_new.unsqueeze(-1).float()).sum(dim=1)    # [B, D]
    assert torch.allclose(old_mass, new_mass, atol=1e-5), (
        f"max diff = {(old_mass - new_mass).abs().max():.2e}"
    )


# -------------------------------------------------------------------- #
# Composition with SourceMap                                            #
# -------------------------------------------------------------------- #


def test_apply_to_source_map_invariants():
    torch.manual_seed(9)
    B, L, W, D = 2, 16, 4, 6
    metric = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=6, W=W, valid=valid)
    sm = SourceMap.identity(B=B, N=L).apply_merge(plan)

    for b in range(B):
        # parent in range
        assert int(sm.parent[b].max()) < int(sm.L[b])
        # size sum invariant
        assert int(sm.size[b, : int(sm.L[b])].sum()) == L
        # all sizes positive on valid tokens
        assert (sm.size[b, : int(sm.L[b])] > 0).all()


# -------------------------------------------------------------------- #
# Greedy ranking: best-globally-available pair is taken first           #
# -------------------------------------------------------------------- #


def test_greedy_takes_best_pair_first():
    """If only one merge is allowed, it should be the globally most-similar pair."""
    B, L, W, D = 1, 8, 4, 4
    # Most-similar pair: (1, 2). All other pairs less similar.
    metric = torch.randn(B, L, D) * 0.01
    metric[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 2] = torch.tensor([1.0, 0.001, 0.0, 0.0])
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric, size, r=1, W=W, valid=valid)
    o2n = plan.old_to_new[0]
    assert int(o2n[1]) == int(o2n[2]), "greedy did not pick the best pair"
