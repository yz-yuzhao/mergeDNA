"""Tests for ``mergedna.modules.source_map.SourceMap``.

Covered:
  - identity invariants
  - apply_merge: parent/size/L bookkeeping after a synthetic merge
  - gather_to_nucleotides: identity case + post-merge round-trip
  - propagate_mask: boolean broadcast through parent array
"""

from __future__ import annotations

import torch

from mergedna.modules.source_map import SourceMap
from mergedna.modules.merge import MergePlan


# ----- helpers ------------------------------------------------------------


def _identity_plan(B: int, L: int) -> MergePlan:
    """A merge plan that does nothing: every token maps to itself."""
    old_to_new = torch.arange(L).unsqueeze(0).expand(B, L).contiguous()
    new_size = torch.ones(B, L, dtype=torch.long)
    L_new = torch.full((B,), L, dtype=torch.long)
    return MergePlan(old_to_new=old_to_new, new_size=new_size, L_new=L_new, L_new_max=L)


def _pair_merge_plan(B: int, L: int) -> MergePlan:
    """Merge tokens 0 and 1 into one (id 0); shift the rest down by one.

    For B=1, L=4:
        old_to_new = [0, 0, 1, 2]
        new_size   = [2, 1, 1, 0]   (last slot is padding)
        L_new      = [3]
    """
    assert L >= 2
    L_new_v = L - 1
    L_new_max = L_new_v  # compact
    old_to_new = torch.zeros(B, L, dtype=torch.long)
    # token 0 stays at id 0; token 1 also goes to id 0; tokens 2..L-1 -> ids 1..L-2
    old_to_new[:, 0] = 0
    old_to_new[:, 1] = 0
    old_to_new[:, 2:] = torch.arange(1, L_new_v).unsqueeze(0).expand(B, -1)
    new_size = torch.ones(B, L_new_max, dtype=torch.long)
    new_size[:, 0] = 2  # the merged token has size 2
    L_new = torch.full((B,), L_new_v, dtype=torch.long)
    return MergePlan(old_to_new=old_to_new, new_size=new_size, L_new=L_new, L_new_max=L_new_max)


# ----- identity -----------------------------------------------------------


def test_identity_invariants():
    sm = SourceMap.identity(B=3, N=8)
    assert sm.parent.shape == (3, 8)
    assert torch.equal(sm.parent[0], torch.arange(8))
    assert torch.equal(sm.size, torch.ones(3, 8, dtype=torch.long))
    assert torch.equal(sm.L, torch.full((3,), 8, dtype=torch.long))
    assert sm.token_mask.all()


def test_identity_size_sums_to_N():
    sm = SourceMap.identity(B=2, N=16)
    for b in range(2):
        assert int(sm.size[b, : sm.L[b]].sum()) == sm.N


# ----- apply_merge --------------------------------------------------------


def test_apply_identity_merge_is_a_noop():
    sm = SourceMap.identity(B=2, N=4)
    sm2 = sm.apply_merge(_identity_plan(B=2, L=4))
    assert torch.equal(sm2.parent, sm.parent)
    assert torch.equal(sm2.size, sm.size)
    assert torch.equal(sm2.L, sm.L)


def test_apply_pair_merge_updates_parent_size_L():
    sm = SourceMap.identity(B=1, N=4)
    sm2 = sm.apply_merge(_pair_merge_plan(B=1, L=4))

    # Nucleotides 0, 1 now share parent 0; nuc 2 -> parent 1; nuc 3 -> parent 2.
    assert sm2.parent.tolist() == [[0, 0, 1, 2]]
    assert sm2.L.tolist() == [3]
    # Sizes: [2, 1, 1] over the 3 valid tokens.
    valid = sm2.size[0, : int(sm2.L[0])]
    assert valid.tolist() == [2, 1, 1]
    # Sum-to-N invariant.
    assert int(valid.sum()) == sm.N
    # parent indices are all in-range.
    assert int(sm2.parent.max()) < int(sm2.L.min())


def test_apply_merge_token_mask_marks_padding_invalid():
    sm = SourceMap.identity(B=1, N=4)
    sm2 = sm.apply_merge(_pair_merge_plan(B=1, L=4))
    assert sm2.token_mask.tolist() == [[True, True, True]]


# ----- gather_to_nucleotides ----------------------------------------------


def test_gather_identity_returns_input():
    sm = SourceMap.identity(B=2, N=5)
    x = torch.randn(2, 5, 3)
    y = sm.gather_to_nucleotides(x)
    assert torch.equal(y, x)


def test_gather_after_merge_broadcasts_within_group():
    """After merging tokens 0 and 1, nucleotides 0 and 1 share the same feature."""
    sm = SourceMap.identity(B=1, N=4).apply_merge(_pair_merge_plan(B=1, L=4))
    # Build [B=1, L_max=3, D=2] features.
    x = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
    y = sm.gather_to_nucleotides(x)
    # Nucleotides 0, 1 -> token 0; nuc 2 -> token 1; nuc 3 -> token 2.
    expected = torch.tensor(
        [[[10.0, 11.0], [10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]]
    )
    assert torch.equal(y, expected)


# ----- propagate_mask ----------------------------------------------------


def test_propagate_mask_identity():
    sm = SourceMap.identity(B=1, N=4)
    m_tok = torch.tensor([[True, False, True, False]])
    m_nuc = sm.propagate_mask(m_tok)
    assert torch.equal(m_nuc, m_tok)


def test_propagate_mask_after_merge():
    sm = SourceMap.identity(B=1, N=4).apply_merge(_pair_merge_plan(B=1, L=4))
    # Mask token 0 (which now contains nucleotides 0 and 1) and token 2.
    m_tok = torch.tensor([[True, False, True]])
    m_nuc = sm.propagate_mask(m_tok)
    assert m_nuc.tolist() == [[True, True, False, True]]


# ----- to_dense ----------------------------------------------------------


def test_to_dense_columns_are_one_hot_in_parent():
    """Each nucleotide has exactly one nonzero entry in its column of S."""
    sm = SourceMap.identity(B=1, N=4).apply_merge(_pair_merge_plan(B=1, L=4))
    S = sm.to_dense()  # [1, 3, 4]
    # Column n has exactly one nonzero in row parent[0, n]; value = 1/size[parent].
    for n in range(4):
        col = S[0, :, n]
        nonzero = (col > 0).nonzero().flatten().tolist()
        assert nonzero == [int(sm.parent[0, n])]
        assert float(col[nonzero[0]]) == 1.0 / float(sm.size[0, nonzero[0]])
