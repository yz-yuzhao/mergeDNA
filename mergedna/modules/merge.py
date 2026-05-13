"""Token merging operations.

Two distinct merge strategies — see the README's architecture overview:

  - ``all_pairs_match_window`` — used by the local encoder. Within each
    window of tokens, computes the full pair-wise cosine similarity, then
    greedily picks the top-r pairs in descending similarity such that each
    token participates in at most one merge.

  - ``bipartite_match_global`` — used by the latent encoder.
    Standard ToMe: split tokens into A (even) and B (odd), each A is matched
    to its best B, top-r get merged into their match.

Both produce a ``MergePlan``, which describes how the current token sequence
of length L_old is being merged into L_new tokens.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from mergedna.modules.windows import pad_to_multiple


@dataclass
class MergePlan:
    """Describes a single merge step.

    Fields:
        old_to_new: LongTensor ``[B, L_old]``. ``old_to_new[b, k]`` is the new
            token index that current-token ``k`` maps to. Multiple old indices
            may map to the same new index (those are the merged ones).
        new_size:   LongTensor ``[B, L_new_max]``. Size of each new token
            (sum of sizes of old tokens mapping to it). Padding entries (when
            ``L_new[b] < L_new_max``) are 0.
        L_new:      LongTensor ``[B]``. Per-batch valid count after the merge.
            With one-r-per-step sampling all entries are equal.
        L_new_max:  int. ``= max(L_new)``; the size of the new tensors.
    """

    old_to_new: torch.Tensor
    new_size: torch.Tensor
    L_new: torch.Tensor
    L_new_max: int


def update_positions(
    positions: torch.Tensor,
    size: torch.Tensor,
    plan: MergePlan,
) -> torch.Tensor:
    """Update per-token positions through a merge step.

    A merged token's position is the size-weighted centroid of its constituent
    tokens.

    Args:
        positions: ``[B, L_old]`` positions of the current tokens.
        size:      ``[B, L_old]`` current sizes (before the merge).
        plan:      a ``MergePlan`` describing the merge.

    Returns:
        ``[B, L_new_max]`` float positions for the merged tokens.
    """
    if positions.dim() != 2:
        raise ValueError(f"positions must be 2-D, got shape {tuple(positions.shape)}")
    B, L_old = positions.shape
    L_new_max = plan.L_new_max
    weighted = positions * size.to(positions.dtype)
    out = torch.zeros(B, L_new_max, device=positions.device, dtype=positions.dtype)
    out.scatter_add_(dim=1, index=plan.old_to_new, src=weighted)
    denom = plan.new_size.to(positions.dtype).clamp_min(1.0)
    return out / denom


def apply_merge_plan(
    x: torch.Tensor,
    size: torch.Tensor,
    plan: MergePlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Size-weighted aggregation of features.

    Args:
        x:    ``[B, L_old, D]`` feature tensor.
        size: ``[B, L_old]`` sizes (number of original nucleotides per token).
        plan: a ``MergePlan`` whose ``old_to_new`` covers the L_old axis.

    Returns:
        ``(x_new, new_size)`` with shapes ``[B, L_new_max, D]`` and
        ``[B, L_new_max]``.
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}")
    B, L_old, D = x.shape
    L_new_max = plan.L_new_max

    weighted = x * size.to(x.dtype).unsqueeze(-1)  # [B, L_old, D]
    out = torch.zeros(B, L_new_max, D, device=x.device, dtype=x.dtype)
    idx = plan.old_to_new.unsqueeze(-1).expand(-1, -1, D)
    out.scatter_add_(dim=1, index=idx, src=weighted)

    # Avoid division by 0 for padding entries (their numerator is also 0).
    denom = plan.new_size.to(x.dtype).clamp_min(1.0).unsqueeze(-1)
    out = out / denom
    return out, plan.new_size.clone()


def soft_apply_merge_window(
    x: torch.Tensor,
    size: torch.Tensor,
    metric: torch.Tensor,
    r: int,
    W: int,
    valid: torch.Tensor,
    plan: MergePlan,
    tau: float = 0.1,
) -> torch.Tensor:
    """DTEM-style soft surrogate for ``apply_merge_plan(x, size, plan)[0]``.

    Used as the differentiable branch of a straight-through estimator:

        x_out = x_hard + x_soft - x_soft.detach()

    Forward value is exactly ``x_hard`` (the bracketed difference is identically
    zero), but the autograd graph runs through ``x_soft``, which depends on
    ``metric`` via continuous soft pair selection. This is what gives the
    grouping head a gradient signal — without it, the hard top-r in
    :func:`all_pairs_match_window` returns only integer indices and the head
    is stranded off the training path.

    Mechanism (DTEM §3.2-3.3, adapted to all-pairs-within-window):

      1. Pair scores are the cosine similarities of ``metric`` restricted to the
         strict upper triangle within each window.
      2. Iterated soft-argmax with suppression: for ``t = 1..r_per_window``,
         compute ``A^t = softmax((S^t) / τ)`` and accumulate it into a soft
         pair-indicator ``E~ ∈ [0,1]^{num_pairs}``. After each step we suppress
         the endpoints of softly-selected pairs by subtracting
         ``SUPPRESS * (u_i + u_j)`` from ``S^{t+1}``, so subsequent steps pick
         disjoint pairs (mimicking the "each token in ≤ 1 pair" rule).
      3. Soft merging: each old position ``k`` keeps fraction ``1 - outgoing[k]``
         of its mass and absorbs ``Σ_b E~_{k,b} · size[b]`` mass from
         higher-indexed partners. The resulting size-weighted features are
         aggregated into the new layout via ``plan.old_to_new``.

    In the discrete limit (``E~`` matching the hard plan exactly), the output
    equals ``apply_merge_plan``'s exactly, so the STE bridge has zero forward
    bias.

    Args:
        x:       ``[B, L, D]`` features (pre-merge).
        size:    ``[B, L]`` current sizes (pre-merge).
        metric:  ``[B, L, D_metric]`` grouping-head output.
        r:       total tokens to remove (same budget as the hard plan).
        W:       window size.
        valid:   ``[B, L]`` BoolTensor.
        plan:    the hard ``MergePlan`` returned by ``all_pairs_match_window``.
        tau:     softmax temperature for the soft-argmax (DTEM default 0.1).
    """
    if x.dim() != 3:
        raise ValueError(f"x must be 3-D, got shape {tuple(x.shape)}")
    B, L, D = x.shape
    device = x.device
    dtype = x.dtype

    metric_p, _ = pad_to_multiple(metric, W, dim=1, value=0.0)
    valid_p, _ = pad_to_multiple(valid, W, dim=1, value=False)
    x_p, _ = pad_to_multiple(x, W, dim=1, value=0.0)
    size_p, _ = pad_to_multiple(size, W, dim=1, value=0)
    L_pad = metric_p.shape[1]
    n_w = L_pad // W

    m_w = metric_p.view(B, n_w, W, -1)
    valid_w = valid_p.view(B, n_w, W)
    x_w = x_p.view(B, n_w, W, D)
    size_w = size_p.view(B, n_w, W).to(dtype)

    m_norm = F.normalize(m_w, dim=-1, eps=1e-8)
    sim = m_norm @ m_norm.transpose(-1, -2)                # [B, n_w, W, W]

    pair_i, pair_j = torch.triu_indices(W, W, offset=1, device=device).unbind(0)
    num_pairs = pair_i.shape[0]
    pair_scores = sim[:, :, pair_i, pair_j]                # [B, n_w, num_pairs]
    pair_valid = valid_w[:, :, pair_i] & valid_w[:, :, pair_j]
    # Large finite negative so softmax stays well-defined for invalid pairs.
    NEG = -1e4
    pair_scores = torch.where(pair_valid, pair_scores, torch.full_like(pair_scores, NEG))

    base, rem = divmod(r, max(n_w, 1))
    r_per_window = torch.full((n_w,), base, dtype=torch.long, device=device)
    r_per_window[:rem] += 1
    r_max = int(r_per_window.max().item()) if n_w > 0 else 0

    E_pair = torch.zeros(B, n_w, num_pairs, device=device, dtype=dtype)

    if r_max > 0:
        used_node = torch.zeros(B, n_w, W, device=device, dtype=dtype)
        pair_i_b = pair_i.view(1, 1, -1).expand(B, n_w, num_pairs)
        pair_j_b = pair_j.view(1, 1, -1).expand(B, n_w, num_pairs)
        # active[t, w]: 1 if this window still has budget at step t.
        steps = torch.arange(r_max, device=device).unsqueeze(1)        # [r_max, 1]
        active_mask = (steps < r_per_window.unsqueeze(0)).to(dtype)    # [r_max, n_w]
        # Suppression strength: with τ=0.1 and similarities in [-1, 1], setting
        # SUPPRESS=10 multiplies an already-used pair's softmax exponent by
        # exp(-100), which is more than enough to zero its probability.
        SUPPRESS = 10.0

        for t in range(r_max):
            u_i = used_node.gather(dim=2, index=pair_i_b)
            u_j = used_node.gather(dim=2, index=pair_j_b)
            s_t = pair_scores - SUPPRESS * (u_i + u_j)
            a_t = F.softmax(s_t / tau, dim=-1) * active_mask[t].view(1, n_w, 1)
            E_pair = E_pair + a_t
            # Out-of-place updates — autograd needs to read prior `used_node`
            # states in subsequent iterations, which the in-place form would
            # invalidate (version-counter mismatch on ScatterAddBackward).
            used_node = used_node.scatter_add(2, pair_i_b, a_t)
            used_node = used_node.scatter_add(2, pair_j_b, a_t)

    E_pair = E_pair.clamp(0.0, 1.0)

    # Soft adjacency in [B, n_w, W, W] (strict upper triangle). Built via
    # out-of-place scatter to keep the autograd graph clean.
    flat_idx = (pair_i * W + pair_j).view(1, 1, -1).expand(B, n_w, num_pairs)
    E_full = torch.zeros(B, n_w, W * W, device=device, dtype=dtype).scatter_add(
        2, flat_idx, E_pair
    ).view(B, n_w, W, W)

    # outgoing[k] = Σ_{a < k} E_full[a, k]; mass position k loses to lower keepers.
    # incoming via einsum over upper-triangle: each i absorbs Σ_{j > i} E_full[i, j] · size[j] · x[j].
    sized_x = size_w.unsqueeze(-1) * x_w                               # [B, n_w, W, D]
    x_incoming = torch.einsum("bwij,bwjd->bwid", E_full, sized_x)      # [B, n_w, W, D]
    size_incoming = torch.einsum("bwij,bwj->bwi", E_full, size_w)      # [B, n_w, W]
    outgoing = E_full.sum(dim=-2)                                      # [B, n_w, W]
    frac_remain = (1.0 - outgoing).clamp_min(0.0)
    size_remain = frac_remain * size_w
    size_soft = size_remain + size_incoming
    x_soft_pos = (size_remain.unsqueeze(-1) * x_w + x_incoming) / size_soft.clamp_min(1e-8).unsqueeze(-1)

    # Project to [B, L_new_max, D] via the hard plan, size-weighted by soft mass.
    x_soft_flat = x_soft_pos.view(B, L_pad, D)[:, :L, :].contiguous()
    size_soft_flat = size_soft.view(B, L_pad)[:, :L].contiguous()

    L_new_max = plan.L_new_max
    weighted = x_soft_flat * size_soft_flat.unsqueeze(-1)
    out_num = torch.zeros(B, L_new_max, D, device=device, dtype=dtype)
    idx = plan.old_to_new.unsqueeze(-1).expand(-1, -1, D)
    out_num.scatter_add_(dim=1, index=idx, src=weighted)
    out_den = torch.zeros(B, L_new_max, device=device, dtype=dtype)
    out_den.scatter_add_(dim=1, index=plan.old_to_new, src=size_soft_flat)
    return out_num / out_den.clamp_min(1e-8).unsqueeze(-1)


def all_pairs_match_window(
    metric: torch.Tensor,
    size: torch.Tensor,
    r: int,
    W: int,
    valid: torch.Tensor,
) -> MergePlan:
    """Greedy top-r all-pairs matching per window.

    Algorithm:
        1. Pad the sequence axis to a multiple of window size ``W``; reshape to
           ``[B, n_w, W, D_metric]``.
        2. For each window, compute the full ``W × W`` cosine-similarity matrix.
        3. Take the strict upper triangle (``i < j``) — i.e. all unordered pairs
           within the window. Mask any pair containing an invalid token
           with ``-inf``.
        4. Distribute the global merge budget ``r`` across windows
        5. Per window, sort pairs by similarity descending. Scan pairs in order:
           accept a pair if neither token is already in an accepted pair AND we
           have not yet reached ``r_per_window[w]`` merges. Stop when full.
        6. For each accepted pair ``(i, j)`` with ``i < j``: token ``j`` merges
           into token ``i`` (lower-index keeper).
        7. Build ``old_to_new``.

    Args:
        metric: ``[B, L, D_metric]`` features used as the similarity metric
            (typically the output of the grouping head).
        size:   ``[B, L]`` current sizes — not used for matching, but needed
            to compute ``new_size`` in the returned ``MergePlan``.
        r:      total number of tokens to remove across the full sequence.
        W:      window size.
        valid:  ``[B, L]`` BoolTensor; ``True`` where the token is valid.

    Returns:
        A ``MergePlan`` whose ``old_to_new`` is on ``[B, L]`` (the unpadded
        length).
    """
    if metric.dim() != 3:
        raise ValueError(f"metric must be 3-D, got shape {tuple(metric.shape)}")
    B, L, _ = metric.shape
    device = metric.device

    if r < 0:
        raise ValueError(f"r must be non-negative, got {r}")
    # Pad sequence axis to multiple of W. Padding metric/valid/size — for sizes
    # the value 0 is correct (padded tokens are size-0 placeholders).
    metric_p, _ = pad_to_multiple(metric, W, dim=1, value=0.0)
    valid_p, _ = pad_to_multiple(valid, W, dim=1, value=False)
    L_pad = metric_p.shape[1]
    n_w = L_pad // W

    # pairwise cosine similarity per window
    m_w = metric_p.view(B, n_w, W, -1)
    valid_w = valid_p.view(B, n_w, W)

    m_norm = F.normalize(m_w, dim=-1, eps=1e-8)
    sim = m_norm @ m_norm.transpose(-1, -2)  # [B, n_w, W, W]

    # Strict upper-triangle pair list (W*(W-1)/2 pairs).
    pair_i, pair_j = torch.triu_indices(W, W, offset=1, device=device).unbind(0)
    num_pairs = pair_i.shape[0]
    pair_scores = sim[:, :, pair_i, pair_j]                # [B, n_w, num_pairs]
    pair_valid = valid_w[:, :, pair_i] & valid_w[:, :, pair_j]
    pair_scores = pair_scores.masked_fill(~pair_valid, float("-inf"))

    # Sort pairs by score descending per (b, w).
    sorted_scores, sorted_idx = pair_scores.sort(dim=-1, descending=True)

    # per-window budget allocation
    base, rem = divmod(r, max(n_w, 1))
    r_per_window = torch.full((n_w,), base, dtype=torch.long, device=device)
    r_per_window[:rem] += 1                                # [n_w]
    r_per_window_b = r_per_window.unsqueeze(0).expand(B, n_w).contiguous()

    # greedy scan
    used = torch.zeros(B, n_w, W, dtype=torch.long, device=device)   # int (0/1) for scatter
    accept_count = torch.zeros(B, n_w, dtype=torch.long, device=device)
    # For each accepted pair: keeper index (i_k) and merger index (j_k).
    keeper_idx = torch.zeros(B, n_w, W, dtype=torch.long, device=device)
    is_merged = torch.zeros(B, n_w, W, dtype=torch.bool, device=device)

    for k in range(num_pairs):
        rank_pair_idx = sorted_idx[:, :, k]                 # [B, n_w]
        i_k = pair_i[rank_pair_idx]                         # [B, n_w]
        j_k = pair_j[rank_pair_idx]                         # [B, n_w]

        i_used = used.gather(dim=2, index=i_k.unsqueeze(-1)).squeeze(-1)
        j_used = used.gather(dim=2, index=j_k.unsqueeze(-1)).squeeze(-1)
        score_finite = sorted_scores[:, :, k] > float("-inf")
        room = accept_count < r_per_window_b
        accept = (i_used == 0) & (j_used == 0) & score_finite & room  # [B, n_w]

        if not accept.any():
            # Either every (b, w) is full or out of valid pairs at this rank;
            # later ranks have lower scores so future iterations may still
            # accept. We can't early-exit safely. Continue.
            continue

        accept_long = accept.to(torch.long)
        used.scatter_add_(dim=2, index=i_k.unsqueeze(-1), src=accept_long.unsqueeze(-1))
        used.scatter_add_(dim=2, index=j_k.unsqueeze(-1), src=accept_long.unsqueeze(-1))

        # Mark j as merged into i.
        is_merged.scatter_(
            dim=2,
            index=j_k.unsqueeze(-1),
            src=(is_merged.gather(dim=2, index=j_k.unsqueeze(-1)) | accept.unsqueeze(-1)),
        )
        cur_keeper = keeper_idx.gather(dim=2, index=j_k.unsqueeze(-1))
        new_keeper = torch.where(accept.unsqueeze(-1), i_k.unsqueeze(-1), cur_keeper)
        keeper_idx.scatter_(dim=2, index=j_k.unsqueeze(-1), src=new_keeper)

        accept_count = accept_count + accept_long

    # build old_to_new
    # A "kept" token is one that is valid AND not absorbed into another.
    kept = valid_w & (~is_merged)                          # [B, n_w, W]
    kept_long = kept.to(torch.long)

    # Local new id within each window (only meaningful where kept=True).
    new_id_local = kept_long.cumsum(dim=-1) - 1            # [B, n_w, W]

    # Global offset across windows: exclusive cumsum of kept counts per window.
    kept_per_window = kept_long.sum(dim=-1)                # [B, n_w]
    offset = kept_per_window.cumsum(dim=-1) - kept_per_window  # [B, n_w]

    new_id_global_kept = new_id_local + offset.unsqueeze(-1)  # [B, n_w, W]

    # For merged tokens, look up the keeper's new id.
    keeper_new_id = new_id_global_kept.gather(dim=2, index=keeper_idx)  # [B, n_w, W]
    new_id_global = torch.where(is_merged, keeper_new_id, new_id_global_kept)

    # Padded-or-invalid tokens get a sentinel id 0 — they won't be consulted by
    # the source map (its parent only points to valid ids), but we must keep
    # the value in-range so scatter_add downstream is safe.
    new_id_global = torch.where(valid_w, new_id_global, torch.zeros_like(new_id_global))

    # Flatten to [B, L_pad] and trim padding.
    old_to_new_padded = new_id_global.view(B, L_pad)
    old_to_new = old_to_new_padded[:, :L].contiguous()

    # L_new and new_size
    L_new = kept_per_window.sum(dim=-1)                   # [B]
    L_new_max = int(L_new.max().item()) if B > 0 else 0
    if L_new_max == 0:
        # Edge case (e.g. r == L) — keep at least 1 to avoid zero-size tensors.
        L_new_max = 1

    # new_size[b, l] = Σ_{k: old_to_new[b,k]==l, valid[b,k]} size[b, k]
    src_size = (size * valid.to(size.dtype))               # zero out invalid contributions
    new_size = torch.zeros(B, L_new_max, dtype=torch.long, device=device)
    new_size.scatter_add_(dim=1, index=old_to_new.long(), src=src_size.long())

    return MergePlan(old_to_new=old_to_new.long(), new_size=new_size, L_new=L_new, L_new_max=L_new_max)


def bipartite_match_global(
    metric: torch.Tensor,
    size: torch.Tensor,
    r: int,
    valid: torch.Tensor,
) -> MergePlan:
    """Standard ToMe bipartite soft matching.

    A = even-indexed tokens, B = odd-indexed. For each ``A_i`` the best
    matching B is found by cosine similarity; the top-r As (by best-match
    score) are merged into their respective Bs. All Bs and the unmerged As
    are kept. 

    Args:
        metric: ``[B, L, D]`` similarity features.
        size:   ``[B, L]`` current sizes.
        r:      total number of tokens to remove (clamped to ``≤ L_a``).
        valid:  ``[B, L]`` BoolTensor.
    """
    if metric.dim() != 3:
        raise ValueError(f"metric must be 3-D, got shape {tuple(metric.shape)}")
    Bsz, L, _ = metric.shape
    device = metric.device

    if L < 2:
        # Nothing to merge.
        return _identity_plan(size, valid)

    a_idx = torch.arange(0, L, 2, device=device)
    b_idx = torch.arange(1, L, 2, device=device)
    L_a = a_idx.shape[0]
    L_b = b_idx.shape[0]

    if r <= 0 or L_a == 0 or L_b == 0:
        return _identity_plan(size, valid)

    metric_a = metric[:, a_idx, :]
    metric_b = metric[:, b_idx, :]
    valid_a = valid[:, a_idx]
    valid_b = valid[:, b_idx]

    a_norm = F.normalize(metric_a, dim=-1, eps=1e-8)
    b_norm = F.normalize(metric_b, dim=-1, eps=1e-8)

    sim = a_norm @ b_norm.transpose(-1, -2)               # [B, L_a, L_b]
    # Mask invalid Bs (whole columns) and invalid As (whole rows).
    sim = sim.masked_fill(~valid_b.unsqueeze(1), float("-inf"))
    sim = sim.masked_fill(~valid_a.unsqueeze(2), float("-inf"))

    best_score, best_b_local = sim.max(dim=-1)            # [B, L_a]

    r_eff = min(r, L_a)
    _, top_a_local = best_score.topk(r_eff, dim=-1)        # [B, r_eff]
    merger_global = a_idx[top_a_local]                     # [B, r_eff]
    dst_local = best_b_local.gather(dim=-1, index=top_a_local)
    dst_global = b_idx[dst_local]                          # [B, r_eff]

    # is_merged: [B, L]
    is_merged = torch.zeros(Bsz, L, dtype=torch.bool, device=device)
    is_merged.scatter_(dim=1, index=merger_global, src=torch.ones_like(merger_global, dtype=torch.bool))

    kept = valid & (~is_merged)
    kept_long = kept.to(torch.long)
    new_id_kept = kept_long.cumsum(dim=-1) - 1             # [B, L]

    # For merged tokens, new id = kept-id of their dst.
    dst_new_id = new_id_kept.gather(dim=-1, index=dst_global)  # [B, r_eff]
    old_to_new = new_id_kept.clone()
    old_to_new.scatter_(dim=-1, index=merger_global, src=dst_new_id)
    old_to_new = old_to_new.clamp_min(0)                   # safety for invalid prefix

    L_new = kept_long.sum(dim=-1)                          # [B]
    L_new_max = int(L_new.max().item()) if Bsz > 0 else 0
    if L_new_max == 0:
        L_new_max = 1

    src_size = (size * valid.to(size.dtype)).long()
    new_size = torch.zeros(Bsz, L_new_max, dtype=torch.long, device=device)
    new_size.scatter_add_(dim=1, index=old_to_new.long(), src=src_size)

    return MergePlan(old_to_new=old_to_new.long(), new_size=new_size, L_new=L_new, L_new_max=L_new_max)


def _identity_plan(size: torch.Tensor, valid: torch.Tensor) -> MergePlan:
    """A no-op plan: each token maps to itself."""
    B, L = size.shape
    device = size.device
    old_to_new = torch.arange(L, device=device, dtype=torch.long).unsqueeze(0).expand(B, L).contiguous()
    new_size = (size * valid.to(size.dtype)).long().clone()
    L_new = valid.long().sum(dim=-1)
    return MergePlan(old_to_new=old_to_new, new_size=new_size, L_new=L_new, L_new_max=L)
