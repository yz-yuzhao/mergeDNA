"""Source map: a compact tracker for nucleotide ↔ merged-token correspondence.

After several layers of merging, every original nucleotide still belongs to
exactly one *current* token. The standard way to express this is a binary
matrix ``S ∈ {0, 1}^{L × N}``; we instead store the equivalent **parent array**
``parent ∈ ℕ^N`` with ``parent[n]`` the current token index that nucleotide
``n`` belongs to. This is O(B·N) memory instead of O(B·L·N), and makes
unmerge a single ``gather`` instead of a matmul.

Two operations dominate downstream usage:

  - ``gather_to_nucleotides`` — broadcast token features back to nucleotide
    resolution (used by the local decoder before its attention blocks).
  - ``propagate_mask`` — broadcast a token-level mask back to nucleotide
    resolution (used by the adaptive masking pipeline).

Both are single ``gather`` ops along the token axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mergedna.modules.merge import MergePlan


@dataclass
class SourceMap:
    """Maps original nucleotide positions to *current* token indices.

    Fields:
        parent:     LongTensor ``[B, N]`` — ``parent[b, n] ∈ [0, L[b])``.
        size:       LongTensor ``[B, L_max]`` — number of originals per current token.
        L:          LongTensor ``[B]`` — per-batch valid token count.
                    With one-r-per-step sampling all entries are equal, but the
                    field is kept per-batch for generality.
        token_mask: BoolTensor ``[B, L_max]`` — ``True`` where the token is valid.

    Invariants (checked in tests):
        - ``size[b, :L[b]].sum() == N`` for every ``b``.
        - ``parent[b].max() < L[b]``.
        - ``size[b, l] > 0`` for every valid ``l``.
        - ``size[b, l] == 0`` for every invalid (padding) ``l``.
    """

    parent: torch.Tensor
    size: torch.Tensor
    L: torch.Tensor
    token_mask: torch.Tensor

    @property
    def B(self) -> int:
        return self.parent.shape[0]

    @property
    def N(self) -> int:
        return self.parent.shape[1]

    @property
    def L_max(self) -> int:
        return self.size.shape[1]

    @staticmethod
    def identity(B: int, N: int, device: torch.device | str | None = None) -> "SourceMap":
        """Return the identity map: every nucleotide is its own token."""
        device = torch.device(device) if device is not None else torch.device("cpu")
        parent = torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, N).contiguous()
        size = torch.ones(B, N, device=device, dtype=torch.long)
        L = torch.full((B,), N, device=device, dtype=torch.long)
        token_mask = torch.ones(B, N, device=device, dtype=torch.bool)
        return SourceMap(parent=parent, size=size, L=L, token_mask=token_mask)

    def apply_merge(self, plan: "MergePlan") -> "SourceMap":
        """Return a new SourceMap reflecting the given merge step.

        ``plan.old_to_new[b, l_old] = l_new`` says where each old-token id is
        sent. We just look up the new id for each nucleotide's current parent.

        Args:
            plan: A ``MergePlan`` describing how the *current* token sequence
                  is being compacted into ``L_new`` tokens.
        """
        # parent_new[b, n] = old_to_new[b, parent_old[b, n]]
        parent_new = plan.old_to_new.gather(dim=1, index=self.parent)
        L_new_max = plan.new_size.shape[1]
        # token_mask_new[b, l] = l < L_new[b]
        token_mask_new = (
            torch.arange(L_new_max, device=self.parent.device).unsqueeze(0)
            < plan.L_new.unsqueeze(1)
        )
        return SourceMap(
            parent=parent_new,
            size=plan.new_size.clone(),
            L=plan.L_new.clone(),
            token_mask=token_mask_new,
        )

    def gather_to_nucleotides(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Broadcast token features back to nucleotide resolution.

        Args:
            x_tokens: ``[B, L_max, D]`` feature tensor over current tokens.

        Returns:
            ``[B, N, D]`` — ``out[b, n] = x_tokens[b, parent[b, n]]``.
        """
        if x_tokens.dim() != 3:
            raise ValueError(f"x_tokens must be 3-D, got shape {tuple(x_tokens.shape)}")
        B, L, D = x_tokens.shape
        if B != self.B:
            raise ValueError(f"batch size mismatch: x has B={B}, source map has B={self.B}")
        idx = self.parent.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
        return x_tokens.gather(dim=1, index=idx)

    def propagate_mask(self, mask_tokens: torch.Tensor) -> torch.Tensor:
        """Broadcast a token-level boolean mask to nucleotide resolution.

        Args:
            mask_tokens: ``[B, L_max]`` BoolTensor.

        Returns:
            ``[B, N]`` BoolTensor — every nucleotide whose parent token is True
            is itself True.
        """
        if mask_tokens.dim() != 2:
            raise ValueError(f"mask_tokens must be 2-D, got shape {tuple(mask_tokens.shape)}")
        return mask_tokens.gather(dim=1, index=self.parent)

    def token_positions(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Centroid of source nucleotide indices per current token.

        ``pos[b, ℓ] = (Σ_n n · 1[parent[b, n] == ℓ]) / size[b, ℓ]``. Used by
        RoPE: every token (merged or not) carries a single coordinate in the
        original nucleotide space, so attention rotations are consistent across
        merge layers and across encoder/decoder stages.

        Args:
            dtype: float dtype for the returned positions.

        Returns:
            ``[B, L_max]`` tensor; padding entries have a 0.0 placeholder.
        """
        B, N = self.parent.shape
        L_max = self.size.shape[1]
        idx = torch.arange(N, device=self.parent.device, dtype=dtype).expand(B, N)
        out = torch.zeros(B, L_max, device=self.parent.device, dtype=dtype)
        out.scatter_add_(dim=1, index=self.parent, src=idx)
        denom = self.size.to(dtype).clamp_min(1.0)
        return out / denom

    def to_dense(self) -> torch.Tensor:
        """Return the dense ``S ∈ ℝ^{B × L_max × N}`` matrix.

        ``S[b, l, n] = 1 / size[b, l]`` if ``parent[b, n] == l``, else 0. The
        normalization makes ``S @ S^T`` close to identity on valid tokens. Used
        only for debugging — the parent array is the authoritative store.
        """
        B, N = self.parent.shape
        L_max = self.size.shape[1]
        # one-hot via scatter
        S = torch.zeros(B, L_max, N, device=self.parent.device, dtype=torch.float32)
        # S[b, parent[b, n], n] = 1
        S.scatter_(dim=1, index=self.parent.unsqueeze(1), src=torch.ones(B, 1, N, device=self.parent.device))
        # divide by size where size > 0
        size_f = self.size.to(torch.float32).clamp_min(1.0).unsqueeze(-1)  # avoid /0 on padding rows
        S = S / size_f
        return S
