"""Top-level MergeDNA model composition.

Two entry points:

  - ``forward(token_ids, mode, ...)`` — runs the full pipeline. Modes:
      * ``"full"``   — full pipline without global merge.
      * ``"masked"`` — substitute ``[MASK]`` ids at ``mask_positions`` first.

  - ``forward_latent_only(h_local, source_map, size_local, token_mask_local, r_global)``
    — bypasses the local encoder. Used by **pass 2** of the loss
    (``L_MTR(theta\\{phi})``). Inside latent encoder, the global merge is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

from mergedna.config import MergeDNAConfig
from mergedna.data.vocab import MASK
from mergedna.model.latent_decoder import LatentDecoder
from mergedna.model.latent_encoder import LatentEncoder
from mergedna.model.local_decoder import LocalDecoder
from mergedna.model.local_encoder import LocalEncoder
from mergedna.modules.merge import MergePlan
from mergedna.modules.source_map import SourceMap


@dataclass
class ForwardOut:
    """Return value of ``MergeDNA.forward`` and ``forward_latent_only``."""

    logits: torch.Tensor                # [B, N, vocab_size]
    source_map: SourceMap
    h_local: torch.Tensor               # [B, L_max, D]
    z_latent: torch.Tensor              # [B, L_or_K, D]
    size_local: torch.Tensor            # [B, L_max]
    size_latent: torch.Tensor           # [B, K_max] when global merge fired, else size_local
    token_mask_local: torch.Tensor      # [B, L_max]
    token_mask_latent: torch.Tensor     # [B, K_max] when global merge fired, else token_mask_local
    global_plan: MergePlan | None


class MergeDNA(nn.Module):
    """The full MergeDNA model."""

    def __init__(self, cfg: MergeDNAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.local_encoder = LocalEncoder(cfg)
        self.latent_encoder = LatentEncoder(cfg)
        self.latent_decoder = LatentDecoder(cfg)
        self.local_decoder = LocalDecoder(cfg)

    def forward(
        self,
        token_ids: torch.Tensor,
        mode: Literal["full", "masked"],
        r_schedule: list[int],
        mask_positions: torch.Tensor | None = None,
    ) -> ForwardOut:
        """Run the full pipeline (no global merge).

        Args:
            token_ids:      ``[B, N]`` LongTensor of nucleotide ids.
            mode:           ``"full"`` or ``"masked"``.
            r_schedule:     per-layer merge counts for the local encoder.
            mask_positions: ``[B, N]`` BoolTensor; required when ``mode="masked"``.
        """
        if mode == "masked":
            if mask_positions is None:
                raise ValueError("mode='masked' requires mask_positions")
            token_ids = token_ids.masked_fill(mask_positions, MASK)
        elif mode != "full":
            raise ValueError(f"unknown mode {mode!r} (use 'full' or 'masked')")

        h_local, source_map, size_local, token_mask_local = self.local_encoder(
            token_ids, r_schedule
        )
        positions_local = source_map.token_positions()
        z_latent, size_latent, token_mask_latent, positions_latent, global_plan = self.latent_encoder(
            h_local, size_local, token_mask_local, positions_local,
            do_global_merge=False, r_global=None,
        )
        z_dec = self.latent_decoder(z_latent, size_latent, token_mask_latent, positions_latent)
        logits = self.local_decoder(z_dec, source_map)

        return ForwardOut(
            logits=logits,
            source_map=source_map,
            h_local=h_local,
            z_latent=z_latent,
            size_local=size_local,
            size_latent=size_latent,
            token_mask_local=token_mask_local,
            token_mask_latent=token_mask_latent,
            global_plan=global_plan,
        )

    def forward_latent_only(
        self,
        h_local: torch.Tensor,
        source_map: SourceMap,
        size_local: torch.Tensor,
        token_mask_local: torch.Tensor,
        r_global: int,
    ) -> ForwardOut:
        """Pass 2 entry point: skip local encoder, enable global merge.
        """
        positions_local = source_map.token_positions()
        z_latent, size_latent, token_mask_latent, positions_latent, global_plan = self.latent_encoder(
            h_local, size_local, token_mask_local, positions_local,
            do_global_merge=True, r_global=r_global,
        )
        z_dec = self.latent_decoder(z_latent, size_latent, token_mask_latent, positions_latent)

        # Un-global-merge K → L: each L token's feature = its K-group feature.
        if global_plan is not None:
            z_at_L = self._un_global_merge(z_dec, global_plan)
        else:
            z_at_L = z_dec

        logits = self.local_decoder(z_at_L, source_map)

        return ForwardOut(
            logits=logits,
            source_map=source_map,
            h_local=h_local,
            z_latent=z_latent,
            size_local=size_local,
            size_latent=size_latent,
            token_mask_local=token_mask_local,
            token_mask_latent=token_mask_latent,
            global_plan=global_plan,
        )

    @staticmethod
    def _un_global_merge(z_K: torch.Tensor, plan: MergePlan) -> torch.Tensor:
        """``[B, K_max, D]`` -> ``[B, L, D]`` via gather on ``plan.old_to_new``.

        ``plan.old_to_new[b, ℓ]`` is the K-token index that local-token ``ℓ``
        was merged into; gathering the K-feature at that index broadcasts each
        K-feature to all of its constituent locals.
        """
        B, _, D = z_K.shape
        idx = plan.old_to_new.unsqueeze(-1).expand(-1, -1, D)
        return z_K.gather(dim=1, index=idx)
