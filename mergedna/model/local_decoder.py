"""Local decoder 

Two steps:

  1. Unmerge by gathering each nucleotide's parent token's.
  2. ``cfg.n_local_dec_layers`` of windowed self-attention (RoPE on Q/K), then
     a linear projection to nucleotide-vocab logits.

"""

from __future__ import annotations

import torch
import torch.nn as nn

from mergedna.config import MergeDNAConfig
from mergedna.modules.source_map import SourceMap
from mergedna.modules.transformer import TransformerBlock


class LocalDecoder(nn.Module):
    def __init__(self, cfg: MergeDNAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    local_window=cfg.window_size,
                )
                for _ in range(cfg.n_local_dec_layers)
            ]
        )
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(
        self,
        z_local: torch.Tensor,
        source_map: SourceMap,
    ) -> torch.Tensor:
        """``[B, L_max, D] -> [B, N, vocab_size]``."""
        x = source_map.gather_to_nucleotides(z_local)
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device, dtype=torch.float32).unsqueeze(0).expand(B, N).contiguous()

        for layer in self.layers:
            x = layer(x, attn_mask=None, size=None, positions=positions)

        return self.head(x)
