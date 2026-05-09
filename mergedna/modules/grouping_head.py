"""Lightweight DTEM-style grouping embedding.

The local encoder uses a separate small MLP to
produce the similarity metric for token merging.

A single GroupingHead is shared across all local-encoder layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GroupingHead(nn.Module):
    """Two-layer MLP producing a low-dimensional similarity metric.

    Args:
        d_model:    input feature dimension.
        d_metric:   output (metric) dimension. Defaults to ``d_model // 4``.
    """

    def __init__(self, d_model: int, d_metric: int | None = None) -> None:
        super().__init__()
        d_metric = d_metric if d_metric is not None else max(1, d_model // 4)
        hidden = max(d_metric, d_model // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_metric),
        )
        self.d_metric = d_metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, d_model] -> [B, L, d_metric]``."""
        return self.net(x)
