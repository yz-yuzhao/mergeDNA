"""Shared pytest fixtures.

The test suite uses tiny configurations so that an end-to-end forward + backward
runs comfortably under a second on CPU. All randomness is seeded.
"""

from __future__ import annotations

import pytest
import torch

from mergedna.config import MergeDNAConfig, TrainConfig, tiny_config


@pytest.fixture
def configs() -> tuple[MergeDNAConfig, TrainConfig]:
    return tiny_config()


@pytest.fixture
def model_cfg() -> MergeDNAConfig:
    cfg, _ = tiny_config()
    return cfg


@pytest.fixture
def train_cfg() -> TrainConfig:
    _, tcfg = tiny_config()
    return tcfg


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Seed torch globally for every test so failures are reproducible."""
    torch.manual_seed(0)


@pytest.fixture
def gen() -> torch.Generator:
    return torch.Generator().manual_seed(123)
