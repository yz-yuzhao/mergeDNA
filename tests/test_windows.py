"""Tests for the window reshape helpers."""

from __future__ import annotations

import pytest
import torch

from mergedna.modules.windows import pad_to_multiple, to_windows, from_windows


def test_pad_already_divisible_is_noop():
    x = torch.randn(2, 8, 3)
    y, pad = pad_to_multiple(x, multiple=4, dim=1)
    assert pad == 0
    assert torch.equal(x, y)


def test_pad_adds_correct_length():
    x = torch.randn(2, 7, 3)
    y, pad = pad_to_multiple(x, multiple=4, dim=1, value=0.0)
    assert pad == 1
    assert y.shape == (2, 8, 3)
    # Original content preserved.
    assert torch.equal(y[:, :7, :], x)
    # Padded slot is zero.
    assert torch.equal(y[:, 7:, :], torch.zeros(2, 1, 3))


def test_to_from_windows_round_trip():
    x = torch.arange(2 * 12 * 5).reshape(2, 12, 5).float()
    W = 4
    xw = to_windows(x, W)
    assert xw.shape == (2, 3, 4, 5)
    x2 = from_windows(xw)
    assert torch.equal(x2, x)


def test_to_windows_rejects_indivisible_length():
    x = torch.zeros(1, 5, 2)
    with pytest.raises(ValueError):
        to_windows(x, W=4)


def test_pad_to_multiple_other_dim():
    x = torch.randn(3, 5)
    y, pad = pad_to_multiple(x, multiple=4, dim=0, value=-1.0)
    assert pad == 1
    assert y.shape == (4, 5)
    assert (y[3] == -1.0).all()
