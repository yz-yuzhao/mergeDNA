"""Tests for proportional attention.

The key invariant: a merged token of size ``g`` should produce identical
attention outputs to ``g`` separate identical tokens. The ``+log(size)`` bias
on the key axis of attention logits is precisely what makes this true.
"""

from __future__ import annotations

import torch

from mergedna.modules.attention import MultiHeadAttention
from mergedna.modules.transformer import TransformerBlock


def test_size_one_is_a_no_op():
    """Passing ``size = ones`` must give the same output as passing nothing."""
    torch.manual_seed(0)
    B, L, D = 2, 8, 16
    attn = MultiHeadAttention(d_model=D, n_heads=4)
    x = torch.randn(B, L, D)

    y_no_size = attn(x)
    y_ones = attn(x, size=torch.ones(B, L))
    assert torch.allclose(y_no_size, y_ones, atol=1e-6)


def test_size_two_token_equivalent_to_duplication():
    """A 1-token sequence with size=2 attended-to by a query produces the same
    outputs (at the query) as a 2-token sequence of two identical tokens.

    Concretely: build sequences ``X = [q, t]`` (size [1, 2]) vs ``X' = [q, t, t]``
    (size [1, 1, 1]). The query at position 0 should produce the same output.
    """
    torch.manual_seed(1)
    B, D = 1, 16
    attn = MultiHeadAttention(d_model=D, n_heads=4)
    q = torch.randn(B, 1, D)
    t = torch.randn(B, 1, D)

    # Length-2 with sizes [1, 2] (proportional attention case).
    x_short = torch.cat([q, t], dim=1)        # [1, 2, D]
    size_short = torch.tensor([[1.0, 2.0]])
    y_short = attn(x_short, size=size_short)

    # Length-3 with two identical t-tokens; standard attention.
    x_long = torch.cat([q, t, t], dim=1)      # [1, 3, D]
    y_long = attn(x_long)

    # The query (position 0) sees the same effective "weight 2" on t in both.
    assert torch.allclose(y_short[:, 0], y_long[:, 0], atol=1e-5), (
        f"max diff = {(y_short[:, 0] - y_long[:, 0]).abs().max():.3e}"
    )


def test_log_size_bias_is_added_to_logits_pre_softmax():
    """Direct mathematical check: rerun the qkv path manually and confirm that
    softmax((qk^T)/sqrt(d) + log(size)) matches the module's output.
    """
    import math
    torch.manual_seed(2)
    B, L, D, H = 1, 4, 16, 4
    attn = MultiHeadAttention(d_model=D, n_heads=H)
    x = torch.randn(B, L, D)
    size = torch.tensor([[1.0, 2.0, 3.0, 1.0]])

    y = attn(x, size=size)

    # Manual recomputation (no positions → no RoPE rotation).
    qkv = attn.qkv(x).view(B, L, 3, H, D // H).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    scale = 1.0 / math.sqrt(D // H)
    logits = (q @ k.transpose(-2, -1)) * scale
    logits = logits + torch.log(size).clamp_min(0.0)[:, None, None, :]
    a = torch.softmax(logits, dim=-1)
    o = a @ v
    o = o.transpose(1, 2).reshape(B, L, D)
    o = attn.out(o)

    assert torch.allclose(y, o, atol=1e-5)


def test_local_encoder_block_does_not_pass_size():
    """A windowed transformer block must not apply the proportional bias even
    if the caller hands it a ``size`` tensor — local encoder/decoder are
    explicitly opted out (size variance within a window of 16 is small).
    """
    torch.manual_seed(3)
    B, L, D = 1, 4, 16
    block = TransformerBlock(d_model=D, n_heads=4, mlp_ratio=2.0, local_window=4)
    x = torch.randn(B, L, D)
    size_dramatic = torch.tensor([[1.0, 100.0, 1.0, 1.0]])

    y_with_size = block(x, size=size_dramatic)
    y_no_size = block(x)
    # Must be identical: windowed mode ignores size.
    assert torch.allclose(y_with_size, y_no_size, atol=1e-6)


def test_global_encoder_block_does_use_size():
    """A non-windowed block *does* honor size."""
    torch.manual_seed(4)
    B, L, D = 1, 4, 16
    block = TransformerBlock(d_model=D, n_heads=4, mlp_ratio=2.0, local_window=None)
    x = torch.randn(B, L, D)
    size_dramatic = torch.tensor([[1.0, 100.0, 1.0, 1.0]])

    y_with_size = block(x, size=size_dramatic)
    y_no_size = block(x)
    # Must differ.
    assert not torch.allclose(y_with_size, y_no_size, atol=1e-3)
