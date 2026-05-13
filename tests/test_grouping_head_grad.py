"""STE bridge: gradients reach the GroupingHead; forward is unchanged."""

from __future__ import annotations

import torch

from mergedna.config import tiny_config
from mergedna.model.local_encoder import LocalEncoder, sample_r_schedule
from mergedna.modules.merge import (
    all_pairs_match_window,
    apply_merge_plan,
    soft_apply_merge_window,
)


def test_grouping_head_receives_gradient():
    cfg, _ = tiny_config()
    enc = LocalEncoder(cfg)
    enc.train()

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    r_sched = sample_r_schedule(
        32, cfg.n_local_enc_layers, target_compression=0.5, training=False
    )

    h, _, _, _ = enc(ids, r_sched)
    h.sum().backward()

    for name, p in enc.grouping_head.named_parameters():
        assert p.grad is not None, f"grouping_head.{name}.grad is None"
        assert p.grad.abs().sum().item() > 0, (
            f"grouping_head.{name}.grad is identically zero"
        )


def test_ste_forward_matches_hard_forward():
    """``x_soft - x_soft.detach()`` is identically 0, so training-mode forward
    must equal eval-mode forward (which skips the soft branch)."""
    cfg, _ = tiny_config()
    enc = LocalEncoder(cfg)

    torch.manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    r_sched = sample_r_schedule(
        32, cfg.n_local_enc_layers, target_compression=0.5, training=False
    )

    enc.eval()
    with torch.no_grad():
        h_eval, _, _, _ = enc(ids, r_sched)

    enc.train()
    h_train, _, _, _ = enc(ids, r_sched)

    torch.testing.assert_close(h_train, h_eval, rtol=0.0, atol=0.0)


def test_soft_equals_hard_in_discrete_limit():
    """When the best pair in each window is strongly separated from the
    runners-up, the iterated soft-argmax is essentially one-hot and the
    soft merge collapses to the hard merge.
    """
    torch.manual_seed(2)
    B, L, W, D, Dm = 1, 8, 4, 4, 4
    x = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    valid = torch.ones(B, L, dtype=torch.bool)
    # Engineer metrics so the best pair in each window is unambiguous.
    # Window 0: tokens 0 and 1 identical along axis 0; tokens 2, 3 orthogonal.
    # Window 1: tokens 4 and 5 identical along axis 1; tokens 6, 7 orthogonal.
    metric = 0.01 * torch.randn(B, L, Dm)
    metric[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    metric[0, 3] = torch.tensor([0.0, 0.0, 1.0, 0.0])
    metric[0, 4] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    metric[0, 5] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    metric[0, 6] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    metric[0, 7] = torch.tensor([0.0, 0.0, 1.0, 0.0])

    plan = all_pairs_match_window(metric, size, r=2, W=W, valid=valid)
    x_hard, _ = apply_merge_plan(x, size, plan)
    x_soft = soft_apply_merge_window(
        x=x, size=size, metric=metric,
        r=2, W=W, valid=valid, plan=plan, tau=0.01,
    )
    torch.testing.assert_close(x_soft, x_hard, rtol=1e-3, atol=1e-3)


def test_metric_receives_gradient_directly():
    """Direct check on the soft-merge function: ``metric`` is in the graph."""
    torch.manual_seed(3)
    B, L, W, D, Dm = 1, 8, 4, 4, 4
    x = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    metric = torch.randn(B, L, Dm, requires_grad=True)
    valid = torch.ones(B, L, dtype=torch.bool)

    plan = all_pairs_match_window(metric.detach(), size, r=2, W=W, valid=valid)
    x_soft = soft_apply_merge_window(
        x=x, size=size, metric=metric,
        r=2, W=W, valid=valid, plan=plan,
    )
    x_soft.sum().backward()

    assert metric.grad is not None
    assert metric.grad.abs().sum().item() > 0
