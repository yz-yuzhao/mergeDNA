"""Integration tests for the three-pass loss.

The most important check is **gradient gating** for pass 2: when only the
``L_MTR(θ\\{φ})`` term is back-propagated, gradients must not flow into the
local encoder or the input embedding (those parameters belong to ``φ``).
"""

from __future__ import annotations

import math

import torch

from mergedna.config import tiny_config
from mergedna.losses.reconstruction import mtr_loss
from mergedna.model.mergedna import MergeDNA
from mergedna.model.local_encoder import sample_r_schedule
from mergedna.modules.source_map import SourceMap
from mergedna.training.three_pass import three_pass_loss


def _make_model_and_batch():
    cfg, train_cfg = tiny_config()
    model = MergeDNA(cfg)
    B, N = 2, train_cfg.seq_len
    token_ids = torch.randint(0, 4, (B, N))
    return model, cfg, train_cfg, token_ids


# -------------------------------------------------------------------- #
# Total loss is finite                                                  #
# -------------------------------------------------------------------- #


def test_total_loss_is_finite():
    model, cfg, train_cfg, token_ids = _make_model_and_batch()
    loss, metrics = three_pass_loss(model, token_ids, cfg, train_cfg)
    assert torch.isfinite(loss)
    for k, v in metrics.items():
        assert torch.isfinite(v), f"non-finite metric {k} = {v}"


# -------------------------------------------------------------------- #
# Backward populates grads on every module                              #
# -------------------------------------------------------------------- #


def test_total_backward_grads_all_modules():
    model, cfg, train_cfg, token_ids = _make_model_and_batch()
    loss, _ = three_pass_loss(model, token_ids, cfg, train_cfg)
    loss.backward()
    for module_name in ("local_encoder", "latent_encoder", "latent_decoder", "local_decoder"):
        module = getattr(model, module_name)
        any_grad = False
        for p in module.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
                assert torch.isfinite(p.grad).all()
        assert any_grad, f"no grads on {module_name}"


# -------------------------------------------------------------------- #
# Gradient gating: pass 2 alone must not grad E_φ or the input embedding
# -------------------------------------------------------------------- #


def test_pass2_does_not_grad_local_encoder_or_embedding():
    """Run only the pass-2 portion and confirm E_φ + token embedding are clean."""
    model, cfg, _, token_ids = _make_model_and_batch()
    B, N = token_ids.shape
    sched = sample_r_schedule(
        N=N,
        n_layers=cfg.n_local_enc_layers,
        target_compression=cfg.target_local_compression,
        jitter=0.0,
        training=False,
    )

    # Pass 1 (only to produce a detached h_local). We will throw away its loss.
    with torch.no_grad():
        out1 = model(token_ids, mode="full", r_schedule=sched)

    # Re-enter grad tape for pass 2 only.
    h_local = out1.h_local.detach().clone().requires_grad_(False)
    size_local = out1.size_local.detach().clone()
    token_mask_local = out1.token_mask_local.detach().clone()
    sm = SourceMap(
        parent=out1.source_map.parent.detach().clone(),
        size=out1.source_map.size.detach().clone(),
        L=out1.source_map.L.detach().clone(),
        token_mask=out1.source_map.token_mask.detach().clone(),
    )
    L = h_local.shape[1]
    r_global = max(1, L // 2)

    # Make sure no stale grads from anywhere.
    model.zero_grad(set_to_none=True)

    out2 = model.forward_latent_only(
        h_local=h_local,
        source_map=sm,
        size_local=size_local,
        token_mask_local=token_mask_local,
        r_global=r_global,
    )
    loss2 = mtr_loss(out2.logits, token_ids)
    loss2.backward()

    # E_φ and its input embedding must have NO grads.
    for name, p in model.local_encoder.named_parameters():
        assert p.grad is None or p.grad.abs().sum() == 0, (
            f"local_encoder.{name} got a non-zero grad in pass 2: "
            f"{p.grad.abs().sum().item():.3e}"
        )

    # And the latent stack + local decoder *must* have grads.
    other_modules = (model.latent_encoder, model.latent_decoder, model.local_decoder)
    for module in other_modules:
        any_grad = False
        for p in module.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                any_grad = True
        assert any_grad, f"no pass-2 grad on {module.__class__.__name__}"


# -------------------------------------------------------------------- #
# Tiny optimization: loss decreases over a few steps on a fixed batch    #
# -------------------------------------------------------------------- #


def test_loss_decreases_over_a_few_steps():
    """Sanity: training on a *fixed* small batch should reduce the total loss
    over a handful of steps. Not strictly required — could be flaky in
    pathological conditions — but useful as an integration smoke test.
    """
    model, cfg, train_cfg, token_ids = _make_model_and_batch()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)

    losses: list[float] = []
    for step in range(8):
        opt.zero_grad(set_to_none=True)
        loss, _ = three_pass_loss(model, token_ids, cfg, train_cfg)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    # Loss at the end should be lower than loss at step 0 (within reason).
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert math.isfinite(losses[-1])
