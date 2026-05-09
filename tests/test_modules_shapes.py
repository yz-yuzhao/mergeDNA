"""Shape and basic forward tests for the four hourglass modules and MergeDNA."""

from __future__ import annotations

import torch

from mergedna.config import tiny_config
from mergedna.data.vocab import VOCAB_SIZE
from mergedna.model.local_encoder import LocalEncoder, sample_r_schedule
from mergedna.model.latent_encoder import LatentEncoder
from mergedna.model.latent_decoder import LatentDecoder
from mergedna.model.local_decoder import LocalDecoder
from mergedna.model.mergedna import MergeDNA


# -------------------------------------------------------------------- #
# r-schedule                                                           #
# -------------------------------------------------------------------- #


def test_sample_r_schedule_sums_correctly():
    sched = sample_r_schedule(N=32, n_layers=4, target_compression=0.5, jitter=0.0, training=False)
    assert len(sched) == 4
    assert sum(sched) == 16  # remove half


def test_sample_r_schedule_clamps_to_valid_range():
    """1000 stochastic samples must all stay within [0.4N, 0.6N] removal range."""
    N = 256
    for _ in range(1000):
        sched = sample_r_schedule(N=N, n_layers=4, target_compression=0.5, jitter=0.05, training=True)
        L = N - sum(sched)
        assert 0.4 * N <= L <= 0.6 * N + 1e-6, f"L = {L} outside [0.4N, 0.6N]"


# -------------------------------------------------------------------- #
# LocalEncoder                                                          #
# -------------------------------------------------------------------- #


def test_local_encoder_shapes():
    cfg, _ = tiny_config()
    enc = LocalEncoder(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)
    h, sm, size, mask = enc(token_ids, sched)

    L = N - sum(sched)
    assert h.shape[0] == B
    assert h.shape[2] == cfg.d_model
    L_max = h.shape[1]
    assert sm.parent.shape == (B, N)
    assert size.shape == (B, L_max)
    assert mask.shape == (B, L_max)
    # source_map.L equals h's L_max (since no per-batch variation in our scheme)
    assert int(sm.L[0]) == L
    # invariants
    assert (sm.size[0, : int(sm.L[0])].sum()) == N
    assert (sm.size[0, : int(sm.L[0])] > 0).all()


# -------------------------------------------------------------------- #
# LatentEncoder                                                         #
# -------------------------------------------------------------------- #


def test_latent_encoder_no_global_merge():
    cfg, _ = tiny_config()
    enc = LatentEncoder(cfg)
    B, L, D = 2, 16, cfg.d_model
    h = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long) * 2  # pretend each token absorbed 2 nucleotides
    mask = torch.ones(B, L, dtype=torch.bool)
    positions = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, L).contiguous()

    z, size_out, mask_out, pos_out, plan = enc(h, size, mask, positions, do_global_merge=False)
    assert z.shape == (B, L, D)
    assert size_out.shape == (B, L)
    assert torch.equal(size_out, size)
    assert mask_out.shape == (B, L)
    assert pos_out.shape == (B, L)
    assert plan is None


def test_latent_encoder_with_global_merge():
    cfg, _ = tiny_config()
    enc = LatentEncoder(cfg)
    B, L, D = 2, 16, cfg.d_model
    h = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long) * 2
    mask = torch.ones(B, L, dtype=torch.bool)
    positions = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, L).contiguous()

    z, size_out, mask_out, pos_out, plan = enc(h, size, mask, positions, do_global_merge=True, r_global=4)
    K = L - 4
    assert z.shape[0] == B and z.shape[2] == D
    assert z.shape[1] == K  # one-r-per-step => same K for all batch elements
    assert plan is not None
    assert int(plan.L_new_max) == K
    assert pos_out.shape == (B, K)
    # size_latent should sum to L * 2 = 32 per batch element (total mass conserved)
    for b in range(B):
        assert int(size_out[b, : int(plan.L_new[b])].sum()) == L * 2


# -------------------------------------------------------------------- #
# LatentDecoder                                                         #
# -------------------------------------------------------------------- #


def test_latent_decoder_preserves_shape():
    cfg, _ = tiny_config()
    dec = LatentDecoder(cfg)
    B, L, D = 2, 12, cfg.d_model
    z = torch.randn(B, L, D)
    size = torch.ones(B, L, dtype=torch.long)
    mask = torch.ones(B, L, dtype=torch.bool)
    positions = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(B, L).contiguous()
    out = dec(z, size, mask, positions)
    assert out.shape == (B, L, D)


# -------------------------------------------------------------------- #
# LocalDecoder                                                          #
# -------------------------------------------------------------------- #


def test_local_decoder_unmerges_to_nucleotides():
    cfg, _ = tiny_config()
    enc = LocalEncoder(cfg)
    dec = LocalDecoder(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)
    h, sm, _, _ = enc(token_ids, sched)
    logits = dec(h, sm)
    assert logits.shape == (B, N, cfg.vocab_size)
    assert cfg.vocab_size == VOCAB_SIZE


# -------------------------------------------------------------------- #
# MergeDNA top-level                                                    #
# -------------------------------------------------------------------- #


def test_mergedna_full_mode():
    cfg, _ = tiny_config()
    model = MergeDNA(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)
    out = model(token_ids, mode="full", r_schedule=sched)
    assert out.logits.shape == (B, N, cfg.vocab_size)
    assert out.global_plan is None


def test_mergedna_masked_mode_substitutes_mask_id():
    cfg, _ = tiny_config()
    model = MergeDNA(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)
    mask_pos = torch.zeros(B, N, dtype=torch.bool)
    mask_pos[:, ::4] = True
    out = model(token_ids, mode="masked", r_schedule=sched, mask_positions=mask_pos)
    assert out.logits.shape == (B, N, cfg.vocab_size)


def test_mergedna_forward_latent_only_changes_K():
    cfg, _ = tiny_config()
    model = MergeDNA(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)

    out1 = model(token_ids, mode="full", r_schedule=sched)
    L = out1.h_local.shape[1]
    r_global = max(1, L // 2)

    out2 = model.forward_latent_only(
        h_local=out1.h_local.detach(),
        source_map=out1.source_map,
        size_local=out1.size_local.detach(),
        token_mask_local=out1.token_mask_local,
        r_global=r_global,
    )
    K = out2.z_latent.shape[1]
    assert K == L - r_global
    assert out2.logits.shape == (B, N, cfg.vocab_size)
    assert out2.global_plan is not None


# -------------------------------------------------------------------- #
# End-to-end forward + backward                                         #
# -------------------------------------------------------------------- #


def test_full_pipeline_backward_produces_finite_grads_on_all_modules():
    cfg, _ = tiny_config()
    model = MergeDNA(cfg)
    B, N = 2, 32
    token_ids = torch.randint(0, 4, (B, N))
    sched = sample_r_schedule(N, cfg.n_local_enc_layers, cfg.target_local_compression, 0.0, False)

    out = model(token_ids, mode="full", r_schedule=sched)
    loss = out.logits.float().mean()
    loss.backward()

    for module_name in ("local_encoder", "latent_encoder", "latent_decoder", "local_decoder"):
        module = getattr(model, module_name)
        any_grad_seen = False
        for p in module.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {module_name}"
                any_grad_seen = True
        assert any_grad_seen, f"no grads on {module_name}"
