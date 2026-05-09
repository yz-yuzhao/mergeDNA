"""Three-pass loss composition.

One optimizer step computes three forward passes through the model and sums
their losses for a single backward:

    L_total = L_MTR(theta)  +  λ · L_MTR(theta\\{phi})  +  L_AMTM(theta)

Why three passes can't collapse to one:

  - Pass 1 uses the original ``X`` and runs the latent encoder without
    a global merge. Loss ``L_MTR``.
  - Pass 2 also uses the original ``X``, but freezes the local encoder
    and enables the latent encoder's global ToMe merge. Loss ``λ · L_MTR(theta\\{phi})``.
  - Pass 3 uses ``X`` with adaptive masking applied. Loss ``L_AMTM``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mergedna.config import MergeDNAConfig, TrainConfig
from mergedna.losses.adaptive_mask import (
    propagate_mask_to_nucleotides,
    sample_adaptive_mask,
)
from mergedna.losses.reconstruction import masked_mtr_loss, mtr_loss
from mergedna.model.local_encoder import sample_r_schedule
from mergedna.model.mergedna import MergeDNA
from mergedna.modules.source_map import SourceMap


@dataclass
class ThreePassMetrics:
    total: torch.Tensor
    loss_mtr: torch.Tensor
    loss_mtr_no_phi: torch.Tensor
    loss_amtm: torch.Tensor


def three_pass_loss(
    model: MergeDNA,
    token_ids: torch.Tensor,
    cfg: MergeDNAConfig,
    train_cfg: TrainConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the full three-term training objective.

    Args:
        model:      a ``MergeDNA`` instance.
        token_ids:  ``[B, N]`` LongTensor of nucleotide ids.
        cfg:        model configuration.
        train_cfg:  training configuration (for ``lambda_latent``, ``n_mask``).
        generator:  optional torch RNG for reproducible sampling.

    Returns:
        ``(total_loss, metrics)`` where ``metrics`` is a dict with the four
        scalar losses (total, mtr, mtr_no_phi, amtm) as detached tensors.
    """
    if token_ids.dim() != 2:
        raise ValueError(f"token_ids must be 2-D, got shape {tuple(token_ids.shape)}")
    B, N = token_ids.shape

    # Sample a single r-schedule for this whole step.
    sched = sample_r_schedule(
        N=N,
        n_layers=cfg.n_local_enc_layers,
        target_compression=cfg.target_local_compression,
        jitter=cfg.local_compression_jitter,
        training=model.training,
        generator=generator,
    )

    # ----- Pass 1: full pipeline, no global merge ------------- #
    out1 = model(token_ids, mode="full", r_schedule=sched)
    loss_mtr = mtr_loss(out1.logits, token_ids)

    # ----- Pass 2: latent-only, with global merge ------------- #
    L_local = out1.h_local.shape[1]
    r_global = max(1, int(round(L_local * (1.0 - cfg.target_global_compression))))

    h_local_detached = out1.h_local.detach()    # block gradient back progated to local decoder
    size_local_detached = out1.size_local.detach()
    sm_detached = SourceMap(
        parent=out1.source_map.parent.detach(),
        size=out1.source_map.size.detach(),
        L=out1.source_map.L.detach(),
        token_mask=out1.source_map.token_mask.detach(),
    )

    out2 = model.forward_latent_only(
        h_local=h_local_detached,
        source_map=sm_detached,
        size_local=size_local_detached,
        token_mask_local=out1.token_mask_local.detach(),
        r_global=r_global,
    )
    loss_mtr_no_phi = mtr_loss(out2.logits, token_ids)

    # ----- Pass 3: adaptive-masked input ----------------------------------- #
    if out2.global_plan is None:
        # No global merge enabled (e.g. r_global was clamped to 0). Skip pass 3.
        loss_amtm = token_ids.new_zeros(()).to(loss_mtr.dtype)
    else:
        K_mask = train_cfg.n_mask if train_cfg.n_mask is not None else int(out2.size_latent.shape[1])
        K_mask = min(K_mask, L_local)

        mask_local = sample_adaptive_mask(
            size_latent=out2.size_latent.detach(),
            parent_global=out2.global_plan.old_to_new.detach(),
            K_mask=K_mask,
            token_mask_local=out1.token_mask_local.detach(),
            generator=generator,
        )
        mask_nuc = propagate_mask_to_nucleotides(mask_local, out1.source_map)

        out3 = model(token_ids, mode="masked", r_schedule=sched, mask_positions=mask_nuc)
        loss_amtm = masked_mtr_loss(out3.logits, token_ids, mask_nuc)

    total = loss_mtr + train_cfg.lambda_latent * loss_mtr_no_phi + loss_amtm
    metrics = {
        "total": total.detach(),
        "loss_mtr": loss_mtr.detach(),
        "loss_mtr_no_phi": loss_mtr_no_phi.detach(),
        "loss_amtm": loss_amtm.detach(),
    }
    return total, metrics
