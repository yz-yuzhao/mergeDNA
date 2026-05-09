"""Tiny test training script.

Trains MergeDNA at a small CPU-friendly config on uniform random DNA. The goal
is a quick end-to-end sanity check, not real performance. The whole run takes
under a minute on a laptop.

Run:
    python -m mergedna.training.train_synthetic
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from mergedna.config import MergeDNAConfig, TrainConfig
from mergedna.data.synthetic import SyntheticDNA
from mergedna.model.mergedna import MergeDNA
from mergedna.training.three_pass import three_pass_loss


def main() -> None:
    cfg = MergeDNAConfig(
        d_model=64,
        n_local_enc_layers=2,
        n_latent_enc_layers=2,
        n_latent_dec_layers=1,
        n_local_dec_layers=1,
        n_heads=4,
        mlp_ratio=2.0,
        window_size=8,
        max_seq_len=128,
        target_local_compression=0.5,
        local_compression_jitter=0.05,
        target_global_compression=0.5,
        global_merge_layer=0,
        grouping_dim=16,
    )
    train_cfg = TrainConfig(
        batch_size=4,
        seq_len=64,
        lr=3e-3,
        steps=50,
        lambda_latent=0.25,
        seed=0,
    )

    torch.manual_seed(train_cfg.seed)
    gen = torch.Generator().manual_seed(train_cfg.seed)

    ds = SyntheticDNA(n_examples=128, seq_len=train_cfg.seq_len, seed=train_cfg.seed)
    dl = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True)

    model = MergeDNA(cfg)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e3:.1f}K params, seq_len={train_cfg.seq_len}, batch={train_cfg.batch_size}")

    t0 = time.time()
    step = 0
    for batch in iter_cycle(dl, train_cfg.steps):
        opt.zero_grad(set_to_none=True)
        loss, metrics = three_pass_loss(model, batch, cfg, train_cfg, generator=gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        if step % 10 == 0 or step == train_cfg.steps - 1:
            print(
                f"step {step:>4d} | total {float(metrics['total']):.4f} "
                f"| mtr {float(metrics['loss_mtr']):.4f} "
                f"| mtr_no_phi {float(metrics['loss_mtr_no_phi']):.4f} "
                f"| amtm {float(metrics['loss_amtm']):.4f}"
            )
        step += 1
        if step >= train_cfg.steps:
            break
    print(f"done in {time.time() - t0:.1f}s")


def iter_cycle(dl: DataLoader, n_steps: int):
    """Yield ``n_steps`` batches by cycling through ``dl`` as needed."""
    yielded = 0
    while yielded < n_steps:
        for batch in dl:
            yield batch
            yielded += 1
            if yielded >= n_steps:
                return


if __name__ == "__main__":
    main()
