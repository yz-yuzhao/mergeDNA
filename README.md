# MergeDNA

A PyTorch reimplementation of MergeDNA ([arxiv 2511.14806](https://arxiv.org/abs/2511.14806)),
a hierarchical genome sequence model that learns its own variable-length DNA tokenization
through a Token-Merging-style differentiable merger inspired by
[ToMe (Bolya et al. 2023)](https://arxiv.org/abs/2210.09461).

The codes were generated in collaboration with Claude Code.

## Architecture overview

MergeDNA is a 4-stage model architecture: Local Encoder, Latent Encoder, Latent Decoder
and Local Decoder.

The pretraining objective has three terms (one optimizer step = three forward passes,
single summed backward):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MTR}}(\theta) + \lambda \cdot \mathcal{L}_{\text{MTR}}(\theta \setminus \lbrace \phi \rbrace) + \mathcal{L}_{\text{AMTM}}(\theta)
$$

- $\mathcal{L}_{\text{MTR}}$ — full-pipeline reconstruction (cross-entropy on nucleotides).
- $\mathcal{L}_{\text{MTR}}(\theta \setminus \lbrace \phi \rbrace)$ — reconstruction with the local encoder frozen and a global ToMe merge
  fired inside the latent encoder.
- $\mathcal{L}_{\text{AMTM}}$ — masked-language-modeling, where the mask is sampled inversely-proportional
  to merge group size.

## Codebase layout

```
mergedna/
    config.py               # MergeDNAConfig, TrainConfig, tiny_config()
    data/                   # vocabulary + synthetic dataset
    modules/                # source_map, windows, grouping_head, merge, attention, transformer
    model/                  # local_encoder, latent_encoder, latent_decoder, local_decoder, mergedna
    losses/                 # reconstruction, adaptive_mask
    training/               # three_pass, train_synthetic
tests/                      # pytest suite, all CPU-friendly
```

## Install / test

```
uv sync --extra test
uv run pytest tests/ -v
```

## Test run

```
uv run python -m mergedna.training.train_synthetic
```

A 50-step training run on uniform random DNA at tiny dims, finishes in well under a minute on CPU.
