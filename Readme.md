# ZetaGPT: A Positional–Encoding–Free State–Space–Attention Compact Language Models

**ZetaGPT** is a compact, positional-encoding-free (NoPE) language model architecture that
uses a state-space-attention mixture block to handle long-context sequences implicitly. It
provides an end-to-end design blueprint covering tokenizer training, pre-training,
supervised instruction tuning (SFT), and reward modelling/RLHF.

[ZetaGPT: A Positional--Encoding--Free State--Space--Attention Compact Language
Model](https://arxiv.org/abs/2608.09432)

## Core design philosophy

- **No positional encoding (NoPE).** Traditional transformers require explicit position
  signals — sinusoidal or rotary embeddings — because self-attention is order-invariant.
  ZetaGPT avoids explicit position vectors entirely: no position table, and no rotation.
- **State-space-attention mixture.** It relies on hybrid mixture blocks that implicitly
  capture token order and long-range dependencies, reflecting modern architectural trends
  aimed at scaling context lengths efficiently.
- **Compact and product-level pipeline.** Built as an open blueprint to demonstrate full
  lifecycle engineering for small-scale, highly efficient language models.

Each block is a causal state-space module, gated multi-head self-attention and a
feed-forward network, pre-normalised with a residual around each sub-layer; sequence order
enters through the recurrent state rather than through a function of the position index.
The repository provides the complete training pipeline — byte-level BPE tokenizer,
pretraining, supervised fine-tuning, reward modelling, RLHF, chain-of-thought reinforcement
learning, direct preference optimisation and distillation — in plain PyTorch, with each
stage runnable on its own.

## Configuration schemes

The head dimension is 64 in every scheme, so width scales by adding heads. Parameter counts
assume the induced vocabulary of 50,259 with the output head tied to the token embedding.

| Scheme | Layers | Heads | `d_model` | `d_h` | MLP | Context | Embedding | Blocks | Parameters |
|---|---|---|---|---|---|---|---|---|---|
| `zetagpt-tiny` | 8 | 8 | 512 | 64 | 4× | 1024 | 25.7M | 35.8M | 61.5M |
| `zetagpt-s` (default) | 24 | 8 | 512 | 64 | 4× | 4096 | 25.7M | 107.3M | **133.0M** |
| `zetagpt-m` | 32 | 8 | 512 | 64 | 4× | 8192 | 25.7M | 143.0M | 168.8M |
| `zetagpt-l` | 32 | 16 | 1024 | 64 | 4× | 32768 | 51.5M | 571.2M | 622.7M |

### Context window scheduling

Pretraining does not sit at one sequence length. Each scheme lists the windows it trains
through — `zetagpt-m` is `[1024, 4096, 8192]` — and each window takes an equal share of the
step budget, shortest first. The batch is sized at the longest window and scales up as the
window shortens, so tokens per step stay constant across the whole run.

### Customized configuration scheme

A scheme is four numbers in `default_config.SCHEMES`, so a size of your own is one entry:

```python
SCHEMES["zetagpt-xl"] = dict(n_layer=32, n_head=20, n_embd=1280, context_window=2048)
```

Then `MODEL_SCHEME=zetagpt-xl` in `config.sh`, with `PRETRAIN_DIR` pointing at a corpus sized
for it. Without editing Python, `CONTEXT_WINDOW` overrides the context alone, and `--set
MODEL.n_layer=20` reaches any other architectural value for one run.

## Niche among compact language models

| Model | Params | Architecture | Positional encoding | Pretrain context |
|---|---|---|---|---|
| TinyStories-1M | 3.7M | Transformer | Learned | 512 |
| Baby GPT (character) | 10.8M | Transformer | Learned | 256 |
| TinyStories-8M | 19.7M | Transformer | Learned | 512 |
| TinyStories-33M | 68.5M | Transformer | Learned | 512 |
| Pythia-70M | 70.4M | Transformer | RoPE | 2048 |
| GPT-2 Small | 124M | Transformer | Learned | 1024 |
| SmolLM2-135M | 134.5M | Transformer | RoPE | 8192 |
| Gemma 3 270M | 268.1M | Transformer | RoPE | 32768 |
| nanochat d20 | ~560M | Transformer | RoPE | 1024 |
| Qwen3-0.6B | ~0.6B | Transformer | RoPE | 32768 |
| TinyLlama | ~1.1B | Transformer | RoPE | 2048 |
| **ZetaGPT-Tiny** | **61.5M** | **State–Space–Attention** | **None** | **512** |
| **ZetaGPT-S** (default) | **97.2M** | **State–Space–Attention** | **None** | **512** |
| **ZetaGPT-M** | **199.3M** | **State–Space–Attention** | **None** | **1024** |
| **ZetaGPT-L** | **479.9M** | **State–Space–Attention** | **None** | **1024** |

## Quick start

```bash
pip install -r requirements.txt

./stage1_download_data.sh          # fetch the corpora into data/download/
```

```bash
./stage1_download_data.sh          # fetch the corpora
./stage2_config.sh                 # choose this run's settings -> config_user.yaml
./stage3_train_bpe_tokenizer.sh    # byte-level BPE
./stage4_tokenize_data.sh          # corpora -> memory-mapped token streams
./stage5_pretrain.sh               # pretraining
./stage6_sft.sh                    # supervised fine-tuning (needs the pretrain checkpoint)
./stage7_train_rlhf_reward.sh      # reward model
./stage8_instruct_tuning_rlhf.sh   # RLHF by PPO (needs the SFT and reward checkpoints)
./stage9_cot_aha_moment.sh         # chain of thought by GRPO
./stage10_instruct_dpo.sh          # direct preference optimisation
python -m distill.run              # sequence-level distillation into gpt2-small
```

Equivalently, bypassing the shell layer:

```bash
python -m pretrain.run
python -m sft.run
python -m instruct_rlhf.run
```

Every knob is a shell variable in `config.sh`, and every default lives in
`default_config.py`; a command-line flag beats both. Overrides apply per run:

```bash
PRETRAIN_STEPS=20000 ./stage5_pretrain.sh
MODEL_SCHEME=zetagpt-m ./stage5_pretrain.sh
GPU=cpu ./stage6_sft.sh
python -m sft.run --sft_steps 5000
```

`python -m tools.config_wizard` writes `config_user.yaml` holding only the settings you choose;
`config.sh` reads it over its own defaults.

The tokenizer -- its special tokens, how to register your own, and how digits and whitespace
are split -- is documented in [`doc/tokenizer.md`](doc/tokenizer.md).

### Using your own data

Point a stage at a directory and it works out the layout by looking at it:

```bash
DATASET=/path/to/my_preferences ./stage7_train_rlhf_reward.sh
SFT_DIR=/path/to/my_demos       ./stage6_sft.sh
PRETRAIN_DIR=/path/to/my_corpus ./stage5_pretrain.sh
```

Talk to a trained checkpoint:

```bash
python chat.py                                # the most aligned checkpoint present
```

## Citation

```bibtex
@article{luo2026zetagpt,
      title={ZetaGPT: A Positional--Encoding--Free State--Space--Attention Compact Language Model},
      author={R\'ois\'in Luo},
      year={2026},
      eprint={2608.09432},
      url={https://arxiv.org/abs/2608.09432},
      notes={https://github.com/roisincrtai/zetagpt},
}
```
