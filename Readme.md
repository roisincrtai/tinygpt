# ZetaGPT: A Positional–Encoding–Free State–Space–Attention Compact Language Models

**ZetaGPT** is a compact, positional-encoding-free (NoPE) language model architecture that
uses a state-space-attention mixture block to handle long-context sequences implicitly. It
provides an end-to-end design blueprint covering tokenizer training, pre-training,
supervised instruction tuning (SFT), and reward modelling/RLHF.

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
| `zetagpt-tiny` | 8 | 8 | 512 | 64 | 4× | 512 | 25.7M | 35.8M | 61.5M |
| `zetagpt-s` (default) | 16 | 8 | 512 | 64 | 4× | 512 | 25.7M | 71.5M | **97.2M** |
| `zetagpt-m` | 16 | 12 | 768 | 64 | 4× | 1024 | 38.6M | 160.7M | 199.3M |
| `zetagpt-l` | 24 | 16 | 1024 | 64 | 4× | 1024 | 51.5M | 428.4M | 479.9M |

Select a scheme with `MODEL_SCHEME=zetagpt-m`, or override the context alone with
`CONTEXT_WINDOW`. The **Context** column is the window each scheme is *trained* at, not a limit:
nothing in the model refers to an absolute position. `zetagpt-tiny` and `zetagpt-s` share a
width, a context and a corpus, differing only in depth; point `PRETRAIN_DIR` at your own for
the two larger schemes.

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

Stage 1 is the only stage that reaches the network; every stage after it reads a local
directory and stops if it is not there. Stage 3 tokenises each corpus once into a
memory-mapped `.tokens` stream, so training pages the corpus in from disk rather than holding
it in RAM -- a 2B-token corpus is 4 GB on disk and costs the training process almost nothing.
It needs no GPU, and training stages rebuild a missing stream themselves, so skipping it costs
time rather than correctness. Already-present datasets are skipped and an
interrupted transfer resumes, so re-running is cheap. `./stage1_download_data.sh --list` shows
what would be fetched and where it goes.

Stages are independent and communicate only through files on disk, so each can be run,
re-run or replaced alone. Every stage requires the tokenizer, and each later stage requires
the checkpoint of the one it starts from.

```bash
./stage1_download_data.sh          # fetch the corpora
./stage2_train_bpe_tokenizer.sh    # byte-level BPE
./stage3_tokenize_data.sh          # corpora -> memory-mapped token streams
./stage4_pretrain.sh               # pretraining
./stage5_sft.sh                    # supervised fine-tuning (needs the pretrain checkpoint)
./stage6_train_rlhf_reward.sh      # reward model
./stage7_instruct_tuning_rlhf.sh   # RLHF by PPO (needs the SFT and reward checkpoints)
./stage8_cot_aha_moment.sh         # chain of thought by GRPO
./stage9_instruct_dpo.sh           # direct preference optimisation
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
PRETRAIN_STEPS=20000 ./stage4_pretrain.sh
MODEL_SCHEME=zetagpt-m ./stage4_pretrain.sh
GPU=cpu ./stage5_sft.sh
python -m sft.run --sft_steps 5000
```

`python config_wizard.py` writes `config_user.yaml` holding only the settings you choose;
`config.sh` reads it over its own defaults.

The tokenizer -- its special tokens, how to register your own, and how digits and whitespace
are split -- is documented in [`doc/tokenizer.md`](doc/tokenizer.md).

### Using your own data

Point a stage at a directory and it works out the layout by looking at it:

```bash
DATASET=/path/to/my_preferences ./stage6_train_rlhf_reward.sh
SFT_DIR=/path/to/my_demos       ./stage5_sft.sh
PRETRAIN_DIR=/path/to/my_corpus ./stage4_pretrain.sh
```

A directory holding `*_train/` and `*_test/` subdirectories of json batches keeps **its own
split**; anything else is read as a plain folder of `.json` / `.jsonl` records, shuffled and
cut at `VAL_FRAC`. Records may be a list, one per jsonl line, or wrapped under `pairs` /
`records` / `data`, and the fields may be spelled `prompt` / `instruction` / `question`,
`chosen` / `response` / `output`, `rejected` / `reject`. A prompt-only bank may also be a
folder of `.txt`, one prompt per line. Every stage prints the directory it read and the layout
it read it as.

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
