# ZetaGPT: A Positional–Encoding–Free State–Space–Attention Compact Language Model

**ZetaGPT** is a positional-encoding-free (NoPE) language model architecture. Sequence order is
carried by the recurrence of a causal state-space module — position is a property of the
operator, not a vector added to it — so there is no position table, no rotation, and nothing to
rescale or interpolate when the context grows. The architecture is the subject of the work; the
full training pipeline, from byte-level BPE through pretraining, instruction tuning, reward
modelling, RLHF, GRPO and DPO, is here so that the claim can be reproduced end to end rather
than asserted from a single loss curve.

[ZetaGPT: A Positional--Encoding--Free State--Space--Attention Compact Language
Model](https://arxiv.org/abs/2608.09432)

## Core design philosophy

- **No positional encoding, by construction.** Not a position signal that has been removed and
  compensated for elsewhere: the causal state-space module is order-sensitive as an operator,
  so order is already present when attention receives it. There is no position table and no
  rotation anywhere in the model.
- **State-space-attention mixture.** Every block is a causal selective state-space module,
  gated multi-head attention and a feed-forward network. The recurrence carries order and
  locality; attention is left free to carry content.
- **Built for length.** Context is what the architecture is for. With no rotary base to retune
  and no interpolation scheme to choose, extending the window is a training decision rather
  than a surgical one — the same weights simply run longer.
- **Reproducible end to end.** Every stage runs on its own, in plain PyTorch, from a bare
  checkout.

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
| `zetagpt-s` (default) | 24 | 8 | 512 | 64 | 4× | 8192 | 25.7M | 107.3M | **133.0M** |
| `zetagpt-m` | 32 | 8 | 512 | 64 | 4× | 16384 | 25.7M | 143.0M | 168.8M |
| `zetagpt-l` | 32 | 16 | 1024 | 64 | 4× | 32768 | 51.5M | 571.2M | 622.7M |

### Context window scheduling

Pretraining does not sit at one sequence length. Each scheme lists the windows it trains
through — `zetagpt-m` is `[1024, 2048, 4096, 8192, 16384]` — and each window takes an equal share of the
step budget, shortest first. The batch is sized at the longest window and scales up as the
window shortens, so tokens per step stay constant across the whole run.

### Mix-Precision Training

**bfloat16 activations, fp32 weights and fp32 optimiser state.** Every stage, on by default
wherever the hardware has it — bf16 for the matmuls and the backward pass, fp32 for everything
that accumulates: the weights, Adam's moments, the gradient accumulation, and the reductions
(layer norm, softmax, logsumexp, the loss). No weight is ever cast, and no stage has a dtype of
its own.

```bash
./stage5_pretrain.sh                          # on, if the GPU is CUDA with bf16
MIXED_PRECISION=0 ./stage5_pretrain.sh        # off: fp32 everywhere
python -m pretrain.run --no-mixed_precision   # off, for one run
```

It resolves against the device rather than being asserted: CPU, MPS and pre-Ampere cards are
told they are training in fp32 instead of being left with a flag that reads as on. The run
prints which it got:

```
precision: bf16 activations, fp32 weights and optimiser state (mixed)
```

bf16 and not fp16 because it keeps fp32's eight exponent bits, so gradients neither overflow nor
underflow and no loss scaling is needed — there is no `GradScaler` here and none is wanted. fp32
weights and not pure bf16 because an update below bf16's three significant digits rounds away
entirely, and training then stalls on a curve that merely looks flat.

### Parallelism

With more than one GPU the model's **layers** are split across them automatically: 2 GPUs and
16 blocks puts blocks 0–7 on `cuda:0` and 8–15 on `cuda:1`, the hidden states crossing at the
boundary. The tied embedding and vocabulary projection share the last device. This buys
**memory, not speed** — the devices take turns, so two cards give roughly the memory of two and
the throughput of one. It is what makes a model or a context window run that does not fit one
card at all. Turn it off with `TENSOR_PARALLEL=0` or `--no-tensor_parallel` whenever the run
already fits one card, where a single device is faster.

### Key-value prefix cache

Generating *n* tokens without a cache recomputes the whole prefix *n* times. `helpers/kv_cache.py`
keeps four tensors per layer rather than two — attention's keys and values, and the state-space
module's convolution window and recurrence state, since a block is SSM → attention → FFN and the
recurrence would otherwise be re-run over the prefix anyway. `--kv_cache_size` bounds it, in GiB (2 by
default): a rollout larger than that is split into groups that fit and decoded a group at a
time, which changes only the peak. `--no-kv_cache` recomputes instead.

### Chain-of-Thought (CoT) from Zero

Stage 9 is two sub-stages. First a supervised fine-tune teaches the model to follow the
instructed output format — the dataset's reference traces, rewritten into
`<think>...</think> <answer>...</answer>` — writing
`checkpoints/cot/checkpoint_<run>_cot-sft.pt`. GRPO then starts from that checkpoint and
optimises against an arithmetically verified answer. The supervised step exists to avoid a
dead reward signal: GRPO's advantage is a reward minus its group's mean, so a format the
policy never samples is a reward every completion earns alike, every advantage is zero, and
the run trains on nothing while its curves look healthy. `--no-cot_sft` skips it, for the
R1-Zero setting.

### Instruction following

Stage 6 fine-tunes the pretrained checkpoint on the Tülu 3 SFT mixture — 939k conversations of
reasoning, code, mathematics, safety and several dozen languages, shipped as parquet shards of
`messages` turns. Every conversation is flattened into **one demonstration per assistant turn**,
each carrying everything said before it as its prompt, so a six-turn conversation is three
demonstrations rather than one and the loss never lands on the user's words. The transcript form
is the preference data's own — `\n\nHuman: … \n\nAssistant:` — which is what lets stages 7, 8 and
10 score, roll out and align a model in the format they prompt it in.

Nothing is pre-tokenised: the demonstrations are held as text and encoded in the collate step,
because the loss mask and the transcript markers are decided there, and a mixture this size
tokenises in minutes against a run of tens of thousands of steps. The budget is one epoch —
`ceil(939,343 / 32) = 29,355` steps at 10⁻⁶ — and the stage measures the true epoch length once
the corpus is loaded and prints it beside the configured one, so a budget that has stopped being
an epoch says so.

This stage sizes its own batch (`SFT_BATCH`, 32) rather than taking `SCHEME_BATCH`, which is 3
for `zetagpt-s` because a *pretraining* sequence fills the whole 8,192-token window while a
demonstration is a few hundred tokens. A batch is padded to its longest record and the mixture
is ragged, so the micro-batch is one sequence per forward pass (`SFT_MICRO_BATCH`): a fixed
count of sequences does not bound the tokens per pass, and one sequence does — at most the
context window, inside the measured per-pass budget. Raise it if throughput matters more than
the headroom.

### Multilingual SFT

Stage 11 adapts the pretrained model to another language in three sub-stages. The new corpus is
first pre-tokenised with the BPE trained on the pretraining set, and that vocabulary is then
*continued* — extended with merges learned on the new corpus, so every existing id keeps its
meaning and only new ones are appended, writing
`checkpoints/lang-sft/<dataset>/bpe/bpe.json`. The corpus is then tokenised with the extended
vocabulary into memory-mapped shards, and the pretrained checkpoint is fine-tuned on that
stream with its embedding grown to match. Extending rather than retraining the vocabulary is
what keeps the adaptation cheap: a byte-level BPE never fails on an unseen language, it just
spends four or five tokens on words an adapted vocabulary spells in one, while a *fresh*
vocabulary would tokenise well and discard the pretrained model with it, since new ids mean the
learned embedding rows sit against different tokens.

### Customized configuration scheme

A scheme fixes depth, width, the context schedule and the corpus together. Here is a small one
end to end — `zetagpt-baby`, 6 layers at `d_model` 512, training through 256 then 512 tokens on
WikiText-103.

**Step 1 — edit `default_config.py`.** Add one entry to each of four dictionaries. They are
already in the file; add your line inside each, beside the existing schemes:

```python
# default_config.py ~line 266 -- the architecture and the context schedule
SCHEMES = {
    ...
    "zetagpt-baby": dict(n_layer=6, n_head=8, n_embd=512,
                         context_window=[256, 512]),   # trains at 256, then at 512
}

# ~line 144 -- which corpus this scheme trains on
PRETRAIN_CORPUS = {
    ...
    "zetagpt-baby": dataset_dir("zetagpt-tiny_pretrain-corpus_wikitext103"),
}

# ~line 286 -- BATCH SIZE: how many SEQUENCES go through one step. Not a length.
# It is the batch for the LONGEST window (512 here); shorter windows scale it up.
SCHEME_BATCH = {
    ...
    "zetagpt-baby": 64,        # 64 sequences x 512 tokens = 32,768 tokens per step
}

# ~line 309 -- sequences per forward pass when a step is too big to fit at once.
# 0 = off: this model is small enough that the whole step goes through in one pass.
SCHEME_MICRO_TOKENS = {
    ...
    "zetagpt-baby": 0,
}
```

Three numbers are easy to confuse, so to be explicit: `context_window=[256, 512]` is the
**sequence length**, `SCHEME_BATCH = 64` is **how many sequences per step**, and
`SCHEME_MICRO_TOKENS = 0` is **how the step is split across forward passes**. Only the first
is a length.

`n_head` is not a free choice either: the head dimension is 64 everywhere, so it is
`n_embd / 64` — 512 / 64 = 8. The largest entry of `context_window` is the model's context, so
this model has a 512-token context.

**Step 2 — set the scheme in `config.sh`**, or override it per run:

```bash
MODEL_SCHEME="${MODEL_SCHEME:-zetagpt-baby}"      # config.sh, ~line 165
```

```bash
MODEL_SCHEME=zetagpt-baby ./stage5_pretrain.sh    # or just for this run
```

The batch is sized at the longest window and scales up as the window shortens, so tokens per
step stay constant — over a 20,000-step budget this scheme runs steps 0–9,999 at context 256
with batch 128, then steps 10,000–19,999 at context 512 with batch 64.

Nothing else needs changing: `PRETRAIN_DIR` is blank in `config.sh` so each scheme uses its own
corpus, and every artefact is named after the scheme
(`checkpoints/pretrain/checkpoint_zetagpt-baby_ssm_pretrain.pt`).

**Without editing Python at all**, `config.sh` reaches the same values for one run:
`CONTEXT_WINDOW=256,512` pins a schedule, `BATCH=64` the batch, and `EXTRA_SET="--set
MODEL.n_layer=6"` any other architectural value.

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
| **ZetaGPT-Tiny** | **61.5M** | **State–Space–Attention** | **None** | **1024** |
| **ZetaGPT-S** (default) | **133.0M** | **State–Space–Attention** | **None** | **8192** |
| **ZetaGPT-M** | **168.8M** | **State–Space–Attention** | **None** | **16384** |
| **ZetaGPT-L** | **622.7M** | **State–Space–Attention** | **None** | **32768** |

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
./stage6_instruct_sft.sh           # instruction SFT (needs the pretrain checkpoint)
./stage7_train_rlhf_reward.sh      # reward model
./stage8_instruct_tuning_rlhf.sh   # RLHF by PPO (needs the SFT and reward checkpoints)
./stage9_cot_aha_moment.sh         # chain of thought by GRPO
./stage10_instruct_dpo.sh          # direct preference optimisation
./stage11_lang_sft.sh              # language adaptation: extend the vocabulary, then adapt
python -m distill.run              # sequence-level distillation into gpt2-small
```

Equivalently, bypassing the shell layer:

```bash
python -m pretrain.run
python -m instruct_sft.run
python -m instruct_rlhf.run
```

Every knob is a shell variable in `config.sh`, and every default lives in
`default_config.py`; a command-line flag beats both. Overrides apply per run:

```bash
PRETRAIN_STEPS=20000 ./stage5_pretrain.sh
MODEL_SCHEME=zetagpt-m ./stage5_pretrain.sh
GPU=cpu ./stage6_instruct_sft.sh
python -m instruct_sft.run --sft_steps 5000
```

`python -m tools.config_wizard` writes `config_user.yaml` holding only the settings you choose;
`config.sh` reads it over its own defaults.

The tokenizer -- its special tokens, how to register your own, and how digits and whitespace
are split -- is documented in [`doc/tokenizer.md`](doc/tokenizer.md).

### Using your own data

Point a stage at a directory and it works out the layout by looking at it:

```bash
DATASET=/path/to/my_preferences ./stage7_train_rlhf_reward.sh
SFT_DIR=/path/to/my_demos       ./stage6_instruct_sft.sh
PRETRAIN_DIR=/path/to/my_corpus ./stage5_pretrain.sh
```

Stage 6 reads three layouts, and says in its log which one it decided yours is: a **conversation
mixture** (parquet or jsonl carrying a `messages` list of `{role, content}` turns), a
**preference tree** (`<dir>/*_train/*.json`, of which it takes the chosen response), or a
**folder of json/jsonl records** with any of the usual prompt and response field names.

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
