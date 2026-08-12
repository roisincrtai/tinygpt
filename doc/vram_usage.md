# VRAM usage

What one training step costs, term by term, so a scheme can be sized for a card without
running it first — and so a number that turns out wrong can be traced to the term that was
wrong rather than to the total.

Everything below assumes fp32 activations and the settings in `config.sh`. All figures are
GiB (2³⁰ bytes). The symbols used throughout:

| Symbol | Meaning | Where it is set |
|---|---|---|
| `L` | layers | `SCHEMES[scheme]["n_layer"]` |
| `d` | `d_model` | `SCHEMES[scheme]["n_embd"]` |
| `H` | heads, `= d / 64` | `SCHEMES[scheme]["n_head"]` |
| `P` | parameters | derived; see the table below |
| `V` | vocabulary, **50,259** | 256 bytes + 50,000 merges + 3 specials |
| `T` | context window in force | one entry of `SCHEMES[scheme]["context_window"]` |
| `B` | batch, sequences per step | `SCHEME_BATCH[scheme]`, scaled by the window |
| `m` | micro-batch, sequences per forward pass | from `SCHEME_MICRO_TOKENS[scheme]` |
| `C` | loss slice, tokens | `LOSS_CHUNK`, default 8192 |

**Peak memory is one forward and backward pass, not one step.** A step of `B` sequences is
run as `⌈B/m⌉ passes of `m` sequences each, whose gradients accumulate into the parameters.
Only one pass is ever resident, so `B` does not appear in the memory below — only `m` does.

---

## The four terms

### 1. Parameters and optimiser state — fixed, independent of context and batch

`MasterAdamW` keeps five parameter-sized copies: the live weights, their gradients, the fp32
master copy, and Adam's two moments. At fp32 throughout that is 20 bytes per parameter.

```
fixed  =  P × 20 bytes
```

### 2. Block activations — the dominant term

Roughly **29 tensors the width of `d`** are kept per block for the backward pass: the state
space module's `2d` input projection, its convolution, its decay and its gate; attention's
`3d` qkv and its `d` gate; and the feed-forward's `4d` hidden twice over, before and after the
non-linearity.

```
per token  =  L × 29 × d × 4 bytes
activations  =  m × T × (per token)
```

Note that this is linear in **tokens**, not in sequences: one sequence of 8,192 costs exactly
what eight sequences of 1,024 cost. That is why the micro-batch is configured as a token
budget (`SCHEME_MICRO_TOKENS`) rather than as a number of sequences.

### 3. The loss — bounded by the slice, not by the step

The vocabulary projection produces one row of `V` per token, and the loss needs three such
tensors: the logits, their log-softmax, and the gradient of that log-softmax. Chunked loss
(on by default) evaluates `C` tokens at a time through `torch.utils.checkpoint`, so only one
slice exists at a time and each is recomputed during the backward pass.

```
loss  =  3 × C × V × 4 bytes                    chunked   (default)
loss  =  3 × m × T × V × 4 bytes                whole     (--no_chunked_loss)
```

At `C = 8192` that is a flat **4.60 GiB** whatever the window. Unchunked at `m × T = 32,768`
it is 18.41 GiB, and it grows with every sequence in the pass.

### 4. Attention — small on CUDA, ruinous without it

On CUDA, `F.scaled_dot_product_attention` computes attention in tiles and recomputes the
scores in the backward pass, so the `(m, H, T, T)` score matrix is never built. CPU and MPS
take the explicit path, which does build it.

```
attention  ≈  L × H × T × 128 × 4 bytes × m     FlashAttention  (CUDA)
attention  =  L × H × T² × 4 bytes × m          explicit        (CPU, MPS)
```

The second term is why long context is impossible without the first: at `L = 32`, `H = 16`,
`T = 32,768` and `m = 1` it is **2,048 GiB** for one sequence.

### Total

```
peak  ≈  fixed  +  activations  +  loss  +  attention
```

---

## Worked figures

Parameters, from `L` and `d` with the head tied to the embedding:

| Scheme | `L` | `d` | `H` | `P` | fixed (`P × 20`) | per token (`L × 29 × d × 4`) |
|---|---|---|---|---|---|---|
| `zetagpt-tiny` | 8 | 512 | 8 | 61,488,128 | 1.15 GiB | 0.45 MB |
| `zetagpt-s` | 24 | 512 | 8 | 132,996,096 | 2.48 GiB | 1.36 MB |
| `zetagpt-m` | 32 | 512 | 8 | 168,750,080 | 3.14 GiB | 1.81 MB |
| `zetagpt-l` | 32 | 1024 | 16 | 622,712,832 | 11.60 GiB | 3.62 MB |

**The per-token column above is the arithmetic. The measured figure is higher.**
`zetagpt-s` at 8,192 tokens per pass came to 18.1 GiB on an L40S, which puts a token at
**1.88 MB** rather than 1.36 MB — about 1.4× the count above, from temporaries the count does
not see: workspace inside kernels, allocator rounding, and intermediates autograd keeps that
are not among the 29. **Size with the measured figure, not the derived one**, and treat the
derivation as the explanation rather than the prediction.

Scaling the measurement by `L × d` gives the per-token cost used to configure each scheme:

| Scheme | measured per token | `SCHEME_MICRO_TOKENS` | peak at that budget |
|---|---|---|---|
| `zetagpt-tiny` | 0.63 MB | 47,104 | ~34.6 GiB |
| `zetagpt-s` | **1.88 MB** (measured) | 14,336 | ~33.4 GiB |
| `zetagpt-m` | 2.51 MB | 10,240 | ~32.8 GiB |
| `zetagpt-l` | 5.02 MB | 3,072 | ~31.2 GiB |

All four are sized to a **35 GiB working budget on a 44–46 GiB card**. The gap is deliberate:
fragmentation, allocator rounding, kernel workspace and the driver's own reservation come out
of the same total, and a step that fits on average but not at its peak fails hours in rather
than at once.

---

## Every window of every scheme

The peak is not one number per scheme: the context schedule changes `T`, which changes the
sequences per pass and therefore the tokens per pass. `B` is the batch the optimiser sees and
does not enter the memory; `m` is what is resident. Figures in GiB.

**`zetagpt-tiny`** — 8 layers, `d` 512, 0.63 MB/token, fixed 1.15 GiB, `SCHEME_BATCH` 32, `SCHEME_MICRO_TOKENS` 47,104

| Window `T` | `B` | `m` | passes | tokens/pass | activations | loss | attention | **peak** |
|---|---|---|---|---|---|---|---|---|
| 512 | 64 | 64 | 1 | 32,768 | 20.1 | 4.6 | 1.0 | **26.8** |
| 1,024 | 32 | 32 | 1 | 32,768 | 20.1 | 4.6 | 1.0 | **26.8** |

**`zetagpt-s`** — 24 layers, `d` 512, 1.88 MB/token, fixed 2.48 GiB, `SCHEME_BATCH` 3, `SCHEME_MICRO_TOKENS` 14,336

| Window `T` | `B` | `m` | passes | tokens/pass | activations | loss | attention | **peak** |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 24 | 14 | 2 | 14,336 | 26.3 | 4.6 | 1.3 | **34.7** |
| 2,048 | 12 | 7 | 2 | 14,336 | 26.3 | 4.6 | 1.3 | **34.7** |
| 4,096 | 6 | 3 | 2 | 12,288 | 22.6 | 4.6 | 1.1 | **30.8** |
| 8,192 | 3 | 1 | 3 | 8,192 | 15.0 | 4.6 | 0.8 | **22.9** |

**`zetagpt-m`** — 32 layers, `d` 512, 2.51 MB/token, fixed 3.14 GiB, `SCHEME_BATCH` 1, `SCHEME_MICRO_TOKENS` 10,240

| Window `T` | `B` | `m` | passes | tokens/pass | activations | loss | attention | **peak** |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 16 | 10 | 2 | 10,240 | 25.1 | 4.6 | 1.2 | **34.1** |
| 2,048 | 8 | 5 | 2 | 10,240 | 25.1 | 4.6 | 1.2 | **34.1** |
| 4,096 | 4 | 2 | 2 | 8,192 | 20.1 | 4.6 | 1.0 | **28.8** |
| 8,192 | 2 | 1 | 2 | 8,192 | 20.1 | 4.6 | 1.0 | **28.8** |
| 16,384 | 1 | 1 | 1 | 16,384 | 40.1 | 4.6 | 2.0 | **49.9**  ⚠ over 35 |

**`zetagpt-l`** — 32 layers, `d` 1024, 5.01 MB/token, fixed 11.60 GiB, `SCHEME_BATCH` 1, `SCHEME_MICRO_TOKENS` 3,072

| Window `T` | `B` | `m` | passes | tokens/pass | activations | loss | attention | **peak** |
|---|---|---|---|---|---|---|---|---|
| 1,024 | 32 | 3 | 11 | 3,072 | 15.0 | 1.7 | 0.8 | **29.1** |
| 2,048 | 16 | 1 | 16 | 2,048 | 10.0 | 1.2 | 0.5 | **23.3** |
| 4,096 | 8 | 1 | 8 | 4,096 | 20.1 | 2.3 | 1.0 | **35.0** |
| 8,192 | 4 | 1 | 4 | 8,192 | 40.1 | 4.6 | 2.0 | **58.3**  ⚠ over 35 |
| 16,384 | 2 | 1 | 2 | 16,384 | 80.2 | 4.6 | 4.0 | **100.4**  ⚠ over 35 |
| 32,768 | 1 | 1 | 1 | 32,768 | 160.4 | 4.6 | 8.0 | **184.6**  ⚠ over 35 |

Two things the tables make plain. `zetagpt-tiny` and `zetagpt-s` fit their whole schedules,
`zetagpt-m` fits until 16,384 and `zetagpt-l` until 4,096 — and the windows that do not fit
are not close, they are two to five times over. And the peak is highest at the SHORT windows
for `-tiny` and `-s`, because the token budget is filled there and the longest window cannot
even reach it with one sequence.

---

## Sizing a scheme for your own card

1. **Choose a working budget.** Leave 20–25% of the card unspent, for the reasons above.
2. **Subtract the fixed terms.** `P × 20 bytes` for the optimiser, plus `3 × C × V × 4` for
   the loss slice — 4.60 GiB at the default `LOSS_CHUNK=8192`.
3. **Divide by the per-token cost** to get the tokens one pass may hold. Use the measured
   1.88 MB/token for 24 layers of width 512, scaled by `(L × d) / (24 × 512)`.
4. **Write that into `SCHEME_MICRO_TOKENS`.** The number of sequences per pass follows from
   the window: `m = clamp(SCHEME_MICRO_TOKENS ÷ T, 1, B)`.
5. **Check it.** `python -m tools.vram --sweep` builds the real model and runs a real step;
   `nvidia-smi` over the first hundred steps is the other half of the answer.

Worked, for `zetagpt-m` on a 24 GiB card:

```
budget                 24 × 0.78            = 18.7 GiB
minus optimiser        168,750,080 × 20     =  3.1 GiB
minus loss slice       3 × 8192 × 50259 × 4 =  4.6 GiB
                                    leaves     11.0 GiB
per token              1.88 MB × (32×512)/(24×512) = 2.51 MB
tokens per pass        11.0 GiB ÷ 2.51 MB   = 4,480  ->  SCHEME_MICRO_TOKENS = 4096
```

A smaller `LOSS_CHUNK` buys that 4.6 GiB back when the card is very small — 2048 tokens costs
1.15 GiB — at the price of more slices, each recomputed in the backward pass.

---

## What each knob moves

| Knob | Term it changes | Effect on the result |
|---|---|---|
| `SCHEME_BATCH` | none directly | more tokens per optimiser step; more passes at the same peak |
| `SCHEME_MICRO_TOKENS` | activations, attention | **the main lever on peak memory** |
| `MICRO_BATCH` | activations, attention | the same lever, expressed in sequences; overrides the budget |
| `LOSS_CHUNK` | loss | `3 × C × V × 4`; smaller is cheaper and slower |
| `CHUNKED_LOSS=0` | loss | the whole pass at once: `3 × m × T × V × 4` |
| `CONTEXT_WINDOW` | activations, attention | linear in `T` for both, on CUDA |
| scheme `n_layer`, `n_embd` | everything | `P`, and the per-token cost through `L × d` |

None of `SCHEME_BATCH`, `MICRO_BATCH`, `LOSS_CHUNK` or `CHUNKED_LOSS` changes what is
computed. The loss, its gradients and the resulting weights are the same to floating-point
noise whichever way the step is divided.

---

## The case that does not fit

`zetagpt-l` at its 32,768 window needs about 185 GiB for a **single sequence**, and one
sequence is the smallest pass there is — no micro-batch can reduce it. What that needs is
gradient checkpointing over the blocks: storing each block's input and recomputing its
interior during the backward pass, which trades roughly 30% more compute for activations that
fall to about their square root. This pipeline does not do that yet, so `zetagpt-l` trains at
the windows it fits and stops above them.
