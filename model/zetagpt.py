"""
zetagpt.py -- the from-scratch GPT this pipeline trains.

nanoGPT / GPT-2 architecture (pre-LN blocks, GELU, 4x MLP, causal attention, weight-tied lm head,
N(0,0.02) init with residual projections scaled by 1/sqrt(2*n_layer)) with TWO changes:

(1) HOW POSITION ENTERS, chosen by `pe`:

      pe="ssm"   (default)  Each block is SSM -> Attention -> FFN. The state space module
                            (ssm.py) is a causal recurrence, so its output at step t already
                            depends on where t is; attention therefore needs no positional
                            signal of its own and gets none. No position table, no rotation:
                            the model is positional-encoding-free.
      pe="rope"  (ablation) The state space module is removed and rotary positions are applied
                            to the queries and keys inside attention, giving the conventional
                            two-sublayer transformer. This exists so the contribution of the
                            recurrence can be measured against a like-for-like control; it is
                            not the architecture the report proposes.

    The model always has EXACTLY ONE source of position, and which one is recorded in the
    checkpoint, so a run can never be reloaded under the wrong assumption.
(2) GATED ATTENTION -- the attention output is modulated elementwise by an input-dependent
    sigmoid gate before the output projection (disable with gated_attn=False).

THE HEAD DIMENSION IS FIXED AT 64 (HEAD_DIM below) in every configuration, so n_head is
not a free parameter but n_embd / 64. Width is scaled by adding heads, never by widening
them, which is what keeps a result measured at one size interpretable at another.

    ZetaGPT-Tiny  8 layers,  8 heads,  512-dim,  61.5M parameters, context  512
    ZetaGPT-S    16 layers,  8 heads,  512-dim,  97.2M,             context  512  (default)
    ZetaGPT-M    16 layers, 12 heads,  768-dim, 199.3M,             context 1024
    ZetaGPT-L    24 layers, 16 heads, 1024-dim, 479.9M,             context 1024

Default config: ZetaGPT-S, MLP 4x, context 512, pe="ssm" (default_config.MODEL).

HuggingFace-compatible forward, so evaluation and generation code runs unchanged:
    out = model(input_ids, attention_mask=..., mask_positions=...)
    out.logits              (B, T, vocab)

`mask_positions` (B,T bool) marks input tokens replaced by a learnable, input-only MASK embedding
(self.mask_embed): an optional corruption hook, never a target and never in the output vocabulary,
and inert unless it is passed. No KV cache (kept deliberately simple), so generation recomputes
the prefix at every step.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pe import RotaryPositionalEmbedding
from .ssm import CausalSSM

# The head dimension, fixed across every ZetaGPT configuration. Not an argument: a size is
# chosen by picking n_embd, and n_head follows as n_embd / HEAD_DIM.
HEAD_DIM = 64


class Output:
    """Minimal stand-in for a HF model output (.logits, plus .mask_logit when requested)."""
    def __init__(self, logits, mask_logit=None):
        self.logits = logits
        self.mask_logit = mask_logit


class CausalSelfAttention(nn.Module):
    """Causal self-attention -- POSITION-FREE -- with (by default) GATED ATTENTION OUTPUT: the head
    outputs are modulated elementwise by an input-dependent sigmoid gate BEFORE the output
    projection,

        y = W_o ( head_out * sigmoid(W_g x) )

    (output gating a la "Gated Attention for Large Language Models", 2025: query-dependent,
    head-specific since W_g spans all head channels, non-linear). The gate sharpens
    selectivity and suppresses attention-sink/massive-activation pathologies at the cost of
    one extra n_embd x n_embd projection. `gated=False` recovers the plain nanoGPT layer.

    Under pe="ssm" the only position-dependent term here is the causal mask: queries and keys
    are used as they come out of the projection, because the state space module in front of
    this layer has already made them position-aware. Under the pe="rope" ablation there is no
    such module, so rotary positions are applied to q and k here instead."""
    def __init__(self, n_embd, n_head, dropout=0.0, gated=True, use_rope=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.hd = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.gate = nn.Linear(n_embd, n_embd) if gated else None      # output gate W_g
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.hd) if use_rope else None

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        if self.rope is not None:                                     # pe="rope" ablation only
            q, k = self.rope(q, k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)          # (B, nh, T, T)
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        att = att.masked_fill(~causal.view(1, 1, T, T), float("-inf"))
        if attn_mask is not None:                                     # (B, T) padding mask
            att = att.masked_fill(~attn_mask.view(B, 1, 1, T).bool(), float("-inf"))
        att = self.drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        if self.gate is not None:
            y = y * torch.sigmoid(self.gate(x))                       # gated attention output
        return self.proj(y)


class Block(nn.Module):
    """SSM -> Attention -> FFN, each pre-normed and residual:

        x = x + SSM(LN(x))          position, and local/recurrent mixing
        x = x + Attention(LN(x))    content-based mixing over the whole prefix
        x = x + FFN(LN(x))          per-position transformation

    Pre-norm throughout (normalise the branch input, keep the residual stream clean), which
    is what makes deep stacks trainable without warmup tricks. The state space module comes
    FIRST because its job is to make position available: attention then reads vectors that
    already know where they are. Under the pe="rope" ablation the module is absent and the
    block is the standard two-sublayer transformer, with position supplied inside attention
    instead."""
    def __init__(self, n_embd, n_head, dropout=0.0, ffn_factor=4, gated_attn=True,
                 d_conv=4, ssm_chunk="optimal", pe="ssm"):
        super().__init__()
        use_ssm = (pe == "ssm")
        self.ln0 = nn.LayerNorm(n_embd) if use_ssm else None
        self.ssm = CausalSSM(n_embd, d_conv, ssm_chunk, dropout) if use_ssm else None
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, gated=gated_attn,
                                        use_rope=not use_ssm)
        self.ln2 = nn.LayerNorm(n_embd)
        hidden = ffn_factor * n_embd
        self.mlp = nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_embd), nn.Dropout(dropout))

    def forward(self, x, attn_mask=None):
        if self.ssm is not None:
            x = x + self.ssm(self.ln0(x), attn_mask)
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class ZetaGPT(nn.Module):
    """GPT-2 / nanoGPT with implicit positions. lm head is WEIGHT-TIED to the token embedding, init is
    N(0,0.02) with residual projections scaled by 1/sqrt(2*n_layer), exactly as nanoGPT. The extra
    input-only `mask_embed` is an optional corruption hook, inert (zero init) unless
    mask_positions/mask_gate are passed. `block_size` is the intended context length, a
    training choice only: nothing in this model imposes a hard limit, because nothing in it
    refers to an absolute position. `self.cfg` records the constructor arguments and travels
    with the weights in every checkpoint."""
    def __init__(self, vocab_size, n_layer=16, n_head=None, n_embd=512, block_size=512,
                 dropout=0.0, ffn_factor=4, gated_attn=True, d_conv=4, ssm_chunk="optimal",
                 pe="ssm"):
        super().__init__()
        if pe not in ("ssm", "rope"):
            raise ValueError(f"pe must be 'ssm' or 'rope', not {pe!r}")
        # n_head=None derives the head count from the fixed head dimension, which is the
        # intended way to ask for a size: give the width, and the heads follow.
        if n_head is None:
            if n_embd % HEAD_DIM:
                raise ValueError(f"n_embd={n_embd} is not a multiple of HEAD_DIM={HEAD_DIM}; "
                                 f"pass n_head explicitly if that is deliberate")
            n_head = n_embd // HEAD_DIM
        # An explicit n_head that implies a different head dimension is WARNED ABOUT, not
        # rejected: a checkpoint written before HEAD_DIM was fixed carries its own n_head in
        # model_cfg, and refusing it here would make old weights unloadable. The warning says
        # what is non-standard so a stale configuration cannot pass unnoticed either.
        if n_embd % n_head or n_embd // n_head != HEAD_DIM:
            import warnings
            warnings.warn(f"head dimension is {n_embd / n_head:g} (n_embd={n_embd}, "
                          f"n_head={n_head}); every ZetaGPT size uses {HEAD_DIM}",
                          stacklevel=2)
        # The full constructor arguments, saved into every checkpoint so a model can be
        # rebuilt from its weights alone rather than from whatever default_config.py happens to say.
        self.cfg = dict(vocab_size=vocab_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                        block_size=block_size, dropout=dropout, ffn_factor=ffn_factor,
                        gated_attn=gated_attn, d_conv=d_conv, ssm_chunk=ssm_chunk, pe=pe)
        self.block_size = block_size
        self.pe = pe
        self.tok = nn.Embedding(vocab_size, n_embd)           # wte; positions are implicit
        self.mask_embed = nn.Parameter(torch.zeros(n_embd))   # learnable, input-only MASK (zero init)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, ffn_factor, gated_attn,
                                           d_conv, ssm_chunk, pe)
                                     for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok.weight = self.head.weight                    # weight tying (wte <-> lm_head)
        self.apply(self._init)
        self._resize_note = ""
        # nanoGPT scaled init for the residual projections: std = 0.02 / sqrt(2*n_layer).
        # The state space module's output projection writes to the same residual stream, so it
        # is scaled too -- otherwise a third branch per block inflates the stream's variance.
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
            nn.init.normal_(blk.mlp[2].weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

            if blk.ssm is not None:
                nn.init.normal_(blk.ssm.out_proj.weight, mean=0.0,
                                std=0.02 / math.sqrt(2 * n_layer))
                nn.init.constant_(blk.ssm.a_proj.bias, -2.0)   # _init overwrote the bias

    def resize_token_embeddings(self, new_vocab_size, seed=0, allow_shrink=False):
        """Grow the vocabulary to `new_vocab_size`, KEEPING every existing row.

        This is the model half of registering a special token. The tokenizer appends the new
        id after everything already learned, so row i of the embedding still means token i for
        every i that existed before; the matrix simply needs more rows. Fine-tuning a
        pretrained checkpoint on a corpus that uses <xyz> therefore costs one new embedding
        row, not a retrained model.

        NEW ROWS ARE INITIALISED AT THE MEAN of the existing embeddings plus a little noise,
        rather than at zero or at N(0, 0.02). The mean is where a token of no particular
        meaning belongs: it starts out maximally unopinionated instead of being a large
        outlier the first gradients have to drag back into the cloud. The noise breaks the
        symmetry between several tokens registered at once, and is scaled to the spread of the
        rows already there, so it does not depend on the width of the model. HuggingFace's
        mean_resizing does the same thing for the same reason.

        Weight tying is re-established explicitly: the embedding and the output head are ONE
        parameter here, and a resize that rebuilt them separately would silently untie them --
        the model would still run, and would train two copies of a matrix the architecture
        says is one.
        """
        old = self.tok.weight.shape[0]
        if new_vocab_size == old:
            return self
        if new_vocab_size < old and not allow_shrink:
            # allow_shrink is for ONE caller: sizing a freshly built model down to match a
            # checkpoint before loading it. Nothing is lost there because nothing is trained
            # yet. Shrinking a trained model is refused, because it can only mean the
            # vocabulary was rebuilt.
            raise ValueError(
                f"refusing to shrink the vocabulary {old} -> {new_vocab_size}: dropping rows "
                f"would change what the surviving ids mean only if the tokenizer also "
                f"changed, in which case this checkpoint belongs to a different vocabulary.")
        w = self.tok.weight.data
        keep = min(old, new_vocab_size)
        emb = nn.Embedding(new_vocab_size, w.shape[1])
        emb.weight.data[:keep] = w[:keep].cpu()
        if new_vocab_size > old:
            mu, sd = w.mean(0, keepdim=True), w.std().item()
            g = torch.Generator(device="cpu").manual_seed(seed)
            emb.weight.data[old:] = (
                mu.cpu().repeat(new_vocab_size - old, 1)
                + 0.02 * sd * torch.randn(new_vocab_size - old, w.shape[1], generator=g))
        emb = emb.to(w.device, w.dtype)
        head = nn.Linear(w.shape[1], new_vocab_size, bias=False).to(w.device, w.dtype)
        self.tok, self.head = emb, head
        self.head.weight = self.tok.weight                    # RE-TIE; see the docstring
        self.cfg["vocab_size"] = new_vocab_size
        self._resize_note = (
            f"vocabulary {old} -> {new_vocab_size}: {new_vocab_size - old} row(s) added at "
            f"the embedding mean" if new_vocab_size > old else
            f"vocabulary {old} -> {new_vocab_size}")
        return self

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _trunk(self, input_ids, attention_mask, mask_positions, mask_gate, mask_vec, sub_ids=None):
        emb = self.tok(input_ids)
        tgt = self.tok(sub_ids) if sub_ids is not None else (
            self.mask_embed if mask_vec is None else mask_vec)
        if mask_positions is not None:                       # hard boolean replace
            emb = torch.where(mask_positions.unsqueeze(-1), tgt.to(emb.dtype), emb)
        if mask_gate is not None:                            # differentiable gate
            g = mask_gate.unsqueeze(-1).to(emb.dtype)        # (B,T,1) in [0,1]
            emb = emb + g * (tgt.to(emb.dtype) - emb)        # g=1 -> substitute/MASK
        x = self.drop(emb)                                   # no position added here: see Block
        for blk in self.blocks:
            x = blk(x, attention_mask)
        return self.lnf(x)

    def hidden(self, input_ids=None, attention_mask=None, mask_positions=None, mask_gate=None,
               mask_vec=None, sub_ids=None, **kw):
        """Final-layer hidden states (B,T,n_embd) -- what a linear mask head reads."""
        return self._trunk(input_ids, attention_mask, mask_positions, mask_gate, mask_vec, sub_ids)

    def forward(self, input_ids=None, attention_mask=None, mask_positions=None, mask_gate=None,
                mask_vec=None, sub_ids=None, mask_logits=False, **kw):
        """`sub_ids` makes the gate interpolate toward substitute-token embeddings (resampling
        corruption); otherwise it interpolates toward the MASK embedding."""
        x = self._trunk(input_ids, attention_mask, mask_positions, mask_gate, mask_vec, sub_ids)
        ml = (x @ self.mask_embed.to(x.dtype)) if mask_logits else None
        return Output(self.head(x), ml)
