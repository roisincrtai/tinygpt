"""
kv_cache.py -- incremental decoding state, and the budget that bounds it.

    Cache(n_layer, budget_bytes)      an empty cache
    budget_bytes(gib)                the configured --kv_cache_size as bytes
    cache.fits(batch, tokens, cfg)    would decoding this batch stay inside the budget?
    plan(n_seq, tokens, cfg, budget)  how many sequences to decode at once
    bytes_per_token(cfg)              the arithmetic, exposed so a caller can size its own run
    check(model, device)              PROVE cached decoding equals uncached (see below)

WHY A CACHE AT ALL. Generation without one recomputes the whole prefix at every step: producing
n tokens costs O(n^2) forward passes' worth of work instead of O(n). At the 200-token rollouts
this pipeline used to do that was tolerable; with the response length now filling the context
window -- up to 8,000 tokens for zetagpt-s -- it is not.

WHAT HAS TO BE CACHED HERE IS NOT ONLY K AND V. Every block is SSM -> attention -> FFN, and the
state space module is a recurrence: its output at step t depends on every earlier step through
a running state. Caching keys and values while recomputing the recurrence would leave the cost
quadratic anyway. So each layer keeps FOUR tensors:

    k, v     (B, heads, tokens, head_dim)   attention's keys and values, as usual
    conv     (B, d_conv - 1, C)             the depthwise causal convolution's input window
    h        (B, C)                         the recurrence state after the last token

The last two are what make the SSM steppable. The convolution looks back d_conv - 1 inputs, so
that many are kept; the recurrence needs only its most recent state, because
h_t = a_t * h_{t-1} + (1 - a_t) * v_t depends on the past through h_{t-1} alone. That is the
whole point of a first-order recurrence and the reason this is cheap: the SSM's state is O(1)
in the sequence length, while attention's is O(tokens).

THE BUDGET. Attention's part grows with every token generated, and a GRPO step decodes
batch x group_size sequences at once -- 128 of them at the default 16 x 8. Left unbounded that
is tens of gigabytes on a long window. `plan()` divides the sequences into groups that fit the
budget and the caller decodes one group at a time; the results are identical, since sequences
are independent, and only the peak changes.

    bytes = 2 (k and v) x layers x C x tokens x 4 x batch      attention, grows with tokens
          + layers x (d_conv - 1 + 1) x C x 4 x batch          the SSM, fixed

VERIFY BEFORE TRUSTING. An incremental decoder that is subtly wrong does not crash: it produces
fluent, plausible, WRONG text, and the run looks healthy while the policy learns from rollouts
the model would never have produced. `check()` runs the same prompt through both paths and
compares the logits. Run it once on the machine you train on, after any change to the model:

    python -c "import torch, helpers.kv_cache as kv, helpers.common as c; \\
               a=c.parse_args([]); ctx=c.setup(a); print(kv.check(ctx['new_model'](), ctx['device']))"
"""
import math


GIB = 1024 ** 3


def budget_bytes(size_gib):
    """A configured size in GiB as bytes. 0 or less means unbounded."""
    try:
        return max(0, int(float(size_gib) * GIB))
    except (TypeError, ValueError):
        return 0


def bytes_per_token(cfg, dtype_bytes=4):
    """Attention cache bytes for ONE token of ONE sequence: k and v, every layer."""
    return 2 * int(cfg["n_layer"]) * int(cfg["n_embd"]) * dtype_bytes


def fixed_bytes(cfg, dtype_bytes=4):
    """The SSM's part, which does not grow with the sequence: the conv window and the state."""
    d_conv = int(cfg.get("d_conv", 4))
    return int(cfg["n_layer"]) * (max(d_conv - 1, 0) + 1) * int(cfg["n_embd"]) * dtype_bytes


def cache_bytes(cfg, batch, tokens, dtype_bytes=4):
    """What decoding `batch` sequences to `tokens` will hold at its peak."""
    return batch * (tokens * bytes_per_token(cfg, dtype_bytes) + fixed_bytes(cfg, dtype_bytes))


def plan(n_seq, tokens, cfg, budget_bytes, dtype_bytes=4):
    """How many sequences to decode at once, and in how many groups.

    Returns (per_group, n_groups). At least one sequence always goes through: a budget too
    small for a single sequence is a budget that cannot be honoured, and refusing to decode at
    all would be worse than exceeding it -- so one sequence is decoded and the caller is told
    what it actually costs rather than silently given nothing."""
    one = cache_bytes(cfg, 1, tokens, dtype_bytes)
    per = max(1, int(budget_bytes // one)) if one else max(1, n_seq)
    per = min(per, max(1, int(n_seq)))
    return per, int(math.ceil(n_seq / per))


class Cache:
    """Per-layer decoding state. A plain container: the model reads and writes it, and nothing
    here knows what a transformer is.

    `append` is the only way tokens enter, so `self.tokens` cannot drift from what the tensors
    hold -- a length counted separately from the thing it counts is a bug waiting for a batch
    that ends early."""

    __slots__ = ("k", "v", "conv", "h", "tokens", "budget")

    def __init__(self, n_layer, budget_bytes=0):
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.conv = [None] * n_layer
        self.h = [None] * n_layer
        self.tokens = 0
        self.budget = int(budget_bytes)

    def __len__(self):
        return self.tokens

    def append(self, layer, k, v):
        """Concatenate this step's keys and values; return the full history for attention."""
        import torch
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def advance(self, n=1):
        """Count the tokens just added. Called ONCE per step by the model, after every layer."""
        self.tokens += int(n)

    def nbytes(self):
        tot = 0
        for group in (self.k, self.v, self.conv, self.h):
            for t in group:
                if t is not None:
                    tot += t.numel() * t.element_size()
        return tot

    def over_budget(self):
        return bool(self.budget) and self.nbytes() > self.budget


def check(model, device, prompt_len=13, new_tokens=9, tol=2e-4, seed=0):
    """PROVE that decoding with the cache gives the logits decoding without it gives.

    Runs a random prompt through the model twice: once feeding the whole sequence at every
    step, once feeding one token at a time with a cache. Returns (ok, max_abs_difference).

    This is the check worth having because the failure it catches is silent. An incremental
    decoder that drops the convolution window, or forgets the recurrence state, or attends to
    one token too few, still emits fluent text -- and a GRPO run then optimises against
    rollouts the model would never have produced, with every curve looking healthy."""
    import torch
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = int(model.cfg["vocab_size"])
    ids = torch.randint(0, V, (2, prompt_len), generator=g).to(device)

    with torch.no_grad():
        # WITHOUT the cache: the whole sequence, every step. The reference.
        ref = []
        seq = ids
        for _ in range(new_tokens):
            logits = model(input_ids=seq).logits[:, -1]
            ref.append(logits)
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)

        # WITH the cache: the prompt once, then one token at a time.
        cache = Cache(len(model.blocks))
        got = []
        logits = model(input_ids=ids, cache=cache).logits[:, -1]
        got.append(logits)
        nxt = logits.argmax(-1, keepdim=True)
        for _ in range(new_tokens - 1):
            logits = model(input_ids=nxt, cache=cache).logits[:, -1]
            got.append(logits)
            nxt = logits.argmax(-1, keepdim=True)

    diff = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    return diff <= tol, diff
