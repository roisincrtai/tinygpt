"""
ssm.py -- the causal state space module that gives ZetaGPT its positions.

Attention is permutation-equivariant, so a transformer must learn position from somewhere.
The usual answers add it to the representation (a learned table) or rotate the queries and
keys (RoPE). This module is the alternative ZetaGPT uses instead, always: a CAUSAL
RECURRENCE placed before attention, whose state at step t summarises tokens 1..t, so the
vector attention reads at position t already encodes where t is. No position table, no
rotation, no explicit index anywhere in the model -- NoPE.

    blocked_scan(a, v, chunk)   h_t = a_t h_{t-1} + v_t, in O(chunk + T/chunk) steps
    CausalSSM                   the layer: selective diagonal SSM + depthwise causal conv

This module is not optional and there is no switch that replaces it. Remove it and the model
would be position-blind, because nothing else in ZetaGPT encodes an index.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def optimal_chunk(T):
    """The chunk size that minimises the blocked scan's sequential depth.

    Depth is  d(c) = c + T/c,  which is convex with  d'(c) = 1 - T/c^2 = 0  at  c = sqrt(T),
    giving  d = 2 sqrt(T)  -- the best any two-level blocking can do. So the chunk is not a
    hyperparameter to guess: it follows from the sequence length, and since T varies from
    batch to batch (rollouts are short, packed documents are long) the right value is computed
    per forward pass rather than fixed once in the configuration.

        T =  128 -> chunk  11, depth  23   (instead of  128)
        T =  512 -> chunk  23, depth  45   (instead of  512)
        T = 2048 -> chunk  45, depth  91   (instead of 2048)
    """
    return max(1, min(int(round(math.sqrt(max(T, 1)))), T))


def blocked_scan(a, v, chunk=None):
    """The first-order recurrence h_t = a_t * h_{t-1} + v_t, over (B, T, C), computed in
    O(chunk + T/chunk) sequential steps instead of T. `chunk=None` (or "optimal") uses
    optimal_chunk(T) = round(sqrt(T)), which minimises that depth.

    Split the sequence into N chunks of `chunk` steps. Chunks are independent GIVEN their
    initial state, so the within-chunk scan runs for every chunk AT ONCE (chunk sequential
    steps, batched over N); the chunks are then stitched by a second scan over N carries.
    For T=512 and chunk=32 that is 32 + 16 = 48 python-level steps rather than 512.

    Stability: the cumulative log-decay is taken WITHIN a chunk only, so every exponent is
    a sum of non-positive terms and exp() cannot overflow. The textbook parallel form
    (divide by the running product) computes exp(+A_s) and overflows on long sequences,
    which is the trap this avoids.
    """
    B, T, C = v.shape
    if chunk is None or isinstance(chunk, str):
        chunk = optimal_chunk(T)
    chunk = max(1, min(int(chunk), T))
    pad = (-T) % chunk
    if pad:
        a = F.pad(a, (0, 0, 0, pad), value=1.0)
        v = F.pad(v, (0, 0, 0, pad))
    N = (T + pad) // chunk
    a = a.view(B, N, chunk, C)
    v = v.view(B, N, chunk, C)

    h = torch.zeros(B, N, C, dtype=v.dtype, device=v.device)
    within = []
    for t in range(chunk):                       # every chunk advances together
        h = a[:, :, t] * h + v[:, :, t]
        within.append(h)
    within = torch.stack(within, dim=2)          # (B, N, chunk, C), zero initial state

    # per-chunk total decay and the running carry the next chunk starts from
    decay = torch.cumprod(a, dim=2)              # (B, N, chunk, C); decay[..., -1, :] = prod
    carry = torch.zeros(B, C, dtype=v.dtype, device=v.device)
    carries = []
    for n in range(N):
        carries.append(carry)
        carry = decay[:, n, -1] * carry + within[:, n, -1]
    carries = torch.stack(carries, dim=1)        # (B, N, C)

    out = within + decay * carries.unsqueeze(2)
    return out.reshape(B, N * chunk, C)[:, :T]


# --------------------------------------------------------------------------- #
# diagnostics
# --------------------------------------------------------------------------- #
# WHAT IS WORTH WATCHING, and why. The decay a_t is the interpretable quantity: it converts
# straight into a MEMORY HORIZON, tau = -1/ln(a), in tokens. At initialisation a ~ 0.87, so
# tau ~ 8 -- the module starts as a local filter, and whether that horizon grows during
# training is the single most informative thing about this architecture. The rest guard
# against the failure modes:
#
#   horizon_med, horizon_p95   how far back the module remembers, in tokens
#   frac_long, frac_local      the split between carry channels (a > 0.99) and detail ones
#   selectivity                std of a ACROSS TOKENS. If it collapses to 0 the decay has
#                              stopped depending on the input and the module has degenerated
#                              into a time-invariant filter -- the "selective" part doing no
#                              work at all
#   write_ratio                ||output|| / ||input||: whether the residual stream is actually
#                              using the module, or the network has routed around it
#   state_norm                 mean |h|, which the convex-average form should keep bounded
#
# Collection is OFF by default and costs nothing when off; the trainer switches it on for one
# step at the figure cadence, so the cost is amortised to nothing.
def collect_stats(model, on=True):
    """Turn diagnostics on or off for every state space module in `model`."""
    n = 0
    for m in model.modules():
        if isinstance(m, CausalSSM):
            m.collect = bool(on)
            n += 1
    return n


def layer_stats(model):
    """The diagnostics of the most recent forward pass, ONE ENTRY PER BLOCK, in depth order.
    Empty list if nothing was collected.

    Nothing is averaged here. Layers differ -- an early block doing local work and a late one
    carrying long context is the interesting case, and a mean over the two hides exactly that.
    The history therefore stores what was measured; the figure, and any later analysis, can
    summarise it however they like from the raw record."""
    return [dict(m.stats) for m in model.modules()
            if isinstance(m, CausalSSM) and m.stats]


class CausalSSM(nn.Module):
    """A selective diagonal state space module -- the block's positional encoder under NoPE.

    Attention is permutation-equivariant, so a transformer needs position from somewhere. The
    usual answer is to add it to the representation (a learned table) or to rotate the
    queries and keys (RoPE). The answer used here is to put a CAUSAL RECURRENCE in front
    of attention: at step t its state summarises tokens 1..t, so the vector attention sees at
    position t already carries where t is. No position table, no rotation, no explicit index
    anywhere in the model.

    The recurrence is diagonal (one scalar state per channel) and SELECTIVE (the decay is a
    function of the input, as in Mamba/S6 rather than the time-invariant S4):

        a_t = exp(-softplus(W_a x_t))    in (0,1), per channel
        h_t = a_t h_{t-1} + (1 - a_t) v_t
        y_t = W_o ( h_t * SiLU(W_g x_t) )

    a_t near 1 holds the state (long memory), near 0 forgets it (local detail), and because
    a_t depends on x_t the layer can decide per token and per channel which to do. The
    (1 - a_t) factor keeps the state a convex average of its inputs, so activations stay
    bounded regardless of sequence length. A short depthwise causal convolution before the
    recurrence supplies exact local offsets (`the cat` vs `cat the`) that a smooth decay
    alone represents poorly.
    """
    def __init__(self, n_embd, d_conv=4, chunk="optimal", dropout=0.0):
        super().__init__()
        # chunk="optimal" (the default) computes round(sqrt(T)) per forward pass, which is the
        # depth-minimising choice; an integer pins it, which is only useful for benchmarking.
        self.d_conv, self.chunk = d_conv, chunk
        self.collect = False        # diagnostics off; see collect_stats()
        self.stats = {}
        self.in_proj = nn.Linear(n_embd, 2 * n_embd)     # value path and gate
        self.a_proj = nn.Linear(n_embd, n_embd)          # per-channel, per-token decay
        self.conv = nn.Conv1d(n_embd, n_embd, d_conv, groups=n_embd, bias=True)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)
        # start with slow forgetting: softplus(1.0) ~ 1.31 -> a ~ 0.27 is too leaky, so bias
        # the decay logits negative, giving a ~ exp(-softplus(-2)) ~ 0.88 at initialisation.
        nn.init.constant_(self.a_proj.bias, -2.0)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        if attn_mask is not None:                        # padding must not enter the state
            x = x * attn_mask.unsqueeze(-1).to(x.dtype)
        v, g = self.in_proj(x).split(C, dim=2)
        # depthwise CAUSAL convolution: pad on the left only, so step t never sees t+1
        v = self.conv(F.pad(v.transpose(1, 2), (self.d_conv - 1, 0))).transpose(1, 2)
        v = F.silu(v)
        a = torch.exp(-F.softplus(self.a_proj(x)))       # (B,T,C) in (0,1)
        h = blocked_scan(a, (1.0 - a) * v, self.chunk)
        y = self.drop(self.out_proj(h * F.silu(g)))
        if self.collect:
            self._measure(x, a, h, y)
        return y

    @torch.no_grad()
    def _measure(self, x, a, h, y):
        """Fill self.stats from one forward pass. Detached and float32 throughout, so nothing
        here touches the graph or the gradient."""
        af = a.detach().float()
        # tau = -1/ln(a), the 1/e memory horizon in tokens; a is clamped off 1 so that a
        # channel that never forgets reports a large horizon rather than an infinity
        tau = -1.0 / torch.log(af.clamp(1e-6, 1 - 1e-7))
        self.stats = {
            "a_mean": af.mean().item(),
            "horizon_med": tau.median().item(),
            "horizon_p95": tau.flatten().quantile(0.95).item(),
            "frac_long": (af > 0.99).float().mean().item(),
            "frac_local": (af < 0.5).float().mean().item(),
            "selectivity": af.std(dim=1).mean().item(),
            "write_ratio": (y.detach().float().norm() /
                            x.detach().float().norm().clamp(min=1e-6)).item(),
            "state_norm": h.detach().float().abs().mean().item(),
        }
