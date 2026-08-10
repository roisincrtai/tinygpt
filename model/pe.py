"""
pe.py -- positional encodings.

    RotaryPositionalEmbedding     -- RoPE (Su et al. 2021), used by zetagpt; rotates the per-head
                                     query/key vectors inside attention rather than adding to the
                                     token embeddings, so there is no position table and the model
                                     extrapolates to any length.
    SinusoidalPositionalEncoding  -- fixed sin/cos table (Vaswani et al.)
    LearnedPositionalEncoding     -- trainable per-position embedding (GPT-2 style)

The additive encodings are cache-aware: forward(x, start=k) adds the encodings for absolute
positions [k, k+T). RoPE takes (q, k) and returns them rotated by the same absolute positions.
"""
import math

import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied to per-head query/key tensors of shape (B, n_head, T, head_dim). `head_dim`
    must be even. No learned parameters; positions are encoded by rotation."""
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, start=0):
        T = q.shape[-2]
        t = torch.arange(start, start + T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)              # (T, head_dim)
        cos = emb.cos()[None, None].to(q.dtype)              # (1, 1, T, head_dim)
        sin = emb.sin()[None, None].to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, start=0):
        T = x.size(1)
        return x + self.pe[start:start + T].unsqueeze(0).to(x.dtype)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=8192):
        super().__init__()
        self.emb = nn.Embedding(max_len, n_embd)

    def forward(self, x, start=0):
        T = x.size(1)
        pos = torch.arange(start, start + T, device=x.device)
        return x + self.emb(pos).unsqueeze(0)


def build_positional_encoding(kind, n_embd, max_len):
    kind = (kind or "sinusoidal").lower()
    if kind.startswith("sin"):
        return SinusoidalPositionalEncoding(n_embd, max_len)
    if kind.startswith("learn"):
        return LearnedPositionalEncoding(n_embd, max_len)
    raise ValueError(f"unknown positional encoding '{kind}' (sinusoidal|learned)")
