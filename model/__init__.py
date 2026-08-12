"""The model: ZetaGPT and the causal state space module that gives it position.

    from model import ZetaGPT          the network every stage trains
    from model.ssm import CausalSSM     the module that supplies position

`pe` selects how position enters: "ssm" (default) uses the state space module and no encoding
at all; "rope" removes the module and rotates queries and keys inside attention instead. The
second exists only as the ablation control for measuring what the recurrence contributes.
"""
from .zetagpt import ZetaGPT, Block, CausalSelfAttention
from .ssm import CausalSSM, blocked_scan
from .pe import RotaryPositionalEmbedding

from . import parallel            # noqa: F401  layer sharding across GPUs
