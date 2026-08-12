"""Stage 9: chain of thought -- a supervised fine-tune on the reference traces, then GRPO
against a verified answer (the aha moment).

`run` is deliberately not imported here; see tokenizer/__init__.py.
"""
from . import cot_sft, grpo, verifier
