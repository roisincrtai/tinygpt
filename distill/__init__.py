"""Sequence-level distillation of the aligned model into gpt2-small.

NOT a numbered stage: an optional extra, run after the pipeline and on its own,
because nothing in the pipeline consumes its output.

    python -m distill.run
"""
from .distill import STAGE, load_prompts, build_student, train, run_stage as run
