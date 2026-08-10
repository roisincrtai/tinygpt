"""Stage 7: chain-of-thought reasoning by GRPO against a verified answer (the aha moment)."""
from .run import STAGE, load_problems, run
from . import grpo, verifier
