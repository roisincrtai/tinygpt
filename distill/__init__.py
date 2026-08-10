"""Stage 8: sequence-level distillation of the aligned model into gpt2-small."""
from .distill import STAGE, load_prompts, build_student, train, run_stage as run
