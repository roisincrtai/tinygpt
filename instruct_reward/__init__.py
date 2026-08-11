"""Stage 7: the RLHF reward model (ZetaGPT trunk + scalar head, sigmoid + BCE)."""
from .rlhf_reward import STAGE, NAME, RewardModel, evaluate, load, run
