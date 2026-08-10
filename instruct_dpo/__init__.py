"""Stage 7: Direct Preference Optimization, and the shared sequence-scoring core."""
from .dpo import (STAGE, NAME, seq_logp, rewards, loss, margin, corrupt, extend_alphabet,
                  run)
