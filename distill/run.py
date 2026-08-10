"""
distill/run.py -- stage 9 entry point: distillation of the aligned model into gpt2-small.

    python -m distill.run [--distill_steps N --distill_lr LR]
    ./stage9_distill.sh
"""
from helpers import common

from . import distill


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)      # the prompt bank and nothing else
    distill.run_stage(ctx)


if __name__ == "__main__":
    main()
