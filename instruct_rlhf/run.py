"""
instruct_rlhf/run.py -- stage 7 entry point: RLHF instruction tuning by PPO.

    python -m instruct_rlhf.run [--rlhf_steps N --rlhf_lr LR]
    ./stage7_instruct_tuning_rlhf.sh
"""
from helpers import common

from . import rlhf


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=True, pretokenize_pairs=True)
    rlhf.run(ctx)


if __name__ == "__main__":
    main()
