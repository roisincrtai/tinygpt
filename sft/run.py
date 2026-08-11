"""
sft/run.py -- stage 6 entry point: domain-adaptive fine-tuning on the fine-tuning data.

SFT MUST be based on the pretrained model: without a pretrain checkpoint this exits.

    python -m sft.run [--sft_steps N --sft_lr LR]
    ./stage6_sft.sh
"""
from helpers import common

from . import sft


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)      # the fine-tuning corpus and nothing else
    corpus, n_files, n_tok = sft.load_corpus(ctx["tok"], args.sft_dir, ctx["log"])
    if not corpus:
        ctx["log"](f"no corpus files under {args.sft_dir}: nothing to fine-tune on")
        return
    init = common.load_stage_model(ctx, "pretrain", train_mode=True)
    ctx["log"]("SFT init <- pretrain checkpoint")
    sft.run(ctx, init, corpus)


if __name__ == "__main__":
    main()
