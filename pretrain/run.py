"""
pretrain/run.py -- stage 5 entry point: language-model pretraining.

    python -m pretrain.run [--pretrain_steps N --pretrain_lr LR]
    ./stage5_pretrain.sh
"""
from helpers import common

from . import pretrain


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)      # the pretraining corpus and nothing else
    corpus, n_files, n_tok = pretrain.load_corpus(ctx["tok"], args.pretrain_dir, ctx["log"],
                                                     args.batch, args.max_len)
    if not corpus:
        ctx["log"](f"no corpus files under {args.pretrain_dir}: nothing to pretrain")
        return
    pretrain.run(ctx, corpus)


if __name__ == "__main__":
    main()
