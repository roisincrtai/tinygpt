"""
instruct_sft/run.py -- stage 6 entry point: domain-adaptive fine-tuning on the fine-tuning data.

SFT MUST be based on the pretrained model: without a pretrain checkpoint this exits.

    python -m instruct_sft.run [--sft_steps N --sft_lr LR]
    ./stage6_instruct_sft.sh
"""
import default_config as config
from helpers import common

from . import instruct_sft as sft


def stage_batch(args):
    """This stage's own batch and micro-batch, unless the run asked for others.

    parse_args has already applied SCHEME_BATCH, which is sized for a PRETRAINING step where
    every sequence fills the context window; an instruction demonstration is a few hundred
    tokens, so that batch would leave the card almost empty and make one epoch ten times longer.
    config.sh passes --batch and --micro_batch for stage 6, so the shell path never reaches
    this; it exists so that `python -m instruct_sft.run` -- which the Readme offers as the
    equivalent of the stage script -- really is equivalent, rather than quietly training at a
    third of the batch and a tenth of the epoch.

    An explicit flag or environment variable still wins, tested exactly as parse_args tests
    it, so nothing anybody typed is replaced here either."""
    if not common._was_given("--batch", "batch"):
        args.batch = int(config.SFT["batch"])
    if not common._was_given("--micro_batch", "micro_batch"):
        args.micro_batch = int(config.SFT["micro_batch"])
    return args


def main():
    args = stage_batch(common.parse_args())
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
