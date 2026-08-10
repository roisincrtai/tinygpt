"""
instruct_reward/run.py -- stage 6 entry point: the RLHF reward model.

A ZetaGPT trunk initialised from the SFT checkpoint plus a scalar head, trained as a binary
classifier (sigmoid + BCE) on the preference pairs: chosen 1, rejected 0.

    python -m instruct_reward.run [--reward_steps N --reward_lr LR]
    ./stage6_train_rlhf_reward.sh
"""
from helpers import common
import default_config as config
import helpers
from helpers import dataset_helpers as dsets

from . import rlhf_reward


def load_pairs(ctx):
    """The preference pairs this stage trains on.

    Read from --dataset: "hh" for the downloaded rlhf_hh tree -- the same records stage 5
    fine-tuned on and stage 7 rolls out over, which is what makes this model's scores
    meaningful where the policy is actually evaluated -- or a path to your own folder. The
    LAYOUT is detected: a tree with its own *_train / *_test split keeps that split, a plain
    folder of json/jsonl is shuffled and cut at --val_frac.

    Pairs without a rejected response are refused by load_pairs, since a classifier trained on
    one label learns to output that label."""
    args, log = ctx["args"], ctx["log"]
    root = dsets.resolve_root(getattr(args, "dataset", "hh"), args.data_dir)
    log(f"reward data: {root}")
    log(f"             read as {dsets.describe(root)}")
    train, val, _ = dsets.load_train_val(
        root, config.REWARD["hh_subsets"], config.REWARD["hh_val_subsets"],
        val_frac=args.val_frac, seed=args.seed, limit=config.REWARD["hh_limit"],
        val_limit=max(config.REWARD["hh_limit"] // 20, 200))
    train = [p for p in train if p["rejected"]]
    val = [p for p in val if p["rejected"]]
    if not train:
        raise SystemExit(f"[reward] {root} holds no records with a REJECTED response; this "
                         f"stage learns from a preference, so it needs pairs.")
    log(f"             {len(train):,} train / {len(val):,} val pairs")
    return train, (val or ctx["ev_pairs"])


def main():
    args = common.parse_args()
    # This stage loads its own pairs from rlhf_hh (load_pairs below), so common.setup is not
    # asked for the shared preference set: loading it twice would cost minutes and the copy
    # this stage trains on is the one it selected subsets for.
    ctx = common.setup(args, need_pairs=False)
    log = ctx["log"]
    log("=== REWARD MODEL (ZetaGPT trunk + scalar head, sigmoid + BCE) ===")
    train_pairs, ev_pairs = load_pairs(ctx)
    rm = rlhf_reward.RewardModel(ctx["new_model"]())
    if helpers.load_ckpt(ctx["ckdir"], "sft"):
        sft_model = common.load_stage_model(ctx, "sft")
        rm.base.load_state_dict(sft_model.state_dict())
        del sft_model
        log("reward trunk init <- sft checkpoint")
    else:
        log("no sft checkpoint: reward trunk starts from random init")
    rm = rm.to(ctx["device"])
    rm.train()
    rlhf_reward.run(rm, ctx["enc"], train_pairs, ev_pairs, ctx["ckdir"],
                    args, log, ctx["monitor"])


if __name__ == "__main__":
    main()
