"""
instruct_reward/run.py -- stage 5 entry point: the RLHF reward model.

A ZetaGPT trunk initialised from the SFT checkpoint plus a scalar head, trained as a binary
classifier (sigmoid + BCE) on the preference pairs: chosen 1, rejected 0.

    python -m instruct_reward.run [--reward_steps N --reward_lr LR]
    ./stage5_train_rlhf_reward.sh
"""
from helpers import common
import default_config as config
import helpers
from helpers import pref_dataset as pdset

from . import rlhf_reward


def load_pairs(ctx):
    """The preference pairs this stage trains on, per config.REWARD["sources"].

    "hh"        the helpful/harmless pairs of <instruct_dir>/rlhf_hh -- the same records
                stage 4 fine-tuned on and stage 6 rolls out over, which is what makes this
                model's scores meaningful where the policy is actually evaluated."""
    cfg, log = config.REWARD, ctx["log"]
    train, val = [], []
    for src in cfg.get("sources", ["hh"]):
        if src == "hh":
            tr = pdset.load_hh_pairs(cfg["hh_subsets"], limit=cfg["hh_limit"],
                                     seed=ctx["args"].seed)
            ev = pdset.load_hh_pairs(cfg["hh_val_subsets"],
                                     limit=max(cfg["hh_limit"] // 20, 200),
                                     seed=ctx["args"].seed)
            log(f"reward data: rlhf_hh -> {len(tr):,} train / {len(ev):,} val pairs")
            train += tr; val += ev
        else:
            log(f"reward data: {src!r} is not a known source; the pipeline reads rlhf_hh")
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
