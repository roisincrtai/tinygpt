"""
instruct_dpo/run.py -- stage 9 entry point: Direct Preference Optimization.

The policy starts from the SFT model and is trained against the frozen SFT reference.

    python -m instruct_dpo.run [--dpo_steps N --dpo_lr LR --beta B]
    ./stage9_instruct_dpo.sh
"""
import copy

from helpers import common

from . import dpo


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=True, pretokenize_pairs=True)
    log = ctx["log"]
    log("=== DPO (policy init = SFT, reference = frozen SFT) ===")
    sft_model = common.load_stage_model(ctx, "sft")
    ref = common.frozen(sft_model)
    policy = copy.deepcopy(sft_model)
    policy.train()
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  ctx["pref_prompts"])
    model = dpo.run(policy, ref, ctx["enc"], ctx["train_pairs"], ctx["ev_pairs"],
                    ctx["tok"], ctx["ckdir"], args, log, ctx["monitor"], preview=preview)
    preview(model, dpo.STAGE, "final")


if __name__ == "__main__":
    main()
