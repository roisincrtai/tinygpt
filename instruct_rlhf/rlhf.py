"""
rlhf.py -- stage 6's logic: RLHF instruction tuning.

Assembles the three models the recipe needs and hands them to the optimiser in ppo.py:

    policy     a copy of the SFT model -- what is trained
    reference  the SFT model, frozen -- what the KL penalty holds the policy near
    reward     the stage-4 reward model, frozen -- what the policy maximises

The algorithm itself (rollout, per-token KL-shaped reward, GAE, clipped surrogate) lives in
ppo.py; this module owns the stage: which checkpoints are loaded, which prompts the previews
use, and what the stage is called on disk. stage6_instruct_tuning_rlhf.sh is only the
command-line wrapper.

    run(ctx) -> the aligned policy
"""
import copy

import default_config as config
from helpers import dataset_helpers as dsets

from . import ppo
import instruct_reward as rlhf_reward

STAGE = "rlhf"


def load_prompts(ctx):
    """The prompt bank PPO rolls out from.

    config.RLHF["prompt_dir"] (<instruct_dir>/alpaca_gpt4 by default)
    holds instruction batches; each record's `prompt` is an instruction with its optional
    input appended. PPO needs nothing else from them -- it learns from the reward model's
    score of its OWN generation, not from the reference output. Falls back to the preference
    file's prompts when no directory is configured."""
    cfg, log = config.RLHF, ctx["log"]
    d = cfg.get("prompt_dir")
    if d:
        prompts, n_files = dsets.load_instruction_prompts(d, cfg.get("prompt_limit", 0))
        if prompts:
            log(f"rlhf prompts: {len(prompts):,} unique instructions "
                f"from {n_files} files in {d}")
            return [{"prompt": p} for p in prompts]
        log(f"rlhf prompts: nothing under {d}; falling back to the preference prompts")
    return ctx["train_pairs"]


def run(ctx):
    """PPO-train a copy of the SFT model against the frozen reward model, preview, return."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log("=== RLHF (PPO from SFT, reward = stage-4 reward model, KL to frozen SFT) ===")
    sft_model = common.load_stage_model(ctx, "sft")
    ref = common.frozen(sft_model)                    # the KL anchor
    rm = rlhf_reward.load(ctx)                        # frozen reward model
    rm.requires_grad_(False)
    policy = copy.deepcopy(sft_model)
    policy.train()
    prompts = load_prompts(ctx)
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  [p["prompt"] for p in prompts[:common.N_PREVIEW]])
    model = ppo.run(policy, ref, rm, ctx["tok"], prompts, ctx["ckdir"],
                    args, log, ctx["monitor"], preview=preview)
    preview(model, STAGE, "final")
    return model
