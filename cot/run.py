"""
cot/run.py -- stage 8 entry point: chain-of-thought reasoning by GRPO, the "aha moment" run.

    python -m cot.run [--cot_steps N --cot_lr LR --cot_init pretrain --cot_group 8]
    ./stage8_cot_aha_moment.sh
    COT_INIT=sft ./stage8_cot_aha_moment.sh        # start from a different checkpoint

This module owns the STAGE: which checkpoint the policy starts from, which problems it is
rolled out on, and what the stage is called on disk. The algorithm is in grpo.py and the
reward is in verifier.py; neither knows about the other's concerns, and no other stage package
is imported.

WHY IT STARTS FROM THE PRETRAINED MODEL BY DEFAULT. This is the DeepSeek-R1-Zero setting:
reinforcement learning applied directly to a base model with no supervised reasoning examples
anywhere in the pipeline. Nothing demonstrates how to reason, so any reasoning that emerges is
produced by the optimisation itself -- which is the claim the stage exists to reproduce.
Starting from the SFT or RLHF checkpoint (--cot_init sft, COT_INIT=rlhf) is supported and is
the practical choice, but it weakens the claim, because the demonstrations have then already
shown the model what an answer looks like.

Checkpoint: checkpoints/cot/checkpoint_<model>_<pe>_cot-grpo.pt
Figure:     outputs/plots/cot/dynamics_<model>_<pe>_cot-grpo.pdf
"""
import glob
import json
import os

from helpers import common
import default_config as config
import helpers

from . import grpo, verifier

STAGE = "cot"


def load_problems(log, split, cfg=None):
    """Every {split}_<batch>.json under config.COT["data_dir"], as [{"question", "answer"}].

    The CoT stage reads ITS OWN directory and nothing else -- it never opens the instruction data/ or
    the pretraining corpus. Each record carries question / reasoning / answer; only the
    question and the answer are used, since the stage never trains on reference reasoning."""
    cfg = cfg or config.COT
    d = cfg["data_dir"]
    files = sorted(glob.glob(os.path.join(d, f"{split}_*.json")))
    if not files:
        raise SystemExit(f"[cot] no {split}_*.json under {d}")
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for r in json.load(fh):
                if r.get("question") and r.get("answer") is not None:
                    out.append({"question": r["question"], "answer": str(r["answer"])})
    limit = cfg.get("limit", 0)
    if limit:
        out = out[:limit]
    return out, len(files)


def run(ctx):
    """GRPO-train a copy of the initial checkpoint against the verifier, preview, return."""
    args, log = ctx["args"], ctx["log"]
    cfg = config.COT
    init = args.cot_init
    log(f"=== CoT by GRPO (policy from {init}, reward = verified answer + format, "
        f"KL to frozen {init}) ===")

    train, n_train_files = load_problems(log, cfg["train_prefix"])
    try:
        test, n_test_files = load_problems(log, cfg["test_prefix"])
    except SystemExit:
        test, n_test_files = [], 0
    # dataset statistics get a table; everything else in this stage is single-line logging
    helpers.table("chain-of-thought dataset", [
        ("dataset", os.path.basename(str(cfg["data_dir"]).rstrip("/"))),
        ("directory", cfg["data_dir"]),
        ("train files", helpers.count(n_train_files)),
        ("train problems", helpers.count(len(train))),
        ("test files", helpers.count(n_test_files)),
        ("test problems", helpers.count(len(test))),
        ("prompts per step", f"{args.batch:,}"),
        ("completions per prompt", f"{args.cot_group:,}"),
        ("rollouts per step", f"{args.batch * args.cot_group:,}"),
        ("reward", f"correct {cfg['correct_reward']} + format {cfg['format_reward']}"),
    ], out=log)

    base = common.load_stage_model(ctx, init)
    ref = common.frozen(base)                       # the KL anchor: the policy's own origin
    policy = common.frozen(base)
    policy.requires_grad_(True)
    policy.train()

    preview = common.make_preview(
        ctx["tok"], ctx["device"], args.max_len, log,
        [verifier.prompt(p["question"]) for p in train[:common.N_PREVIEW]])
    model = grpo.run(policy, ref, ctx["tok"], train, ctx["ckdir"], args, log,
                     ctx["monitor"], preview=preview, eval_set=test or None)
    preview(model, STAGE, "final")
    return model


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)      # this stage reads its own corpus only
    run(ctx)


if __name__ == "__main__":
    main()
