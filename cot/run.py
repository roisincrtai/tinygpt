"""
cot/run.py -- stage 9 entry point: chain-of-thought reasoning by GRPO, the "aha moment" run.

    python -m cot.run [--cot_steps N --cot_lr LR --cot_init pretrain --cot_group 8]
    ./stage9_cot_aha_moment.sh
    COT_INIT=sft ./stage9_cot_aha_moment.sh        # start from a different checkpoint

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


def task_dir(cfg):
    """The directory this run's task reads. One place decides it, so the loader, the log line
    and the error message cannot name different directories."""
    if cfg.get("task", "countdown") == "gsm8k":
        return cfg.get("data_dir_gsm8k") or cfg["data_dir"]
    return cfg["data_dir"]


def load_problems(log, split, cfg=None):
    """The problem bank for `split`, as [{"question", "answer"}].

    TWO LAYOUTS, ONE LOADER. A dataset that ships its own split names its files
    {train,test}_<batch>.json and those are read directly. One that does not -- the countdown
    shards are just <name>_00-of-21.json -- is read whole and CUT AT THE END: the last
    `val_frac` of the records become the test split and the rest the train split. Cutting at
    the end rather than sampling keeps the split a pure function of the files, so two runs on
    one machine, or the same run resumed, hold out exactly the same problems.

    ONLY THE QUESTION AND THE ANSWER ARE READ. Every record here also carries a reference
    reasoning trace, and the stage never looks at it: that is what makes this R1-Zero-style
    rather than supervised. The answer is kept in whatever shape the task needs -- a number for
    gsm8k, {target, numbers} for countdown -- because the verifier, not the loader, decides
    what being right means.

    The CoT stage reads ITS OWN directory and nothing else: never the instruction data, never
    the pretraining corpus."""
    cfg = cfg or config.COT
    d = task_dir(cfg)
    qf, af = cfg.get("question_field", "question"), cfg.get("answer_field", "answer")
    split_files = sorted(glob.glob(os.path.join(d, f"{split}_*.json")))
    files = split_files or sorted(glob.glob(os.path.join(d, "*.json")))
    if not files:
        raise SystemExit(
            f"[cot] no json under {d}\n"
            f"      Fetch it first:  ./stage1_download_data.sh --only "
            f"{os.path.basename(os.path.normpath(d))}")
    out = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            payload = json.load(fh)
        for r in (payload if isinstance(payload, list) else payload.get("data", [])):
            q = r.get(qf) if r.get(qf) is not None else r.get("question")
            a = r.get(af) if r.get(af) is not None else r.get("answer")
            if q and a is not None:
                out.append({"question": q, "answer": a if isinstance(a, dict) else str(a)})
    if not split_files:                       # no split of its own: cut the tail off
        n_val = max(1, int(len(out) * float(cfg.get("val_frac", 0.02))))
        out = out[:-n_val] if split == cfg.get("train_prefix", "train") else out[-n_val:]
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
