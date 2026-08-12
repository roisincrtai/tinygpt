"""
cot/run.py -- stage 9 entry point: chain of thought, supervised first and then by GRPO.

    python -m cot.run [--cot_sft_steps N --cot_steps N --cot_lr LR --cot_group 8]
    ./stage9_cot_aha_moment.sh
    ./stage9_cot_aha_moment.sh --no-cot_sft        # GRPO alone: the R1-Zero setting

THE STAGE IS TWO RUNS OVER ONE PROBLEM BANK, and this module owns the order between them:

    1. SFT   the dataset's reference traces, rewritten into <think>/<answer> form and trained
             by maximum likelihood            -> checkpoint_<run>_cot-sft.pt
    2. GRPO  reinforcement learning FROM THAT CHECKPOINT against the verifier
             -> checkpoint_<run>_cot-grpo.pt

WHY THE SUPERVISED RUN COMES FIRST. GRPO's advantage is a reward minus its group's mean, so a
reward every completion earns equally teaches nothing. A base model has never produced a
<think> tag, so before anything demonstrates the format the format terms are constant across
the group, the correctness term is constant at zero, and the run trains on nothing while every
curve looks healthy. The SFT is what makes the reward have something to grip.

THIS IS R1, NOT R1-ZERO. DeepSeek-R1-Zero is RL directly on a base model with no supervised
reasoning anywhere, which is a claim about what RL alone can do; R1 itself uses a small
supervised cold start for exactly the reason above. `--no-cot_sft` restores the zero setting,
and then --cot_init decides what GRPO starts from.

THE TRACE IS THE SFT'S AND NOBODY ELSE'S. grpo.py is handed the question and the answer; it
never receives the reference reasoning, which is what keeps the reinforcement half honest.

The algorithm is in grpo.py, the supervised half in cot_sft.py and the reward in verifier.py;
no other stage package is imported.
"""
import glob
import json
import os

from helpers import common
import default_config as config
import helpers

from . import cot_sft, grpo, verifier

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

    THE TRACE IS READ BUT KEPT SEPARATE. Every record carries a reference reasoning trace, and
    the two halves of this stage want different things from it: the SFT trains on it, GRPO must
    never see it. So it is loaded once, under its own key, and WHICH CALLER USES IT is what
    separates supervised reasoning from reinforcement -- grpo.py is handed the question and the
    answer, never the trace. The answer is kept in whatever shape the task needs -- a number for
    gsm8k, {target, numbers} for countdown -- because the verifier, not the loader, decides
    what being right means.

    The CoT stage reads ITS OWN directory and nothing else: never the instruction data, never
    the pretraining corpus."""
    cfg = cfg or config.COT
    d = task_dir(cfg)
    qf, af = cfg.get("question_field", "question"), cfg.get("answer_field", "answer")
    tf = cfg.get("trace_field", "response")
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
                out.append({"question": q,
                            "answer": a if isinstance(a, dict) else str(a),
                            "trace": r.get(tf) or ""})
    if not split_files:                       # no split of its own: cut the tail off
        n_val = max(1, int(len(out) * float(cfg.get("val_frac", 0.02))))
        out = out[:-n_val] if split == cfg.get("train_prefix", "train") else out[-n_val:]
    limit = cfg.get("limit", 0)
    if limit:
        out = out[:limit]
    return out, len(files)


def run(ctx):
    """The whole of stage 9: the reasoning SFT, then GRPO from its checkpoint."""
    args, log = ctx["args"], ctx["log"]
    cfg = config.COT

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
        # every term of the reward, by name -- the figure plots them separately, so a table
        # that collapsed them into one number could not be checked against it
        ("reward", f"correct {cfg['correct_reward']} "
                   f"+ <think> {cfg['think_format_reward']} "
                   f"+ <answer> {cfg['answer_format_reward']} "
                   f"+ grounded {cfg['think_reward']}"),
        ("supervised first", f"yes, {args.cot_sft_steps:,} steps at lr {args.cot_sft_lr:g}"
                             if args.cot_sft else "no (--no-cot_sft: the R1-Zero setting)"),
    ], out=log)

    # THE PLAN, PRINTED BEFORE ANY OF IT RUNS. Stage 9 is two training runs, and which of them
    # a log is showing has to be readable at a glance -- otherwise a run that skipped the first
    # looks exactly like a run that did it, until the checkpoints are inspected.
    log("")
    log("=" * 78)
    log(f"STAGE 9, SUB-STAGE 1 of 2: CoT SFT   -> {helpers.ckpt_path(ctx['ckdir'], 'cot_sft')}"
        if args.cot_sft else
        "STAGE 9, SUB-STAGE 1 of 2: CoT SFT   -> SKIPPED (--no-cot_sft, the R1-Zero setting)")
    log(f"STAGE 9, SUB-STAGE 2 of 2: CoT GRPO  -> {helpers.ckpt_path(ctx['ckdir'], STAGE)}")
    log("=" * 78)
    log("")

    # --- 1. the supervised half: the format, from the dataset's own traces --------------- #
    # GRPO STARTS FROM WHATEVER THIS LEAVES BEHIND. When the SFT is on, that is its checkpoint
    # and not args.cot_init: a policy initialised from the base model would have none of the
    # format the SFT just paid for.
    init = args.cot_init
    if args.cot_sft:
        log(f"--- sub-stage 1/2: CoT SFT ({args.cot_sft_steps:,} steps, lr "
            f"{args.cot_sft_lr:g}, from {init}) ---")
        demos = cot_sft.build_corpus(train, cfg, log, task=cfg.get("task", "countdown"),
                                     tok=ctx["tok"], max_len=args.max_len)
        if not demos:
            raise SystemExit(
                "[cot] no usable demonstrations: every reference trace was dropped.\n"
                "      The traces need a <think>...</think> span and an extractable answer;\n"
                "      set COT.trace_field to the field that holds them, or --no-cot_sft to\n"
                "      run GRPO alone.")
        cot_sft.run(ctx, common.load_stage_model(ctx, init, train_mode=True), demos)
        init = cot_sft.STAGE

    # --- 2. the reinforcement half: the answer, against the verifier --------------------- #
    log(f"--- sub-stage 2/2: CoT GRPO ({args.cot_steps:,} steps, lr {args.cot_lr:g}, "
        f"from {init}) ---")
    log(f"=== CoT by GRPO (policy from {init}, reward = verified answer + format, "
        f"KL to frozen {init}) ===")
    base = common.load_stage_model(ctx, init)
    ref = common.frozen(base)                       # the KL anchor: the policy's own origin
    policy = common.frozen(base)
    policy.requires_grad_(True)
    policy.train()

    # fill_window: untruncated, to the end of the context. This stage is looking for a policy
    # that thinks for LONGER, and a fixed cap would cut every sample at the same place -- which
    # hides the very quantity the run is watching.
    preview = common.make_preview(
        ctx["tok"], ctx["device"], args.max_len, log,
        [verifier.prompt(p["question"], cfg) for p in train[:common.N_PREVIEW]],
        fill_window=True)
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
