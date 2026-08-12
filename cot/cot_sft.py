"""
cot/cot_sft.py -- the supervised half of stage 9: teach the FORMAT, then let GRPO teach the
answer.

    build_corpus(problems, cfg, log)   reference traces -> demonstrations, filtered
    run(ctx, init_model, corpus)       maximum likelihood on them -> the cot-sft checkpoint

WHY THIS EXISTS. GRPO can only amplify behaviour the policy already samples. Its reward pays
for a closed <think> span, a closed <answer> span and a verified answer -- and a base model has
produced none of the three, ever. When no completion in a group scores differently from any
other, every advantage is zero: the run consumes rollouts, plots curves and learns nothing.
That is not a hypothetical failure here, it is the one this pipeline already hit.

So the container is demonstrated first, by maximum likelihood, on the traces the dataset ships.
The demonstration is REWRITTEN into this pipeline's format rather than trained on as written
(verifier.demonstration explains the rewrite), because a supervised target that differs from
what the reward pays for teaches the model to be graded down.

WHAT IT DOES NOT TEACH. Only where the reasoning goes. Whether the reasoning is any good is
what stage 9's second half optimises, against a verifier that cannot be talked round.

    checkpoint:  checkpoints/cot/checkpoint_<model>_<pe>_cot-sft.pt
    figure:      outputs/plots/cot/dynamics_<model>_<pe>_cot-sft.pdf

THE OBJECTIVE IS THE PIPELINE'S ONE LM LOSS (helpers.lm.train), the same loop stages 5 and 6
use: the loss lands on the response tokens with the prompt as context. Sharing it is deliberate
-- three stages cannot drift apart in their treatment of masking, normalisation or padding if
there is only one implementation of them.
"""
import helpers
from helpers import lm

from . import verifier

STAGE = "cot_sft"


def _length_rows(corpus, tok, max_len, sample=500, seed=0):
    """Rows reporting how the demonstrations sit against the context window.

    WHY THIS IS REPORTED AT ALL. An example longer than the window is LEFT-truncated by the
    encoder (dataset_helpers.Encoder), which drops from the FRONT: the answer survives and the
    beginning of the prompt does not. That is the right direction -- a demonstration cut at the
    end would teach the model to open <answer> and never close it -- but a corpus where most
    examples lose their prompt is training on something other than what it looks like, and that
    should be a number on screen rather than a surprise.

    The tokens-per-word ratio is MEASURED on a sample and the count extrapolated from it:
    tokenising twenty thousand traces to print one row costs minutes, and this is a report."""
    import random as _random
    if not corpus or tok is None:
        return []
    words = [len(d["prompt"].split()) + len(d["chosen"].split()) for d in corpus]
    smp = corpus if len(corpus) <= sample else _random.Random(seed).sample(corpus, sample)
    t = w = 0
    for d in smp:
        text = d["prompt"] + " " + d["chosen"]
        t += len(tok(text, add_special_tokens=False)["input_ids"])
        w += max(1, len(text.split()))
    per = t / max(1, w)
    over = sum(1 for n in words if n * per > max_len)
    words.sort()
    return [("mean / p90 / max length (words)",
             f"{sum(words)/len(words):.0f} / {words[int(0.9 * (len(words)-1))]:,} / "
             f"{words[-1]:,}"),
            ("tokens per word (measured on a sample)", f"{per:.2f}"),
            (f"longer than the {max_len:,}-token window (estimated)",
             f"{helpers.count(over)}  ({100 * over / len(words):.1f}%, left-truncated: the "
             f"answer is kept, the start of the prompt is not)")]


def build_corpus(problems, cfg, log=print, task=None, tok=None, max_len=0):
    """The problems' reference traces as {prompt, chosen, rejected} demonstrations.

    Each record becomes the SAME prompt GRPO will roll out on -- verifier.prompt(), system
    instruction and all -- paired with the dataset's trace rewritten into <think>/<answer>
    form. Using the rollout prompt rather than a plainer one is the point: the model is being
    shown what to produce after exactly the text it will later be asked to continue.

    TWO FILTERS, and both drop rather than repair.

      no demonstration   the trace has no think span or no extractable answer, so it cannot be
                         rewritten into the format without inventing the missing half
      verifier rejects   the trace's own final answer does not solve the problem

    The second matters more than it looks. A trace that reasons fluently to a wrong equation is
    a demonstration of confident, well-formatted error -- and this stage's whole advantage is
    that correctness here is DECIDABLE, so there is no reason to train on an answer that was
    never checked. Both counts are reported: a filter that silently removes most of a corpus
    should be visible as a number, not inferred from a training curve.

    `rejected` is empty because nothing downstream of this reads it -- the field exists so the
    records have the shape helpers.dataset_helpers.Encoder already takes."""
    task = task or cfg.get("task", "countdown")
    verify = bool(cfg.get("sft_verify", True))
    corpus, no_demo, wrong = [], 0, 0
    for p in problems:
        # THE ANSWER IS OFFERED AS A FALLBACK. A countdown trace states its own answer in a
        # \boxed{} and p["answer"] is a {target, numbers} dict that is not one; gsm8k's prose
        # states no answer at all and p["answer"] is the number. demonstration() prefers what
        # the trace says and falls back to this, so one call serves both.
        target = verifier.demonstration(p.get("trace", ""), p.get("answer"))
        if not target:
            no_demo += 1
            continue
        if verify and not verifier.is_correct(target, p["answer"], task):
            wrong += 1
            continue
        corpus.append({"prompt": verifier.prompt(p["question"], cfg),
                       "chosen": target,
                       "rejected": ""})
    n_think = sum(len(verifier.think_text(d["chosen"]).split()) for d in corpus)
    helpers.table("chain-of-thought demonstrations", [
        ("problems read", helpers.count(len(problems))),
        ("no think span or no answer (dropped)", helpers.count(no_demo)),
        ("answer rejected by the verifier (dropped)",
         helpers.count(wrong) if verify else "(not checked: sft_verify=False)"),
        ("demonstrations", helpers.count(len(corpus))),
        ("mean think length (words)", f"{n_think / max(1, len(corpus)):.1f}"),
        *(_length_rows(corpus, tok, max_len) if max_len else []),
        ("trained on", "the rewritten trace, conditioned on the rollout prompt"),
    ], out=log)
    return corpus


def run(ctx, init_model, corpus):
    """Fine-tune `init_model` on the demonstrations and return it, in eval mode.

    The previews are prompted from this corpus, so what they show is the format being learned
    -- which is the one thing worth watching here, and the thing GRPO cannot report until it
    is already present."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log(f"=== CoT SFT (reasoning format from {len(corpus):,} demonstrations, "
        f"from {args.cot_init}) ===")
    # fill_window: the samples run to the end of the context window and are printed whole.
    # What this sub-stage teaches is a LONG structured completion, and a preview cut at a few
    # hundred tokens would stop before the </think><answer> it exists to demonstrate.
    #
    # FIVE OF THEM, not twenty, for the same reason: each is thousands of tokens rather than
    # sixty, so twenty would be minutes of generation between training steps and more text than
    # anyone reads. The cadence is unchanged (--print_samples_every_steps).
    preview = common.make_preview(
        ctx["tok"], ctx["device"], args.max_len, log,
        [d["prompt"] for d in corpus[:common.N_PREVIEW_LONG]], n=common.N_PREVIEW_LONG,
        fill_window=True)
    model = lm.train(init_model, ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.cot_sft_steps,
                     lr=args.cot_sft_lr, preview=preview)
    preview(model, STAGE, "final")
    return model
