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


def build_corpus(problems, cfg, log=print, task=None):
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
        target = verifier.demonstration(p.get("trace", ""))
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
    preview = common.make_preview(
        ctx["tok"], ctx["device"], args.max_len, log,
        [d["prompt"] for d in corpus[:common.N_PREVIEW]])
    model = lm.train(init_model, ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.cot_sft_steps,
                     lr=args.cot_sft_lr, preview=preview)
    preview(model, STAGE, "final")
    return model
