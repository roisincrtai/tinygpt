"""
instruct_sft.py -- stage 6's logic: domain-adaptive supervised fine-tuning on the fine-tuning data.

The objective is the SAME length-normalized LM likelihood as pretraining (lm.train); what
changes is the corpus. Stage 5 buys general language from a large heterogeneous corpus,
this stage moves the model onto the distribution the alignment stages operate in. Sharing
one loop is deliberate: the two stages cannot drift apart in their treatment of masking,
normalisation or padding.

    load_corpus(tok, corpus_dir, log)  -> (docs, n_files, n_tokens), pre-tokenised and cached
    run(ctx, init_model, corpus) -> the fine-tuned model

Its output anchors everything downstream: it initialises the reward trunk, it is the
starting policy for both alignment stages, and it is the frozen reference their divergence
terms are measured against. Nothing outside the fine-tuning data is read.
"""
import glob
import os
import random

import default_config as config
import helpers
from helpers import dataset_helpers as dsets
from helpers import lm

STAGE = "sft"


def load_corpus(tok, corpus_dir, log=print):
    """The fine-tuning set: every demonstration under `corpus_dir`, conditioned on its prompt.

    THREE LAYOUTS, ONE RESULT. A CONVERSATION MIXTURE (the Tulu 3 SFT mixture: parquet shards
    of `messages` lists) is flattened to one demonstration per assistant turn. A PREFERENCE
    TREE is read for its chosen continuation and nothing else -- the rejected one is what
    stages 7 and 10 need, and feeding it here would teach the model the very responses the rest
    of the pipeline then spends its time pushing down. A LOCAL FOLDER of json/jsonl records is
    read for whatever prompt and response fields it uses. All three arrive in the same
    {prompt, chosen, rejected} shape the encoder already takes, so the loss lands on the
    response tokens with the prompt as context: the prompt is conditioning, not something to be
    modelled.

    A demonstration whose prompt and response are identical to another's is dropped. rlhf_hh
    repeats prompts across its helpful and harmless subsets, and a mixture assembled from
    several sources repeats whole examples; a duplicate is simply that example weighted twice.

    The layout is detected and REPORTED, not declared -- so pointing this stage at your own
    folder works, and what it decided your folder was is in the log rather than in someone's
    head. Only a prompt and a response are needed here, so a mixture with no rejected field
    anywhere in it is enough for this stage even though stages 7 and 10 would refuse it."""
    layout = dsets.detect_layout(corpus_dir)
    if layout == "missing":
        raise SystemExit(
            f"[sft] fine-tuning data not found: {corpus_dir or '(unset)'}\n"
            f"      Expected a conversation mixture (parquet/jsonl with a `messages` list),\n"
            f"      an hh tree (<dir>/*_train/*.json), or a folder of json/jsonl records\n"
            f"      carrying a prompt and a response.\n"
            f"      No stage downloads. Run:  ./stage1_download_data.sh\n"
            f"      or set SFT_DIR in config.sh (--sft_dir) to your own.")
    if layout == "hh":
        pairs = dsets.load_hh_pairs(subsets=dsets.subsets_for(corpus_dir,
                                                              config.SFT["subsets"]),
                                    hh_dir=corpus_dir, limit=config.SFT["limit"], seed=0)
    elif layout == "chat":
        pairs = dsets.load_chat_pairs(corpus_dir, limit=config.SFT["limit"], seed=0)
    else:
        pairs = dsets.load_local_pairs(corpus_dir, limit=config.SFT["limit"], seed=0)
    seen, corpus = set(), []
    for p in pairs:
        key = (p["prompt"], p["chosen"])
        if key in seen:
            continue
        seen.add(key)
        corpus.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": ""})
    n_files = (len(dsets.chat_files(corpus_dir)) if layout == "chat" else
               len(glob.glob(os.path.join(corpus_dir, "**", "*.json*"), recursive=True)))
    n_tok = 0
    if corpus:
        # measured on a sample: tokenising every demonstration to print one number would cost
        # minutes, and this is a report rather than something the training reads
        smp = corpus if len(corpus) <= 2000 else random.Random(0).sample(corpus, 2000)
        per = sum(len(tok(d["prompt"] + " " + d["chosen"],
                          add_special_tokens=False)["input_ids"]) for d in smp) / len(smp)
        n_tok = int(per * len(corpus))
        helpers.table("sft data", [
            ("dataset", os.path.basename(corpus_dir.rstrip("/")) or corpus_dir),
            ("directory", corpus_dir),
            ("layout", dsets.describe(corpus_dir)),
            ("subsets", ", ".join(dsets.subsets_for(corpus_dir, config.SFT["subsets"]))
                        if layout == "hh" else
                        "(every shard, read recursively)" if layout == "chat" else
                        "(local folder: read recursively)"),
            ("batch files", helpers.count(n_files)),
            ("demonstrations read", helpers.count(len(pairs))),
            ("demonstrations (deduplicated)", helpers.count(len(corpus))),
            ("mean tokens/demonstration", f"{per:.1f}"),
            ("TOTAL TOKENS (estimated)", f"~{helpers.human(n_tok)}"),
            ("trained on", "every ASSISTANT TURN, conditioned on everything said before it"
                           if layout == "chat" else
                           "the CHOSEN response, conditioned on the prompt")], out=log)
    return corpus, n_files, n_tok


def run(ctx, init_model, corpus):
    """LM-fine-tune `init_model` (the pretrained zetagpt) on `corpus`, preview, and return
    it. The previews are prompted from the SFT corpus, so they show the domain being
    adapted to."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log(f"=== SFT (domain-adaptive LM on {len(corpus):,} documents, from pretrain) ===")
    # WHAT AN EPOCH ACTUALLY IS, beside what was asked for. The step budget is a number in
    # config.sh, chosen against an expected corpus size; the corpus is whatever is on disk,
    # after deduplication and after a conversation mixture has been expanded per assistant
    # turn. Those two are allowed to differ -- but not silently, because a budget that is
    # half an epoch and a budget that is three looks identical from a loss curve.
    epoch = -(-len(corpus) // max(args.batch, 1))            # ceil
    mb = int(getattr(args, "micro_batch", 0) or 0)
    log(f"[sft] one epoch = {epoch:,} steps at batch {args.batch} "
        f"({len(corpus):,} demonstrations); this run does {args.sft_steps:,} "
        f"({args.sft_steps / max(epoch, 1):.2f} epochs)")
    # THE MICRO-BATCH, SAID OUT LOUD. lm.train only announces it when there is a context
    # SCHEDULE to announce it alongside, and this stage has one window -- so without this the
    # single number that decides whether the step fits in memory would never appear in the log.
    log(f"[sft] {mb} sequence{'s' if mb != 1 else ''} per forward pass, "
        f"{-(-args.batch // max(mb, 1))} passes per step (a batch is padded to its longest "
        f"record, so this is what bounds the tokens per pass)"
        if mb else
        f"[sft] the whole batch of {args.batch} in one forward pass, padded to its longest "
        f"record -- set SFT_MICRO_BATCH to split it")
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  common.corpus_prompts(corpus, seed=args.seed,
                                                        tok=ctx["tok"]))
    model = lm.train(init_model, ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.sft_steps, lr=args.sft_lr,
                     preview=preview)
    preview(model, STAGE, "final")
    return model
