"""
sft.py -- stage 5's logic: domain-adaptive supervised fine-tuning on the fine-tuning data.

The objective is the SAME length-normalized LM likelihood as pretraining (lm.train); what
changes is the corpus. Stage 4 buys general language from a large heterogeneous corpus,
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
    """The fine-tuning set: the PREFERRED response of every preference pair, conditioned on its
    prompt.

    Supervised fine-tuning on preference data means training on the chosen continuation and
    nothing else -- the rejected one is what stages 6 and 9 need, and feeding it here would
    teach the model the very responses the rest of the pipeline then spends its time pushing
    down. The records are handed back in the same {prompt, chosen, rejected} shape the encoder
    already takes, so the loss lands on the response tokens with the prompt as context: the
    prompt is conditioning, not something to be modelled.

    A pair whose prompt and chosen response are identical to another's is dropped. rlhf_hh
    repeats prompts across its helpful and harmless subsets, and a duplicated demonstration is
    simply that example weighted twice.

    `corpus_dir` may be the downloaded hh tree or YOUR OWN FOLDER of json/jsonl records; the
    layout is detected and reported, not declared. Only a prompt and a preferred response are
    needed here, so a folder of plain demonstrations (no rejected field) is enough for this
    stage even though stages 6 and 9 would refuse it."""
    layout = dsets.detect_layout(corpus_dir)
    if layout == "missing":
        raise SystemExit(
            f"[sft] fine-tuning data not found: {corpus_dir or '(unset)'}\n"
            f"      Expected an hh tree (<dir>/*_train/*.json) or a folder of json/jsonl\n"
            f"      records carrying a prompt and a response.\n"
            f"      No stage downloads. Run:  ./stage1_download_data.sh\n"
            f"      or set SFT_DIR in config.sh (--sft_dir) to your own.")
    if layout == "hh":
        pairs = dsets.load_hh_pairs(subsets=dsets.subsets_for(corpus_dir,
                                                              config.SFT["subsets"]),
                                    hh_dir=corpus_dir, limit=config.SFT["limit"], seed=0)
    else:
        pairs = dsets.load_local_pairs(corpus_dir, limit=config.SFT["limit"], seed=0)
    seen, corpus = set(), []
    for p in pairs:
        key = (p["prompt"], p["chosen"])
        if key in seen:
            continue
        seen.add(key)
        corpus.append({"prompt": p["prompt"], "chosen": p["chosen"], "rejected": ""})
    n_files = len(glob.glob(os.path.join(corpus_dir, "**", "*.json*"), recursive=True))
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
                        if layout == "hh" else "(local folder: read recursively)"),
            ("batch files", helpers.count(n_files)),
            ("pairs read", helpers.count(len(pairs))),
            ("demonstrations (deduplicated)", helpers.count(len(corpus))),
            ("mean tokens/demonstration", f"{per:.1f}"),
            ("TOTAL TOKENS (estimated)", f"~{helpers.human(n_tok)}"),
            ("trained on", "the CHOSEN response, conditioned on the prompt")], out=log)
    return corpus, n_files, n_tok


def run(ctx, init_model, corpus):
    """LM-fine-tune `init_model` (the pretrained zetagpt) on `corpus`, preview, and return
    it. The previews are prompted from the SFT corpus, so they show the domain being
    adapted to."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log(f"=== SFT (domain-adaptive LM on {len(corpus):,} documents, from pretrain) ===")
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  common.corpus_prompts(corpus, seed=args.seed,
                                                        tok=ctx["tok"]))
    model = lm.train(init_model, ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.sft_steps, lr=args.sft_lr,
                     preview=preview)
    preview(model, STAGE, "final")
    return model
