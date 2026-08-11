"""
pretrain.py -- stage 5's logic: language-model pretraining.

Owns everything specific to pretraining and nothing else: which corpus is read, how it is
reported, which prompts the generation previews use, and which budget and learning rate the
shared LM loop (lm.train) is run with. run.py is only the command-line wrapper.

    load_corpus(tok, corpus_dir, log)  -> (docs, n_files, n_tokens), pre-tokenised and cached
    run(ctx, corpus)        -> the pretrained model

State space diagnostics (memory horizon, selectivity, residual write ratio) are recorded every
--ssm_stats_every steps and drawn in the second row of outputs/plots/pretrain/.

The corpus is every file with a CORPUS_EXTENSIONS extension under the pretraining
corpus (prose, markdown, source code, parquet), scanned recursively minus
config.PRETRAIN["exclude_dirs"], packed into ~max_words-word documents that never cross a
file boundary. Nothing outside the pretraining corpus is read.
"""
import os

import default_config as config
import helpers
from helpers import lm

STAGE = "pretrain"


def load_corpus(tok, corpus_dir, log=print, ctx_batch=0, ctx_len=0):
    """Pre-tokenised pretraining documents (cached under cache/tokens/bpe_<256+merges>_<fp>/), reported
    as a table: the file count and the total token count the scan must announce.

    `corpus_dir` is passed in rather than read from the configuration, because it depends on
    the scheme being trained: tiny and -s ship with one, -m and -l do not. An empty path is
    refused outright instead of scanned -- a scan of nothing reports "0 files" and reads like
    an empty corpus rather than like a missing setting."""
    if not corpus_dir:
        raise SystemExit(
            "[pretrain] no pretraining corpus configured for this scheme.\n"
            "           zetagpt-tiny and zetagpt-s use the downloaded FineWeb-Edu subset; run\n"
            "               python -m tools.download_data\n"
            "           zetagpt-m and zetagpt-l ship with none: set PRETRAIN_DIR in config.sh\n"
            "           (or --pretrain_dir) to your own corpus.")
    if not os.path.isdir(corpus_dir):
        raise SystemExit(
            f"[pretrain] corpus directory does not exist: {corpus_dir}\n"
            f"           No stage downloads. Run:  python -m tools.download_data")
    corpus, n_files, n_tok = helpers.load_token_corpus(
        STAGE, corpus_dir, tok, config.PRETRAIN["max_words"],
        exclude_dirs=config.PRETRAIN["exclude_dirs"], log=log,
        text_column=config.PRETRAIN["text_column"])
    if corpus:
        # WHAT WAS LOADED, stated plainly and in full. A corpus is the one input to a run that
        # the checkpoint does not record, so if the log does not say which directory was read
        # and how many tokens came out of it, nothing later can establish what the model was
        # trained on. The epoch figure is derived rather than asserted: it is what the step
        # budget actually buys over THIS corpus, which is how a mismatch between the two gets
        # noticed at the start of a run rather than at the end.
        seen = config.PRETRAIN["steps"] * ctx_batch * ctx_len if ctx_batch else 0
        rows = [
            ("dataset", os.path.basename(corpus_dir.rstrip("/")) or corpus_dir),
            ("directory", corpus_dir),
            ("excluded subdirs", ", ".join(config.PRETRAIN["exclude_dirs"]) or "(none)"),
            ("token stream", corpus.path),
            ("stream size", f"{corpus.nbytes / 1048576:.1f} MB {corpus.dtype}, "
                            f"memory-mapped (not loaded)"),
            ("corpus files", helpers.count(n_files)),
            ("documents", helpers.count(len(corpus))),
            ("words per document", f"~{config.PRETRAIN['max_words']}"),
            ("TOTAL TOKENS", helpers.count(n_tok)),
            ("mean tokens/document", f"{n_tok / max(len(corpus), 1):.1f}")]
        if seen:
            rows += [("tokens per step", helpers.count(ctx_batch * ctx_len)),
                     ("budget", f"{config.PRETRAIN['steps']:,} steps = "
                                f"{helpers.human(seen)} tokens"),
                     ("epochs over this corpus", f"{seen / max(n_tok, 1):.2f}")]
        helpers.table("pretrain corpus scan", rows, out=log)
    return corpus, n_files, n_tok


def run(ctx, corpus):
    """LM-pretrain a fresh zetagpt on `corpus`, preview, and return it. The periodic
    generation previews are prompted from the corpus itself."""
    from helpers import common
    args, log = ctx["args"], ctx["log"]
    log(f"=== PRETRAIN (LM on {len(corpus):,} documents) ===")
    preview = common.make_preview(ctx["tok"], ctx["device"], args.max_len, log,
                                  common.corpus_prompts(corpus, seed=args.seed,
                                                        tok=ctx["tok"]))
    # the state space module's diagnostics are collected during PRETRAINING specifically:
    # this is the stage long enough for the memory horizon to move, and the one where a
    # degenerate module (selectivity collapsing, or the residual stream routing around it)
    # would otherwise go unnoticed until much later
    model = lm.train(ctx["new_model"](), ctx["enc"], corpus, ctx["ckdir"], args, log,
                     ctx["monitor"], stage=STAGE, steps=args.pretrain_steps,
                     lr=args.pretrain_lr, preview=preview,
                     ssm_stats_every=args.ssm_stats_every)
    preview(model, STAGE, "final")
    return model
