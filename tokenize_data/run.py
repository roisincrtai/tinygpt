"""
tokenize_data/run.py -- stage 4: turn this run's corpus into a token stream.

    python -m tokenize_data.run [--force] [--list] [--all] [--only tiny]
    ./stage4_tokenize_data.sh

WHY THIS IS ITS OWN STAGE. Tokenising a corpus is the one expensive, entirely deterministic
step in the pipeline: it depends on the corpus and the tokenizer and on nothing else, it
takes hours on a 2B-token corpus, and its result is reused by every run that follows. Folding
it into the start of pretraining made those hours look like part of training -- a run that
appeared to hang before its first step, a cost paid again on every machine, and a failure
(an unreadable file, a wrong text column, a full disk) discovered only after a GPU had been
allocated. As a stage of its own it is visible, resumable, runnable on a machine with no GPU
at all, and shared: the same streams serve pretraining, evaluation and the scaling-law study.

This is the standard shape of a large-scale pipeline -- GPT-2's encoder pass, nanoGPT's
prepare.py, Megatron's preprocess_data.py all exist for the same reason.

WHAT IT TOKENISES. The corpora THIS RUN will train on, and only those: PRETRAIN_DIR from
config.sh (a comma-separated list), or the list belonging to MODEL_SCHEME. A pipeline
configured for the 61M model has no reason to spend hours on the ~10 GB corpus only the 97M
model reads. --all tokenises every configured corpus, for the case where several sizes are
prepared at once.

ONE STREAM PER CORPUS, AND ADDING ONE COSTS ONLY THAT ONE. A scheme's corpora are never
merged: each is tokenised alone, under its own signature, into its own cache directory. So
the way to add a corpus to a model that is already training is simply to add it to the list
and re-run this stage -- every existing stream is found by its unchanged signature and left
exactly as it is, the new one is built beside it, and the run reading them neither restarts
nor re-tokenises. The corpora are combined only in the batch sampler, in proportion to their
token counts.

WHAT IT WRITES. ~100 MB .tokens shards per (corpus, tokenizer, packing), under
cache/tokens/bpe_<encoding>_<fingerprint>/. See helpers/token_store.py. Re-running is cheap: a
stream whose signature still matches its corpus is left alone, an interrupted one resumes from
its last shard, and --force starts again.

Training stages do NOT depend on this having been run. They build a missing stream themselves,
so skipping it costs time rather than correctness -- the same rule stage 1 follows for
downloads, and for the same reason: a pipeline that breaks when a step is skipped teaches
people to run every step blindly.
"""
import argparse
import os
import sys

import default_config as config
import helpers


def corpora(args):
    """(name, directory) for the corpus THIS CONFIGURATION will train on -- normally one.

    By default that is what config.sh says: PRETRAIN_DIR if it is set, otherwise the corpus
    belonging to MODEL_SCHEME. Tokenising every scheme's corpus instead would mean a run
    configured for the 61M model spending hours on the ~10 GB FineWeb subset that only the
    97M model reads -- work nobody asked for, on a machine that may not even have the disk.

    --only <name> picks a different scheme by name, and --all does the lot, for the case
    where several sizes are to be trained from one prepared cache.

    A scheme owns a LIST of corpora, and each one is tokenised into its OWN stream under its
    own signature. Nothing is merged and nothing is re-read: adding a corpus to a scheme that
    has already been prepared costs exactly the new corpus, because every other stream is
    found in the cache by its unchanged signature and reused. That is what makes it safe to
    extend a run that is already training -- the shards it is reading are never rewritten."""
    def expand(scheme, roots, out, seen):
        for path in (roots if isinstance(roots, (list, tuple)) else [roots]):
            if not path:
                continue                   # a scheme with no corpus configured at all
            key = os.path.abspath(path)
            if key in seen:                # schemes share corpora; tokenise each one once
                continue
            seen.add(key)
            # The label carries the corpus name as well as the scheme, because a scheme now
            # contributes several rows and "zetagpt-s TOKENS" three times over says nothing.
            name = scheme if roots is path else f"{scheme}/{os.path.basename(key)}"
            out.append((name, path))

    if args.all or args.only:
        out, seen = [], set()
        for scheme, roots in config.PRETRAIN_CORPUS.items():
            if args.only and args.only not in scheme:
                continue
            expand(scheme, roots, out, seen)
        return out
    scheme = args.model_scheme or config.PRETRAIN["model_scheme"]
    roots = ([d.strip() for d in str(args.pretrain_dir).split(",") if d.strip()]
             if args.pretrain_dir else config.PRETRAIN_CORPUS.get(scheme, []))
    out, seen = [], set()
    expand(scheme, roots, out, seen)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="tokenise the corpora into token streams")
    ap.add_argument("--model_scheme", default="",
                    help="which scheme's corpus to tokenise; empty = the configured one")
    ap.add_argument("--pretrain_dir", default="",
                    help="the corpus directories themselves, comma-separated, overriding the "
                         "scheme's own list")
    ap.add_argument("--only", default="", help="substring of a scheme name, e.g. tiny")
    ap.add_argument("--all", action="store_true",
                    help="every configured corpus, not just this run's")
    ap.add_argument("--force", action="store_true",
                    help="discard the existing shards and tokenise from the start")
    # --no-resume IS NOT --force HERE, though it was, and that was a trap. This stage receives
    # COMMON_FLAGS (see stage4_tokenize_data.sh), and COMMON_FLAGS carries the trainers'
    # --no-resume, which means "ignore the checkpoints and train from step 0". Treating it as
    # --force meant that starting a model afresh silently deleted and re-tokenised every
    # corpus -- hours of work, and the very shards a concurrent run was reading. Resuming is
    # this stage's only behaviour; rebuilding is --force, typed deliberately, and nothing
    # forwards it.
    ap.add_argument("--no-resume", dest="no_resume", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--list", action="store_true", help="show what would be built, and where")
    # THE PACKING DEFAULTS ARE RESOLVED AFTER --set, NOT BEFORE IT. An argparse default is
    # evaluated when the parser is built, which is before apply_overrides has run, so
    # `default=config.PRETRAIN["max_words"]` would freeze the value the override was about to
    # change and this stage would pack the corpus differently from every stage that reads
    # config.PRETRAIN directly. A different packing is a different stream, and a different
    # stream is the whole corpus tokenised again. The sentinels defer the question.
    ap.add_argument("--max_words", type=int, default=0,
                    help="words per packed document; 0 = PRETRAIN['max_words'] after --set")
    ap.add_argument("--text_column", default="",
                    help="parquet column holding the text; empty = PRETRAIN['text_column']")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="SEC.key=v", help="override any default_config value, e.g. "
                         "--set PRETRAIN.max_words=512")
    a, _ = ap.parse_known_args(argv)
    config.apply_overrides(a.overrides, print)
    a.max_words = a.max_words or config.PRETRAIN["max_words"]
    a.text_column = a.text_column or config.PRETRAIN["text_column"]

    # The tokenizer is READ, never built here. A stream is tied to the vocabulary that
    # produced it -- that is what its cache directory is named after -- so a stage that
    # quietly trained its own BPE would write streams no other stage could use.
    from tokenizer import run as train_bpe
    tok = train_bpe.peek(config.BPE_PATH)
    if tok is None:
        raise SystemExit(
            f"[tokenize] no usable tokenizer at {config.BPE_PATH}\n"
            f"           Run ./stage3_train_bpe_tokenizer.sh first: a token stream is tied "
            f"to the vocabulary that produced it.")

    todo = corpora(a)
    if not todo:
        print(f"[tokenize] no corpus configured for "
              f"{a.model_scheme or config.PRETRAIN['model_scheme']}; nothing to do.\n"
              f"           Set PRETRAIN_DIR in config.sh, or add the scheme to "
              f"PRETRAIN_CORPUS in default_config.py.")
        return 0
    print(f"[tokenize] {'every configured corpus' if (a.all or a.only) else 'this run'}: "
          f"{', '.join(n for n, _ in todo)}")

    print(f"[tokenize] tokenizer: {config.BPE_PATH}")
    print(f"[tokenize] vocabulary {len(tok):,} -> "
          f"{helpers.token_cache_root(tok)}", flush=True)
    rows, total = [], 0
    for scheme, path in todo:
        if not os.path.isdir(path):
            print(f"[tokenize] {scheme}: corpus directory missing, skipped: {path}")
            rows += [(scheme, "MISSING: " + path)]
            continue
        if a.list:
            files = helpers.corpus_files(path, config.PRETRAIN["exclude_dirs"])
            sig = helpers.corpus_signature(tok, path, files, a.max_words, a.text_column)
            print(f"[tokenize] {scheme}: {len(files):,} files\n"
                  f"           {path}\n"
                  f"        -> {helpers.corpus_stream_path(tok, path, sig)}")
            continue
        stream, n_files = helpers.build_token_stream(
            path, tok, a.max_words, config.PRETRAIN["exclude_dirs"], log=print,
            text_column=a.text_column, stage=f"tokenize/{scheme}",
            force=a.force)
        if stream is None:
            print(f"[tokenize] {scheme}: no corpus files under {path}")
            rows += [(scheme, "no corpus files")]
            continue
        total += stream.n_tokens
        rows += [(f"{scheme} corpus", path),
                 (f"{scheme} files", helpers.count(n_files)),
                 (f"{scheme} documents", helpers.count(stream.n_docs)),
                 (f"{scheme} TOKENS", helpers.count(stream.n_tokens)),
                 (f"{scheme} stream", f"{stream.nbytes / 1048576:.1f} MB {stream.dtype}"),
                 (f"{scheme} path", stream.path)]
    if a.list:
        return 0
    helpers.table("tokenized corpora", rows + [("TOTAL TOKENS", helpers.count(total))])
    return 0


if __name__ == "__main__":
    sys.exit(main())
