"""
tokenize_data/run.py -- stage 3: turn every configured corpus into a token stream.

    python -m tokenize_data.run [--only tiny] [--force] [--list]
    ./stage3_tokenize_data.sh

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

WHAT IT WRITES. One memory-mapped .tokens file per (corpus, tokenizer, packing), under
cache/tokens/bpe_<vocab>_<fingerprint>/. See helpers/token_store.py for the format. Re-running is
cheap: a stream whose signature still matches its corpus is left alone, and --force rebuilds.

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
    """(name, directory) for every corpus to tokenise: one per scheme that configures one,
    deduplicated by directory -- Tiny and S may well be pointed at the same corpus, and
    tokenising it twice would write the same stream twice under the same name."""
    out, seen = [], set()
    for scheme, path in config.PRETRAIN_CORPUS.items():
        if args.only and args.only not in scheme:
            continue
        if not path:
            continue                       # -M and -L ship with none; that is not an error
        key = os.path.abspath(path)
        if key in seen:
            continue
        seen.add(key)
        out.append((scheme, path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="tokenise the corpora into token streams")
    ap.add_argument("--only", default="", help="substring of a scheme name, e.g. tiny")
    ap.add_argument("--force", action="store_true", help="rebuild even if current")
    ap.add_argument("--list", action="store_true", help="show what would be built, and where")
    ap.add_argument("--max_words", type=int, default=config.PRETRAIN["max_words"])
    ap.add_argument("--text_column", default=config.PRETRAIN["text_column"])
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="SEC.key=v", help="override any default_config value, e.g. "
                         "--set PRETRAIN.max_words=512")
    a, _ = ap.parse_known_args(argv)
    config.apply_overrides(a.overrides, print)

    # The tokenizer is READ, never built here. A stream is tied to the vocabulary that
    # produced it -- that is what its cache directory is named after -- so a stage that
    # quietly trained its own BPE would write streams no other stage could use.
    from tokenizer import run as train_bpe
    tok = train_bpe.peek(config.BPE_PATH)
    if tok is None:
        raise SystemExit(
            f"[tokenize] no usable tokenizer at {config.BPE_PATH}\n"
            f"           Run ./stage2_train_bpe_tokenizer.sh first: a token stream is tied "
            f"to the vocabulary that produced it.")

    todo = corpora(a)
    if not todo:
        print("[tokenize] no corpus is configured for any scheme; nothing to do.\n"
              "           PRETRAIN_CORPUS in default_config.py, or PRETRAIN_DIR in config.sh.")
        return 0

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
            text_column=a.text_column, stage=f"tokenize/{scheme}", force=a.force)
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
