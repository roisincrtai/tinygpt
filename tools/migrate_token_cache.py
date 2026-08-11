"""
tools/migrate_token_cache.py -- move an existing token cache onto the portable naming.

    python -m tools.migrate_token_cache --dry_run     what would move, and where
    python -m tools.migrate_token_cache               move it

WHY THIS EXISTS. Token streams used to be named after WHERE a corpus sat and WHEN its files
were last written: the cache mirrored the corpus's path, and the signature hashed every file's
modification time. Both are properties of a machine rather than of a dataset, so the same
one corpus could reach two different names on two hosts, and the second would retokenise what
the first had already finished. The naming is now the dataset's name and the dataset's
contents, which are the same everywhere.

Streams already on disk carry the old names. They are correct -- nothing about the TOKENS was
wrong, only the label -- and on a 2B-token corpus they represent hours. This renames them to
what the current code computes, so the ordinary lookup finds them exactly and nothing is
tokenised again.

    cache/tokens/<tokenizer>/data/download/<dataset>/<dataset>_<old sig>_00000.tokens
 -> cache/tokens/<tokenizer>/<dataset>/<dataset>_<new sig>_00000.tokens

NOTHING IS COPIED AND NOTHING IS DELETED. Every step is a rename within one filesystem, which
is instant whether the stream is 40 MB or 4 TB, and cannot half-succeed the way a copy of a
4 GB file can. The manifest is rewritten last, so a run interrupted mid-migration leaves the
shards where they are and can simply be run again.

THE TARGET NAME COMES FROM THE LIVE CODE. helpers.corpus_signature and
helpers.corpus_stream_path are called here exactly as a training stage calls them, so this
cannot compute one name while the pipeline looks for another -- which would be the one way a
migration could quietly cost more than it saved. Afterwards each stream is opened through the
ordinary lookup and the result reported.
"""
import argparse
import json
import os
import sys


def _corpus_root(dataset, log):
    """Where THIS machine keeps `dataset`, or None. The name is the identity; the path is not."""
    import default_config as config
    for path in list(config.PRETRAIN_CORPUS.values()) + [config.dataset_dir(dataset)]:
        if path and os.path.basename(os.path.normpath(path)) == dataset and os.path.isdir(path):
            return path
    return None


def _manifests(cache_root):
    """Every token-store manifest under the cache, old layout or new."""
    out = []
    for dirpath, dirnames, filenames in os.walk(cache_root):
        dirnames.sort()
        for fn in sorted(filenames):
            # `._name` is an AppleDouble stub left by a copy from a Mac: it ends in
            # _index.json, is not JSON, and describes a stream that does not exist.
            if not fn.endswith("_index.json") or fn.startswith("._"):
                continue
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                    idx = json.load(f)
            except (OSError, ValueError):
                continue
            if idx.get("magic") == "zetagpt-tokens":
                out.append((os.path.join(dirpath, fn[:-len("_index.json")]), idx))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="rename token streams onto the portable naming")
    ap.add_argument("--dry_run", action="store_true", help="report only; move nothing")
    ap.add_argument("--bpe", default="", help="tokenizer json; default config.BPE_PATH")
    a = ap.parse_args(argv)

    import default_config as config
    import helpers
    from helpers import token_store
    from tokenizer import run as train_bpe

    bpe_path = a.bpe or config.BPE_PATH
    tok = train_bpe.peek(bpe_path)
    if tok is None:
        raise SystemExit(f"[migrate] no usable tokenizer at {bpe_path}\n"
                         f"          The streams are named after the vocabulary that produced "
                         f"them, so it has to be readable to name them again.")
    cache_root = os.path.join(config.CACHE_DIR, "tokens")
    if not os.path.isdir(cache_root):
        print(f"[migrate] nothing to do: {cache_root} does not exist")
        return 0

    found = _manifests(cache_root)
    print(f"[migrate] tokenizer {bpe_path}  ({len(tok):,} tokens -> "
          f"{os.path.basename(helpers.token_cache_root(tok))})")
    print(f"[migrate] {len(found)} stream(s) under {cache_root}")

    rows, moved, already, skipped = [], 0, 0, 0
    for stem, idx in found:
        dataset = os.path.basename(os.path.dirname(stem))
        root = _corpus_root(dataset, print)
        n_tok = idx.get("n_tokens", 0)
        state = "complete" if idx.get("complete") else f"partial/{len(idx.get('shards', []))}"
        if root is None:
            print(f"[migrate] SKIP {dataset}: no corpus of that name on this machine, so the "
                  f"signature cannot be recomputed. Point PRETRAIN_DIR at it and re-run.")
            rows.append((dataset, f"skipped ({state}, {n_tok:,} tokens)")); skipped += 1
            continue

        files = helpers.corpus_files(root, config.PRETRAIN["exclude_dirs"])
        sig = helpers.corpus_signature(tok, root, files, config.PRETRAIN["max_words"],
                                       config.PRETRAIN["text_column"])
        want = helpers.corpus_stream_path(tok, root, sig)
        packing = (f"{helpers.utils._tok_signature(tok, config.PRETRAIN['max_words'], config.PRETRAIN['text_column'])}"
                   f"|files={len(files)}")
        if os.path.abspath(want) == os.path.abspath(stem):
            print(f"[migrate] OK   {dataset}: already correctly named ({state}, {n_tok:,} tokens)")
            rows.append((dataset, f"already correct ({n_tok:,} tokens)")); already += 1
            continue

        shards = [s["file"] for s in idx.get("shards", [])]
        print(f"[migrate] MOVE {dataset}  ({state}, {n_tok:,} tokens, {len(shards)} shard(s))")
        print(f"[migrate]   from {stem}")
        print(f"[migrate]   to   {want}")
        if a.dry_run:
            rows.append((dataset, f"would move {len(shards)} shard(s), {n_tok:,} tokens"))
            continue

        os.makedirs(os.path.dirname(want) or ".", exist_ok=True)
        old_dir, new_dir = os.path.dirname(stem), os.path.dirname(want)
        new_shards = []
        for s in idx["shards"]:
            tail = s["file"][len(os.path.basename(stem)):]          # _00007.tokens
            dst = os.path.basename(want) + tail
            src_p, dst_p = os.path.join(old_dir, s["file"]), os.path.join(new_dir, dst)
            if os.path.exists(src_p):
                os.replace(src_p, dst_p)
            s = dict(s); s["file"] = dst
            new_shards.append(s)

        # The manifest is written LAST and under its new name, so an interruption anywhere
        # above leaves the shards renamed and this run repeatable rather than a half-state.
        token_store._write_index(want, sig, idx["dtype"], idx["eos"], idx["vocab_size"],
                                 new_shards, idx.get("complete", False),
                                 idx.get("cursor"), packing)
        try:
            os.remove(stem + "_index.json")
        except OSError:
            pass
        rows.append((dataset, f"moved {len(new_shards)} shard(s), {n_tok:,} tokens"))
        moved += 1

    # VERIFY THROUGH THE ORDINARY LOOKUP, not through the migration's own idea of the name.
    if not a.dry_run:
        print()
        for stem, idx in _manifests(cache_root):
            dataset = os.path.basename(os.path.dirname(stem))
            root = _corpus_root(dataset, print)
            if root is None:
                continue
            files = helpers.corpus_files(root, config.PRETRAIN["exclude_dirs"])
            sig = helpers.corpus_signature(tok, root, files, config.PRETRAIN["max_words"],
                                           config.PRETRAIN["text_column"])
            st = token_store.open_if_current(helpers.corpus_stream_path(tok, root, sig), sig)
            print(f"[migrate] {dataset}: "
                  + (f"FOUND by the ordinary lookup -- {st.describe()}" if st else
                     "not found as complete; a training stage will resume it, not restart it"))

    helpers.table("token cache migration", rows + [
        ("moved", str(moved)), ("already correct", str(already)), ("skipped", str(skipped))])
    if a.dry_run:
        print("  --dry_run: nothing was moved. Re-run without it to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
