"""
tools/rename_dataset.py -- rename a dataset ON DISK, and carry its token cache across.

    python -m tools.rename_dataset <old name> <new name> [--dry-run] [--cache-only]
    python -m tools.rename_dataset zetagpt-small_pretrain-corpus_fineweb-edu_10GB \\
                                   zetagpt-pretrain_fineweb-edu-2BT

WHY THIS EXISTS RATHER THAN `mv`. The dataset NAME is an input to the corpus signature -- and
deliberately so: a signature must be identical on every machine that has a copy, so it is
built from the name, the file count and the byte count, and from nothing about the host (see
helpers/utils.corpus_signature). The consequence is that renaming a corpus CHANGES ITS
SIGNATURE, and a signature is:

    * hashed into every shard and index FILENAME     <name>_<sig8>_00000.tokens
    * stored inside the index                        {"sig": "corpus|v2|files=N|<hash>"}
    * the thing a stage compares before reusing a stream

So `mv` alone leaves a cache that the pipeline cannot see. It does not fail: it silently
re-tokenises, which on the 2B-token corpus is hours, and on the 10B one considerably more.
That is the exact failure this project has already paid for once.

WHAT IS MOVED

    data/download/<old>/                    ->  data/download/<new>/
    cache/tokens/<tokenizer>/<old>/         ->  cache/tokens/<tokenizer>/<new>/
      <old>_<oldsig8>_NNNNN.tokens          ->    <new>_<newsig8>_NNNNN.tokens
      <old>_<oldsig8>_index.json            ->    <new>_<newsig8>_index.json
        {"sig": <old signature>}            ->      {"sig": <new signature>}
        {"shards": [{"file": ...}]}         ->      rewritten to the new names

EVERY TOKENIZER'S CACHE, not just the current one: cache/tokens/ holds one directory per
vocabulary, and a corpus that was tokenised under two of them has two streams to carry across.
Both layouts are handled -- the current cache/tokens/<tok>/<name>/ and the older one that
mirrored the corpus's path under the cache root -- because a machine that has been running
this pipeline for a while has both.

THE TOKENS THEMSELVES ARE NEVER READ OR REWRITTEN. Renaming a corpus does not change one byte
of it, so the shards are correct as they stand; only their names and the manifest that points
at them have to move. That is what makes this seconds rather than hours.

--dry-run prints every move and changes nothing. Run it first.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import default_config as config                                        # noqa: E402
import helpers                                                         # noqa: E402
from helpers.utils import _sig_tag                                     # noqa: E402


class TokenizerTag:
    """Just enough tokenizer for the signature: the MERGE COUNT, read off the cache directory.

    corpus_signature reaches into the tokenizer for exactly one thing -- how many merges it has
    (helpers.utils._tok_signature) -- and the cache directory is named bpe_<256 + merges>_<fp>,
    so the count is already in the path. Standing in for the vocabulary this way means the
    pipeline's OWN signature function computes the new signature, with no second implementation
    of the hash to drift from the first, and means four vocabularies can be re-signed without
    loading a single one of them.

    A directory whose name does not parse leaves `merges` empty, and migrate() refuses rather
    than signing with a merge count of zero -- which would produce a plausible, wrong hash."""

    __slots__ = ("merges", "tag")

    def __init__(self, tag):
        self.tag = tag
        parts = tag.split("_")
        n_base = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        self.merges = [None] * max(0, n_base - 256)

    def __bool__(self):
        return bool(self.merges)


def corpus_files_and_bytes(root):
    """The two numbers the signature is built from, for the corpus AT ITS NEW PATH."""
    files = helpers.corpus_files(root, config.PRETRAIN["exclude_dirs"])
    total = 0
    for f in files:
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return files, total


def streams_for(name):
    """Every (directory, stem) pair holding a token stream of `name`, under any tokenizer.

    TWO LAYOUTS. The current one puts a corpus at cache/tokens/<tok>/<name>/; the older one
    mirrored the corpus's path, cache/tokens/<tok>/data/download/<name>/. A machine that has
    been running this pipeline across the change has both, and a migration that only knew about
    one would leave the other stranded -- invisible to the pipeline and impossible to explain
    later."""
    root = os.path.join(config.CACHE_DIR, "tokens")
    out = []
    if not os.path.isdir(root):
        return out
    for tokdir in sorted(os.listdir(root)):
        base = os.path.join(root, tokdir)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            if os.path.basename(dirpath) != name:
                continue
            for fn in filenames:
                if fn.endswith("_index.json"):
                    out.append((dirpath, fn))
            dirnames[:] = []          # a corpus directory holds no corpus directories
    return out


def migrate(old, new, dry_run=False, cache_only=False, log=print):
    """Rename the dataset and re-sign every token stream of it. Returns the number of streams
    carried across.

    THE DATA MOVES FIRST, because the new signature is computed from the corpus at its NEW
    name -- the name is what changed, and the file count and byte total must be read from the
    files that will actually be there."""
    old_dir = config.dataset_dir(old)
    new_dir = config.dataset_dir(new)

    if not cache_only:
        if not os.path.isdir(old_dir) and os.path.isdir(new_dir):
            log(f"[rename] {new_dir} already exists and {old} does not; data already moved")
        elif not os.path.isdir(old_dir):
            raise SystemExit(f"[rename] no such dataset: {old_dir}")
        elif os.path.isdir(new_dir):
            raise SystemExit(
                f"[rename] BOTH exist:\n  {old_dir}\n  {new_dir}\n"
                f"          Refusing to merge two corpora. Move or delete one first.")
        else:
            log(f"[rename] data  {old_dir}\n              -> {new_dir}")
            if not dry_run:
                os.rename(old_dir, new_dir)

    # THE FILES ARE READ FROM WHEREVER THEY ARE; THE NAME IS ALWAYS THE NEW ONE. Those are two
    # different inputs to the signature and a dry run has to get both right: reading the name
    # from the directory that has not been renamed yet made --dry-run report a signature the
    # real run would never write, which is worse than no dry run at all. corpus_signature takes
    # `root` only to derive the dataset name, so it is handed the destination even before the
    # destination exists.
    here = new_dir if os.path.isdir(new_dir) else old_dir
    if not os.path.isdir(here):
        raise SystemExit(f"[rename] cannot sign a corpus that is not on disk: {new_dir}")
    files, total = corpus_files_and_bytes(here)
    log(f"[rename] corpus: {len(files):,} files, {total:,} bytes")

    streams = streams_for(old)
    if not streams:
        log(f"[rename] no token streams found for {old} -- nothing further to do")
        return 0

    # ONE TOKENIZER PER CACHE DIRECTORY, and the signature depends on it, so each stream is
    # re-signed under the tokenizer whose directory it sits in rather than under a single
    # global one. Reading the tokenizer from the path is what makes that possible without
    # loading four vocabularies.
    n = 0
    for dirpath, index_name in streams:
        old_index = os.path.join(dirpath, index_name)
        with open(old_index, encoding="utf-8") as fh:
            idx = json.load(fh)
        old_sig = idx.get("sig", "")
        old_tag = index_name[len(old) + 1:-len("_index.json")]

        # The tokenizer directory is the one two-or-more levels up whose name starts with bpe_;
        # its own tag is all the signature needs from it.
        tok_dir = dirpath
        while tok_dir and os.path.basename(tok_dir) != "tokens":
            parent = os.path.dirname(tok_dir)
            if os.path.basename(parent) == "tokens":
                break
            tok_dir = parent
        tok_tag = os.path.basename(tok_dir)

        # THE NEW SIGNATURE IS DERIVED THE SAME WAY THE PIPELINE DERIVES IT, by calling the
        # pipeline's own function on a fake tokenizer carrying the recorded fingerprint. A
        # migration that computed the hash its own way would be a second implementation of the
        # identity, and the two would eventually disagree -- which is the whole class of bug
        # this file exists to avoid.
        shim = TokenizerTag(tok_tag)
        if not shim:
            raise SystemExit(
                f"[rename] cannot read a merge count from the cache directory {tok_tag!r}, so "
                f"the new signature cannot be computed. Expected bpe_<256+merges>_<fingerprint>.")
        new_sig = helpers.corpus_signature(shim, new_dir, files,
                                           config.PRETRAIN["max_words"],
                                           config.PRETRAIN["text_column"])
        new_tag = _sig_tag(new_sig)

        target_dir = os.path.join(os.path.dirname(dirpath), new)
        log(f"[rename] stream under {tok_tag}")
        log(f"           {dirpath}")
        log(f"        -> {target_dir}")
        log(f"           sig {old_sig}")
        log(f"        ->     {new_sig}")
        log(f"           tag {old_tag} -> {new_tag}, {len(idx.get('shards', []))} shard(s)")

        if dry_run:
            n += 1
            continue

        if os.path.isdir(target_dir) and os.path.abspath(target_dir) != os.path.abspath(dirpath):
            raise SystemExit(f"[rename] {target_dir} already exists; refusing to merge")
        if os.path.abspath(target_dir) != os.path.abspath(dirpath):
            os.rename(dirpath, target_dir)

        for shard in idx.get("shards", []):
            src = os.path.join(target_dir, shard["file"])
            dst_name = shard["file"].replace(f"{old}_{old_tag}_", f"{new}_{new_tag}_", 1)
            if os.path.isfile(src):
                os.rename(src, os.path.join(target_dir, dst_name))
            elif not os.path.isfile(os.path.join(target_dir, dst_name)):
                raise SystemExit(f"[rename] shard missing, refusing to write a manifest that "
                                 f"points at nothing: {src}")
            shard["file"] = dst_name

        idx["sig"] = new_sig
        # WRITTEN THROUGH A TEMPORARY AND RENAMED, like every other manifest write here: a
        # half-written index is an unreadable stream, and this runs on a cache that took hours.
        tmp = os.path.join(target_dir, f"{new}_{new_tag}_index.json.part")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=1)
        os.replace(tmp, os.path.join(target_dir, f"{new}_{new_tag}_index.json"))
        stale = os.path.join(target_dir, index_name)
        if os.path.isfile(stale) and index_name != f"{new}_{new_tag}_index.json":
            os.remove(stale)
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="rename a dataset on disk and carry its token cache across")
    ap.add_argument("old", help="the current directory name under data/download/")
    ap.add_argument("new", help="the new one")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every move and change nothing")
    ap.add_argument("--cache-only", action="store_true",
                    help="the data directory is already renamed; only re-sign the cache")
    a = ap.parse_args(argv)
    if a.old == a.new:
        raise SystemExit("[rename] the two names are the same")
    n = migrate(a.old, a.new, dry_run=a.dry_run, cache_only=a.cache_only)
    print(f"\n[rename] {'would carry' if a.dry_run else 'carried'} {n} token stream(s) across"
          + ("  (dry run: nothing changed)" if a.dry_run else ""))
    if not a.dry_run and n:
        print("[rename] verify with:  ./stage4_tokenize_data.sh --list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
