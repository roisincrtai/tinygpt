"""
tools/download_data.py -- fetch the datasets into data/download/. The ONLY code in this
repository that downloads.

    python -m tools.download_data                     every dataset
    python -m tools.download_data --list              what would be fetched, and where
    python -m tools.download_data --only zetagpt-grpo-cot_gsm8k
    python -m tools.download_data --force             re-fetch even if present
    python -m tools.download_data --reshard           report oversized dirs (splits nothing)

    data/download/zetagpt-tiny_pretrain-corpus_wikitext103/         Tiny's corpus
    data/download/zetagpt-pretrain_fineweb-edu-2BT/   ~10 GB, ~2B tokens
    data/download/zetagpt-rlhf-instruction_following/               instruction-tuning data
    data/download/zetagpt-grpo-cot_gsm8k/                           chain-of-thought problems

WHY THE TRAINERS DO NOT DO THIS. A stage that fetches its own corpus on first use is a stage
whose corpus is not pinned: the run cannot be reproduced offline, a network hiccup surfaces as
a training failure hours in, and nobody can tell from a log whether the data on disk is the
data that was intended. Fetching is therefore a separate, deliberate act with its own command,
and every trainer reads a local directory and stops if it is not there.

WHAT IS DOWNLOADED is listed in default_config.DATASETS, one entry per dataset, each naming
the source repository and what it is for. The two larger schemes ship with NO pretraining
corpus and nothing here provides one; point PRETRAIN_DIR at your own.

HOW THE FETCH RESUMES, and why it is done the long way round. `snapshot_download` resumes
through THE SHARED CACHE and only through it: a blob is keyed by its hash, a partial transfer
is an `.incomplete` beside it, and a second run continues that same file. Handed `local_dir=`
instead, recent versions write each attempt to a UNIQUELY NAMED temporary file, so nothing is
ever continued -- an interrupted 17 GB fetch starts again from zero and leaves its partial
behind. That is not a hypothesis: this project accumulated 24 orphaned `.incomplete` files,
19 GB of them, three per shard, one per interruption, for a dataset that had not landed a
single visible file.

So the download goes to `data/.hf_cache/` and the snapshot is then HARD-LINKED into
data/download/<name>/. A hard link costs no space -- the cache and the dataset directory are
deliberately on one filesystem so that it cannot silently become a copy -- and it leaves real
files rather than symlinks for the corpus scanner to follow. Deleting the cache afterwards is
safe: the links keep the data alive.

THE REPOSITORY'S OWN LAYOUT IS PRESERVED VERBATIM, with no exception. A downloaded corpus is
read-only input from the moment it lands: nothing here rearranges it, splits it, renames it or
adds to it afterwards. `shard()` once split oversized directories into part_NNNN/ and no
longer does -- it reports and returns; `--reshard` declines out loud. The loaders walk a
corpus recursively, so a flat directory of thirty thousand files reads exactly as a split one
does, and the split bought nothing that justified editing the input.
"""
import argparse
import json
import os
import shutil

import default_config as config


def target(name):
    return config.dataset_dir(name)


# THE SHARED CACHE, ON THE SAME FILESYSTEM AS THE DATASETS. Beside data/download/ rather than in
# ~/.cache, for one reason: the snapshot is hard-linked into the dataset directory afterwards,
# and a hard link cannot cross a filesystem. Under $HOME it would silently degrade to a copy on
# any machine whose data lives on a separate mount -- which is every cluster -- and a 17 GB
# corpus would occupy 34 GB with nothing in the log to say why.
HF_CACHE = os.path.join(os.path.dirname(os.path.normpath(config.DOWNLOAD_DIR)), ".hf_cache")


def stale_staging(name):
    """(files, bytes) of the DEAD `.incomplete` staging a local_dir download left behind.

    These are unusable by anything: they are named with a per-attempt random suffix, so the
    library that wrote them will not continue them either, and this tool no longer downloads
    that way. Reported rather than deleted -- gigabytes are not removed on a tool's own
    initiative -- with the command to remove them printed beside the number."""
    root = os.path.join(target(name), ".cache", "huggingface", "download")
    n = total = 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if not f.endswith(".incomplete"):
                continue
            n += 1
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return n, total


def _link(src, dst):
    """One file from the snapshot into the dataset tree: hard link, else symlink, else copy.

    HARD LINK FIRST because it is free and indistinguishable from a real file afterwards --
    which matters, since the corpus scanner opens these and a broken symlink is a corpus that
    reads as empty. os.link follows the snapshot's own symlink into the blob, so what is linked
    is the data and not the pointer."""
    if os.path.exists(dst):
        return "kept"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(os.path.realpath(src), dst)
        return "linked"
    except OSError:
        pass
    try:
        os.symlink(os.path.realpath(src), dst)
        return "symlinked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def materialise(snapshot, dest, log):
    """The cache snapshot -> the dataset directory. Returns how many files arrived, by method."""
    try:
        from tqdm import tqdm
    except ImportError:                                            # noqa: BLE001
        tqdm = None
    files = []
    for dirpath, dirs, names in os.walk(snapshot):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        files += [os.path.join(dirpath, f) for f in names if not f.startswith(".")]
    how = {}
    it = tqdm(files, desc="[download] linking into place", unit="file") if tqdm else files
    for src in it:
        rel = os.path.relpath(src, snapshot)
        k = _link(src, os.path.join(dest, rel))
        how[k] = how.get(k, 0) + 1
    if tqdm and hasattr(it, "close"):
        it.close()
    log(f"[download]   {len(files):,} files into place ("
        + ", ".join(f"{v:,} {k}" for k, v in sorted(how.items())) + ")")
    return len(files)


def _size(n):
    """Bytes at a scale a reader can check against `du`, whatever the scale."""
    for unit, div in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:,} B"


MARKER = ".zetagpt_download.json"
MAGIC = "zetagpt-download"


def state(name):
    """(file count, total bytes) of what is on disk now. Dotfiles are ignored -- `._*` stubs
    and `.cache/` are written by macOS and by huggingface_hub, belong to neither the repo nor
    the corpus, and would make an otherwise identical tree count differently."""
    n = total = 0
    for root, dirs, files in os.walk(target(name)):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("."):
                continue
            n += 1
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n, total


def read_marker(name):
    """The completion record written after a successful fetch, or None."""
    p = os.path.join(target(name), MARKER)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            m = json.load(fh)
    except (OSError, ValueError):
        return None
    return m if m.get("magic") == MAGIC else None


def write_marker(name, sharded):
    """Record that this dataset is COMPLETE, and what complete looks like.

    A download is finished when the thing that downloads it says so -- not when a directory
    exists, and not when it holds a file. Those are true a second after the first byte lands.
    So completion is recorded explicitly, once snapshot_download has returned, and the file
    count and byte total go in with it so that a later truncation, a half-deleted tree or an
    interrupted copy between machines is caught rather than assumed away.

    NOTHING MACHINE-DEPENDENT GOES IN, for the same reason it does not go in a corpus
    signature: no path, no host, no timestamp. The repo it came from and the shape of what
    arrived are properties of the data."""
    n, total = state(name)
    p = os.path.join(target(name), MARKER)
    tmp = p + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"magic": MAGIC, "version": 1,
                   "repo": config.DATASETS[name]["repo"],
                   "n_files": n, "bytes": total, "sharded": bool(sharded)}, fh, indent=1)
    os.replace(tmp, p)          # atomic: a half-written marker must never look like completion
    return n, total


def is_complete(name):
    """(complete?, why not). COMPLETION IS A RECORD, NOT AN OBSERVATION.

    This used to be "the directory holds at least one file", and that is the bug: an
    interrupted fetch leaves hundreds of files and one missing, which passes that test exactly
    as a finished fetch does. The stage then reported "already present" and skipped, and the
    corpus trained on whatever happened to have arrived -- quietly, since a short corpus scans,
    tokenises and trains without complaint."""
    d = target(name)
    if not os.path.isdir(d):
        return False, "not downloaded"
    m = read_marker(name)
    n, total = state(name)
    if m is None:
        return False, ("no completion marker" if n else "empty directory")
    if n != m.get("n_files") or total != m.get("bytes"):
        return False, (f"changed since it was fetched: "
                       f"{n:,} files / {_size(total)} now, "
                       f"{m.get('n_files', 0):,} files / {_size(m.get('bytes', 0))} "
                       f"when it completed")
    return True, ""


def looks_sharded(name):
    """Has this tree already been split into part_NNNN/ directories?

    It matters because sharding MOVES files out of the layout the hub published, so re-running
    snapshot_download over a sharded tree would fetch every file again into the flat layout and
    leave two copies. A tree that is sharded but has no marker therefore cannot be verified by
    re-fetching, and this is what lets that case be refused rather than blundered into."""
    for _root, dirs, _files in os.walk(target(name)):
        if any(d.startswith("part_") for d in dirs):
            return True
    return False


def is_present(name):
    """Kept for the summary: is there anything here at all? Completion is is_complete()."""
    n, _ = state(name)
    return n > 0


def size_on_disk(name):
    total = 0
    for root, _dirs, files in os.walk(target(name)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def describe(log=print):
    log("")
    log("datasets")
    log("-" * 96)
    log(f"  {'name':<52} {'status':<10} {'destination'}")
    for name, spec in config.DATASETS.items():
        done, why = is_complete(name)
        if done:
            status = f"{size_on_disk(name) / 2**30:.1f} GiB"
        elif is_present(name):
            status = "PARTIAL"          # present is not complete; say which
        else:
            status = "missing"
        log(f"  {name:<52} {status:<10} {os.path.relpath(target(name), config.ROOT)}")
        if status == "PARTIAL":
            log(f"  {'':<52} {'':<10} {why} -- re-run to resume")
        log(f"  {'':<52} {'':<10} from {spec['repo']}")
        log(f"  {'':<52} {'':<10} {spec['what']}")
    log("")
    log("  pretraining corpus per scheme")
    for scheme, paths in config.PRETRAIN_CORPUS.items():
        # A scheme reads SEVERAL corpora, each one standalone in the cache, so list them one
        # per line rather than joining them into a path-shaped string nobody can parse.
        paths = [p for p in (paths if isinstance(paths, (list, tuple)) else [paths]) if p]
        if not paths:
            log(f"    {scheme:<12} (unset -- configure PRETRAIN_DIR for your own corpus)")
            continue
        for i, path in enumerate(paths):
            log(f"    {scheme if i == 0 else '':<12} {os.path.relpath(path, config.ROOT)}")
    log("-" * 96)


def shard(name, per_dir, log, dry_run=False):
    """DISABLED. It reports, and it moves nothing.

    A downloaded corpus is left EXACTLY as it was fetched. Its directory structure is part of
    the dataset -- it is what the repository published, what a signature is computed over, and
    what every other machine holding the same dataset has -- and rearranging it in place edits
    the one tree in this project that cannot be regenerated from anything else here. The files
    are not copied first, they are `os.replace`d, so an interruption leaves a corpus half in
    `part_0000/` and half beside it, matching neither what was downloaded nor what a re-run
    expects.

    Nothing needed this. `corpus_files` walks recursively, so a flat directory of thirty
    thousand files reads exactly as a split one does; the split was cosmetic, and it was
    cosmetic on the input.

    The function is kept, reporting what it would once have done, so a caller does not vanish
    into an AttributeError and so `--reshard` says why it declines instead of appearing to
    work."""
    root = target(name)
    todo = []
    for here, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        loose = sorted(f for f in files if not f.startswith("."))
        if len(loose) > per_dir:
            todo.append((here, loose))
    if not todo:
        return 0
    log(f"[download] {name}: NOT resharding. {len(todo)} directory(ies) hold more than "
        f"{per_dir} files, and they stay exactly as downloaded:")
    for here, loose in todo:
        log(f"[download]   {os.path.relpath(here, config.ROOT)}: {len(loose):,} files")
    log(f"[download] a downloaded corpus is read-only input. The loaders walk it recursively, "
        f"so its layout costs nothing.")
    return 0


def free_space(path):
    d = path
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d or ".").free


def fetch(name, force, log):
    spec = config.DATASETS[name]
    dest = target(name)
    rel = os.path.relpath(dest, config.ROOT)
    done, why = is_complete(name)

    if done and not force:
        log(f"[download] {name}: complete at {rel} "
            f"({size_on_disk(name) / 2**30:.1f} GiB); --force to re-fetch")
        return dest

    if is_present(name) and not force and looks_sharded(name):
        # A SHARDED TREE CANNOT BE RE-FETCHED, so it is recorded rather than re-downloaded.
        # Sharding MOVES files out of the layout the hub published, so snapshot_download would
        # fetch every one of them again into the flat layout and leave two copies -- a corpus
        # that scans at twice its size. There is no way to verify it against the hub, and the
        # tree in front of us is the only evidence there is, so it is taken as the answer and
        # written down. A dataset from before completion was recorded lands here exactly once.
        if read_marker(name) is None:
            n, total = write_marker(name, sharded=True)
            log(f"[download] {name}: already split into part_NNNN/ and predates the completion "
                f"record -- taking it as complete ({n:,} files, {_size(total)})")
            return dest
        # It HAD a marker and no longer matches it: something was removed or truncated after
        # the fetch. Re-fetching is still unsafe for the reason above, so say so plainly.
        raise SystemExit(
            f"[download] {name}: {why}\n"
            f"           This dataset is split into part_NNNN/ directories, so it cannot be\n"
            f"           re-fetched in place -- the hub's flat layout would land beside the\n"
            f"           split one and leave two copies of everything.\n"
            f"           Delete {rel} and run this again.")

    if is_present(name) and not force:
        log(f"[download] {name}: incomplete -- {why}")
        log(f"[download] {name}: resuming from {os.path.relpath(HF_CACHE, config.ROOT)}")
    # DEAD STAGING FROM THE OLD local_dir FETCH. Reported here, once, with its size and the
    # command that removes it -- it is never continued by anything and never will be.
    n_dead, b_dead = stale_staging(name)
    if n_dead:
        log(f"[download] {name}: {n_dead:,} orphaned .incomplete files ({_size(b_dead)}) from "
            f"the old local_dir fetch,")
        log(f"[download]   which nothing resumes. Remove with:")
        log(f"[download]   rm -rf {os.path.relpath(target(name), config.ROOT)}"
            f"/.cache/huggingface/download")
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:                                         # noqa: BLE001
        raise SystemExit(
            f"[download] huggingface_hub is not importable ({e}).\n"
            f"           pip install huggingface_hub, or fetch {spec['repo']} by hand into\n"
            f"           {os.path.relpath(dest, config.ROOT)}") from e

    os.makedirs(dest, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    log(f"[download] {name}")
    log(f"[download]   from {spec['repo']}")
    log(f"[download]   via  {os.path.relpath(HF_CACHE, config.ROOT)}  (resumable; hard-linked "
        f"into place after)")
    log(f"[download]   into {os.path.relpath(dest, config.ROOT)}")
    log(f"[download]   {free_space(dest) / 2**30:.1f} GiB free on that filesystem")
    try:
        # NO local_dir=. That is the whole fix: with local_dir the library writes each attempt
        # to a uniquely named temporary file and resumes nothing, so an interrupted transfer
        # restarts from zero and leaves its partial behind. Into the cache, a partial file is
        # an .incomplete keyed by the blob's hash and the next run CONTINUES IT.
        snapshot = snapshot_download(repo_id=spec["repo"], repo_type="dataset",
                                     cache_dir=HF_CACHE)
    except Exception as e:                                         # noqa: BLE001
        # Left partially fetched ON PURPOSE: the cache resumes, so the right response to a
        # failure is to run this again, and deleting the partial would throw away everything
        # already transferred.
        raise SystemExit(
            f"[download] {name} failed: {type(e).__name__}: {e}\n"
            f"           Whatever transferred is kept in "
            f"{os.path.relpath(HF_CACHE, config.ROOT)};\n"
            f"           re-run this command to resume from there.") from e
    materialise(snapshot, dest, log)
    log(f"[download] {name}: transferred, {size_on_disk(name) / 2**30:.1f} GiB")
    # THE MARKER IS WRITTEN THE MOMENT THE FETCH RETURNS. Nothing rearranges the corpus after
    # this point -- shard() reports and moves nothing -- so the tree the marker describes is
    # the tree that stays on disk.
    write_marker(name, sharded=False)
    per_dir = spec.get("files_per_subdir")
    if per_dir:
        n = shard(name, per_dir, log)
        if n:
            log(f"[download] {name}: moved {n:,} files into subdirectories of {per_dir}")
        write_marker(name, sharded=True)
    n, total = state(name)
    log(f"[download] {name}: COMPLETE -- {n:,} files, {_size(total)}, recorded in "
        f"{MARKER}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="fetch the datasets into data/download/")
    ap.add_argument("--only", action="append", choices=list(config.DATASETS),
                    help="fetch just this dataset; repeatable")
    ap.add_argument("--list", action="store_true", help="show what would be fetched, then stop")
    ap.add_argument("--force", action="store_true", help="re-fetch even if already complete")
    ap.add_argument("--reshard", action="store_true",
                    help="only split oversized directories of what is already downloaded; "
                         "fetches nothing")
    a = ap.parse_args(argv)
    def log(m): print(m, flush=True)

    describe(log)
    if a.list:
        return
    if a.reshard:
        for name in (a.only or list(config.DATASETS)):
            per_dir = config.DATASETS[name].get("files_per_subdir")
            if per_dir and is_present(name):
                n = shard(name, per_dir, log)
                log(f"[download] {name}: {'moved ' + format(n, ',') + ' files' if n else 'already split'}")
        return
    for name in (a.only or list(config.DATASETS)):
        fetch(name, a.force, log)
    log("")
    describe(log)
    log("[download] the trainers read these directories; nothing else fetches.")


if __name__ == "__main__":
    main()
