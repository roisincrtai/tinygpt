"""
tools/download_data.py -- fetch the datasets into data/download/. The ONLY code in this
repository that downloads.

    python -m tools.download_data                     every dataset
    python -m tools.download_data --list              what would be fetched, and where
    python -m tools.download_data --only zetagpt-grpo-cot_gsm8k
    python -m tools.download_data --force             re-fetch even if present
    python -m tools.download_data --reshard           only split oversized directories

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

The fetch itself is `huggingface_hub.snapshot_download`, which is resumable, verifies file
hashes, and skips what is already present -- so an interrupted download is continued by
re-running this, not restarted. The repository's own layout is preserved verbatim under the
target directory, with ONE exception: a dataset carrying `files_per_subdir` has its oversized
directories split into part_NNNN/ afterwards. See shard() for why, and note that it changes
where files sit, never which files exist.
"""
import argparse
import os
import shutil

import default_config as config


def target(name):
    return config.dataset_dir(name)


def is_present(name):
    """A dataset counts as present if its directory exists and holds at least one file. The
    file check matters: an interrupted fetch can leave the directory created and empty, and
    "the directory exists" would then report success for a corpus that is not there."""
    d = target(name)
    if not os.path.isdir(d):
        return False
    for _root, _dirs, files in os.walk(d):
        if files:
            return True
    return False


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
        here = is_present(name)
        status = f"{size_on_disk(name) / 2**30:.1f} GiB" if here else "missing"
        log(f"  {name:<52} {status:<10} {os.path.relpath(target(name), config.ROOT)}")
        log(f"  {'':<52} {'':<10} from {spec['repo']}")
        log(f"  {'':<52} {'':<10} {spec['what']}")
    log("")
    log("  pretraining corpus per scheme")
    for scheme, path in config.PRETRAIN_CORPUS.items():
        shown = os.path.relpath(path, config.ROOT) if path else \
            "(unset -- configure PRETRAIN_DIR for your own corpus)"
        log(f"    {scheme:<12} {shown}")
    log("-" * 96)


def shard(name, per_dir, log, dry_run=False):
    """Split any directory under the dataset that holds more than `per_dir` loose files into
    part_0000/, part_0001/, ... of at most `per_dir` each.

    WHY. This corpus arrives as one directory of ~29,500 files. A directory that large is slow
    to list on most filesystems, slow to complete in a shell, and unpleasant to open in
    anything; on some it degrades badly enough to be felt on every scan. Splitting costs
    nothing at read time because the corpus scanner walks RECURSIVELY -- every stage sees the
    same files in the same order whether they sit in one directory or sixty.

    IDEMPOTENT BY CONSTRUCTION. It shards a directory only when that directory holds more than
    `per_dir` LOOSE files, and after a pass none does -- the parts hold at most `per_dir` each
    and the parent holds none. Re-running the download therefore does nothing, and a dataset
    that arrives already split is left exactly as it is.

    Names are sorted before splitting, so which file lands in which part is a function of the
    dataset and not of the order the filesystem happened to return."""
    root = target(name)
    todo = []
    for here, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        loose = sorted(f for f in files if not f.startswith("."))
        if len(loose) > per_dir:
            todo.append((here, loose))
    if not todo:
        return 0
    moved = 0
    for here, loose in todo:
        n_parts = (len(loose) + per_dir - 1) // per_dir
        log(f"[download] {os.path.relpath(here, config.ROOT)}: {len(loose):,} files -> "
            f"{n_parts} directories of at most {per_dir}")
        if dry_run:
            continue
        for i in range(0, len(loose), per_dir):
            part = os.path.join(here, f"part_{i // per_dir:04d}")
            os.makedirs(part, exist_ok=True)
            for fn in loose[i:i + per_dir]:
                os.replace(os.path.join(here, fn), os.path.join(part, fn))
                moved += 1
    return moved


def free_space(path):
    d = path
    while d and not os.path.isdir(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d or ".").free


def fetch(name, force, log):
    spec = config.DATASETS[name]
    dest = target(name)
    if is_present(name) and not force:
        log(f"[download] {name}: already present at "
            f"{os.path.relpath(dest, config.ROOT)} ({size_on_disk(name) / 2**30:.1f} GiB); "
            f"--force to re-fetch")
        # still worth a pass: a dataset fetched before this setting existed, or by hand, is
        # unsharded, and the check costs one directory walk
        per_dir = spec.get("files_per_subdir")
        if per_dir:
            n = shard(name, per_dir, log)
            if n:
                log(f"[download] {name}: moved {n:,} files into subdirectories of {per_dir}")
        return dest
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:                                         # noqa: BLE001
        raise SystemExit(
            f"[download] huggingface_hub is not importable ({e}).\n"
            f"           pip install huggingface_hub, or fetch {spec['repo']} by hand into\n"
            f"           {os.path.relpath(dest, config.ROOT)}") from e

    os.makedirs(dest, exist_ok=True)
    log(f"[download] {name}")
    log(f"[download]   from {spec['repo']}")
    log(f"[download]   into {os.path.relpath(dest, config.ROOT)}")
    log(f"[download]   {free_space(dest) / 2**30:.1f} GiB free on that filesystem")
    try:
        snapshot_download(repo_id=spec["repo"], repo_type="dataset", local_dir=dest,
                          resume_download=True)
    except Exception as e:                                         # noqa: BLE001
        # Left partially fetched ON PURPOSE: snapshot_download resumes, so the right response
        # to a failure is to run this again, and deleting the partial copy would throw away
        # everything already transferred.
        raise SystemExit(
            f"[download] {name} failed: {type(e).__name__}: {e}\n"
            f"           Whatever transferred is kept at "
            f"{os.path.relpath(dest, config.ROOT)};\n"
            f"           re-run this command to resume.") from e
    log(f"[download] {name}: done, {size_on_disk(name) / 2**30:.1f} GiB")
    per_dir = spec.get("files_per_subdir")
    if per_dir:
        n = shard(name, per_dir, log)
        if n:
            log(f"[download] {name}: moved {n:,} files into subdirectories of {per_dir}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="fetch the datasets into data/download/")
    ap.add_argument("--only", action="append", choices=list(config.DATASETS),
                    help="fetch just this dataset; repeatable")
    ap.add_argument("--list", action="store_true", help="show what would be fetched, then stop")
    ap.add_argument("--force", action="store_true", help="re-fetch even if already present")
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
