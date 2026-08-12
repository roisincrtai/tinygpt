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
import json
import os
import shutil

import default_config as config


def target(name):
    return config.dataset_dir(name)


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
        with open(p, encoding="utf-8") as fh:
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


def fetch(name, force, log, adopt=False):
    spec = config.DATASETS[name]
    dest = target(name)
    rel = os.path.relpath(dest, config.ROOT)
    done, why = is_complete(name)

    if adopt and not done and is_present(name):
        # THE TREE IS VOUCHED FOR BY A PERSON, not by this tool. --adopt exists for corpora
        # fetched before completion was recorded -- including sharded ones, which cannot be
        # re-verified against the hub without duplicating them.
        n, total = write_marker(name, sharded=looks_sharded(name))
        log(f"[download] {name}: adopted as complete -- {n:,} files, {_size(total)}")
        return dest

    if done and not force:
        log(f"[download] {name}: complete at {rel} "
            f"({size_on_disk(name) / 2**30:.1f} GiB); --force to re-fetch")
        return dest

    if is_present(name) and not force:
        log(f"[download] {name}: INCOMPLETE at {rel} -- {why}")
        if looks_sharded(name):
            # Re-fetching would put the hub's flat layout back beside part_0000/, leaving two
            # copies of everything and a corpus that scans at twice its size. Refuse and say
            # what the two safe moves are.
            raise SystemExit(
                f"[download] {name} is split into part_NNNN/ directories and has no "
                f"completion marker.\n"
                f"           Re-fetching would download the hub's flat layout beside the "
                f"split one and leave TWO copies.\n"
                f"           If the data is good:   ./stage1_download_data.sh --adopt --only "
                f"{name}\n"
                f"           If it is not:          delete {rel} and run this again")
        log(f"[download] {name}: resuming -- snapshot_download skips whatever already matches")
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
    log(f"[download] {name}: transferred, {size_on_disk(name) / 2**30:.1f} GiB")
    # THE MARKER IS WRITTEN THE MOMENT THE FETCH RETURNS, before sharding moves anything. If
    # the process dies during sharding the marker is stale, is_complete() sees the counts still
    # agree (sharding moves files, it does not add or remove them) and the next run re-shards,
    # which is idempotent.
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
    ap.add_argument("--verify", action="store_true",
                    help="report what is complete, partial or missing, and stop")
    ap.add_argument("--adopt", action="store_true",
                    help="record an ALREADY-DOWNLOADED dataset as complete, without fetching. "
                         "For corpora that arrived before completion was recorded -- you are "
                         "vouching that the tree is whole")
    ap.add_argument("--reshard", action="store_true",
                    help="only split oversized directories of what is already downloaded; "
                         "fetches nothing")
    a = ap.parse_args(argv)
    def log(m): print(m, flush=True)

    describe(log)
    if a.list or a.verify:
        if a.verify:
            bad = [n for n in (a.only or list(config.DATASETS)) if not is_complete(n)[0]]
            log(f"[download] {len(bad)} of {len(a.only or config.DATASETS)} not complete"
                + (": " + ", ".join(bad) if bad else ""))
        return
    if a.reshard:
        for name in (a.only or list(config.DATASETS)):
            per_dir = config.DATASETS[name].get("files_per_subdir")
            if per_dir and is_present(name):
                n = shard(name, per_dir, log)
                log(f"[download] {name}: {'moved ' + format(n, ',') + ' files' if n else 'already split'}")
        return
    for name in (a.only or list(config.DATASETS)):
        fetch(name, a.force, log, adopt=a.adopt)
    log("")
    describe(log)
    log("[download] the trainers read these directories; nothing else fetches.")


if __name__ == "__main__":
    main()
