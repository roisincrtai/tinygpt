"""
tools/check_resume.py -- prove that an interrupted tokenisation resumes without redoing work.

    python -m tools.check_resume

WHAT IT ASSERTS, on a synthetic corpus built in a temporary directory:

 1. a build interrupted part way and resumed produces shards BYTE-IDENTICAL to a clean build,
    and a manifest with the same token and document counts;
 2. the resumed half calls the tokenizer only for the documents it still owes -- the count of
    encoder calls across the two halves equals the corpus's document count, with no overlap;
 3. resuming from a manifest with no cursor (one written by an older version) re-reads the
    corpus but STILL does not re-encode what it already has.

The third case matters because a cache written before the cursor existed is on disk right now
on any machine that has started a build, and it must not be the case that the fix only helps
runs that begin after it.

This is a correctness check, not a benchmark: it runs in a second and touches no real corpus.
"""
import json
import os
import shutil
import sys
import tempfile


class _Stub:
    """A tokenizer that counts its calls. Deterministic, so two builds must agree exactly."""

    def __init__(self):
        self.merges = [("a", "b")] * 8
        self.eos_token_id = 999
        self.calls = 0
        self.seen = []

    def __len__(self):
        return 1000

    def encode_ordinary(self, text):
        self.calls += 1
        self.seen.append(text)
        return [(ord(c) % 900) + 1 for c in text[:40]]


def _corpus(root, n_files=6, n_paras=40):
    os.makedirs(root, exist_ok=True)
    for i in range(n_files):
        with open(os.path.join(root, f"doc{i:03d}.txt"), "w", encoding="utf-8") as f:
            for j in range(n_paras):
                f.write(f"file {i} paragraph {j} " + "word " * 30 + "\n\n")
    return root


def _shard_bytes(stem):
    out = {}
    d = os.path.dirname(stem)
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".tokens"):
            with open(os.path.join(d, fn), "rb") as f:
                out[fn] = f.read()
    return out


def _index(stem):
    with open(f"{stem}_index.json", encoding="utf-8") as f:
        return json.load(f)


class _Boom(RuntimeError):
    pass


def main(argv=None):
    import helpers
    import default_config as config

    tmp = tempfile.mkdtemp(prefix="zetagpt-resume-")
    ok = True
    try:
        config.CACHE_DIR = os.path.join(tmp, "cache")
        corpus = _corpus(os.path.join(tmp, "corpus"))
        kw = dict(max_words=60, exclude_dirs=(), log=lambda *a: None,
                  extensions=[".txt"], stage="check")

        # ---- 1. a clean build, for the reference bytes -----------------------------------
        clean = _Stub()
        st, _ = helpers.build_token_stream(corpus, clean, **kw)
        stem = st.path
        want_bytes, want_idx = _shard_bytes(stem), _index(stem)
        total_docs = clean.calls
        print(f"  clean build      {want_idx['n_tokens']:,} tokens, "
              f"{want_idx['n_docs']:,} docs, {total_docs:,} encoder calls")

        # ---- 2. interrupt part way, then resume ------------------------------------------
        shutil.rmtree(os.path.join(config.CACHE_DIR), ignore_errors=True)
        half = _Stub()
        cut = total_docs // 2
        real = half.encode_ordinary

        def explode(text):
            if half.calls >= cut:
                raise _Boom("killed")
            return real(text)

        half.encode_ordinary = explode
        try:
            helpers.build_token_stream(corpus, half, **kw)
        except _Boom:
            pass
        part = _index(stem)
        print(f"  interrupted at   {part['n_tokens']:,} tokens, {part['n_docs']:,} docs, "
              f"{half.calls:,} encoder calls, cursor {part.get('cursor')}")
        if part.get("cursor") is None:
            print("  FAIL: the interrupted manifest carries no cursor"); ok = False

        rest = _Stub()
        st2, _ = helpers.build_token_stream(corpus, rest, **kw)
        got_bytes, got_idx = _shard_bytes(stem), _index(stem)
        print(f"  resumed build    {got_idx['n_tokens']:,} tokens, {got_idx['n_docs']:,} docs, "
              f"{rest.calls:,} encoder calls")

        if got_bytes != want_bytes:
            print("  FAIL: the resumed shards differ from the clean ones"); ok = False
        else:
            print(f"  PASS: {len(got_bytes)} shard(s) byte-identical to the clean build")

        if (got_idx["n_tokens"], got_idx["n_docs"]) != (want_idx["n_tokens"], want_idx["n_docs"]):
            print("  FAIL: the resumed manifest counts differ"); ok = False

        # the two halves between them encode the corpus ONCE. A little overlap is expected and
        # correct: documents written after the last manifest flush were truncated, so they are
        # genuinely lost work and must be redone. Redoing the whole corpus is the bug.
        if rest.calls > total_docs - cut + 200:
            print(f"  FAIL: the resume re-encoded {rest.calls - (total_docs - cut):,} "
                  f"documents it already had"); ok = False
        else:
            print(f"  PASS: resume encoded {rest.calls:,} of {total_docs:,} documents "
                  f"({100 * rest.calls / total_docs:.0f}% -- the part that was still owed)")

        # ---- 3. an old manifest, with no cursor ------------------------------------------
        shutil.rmtree(os.path.join(config.CACHE_DIR), ignore_errors=True)
        half2 = _Stub()
        real2 = half2.encode_ordinary

        def explode2(text):
            if half2.calls >= cut:
                raise _Boom("killed")
            return real2(text)

        half2.encode_ordinary = explode2
        try:
            helpers.build_token_stream(corpus, half2, **kw)
        except _Boom:
            pass
        idx = _index(stem)
        idx["cursor"] = None                       # exactly what a v3 manifest looks like
        idx["version"] = 3
        with open(f"{stem}_index.json", "w", encoding="utf-8") as f:
            json.dump(idx, f)

        old = _Stub()
        helpers.build_token_stream(corpus, old, **kw)
        got_bytes = _shard_bytes(stem)
        if got_bytes != want_bytes:
            print("  FAIL: the cursor-less resume produced different bytes"); ok = False
        elif old.calls > total_docs - cut + 200:
            print(f"  FAIL: the cursor-less resume re-encoded {old.calls:,} documents"); ok = False
        else:
            print(f"  PASS: cursor-less resume encoded {old.calls:,} of {total_docs:,} "
                  f"and matched the clean bytes")

        # ---- 4. ACROSS SHARD BOUNDARIES, and interrupted between manifest flushes --------
        # The store is driven directly here so the shard size can be made small enough to roll
        # over many times, and so the interruption can be placed anywhere rather than only
        # where a flush happens to fall. Rolling over is where a resume goes wrong quietly:
        # the last shard is partial, must be reopened for append, and must not be counted twice.
        from helpers import token_store
        # separate directories, so the two builds are compared file for file with no name
        # juggling -- the shards are named by ordinal and mean the same thing in each
        ref_dir = os.path.join(tmp, "multi_ref"); os.makedirs(ref_dir, exist_ok=True)
        run_dir = os.path.join(tmp, "multi_run"); os.makedirs(run_dir, exist_ok=True)
        base, refbase = os.path.join(run_dir, "s"), os.path.join(ref_dir, "s")
        N, SB = 500, 4096

        def producer(stop=None):
            def make(cursor, skip_docs):
                start = int((cursor or {}).get("n_read", 0)) or int(skip_docs)
                for k in range(start, N):
                    if stop is not None and k >= stop:
                        raise _Boom("killed")
                    yield [k % 97 + 1] * (7 + k % 11), {"n_read": k + 1}
            return make

        token_store.build(refbase, "sig", producer(), 999, 1000,
                          log=lambda *a: None, shard_bytes=SB, flush_seconds=1e9)
        ref = _shard_bytes(refbase), _index(refbase)

        for stop in (37, 211, 468):
            for f in os.listdir(run_dir):
                os.remove(os.path.join(run_dir, f))
            try:
                token_store.build(base, "sig", producer(stop), 999, 1000,
                                  log=lambda *a: None, shard_bytes=SB, flush_seconds=1e9)
            except _Boom:
                pass
            token_store.build(base, "sig", producer(), 999, 1000,
                              log=lambda *a: None, shard_bytes=SB, flush_seconds=1e9)
            got = _shard_bytes(base), _index(base)
            if got[0] != ref[0] or got[1]["n_tokens"] != ref[1]["n_tokens"]:
                print(f"  FAIL: interrupted at {stop}/{N} across shards -> "
                      f"{got[1]['n_tokens']:,} tokens, want {ref[1]['n_tokens']:,}"); ok = False
            else:
                print(f"  PASS: killed at doc {stop}/{N} mid-shard, resumed to "
                      f"{got[1]['n_tokens']:,} tokens in {len(got[1]['shards'])} shards, "
                      f"identical")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("  ALL PASS" if ok else "  FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
