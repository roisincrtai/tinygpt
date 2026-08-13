"""
token_store.py -- the pre-tokenised corpus as a set of memory-mapped SHARD FILES.

    build(stem, sig, documents, eos, vocab, log)   write it, appending as it goes
    TokenStream(stem)                              open it; nothing is read into memory
    stream.batch(B, T, generator, device)          (ids, attn, rmask) for one training step

NOTHING IS HELD BACK. Each document is appended to the open shard the moment it is tokenised,
and the manifest is rewritten every few seconds with the counts that are actually on disk.
Buffering a shard's worth of tokens before writing would mean a run killed part way through
left an empty directory -- which is exactly what a corpus of 10 TB cannot afford, and what
this module used to do. The invariant is: what has been read has been written.

RESUME IS THE DEFAULT, AND IT COSTS NOTHING. A build finds the manifest, truncates the last
shard back to the last recorded position (the bytes after it were written but not yet
accounted for), and continues from the document after the last one counted. This is sound
because the document order is deterministic -- the corpus files are sorted, each is packed in
order -- so document N is always the same document. Pass resume=False to start again.

What makes it cost nothing is the CURSOR. The manifest carries an opaque position handed back
to the producer, which uses it to seek: files already consumed are never opened, and documents
already stored are never tokenised again. Resuming by counting instead -- reading the corpus
from the top and discarding the first N documents -- runs the tokenizer over work already on
disk, which at 10 TB is days of a GPU-free machine spent producing bytes it already has. The
rule is that the cost of resuming is proportional to what was LOST, not to what was done.

The cursor is written by the producer and never interpreted here; the store only persists it
next to the counts it belongs with, and only at the moment those counts are written, so the
position and the shard contents can never disagree. A manifest from before the cursor existed
resumes by count instead, which is slower but still correct.

    <corpus dir>/<name>_<sig>_00000.tokens     shard 0: RAW tokens, nothing else
    <corpus dir>/<name>_<sig>_00001.tokens     shard 1
    <corpus dir>/<name>_<sig>_index.json       the manifest: dtype, eos, per-shard counts

A shard is raw tokens with no header, which is what makes appending to it trivial and makes a
partial file a valid prefix of a whole one rather than a corrupt file. Everything a reader
needs is in the manifest; document boundaries are recovered from the EOS separators, lazily,
per shard, and only for the shards a caller actually asks about.

WHY SHARDS. A corpus can be 10 TB, and one file that size cannot be resumed, cannot be copied
incrementally, and loses everything to a single bad byte. ~100 MB pieces are re-fetchable,
verifiable one at a time, and readable in parallel.

WHY MEMORY-MAPPED. The corpus was once a Python list of lists of ints: a 28-byte object per
token plus 8 bytes of list slot, so 2 billion tokens would want ~70 GB of RAM to hold 4 GB of
data. Flat arrays paged in by the OS make resident memory a function of the batch, not of the
corpus -- the same choice as GPT-2's and nanoGPT's .bin, Megatron's indexed dataset and the
mmap'd shards of GPT-NeoX.

uint16 WHEN THE VOCABULARY ALLOWS IT. 50,259 tokens fit in 16 bits, so each costs 2 bytes
rather than 4 -- half the file, half the page cache, half the bytes per step. The dtype is
chosen from the vocabulary size and recorded, so a larger vocabulary widens to uint32 instead
of wrapping around into a silent corruption.

CONTIGUOUS, EOS-SEPARATED sampling, as in GPT-2, nanoGPT and Megatron: documents concatenated
with an end-of-sequence token between them, a sample being T+1 tokens from a random offset.
No padding, so every position carries a gradient. A window that begins near the end of a shard
is completed from the head of the next, so the shard size is invisible to training -- drawing
a shard first and then an offset inside it would mean the last T positions of every shard could
never begin a window, a bias nothing downstream could detect.
"""
import glob
import json
import os
import time

MAGIC = "zetagpt-tokens"
VERSION = 4                              # 4 adds "cursor"; a v3 manifest still resumes, by count
SHARD_BYTES = 100 * 1024 * 1024          # ~100 MB of tokens per shard
FLUSH_SECONDS = 5.0                      # how often the manifest catches up with the file


def dtype_for(vocab_size):
    """The narrowest integer type that can hold every id. numpy names, stored in the manifest."""
    return "uint16" if int(vocab_size) <= 65535 else "uint32"


def _itemsize(dtype):
    return 2 if dtype == "uint16" else 4


def _np():
    try:
        import numpy as np
        return np
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(f"[tokens] numpy is required for the token store ({e}).\n"
                         f"         pip install numpy") from e


# `stem` throughout is a FILE STEM, not a directory: <corpus dir>/<name>_<sig>
def index_path(stem):
    return f"{stem}_index.json"


def shard_path(stem, i):
    return f"{stem}_{i:05d}.tokens"


def _read_index(stem):
    try:
        with open(index_path(stem), "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_index(stem, sig, dtype, eos, vocab_size, shards, complete, cursor=None, packing=""):
    """Rewrite the manifest, through a .part rename so a reader never sees half of it.

    `cursor` is the producer's position AT THE MOMENT these counts were taken. The two are
    written together, in one file, by one rename: a cursor that could land in the manifest
    without the shard counts it belongs to would resume from the wrong document, and the
    corpus would silently gain or lose a stretch of text that nothing downstream could see."""
    tmp = index_path(stem) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": MAGIC, "version": VERSION, "sig": sig, "dtype": dtype,
                   "eos": int(eos), "vocab_size": int(vocab_size), "complete": bool(complete),
                   "n_tokens": sum(s["n_tokens"] for s in shards),
                   "n_docs": sum(s["n_docs"] for s in shards),
                   "cursor": cursor,
                   # WHAT PACKING PRODUCED THESE TOKENS, in words rather than as a hash. `sig`
                   # decides whether a stream may be reused; this says WHY when it may not, so
                   # a stage about to retokenise a corpus it already tokenised can name the
                   # setting that differs instead of starting silently from zero.
                   "packing": packing,
                   "shards": shards}, f, indent=1)
    os.replace(tmp, index_path(stem))


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(stem, sig, documents, eos, vocab_size, log=print, shard_bytes=SHARD_BYTES,
          resume=True, flush_seconds=FLUSH_SECONDS, packing=""):
    """Append documents to the shards at `stem`. Returns (n_tokens, n_docs).

    `documents` is a PRODUCER FACTORY: `documents(cursor, skip_docs)` returns an iterator of
    `(ids, cursor)` pairs, having already positioned itself. It is called once, here, with
    whatever the manifest recorded:

        cursor is not None    seek to it; nothing before it is read or tokenised
        cursor is None,       an older manifest: read from the top but do not TOKENISE the
        skip_docs > 0         first `skip_docs` documents -- still the expensive half saved
        neither               a fresh build

    Skipping used to happen here, on the store's side of the generator, which meant every
    skipped document had already been through the tokenizer before it was thrown away. The
    producer is the only place that can skip cheaply, so the position is handed to it.

    The producer yields a pair for EVERY document it packs, including one that tokenises to
    nothing. Empty ids are written as nothing but still advance the cursor, so the recorded
    position counts packed documents while the manifest's n_docs counts written ones, and the
    two are never required to agree -- an equality that held by luck would be a desynchronised
    resume the first time it did not.

    Every document is written as it arrives; the manifest, and the cursor with it, catch up
    every `flush_seconds`. A plain iterable is still accepted, and resumes by count."""
    import array
    dtype = dtype_for(vocab_size)
    isz = _itemsize(dtype)
    typecode = "H" if isz == 2 else "I"
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)

    idx = _read_index(stem)
    stale = idx is not None and idx.get("sig") != sig
    if stale or not resume:
        if idx is not None:
            why = "a different corpus or packing" if stale else "--no-resume"
            log(f"[tokens] discarding the existing shards ({why})")
        for f in glob.glob(f"{stem}_*.tokens") + glob.glob(f"{stem}_*.part"):
            try:
                os.remove(f)
            except OSError:
                pass
        idx = None

    shards = list(idx["shards"]) if idx else []
    if idx and idx.get("complete"):
        log(f"[tokens] already complete: {idx['n_tokens']:,} tokens in {len(shards)} shard(s)")
        return idx["n_tokens"], idx["n_docs"]

    # RESUME. The last shard may hold bytes written after the manifest was last updated; they
    # belong to documents the manifest does not count, so they are cut off. Truncating to a
    # recorded boundary is what makes "written" and "counted" agree again, and it is what lets
    # the cursor beside those counts be trusted.
    skip_docs = sum(s["n_docs"] for s in shards)
    cursor = idx.get("cursor") if idx else None
    if shards:
        last = shard_path(stem, len(shards) - 1)
        want = shards[-1]["n_tokens"] * isz
        have = os.path.getsize(last) if os.path.exists(last) else 0
        if have != want:
            with open(last, "r+b") as f:
                f.truncate(want)
            log(f"[tokens] trimmed {(have - want) / 1048576:.1f} MB of uncounted tail from "
                f"{os.path.basename(last)}")
        where = "seeking to the recorded position" if cursor else \
                "no cursor in this manifest: re-reading the corpus, but NOT re-tokenising it"
        log(f"[tokens] resuming after {skip_docs:,} documents in {len(shards)} shard(s); "
            f"{where}")

    # The producer positions itself ONCE, here. Everything after this point is new work.
    stream = documents(cursor, skip_docs) if callable(documents) else documents

    per_shard = max(shard_bytes // isz, 1)
    fh = None
    cur = shards[-1] if shards and shards[-1]["n_tokens"] < per_shard else None
    if cur is None and shards:
        pass                                     # the last shard is full; a new one starts below
    last_flush = time.time()

    def open_shard():
        nonlocal fh, cur
        if cur is None:
            cur = {"file": os.path.basename(shard_path(stem, len(shards))),
                   "n_tokens": 0, "n_docs": 0}
            shards.append(cur)
        fh = open(os.path.join(os.path.dirname(stem) or ".", cur["file"]), "ab")

    def close_shard():
        nonlocal fh, cur
        if fh is not None:
            fh.flush(); os.fsync(fh.fileno()); fh.close(); fh = None
        cur = None

    try:
        for item in stream:
            # (ids, cursor) from a producer that can seek; a bare list from one that cannot,
            # in which case the position simply stays as it was and resume falls back to count
            ids, cursor = item if isinstance(item, tuple) else (item, cursor)
            if not ids:                          # tokenised to nothing: the cursor still moved
                continue
            if fh is None:
                open_shard()
                log(f"[tokens] shard {len(shards) - 1:05d} open")
            # WRITTEN NOW, not when the shard fills: a run killed at any moment leaves every
            # document it read on disk, and the manifest at most `flush_seconds` behind.
            array.array(typecode, ids + [eos]).tofile(fh)
            cur["n_tokens"] += len(ids) + 1
            cur["n_docs"] += 1
            now = time.time()
            if now - last_flush >= flush_seconds:
                fh.flush()
                _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)
                last_flush = now
            if cur["n_tokens"] >= per_shard:
                n = len(shards) - 1
                log(f"[tokens] shard {n:05d} done: {cur['n_tokens']:,} tokens, "
                    f"{cur['n_tokens'] * isz / 1048576:.1f} MB")
                close_shard()
                _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)
                last_flush = time.time()
    finally:
        if fh is not None:
            fh.flush(); os.fsync(fh.fileno()); fh.close(); fh = None
        # the manifest is brought up to date even when the loop raised, so an interrupted
        # build resumes from where it truly stopped rather than from the last periodic flush
        if shards:
            _write_index(stem, sig, dtype, eos, vocab_size, shards, False, cursor, packing)

    _write_index(stem, sig, dtype, eos, vocab_size, shards, True, cursor, packing)
    n_tok = sum(s["n_tokens"] for s in shards)
    n_doc = sum(s["n_docs"] for s in shards)
    log(f"[tokens] wrote {n_tok:,} tokens ({dtype}) in {len(shards)} shard(s) from "
        f"{n_doc:,} documents -> {stem}_*.tokens")
    return n_tok, n_doc


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
class _Shard:
    """One shard, memory-mapped lazily: a corpus of ten thousand shards must not open ten
    thousand files to be described, so the mapping is created on first touch."""

    def __init__(self, path, dtype, n_tokens, n_docs, eos, np):
        self.path, self.dtype, self.eos, self._np = path, dtype, eos, np
        self.n_tokens, self.n_docs = n_tokens, n_docs
        self._tokens = self._starts = None

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = self._np.memmap(self.path, dtype=self.dtype, mode="r",
                                           shape=(self.n_tokens,))
        return self._tokens

    @property
    def starts(self):
        """Where each document begins, RECOVERED from the EOS separators.

        Not stored: a manifest carrying an offset per document would be gigabytes for a corpus
        of this size, and the information is already in the stream -- every document ends with
        exactly one EOS. Derived per shard on first use, so previews touch a handful of shards
        and training, which never asks, touches none."""
        if self._starts is None:
            np = self._np
            ends = np.flatnonzero(np.asarray(self.tokens) == self.eos)
            self._starts = np.concatenate(([0], ends[:-1] + 1)) if len(ends) else np.zeros(0, "int64")
            self._ends = ends
        return self._starts


class TokenStream:
    """The shards at a stem, read as one contiguous stream.

    `len(stream)` is the DOCUMENT count, so stage logs and tables read as they did when the
    corpus was a list of documents; `stream.n_tokens` is the token count."""

    def __init__(self, path):
        np = _np()
        self._np = np
        self.path = path
        idx = _read_index(path)
        if not idx or idx.get("magic") != MAGIC:
            raise ValueError(f"{index_path(path)} is not a {MAGIC} manifest")
        if not idx.get("complete"):
            raise ValueError(f"{path} is an unfinished build "
                             f"({idx.get('n_tokens', 0):,} tokens so far)")
        self.head, self.sig, self.dtype = idx, idx.get("sig", ""), idx["dtype"]
        self.eos, self.n_tokens, self.n_docs = idx["eos"], idx["n_tokens"], idx["n_docs"]
        base = os.path.dirname(path) or "."
        self.shards = [_Shard(os.path.join(base, s["file"]), self.dtype, s["n_tokens"],
                              s["n_docs"], self.eos, np) for s in idx["shards"]]
        self._tok_start, self._doc_start, t, d = [], [], 0, 0
        for s in self.shards:
            self._tok_start.append(t); self._doc_start.append(d)
            t += s.n_tokens; d += s.n_docs
        self._tok_start.append(t); self._doc_start.append(d)

    def __len__(self):
        return self.n_docs

    @property
    def nbytes(self):
        return sum(os.path.getsize(s.path) for s in self.shards)

    def _locate(self, starts, i):
        import bisect
        return max(0, min(bisect.bisect_right(starts, i) - 1, len(self.shards) - 1))

    def read(self, start, n):
        """`n` tokens from global offset `start`, ACROSS SHARD BOUNDARIES."""
        np = self._np
        out, k = [], self._locate(self._tok_start, start)
        off = start - self._tok_start[k]
        while n > 0 and k < len(self.shards):
            s = self.shards[k]
            take = min(n, s.n_tokens - off)
            if take > 0:
                out.append(np.asarray(s.tokens[off:off + take], dtype="int64"))
                n -= take
            k += 1
            off = 0
        return out[0] if len(out) == 1 else np.concatenate(out)

    def doc(self, i):
        """Document `i` as a list of ids, WITHOUT its trailing eos."""
        k = self._locate(self._doc_start, i)
        s = self.shards[k]
        j = i - self._doc_start[k]
        starts = s.starts
        a = int(starts[j])
        b = int(s._ends[j])
        return s.tokens[a:b].tolist()

    def batch(self, batch, seq_len, generator=None, device=None):
        """One training batch: (ids, attn, rmask), each (batch, seq_len + 1).

        seq_len + 1 tokens are drawn so the loop's `logits[:, :-1]` against `ids[:, 1:]` yields
        exactly seq_len predictions. attn and rmask are all ones: a packed stream has no
        padding to hide and no prompt to exclude, so every position is a target.

        Offsets come from the SAME torch generator the training loop checkpoints, so a resumed
        run continues the data order it would have had."""
        import torch
        np = self._np
        n = seq_len + 1
        hi = max(self.n_tokens - n, 1)
        starts = torch.randint(0, hi, (batch,), generator=generator).tolist()
        ids = torch.from_numpy(np.stack([self.read(s, n) for s in starts]))
        if device is not None:
            ids = ids.to(device)
        ones = torch.ones_like(ids)
        return ids, ones, ones

    def describe(self):
        return (f"{self.n_tokens:,} tokens ({self.dtype}, {self.nbytes / 1048576:.1f} MB in "
                f"{len(self.shards)} shard(s), memory-mapped) in {self.n_docs:,} documents")


class MultiStream:
    """Several TokenStreams read as ONE corpus, without being merged on disk.

    A CORPUS KEEPS ITS OWN STREAM, ITS OWN SIGNATURE AND ITS OWN CACHE. That is the whole point:
    adding a second corpus to a training run must not touch the first one's tokens. Merging them
    into a single stream would change the signature of the result, discard the shards already
    built, and re-tokenise hours of text to arrive at the same tokens in a different file --
    which for a run already half-way through its budget is not a data change but a restart.

    Instead each corpus is built and cached exactly as if it were alone, and this reads across
    them. Add a corpus tomorrow and only the new one is tokenised; remove it and the other's
    cache is still there, still current, still keyed by the same signature it always had.

    SEQUENCES ARE DRAWN IN PROPORTION TO TOKENS, which is what concatenating the corpora would
    have given: a corpus of 4 GiB contributes twenty times as often as one of 0.2 GiB. The
    mixture is therefore a fact about the data on disk rather than a weighting somebody chose,
    and it is reproducible -- the draw uses the caller's generator, so a resumed run continues
    the same sequence of batches.

    The batch is shuffled after assembly so it is not ordered by corpus: gradients are averaged
    over a batch, and a batch whose first half is one corpus and second half another is fine for
    the mean but reads as a bug in any per-example diagnostic."""

    def __init__(self, streams, names=()):
        self.streams = list(streams)
        self.names = list(names) or [f"corpus {i + 1}" for i in range(len(self.streams))]
        self.n_tokens = sum(s.n_tokens for s in self.streams)
        self.n_docs = sum(s.n_docs for s in self.streams)
        self.eos = self.streams[0].eos
        self.dtype = self.streams[0].dtype
        self.nbytes = sum(s.nbytes for s in self.streams)

    def __len__(self):
        return self.n_docs

    def doc(self, i):
        """Document `i` as a list of ids, the corpora laid end to end in their listed order.

        A TokenStream has this and the sample previews use it (helpers.common.corpus_prompts
        tests for `doc` and `n_docs` together), so a corpus that had `n_docs` but no `doc`
        would take the branch meant for a list of records and fail on a type that is not a
        sequence. Nothing here reads a corpus by index during training -- batches are drawn,
        not indexed -- so this exists for the previews and for anything else that reasonably
        expects a corpus to be enumerable."""
        j = i
        for s in self.streams:
            if j < s.n_docs:
                return s.doc(j)
            j -= s.n_docs
        raise IndexError(f"document {i:,} past the end of {self.n_docs:,} documents")

    def batch(self, batch, seq_len, generator=None, device=None):
        """One training batch, drawn across the corpora. Same contract as TokenStream.batch."""
        import torch
        w = torch.tensor([float(s.n_tokens) for s in self.streams], dtype=torch.double)
        pick = torch.multinomial(w / w.sum(), batch, replacement=True, generator=generator)
        parts = []
        for k, s in enumerate(self.streams):
            n = int((pick == k).sum())
            if n:
                parts.append(s.batch(n, seq_len, generator=generator, device=None)[0])
        ids = parts[0] if len(parts) == 1 else torch.cat(parts)
        ids = ids[torch.randperm(ids.shape[0], generator=generator)]
        if device is not None:
            ids = ids.to(device)
        ones = torch.ones_like(ids)
        return ids, ones, ones

    def describe(self):
        share = [f"{n} {s.n_tokens / max(self.n_tokens, 1):.1%}"
                 for n, s in zip(self.names, self.streams)]
        return (f"{self.n_tokens:,} tokens across {len(self.streams)} corpora "
                f"({', '.join(share)}) in {self.n_docs:,} documents, each memory-mapped from "
                f"its own cache")


def open_if_current(path, sig):
    """The stream at the stem `path` when it is complete and its signature matches."""
    idx = _read_index(path)
    if not idx or idx.get("sig") != sig or not idx.get("complete"):
        return None
    try:
        return TokenStream(path)
    except Exception:                                          # noqa: BLE001
        return None
