"""
token_store.py -- the pre-tokenised corpus as a set of memory-mapped SHARD FILES.

    build(stem, sig, documents, eos, vocab, log)   write it, appending as it goes
    TokenStream(stem)                              open it; nothing is read into memory
    stream.batch(B, T, generator, device)          (ids, attn, rmask) for one training step

NOTHING IS HELD BACK. Each document is appended to the open shard the moment it is tokenised,
and the manifest is rewritten every few seconds with the counts that are actually on disk.
Buffering a shard's worth of tokens before writing would mean a run killed after nine hours
left an empty directory -- which is exactly what a corpus of 10 TB cannot afford, and what
this module used to do. The invariant is: what has been read has been written.

RESUME IS THE DEFAULT. A build finds the manifest, truncates the last shard back to the last
recorded position (the bytes after it were written but not yet accounted for), and continues
from the document after the last one counted. This is sound because the document order is
deterministic -- the corpus files are sorted, each is packed in order -- so document N is
always the same document. Pass resume=False to start again.

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
VERSION = 3
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
        with open(index_path(stem), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_index(stem, sig, dtype, eos, vocab_size, shards, complete):
    """Rewrite the manifest, through a .part rename so a reader never sees half of it."""
    tmp = index_path(stem) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": MAGIC, "version": VERSION, "sig": sig, "dtype": dtype,
                   "eos": int(eos), "vocab_size": int(vocab_size), "complete": bool(complete),
                   "n_tokens": sum(s["n_tokens"] for s in shards),
                   "n_docs": sum(s["n_docs"] for s in shards),
                   "shards": shards}, f, indent=1)
    os.replace(tmp, index_path(stem))


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(stem, sig, documents, eos, vocab_size, log=print, shard_bytes=SHARD_BYTES,
          resume=True, flush_seconds=FLUSH_SECONDS):
    """Append `documents` (an iterable of id lists) to the shards at `stem`.

    Every document is written as it arrives; the manifest catches up every `flush_seconds`.
    Returns (n_tokens, n_docs)."""
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
    # recorded boundary is what makes "written" and "counted" agree again.
    skip_docs = sum(s["n_docs"] for s in shards)
    if shards:
        last = shard_path(stem, len(shards) - 1)
        want = shards[-1]["n_tokens"] * isz
        have = os.path.getsize(last) if os.path.exists(last) else 0
        if have != want:
            with open(last, "r+b") as f:
                f.truncate(want)
            log(f"[tokens] trimmed {(have - want) / 1048576:.1f} MB of uncounted tail from "
                f"{os.path.basename(last)}")
        log(f"[tokens] resuming after {skip_docs:,} documents in {len(shards)} shard(s)")

    per_shard = max(shard_bytes // isz, 1)
    seen = 0
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
        for ids in documents:
            if not ids:
                continue
            seen += 1
            if seen <= skip_docs:                # already on disk and counted
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
                _write_index(stem, sig, dtype, eos, vocab_size, shards, complete=False)
                last_flush = now
            if cur["n_tokens"] >= per_shard:
                n = len(shards) - 1
                log(f"[tokens] shard {n:05d} done: {cur['n_tokens']:,} tokens, "
                    f"{cur['n_tokens'] * isz / 1048576:.1f} MB")
                close_shard()
                _write_index(stem, sig, dtype, eos, vocab_size, shards, complete=False)
                last_flush = time.time()
    finally:
        if fh is not None:
            fh.flush(); os.fsync(fh.fileno()); fh.close(); fh = None
        # the manifest is brought up to date even when the loop raised, so an interrupted
        # build resumes from where it truly stopped rather than from the last periodic flush
        if shards:
            _write_index(stem, sig, dtype, eos, vocab_size, shards, complete=False)

    _write_index(stem, sig, dtype, eos, vocab_size, shards, complete=True)
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


def open_if_current(path, sig):
    """The stream at the stem `path` when it is complete and its signature matches."""
    idx = _read_index(path)
    if not idx or idx.get("sig") != sig or not idx.get("complete"):
        return None
    try:
        return TokenStream(path)
    except Exception:                                          # noqa: BLE001
        return None
