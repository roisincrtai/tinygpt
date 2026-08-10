"""
token_store.py -- the pre-tokenised corpus as a directory of memory-mapped SHARDS.

    build(dir, sig, documents, eos, vocab, log)    write it, streaming, once
    TokenStream(dir)                               open it; nothing is read into memory
    stream.batch(B, T, generator, device)          (ids, attn, rmask) for one training step

WHY MEMORY-MAPPED. The corpus used to be a Python list of lists of ints, built at startup and
held for the whole run. A Python int is a 28-byte object and a list of them costs another 8
bytes per slot, so 2 billion tokens -- the budget ZetaGPT-S is trained on -- would want
something like 70 GB of RAM to represent 4 GB of data. This module stores tokens as flat
arrays of machine integers and lets the OS page in the parts being read, which is what every
large-scale pretraining pipeline does (GPT-2's and nanoGPT's .bin, Megatron's indexed dataset,
the mmap'd shards of GPT-NeoX). Resident memory becomes a function of the batch, not of the
corpus.

WHY SHARDS, AND NOT ONE FILE. A corpus can be 10 TB. One file that size is a single point of
failure in every sense: an interrupted build starts again from nothing, a copy that fails at
99% copies nothing, no filesystem or transfer tool likes it, and it cannot be produced or
consumed in parallel. A directory of ~100 MB pieces is resumable (a completed shard is never
rebuilt), copyable incrementally, and checkable one piece at a time. The layout is

    <dir>/<name>_<sig>_00000.tokens     shard 0
    <dir>/<name>_<sig>_00001.tokens     shard 1
    <dir>/<name>_<sig>_index.json       the manifest: dtype, eos, per-shard counts

SAMPLING CROSSES SHARD BOUNDARIES. A window that begins near the end of a shard is completed
from the head of the next one, so the shard size is invisible to training. Drawing the shard
first and then an offset inside it would have been simpler and slightly wrong: the last T
positions of every shard would never start a window, which is a small bias that grows as the
shards get smaller and that nothing downstream could detect.

uint16 WHEN THE VOCABULARY ALLOWS IT. A byte-level BPE with 50,000 merges has 50,259 tokens,
which fits in 16 bits, so every token costs 2 bytes instead of 4 -- half the file, half the
page cache, half the bytes moved per step. The dtype is chosen from the vocabulary size and
recorded, so a larger vocabulary silently widens to uint32 rather than wrapping around, which
would be a silent corruption.

CONTIGUOUS, EOS-SEPARATED, the GPT-2/nanoGPT/Megatron arrangement: documents concatenated with
an end-of-sequence token between them, a sample being T+1 tokens from a random offset. There
is no padding, so every position in every batch carries a gradient; one document per row
padded to T would spend 20-40% of a batch on padding at T=512. A sample may straddle a
document boundary, and the EOS sitting there is precisely what teaches the model where a
document ends.

EACH SHARD IS SELF-DESCRIBING -- a JSON header line, then its document offsets, then its
tokens -- so a shard can be inspected alone, and the manifest is a convenience rather than the
only thing that knows the format.
"""
import json
import os
import shutil

MAGIC = "zetagpt-tokens"
VERSION = 2
SHARD_BYTES = 100 * 1024 * 1024          # ~100 MB of tokens per shard


def dtype_for(vocab_size):
    """The narrowest integer type that can hold every id. numpy names, stored in the header."""
    return "uint16" if int(vocab_size) <= 65535 else "uint32"


def _np():
    try:
        import numpy as np
        return np
    except ImportError as e:                                   # noqa: BLE001
        raise SystemExit(f"[tokens] numpy is required for the token store ({e}).\n"
                         f"         pip install numpy") from e


def _tag(path):
    """<name>_<sig> -- the stem every file in the directory shares."""
    return os.path.basename(os.path.normpath(path))


def index_path(dir_path):
    return os.path.join(dir_path, f"{_tag(dir_path)}_index.json")


def shard_path(dir_path, i):
    return os.path.join(dir_path, f"{_tag(dir_path)}_{i:05d}.tokens")


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def _write_shard(path, dtype, eos, sig, offsets, flat, np):
    """One shard: header line, its document offsets, then its tokens. Written to <path>.part
    and renamed, so an interrupted write can never leave a file a later run would trust."""
    head = {"magic": MAGIC, "version": VERSION, "dtype": dtype, "eos": int(eos),
            "sig": sig, "n_tokens": int(len(flat)), "n_docs": len(offsets) - 1}
    blob = (json.dumps(head) + "\n").encode("utf-8")
    pad = (-len(blob)) % 4096            # page-align the arrays; a misaligned mmap can cost a
    head["off_docs"] = len(blob) + pad   # copy on every read
    head["off_tokens"] = head["off_docs"] + 8 * len(offsets)
    blob = (json.dumps(head) + "\n").encode("utf-8")
    blob += b" " * (head["off_docs"] - len(blob))
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(blob)
        offsets.tofile(f)
        flat.tofile(f)
    os.replace(tmp, path)
    return head


def build(dir_path, sig, documents, eos, vocab_size, log=print, shard_bytes=SHARD_BYTES):
    """Write `documents` (an iterable of id lists) as a directory of token shards.

    STREAMING, and RESUMABLE. Documents are consumed from an iterator and flushed a shard at a
    time, so neither the corpus nor the output is ever fully in memory; and the manifest is
    rewritten after every completed shard, so a build killed at 8 TB restarts at 8 TB rather
    than at zero. Resuming skips the documents the finished shards already hold, which is
    sound because the document order is deterministic: the corpus files are sorted and each is
    packed in order.

    Returns (n_tokens, n_docs)."""
    import array
    np = _np()
    dtype = dtype_for(vocab_size)
    os.makedirs(dir_path, exist_ok=True)

    # what a previous, interrupted run of THIS signature already finished
    done = _read_index(dir_path)
    if done and done.get("sig") == sig and not done.get("complete"):
        shards = done["shards"]
        skip_docs = sum(s["n_docs"] for s in shards)
        log(f"[tokens] resuming: {len(shards)} shard(s) already written, "
            f"{skip_docs:,} documents; skipping ahead")
    else:
        shards, skip_docs = [], 0
        if done and done.get("sig") != sig:
            for f in os.listdir(dir_path):                     # a different corpus lived here
                if f.endswith((".tokens", ".tokens.part", "_index.json")):
                    try:
                        os.remove(os.path.join(dir_path, f))
                    except OSError:
                        pass

    # array("Q") for the offsets, not a list: a shard of five million documents costs 40 MB as
    # machine integers and ~200 MB as Python ints, and this is the one structure that scales
    # with the corpus rather than with the batch
    offsets = array.array("Q", [0])
    buf = array.array("H" if dtype == "uint16" else "I")
    seen, n_tokens_total, n_docs_total = 0, sum(s["n_tokens"] for s in shards), skip_docs
    per_shard = max(shard_bytes // (2 if dtype == "uint16" else 4), 1)

    def flush():
        nonlocal offsets, buf, n_tokens_total
        if not len(buf):
            return
        p = shard_path(dir_path, len(shards))
        head = _write_shard(p, dtype, eos, sig, offsets, buf, np)
        shards.append({"file": os.path.basename(p), "n_tokens": head["n_tokens"],
                       "n_docs": head["n_docs"]})
        n_tokens_total += head["n_tokens"]
        _write_index(dir_path, sig, dtype, eos, vocab_size, shards, complete=False)
        log(f"[tokens] shard {len(shards) - 1:05d}: {head['n_tokens']:,} tokens, "
            f"{os.path.getsize(p) / 1048576:.1f} MB")
        offsets = array.array("Q", [0])
        buf = array.array("H" if dtype == "uint16" else "I")

    for ids in documents:
        if not ids:
            continue
        seen += 1
        if seen <= skip_docs:                                  # already in a finished shard
            continue
        buf.extend(ids)
        buf.append(eos)                                        # the join the model learns
        offsets.append(len(buf))
        n_docs_total += 1
        if len(buf) >= per_shard:
            flush()
    flush()
    _write_index(dir_path, sig, dtype, eos, vocab_size, shards, complete=True)
    log(f"[tokens] wrote {n_tokens_total:,} tokens ({dtype}) in {len(shards)} shard(s) "
        f"from {n_docs_total:,} documents -> {dir_path}")
    return n_tokens_total, n_docs_total


def _write_index(dir_path, sig, dtype, eos, vocab_size, shards, complete):
    tmp = index_path(dir_path) + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": MAGIC, "version": VERSION, "sig": sig, "dtype": dtype,
                   "eos": int(eos), "vocab_size": int(vocab_size), "complete": bool(complete),
                   "n_tokens": sum(s["n_tokens"] for s in shards),
                   "n_docs": sum(s["n_docs"] for s in shards),
                   "shards": shards}, f, indent=1)
    os.replace(tmp, index_path(dir_path))


def _read_index(dir_path):
    try:
        with open(index_path(dir_path), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
class _Shard:
    """One shard, memory-mapped lazily: opening a stream of ten thousand shards must not open
    ten thousand files, so the mapping is created on first touch."""

    def __init__(self, path, np):
        self.path, self._np = path, np
        with open(path, "rb") as f:
            self.head = json.loads(f.readline().decode("utf-8").rstrip("\0 \n"))
        self.n_tokens = self.head["n_tokens"]
        self.n_docs = self.head["n_docs"]
        self._tokens = self._offsets = None

    @property
    def tokens(self):
        if self._tokens is None:
            self._tokens = self._np.memmap(self.path, dtype=self.head["dtype"], mode="r",
                                           offset=self.head["off_tokens"],
                                           shape=(self.n_tokens,))
        return self._tokens

    @property
    def offsets(self):
        if self._offsets is None:
            self._offsets = self._np.memmap(self.path, dtype="uint64", mode="r",
                                            offset=self.head["off_docs"],
                                            shape=(self.n_docs + 1,))
        return self._offsets


class TokenStream:
    """A directory of memory-mapped token shards, read as one contiguous stream.

    `len(stream)` is the DOCUMENT count, so the stage logs and tables read as they did when the
    corpus was a list of documents; `stream.n_tokens` is the token count."""

    def __init__(self, path):
        np = _np()
        self._np = np
        self.path = path
        idx = _read_index(path)
        if not idx or idx.get("magic") != MAGIC:
            raise ValueError(f"{path} is not a {MAGIC} directory")
        if not idx.get("complete"):
            raise ValueError(f"{path} is an unfinished build ({len(idx['shards'])} shards)")
        self.head, self.sig, self.dtype = idx, idx.get("sig", ""), idx["dtype"]
        self.eos, self.n_tokens, self.n_docs = idx["eos"], idx["n_tokens"], idx["n_docs"]
        self.shards = [_Shard(os.path.join(path, s["file"]), np) for s in idx["shards"]]
        # cumulative starts, so a global offset maps to (shard, offset) by one bisect
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
        """`n` tokens from global offset `start`, ACROSS SHARD BOUNDARIES.

        A window is completed from the following shard when it runs off the end of one, so the
        shard size is invisible to training. Drawing within a single shard would have been
        simpler and quietly biased: the last n positions of every shard could never begin a
        window."""
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
        a, b = int(s.offsets[j]), int(s.offsets[j + 1])
        return s.tokens[a:b - 1].tolist()

    def batch(self, batch, seq_len, generator=None, device=None):
        """One training batch: (ids, attn, rmask), each (batch, seq_len + 1).

        seq_len + 1 tokens are drawn so the loop's `logits[:, :-1]` against `ids[:, 1:]` yields
        exactly seq_len predictions. attn and rmask are all ones: a packed stream has no
        padding to hide and no prompt to exclude, so every position is a target -- which is the
        entire point of packing.

        Offsets come from the SAME torch generator the training loop checkpoints, so a resumed
        run continues the data order it would have had rather than starting a new one."""
        import torch
        np = self._np
        n = seq_len + 1
        hi = max(self.n_tokens - n, 1)
        starts = torch.randint(0, hi, (batch,), generator=generator).tolist()
        arr = np.stack([self.read(s, n) for s in starts])
        ids = torch.from_numpy(arr)
        if device is not None:
            ids = ids.to(device)
        ones = torch.ones_like(ids)
        return ids, ones, ones

    def describe(self):
        return (f"{self.n_tokens:,} tokens ({self.dtype}, {self.nbytes / 1048576:.1f} MB in "
                f"{len(self.shards)} shard(s), memory-mapped) in {self.n_docs:,} documents")


def open_if_current(path, sig):
    """The stream at `path` when it is complete and its signature matches, else None."""
    idx = _read_index(path)
    if not idx or idx.get("sig") != sig or not idx.get("complete"):
        return None
    try:
        return TokenStream(path)
    except Exception:                                          # noqa: BLE001
        return None
