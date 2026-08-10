"""
token_store.py -- the pre-tokenised corpus as ONE memory-mapped file per corpus.

    build(path, sig, documents, eos, vocab, log)   write it, streaming, once
    TokenStream(path)                              open it; nothing is read into memory
    stream.batch(B, T, generator, device)          (ids, attn, rmask) for one training step

WHY. The corpus used to be a Python list of lists of ints, built at startup and held for the
whole run. A Python int is a 28-byte object and a list of them costs another 8 bytes per
slot, so 2 billion tokens -- the budget ZetaGPT-S is trained on -- would want something like
70 GB of RAM to represent 4 GB of data. No amount of tuning fixes a representation that is
twenty times the size of what it represents. This module stores tokens as a flat array of
machine integers in a file and lets the OS page in the parts being read, which is what every
large-scale pretraining pipeline does (GPT-2's and nanoGPT's .bin, Megatron's indexed
dataset, the mmap'd shards of GPT-NeoX). Resident memory becomes a function of the batch,
not of the corpus.

uint16 WHEN THE VOCABULARY ALLOWS IT. A byte-level BPE with 50,000 merges has 50,259 tokens,
which fits in 16 bits with room to spare, so every token costs 2 bytes instead of 4. That is
not a micro-optimisation: it halves the file, halves the page cache the corpus occupies, and
halves the bytes moved per step. The dtype is chosen from the vocabulary size and recorded in
the header, so a larger vocabulary silently widens to uint32 rather than wrapping around --
a wrapped token id is a silent corruption that would train the model on the wrong words.

CONTIGUOUS, EOS-SEPARATED SAMPLING, the GPT-2/nanoGPT/Megatron arrangement. Documents are
concatenated with an end-of-sequence token between them and a sample is T+1 tokens from a
random offset. There is no padding at all, so every position in every batch carries a
gradient; the alternative -- one document per row, padded to T -- spends 20-40% of a batch
on padding at T=512 and lets short documents dominate the step count. A sample may straddle
a document boundary, and the EOS sitting at that boundary is precisely what teaches the model
where a document ends; a model that never saw the join would not know how to stop.

THE FILE IS SELF-DESCRIBING and there is only one of it -- no sidecar index:

    line 1        JSON header, newline-terminated: dtype, counts, eos, signature, offsets
    offsets       (n_docs + 1) x uint64, the start of each document in tokens
    tokens        n_tokens x uint16 (or uint32)

The offsets are kept even though contiguous sampling does not need them, because the corpus
statistics, the generation previews and any future document-aligned sampler all want to know
where documents begin, and recovering that by scanning several billion tokens for EOS is a
poor trade against 8 bytes per document.
"""
import json
import os
import shutil

MAGIC = "zetagpt-tokens"
VERSION = 1


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


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #
def build(path, sig, documents, eos, vocab_size, log=print, flush_every=1 << 20):
    """Write `documents` (an iterable of id lists) to `path` as one token stream.

    STREAMING, in both directions: documents are consumed from an iterator and appended to a
    scratch file in blocks, so neither the caller's corpus nor the output is ever fully in
    memory. Only the document offsets are held, at 8 bytes each -- a million documents costs
    8 MB, which is the one part of the corpus small enough to keep.

    Written to <path>.part and renamed at the end, so an interrupted build leaves no file that
    a later run would mistake for a complete one. Returns (n_tokens, n_docs)."""
    import array
    np = _np()
    dtype = dtype_for(vocab_size)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw_path, part_path = path + ".raw", path + ".part"
    # array("Q"), not a list: a corpus of five million documents would cost ~200 MB as Python
    # ints and costs 40 MB as machine integers. This is the one structure that scales with the
    # corpus rather than with the batch, so it is the one worth being careful about.
    offsets = array.array("Q", [0])
    n_tokens = 0
    buf = []

    def flush(fh):
        nonlocal buf
        if buf:
            np.asarray(buf, dtype=dtype).tofile(fh)
            buf = []

    try:
        with open(raw_path, "wb") as raw:
            for ids in documents:
                if not ids:
                    continue
                buf.extend(ids)
                buf.append(eos)                  # the join the model learns to produce
                n_tokens += len(ids) + 1
                offsets.append(n_tokens)
                if len(buf) >= flush_every:
                    flush(raw)
            flush(raw)

        head = {"magic": MAGIC, "version": VERSION, "dtype": dtype, "eos": int(eos),
                "vocab_size": int(vocab_size), "n_tokens": int(n_tokens),
                "n_docs": len(offsets) - 1, "sig": sig}
        blob = (json.dumps(head) + "\n").encode("utf-8")
        head["off_docs"] = len(blob)             # recorded after the header's own length is
        head["off_tokens"] = len(blob)           # known, so both are exact
        # the two offsets change the header's length, so it is re-serialised at a FIXED width:
        # pad to a multiple of 4096 and record that, which also aligns the token array to a
        # page boundary -- a misaligned mmap costs a copy on every read on some platforms
        blob = (json.dumps(head) + "\n").encode("utf-8")
        pad = (-len(blob)) % 4096
        head["off_docs"] = len(blob) + pad
        head["off_tokens"] = head["off_docs"] + 8 * len(offsets)
        blob = (json.dumps(head) + "\n").encode("utf-8")
        blob += b" " * ((head["off_docs"]) - len(blob))
        with open(part_path, "wb") as out:
            out.write(blob)
            offsets.tofile(out)                       # array("Q") is uint64 on every platform
            with open(raw_path, "rb") as raw:
                shutil.copyfileobj(raw, out, 1 << 22)
        os.replace(part_path, path)
    finally:
        for p in (raw_path, part_path):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    log(f"[tokens] wrote {n_tokens:,} tokens ({dtype}, "
        f"{os.path.getsize(path) / 1048576:.1f} MB) from {len(offsets) - 1:,} documents "
        f"-> {path}")
    return n_tokens, len(offsets) - 1


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
class TokenStream:
    """A memory-mapped token stream. Nothing is read until it is indexed.

    `len(stream)` is the DOCUMENT count, so the stage logs and tables read the same as they did
    when the corpus was a list of documents; `stream.n_tokens` is the token count."""

    def __init__(self, path):
        np = _np()
        self.path = path
        with open(path, "rb") as f:
            line = f.readline()
        self.head = json.loads(line.decode("utf-8").rstrip("\0 \n"))
        if self.head.get("magic") != MAGIC:
            raise ValueError(f"{path} is not a {MAGIC} file")
        h = self.head
        self.n_tokens, self.n_docs = h["n_tokens"], h["n_docs"]
        self.eos, self.sig, self.dtype = h["eos"], h.get("sig", ""), h["dtype"]
        self.tokens = np.memmap(path, dtype=h["dtype"], mode="r",
                                offset=h["off_tokens"], shape=(h["n_tokens"],))
        self.offsets = np.memmap(path, dtype="uint64", mode="r",
                                 offset=h["off_docs"], shape=(h["n_docs"] + 1,))

    def __len__(self):
        return self.n_docs

    @property
    def nbytes(self):
        return os.path.getsize(self.path)

    def doc(self, i):
        """Document `i` as a list of ids, WITHOUT its trailing eos."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.tokens[a:b - 1].tolist()

    def batch(self, batch, seq_len, generator=None, device=None):
        """One training batch: (ids, attn, rmask), each (batch, seq_len + 1).

        seq_len + 1 tokens are drawn so the loop's `logits[:, :-1]` against `ids[:, 1:]` yields
        exactly seq_len predictions. attn and rmask are all ones: a packed stream has no
        padding to hide and no prompt to exclude, so every position is a target -- which is the
        entire point of packing.

        Offsets are drawn from the SAME torch generator the training loop checkpoints, so a
        resumed run continues the data order it would have had rather than starting a new one."""
        import torch
        np = _np()
        n = seq_len + 1
        hi = max(self.n_tokens - n, 1)
        starts = torch.randint(0, hi, (batch,), generator=generator).tolist()
        # one numpy copy per row, then a single conversion: np.stack on memmap slices reads
        # only the pages touched, which is what keeps resident memory at O(batch * seq_len)
        arr = np.stack([np.asarray(self.tokens[s:s + n], dtype="int64") for s in starts])
        ids = torch.from_numpy(arr)
        if device is not None:
            ids = ids.to(device)
        ones = torch.ones_like(ids)
        return ids, ones, ones

    def describe(self):
        return (f"{self.n_tokens:,} tokens ({self.dtype}, {self.nbytes / 1048576:.1f} MB "
                f"memory-mapped) in {self.n_docs:,} documents")


def open_if_current(path, sig):
    """The stream at `path` when it exists and its signature matches, else None."""
    if not os.path.isfile(path):
        return None
    try:
        st = TokenStream(path)
    except Exception:                                          # noqa: BLE001
        return None
    return st if st.sig == sig else None
