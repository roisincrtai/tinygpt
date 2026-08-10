"""
scaling_laws/wikitext.py -- WikiText-103, read from disk, tokenised once, cached as raw ids.

    tokens(tok, log)  -> (train_ids, valid_ids)   two array('I') of token ids

WHERE IT COMES FROM. data/download/wikitext-103/wiki.{train,valid}.tokens, the files as
distributed. This module NEVER downloads: fetching is tools/download_data.py's job, and a
study that begins by silently pulling half a gigabyte is a study whose corpus is not pinned.
Place the files and re-run. Use the RAW variant (wikitext-103-raw-v1) -- the non-raw one has
already been through a word-level tokenizer and had rare words replaced by <unk>, which would
make a byte-level BPE measure the vocabulary someone else chose rather than the corpus.

data/download/ is git-ignored in full: the corpus is large and is not ours to redistribute.

TOKENISED WITH THE PIPELINE'S OWN BPE (checkpoints/bpe/bpe.json), not with a tokenizer of its
own. A scaling law is read off the loss in nats per token, and "a token" has to mean the same
thing here as everywhere else in the project or the numbers cannot be compared with anything.

THE CACHE EARNS ITS COMPLEXITY. The BPE is a pure-Python merge loop, so ~100M tokens is tens of
minutes even across every core; doing it once per grid point instead of once per corpus would
dominate the whole study. The cached file records the tokenizer's signature, so a rebuilt
vocabulary invalidates it rather than silently training on ids that no longer mean anything.
"""
import array
import json
import os

import default_config as config

DIR = os.path.join(config.DOWNLOAD_DIR, "wikitext-103")
HF_NAME, HF_CONFIG = "wikitext", "wikitext-103-raw-v1"
RAW = {"train": "wiki.train.tokens", "valid": "wiki.valid.tokens"}


def signature(tok):
    """Identifies the vocabulary that produced a cache. Any change invalidates it."""
    return f"wikitext103|v1|vocab={len(tok)}|merges={len(getattr(tok, 'merges', []) or [])}"


def _cache_path(split):
    return os.path.join(DIR, f"tokens.{split}.bin")


def _read_cache(split, sig):
    p = _cache_path(split)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rb") as fh:
            head = json.loads(fh.readline().decode("utf-8"))
            if head.get("sig") != sig:
                return None
            ids = array.array("I")
            ids.frombytes(fh.read())
    except Exception:                                              # noqa: BLE001
        return None
    return ids if len(ids) == head.get("n", -1) else None


def _write_cache(split, sig, ids):
    os.makedirs(DIR, exist_ok=True)
    p = _cache_path(split)
    tmp = p + ".tmp"
    with open(tmp, "wb") as fh:                    # atomic: never leave a half-written cache
        fh.write((json.dumps({"sig": sig, "n": len(ids)}) + "\n").encode("utf-8"))
        ids.tofile(fh)
    os.replace(tmp, p)


# --------------------------------------------------------------------------- #
# raw text
# --------------------------------------------------------------------------- #
def _raw_text(split, log):
    """The split's text, from the distributed files if present, else from HuggingFace."""
    local = os.path.join(DIR, RAW[split])
    if os.path.isfile(local):
        log(f"[scaling] {split}: reading {os.path.relpath(local, config.ROOT)}")
        return open(local, encoding="utf-8", errors="replace").read()
    # NOTHING HERE DOWNLOADS. Fetching is tools/download_data.py's job alone, so a study that
    # runs for hours cannot begin by silently pulling half a gigabyte of someone else's corpus.
    raise SystemExit(
        f"[scaling] WikiText-103 is not present locally, and this study does not download.\n"
        f"          Place the distributed files at\n"
        f"              {os.path.relpath(DIR, config.ROOT)}/{RAW['train']}\n"
        f"              {os.path.relpath(DIR, config.ROOT)}/{RAW['valid']}\n"
        f"          (the RAW variant, {HF_NAME}/{HF_CONFIG}: the non-raw one has already been\n"
        f"          word-tokenised and had rare words replaced by <unk>, which would make a\n"
        f"          byte-level BPE measure someone else's vocabulary rather than the corpus.)")


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #
_TOK = None          # per-worker tokenizer, set by _init_worker


def _init_worker(bpe_path):
    global _TOK
    from tokenizer import BPETokenizer
    _TOK = BPETokenizer.load(bpe_path)


def _encode_chunk(text):
    return _TOK.encode(text)


def _chunks(text, n_chars):
    """Split on line boundaries near every `n_chars`, so a chunk never cuts a word in half.

    Chunking at all is what makes the work parallelisable, and doing it on newlines is what
    makes the result identical to encoding the whole string: the BPE pre-tokenizer already
    treats a newline as a boundary, so no merge spans one."""
    out, start = [], 0
    n = len(text)
    while start < n:
        end = min(start + n_chars, n)
        if end < n:
            nl = text.find("\n", end)
            end = n if nl == -1 else nl + 1
        out.append(text[start:end])
        start = end
    return out


def _encode(text, tok, workers, log):
    """Token ids for one split, across `workers` processes (1 = in this process).

    Falls back to single-process on any failure to start a pool: a machine that forbids
    multiprocessing should be slow, not broken."""
    from helpers import progress
    pieces = _chunks(text, 4_000_000)
    log(f"[scaling] tokenizing {len(text) / 1e6:.1f}M characters in {len(pieces)} chunk(s) "
        f"across {workers} worker(s) -- this is the slow part, and it is cached")
    ids = array.array("I")
    if workers > 1:
        try:
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_init_worker,
                          initargs=(config.BPE_PATH,)) as pool:
                for part in progress(pool.imap(_encode_chunk, pieces), desc="[scaling] bpe",
                                     total=len(pieces)):
                    ids.extend(part)
            return ids
        except Exception as e:                                     # noqa: BLE001
            log(f"[scaling] worker pool unavailable ({e}); tokenizing in this process")
            ids = array.array("I")
    for piece in progress(pieces, desc="[scaling] bpe", total=len(pieces)):
        ids.extend(tok.encode(piece))
    return ids


def tokens(tok, workers=0, log=print):
    """(train_ids, valid_ids) as array('I'), from cache when the vocabulary is unchanged."""
    sig = signature(tok)
    if workers <= 0:
        workers = max(1, (os.cpu_count() or 2) - 1)
    out = {}
    for split in ("train", "valid"):
        cached = _read_cache(split, sig)
        if cached is not None:
            log(f"[scaling] {split}: {len(cached):,} tokens from cache "
                f"({os.path.relpath(_cache_path(split), config.ROOT)})")
            out[split] = cached
            continue
        ids = _encode(_raw_text(split, log), tok, workers, log)
        _write_cache(split, sig, ids)
        log(f"[scaling] {split}: {len(ids):,} tokens -> "
            f"{os.path.relpath(_cache_path(split), config.ROOT)}")
        out[split] = ids
    return out["train"], out["valid"]
