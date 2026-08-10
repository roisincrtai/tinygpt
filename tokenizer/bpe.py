"""
bpe_tokenizer.py -- a small, dependency-free byte-level BPE tokenizer for the from-scratch
ZetaGPT.

The token-bag synthetic datasets used a word-level vocabulary (models.WordTokenizer). The readable
pref_<theme>.json datasets are grammatical English sentences, so ZetaGPT needs sub-word units: a
word-level table would be huge and share nothing across inflections. This BPE is byte-level (like
GPT-2), so EVERY string encodes -- there is no unknown token to fall back to -- and it is
trained on the dataset corpus.

Interface parity with models.WordTokenizer so it is a drop-in for build_tokenizer/build_model:
    tok(text, add_special_tokens=False) -> {"input_ids": [...]}
    tok.pad_token_id / eos_token_id / unk_token_id / bos_token_id
    len(tok)                      vocabulary size (256 bytes + #merges + #specials)
    tok.decode(ids)               bytes back to text
    tok.save(path) / BPETokenizer.load(path)
    BPETokenizer.build(texts, num_merges=...)

ID LAYOUT, GPT-2's: the 256 raw bytes [0..255], then the merged symbols in merge order, then
the SPECIALS LAST. A symbol is a tuple of byte values; encoding applies the learned merges
greedily by rank, exactly as classic BPE.

The specials sit at the end because that is the only arrangement in which REGISTERING A NEW
ONE IS CHEAP. Put them first and every byte and every merge shifts by one the moment a token
is added, which invalidates every checkpoint and every cached token stream ever produced with
that vocabulary. Put them last and a new special is simply the next id: nothing already
trained changes meaning, and a model only needs its embedding matrix extended by a row --
which is exactly what `resize_token_embeddings` does elsewhere in the ecosystem.

The predefined set follows GPT-2's naming:

    <|endoftext|>   end of a document, and the bos/eos the pipeline generates against
    <|pad|>         padding, kept DISTINCT from eos

GPT-2 itself uses one token and pads with it. A separate pad costs one id and removes a real
ambiguity: with pad == eos, a padded batch and a batch of empty completions have identical
ids, and only the attention mask tells them apart -- so a mask bug becomes a silent training
error rather than a loud one.

There is no <unk>. A byte-level vocabulary cannot produce one: every string encodes into the
256 byte tokens at worst. `unk_token_id` remains as an alias for eos so callers expecting the
HuggingFace attribute keep working.

REGISTERING A TOKEN makes it an ATOM of the vocabulary:

    tok.register_special("<vp>")
    tok.encode("a <vp> b")        -> [.., <vp>, ..]   one id, never split into < v p >

which is what lets a later model attach a meaning to it -- a control marker, a role tag, a
modality boundary. Matching happens on the raw string before the byte-level pre-tokenizer, so
the token is never split and never merged with its neighbours. `encode_ordinary` is the
escape hatch that treats specials as ordinary text, and is the right call for untrusted input:
otherwise a scraped document containing the characters "<|endoftext|>" could insert a document
boundary of its own.
"""
import json
import os
import re
from collections import Counter

# pre-tokenizer: a run of non-space with its leading whitespace (GPT-2-style leading space), or a
# run of whitespace on its own. Splitting first keeps merges from spanning word boundaries.
_PAT = re.compile(r"\s*\S+|\s+")


class BPETokenizer:
    # The predefined set. EOS first so that adding a special later never moves it -- the one
    # id the generation loops, the packer and every checkpoint agree on.
    EOS_TOKEN = "<|endoftext|>"
    PAD_TOKEN = "<|pad|>"
    SPECIALS = [EOS_TOKEN, PAD_TOKEN]
    LAYOUT = "specials-last"          # written into the file; see load()

    def __init__(self, merges, build_history=None, specials=None):
        # merges: ordered list of (sym_a, sym_b), each sym a tuple of byte ints (priority = index)
        self.merges = [(tuple(a), tuple(b)) for a, b in merges]
        # per-merge training dynamics (empty unless this tokenizer was just build()-ed or loaded
        # from a file that carried them); see build() for the recorded fields.
        self.build_history = list(build_history or [])
        # signature of the corpus this vocabulary was trained on, when known: it is what
        # lets a later run confirm the corpus is unchanged and CONTINUE adding merges.
        self.corpus_sig = None
        self.specials = list(specials or self.SPECIALS)
        self._rebuild()

    def _rebuild(self):
        """(Re)derive every id table. Called on construction and after a special is registered,
        so there is ONE description of the layout rather than two that can disagree."""
        self.sym2id, nxt = {}, 0
        for b in range(256):                                 # 0..255: the raw bytes
            self.sym2id[(b,)] = nxt; nxt += 1
        for a, b in self.merges:                             # then the merges, in rank order
            self.sym2id[a + b] = nxt; nxt += 1
        self.n_base = nxt                                    # first special id
        self.special2id = {s: self.n_base + i for i, s in enumerate(self.specials)}
        self.id2special = {i: s for s, i in self.special2id.items()}
        self.n_special = len(self.specials)
        self.id2sym = {i: s for s, i in self.sym2id.items()}
        self.ranks = {(a, b): i for i, (a, b) in enumerate(self.merges)}
        self._size = self.n_base + self.n_special
        self.eos_token_id = self.special2id[self.EOS_TOKEN]
        self.pad_token_id = self.special2id.get(self.PAD_TOKEN, self.eos_token_id)
        self.bos_token_id = self.eos_token_id                # no separate BOS; reuse EOS
        # A byte-level vocabulary has no unknown token -- every string encodes into the 256
        # byte tokens at worst. The attribute is an alias so callers written against the
        # HuggingFace interface keep working rather than raising AttributeError.
        self.unk_token_id = self.eos_token_id
        # Recogniser for the specials, LONGEST FIRST so that when one token is a prefix of
        # another the longer one wins -- with <|im|> and <|im_end|> both registered, the
        # shorter must not claim the first four characters of the longer.
        self._special_re = (re.compile("|".join(re.escape(t) for t in
                                                sorted(self.specials, key=len, reverse=True)))
                            if self.specials else None)

    def register_special(self, token):
        """Add a special token and return its id; registering an existing one returns it
        unchanged, so this is safe to call repeatedly.

        The new id is the next one after the current vocabulary, so NOTHING already trained
        changes meaning. A model built before the call has an embedding matrix one row short:
        extend it (and the tied output head) before using the new id, exactly as
        resize_token_embeddings does. This is why the specials are last."""
        if token in self.special2id:
            return self.special2id[token]
        if not isinstance(token, str) or not token:
            raise ValueError("a special token must be a non-empty string")
        # From here on encode() RECOGNISES the token: "a <vp> b" yields one id for <vp>
        # instead of the subwords of "<", "vp", ">". That is the whole point -- an atom the
        # model can attach a meaning to, which it cannot do with pieces that also occur in
        # ordinary text. The cost is that any text containing the literal characters now
        # produces the token, so untrusted input should go through encode_ordinary.
        self.specials = list(self.specials) + [token]
        self._rebuild()
        return self.special2id[token]

    # ----- construction ----------------------------------------------------- #
    @classmethod
    def _pretok(cls, text):
        return _PAT.findall(text)

    @classmethod
    def build(cls, texts, num_merges=10000, checkpoint_path=None, checkpoint_every=250,
              log=print, monitor=None, plot_every=200, init_merges=None, init_history=None,
              init_sig=None):
        """Train byte-level BPE on `texts` (an iterable of strings).

        INCREMENTAL. The textbook loop recounts every adjacent pair in the whole corpus
        before each merge, which is O(corpus) per merge and quadratic overall -- hours on a
        corpus of this size. Here the pair counts are built ONCE and then maintained: a merge
        only touches the words that actually contain the merged pair, found through a
        pair -> word-index inverted index, and the best pair comes off a lazy max-heap
        instead of a full scan. Same greedy vocabulary, very different cost.

        Two further things make the inner loop cheap:

        * SYMBOLS ARE INTERNED TO INTEGERS. The natural representation -- a symbol is a
          tuple of byte values, a pair a tuple of two of those -- makes every dictionary
          operation hash a nested tuple, and this loop is nothing but dictionary operations.
          Symbols are therefore integer ids into a table and a pair is packed into ONE
          integer, `a << 20 | b`, so the hot dicts are keyed by small ints.

        * THE HEAP KEEPS AN INVARIANT INSTEAD OF CHURNING. Every pair whose count changed
          during a merge is pushed ONCE, at its final count, after the merge is applied. A
          correct entry therefore exists for every live pair, which means a stale entry can
          be discarded on sight -- no re-pushing, and no repeated pushes of one pair inside
          a single merge. (Discarding stale entries WITHOUT that invariant is the classic
          bug: it silently removes a still-competitive pair from consideration.)

        RESUMABLE, IN TWO SENSES.

        * An INTERRUPTED build continues: `checkpoint_path` writes the merges learned so far,
          with their history, to `<path>.partial` every `checkpoint_every` merges and on
          exit, and a later call reloads and REPLAYS them onto the freshly counted corpus --
          seconds -- before carrying on where it stopped.

        * A FINISHED but SHORTER vocabulary is EXTENDED: pass its merges as `init_merges`
          (with `init_history`, and `init_sig` if it recorded one) and raising `num_merges`
          continues from merge N rather than restarting at zero. This is what happens when
          the configured merge budget is increased.

        Both are guarded by a signature over the word table, so a changed corpus can never
        silently continue someone else's vocabulary; whichever source carries more merges is
        the one replayed.

        `monitor(history, merge)` is called every `plot_every` merges, the same live-figure
        cadence every trainer uses, so the merge dynamics are watchable while the vocabulary
        is still being learned rather than only at the end.
        """
        import heapq
        from tqdm import tqdm

        SHIFT = 20                                   # pair key = a << SHIFT | b
        MASK = (1 << SHIFT) - 1

        words = Counter()
        for t in tqdm(list(texts), desc="BPE: counting words", unit="seg", dynamic_ncols=True):
            for chunk in cls._pretok(t):
                words[chunk] += 1
        items = [(w, c) for w, c in words.items() if w]
        sig = cls._corpus_signature(items)

        # ---- intern symbols: id <-> bytes ---- #
        sym = [bytes([b]) for b in range(256)]       # ids 0..255 are the raw bytes
        wordsyms, freqs = [], []
        for w, c in items:
            enc = w.encode("utf-8")
            if enc:
                wordsyms.append(list(enc))           # byte values ARE their symbol ids
                freqs.append(c)

        n_special = len(cls.SPECIALS)
        total_bytes = sum(sum(len(sym[s]) for s in ws) * c for ws, c in zip(wordsyms, freqs))
        tok_count = sum(len(ws) * c for ws, c in zip(wordsyms, freqs))

        # ---- pair counts + inverted index, built once ---- #
        counts, where = {}, {}
        for i in tqdm(range(len(wordsyms)), desc="BPE: indexing pairs", unit="word",
                      dynamic_ncols=True):
            s, c = wordsyms[i], freqs[i]
            for j in range(len(s) - 1):
                k = s[j] << SHIFT | s[j + 1]
                counts[k] = counts.get(k, 0) + c
                w = where.get(k)
                if w is None:
                    where[k] = {i}
                else:
                    w.add(i)
        heap = [(-c, k) for k, c in counts.items() if c > 0]
        heapq.heapify(heap)

        def apply_merge(a, b, ab):
            """Merge symbol pair (a,b) into `ab` everywhere; returns its corpus frequency."""
            nonlocal tok_count
            key = a << SHIFT | b
            hit, touched = 0, set()
            for i in where.pop(key, ()):
                s = wordsyms[i]
                if len(s) < 2:
                    continue
                out, j, n, ln = [], 0, 0, len(s)
                while j < ln:
                    if j < ln - 1 and s[j] == a and s[j + 1] == b:
                        out.append(ab); j += 2; n += 1
                    else:
                        out.append(s[j]); j += 1
                if not n:
                    continue
                c = freqs[i]
                for j in range(ln - 1):              # retract the word's old pairs
                    k = s[j] << SHIFT | s[j + 1]
                    counts[k] = counts.get(k, 0) - c
                    touched.add(k)
                    w = where.get(k)
                    if w is not None:
                        w.discard(i)
                wordsyms[i] = out
                for j in range(len(out) - 1):        # and post its new ones
                    k = out[j] << SHIFT | out[j + 1]
                    counts[k] = counts.get(k, 0) + c
                    touched.add(k)
                    w = where.get(k)
                    if w is None:
                        where[k] = {i}
                    else:
                        w.add(i)
                hit += n * c
                tok_count -= n * c
            # ONE push per changed pair, at its final count: this is the heap invariant
            for k in touched:
                v = counts.get(k, 0)
                if v > 0:
                    heapq.heappush(heap, (-v, k))
            return hit

        def intern(a, b):
            """Symbol id for the concatenation of `a` and `b`, creating it if new."""
            s = sym[a] + sym[b]
            sym.append(s)
            return len(sym) - 1

        merges, history = [], []
        # per-merge learning dynamics, recorded so bpe_dynamics can show how the vocabulary is
        # learned over the merge "steps" (bytes are conserved, so total_bytes is a constant
        # that turns the corpus token count into a compression ratio):
        #   merge          1-based merge index (the "step")
        #   pair_freq      corpus frequency of the pair just merged (the greedy objective)
        #   distinct_pairs number of distinct adjacent pairs still available before this merge
        #   vocab_size     3 specials + 256 bytes + merges so far
        #   tokens         total corpus length in tokens AFTER this merge
        #   new_token_len  #bytes in the symbol this merge created
        #   bytes_per_token total_bytes / tokens (compression, rises toward longer tokens)

        # ---- resume: replay whichever source carries the most merges ---- #
        done = cls._load_partial(checkpoint_path, sig)
        if init_merges:
            if init_sig is not None and init_sig != sig:
                log("BPE: the existing vocabulary was trained on a DIFFERENT corpus; "
                    "starting from scratch")
            else:
                if init_sig is None:
                    log("BPE: the existing vocabulary carries no corpus signature; "
                        "assuming it belongs to this corpus and extending it")
                if not done or len(init_merges) > len(done[0]):
                    done = (list(init_merges), list(init_history or []))
        if done:
            m_done, history = done
            m_done = m_done[:num_merges]             # a shorter budget: keep the prefix
            history = history[:num_merges]
            sid = {bytes(sym[i]): i for i in range(256)}
            for a_t, b_t in tqdm(m_done, desc="BPE: replaying checkpoint", unit="merge",
                                 dynamic_ncols=True):
                a, b = sid[bytes(a_t)], sid[bytes(b_t)]
                ab = intern(a, b)
                sid[sym[ab]] = ab
                apply_merge(a, b, ab)
                merges.append((a_t, b_t))
            log(f"BPE: resumed at merge {len(merges):,}/{num_merges:,}")
            if len(merges) >= num_merges:
                tok = cls(merges[:num_merges], build_history=history[:num_merges])
                tok.corpus_sig = sig
                return tok

        bar = tqdm(range(len(merges), num_merges), desc="BPE: learning merges", unit="merge",
                   initial=len(merges), total=num_merges, dynamic_ncols=True)
        for _ in bar:
            best = None
            while heap:                              # lazy heap: the invariant makes it safe
                nc, k = heapq.heappop(heap)          # to DISCARD a stale entry
                if counts.get(k, 0) == -nc and -nc > 0:
                    best = (k, -nc)
                    break
            if best is None:
                break
            k, freq = best
            if freq < 2:                             # no pair repeats; nothing left to gain
                break
            a, b = k >> SHIFT, k & MASK
            n_pairs = len(heap)
            ab = intern(a, b)
            if not apply_merge(a, b, ab):
                sym.pop()                            # the pair vanished under earlier merges
                continue
            merges.append((tuple(sym[a]), tuple(sym[b])))
            history.append({"merge": len(merges), "pair_freq": int(freq),
                            "distinct_pairs": n_pairs,
                            "vocab_size": n_special + 256 + len(merges),
                            "tokens": int(tok_count), "new_token_len": len(sym[ab]),
                            "bytes_per_token": total_bytes / max(tok_count, 1)})
            if hasattr(bar, "set_postfix"):
                bar.set_postfix(pair_freq=int(freq), vocab=n_special + 256 + len(merges),
                                bytes_per_tok=f"{total_bytes / max(tok_count, 1):.2f}")
            if (checkpoint_path and checkpoint_every > 0
                    and len(merges) % checkpoint_every == 0):
                cls._save_partial(checkpoint_path, sig, merges, history)
            if monitor and plot_every > 0 and len(merges) % plot_every == 0:
                try:
                    monitor(history, len(merges))    # live bpe_dynamics.pdf
                except Exception as e:               # plotting must never kill a long build
                    log(f"[plot] bpe: {e}")
        tok = cls(merges, build_history=history)
        tok.corpus_sig = sig                         # lets a later run verify and EXTEND it
        if checkpoint_path:                          # final state, marked complete
            cls._save_partial(checkpoint_path, sig, merges, history, done=True)
        return tok

    # ----- resumable-build checkpoint --------------------------------------- #
    @staticmethod
    def _corpus_signature(items):
        """Identifies the word table, so a partial build is only resumed on ITS corpus."""
        import hashlib
        h = hashlib.sha1()
        h.update(str(len(items)).encode())
        for w, c in sorted(items)[:20000]:           # a large, deterministic sample
            h.update(w.encode("utf-8", "replace")); h.update(str(c).encode())
        h.update(str(sum(c for _, c in items)).encode())
        return h.hexdigest()

    @staticmethod
    def _partial_path(path):
        return (path or "") + ".partial"

    @classmethod
    def _save_partial(cls, path, sig, merges, history, done=False):
        p = cls._partial_path(path)
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"sig": sig, "done": done, "n_merges": len(merges),
                       "merges": [[list(a), list(b)] for a, b in merges],
                       "build_history": history}, f)
        os.replace(tmp, p)                           # atomic: never a half-written resume file

    @classmethod
    def _load_partial(cls, path, sig):
        """(merges, history) from a matching partial build, or None."""
        p = cls._partial_path(path)
        if not path or not os.path.isfile(p):
            return None
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            return None
        if d.get("sig") != sig or not d.get("merges"):
            return None
        return ([(tuple(a), tuple(b)) for a, b in d["merges"]], d.get("build_history", []))

    # ----- encode / decode -------------------------------------------------- #
    def _merge_word(self, syms):
        while len(syms) >= 2:
            best_rank, bi = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, bi = r, i
            if bi < 0:
                break
            syms = syms[:bi] + [syms[bi] + syms[bi + 1]] + syms[bi + 2:]
        return syms

    def encode_ordinary(self, text):
        """Text -> ids, treating EVERY character as text. A registered special appearing in
        the string encodes as its literal bytes and cannot be produced. tiktoken's name for
        the same idea, and the right call for untrusted input -- a scraped document containing
        the characters "<|endoftext|>" should not be able to insert a document boundary."""
        ids = []
        for chunk in self._pretok(text):
            syms = self._merge_word([(b,) for b in chunk.encode("utf-8")])
            ids.extend(self.sym2id[s] for s in syms)         # base bytes + merges always present
        return ids

    def encode(self, text, split_specials=True):
        """Text -> ids, with registered specials RECOGNISED as single tokens.

        This is what registering a token is for: after `tok.register_special("<vp>")`, the
        string "a <vp> b" encodes with one id for <vp> rather than splitting it into "<", "vp"
        and ">". The token becomes an atom the model can learn a meaning for -- a control
        marker, a role tag, a modality boundary -- which it cannot if it arrives as four
        subwords that also occur in ordinary text.

        The specials are matched on the RAW STRING before the byte-level pre-tokenizer runs,
        so they are never split and never merged with their neighbours. Pass
        split_specials=False, or call encode_ordinary, to treat them as plain text."""
        if not split_specials or self._special_re is None:
            return self.encode_ordinary(text)
        ids, pos = [], 0
        for m in self._special_re.finditer(text):
            if m.start() > pos:
                ids.extend(self.encode_ordinary(text[pos:m.start()]))
            ids.append(self.special2id[m.group(0)])
            pos = m.end()
        if pos < len(text):
            ids.extend(self.encode_ordinary(text[pos:]))
        return ids

    def __call__(self, text, add_special_tokens=False, split_specials=True):
        ids = self.encode(text, split_specials=split_specials)
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        """Ids back to text. Specials carry no bytes; with skip_special_tokens=False they are
        rendered in their written form, which is what makes a transcript readable when the
        question is where the model actually emitted an end of text."""
        buf, out = bytearray(), []
        for i in ids:
            i = int(i)
            if i >= self.n_base:                             # a special: no bytes to add
                if not skip_special_tokens:
                    tokn = self.id2special.get(i)
                    if tokn:
                        out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                        out.append(tokn)
                continue
            s = self.id2sym.get(i)
            if s:
                buf.extend(s)
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    # ----- parity helpers --------------------------------------------------- #
    def __len__(self):
        return self._size

    def get_vocab(self):
        v = {bytes(s).decode("utf-8", "replace"): i for s, i in self.sym2id.items()}
        v.update(self.special2id)
        return v

    def convert_tokens_to_ids(self, t):
        if t in self.special2id:
            return self.special2id[t]
        ids = self.encode(t)
        return ids[0] if ids else self.unk_token_id

    # ----- persistence ------------------------------------------------------ #
    def vocab_records(self):
        """Full vocabulary as an id-ordered list of records, one per token:
            {"id": int, "token": readable-string, "bytes": [byte ints] | null}
        `token` is the UTF-8 decoding (invalid bytes shown as U+FFFD) for reading; `bytes` is the
        exact, lossless byte sequence (null for the specials, which carry no bytes)."""
        out = []
        for i in range(self._size):
            if i >= self.n_base:                             # the specials, at the end
                out.append({"id": i, "token": self.id2special[i], "bytes": None})
            else:
                sym = self.id2sym[i]
                out.append({"id": i, "token": bytes(sym).decode("utf-8", "replace"),
                            "bytes": list(sym)})
        return out

    def save(self, path):
        """Write the tokenizer as human-readable JSON. Keeps FULL information: the ordered `merges`
        (needed to encode) AND the complete `vocab` (readable token + exact bytes for every id) so
        the file can be inspected/printed/debugged without re-deriving anything."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        json.dump({"type": "byte_bpe",
                   "vocab_size": self._size,
                   "corpus_sig": self.corpus_sig,
                   "layout": self.LAYOUT,        # see load(): an older file means other ids
                   "specials": self.specials,
                   "vocab": self.vocab_records(),
                   "merges": [[list(a), list(b)] for a, b in self.merges],
                   "build_history": self.build_history},   # per-merge learning dynamics (may be [])
                  open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        # `merges` is the source of truth for encoding; `vocab` in the file is informational and
        # is rebuilt deterministically from the merges here.
        d = json.load(open(path, encoding="utf-8"))
        # A FILE WITHOUT `layout` PREDATES specials-last, and its ids mean something else: the
        # specials were 0..2 and every byte and merge sat three higher. Loading it under the
        # current layout would not fail -- it would silently return a tokenizer that decodes
        # every checkpoint's output shifted by three, which is far worse than refusing.
        layout = d.get("layout")
        if layout != cls.LAYOUT:
            raise SystemExit(
                f"[bpe] {path} was written with the old id layout "
                f"({layout or 'specials-first, pre-<|endoftext|>'}), which is not compatible: "
                f"its ids are offset by the specials.\n"
                f"      Delete it and re-run ./stage2_train_bpe_tokenizer.sh. Any checkpoint "
                f"or token stream built with it must be rebuilt too -- the cache directory is "
                f"named after the tokenizer's fingerprint, so that happens by itself.")
        tok = cls([(tuple(a), tuple(b)) for a, b in d["merges"]],
                  build_history=d.get("build_history", []),
                  specials=d.get("specials"))
        tok.corpus_sig = d.get("corpus_sig")
        return tok
