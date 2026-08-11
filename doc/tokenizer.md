# The ZetaGPT Tokenizer

**Files:** `tokenizer/bpe.py` (the algorithm), `tokenizer/run.py` (stage 3), `checkpoints/bpe/bpe.json` (the trained vocabulary)

A byte-level byte-pair encoding tokenizer, trained from scratch on the pipeline's own corpora.
Byte-level in the sense GPT-2 introduced: the base alphabet is the 256 possible byte values, so
**every string encodes**, in any script, with no unknown token to fall back on.

```
vocabulary = 256 bytes + 50,000 merges + 3 specials = 50,259
```

---

## 1. Id layout

Ids are laid out in GPT-2's order — bytes, then merges, then **specials last**:

| ids | contents |
|-----|----------|
| `0 … 255` | the raw byte values |
| `256 … 50,255` | the learned merges, in merge order (rank = position) |
| `50,256 …` | the special tokens |

The specials are last for one reason: it is the only arrangement in which **registering a new
token is cheap**. Were they first, adding one would shift all 256 byte ids and every merge, and
every checkpoint and cached token stream produced with that vocabulary would become meaningless.
Last, a new special is simply the next free id, and nothing already trained changes.

A `bpe.json` written before this layout is **refused** by `BPETokenizer.load`, rather than read.
Its ids are offset, so loading it would not raise — it would silently decode every existing
checkpoint's output shifted by three.

---

## 2. Special tokens

Three are predefined, named as GPT-2 names them:

| token | id | role |
|-------|-----|------|
| `<|endoftext|>` | 50,256 | end of a document; also the bos and the eos the generation loops stop on |
| `<|pad|>` | 50,257 | padding, deliberately **distinct** from eos |
| `<|unk|>` | 50,258 | reserved; nothing can produce it |

**Why pad is separate.** GPT-2 pads with its eos. Here they are different ids, which costs one
row and removes a real ambiguity: with `pad == eos`, a padded batch and a batch of empty
completions have identical ids and only the attention mask distinguishes them — so a masking
mistake becomes a silent training error instead of a loud one.

**Why unk exists but cannot occur.** A byte-level vocabulary never needs an unknown token: every
string decomposes into the 256 byte tokens at worst, so `encode` has no path that produces one.
It is kept as a real id because interfaces expect `unk_token_id` to exist, and because a reserved
slot costs one embedding row and is there if a later use wants it.

`bos_token_id` is `eos_token_id`; there is no separate beginning-of-sequence token.

---

## 3. Registering tokens

A registered token becomes an **atom** of the vocabulary: the tokenizer emits one id for it and
never breaks it into subwords.

```python
tok.register_special("<vp>")

tok.encode("a <vp> b")     # -> [.., 50259, ..]      one id
                           # not the subwords of "<", "vp", ">"
```

This is what lets a later model attach a meaning to it — a control marker, a role tag, a modality
boundary. It cannot do that with three pieces that also occur in ordinary text.

Or from configuration, applied on every load and build:

```python
# default_config.py
EXTRA_SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>"]
```

### What registering costs

Nothing that already exists changes.

| | effect |
|---|---|
| existing token ids | **unchanged** — the new id is appended after the vocabulary |
| `<|endoftext|>`, `<|pad|>`, `<|unk|>` | **unchanged** |
| a trained checkpoint | needs **one more embedding row**, not retraining |
| the token cache | **untouched** — see §6 |

The model side is handled by `ZetaGPT.resize_token_embeddings(n)`, called automatically by
`helpers.common.load_growing` when a checkpoint is loaded into a model whose vocabulary has since
grown. Existing rows are copied bit-for-bit; new rows are initialised at the **mean** of the
existing embeddings plus a little noise. The mean is where a token of no particular meaning
belongs — starting at zero or at `N(0, 0.02)` would place it outside the trained cloud, and the
first gradients would be spent dragging it back rather than teaching it anything. The noise breaks
the symmetry between several tokens registered at once. Weight tying between the embedding and the
output head is re-established explicitly, since they are one parameter.

A checkpoint from a **larger** vocabulary is refused. Tokens can be registered but not removed, so
that case means the tokenizer was rebuilt and its ids mean different words.

### Matching, and the escape hatch

Specials are matched on the **raw string, before** the pre-tokenizer runs, so a registered token is
never split and never merged with its neighbours. Where one is a prefix of another, the longer wins
(`<|im_end|>` beats `<|im|>`).

The consequence is that text containing those exact characters now produces the token. That is
correct for curated data, where the marker is written on purpose, and wrong for scraped text:

```python
tok.encode("... <|endoftext|> ...")            # -> contains a real eos
tok.encode_ordinary("... <|endoftext|> ...")   # -> just characters
```

`encode_ordinary` (tiktoken's name for the same idea) treats every special as ordinary text.
**The pretraining corpus is tokenised with `encode_ordinary`**, so a scraped page that merely
mentions `<|endoftext|>` — any page discussing GPT-2 does — cannot forge a document boundary in a
packed stream. The instruction data still uses `encode`, since that is where a registered marker is
deliberately written.

---

## 4. Pre-tokenization

Text is cut into chunks before any merge is applied, and **merges never span a chunk**, so this
rule decides what the vocabulary is able to learn at all.

```
_PAT = \s*\d  |  \s*[^\s\d]+  |  \s+
        ↑         ↑              ↑
        one       a run of       whitespace
        digit     non-space      on its own
                  non-digit
```

Every character is whitespace, a digit, or neither, so the three alternatives cover any string
exactly: **the chunks concatenate back to the input**, and nothing can be dropped.

### Digits are split one by one

Following LLaMA and DeepSeek. A number is always exactly its digits:

```
 1000     ->  [' 1', '0', '0', '0']
 1001     ->  [' 1', '0', '0', '1']
 999      ->  [' 9', '9', '9']
 1234567  ->  seven digit chunks
```

Left whole, BPE learns whichever number strings happen to be frequent in the corpus — `2015`,
` 100`, `42` — and the model must then do arithmetic over units that vary with the corpus rather
than with the mathematics: `1000` one token, `1001` two, and the relationship between them
invisible. Per-digit tokens make place value explicit and uniform. This matters here in particular
because the chain-of-thought stage is graded on a **verified** answer, so a tokenisation that hides
place value taxes exactly the capability that stage measures. The cost is a few more tokens per
number.

### Whitespace attaches to what follows it

```
'a  b'                    ->  ['a', '  b']
'def f():\n    return 1'  ->  ['def', ' f():', '\n    return', ' 1']
```

A run of indentation therefore stays inside one chunk and can become a single token. GPT-2's rule
peels runs of spaces apart one at a time, which is why it handles Python indentation poorly; this
is better on code.

Two consequences worth knowing:

- **Merges can cross a letter/punctuation boundary** inside a chunk — `f():` could become one
  token, where GPT-2 would keep `f` and `():` separate and get two reusable pieces. Bytes-per-token
  is therefore not directly comparable to GPT-2's at equal merge counts.
- **Token boundary effects.** Combining whitespace with the following word is what DeepSeek-V3
  does for compression, and V3 documents the resulting bias: a multi-line prompt that ends without
  a terminal line break tokenises differently at the join. Their mitigation — randomly splitting a
  proportion of such tokens during training — is not implemented here.

Whitespace, tabs, CR/LF, blank lines and trailing spaces all survive exactly; round-tripping is
byte-exact for every one of the 256 byte values, for CJK, for emoji including ZWJ sequences, and
for source code.

---

## 5. Training the vocabulary

Stage 3 (`./stage3_train_bpe_tokenizer.sh`) trains the merges on **every text the pipeline will
see**, so one vocabulary serves every stage: the pretraining corpus as documents, plus the
instruction data in its parsed form (each prompt, chosen and rejected string). The corpus is read
through the same packer the training stages use, so parquet, markdown and source code are decoded
properly rather than read as bytes of a container format.

The merge loop is incremental — pair counts are built once and maintained through a
pair → word inverted index, with the best pair taken from a lazy max-heap — rather than recounting
the corpus before every merge. Per-merge dynamics are recorded and drawn to
`outputs/plots/bpe/bpe_dynamics.pdf`.

`num_merges` is `default_config.BPE["num_merges"]`, 50,000 — the same count GPT-2 used. On a small
corpus the achievable number of merges can be lower, and the build simply stops when no pair
repeats.

---

## 6. Identity, and the token cache

Pre-tokenised corpora live in `cache/tokens/<tokenizer>/`, where the directory names the tokenizer:

```
cache/tokens/bpe_50256_<fingerprint>/…
             ↑     ↑
             |     8 characters of the git blob hash of the ENCODING
             256 + #merges
```

The fingerprint covers the **merges and the pre-tokenizer version**, and nothing else. Both parts
of that are deliberate:

- the **specials are excluded**, so registering `<|im_start|>` after pretraining does not throw
  away a corpus that has not changed by a single token. Nothing in a token stream depends on them:
  the corpus is tokenised with `encode_ordinary`, and `eos = 256 + #merges` regardless of how many
  specials exist;
- the **pre-tokenizer version is included**, because the same merges under a different chunking
  rule produce different ids. `PRETOK_VERSION` in `tokenizer/bpe.py` is bumped whenever `_PAT`
  changes.

It uses git's blob-hash construction so the value can be checked independently:

```bash
python -c "import json,hashlib; \
  m=json.load(open('checkpoints/bpe/bpe.json'))['merges']; \
  b=json.dumps({'pretok':2,'merges':m},separators=(',',':')).encode(); \
  print(hashlib.sha1(b'blob %d\0'%len(b)+b).hexdigest()[:8])"
```

Retraining the vocabulary therefore moves the cache to a new directory rather than invalidating the
old one in place, so two vocabularies coexist and switching back costs nothing.

---

## 7. Interface

Compatible with the HuggingFace tokenizer surface used elsewhere in the pipeline:

```python
tok(text, add_special_tokens=False)   # -> {"input_ids": [...]}
tok.encode(text)                      # specials recognised
tok.encode_ordinary(text)             # specials treated as text
tok.decode(ids, skip_special_tokens=True)
tok.register_special(token)           # -> the new id
len(tok)                              # vocabulary size
tok.pad_token_id / eos_token_id / unk_token_id / bos_token_id
tok.save(path)  /  BPETokenizer.load(path)
BPETokenizer.build(texts, num_merges=...)
```

`decode(ids, skip_special_tokens=False)` renders specials in their written form, which is what makes
a transcript readable when the question is where the model actually emitted an end of text.
