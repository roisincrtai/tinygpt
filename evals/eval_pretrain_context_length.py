"""
evals/eval_pretrain_context_length.py -- does a PRETRAINED model actually use a long context,
and does it still work past the length it was trained at?

    python evals/eval_pretrain_context_length.py
    python evals/eval_pretrain_context_length.py --only zetagpt
    python evals/eval_pretrain_context_length.py --no-resume     # measure everything again
    python evals/eval_pretrain_context_length.py --plot_only

    outputs/plots/evaluation/pretrain_context_generalization.pdf
    outputs/eval/pretrain_context_generalization.json
    cache/evals/context_length/<model>.json                      # measured points, for resume

IT RESUMES, and it is built to. Attention materialises a (heads x T x T) matrix, so this sweep
is GUARANTEED to end in an allocation failure at some length -- that is what the last point of
each curve IS, not a fault -- and reaching it takes hours. Every measured point is written to
cache/evals/context_length/ the moment it exists, and a second run continues from there. The
cache key is the checkpoint's NAME and step, the corpus's NAME, the probe grids and the
sampling settings: no path, no device, no timestamp, so two machines agree on what a cached
point means. Change any of them and the file is discarded rather than merged into.

FOR BASE MODELS, WHICH MEANS EVERYTHING IS SCORED BY LIKELIHOOD AND NOTHING IS GENERATED. A
pretrained model cannot be asked a question -- it has never been shown that answering is what
one does with a question -- so every long-context benchmark built on instructions (LongBench,
InfiniteBench, RULER's QA half) measures instruction tuning and reports the absence of it as
the absence of context. The three probes here ask only what a language model is: how surprised
is it by these exact bytes.

    1. CONTEXT CURVE     the same target span, preceded by 128, 256, ... tokens of its own
                         document. If the curve keeps falling, the model is USING the extra
                         context; if it flattens at 512 while being fed 8,192, it merely
                         ACCEPTS it. This is the whole distinction, and one number cannot make
                         it -- only the same targets under a growing prefix can.

    2. COPY AT DISTANCE  a random passage, then d tokens of filler, then THE SAME PASSAGE. The
                         second copy is free information: a model that can reach back d tokens
                         pays almost nothing for it, one that cannot pays full price. No
                         semantics, no vocabulary effects, no corpus confound -- it isolates
                         retrieval from language modelling, which is exactly what a positional
                         scheme is or is not providing.

    3. PASSKEY           a five-digit key buried at a known depth, and the continuation
                         "The pass key is". Scored as the likelihood of the digits, never as
                         generated text: a weak model that puts 30% on the right key scores 30%
                         here and zero under exact match, and the graded number is the one that
                         still means something at 133M.

EVERY FIGURE IS BITS PER BYTE OVER AN IDENTICAL SPAN OF CHARACTERS. The models compared here
have four different tokenizers, and perplexity is per TOKEN: a tokenizer that cuts text into
fewer, longer pieces reports a lower perplexity for identical predictions. Bits per byte
divides that out -- the same sentence is the same number of bytes to everyone -- and it is the
only unit in which these four rows may be plotted on one axis.

WHAT THE COMPARISON MODELS ARE FOR. GPT-2 has a learned absolute position table of 1,024 rows
and there is no row 1,025: past it the model cannot be evaluated at all, and that wall is the
point. TinyLlama and Qwen2.5 use RoPE, trained at 2,048 and 32,768; they run past their
training length and degrade instead of stopping. ZetaGPT refers to no position index anywhere,
so the sweep continues until memory rather than architecture ends it. Three ways of answering
the same question, on one axis.

Nothing here is fetched by this script except the comparison weights, through transformers'
own cache (config.MODEL_DIR). A model that cannot be loaded is SKIPPED AND REPORTED, never
fatal: an evaluation that dies because one baseline is missing has wasted the others.
"""
import os
import sys

# RUNNABLE AS A SCRIPT, not only as a module. `python evals/<name>.py` is what a person
# types, and without this it fails on `import default_config` -- the project root is on
# the path when python is given a module (-m) and is NOT when it is given a file.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import json
import math
import random

import torch

import default_config as config
from helpers.kv_cache import Cache as KVCache
from helpers.utils import progress
from model import ZetaGPT
from tokenizer import BPETokenizer

# The comparison set. Base checkpoints only -- an instruct model would be a different
# experiment, and a chat model's likelihoods are shaped by a template that is not present here.
BASELINES = {
    "gpt2": "gpt2",                                   # 124M, learned absolute, 1,024
    "tinyllama": "TinyLlama/TinyLlama_v1.1",          # 1.1B, RoPE, 2,048
    "qwen": "Qwen/Qwen2.5-0.5B",                      # 0.5B, RoPE, 32,768
}
OUT_PDF = os.path.join(config.PLOT_DIR, "evaluation", "pretrain_context_generalization.pdf")
OUT_JSON = os.path.join(config.OUTPUT_DIR, "eval", "pretrain_context_generalization.json")

BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#d1701c", "#2e7d5b", "#8a8a8a"
COLOURS = {"zetagpt": BLUE, "gpt2": GREY, "tinyllama": ORANGE, "qwen": GREEN}
CACHE_DIR = os.path.join(config.CACHE_DIR, "evals", "context_length")

# WHAT AN ALLOCATOR SAYS WHEN IT CANNOT ALLOCATE, on each backend. There is no common exception
# type and no common message: CUDA raises OutOfMemoryError and says "out of memory", MPS raises
# a plain RuntimeError and says "Invalid buffer size: 19.44 GiB", and the CPU allocator says
# "can't allocate memory" or raises MemoryError outright.
#
# THIS LIST IS NOT COSMETIC. Attention materialises a (heads x T x T) matrix, so the sweep is
# GUARANTEED to hit the ceiling at some length -- that is what the last point of every curve
# IS. Failing to recognise the message turns the expected end of a curve into a traceback that
# loses the whole run, which is exactly what happened at 27k tokens on MPS: 8 x 27k x 27k x 4
# bytes is 19.44 GiB, reported in words this did not know.
_ALLOC_MSGS = ("out of memory", "invalid buffer size", "can't allocate", "cannot allocate",
               "failed to allocate", "not enough memory", "insufficient memory",
               "mps backend out of memory")


def is_alloc_error(e):
    return isinstance(e, MemoryError) or any(m in str(e).lower() for m in _ALLOC_MSGS)


# --------------------------------------------------------------------------- #
# the models, behind one interface
# --------------------------------------------------------------------------- #
def _finish(spec, model, log):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    spec["model"] = model
    spec["params"] = sum(p.numel() for p in model.parameters())
    log(f"[context] {spec['key']:<10} {spec['params'] / 1e6:7.1f}M params  "
        f"trained at {spec['train_len']:,}  "
        f"{'hard position limit ' + format(spec['max_pos'], ',') if spec['max_pos'] else 'no position limit'}")
    return spec


def spec_zetagpt(args, device, dtype, log):
    """The pipeline's own checkpoint, rebuilt from the architecture stored inside it."""
    path = args.checkpoint
    if not os.path.isfile(path):
        log(f"[context] zetagpt skipped: no checkpoint at {path}")
        return None
    if not os.path.isfile(config.BPE_PATH):
        log(f"[context] zetagpt skipped: no tokenizer at {config.BPE_PATH} (run stage 3)")
        return None
    tok = BPETokenizer.load(config.BPE_PATH)
    ck = torch.load(path, map_location="cpu")
    saved = ck.get("model_cfg") or dict(config.MODEL)
    cfg = {k: v for k, v in saved.items() if k != "vocab_size"}
    cfg.setdefault("pe", "ssm")
    model = ZetaGPT(vocab_size=len(tok), **cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"])
    return _finish({
        "key": "zetagpt",
        "name": f"ZetaGPT {cfg.get('n_layer')}L-{cfg.get('n_embd')}d (NoPE)"
                if cfg.get("pe", "ssm") == "ssm" else
                f"ZetaGPT {cfg.get('n_layer')}L-{cfg.get('n_embd')}d (pe={cfg.get('pe')})",
        "encode": lambda t: tok(t, add_special_tokens=False)["input_ids"],
        # MAX_POS = 0 MEANS "NO ARCHITECTURAL LIMIT", which is the claim under test and not a
        # missing value: there is no position table to run out of and no rotary base to
        # extrapolate, so the only ceiling is the memory of the machine.
        "max_pos": 0, "train_len": args.base or int(cfg.get("block_size", 512)),
        "step": ck.get("step"), "total": ck.get("total"),
        "checkpoint": os.path.relpath(path, config.ROOT),
        "positional": "none (state space recurrence)",
        # ONE CHUNK OF TOKENS, ATTENDING TO EVERYTHING ALREADY CACHED. helpers.kv_cache.Cache
        # carries four tensors per layer -- attention's keys and values, and the state space
        # module's convolution window and recurrence state -- because a block is SSM ->
        # attention -> FFN and the recurrence would otherwise be re-run over the whole prefix.
        "new_state": lambda: KVCache(len(model.blocks)),
        "forward_chunk": lambda x, st: (
            model.head(model.hidden_states(input_ids=x, cache=st)),
            st),
    }, model, log)


def spec_hf(key, repo, device, dtype, log):
    """A Hugging Face base model, or None. A baseline that cannot be loaded is reported and
    dropped -- the run continues with whichever rows it has."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: transformers not importable ({e})")
        return None
    try:
        tk = AutoTokenizer.from_pretrained(repo, cache_dir=config.MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=config.MODEL_DIR)
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: could not load {repo!r} ({e})")
        return None
    tk.model_max_length = int(1e9)          # it warns per sequence otherwise; long is the point
    model = model.to(device=device, dtype=dtype)
    c = model.config
    n_pos = int(getattr(c, "n_positions", 0) or getattr(c, "max_position_embeddings", 0) or 0)
    learned = str(getattr(c, "model_type", "")).startswith("gpt2")
    trunk = getattr(model, "transformer", None) or getattr(model, "model", None)
    return _finish({
        "key": key, "name": f"{repo.split('/')[-1]}",
        "encode": lambda t: tk(t, add_special_tokens=False)["input_ids"],
        # A LEARNED TABLE IS A WALL; RoPE IS A SLOPE. GPT-2 cannot be run past its rows at all,
        # so max_pos is a hard stop; a rotary model runs and degrades, so its training length
        # is recorded but nothing is refused.
        "max_pos": n_pos if learned else 0, "train_len": n_pos or 2048,
        "step": None, "total": None, "checkpoint": repo,
        "positional": "learned absolute table" if learned else "RoPE",
        # transformers' OWN incremental path: past_key_values in, past_key_values out. Same
        # reason as above -- the attention matrix becomes (chunk x T) instead of (T x T).
        "new_state": lambda: None,
        "forward_chunk": lambda x, st: (lambda o: (o.logits, o.past_key_values))(
            model(input_ids=x, past_key_values=st, use_cache=True)),
    }, model, log)


# --------------------------------------------------------------------------- #
# scoring: nats over an exact span, and the bytes that span covers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_ids(spec, pid, sid, device, chunk):
    """Score `sid` given `pid`, FEEDING THE SEQUENCE THROUGH THE MODEL'S KV CACHE IN CHUNKS.

    THIS IS WHAT MAKES 32k REACHABLE. A single forward over T tokens materialises attention as
    (heads x T x T): at T = 27,000 and 8 heads that is 19.44 GiB, which is the allocation that
    ended the first run. Fed `chunk` tokens at a time with the keys and values of everything
    before them already in the cache, the matrix is (heads x chunk x T) instead -- at chunk 512
    and T = 32,768, 0.5 GiB. LINEAR IN T RATHER THAN QUADRATIC, for identical arithmetic: each
    position still attends to the whole prefix, it is simply not asked to do so all at once.

    The result is exactly the single-pass result. Causal attention means a position's output
    depends only on positions at or before it, so splitting the sequence changes what is
    resident, never what is computed.

    Returns (nats, tokens, correct, T, ok)."""
    T = len(pid) + len(sid)
    if spec["max_pos"] and T > spec["max_pos"]:
        return None, len(sid), 0, T, False
    ids_all = pid + sid
    n_pre = len(pid)
    state = spec["new_state"]()
    total, correct = 0.0, 0
    try:
        for a in range(0, T, chunk):
            b = min(a + chunk, T)
            x = torch.tensor([ids_all[a:b]], device=device)
            logits, state = spec["forward_chunk"](x, state)
            # positions lo..hi-1 of THIS chunk have a next token AND lie in the span
            lo, hi = max(n_pre - 1, a), min(b, T - 1)
            if lo >= hi:
                del logits
                continue
            lg = logits[:, lo - a:hi - a].float()
            tgt = torch.tensor([ids_all[lo + 1:hi + 1]], device=device)
            total += float(-lg.log_softmax(-1).gather(-1, tgt.unsqueeze(-1)).sum())
            correct += int((lg.argmax(-1) == tgt).sum())
            del lg, tgt, logits
    except (RuntimeError, MemoryError) as e:
        # RUNNING OUT OF MEMORY IS THE EXPECTED END OF A CURVE, not a failure of the run: even
        # chunked, the cache itself grows with T. Anything that is NOT an allocation failure is
        # a real bug and is re-raised.
        if not is_alloc_error(e):
            raise
        empty_cache()
        return None, len(sid), 0, T, False
    return total, len(sid), correct, T, True


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def score_span(spec, prefix, span, device, chunk=512):
    """score_ids, for callers that hold text rather than ids.

    THE PREFIX AND THE SPAN ARE TOKENISED SEPARATELY and their ids concatenated. Tokenising the
    join would let a merge straddle the boundary, and then the tokens being scored would differ
    from one model to the next by more than the tokenizer -- the span itself would have moved.

    Returns (nats, bytes, n_tokens, n_correct, ctx_tokens, ok)."""
    nb = len(span.encode("utf-8"))
    sid = spec["encode"](span)
    if not sid:
        return 0.0, 0, 0, 0, 0, False
    pid = spec["encode"](prefix) if prefix else []
    n, t, c, T, ok = score_ids(spec, pid, sid, device, chunk)
    return n, nb, t, c, T, ok


def bpb(nats, nbytes):
    """Nats over a span -> BITS PER BYTE, the one unit four tokenizers share."""
    return None if nats is None or not nbytes else nats / math.log(2) / nbytes


# --------------------------------------------------------------------------- #
# probe 1 -- the same targets under a growing prefix
# --------------------------------------------------------------------------- #
def long_documents(data_dir, want_chars, log, limit=64):
    """The longest documents available locally, as text. NOTHING IS CONCATENATED.

    Stitching short documents into a long one manufactures a context whose distant half is
    genuinely irrelevant to the current token: no model can exploit it, every model scores the
    same, and the flat curve reads as success. So documents are used whole, the longest first,
    and if none is long enough the run SAYS SO rather than quietly measuring nothing."""
    from helpers import corpus_files
    from helpers.utils import _pack
    files = corpus_files(data_dir, config.PRETRAIN["exclude_dirs"])[:limit]
    if not files:
        raise SystemExit(f"[context] no corpus under {data_dir}\n"
                         f"          ./stage1_download_data.sh")
    docs = []
    for fp in progress(files, desc="[context] scanning for long documents", total=len(files)):
        try:
            for d in _pack(fp, 10 ** 9, config.PRETRAIN["text_column"]):
                if len(d) > 2000:
                    docs.append(d)
        except Exception:                                          # noqa: BLE001
            continue
        if len(docs) > 20000:
            break
    docs.sort(key=len, reverse=True)
    if not docs:
        raise SystemExit(f"[context] no documents over 2,000 characters under {data_dir}")
    log(f"[context] {len(docs):,} documents, longest {len(docs[0]):,} characters")
    if len(docs[0]) < want_chars:
        log(f"[context] WARNING: the longest document is {len(docs[0]):,} characters and the "
            f"sweep wants {want_chars:,}.")
        log(f"[context]          The context curve is therefore capped by the CORPUS, not by "
            f"the models. Point --data_dir at a corpus of long documents (PG-19, arXiv) for "
            f"the full sweep; probes 2 and 3 are synthetic and unaffected.")
    return docs


def probe_context_curve(spec, docs, args, device, log, live, cache):
    """The FIXED target span at CONTEXT LENGTHS OF 512, 1k, 2k, ... 32k TOKENS.

    THE X AXIS IS EXACTLY 2^n FOR EVERY MODEL. The document is tokenised once per model, the
    target is a fixed span of CHARACTERS -- the same text for everyone -- and the prefix is then
    truncated BY TOKEN ID to whatever leaves the total at L. Truncating ids rather than
    characters is what makes the length exact: a character budget lands on a different token
    count in each of four vocabularies, and the curves would not share an x value anywhere.

    FOUR CURVES OUT OF ONE SWEEP, because they are four readings of the same logits and they
    fail in different ways:

        bits per byte     comparable ACROSS models -- the only unit four tokenizers share
        perplexity        the number everyone quotes, exp(nats per token), per-tokenizer
        nats per token    the same thing before the exponential, where small differences read
        token accuracy    how often the argmax is right, which moves for different reasons than
                          perplexity does -- a model can sharpen a distribution it already had
                          roughly right and gain a lot of one and none of the other

    The target never moves. Only how much of the document precedes it changes, so a curve that
    falls is the model extracting something from text it did not have before, and a curve that
    flattens is the model having stopped reading.

    THE PDF IS REWRITTEN AFTER EVERY POINT, and every point is cached. A sweep to 32k is hours,
    and a result that exists only in memory until the end is a result an interruption throws
    away."""
    out = []
    tail = args.target_chars
    # tokenised ONCE per document, not once per length: the same ids are re-cut for every L,
    # and re-encoding 130k characters seven times is minutes spent on an answer already held
    encoded = []
    for d in docs[:args.documents]:
        if len(d) <= tail:
            continue
        cut = len(d) - tail
        encoded.append((spec["encode"](d[:cut]), spec["encode"](d[cut:]),
                        len(d[cut:].encode("utf-8"))))
    if not encoded:
        log(f"[context] {spec['key']:<10} curve  no document longer than the target span")
        return out

    for L in args.context_lengths:
        hit = cache.get(f"curve:{L}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   bpb {hit['bpb']:.4f}   "
                f"ppl {hit['ppl']:8.2f}   nats/tok {hit['nats_per_token']:.4f}   "
                f"acc {hit['acc']:.4f}   (cached)")
            live()
            continue
        nats = 0.0
        nbytes = n_tok = n_cor = ctx = n_ok = 0
        for pre_ids, span_ids, span_bytes in encoded:
            room = L - len(span_ids)
            if room < 0:                       # the target alone is longer than this context
                continue
            n, t, c, T, ok = score_ids(spec, pre_ids[-room:] if room else [], span_ids,
                                       device, args.chunk_tokens)
            if not ok:
                n_ok = 0
                break
            nats += n; nbytes += span_bytes; n_tok += t; n_cor += c; ctx += T; n_ok += 1
        npt = (nats / n_tok) if n_ok and n_tok else None
        rec = {"context_length": L, "documents": n_ok,
               "ctx_tokens": round(ctx / n_ok) if n_ok else L,
               "bpb": bpb(nats, nbytes) if n_ok else None,
               "nats_per_token": npt,
               "ppl": math.exp(npt) if npt is not None else None,
               "acc": (n_cor / n_tok) if n_ok and n_tok else None}
        out.append(rec)
        log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   "
            + (f"bpb {rec['bpb']:.4f}   ppl {rec['ppl']:8.2f}   "
               f"nats/tok {rec['nats_per_token']:.4f}   acc {rec['acc']:.4f}   "
               f"({n_ok} docs)" if rec["bpb"] is not None else "unreachable"))
        if rec["bpb"] is not None:
            cache.put(f"curve:{L}", rec)
        live()
        if rec["bpb"] is None:
            break
    return out


# --------------------------------------------------------------------------- #
# probe 2 -- copy at distance
# --------------------------------------------------------------------------- #
def _filler_words(rng, n):
    """Neutral filler. Common English words, so the filler is cheap to model and whatever the
    second copy costs is about RETRIEVAL and not about the filler being surprising."""
    bank = ("the of and to in a is that for it as with was on be by at this from or an "
            "which we can has been are but not they their more when there all would some "
            "such time other into over after where these two may than first also its").split()
    return " ".join(rng.choice(bank) for _ in range(n))


def probe_copy(spec, args, device, log, live, cache):
    """A passage, d words of filler, then THE SAME PASSAGE. Bits per byte on each copy.

    THE SECOND COPY IS FREE INFORMATION. A model that can reach back over the filler pays
    almost nothing for it; one that cannot pays what it paid the first time. The GAIN --
    first minus second -- is the retrieval signal, in bits per byte, and it is the cleanest
    statement of what a positional mechanism buys: no semantics, no vocabulary, no corpus.

    The passage is drawn from the same word bank, so the first copy is not itself hard; what
    is being measured is the DIFFERENCE, and a difficult passage would only add noise to both
    ends of it."""
    out = []
    for dist in args.copy_distances:
        hit = cache.get(f"copy:{dist}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
                f"1st {hit['bpb_first']:.3f} -> 2nd {hit['bpb_second']:.3f} bpb   "
                f"gain {hit['gain']:+.3f}   (cached)")
            live()
            continue
        first_n = first_b = second_n = second_b = 0.0
        reached = 0
        for trial in range(args.trials):
            rng = random.Random(args.seed * 1009 + dist * 31 + trial)
            passage = _filler_words(rng, args.copy_words)
            filler = _filler_words(rng, dist)
            n1, b1, _, _, _, ok1 = score_span(spec, "", passage, device)
            n2, b2, _, _, _, ok2 = score_span(spec, passage + " " + filler + " ", passage,
                                              device)
            if not (ok1 and ok2):
                break
            first_n += n1; first_b += b1; second_n += n2; second_b += b2; reached += 1
        rec = {"filler_words": dist, "trials": reached,
               "bpb_first": bpb(first_n, first_b) if reached else None,
               "bpb_second": bpb(second_n, second_b) if reached else None}
        rec["gain"] = (None if rec["bpb_first"] is None
                       else rec["bpb_first"] - rec["bpb_second"])
        out.append(rec)
        log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
            + (f"1st {rec['bpb_first']:.3f} -> 2nd {rec['bpb_second']:.3f} bpb   "
               f"gain {rec['gain']:+.3f}" if rec["gain"] is not None else "unreachable"))
        if rec["gain"] is not None:
            cache.put(f"copy:{dist}", rec)
        live()
        if rec["gain"] is None:
            break
    return out


# --------------------------------------------------------------------------- #
# probe 3 -- passkey, scored as a likelihood
# --------------------------------------------------------------------------- #
def probe_passkey(spec, args, device, log, live, cache):
    """A five-digit key at a known depth, scored as -log P(the digits | everything before).

    NEVER GENERATED, NEVER EXACT-MATCHED. A 133M model that puts a third of its mass on the
    right key has found it; exact match would call that a zero and the whole probe would read
    as a flat floor for every small model. The likelihood is graded, and it is the same
    quantity for a model that would have got it right."""
    out = []
    for total in args.passkey_lengths:
        for depth in args.passkey_depths:
            pid = f"key:{total}:{depth:.2f}"
            hit = cache.get(pid)
            if hit is not None:
                out.append(hit)
                log(f"[context] {spec['key']:<10} key    filler {total:>7,} words  depth "
                    f"{depth:.2f}  {hit['nats_per_key']:.2f} nats/key   (cached)")
                live()
                continue
            nats = nbytes = 0.0
            reached = 0
            for trial in range(args.trials):
                rng = random.Random(args.seed * 7919 + total * 17 + int(depth * 100) + trial)
                key = f"{rng.randrange(10000, 100000)}"
                before = int(total * depth)
                text = (_filler_words(rng, before)
                        + f" The pass key is {key}. Remember it. "
                        + _filler_words(rng, total - before)
                        + " The pass key is")
                n, b, _, _, _, ok = score_span(spec, text, f" {key}", device)
                if not ok:
                    break
                nats += n; nbytes += b; reached += 1
            rec = {"filler_words": total, "depth": depth, "trials": reached,
                   "bpb": bpb(nats, nbytes) if reached else None,
                   # per-key nats is the readable form: -log P over the five digits, so 0 is
                   # certainty and ln(10^5) = 11.5 is a model guessing uniformly at random
                   "nats_per_key": (nats / reached) if reached else None}
            out.append(rec)
            log(f"[context] {spec['key']:<10} key    filler {total:>7,} words  depth "
                f"{depth:.2f}  "
                + (f"{rec['nats_per_key']:.2f} nats/key (random = 11.51)"
                   if rec["nats_per_key"] is not None else "unreachable"))
            if rec["nats_per_key"] is not None:
                cache.put(pid, rec)
            live()
    return out


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
def _curve_panel(ax, models, field, xlabel, ylabel, title, logy=False, marker="o-"):
    """One metric of probe 1 against CONTEXT LENGTH IN TOKENS.

    x is each model's OWN token count, not a shared axis of characters, because perplexity,
    nats per token and accuracy are per-token quantities: plotting them against a character
    count would divide by one thing and index by another. What is held identical across models
    is the TARGET -- the same span of text is being predicted in every case -- and the x
    positions differ only because the tokenizers do."""
    drew = False
    for m in models:
        pts = [(r.get("context_length") or r["ctx_tokens"], r[field])
               for r in m["curve"] if r.get(field) is not None]
        if not pts:
            continue
        c = COLOURS.get(m["key"], GREY)
        ax.plot(*zip(*pts), marker, color=c, lw=1.5, ms=3.4, label=m["name"])
        drew = True
        # WHERE A MODEL STOPPED, and that it did. A line ending because the architecture
        # refused the length must LOOK like it ended -- an x, not a line run to the axis edge.
        if any(r.get(field) is None for r in m["curve"]):
            ax.plot(pts[-1][0], pts[-1][1], "x", color=c, ms=8, mew=1.8)
        # the training length of each model, where it falls inside the plotted range
        if m.get("train_len"):
            ax.axvline(m["train_len"], color=c, lw=0.7, ls=":", alpha=0.45)
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5)
    if drew:
        ax.legend(fontsize=6.5)


def figure(res, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 8.5, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    ax = axes.ravel()
    models = res["models"]

    # ---- row 1: four readings of the same sweep, against context length ---------------- #
    # DOTTED VERTICALS ARE TRAINING LENGTHS. Everything to the right of a model's own line is
    # extrapolation, which is the whole question, and it has to be visible without arithmetic.
    _curve_panel(ax[0], models, "bpb", "context length (tokens)",
                 "bits per byte on the fixed target",
                 "1. bits/byte -- the cross-model unit", logy=True)
    _curve_panel(ax[1], models, "ppl", "context length (tokens)", "perplexity",
                 "2. perplexity (per tokenizer: shape, not level)", logy=True, marker="s-")
    _curve_panel(ax[2], models, "nats_per_token", "context length (tokens)", "nats / token",
                 "3. nats per token", marker="d-")
    _curve_panel(ax[3], models, "acc", "context length (tokens)",
                 "next-token accuracy on the target",
                 "4. token accuracy", marker="v-")

    # ---- row 2: the two synthetic probes ----------------------------------------------- #
    for m in models:
        pts = [(r["filler_words"], r["gain"]) for r in m["copy"] if r.get("gain") is not None]
        if pts:
            ax[4].plot(*zip(*pts), "s-", color=COLOURS.get(m["key"], GREY), lw=1.5, ms=3.4,
                       label=m["name"])
    ax[4].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[4].set_xscale("log", base=2)
    ax[4].set_xlabel("filler words between the two copies")
    ax[4].set_ylabel("bits/byte saved on the second copy")
    ax[4].set_title("5. can it reach back?", loc="left", fontsize=9.5)
    if ax[4].get_legend_handles_labels()[0]:
        ax[4].legend(fontsize=6.5)

    for m in models:
        by_len = {}
        for r in m["passkey"]:
            if r.get("nats_per_key") is not None:
                by_len.setdefault(r["filler_words"], []).append(r["nats_per_key"])
        pts = sorted((k, sum(v) / len(v)) for k, v in by_len.items())
        if pts:
            ax[5].plot(*zip(*pts), "^-", color=COLOURS.get(m["key"], GREY), lw=1.5, ms=4.0,
                       label=m["name"])
    ax[5].axhline(math.log(1e5), color="k", lw=0.8, ls=":", alpha=0.6)
    ax[5].text(0.02, math.log(1e5), " uniform guess", va="bottom", fontsize=7,
               transform=ax[5].get_yaxis_transform())
    ax[5].set_xscale("log", base=2)
    ax[5].set_xlabel("filler words around the key")
    ax[5].set_ylabel("nats to name the key  (lower is better)")
    ax[5].set_title("6. can it retrieve a fact?", loc="left", fontsize=9.5)
    if ax[5].get_legend_handles_labels()[0]:
        ax[5].legend(fontsize=6.5)

    # NO SUPTITLE. The panels carry their own titles and a figure that goes into a paper is
    # captioned there; a banner across the top is duplication in the document and noise in the
    # figure. The progress note lives in the log, where a progress note belongs.
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # ATOMIC. The figure is rewritten after every measured point, so a reader may well have it
    # open; replacing it whole means they never catch it half-written.
    #
    # FORMAT IS PASSED EXPLICITLY, because matplotlib infers it from the EXTENSION and the
    # temporary file ends in `.part` -- which it rejects as an unknown format. Without this the
    # write raised every time, the guard around the redraw swallowed it, and the pdf silently
    # stopped updating after the first point. THE ONE THING THIS FEATURE IS FOR.
    #
    # AND THE FIGURE IS CLOSED IN A `finally`: closing after a save that raised leaks it, and a
    # hundred live updates then leak a hundred figures.
    tmp = out_path + ".part"
    try:
        fig.savefig(tmp, format="pdf", bbox_inches="tight")
    finally:
        plt.close(fig)
    os.replace(tmp, out_path)
    return out_path


class Cache:
    """Measured points, on disk, so a crashed sweep resumes instead of starting again.

        cache/evals/context_length/<model>.json

    THIS SWEEP ENDS IN AN ALLOCATION FAILURE BY DESIGN -- attention is quadratic in T, so every
    curve runs until the machine says no -- and it is hours long. Recomputing the first five
    points to reach the sixth, every time, is the whole cost of the run spent on arithmetic
    already done.

    THE SIGNATURE IS MACHINE-INDEPENDENT, and deliberately: the checkpoint's NAME and step, the
    corpus's NAME, the probe grids and the sampling settings -- never a path, never a device,
    never a time. Two machines with the same checkpoint and the same corpus must agree on what
    a cached point means, or the cache is a source of wrong numbers rather than fast ones. Any
    change to it discards the file rather than merging into it.

    ONLY SUCCESSFUL POINTS ARE KEPT. An allocation failure is a fact about THIS MACHINE, and
    writing it into a signature that claims to be machine-independent would tell a larger card
    that 27k tokens is unreachable because a laptop could not do it. Failures are re-tried on
    every run; that costs one forward pass, which is the point at which it fails anyway."""

    def __init__(self, key, signature, log, enabled=True):
        self.path = os.path.join(CACHE_DIR, f"{key}.json")
        self.sig, self.log, self.enabled, self.points = signature, log, enabled, {}
        if not enabled:
            if os.path.isfile(self.path):
                log(f"[context] {key}: --no-resume, ignoring {len(self._read()[1]):,} "
                    f"cached points")
            return
        old_sig, pts = self._read()
        if old_sig is None:
            return
        if old_sig != signature:
            differ = [k for k in signature if old_sig.get(k) != signature.get(k)]
            log(f"[context] {key}: cache discarded, {', '.join(differ) or 'signature'} changed")
            return
        self.points = pts
        if pts:
            log(f"[context] {key}: resuming, {len(pts):,} points already measured")

    def _read(self):
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            return d.get("signature"), d.get("points", {})
        except (OSError, ValueError):
            return None, {}

    def get(self, pid):
        return self.points.get(pid) if self.enabled else None

    def put(self, pid, rec):
        """Written THE MOMENT the point is measured, atomically. A cache flushed at the end
        would be empty in exactly the case it exists for."""
        self.points[pid] = rec
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"signature": self.sig, "points": self.points}, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def cache_signature(spec, args, corpus):
    """What a cached point depends on. NO PATH, NO DEVICE, NO TIME -- see Cache."""
    return {
        "model": spec["key"],
        "checkpoint": os.path.basename(str(spec["checkpoint"]).rstrip("/")),
        "step": spec.get("step"),
        "corpus": os.path.basename(str(corpus).rstrip("/")),
        "target_chars": args.target_chars, "documents": args.documents,
        "trials": args.trials, "copy_words": args.copy_words,
        "seed": args.seed, "dtype": args.dtype,
        "context_lengths": list(args.context_lengths),
        "copy_distances": list(args.copy_distances),
        "passkey_lengths": list(args.passkey_lengths),
        "passkey_depths": list(args.passkey_depths),
    }


class Live:
    """The figure and the json, rewritten AFTER EVERY MEASURED POINT.

    NEVER BUFFERED TO THE END. This sweep is four models over six context lengths plus two
    synthetic probes -- hours, most of it in forwards at 32k -- and a result that exists only in
    memory until the last line is a result that any interruption destroys. Writing costs about
    a second against minutes of compute, so the trade is not close.

    It is also the only way to WATCH the thing: open the pdf and the curves grow. A run that
    produces nothing until it finishes is indistinguishable from a run that has hung."""

    def __init__(self, res, json_path, pdf_path, log):
        self.res, self.json_path, self.pdf_path, self.log = res, json_path, pdf_path, log
        self.n = 0
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

    def __call__(self, note=""):
        self.n += 1
        self.res["progress"] = note or f"{self.n} points measured"
        tmp = self.json_path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.res, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())          # on disk, not in the page cache, before the rename
        os.replace(tmp, self.json_path)
        try:
            figure(self.res, self.pdf_path)
        except Exception as e:             # noqa: BLE001
            # A FIGURE MUST NEVER KILL A MEASUREMENT. The numbers are the expensive part and
            # they are already on disk; a broken draw is reported and the sweep continues.
            self.log(f"[context] [plot] skipped this update: {type(e).__name__}: {e}")


def summarise(res, log):
    from helpers import table
    rows = []
    for m in res["models"]:
        curve = [r for r in m["curve"] if r["bpb"] is not None]
        gains = [r["gain"] for r in m["copy"] if r.get("gain") is not None]
        keys = [r["nats_per_key"] for r in m["passkey"] if r.get("nats_per_key") is not None]
        far = max((r["filler_words"] for r in m["copy"]
                   if (r.get("gain") or 0) > 0.05), default=0)
        rows += [
            (m["name"], ""),
            ("  positional", m["positional"]),
            ("  parameters", f"{m['params'] / 1e6:.1f}M"),
            ("  trained at", f"{m['train_len']:,} tokens"),
            ("  best bpb / at", f"{min(r['bpb'] for r in curve):.4f} at "
                                f"{max(r.get('context_length') or 0 for r in curve):,} tokens"
                                if curve else "-"),
            ("  furthest copy", f"{far:,} filler words still worth over 0.05 bpb"),
            ("  best key", f"{min(keys):.2f} nats (uniform = 11.51)" if keys else "-"),
            ("", ""),
        ]
    table("context generalization", rows, out=log)


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="does a pretrained model use, and generalise, a long context")
    p.add_argument("--checkpoint", default=os.path.join(
        config.CHECKPOINT_DIR, "pretrain", "checkpoint_zetagpt-s_ssm_pretrain.pt"))
    p.add_argument("--only", action="append",
                   choices=["zetagpt"] + list(BASELINES),
                   help="evaluate just this model; repeatable. Default: all four")
    p.add_argument("--data_dir", default="",
                   help="corpus of LONG documents for probe 1; empty = the scheme's own "
                        "pretraining corpus. Nothing is concatenated, so a corpus of short "
                        "documents caps the sweep and says so")
    p.add_argument("--documents", type=int, default=8, help="documents averaged in probe 1")
    p.add_argument("--target_chars", type=int, default=2000,
                   help="the fixed target span every context length is scored on")
    p.add_argument("--chunk_tokens", type=int, default=512,
                   help="tokens per forward pass through the KV cache; attention is "
                        "(heads x chunk x T), so this and not the context length is what "
                        "bounds memory. Lower it if 32k still will not fit")
    p.add_argument("--trials", type=int, default=4, help="repeats per synthetic point")
    p.add_argument("--copy_words", type=int, default=48, help="length of the copied passage")
    p.add_argument("--base", type=int, default=0, help="ZetaGPT's training length; 0 = the "
                                                       "checkpoint's own block_size")
    p.add_argument("--gpu", default="auto")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT_PDF)
    p.add_argument("--json", default=OUT_JSON)
    p.add_argument("--plot_only", action="store_true", help="redraw from the saved json")
    # RESUME IS THE DEFAULT, because this sweep ends in an allocation failure by design and is
    # hours long: the normal case is a second run, and the normal thing for it to do is to
    # continue. --no-resume is how a measurement is repeated from nothing.
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help=f"reuse points already measured, from {os.path.relpath(CACHE_DIR, config.ROOT)}/ (default)")
    p.add_argument("--no-resume", "--no_resume", dest="resume", action="store_false",
                   help="measure every point again, ignoring the cache")
    a = p.parse_args()
    # CONTEXT LENGTHS ARE POWERS OF TWO IN TOKENS, so every model is measured at the same x
    # and the doubling is legible on a log axis. 32,768 is reachable only because the sequence
    # is fed through the KV cache in chunks (score_ids); a single pass at that length is a
    # 19 GiB attention matrix.
    a.context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    a.copy_distances = [16, 64, 256, 1024, 4096, 16384]
    a.passkey_lengths = [64, 256, 1024, 4096, 16384]
    a.passkey_depths = [0.1, 0.5, 0.9]
    return a


def main():
    args = parse_args()
    def log(m): print(m, flush=True)

    if args.plot_only:
        if not os.path.isfile(args.json):
            raise SystemExit(f"--plot_only: no results at {args.json}")
        with open(args.json, "r", encoding="utf-8") as fh:
            res = json.load(fh)
        summarise(res, log)
        log(f"[context] [figure] {figure(res, args.out)}")
        return

    from helpers import resolve_device
    device = resolve_device(args.gpu)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    log(f"[context] device={device} dtype={args.dtype} seed={args.seed}")

    want = list(args.only) if args.only else ["zetagpt"] + list(BASELINES)
    specs = []
    for key in want:
        s = (spec_zetagpt(args, device, dtype, log) if key == "zetagpt"
             else spec_hf(key, BASELINES[key], device, dtype, log))
        if s is not None:
            specs.append(s)
    if not specs:
        raise SystemExit("[context] no model could be loaded")

    data_dir = args.data_dir or config.PRETRAIN_CORPUS.get(
        config.PRETRAIN["model_scheme"], config.PRETRAIN_DIR)
    # a token is ~4 characters, so the longest context wants roughly 4x its tokens in text
    docs = long_documents(data_dir, max(args.context_lengths) * 4 + args.target_chars, log)

    # THE RESULT OBJECT EXISTS BEFORE THE FIRST MEASUREMENT, and every model's row is appended
    # to it as soon as that model starts -- so the figure shows a model in progress rather than
    # appearing only once it is finished.
    res = {"models": [], "corpus": os.path.relpath(data_dir, config.ROOT),
           "target_chars": args.target_chars, "trials": args.trials,
           "copy_words": args.copy_words, "dtype": args.dtype, "seed": args.seed,
           "progress": "starting"}
    live = Live(res, args.json, args.out, log)
    log(f"[context] writing {os.path.relpath(args.out, config.ROOT)} after every point")

    for i, spec in enumerate(specs, 1):
        log(f"\n[context] === {spec['name']}  ({i}/{len(specs)}) ===")
        m = {k: spec[k] for k in ("key", "name", "params", "train_len", "max_pos",
                                  "positional", "checkpoint", "step", "total")}
        m["curve"], m["copy"], m["passkey"] = [], [], []
        res["models"].append(m)
        tag = f"{spec['key']} ({i}/{len(specs)})"
        cache = Cache(spec["key"], cache_signature(spec, args, data_dir), log,
                      enabled=args.resume)
        m["curve"] = probe_context_curve(spec, docs, args, device, log,
                                         lambda: live(f"{tag}: context curve"), cache)
        m["copy"] = probe_copy(spec, args, device, log, lambda: live(f"{tag}: copy"), cache)
        m["passkey"] = probe_passkey(spec, args, device, log,
                                     lambda: live(f"{tag}: passkey"), cache)
        spec["model"] = None                       # let the weights go before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        live(f"{tag} complete")

    live("complete")
    log(f"\n[context] [json] {os.path.relpath(args.json, config.ROOT)}")
    summarise(res, log)
    log(f"[context] [figure] {os.path.relpath(args.out, config.ROOT)}")


if __name__ == "__main__":
    main()
