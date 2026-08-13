"""
evals/eval_pretrain_context_length.py -- does a PRETRAINED model actually use a long context,
and does it still work past the length it was trained at?

    python evals/eval_pretrain_context_length.py
    python evals/eval_pretrain_context_length.py --only zetagpt
    python evals/eval_pretrain_context_length.py --no-resume     # measure everything again
    python evals/eval_pretrain_context_length.py --plot_only

    outputs/plots/evaluation/pretrain_context_generalization.pdf
    outputs/eval/pretrain_context_generalization.json
    cache/evals/eval_context_length/<model>.json                      # measured points, for resume

MEMORY IS BUDGETED, NOT HOPED FOR. --vram_budget (20 GiB) is what the whole thing may occupy:
weights, the KV cache, and the pass in flight. The sequence goes through the cache in chunks,
and the CHUNK IS SOLVED FOR THE BUDGET AT EACH LENGTH -- smaller at 32k than at 512, because by
then the cache has taken the room. A length whose cache ALONE exceeds the budget is reported
unreachable before anything is allocated, rather than after the allocator says so.

IT RESUMES, and it is built to: the sweep runs to 32k and takes hours. Every measured point is written to
cache/evals/eval_context_length/ the moment it exists, and a second run continues from there. The
cache key is the checkpoint's NAME and step, the corpus's NAME, the probe grids and the
sampling settings: no path, no device, no timestamp, so two machines agree on what a cached
point means. Change any of them and the file is discarded rather than merged into.

FOR BASE MODELS, WHICH MEANS EVERYTHING IS SCORED BY LIKELIHOOD AND NOTHING IS GENERATED. A
pretrained model cannot be asked a question -- it has never been shown that answering is what
one does with a question -- so every long-context benchmark built on instructions (LongBench,
InfiniteBench, RULER's QA half) measures instruction tuning and reports the absence of it as
the absence of context. The three probes here ask only what a language model is: how surprised
is it by these exact bytes.

    1. CONTEXT CURVE     the same target span, preceded by 512, 1k, 2k ... 32k TOKENS of its
                         own document. If the curve keeps falling, the model is USING the extra
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

THE COMPARISON SET IS THE README'S TABLE -- every compact model in it that exists on the Hub,
smallest first: TinyStories 1M/8M/33M, Pythia-70M, GPT-2 small, SmolLM2-135M, Gemma 3 270M,
Qwen3-0.6B-Base, Qwen2.5-0.5B, TinyLlama v1.1. So the table stops being a literature survey
and becomes a measurement.

EVERY FIGURE IS NATS PER BYTE OVER AN IDENTICAL SPAN OF CHARACTERS. Those models have as many
tokenizers between them, and perplexity is per TOKEN: a vocabulary that cuts text into fewer,
longer pieces reports a lower perplexity for identical predictions. Nats per byte divides that
out -- the same sentence is the same number of bytes to everyone -- and it is the only unit in
which these rows may share an axis.

THREE ANSWERS TO ONE QUESTION, which is why the set is worth its cost. A LEARNED POSITION TABLE
IS A WALL: GPT-2 has 1,024 rows and no row 1,025, so past it the model cannot be evaluated at
all, and the sweep reports that rather than a bad number. RoPE IS A SLOPE: the rotary models
run past their training length and degrade. ZetaGPT refers to no position index anywhere, so
its sweep ends where memory ends and not where the architecture does.

THE CORPUS IS A HELD-OUT SPLIT, AND ONLY A HELD-OUT SPLIT. --data_dir defaults to
data/download/zetagpt-tiny_pretrain-corpus_wikitext103, of which only validation-*.parquet and
test-*.parquet are read; a corpus with no such files is REFUSED rather than scored. It is not
the pretraining corpus, deliberately: FineWeb-Edu ships as one undifferentiated set of shards
and every one of them is trained on, so scoring there would measure memorisation and report it
as generalisation. That the held-out text comes from a different distribution is a feature --
the claim under test is about LENGTH, not about domain.

DOCUMENTS ARE USED WHOLE AND NEVER CONCATENATED, and a document that cannot fill the context is
not scored at that length: stitching short articles into a long one manufactures a context
whose distant half cannot be exploited by anyone, and a curve drawn over it is flat for reasons
that have nothing to do with the models. Where no document is long enough the sweep stops and
says that the CORPUS ended it.

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
from helpers.utils import model_name, progress
from model import ZetaGPT
from tokenizer import BPETokenizer

# THE COMPARISON SET IS THE README'S TABLE, every row of it that exists on the Hub -- so the
# table stops being a literature survey and becomes a measurement. SMALLEST FIRST, because the
# cheap rows then land in the figure within minutes and a run stopped early still compares
# something.
#
# BASE CHECKPOINTS ONLY. An instruct model is a different experiment, and a chat model's
# likelihoods are shaped by a template that is not present here -- scoring raw prose against a
# model tuned to expect a chat header measures the header.
#
# `Baby GPT (character)` is in the table and NOT here: it is nanoGPT's character-level demo,
# trained per-reader rather than published, so there is nothing to fetch. Gemma 3 is gated on
# the Hub and will be skipped with a report unless `huggingface-cli login` has been run --
# which is exactly what a skipped baseline is for.
BASELINES = {
    "tinystories-1m": "roneneldan/TinyStories-1M",       # 3.7M,   learned,   512
    "tinystories-8m": "roneneldan/TinyStories-8M",       # 19.7M,  learned,   512
    "tinystories-33m": "roneneldan/TinyStories-33M",     # 68.5M,  learned,   512
    "pythia-70m": "EleutherAI/pythia-70m",               # 70.4M,  RoPE,      2,048
    "gpt2": "gpt2",                                      # 124M,   learned,   1,024
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",        # 134.5M, RoPE,      8,192
    "gemma3-270m": "google/gemma-3-270m",                # 268.1M, RoPE,      32,768
    "qwen3-0.6b": "Qwen/Qwen3-0.6B-Base",                # 0.6B,   RoPE,      32,768
    "qwen": "Qwen/Qwen2.5-0.5B",                         # 0.5B,   RoPE,      32,768
    "tinyllama": "TinyLlama/TinyLlama_v1.1",             # 1.1B,   RoPE,      2,048
}
OUT_PDF = os.path.join(config.PLOT_DIR, "evaluation", "pretrain_context_generalization.pdf")
OUT_JSON = os.path.join(config.OUTPUT_DIR, "eval", "pretrain_context_generalization.json")

# THE PALETTE. Eleven lines on one axis need to be told apart by MORE THAN HUE, so each model
# gets a colour, a marker and a dash pattern, and keeps all three in every panel -- a reader who
# has found a line in one panel finds the same line in the next without re-reading the legend.
#
# A COLORMAP WAS THE WRONG INSTRUMENT. Sampling `cividis` at eleven points is a SEQUENTIAL scale
# cut into categories: neighbouring models came out as neighbouring shades of the same
# blue-to-yellow ramp, indistinguishable in the middle and ugly throughout. A sequential
# colormap encodes an ordered quantity; these are eleven different objects, which is what a
# QUALITATIVE palette is for.
#
# The colours are Okabe-Ito's colourblind-safe set extended with four more of the same
# saturation, chosen to stay separable under deuteranopia and to survive being printed in
# greyscale -- where the markers and dashes carry the distinction on their own.
OWN = "#12386B"            # ZetaGPT: a deep navy nothing else is near, drawn heavier and on top
PALETTE = [
    "#E69F00",             # orange
    "#56B4E9",             # sky blue
    "#009E73",             # bluish green
    "#D55E00",             # vermillion
    "#CC79A7",             # reddish purple
    "#6A51A3",             # violet
    "#8C6D31",             # olive brown
    "#17A2A2",             # teal
    "#B03060",             # maroon
    "#5A5A5A",             # graphite
]
MARKERS = ["s", "^", "v", "D", "P", "X", "<", ">", "*", "h"]

# THE LINE STYLE IS NOT IDENTITY, IT IS MEANING. Solid up to a model's own training length,
# dashed past it, and a ring on the boundary. Every curve then says where it stops being
# interpolation and starts being extrapolation, without the reader holding eleven training
# lengths in mind or counting dotted verticals. Identity is carried by colour and marker, which
# is enough for eleven lines and leaves the dashes free to mean something.
SOLID, BEYOND = "-", (0, (4.5, 1.8))


def style_of(key, i):
    """Colour, marker and dash for one model -- the same three in every panel.

    ZetaGPT is solid, navy, round-markered and on top; a baseline takes the next entry of each
    cycle. Three channels rather than one means the figure survives a colourblind reader, a
    greyscale printer, and two lines that happen to cross."""
    if key == "zetagpt":
        return {"color": OWN, "marker": "o", "lw": 2.6, "ms": 5.0, "zorder": 5,
                "markeredgecolor": "white", "markeredgewidth": 0.6}
    j = i % len(PALETTE)
    return {"color": PALETTE[j], "marker": MARKERS[j], "lw": 1.3, "ms": 3.4, "zorder": 2}


CACHE_DIR = os.path.join(config.CACHE_DIR, "evals", "eval_context_length")

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
    # A CHECKPOINT THAT WILL NOT LOAD IS A SKIPPED MODEL, not a dead run. Every baseline is
    # already treated that way and there is no reason ours should be the one that takes the
    # evaluation down -- least of all when the likeliest cause is a TRAINING RUN STILL WRITING
    # IT, which is a matter of waiting rather than of anything being wrong.
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:                                        # noqa: BLE001
        size = os.path.getsize(path) if os.path.isfile(path) else 0
        import time
        age = time.time() - os.path.getmtime(path) if size else 0
        log(f"[context] zetagpt skipped: {os.path.relpath(path, config.ROOT)} would not load")
        log(f"[context]   {type(e).__name__}: {str(e).splitlines()[0]}")
        log(f"[context]   {size / 2**30:.2f} GiB, last written {age / 60:.1f} minutes ago")
        if age < 600:
            log(f"[context]   THAT IS RECENT. A pretraining run writes this file every "
                f"--checkpoint_every_steps; the write is atomic (helpers._atomic_torch_save "
                f"renames a temporary over it), so a complete file is always there -- but a "
                f"copy taken while the rename lands can still be short. Try again, or")
        # THE SNAPSHOT GOES IN THE PROJECT'S OWN CACHE, not in /tmp. cache/ is where this
        # project puts everything it can rebuild -- it is on the same filesystem as the
        # checkpoint, it survives a reboot, and it is covered by the same rules as the token
        # streams and these very results. /tmp is a machine's business, not a run's.
        snap = os.path.relpath(os.path.join(config.CACHE_DIR, "evals", "checkpoint_snapshot.pt"),
                               config.ROOT)
        log(f"[context]   evaluate a snapshot, which a live trainer cannot touch:")
        log(f"[context]       mkdir -p {os.path.dirname(snap)} && "
            f"cp {os.path.relpath(path, config.ROOT)} {snap}")
        log(f"[context]       python evals/eval_pretrain_context_length.py --checkpoint {snap}")
        log(f"[context]   If a copy fails the same way the file itself is damaged; "
            f"checkpoints/pretrain/ keeps only the newest, so the run must be resumed from "
            f"its history.")
        return None
    saved = ck.get("model_cfg") or dict(config.MODEL)
    cfg = {k: v for k, v in saved.items() if k != "vocab_size"}
    cfg.setdefault("pe", "ssm")
    model = ZetaGPT(vocab_size=len(tok), **cfg).to(device=device, dtype=dtype)
    model.load_state_dict(ck["model"])
    return _finish({
        "key": "zetagpt",
        # THE SCHEME'S NAME, not its dimensions. `24L-512d` is a description of a model, and
        # this project already has a NAME for that model -- zetagpt-s -- derived from
        # default_config.SCHEMES by helpers.model_name, which is also what every checkpoint,
        # history and figure is filed under. Two vocabularies for one object is one too many,
        # and the dimensions are in the startup table for anyone who wants them.
        "name": f"{model_name(cfg)}"
                + ("" if cfg.get("pe", "ssm") == "ssm" else f" (pe={cfg.get('pe')})"),
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
        "shape": {"n_layer": int(cfg["n_layer"]), "n_head": int(cfg["n_head"]),
                  # no grouped-query attention here: every head keeps its own K and V
                  "kv_per_token": 2 * int(cfg["n_embd"]), "vocab": len(tok)},
        "ensure": lambda T: None,      # nothing to grow: no position index exists
        "new_state": lambda: KVCache(len(model.blocks)),
        "forward_chunk": lambda x, st: (
            model.head(model.hidden_states(input_ids=x, cache=st)),
            st),
    }, model, log)


def extend_positions(model, target, log):
    """Stretch a learned position table to `target` rows, so a model with a wall can be run past
    it. Returns the new limit, or 0 when there is nothing to stretch.

    WHY NOT JUST REFUSE. A learned table stops the sweep at 1,024 and the curve ends there,
    which is true but uninformative: it says the model was BUILT for 1,024, not what happens to
    a learned scheme asked for more. Interpolating the table is the standard way to ask, and it
    is the comparison that matters -- ZetaGPT needs no such operation, and the point is what the
    operation COSTS.

    THE ROWS ARE INTERPOLATED, NOT APPENDED. Appending fresh rows would hand the model positions
    it has never seen, which is not an extension but a partial re-initialisation. Linear
    interpolation instead RESCALES the table: old row i lands at i x target / n, and the
    positions in between are averages of learned neighbours. Position 4,000 of 8,192 then means
    what position 500 of 1,024 meant -- the same fraction of the way through -- which is
    Position Interpolation's idea, applied to a table rather than to a rotation.

    THE CAUSAL MASK IS NOT REBUILT, AND MUST NOT BE. GPT-2 and GPT-Neo carry a lower-triangular
    `bias` buffer sized to the original length, and rebuilding it at the target is a (T x T)
    tensor -- at 32,768 that is 1.07 billion entries, 4 GiB PER LAYER, 32 GiB across eight of
    them, which is the very quadratic object the chunked scorer exists to avoid. Trying it is
    what produced

        RuntimeError: MPS backend out of memory (tried to allocate 4.00 GiB)

    The model is loaded with attn_implementation="sdpa" instead, where the causal mask is a flag
    to scaled_dot_product_attention and no dense buffer is consulted at all. WITHOUT SDPA THE
    TABLE IS NOT STRETCHED: an eager model would read its stale 2,048-row buffer and fail on
    shapes, so the extension is refused and reported rather than attempted.

    THIS CHANGES THE MODEL, and the figure says so: the label gains "positions interpolated".
    A curve drawn under a published model's name must be that published model."""
    import torch.nn as nn
    emb = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and name.split(".")[-1] in (
                "wpe", "embed_positions", "position_embeddings"):
            emb = (name, mod)
            break
    if emb is None:
        return 0
    name, mod = emb
    n_old, d = mod.weight.shape
    if target <= n_old:
        return n_old
    # A DENSE CAUSAL MASK IS REBUILT, NOT USED AS A REASON TO REFUSE. With sdpa there is none --
    # the mask is a flag to scaled_dot_product_attention. An eager path indexes a real buffer,
    # so it is rebuilt as BOOL (one byte an entry, not four) at the length being asked for.
    impl = str(getattr(getattr(model, "config", None), "_attn_implementation", "")).lower()
    dense = [(m, bn, b) for m in model.modules()
             for bn, b in m.named_buffers(recurse=False)
             if b is not None and b.dim() == 4 and b.shape[-1] == n_old]
    if dense and impl not in ("sdpa", "flash_attention_2"):
        for m, bn, b in dense:
            m.register_buffer(bn, torch.tril(torch.ones(
                target, target, dtype=torch.bool, device=b.device)).view(1, 1, target, target))
        log(f"[context]            {len(dense)} dense causal mask(s) rebuilt at {target:,} "
            f"as bool ({target * target * len(dense) / 2**30:.2f} GiB)")
    w = mod.weight.data.detach().float().t().unsqueeze(0)          # (1, d, n_old)
    w = torch.nn.functional.interpolate(w, size=target, mode="linear", align_corners=True)
    new = nn.Embedding(target, d).to(mod.weight.device)
    new.weight.data.copy_(w.squeeze(0).t().to(mod.weight.dtype))
    parent = model
    for part in name.split(".")[:-1]:
        parent = getattr(parent, part)
    setattr(parent, name.split(".")[-1], new)

    if hasattr(model, "config"):
        model.config.max_position_embeddings = target
        if hasattr(model.config, "n_positions"):
            model.config.n_positions = target
    log(f"[context]            position table {n_old:,} -> {target:,} rows by linear "
        f"interpolation (the causal mask is sdpa's, so none is materialised)")
    return target


def spec_hf(key, repo, device, dtype, log, args):
    """A Hugging Face base model, or None. A baseline that cannot be loaded is reported and
    dropped -- the run continues with whichever rows it has."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: transformers not importable ({e})")
        return None
    try:
        tk = AutoTokenizer.from_pretrained(repo, cache_dir=config.MODEL_DIR)
        # SDPA WHERE IT EXISTS. scaled_dot_product_attention takes the causal mask as a flag,
        # so no (T x T) buffer is built and no (T x T) score matrix is materialised either --
        # which is what makes a 32k forward possible at all. Older architectures without an
        # sdpa path fall back to eager, and extend_positions then declines to stretch them.
        try:
            model = AutoModelForCausalLM.from_pretrained(
                repo, cache_dir=config.MODEL_DIR, attn_implementation="sdpa")
        except Exception:                                          # noqa: BLE001
            model = AutoModelForCausalLM.from_pretrained(repo, cache_dir=config.MODEL_DIR)
    except Exception as e:                                        # noqa: BLE001
        log(f"[context] {key} skipped: could not load {repo!r} ({e})")
        return None
    tk.model_max_length = int(1e9)          # it warns per sequence otherwise; long is the point
    model = model.to(device=device, dtype=dtype)
    c = model.config
    n_pos = int(getattr(c, "n_positions", 0) or getattr(c, "max_position_embeddings", 0) or 0)
    # A LEARNED POSITION TABLE IS FOUND BY LOOKING FOR ONE, not by recognising a family name.
    # This tested `model_type.startswith("gpt2")`, which is false for GPT-Neo -- and TinyStories
    # is GPT-Neo, with a `wpe` table of 2,048 rows exactly like GPT-2's. It was therefore
    # treated as rotary, run out to 2,560, and died inside its own causal mask:
    #     RuntimeError: The size of tensor a (2048) must match the size of tensor b (2560)
    # A parameter named wpe / embed_positions / position_embeddings IS the table, whatever the
    # architecture is called, and every rotary model lacks all three.
    learned = any(("wpe" in n) or ("embed_positions" in n) or ("position_embeddings" in n)
                  for n, _ in model.named_parameters())
    native = n_pos                     # what it was TRAINED at, whatever is done to it after
    state = {"rows": n_pos}
    grows = bool(learned and args.extend_positions)

    def ensure(T):
        """Make the model able to take T tokens. ON DEMAND, and never a refusal.

        AN AUTOREGRESSIVE MODEL HAS NO INHERENT LENGTH. A learned position table is a table and
        a table can be stretched; refusing the length instead was a limit this script invented
        and then went on to measure. Every length asked for is attempted, and the table grows to
        meet it -- to the next power of two past T, so a sweep of doubling lengths interpolates
        once per doubling rather than once per point."""
        if not grows or T <= state["rows"]:
            return
        want = 1 << max(T - 1, 1).bit_length()
        state["rows"] = max(state["rows"], extend_positions(model, want, log) or state["rows"])
    trunk = getattr(model, "transformer", None) or getattr(model, "model", None)
    return _finish({
        "key": key,
        "name": repo.split("/")[-1] + (" (positions interpolated)" if grows else ""),
        "encode": lambda t: tk(t, add_special_tokens=False)["input_ids"],
        # NOTHING IS REFUSED FOR ITS LENGTH. A learned table is stretched by `ensure` when a
        # longer sequence arrives and a rotary model needs nothing, so the only things that can
        # end a curve are MEMORY and the CORPUS -- both of which are facts about the run rather
        # than limits this script imposed. --extend_positions 0 puts the wall back, for a run
        # that wants to see where a table would have stopped.
        #
        # TRAIN_LEN IS THE NATIVE WINDOW AND STAYS THERE. Stretching a table does not train a
        # model; reporting "trained at 32,768" of a model trained at 2,048 would put the
        # extrapolation marker in the wrong place and hide the entire finding.
        "max_pos": 0 if grows else (n_pos if learned else 0),
        "train_len": native or 2048, "ensure": ensure,
        # WHAT THE MODEL SAYS ABOUT ITSELF, kept even for the rotary ones. It is not enforced
        # there -- running past it is the measurement -- but it is what tells a failure beyond
        # this length apart from a failure inside it, which is the difference between a fact to
        # plot and a bug to raise. See score_ids.
        "declared_max": n_pos,
        "step": None, "total": None, "checkpoint": repo,
        "positional": "learned absolute table" if learned else "RoPE",
        # transformers' OWN incremental path: past_key_values in, past_key_values out. Same
        # reason as above -- the attention matrix becomes (chunk x T) instead of (T x T).
        "shape": {
            "n_layer": int(getattr(c, "num_hidden_layers", None) or getattr(c, "n_layer", 12)),
            "n_head": int(getattr(c, "num_attention_heads", None) or getattr(c, "n_head", 12)),
            # GROUPED-QUERY ATTENTION MAKES THE CACHE SMALLER THAN THE HEAD COUNT SUGGESTS:
            # Qwen2.5-0.5B has 14 query heads and 2 key/value heads, so its cache is a seventh
            # of what num_attention_heads would imply. Reading num_key_value_heads is the
            # difference between a usable estimate and a seven-fold overestimate.
            "kv_per_token": 2 * int(getattr(c, "num_key_value_heads", None)
                                    or getattr(c, "num_attention_heads", None)
                                    or getattr(c, "n_head", 12))
            * int((getattr(c, "hidden_size", None) or getattr(c, "n_embd", 768))
                  // max(int(getattr(c, "num_attention_heads", None)
                             or getattr(c, "n_head", 12)), 1)),
            "vocab": int(getattr(c, "vocab_size", 50257))},
        "new_state": lambda: None,
        "forward_chunk": lambda x, st: (lambda o: (o.logits, o.past_key_values))(
            model(input_ids=x, past_key_values=st, use_cache=True)),
    }, model, log)


# --------------------------------------------------------------------------- #
# scoring: nats over an exact span, and the bytes that span covers
# --------------------------------------------------------------------------- #
def plan_chunk(spec, T, budget_bytes, itemsize, cap, log=None):
    """How many tokens may go through one forward pass, so that the whole thing fits `budget`.

    THE CHUNK IS NOT A CONSTANT, because what it has to fit is not. Three things are resident
    while a chunk is scored, and only the first two grow with the CONTEXT:

        weights            n parameters x itemsize                  fixed
        the KV cache       2 x layers x T x d_kv x itemsize         LINEAR IN T, and unavoidable
        the chunk itself   heads x chunk x T x itemsize (scores)    linear in BOTH
                           + chunk x vocab x 4        (logits)

    So the budget left for a pass shrinks as T grows, and a chunk of 512 that is comfortable at
    8k is four times too large at 32k. Solving for the chunk:

        chunk = (budget - weights - cache) / (2 x heads x T x itemsize + 4 x vocab)

    with the 2 because the scores and their softmax are alive together. THE CACHE IS THE HARD
    FLOOR: at 32k it is gigabytes on its own, and when it alone exceeds the budget no chunk
    size helps -- 0 is returned and the length is reported unreachable BEFORE anything is
    allocated, rather than after the allocator says so.

    Returns (chunk, reason). `chunk` is 0 when nothing fits, and `reason` says WHICH THING RAN
    OUT -- which matters because they are not the same finding:

        "weights"   the model does not fit the budget AT ANY LENGTH. Nothing to do with
                    context; it would fail identically at 512. Reported once and the model
                    skipped, NOT drawn as a curve that ends early.
        "cache"     the KV cache for THIS length exceeds what is left. A genuine limit of the
                    length, and the point the curve should stop at.
        ""          it fits.

    CHUNKING IS A TECHNIQUE, NOT A CEILING. The chunk decides how much is resident during one
    pass; it decides nothing about how long a context may be. Conflating the two would let a
    budget too small for the WEIGHTS be drawn as a model that cannot handle long contexts."""
    sh = spec.get("shape") or {}
    n_layer = sh.get("n_layer", 12)
    n_head = sh.get("n_head", 12)
    kv_tok = sh.get("kv_per_token", 2 * 768)
    vocab = sh.get("vocab", 50257)
    weights = spec.get("params", 0) * itemsize
    cache = n_layer * T * kv_tok * itemsize
    per_token = 2 * n_head * T * itemsize + 4 * vocab
    if weights >= budget_bytes:
        if log is not None:
            log(f"[context] {spec['key']:<10} plan   weights alone are "
                f"{weights / 2**30:.2f} GiB against a {budget_bytes / 2**30:.2f} GiB budget: "
                f"this model does not fit AT ANY LENGTH")
        return 0, "weights"
    room = budget_bytes - weights - cache
    chunk = int(room // max(per_token, 1))
    if log is not None:
        log(f"[context] {spec['key']:<10} plan   T={T:,}  weights {weights / 2**30:.2f} + "
            f"cache {cache / 2**30:.2f} GiB, {room / 2**30:.2f} left -> chunk "
            f"{max(min(chunk, cap), 0) if chunk > 0 else 0}"
            + ("  (does not fit: the cache alone exceeds the budget)" if chunk <= 0 else
               f", pass ~{min(chunk, cap) * per_token / 2**30:.2f} GiB"))
    return (max(min(chunk, cap), 0), "" if chunk > 0 else "cache")


@torch.no_grad()
def score_ids(spec, pid, sid, device, chunk, desc=""):
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
    # GROW THE MODEL TO THE LENGTH rather than turn the length away. A no-op for anything with
    # no position index, and for a table already large enough.
    if spec.get("ensure"):
        spec["ensure"](T)
    if spec["max_pos"] and T > spec["max_pos"]:
        return None, len(sid), 0, T, False
    # THE CHUNK IS PLANNED FOR THIS LENGTH, not carried over from a shorter one. `chunk` may
    # arrive as a plan already made (the context curve makes one per length and logs it) or as
    # a cap to plan under.
    if isinstance(chunk, tuple):
        chunk, budget, itemsize = chunk
    else:
        budget, itemsize = 0, 4
    if budget:
        chunk, _why = plan_chunk(spec, T, budget, itemsize, chunk)
        if chunk <= 0:
            return None, len(sid), 0, T, False
    ids_all = pid + sid
    n_pre = len(pid)
    state = spec["new_state"]()
    total, correct = 0.0, 0
    # RULE1: A LOOP THAT CAN TAKE MORE THAN A SECOND REPORTS WHILE IT RUNS. At 32,768 tokens
    # and a chunk of 512 this is 64 forward passes, each one slower than the last because the
    # cache it attends to keeps growing -- minutes for ONE document, in silence, which is
    # indistinguishable from a hang. The bar is self-erasing (progress passes leave=False), so
    # it shows the work and leaves no trace between the lines of the result.
    steps = progress(range(0, T, chunk), desc=desc or f"[context] {spec['key']} {T:,} tok",
                     total=-(-T // chunk))
    try:
        for a in steps:
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
            n_done = hi - max(n_pre - 1, 0)
            if hasattr(steps, "set_postfix_str") and n_done:
                steps.set_postfix_str(f"{n_done:,} scored, "
                                      f"{total / max(n_done, 1):.3f} nats/tok")
            del lg, tgt, logits
    except (RuntimeError, MemoryError, IndexError) as e:
        # TWO EXPECTED ENDS OF A CURVE, and everything else is a bug.
        #
        # RUNNING OUT OF MEMORY: even chunked, the cache itself grows with T.
        #
        # AND RUNNING PAST WHAT THE MODEL DECLARES. An implementation may hold a causal mask or
        # a position table sized to max_position_embeddings and fail on its own shapes rather
        # than on ours -- which is a FACT ABOUT THE LENGTH and belongs in the figure. It counts
        # only BEYOND the declared length: the same error inside it is a real bug, and is
        # raised. That distinction is the whole guard; without it this would swallow the next
        # genuine shape error and report it as a context limit.
        dm = spec.get("declared_max") or 0
        beyond = dm and T > dm
        if not (is_alloc_error(e) or beyond):
            raise
        if beyond and not is_alloc_error(e):
            spec["_note"] = (f"refused {T:,} tokens past its declared "
                             f"{dm:,}: {type(e).__name__}")
        empty_cache()
        return None, len(sid), 0, T, False
    finally:
        if hasattr(steps, "close"):
            steps.close()
    return total, len(sid), correct, T, True


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


@torch.no_grad()
def score_span(spec, prefix, span, device, chunk=512, desc=""):
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
    n, t, c, T, ok = score_ids(spec, pid, sid, device, chunk, desc=desc)
    return n, nb, t, c, T, ok


def npb(nats, nbytes):
    """Nats over a span -> NATS PER BYTE, the one unit every tokenizer shares.

    NATS, NOT BITS. The loss is in nats, so dividing by the byte count and stopping there is
    the whole conversion; going on to divide by ln 2 introduces a second unit for no gain and
    makes every figure disagree with every log line. One unit, and it is the one the objective
    is already written in."""
    return None if nats is None or not nbytes else nats / nbytes


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
    from helpers.corpus_id import _split_token
    from helpers.utils import _pack
    # THE HELD-OUT SPLIT, AND NOTHING ELSE. This is an EVALUATION: scoring a model on the text
    # it was trained on measures how much of it the model memorised, and a context curve
    # measured that way says nothing about generalisation. corpus_files' `exclude_dirs` is the
    # wrong instrument -- it drops directories named test/valid/validation, which a corpus of
    # flat `<name>_00000.parquet` shards does not have, so it excluded NOTHING and the sweep
    # read the training set. The split is selected here by name instead, and its ABSENCE IS
    # FATAL rather than a silent fall back to training data.
    every = corpus_files(data_dir, ())
    files = [f for f in every
             if _split_token(os.path.basename(f)) in ("validation", "valid", "test", "eval")]
    if not every:
        raise SystemExit(f"[context] no corpus under {data_dir}\n"
                         f"          ./stage1_download_data.sh")
    if not files:
        wiki = config.dataset_dir("zetagpt-tiny_pretrain-corpus_wikitext103")
        raise SystemExit(
            f"[context] {os.path.relpath(data_dir, config.ROOT)} has no held-out split: all "
            f"{len(every):,} of its files are training data.\n"
            f"          Evaluating on them would measure memorisation, so this refuses "
            f"rather than report it as generalisation.\n"
            f"          Point --data_dir at a corpus that ships one, e.g.\n"
            f"              --data_dir {os.path.relpath(wiki, config.ROOT)}\n"
            f"          (its validation-*.parquet is genuinely unseen), or hold shards of "
            f"this corpus out of pretraining\n"
            f"          by naming them validation-*.parquet -- PRETRAIN.exclude_dirs already "
            f"keeps such a split out of training.")
    log(f"[context] held-out split: {len(files):,} of {len(every):,} files under "
        f"{os.path.relpath(data_dir, config.ROOT)} "
        f"({', '.join(sorted({_split_token(os.path.basename(f)) for f in files}))})")
    files = files[:limit]
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


def prefix_ids(spec, text, want, chars_per_token=4.5):
    """The LAST `want` token ids of `text`, tokenising ONLY AS MUCH TEXT AS THAT NEEDS.

    THIS IS WHY THERE IS NO PRE-TOKENISATION PASS. Every model here has its own tokenizer, so
    nothing can be shared between them, and tokenising a whole 130,000-character document to
    use the last 512 tokens of it is ten minutes of work thrown away -- which is exactly what
    the first version did, at 74 seconds a document before a single forward pass had run.

    A token is about 4.5 characters, so `want` tokens need about 4.5 x want characters. The
    estimate is checked rather than trusted: if the slice came out short, the window is doubled
    and the slice retaken, until it is long enough or the document is used up. Overshooting
    costs a little tokenising; undershooting would silently measure a shorter context than the
    x axis claims, which is the failure worth spending a retry to avoid.

    Taking the last N CHARACTERS and then cutting to the last N TOKENS leaves the left edge on
    whatever boundary the merges land on -- but the ids are cut from the left anyway, so that
    edge is arbitrary in both cases and nothing is lost by not seeing the whole document."""
    if want <= 0:
        return []
    take = min(len(text), int(want * chars_per_token) + 64)
    while True:
        ids = spec["encode"](text[len(text) - take:])
        if len(ids) >= want or take >= len(text):
            return ids[-want:]
        take = min(len(text), take * 2)


def probe_context_curve(spec, docs, args, device, log, live, cache, out):
    """The FIXED target span at CONTEXT LENGTHS OF 512, 1k, 2k, ... 32k TOKENS.

    THE X AXIS IS EXACTLY 2^n FOR EVERY MODEL. The document is tokenised once per model, the
    target is a fixed span of CHARACTERS -- the same text for everyone -- and the prefix is then
    truncated BY TOKEN ID to whatever leaves the total at L. Truncating ids rather than
    characters is what makes the length exact: a character budget lands on a different token
    count in each of four vocabularies, and the curves would not share an x value anywhere.

    FOUR CURVES OUT OF ONE SWEEP, because they are four readings of the same logits and they
    fail in different ways:

        nats per byte     comparable ACROSS models -- the only unit every tokenizer shares
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
    tail = args.target_chars
    chosen = [d for d in docs[:args.documents] if len(d) > tail]
    if not chosen:
        log(f"[context] {spec['key']:<10} curve  no document longer than the target span")
        return out
    # NOTHING IS TOKENISED IN ADVANCE, NOT EVEN THE TARGET. Every model here has its own
    # vocabulary, so no ids can be shared between them and none are worth keeping: a token
    # cache would be per-model, per-text, and correct only until a tokenizer changed under it.
    # The target span is re-encoded where it is used, inside the loop, at a cost of two
    # thousand characters -- and that is the whole of the rule, applied without exception so
    # there is no second place for a stale id to come from.
    for L in args.context_lengths:
        hit = cache.get(f"curve:{L}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   npb {hit['npb']:.4f}   "
                f"ppl {hit['ppl']:8.2f}   nats/tok {hit['nats_per_token']:.4f}   "
                f"acc {hit['acc']:.4f}   (cached)")
            live()
            continue
        chunk, why = plan_chunk(spec, L, args._budget_bytes, args._itemsize,
                                args.chunk_tokens, log)
        if why == "weights":
            # NOT A CONTEXT FINDING. The model is too big for the budget and would fail the
            # same way at 512; drawing a curve that stops here would say the model cannot
            # handle long contexts, which is not what was measured. Nothing is recorded.
            log(f"[context] {spec['key']:<10} curve  SKIPPED: the model does not fit "
                f"{args.vram_budget:g} GiB at any length -- raise --vram_budget or use "
                f"--dtype bf16")
            break
        if chunk <= 0:
            log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   the KV cache for this "
                f"LENGTH exceeds {args.vram_budget:g} GiB -- a limit of the length, not of "
                f"the chunk")
            live()
            break
        nats = 0.0
        nbytes = n_tok = n_cor = ctx = n_ok = short = 0
        docs_bar = progress(list(enumerate(chosen, 1)),
                            desc=f"[context] {spec['key']} ctx {L:,}", total=len(chosen))
        for i_doc, d in docs_bar:
            # ON THE FLY, here, every time. See above: no ids are kept anywhere.
            span_text = d[len(d) - tail:]
            span_ids = spec["encode"](span_text)
            span_bytes = len(span_text.encode("utf-8"))
            room = L - len(span_ids)
            if room < 0:                       # the target alone is longer than this context
                continue
            pre_ids = prefix_ids(spec, d[:len(d) - tail], room)
            # THE DOCUMENT MUST BE ABLE TO FILL THE CONTEXT. prefix_ids returns what it has,
            # so a short article would be scored at its own length while the axis said 32,768 --
            # a curve whose x values are fiction. Short documents are skipped at this length,
            # and if none is long enough the sweep stops and says which.
            if len(pre_ids) < room:
                short += 1
                continue
            n, t, c, T, ok = score_ids(spec, pre_ids, span_ids, device, chunk,
                                       desc=f"[context] {spec['key']} ctx {L:,} "
                                            f"doc {i_doc}/{len(chosen)}")
            if not ok:
                n_ok = 0
                break
            nats += n; nbytes += span_bytes; n_tok += t; n_cor += c; ctx += T; n_ok += 1
            if hasattr(docs_bar, "set_postfix_str"):
                docs_bar.set_postfix_str(f"npb {npb(nats, nbytes):.4f}, "
                                         f"acc {n_cor / max(n_tok, 1):.3f}")
        if hasattr(docs_bar, "close"):
            docs_bar.close()
        npt = (nats / n_tok) if n_ok and n_tok else None
        rec = {"context_length": L, "documents": n_ok,
               "ctx_tokens": round(ctx / n_ok) if n_ok else L,
               "npb": npb(nats, nbytes) if n_ok else None,
               "nats_per_token": npt,
               "ppl": math.exp(npt) if npt is not None else None,
               "acc": (n_cor / n_tok) if n_ok and n_tok else None}
        out.append(rec)
        log(f"[context] {spec['key']:<10} curve  ctx {L:>7,} tok   "
            + (f"npb {rec['npb']:.4f}   ppl {rec['ppl']:8.2f}   "
               f"nats/tok {rec['nats_per_token']:.4f}   acc {rec['acc']:.4f}   "
               f"({n_ok} docs" + (f", {short} too short)" if short else ")")
               if rec["npb"] is not None else
               (f"no document reaches {L:,} tokens -- the CORPUS ends the sweep, "
                f"not the model" if short and not n_ok else "unreachable")))
        if rec["npb"] is not None:
            cache.put(f"curve:{L}", rec)
        live()
        if rec["npb"] is None:
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


def probe_copy(spec, args, device, log, live, cache, out):
    """A passage, d words of filler, then THE SAME PASSAGE. Bits per byte on each copy.

    THE SECOND COPY IS FREE INFORMATION. A model that can reach back over the filler pays
    almost nothing for it; one that cannot pays what it paid the first time. The GAIN --
    first minus second -- is the retrieval signal, in nats per byte, and it is the cleanest
    statement of what a positional mechanism buys: no semantics, no vocabulary, no corpus.

    The passage is drawn from the same word bank, so the first copy is not itself hard; what
    is being measured is the DIFFERENCE, and a difficult passage would only add noise to both
    ends of it."""
    for dist in args.copy_distances:
        hit = cache.get(f"copy:{dist}")
        if hit is not None:
            out.append(hit)
            log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
                f"1st {hit['npb_first']:.3f} -> 2nd {hit['npb_second']:.3f} npb   "
                f"gain {hit['gain']:+.3f}   (cached)")
            live()
            continue
        first_n = first_b = second_n = second_b = 0.0
        reached = 0
        trials = progress(range(args.trials), total=args.trials,
                          desc=f"[context] {spec['key']} copy filler {dist:,}")
        for trial in trials:
            rng = random.Random(args.seed * 1009 + dist * 31 + trial)
            passage = _filler_words(rng, args.copy_words)
            filler = _filler_words(rng, dist)
            tag = f"[context] {spec['key']} copy filler {dist:,} trial {trial + 1}"
            n1, b1, _, _, _, ok1 = score_span(spec, "", passage, device,
                                              (args.chunk_tokens, args._budget_bytes,
                                               args._itemsize), desc=tag + " (1st copy)")
            n2, b2, _, _, _, ok2 = score_span(spec, passage + " " + filler + " ", passage,
                                              device, (args.chunk_tokens,
                                              args._budget_bytes, args._itemsize),
                                              desc=tag + " (2nd copy)")
            if not (ok1 and ok2):
                break
            first_n += n1; first_b += b1; second_n += n2; second_b += b2; reached += 1
            if hasattr(trials, "set_postfix_str"):
                trials.set_postfix_str(
                    f"gain {npb(first_n, first_b) - npb(second_n, second_b):+.3f} npb")
        if hasattr(trials, "close"):
            trials.close()
        rec = {"filler_words": dist, "trials": reached,
               "npb_first": npb(first_n, first_b) if reached else None,
               "npb_second": npb(second_n, second_b) if reached else None}
        rec["gain"] = (None if rec["npb_first"] is None
                       else rec["npb_first"] - rec["npb_second"])
        out.append(rec)
        log(f"[context] {spec['key']:<10} copy   filler {dist:>7,} words  "
            + (f"1st {rec['npb_first']:.3f} -> 2nd {rec['npb_second']:.3f} npb   "
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
def probe_passkey(spec, args, device, log, live, cache, out):
    """A five-digit key at a known depth, scored as -log P(the digits | everything before).

    NEVER GENERATED, NEVER EXACT-MATCHED. A 133M model that puts a third of its mass on the
    right key has found it; exact match would call that a zero and the whole probe would read
    as a flat floor for every small model. The likelihood is graded, and it is the same
    quantity for a model that would have got it right."""
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
            trials = progress(range(args.trials), total=args.trials,
                              desc=f"[context] {spec['key']} key {total:,} @ {depth:.2f}")
            for trial in trials:
                rng = random.Random(args.seed * 7919 + total * 17 + int(depth * 100) + trial)
                key = f"{rng.randrange(10000, 100000)}"
                before = int(total * depth)
                text = (_filler_words(rng, before)
                        + f" The pass key is {key}. Remember it. "
                        + _filler_words(rng, total - before)
                        + " The pass key is")
                n, b, _, _, _, ok = score_span(
                    spec, text, f" {key}", device,
                    (args.chunk_tokens, args._budget_bytes, args._itemsize),
                    desc=f"[context] {spec['key']} key {total:,} @ {depth:.2f} "
                         f"trial {trial + 1}")
                if not ok:
                    break
                nats += n; nbytes += b; reached += 1
                if hasattr(trials, "set_postfix_str"):
                    trials.set_postfix_str(f"{nats / reached:.2f} nats/key")
            if hasattr(trials, "close"):
                trials.close()
            rec = {"filler_words": total, "depth": depth, "trials": reached,
                   "npb": npb(nats, nbytes) if reached else None,
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
def split_at(pts, x0):
    """A polyline cut at x0 into (within, beyond), SHARING an interpolated point at the cut.

    Sharing the boundary point is what keeps the line continuous: drawn as two polylines a gap
    would open at exactly the length the reader is being asked to look at. Where x0 falls
    between two measured points the y there is interpolated."""
    pts = sorted(pts)
    if not pts or not x0 or x0 <= pts[0][0]:
        return [], pts
    if x0 >= pts[-1][0]:
        return pts, []
    within = [q for q in pts if q[0] <= x0]
    beyond = [q for q in pts if q[0] > x0]
    # INTERPOLATE ONLY WHEN x0 IS NOT ALREADY A MEASURED POINT -- but SHARE the boundary in
    # BOTH cases, which is where this went wrong. When a training length coincides with a
    # measured length, as 2,048 does for TinyStories and Pythia, the interpolation branch was
    # skipped and so was the sharing, so `beyond` began at the NEXT length and the figure had a
    # visible gap between 2^11 and 2^12 -- exactly at the boundary the panel exists to show.
    if within[-1][0] != x0:
        (xa, ya), (xb, yb) = within[-1], beyond[0]
        within = within + [(x0, ya + (yb - ya) * (x0 - xa) / (xb - xa))]
    return within, [within[-1]] + beyond


def _curve_panel(ax, models, field, xlabel, ylabel, title, logy=False):
    """One metric of probe 1 against CONTEXT LENGTH IN TOKENS.

    x is each model's OWN token count, not a shared axis of characters, because perplexity,
    nats per token and accuracy are per-token quantities: plotting them against a character
    count would divide by one thing and index by another. What is held identical across models
    is the TARGET -- the same span of text is being predicted in every case -- and the x
    positions differ only because the tokenizers do."""
    drew = False
    n = len(models)
    for i, m in enumerate(models):
        pts = [(r.get("context_length") or r["ctx_tokens"], r[field])
               for r in m["curve"] if r.get(field) is not None]
        if not pts:
            continue
        st = style_of(m["key"], i)
        within, beyond = split_at(pts, m.get("train_len"))
        if within:
            ax.plot(*zip(*within), linestyle=SOLID, label=m["name"], **st)
        if beyond:
            ax.plot(*zip(*beyond), linestyle=BEYOND,
                    **({**st, "label": m["name"]} if not within else st))
        # THE TRAINING LENGTH, RINGED. A hollow marker in the model's own colour, on the point
        # where the curve changes meaning: the last length it was trained for, and the first at
        # which it is extrapolating.
        if within and beyond:
            ax.plot([within[-1][0]], [within[-1][1]], marker="o", markersize=9.5,
                    markerfacecolor="none", markeredgecolor=st["color"], markeredgewidth=1.5,
                    linestyle="none", zorder=st["zorder"] + 1)
        drew = True
        # WHERE A MODEL STOPPED, and that it did. A line ending because the architecture
        # refused the length must LOOK like it ended -- an x, not a line run to the axis edge.
        if any(r.get(field) is None for r in m["curve"]):
            ax.plot(pts[-1][0], pts[-1][1], "x", color=st["color"], ms=8, mew=1.8,
                    zorder=st["zorder"] + 1)
        # NO VERTICAL RULES. Each curve carries its own boundary as a ring, so eleven of them
        # would repeat what the lines already say and turn the panel into a grid.
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9.5)
    if drew:
        # THREE PROXY ENTRIES, so the convention is stated in the figure rather than in a
        # caption somebody may not have. They carry no data and are drawn in grey.
        from matplotlib.lines import Line2D
        h, l = ax.get_legend_handles_labels()
        h += [Line2D([], [], color="0.35", linestyle=SOLID),
              Line2D([], [], color="0.35", linestyle=BEYOND),
              Line2D([], [], color="0.35", marker="o", markerfacecolor="none", markersize=8,
                     linestyle="none")]
        l += ["within trained context", "beyond it (extrapolating)", "training length"]
        ax.legend(h, l, fontsize=6.2, ncol=2, loc="best", handlelength=2.6)


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
    _curve_panel(ax[0], models, "npb", "context length (tokens)", "nats / byte",
                 "average encoding length per byte (nats / byte)", logy=True)
    _curve_panel(ax[1], models, "ppl", "context length (tokens)", "perplexity",
                 "perplexity", logy=True)
    _curve_panel(ax[2], models, "nats_per_token", "context length (tokens)", "nats / token",
                 "average encoding length per token")
    _curve_panel(ax[3], models, "acc", "context length (tokens)", "accuracy",
                 "next-token prediction accuracy")

    # ---- row 2: the two synthetic probes ----------------------------------------------- #
    for i, m in enumerate(models):
        pts = [(r["filler_words"], r["gain"]) for r in m["copy"] if r.get("gain") is not None]
        if pts:
            ax[4].plot(*zip(*pts), label=m["name"], **style_of(m["key"], i))
    ax[4].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[4].set_xscale("log", base=2)
    ax[4].set_xlabel("filler words between the two copies")
    ax[4].set_ylabel("nats / byte saved on the second copy")
    ax[4].set_title("encoding length saved by a repeated passage", loc="left", fontsize=9.5)
    if ax[4].get_legend_handles_labels()[0]:
        ax[4].legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)

    for i, m in enumerate(models):
        by_len = {}
        for r in m["passkey"]:
            if r.get("nats_per_key") is not None:
                by_len.setdefault(r["filler_words"], []).append(r["nats_per_key"])
        pts = sorted((k, sum(v) / len(v)) for k, v in by_len.items())
        if pts:
            ax[5].plot(*zip(*pts), label=m["name"], **style_of(m["key"], i))
    ax[5].axhline(math.log(1e5), color="k", lw=0.8, ls=":", alpha=0.6)
    ax[5].text(0.02, math.log(1e5), " uniform guess", va="bottom", fontsize=7,
               transform=ax[5].get_yaxis_transform())
    ax[5].set_xscale("log", base=2)
    ax[5].set_xlabel("filler words around the key")
    ax[5].set_ylabel("nats to name the key")
    ax[5].set_title("encoding length of a buried key", loc="left", fontsize=9.5)
    if ax[5].get_legend_handles_labels()[0]:
        ax[5].legend(fontsize=6.2, ncol=2, loc="best", handlelength=2.6)

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
    """Measured points, on disk, so a crashed or changed sweep never measures twice.

        cache/evals/eval_context_length/<model>.json

    NOTHING IS EVER OVERWRITTEN. THIS IS THE RULE THE FILE IS BUILT AROUND, and it was broken
    twice: a signature that did not match left `points` empty and the first put() then wrote
    that empty dict over the file, destroying every measurement it had just declined to use.
    --no-resume did the same. Hours of forwards, gone, to record that a setting had changed.

    SO THE FILE HOLDS BUCKETS, one per signature:

        {"buckets": {<sig hash>: {"signature": {...}, "points": {...}}, ...}}

    A signature change starts a NEW bucket beside the old ones rather than replacing anything.
    Change --dtype and come back tomorrow and both sets are there. put() RE-READS the file,
    edits its own bucket and writes the whole thing back, so a second run measuring a different
    model, or the same model under different settings, cannot clobber the first.

    A file in the old flat shape is read as a single bucket, so existing caches keep working.

    THE SIGNATURE IS MACHINE-INDEPENDENT -- the checkpoint's NAME and step, the corpus's NAME,
    the sampling settings; never a path, a device or a time -- and it is compared only on the
    keys the CURRENT version declares, so narrowing it later cannot invalidate anything.

    ONLY SUCCESSFUL POINTS ARE KEPT. An allocation failure is a fact about one machine, and a
    machine-independent key must not carry it."""

    def __init__(self, key, signature, log, enabled=True):
        self.path = os.path.join(CACHE_DIR, f"{key}.json")
        self.sig, self.log, self.enabled, self.points = signature, log, enabled, {}
        self.hash = _sig_hash(signature)
        buckets = self._read()
        mine, why = _match(buckets, signature)
        if mine and why:
            log(f"[context] {key}: {why}")
        if not enabled:
            if mine:
                log(f"[context] {key}: --no-resume, re-measuring "
                    f"{len(mine.get('points', {})):,} points (they are KEPT on disk until "
                    f"each is replaced)")
            return
        if mine:
            self.points = dict(mine.get("points", {}))
            if self.points:
                log(f"[context] {key}: resuming, {len(self.points):,} points already measured")
        elif buckets:
            log(f"[context] {key}: no bucket matches these settings; "
                f"{len(buckets)} other set(s) on disk are left untouched")

    def _read(self):
        """Every bucket in the file, keyed by signature hash. A file in the old flat shape
        becomes one bucket; an unreadable file becomes none, and is not written over until a
        point is actually measured."""
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            return {}
        if "buckets" in d:
            return d["buckets"]
        if "signature" in d:                        # the older, single-signature layout
            return {_sig_hash(d["signature"]): {"signature": d["signature"],
                                                "points": d.get("points", {})}}
        return {}

    def get(self, pid):
        return self.points.get(pid) if self.enabled else None

    def describe(self, meta):
        """What the FIGURE needs to draw this model, stored beside its points.

        Without it a cached model can only be redrawn by loading its weights again, which is
        absurd for a row whose numbers are already on disk -- and impossible for a model whose
        checkpoint has since gone bad. Name, parameter count, training length and positional
        scheme are a few hundred bytes and they make the cache self-sufficient."""
        self.meta = dict(meta)

    def put(self, pid, rec):
        """Written THE MOMENT the point is measured, atomically, and MERGED rather than
        replacing: the file is re-read first, so buckets this run knows nothing about survive,
        and so does a bucket written by a run in another terminal."""
        self.points[pid] = rec
        buckets = self._read()
        buckets[self.hash] = {"signature": self.sig, "meta": getattr(self, "meta", {}),
                              "points": self.points}
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = self.path + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"buckets": buckets}, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def _sig_hash(sig):
    """A stable id for a signature. Sorted keys so two runs agree, and short so the file stays
    readable by eye."""
    import hashlib
    return hashlib.sha1(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:12]


def _match(buckets, signature):
    """The bucket whose signature agrees with `signature` ON THE KEYS IT DECLARES, and a note
    when it was written by a version that keyed on more than this one does."""
    for b in buckets.values():
        old = b.get("signature", {})
        if any(old.get(k) != v for k, v in signature.items()):
            continue
        extra = [k for k in old if k not in signature]
        return b, (f"cache written by an older version ({', '.join(sorted(extra))} no longer "
                   f"part of the key) -- kept" if extra else "")
    return None, ""


# THE KEYS THAT BELONG TO THE RUN rather than to a model. A cached row can be redrawn without
# loading the model at all, but only if it was measured under the same corpus and settings --
# these are what that means, and the model's own keys (which checkpoint, which step) are
# deliberately not among them because a row is being IDENTIFIED, not re-verified.
RUN_KEYS = ("corpus", "target_chars", "documents", "trials", "copy_words", "seed", "dtype",
            "unit")


def rebuild_rows(points):
    """Cached points -> the (curve, copy, passkey) lists the figure reads, in x order."""
    curve, cp, pk = [], [], []
    for pid, rec in points.items():
        (curve if pid.startswith("curve:") else cp if pid.startswith("copy:") else pk).append(rec)
    curve.sort(key=lambda r: r.get("context_length") or r.get("ctx_tokens") or 0)
    cp.sort(key=lambda r: r.get("filler_words") or 0)
    pk.sort(key=lambda r: (r.get("filler_words") or 0, r.get("depth") or 0))
    return curve, cp, pk


def cached_models(args, corpus, log):
    """Every model with measurements on disk, as figure rows, WITHOUT LOADING A SINGLE WEIGHT.

    THIS IS WHAT MAKES A RESUMED RUN DRAW THE WHOLE FIGURE. The rows used to be built only from
    the models loaded THIS time, so a model skipped for any reason -- a checkpoint that would
    not open, a baseline that would not download, or simply `--only zetagpt` -- vanished from
    the pdf even though every one of its points was sitting in the cache. The figure then
    silently showed less than had been measured, which is worse than showing nothing: it looks
    like a finished comparison.

    A bucket carries its own `meta` (name, parameters, training length, positional scheme), so
    a row is redrawn from the cache alone -- including for a model whose checkpoint has since
    become unreadable, which is exactly when it matters most."""
    out = {}
    if not os.path.isdir(CACHE_DIR):
        return out
    run = {k: v for k, v in cache_signature({"key": "", "checkpoint": "", "step": None},
                                            args, corpus).items() if k in RUN_KEYS}
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".json"):
            continue
        key = fn[:-len(".json")]
        try:
            with open(os.path.join(CACHE_DIR, fn), "r", encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        buckets = d.get("buckets") or ({"flat": d} if "signature" in d else {})
        for b in buckets.values():
            sig = b.get("signature", {})
            if any(sig.get(k) != v for k, v in run.items()):
                continue
            pts = b.get("points") or {}
            if not pts:
                continue
            row = {"key": key, "name": key, "params": 0, "train_len": 0, "max_pos": 0,
                   "positional": "", "checkpoint": sig.get("checkpoint", ""),
                   "step": sig.get("step"), "total": None}
            row.update({k: v for k, v in (b.get("meta") or {}).items() if v is not None})
            row["curve"], row["copy"], row["passkey"] = rebuild_rows(pts)
            out[key] = row
            break
    if out:
        log(f"[context] redrawing {len(out)} model(s) from the cache: "
            + ", ".join(f"{k} ({len(v['curve'])}+{len(v['copy'])}+{len(v['passkey'])} pts)"
                        for k, v in out.items()))
    return out


def cache_signature(spec, args, corpus):
    """What a cached POINT depends on -- and nothing else.

    THE GRID IS NOT IN HERE, and putting it in was the bug: `context_lengths` was part of the
    signature, so adding a length to the sweep discarded every point already measured at 512
    through 32,768. Those points had not changed. `curve:512` means "the nats per byte at
    context 512, for this checkpoint, on this corpus, over these documents" -- which of the
    other lengths were also asked for is a property of the RUN, not of the measurement. The same
    went for the copy and passkey grids.

    NOR IS ANYTHING THAT ONLY AFFECTS HOW THE NUMBER IS COMPUTED. --chunk_tokens and
    --vram_budget decide how much is resident during a forward pass; they do not change the
    logits, so a cache must survive a change to either. Wiping hours of measurement because the
    memory budget moved is exactly the failure this cache exists to prevent.

    WHAT IS LEFT is what would make the number different: which model (spec["name"] carries
    "positions interpolated" when the table was stretched, so an extended model is a different
    model here), which checkpoint and step, which corpus, how much text is scored, how many
    documents and trials, the seed, the precision, and the unit.

    NO PATH, NO DEVICE, NO TIME -- so two machines with the same checkpoint agree on what a
    cached point means."""
    return {
        # THE LABEL IS NOT IN THE KEY. It carried "(positions interpolated)", so enabling the
        # extension would have discarded every point already measured -- and those points are
        # still correct: a stretched table is identical to a native one BELOW the native
        # window, and above it the native model recorded nothing, because a refused length is
        # never cached. There is nothing for the two to disagree about.
        "model": spec["key"],
        "checkpoint": os.path.basename(str(spec["checkpoint"]).rstrip("/")),
        "step": spec.get("step"),
        "corpus": os.path.basename(str(corpus).rstrip("/")),
        "target_chars": args.target_chars, "documents": args.documents,
        "trials": args.trials, "copy_words": args.copy_words,
        "seed": args.seed, "dtype": args.dtype, "unit": "nats_per_byte",
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
        curve = [r for r in m["curve"] if r["npb"] is not None]
        gains = [r["gain"] for r in m["copy"] if r.get("gain") is not None]
        keys = [r["nats_per_key"] for r in m["passkey"] if r.get("nats_per_key") is not None]
        far = max((r["filler_words"] for r in m["copy"]
                   if (r.get("gain") or 0) > 0.05), default=0)
        rows += [
            (m["name"], ""),
            ("  positional", m["positional"]),
            ("  parameters", f"{m['params'] / 1e6:.1f}M"),
            ("  trained at", f"{m['train_len']:,} tokens"),
            ("  best npb / at", f"{min(r['npb'] for r in curve):.4f} at "
                                f"{max(r.get('context_length') or 0 for r in curve):,} tokens"
                                if curve else "-"),
            ("  furthest copy", f"{far:,} filler words still worth over 0.05 npb"),
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
    # WIKITEXT-103's VALIDATION SPLIT BY DEFAULT, and not the pretraining corpus, because it is
    # the only downloaded corpus that HOLDS ANYTHING OUT. Scoring a model on its own training
    # text measures memorisation; that a held-out set comes from a different distribution than
    # FineWeb-Edu is a feature here rather than a flaw, since the claim under test is about
    # length rather than about domain.
    p.add_argument("--data_dir",
                   default=config.dataset_dir("zetagpt-tiny_pretrain-corpus_wikitext103"),
                   help="corpus for probe 1; only its HELD-OUT files are read "
                        "(validation/test/eval by name) and a corpus with none is refused. "
                        "Nothing is concatenated, so short documents cap the sweep and say so")
    p.add_argument("--documents", type=int, default=8, help="documents averaged in probe 1")
    p.add_argument("--target_chars", type=int, default=2000,
                   help="the fixed target span every context length is scored on")
    p.add_argument("--chunk_tokens", type=int, default=512,
                   help="UPPER BOUND on tokens per forward pass; the actual chunk is planned "
                        "per context length to fit --vram_budget, and is smaller at 32k than "
                        "at 512 because the KV cache has taken the room")
    # FORCE A LEARNED TABLE PAST ITS ROWS rather than stopping the curve there. 0 refuses, as
    # before; a number stretches the table to that many rows by interpolation and the model is
    # relabelled in the figure, because it is no longer the published one. The comparison this
    # buys is the interesting one: ZetaGPT needs no such operation at any length.
    p.add_argument("--extend_positions", type=int, default=32768, metavar="ROWS",
                   help="interpolate a learned position table up to this many rows so the model "
                        "can be measured past its native window; 0 = stop the curve at the wall "
                        "(default: 32768, the longest context in the sweep)")
    p.add_argument("--vram_budget", type=float, default=20.0, metavar="GiB",
                   help="what the whole thing may occupy: weights + KV cache + the pass in "
                        "flight. The chunk is solved for this, and a length whose CACHE ALONE "
                        "exceeds it is reported unreachable before anything is allocated "
                        "(default: 20)")
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
    a._budget_bytes = int(a.vram_budget * 2 ** 30)
    a._itemsize = {"fp32": 4, "bf16": 2, "fp16": 2}[a.dtype]
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
    data_dir = args.data_dir
    # WHAT IS ALREADY ON DISK, FIRST. Rows are rebuilt from the cache before a single model is
    # loaded, so the figure carries every measurement ever taken under these settings -- and a
    # model that cannot be loaded at all keeps the curve it earned earlier.
    seeded = cached_models(args, data_dir, log)

    specs = []
    for key in want:
        s = (spec_zetagpt(args, device, dtype, log) if key == "zetagpt"
             else spec_hf(key, BASELINES[key], device, dtype, log, args))
        if s is not None:
            specs.append(s)
    if not specs and not seeded:
        raise SystemExit("[context] no model could be loaded and nothing is cached")
    if not specs:
        log("[context] no model could be loaded; redrawing what is cached and stopping")
    # a token is ~4 characters, so the longest context wants roughly 4x its tokens in text
    docs = long_documents(data_dir, max(args.context_lengths) * 4 + args.target_chars, log)

    # THE RESULT OBJECT STARTS FROM THE CACHE, in the canonical size order, so the very first
    # figure of a resumed run already carries everything measured before it. A model being
    # measured now REPLACES its seeded row rather than adding a second one -- the probes replay
    # their cached points into the fresh list, and two rows for one model would draw it twice.
    order = ["zetagpt"] + list(BASELINES)
    res = {"models": [seeded[k] for k in order if k in seeded],
           "corpus": os.path.relpath(data_dir, config.ROOT),
           "target_chars": args.target_chars, "trials": args.trials,
           "copy_words": args.copy_words, "dtype": args.dtype, "seed": args.seed,
           "progress": "starting"}
    live = Live(res, args.json, args.out, log)
    live("redrawn from the cache")             # the pdf is correct before anything is measured
    log(f"[context] writing {os.path.relpath(args.out, config.ROOT)} after every point")

    for i, spec in enumerate(specs, 1):
        log(f"\n[context] === {spec['name']}  ({i}/{len(specs)}) ===")
        m = {k: spec[k] for k in ("key", "name", "params", "train_len", "max_pos",
                                  "positional", "checkpoint", "step", "total")}
        m["curve"], m["copy"], m["passkey"] = [], [], []
        at = next((j for j, r in enumerate(res["models"]) if r["key"] == spec["key"]), None)
        if at is None:
            res["models"].append(m)
        else:
            res["models"][at] = m
        tag = f"{spec['key']} ({i}/{len(specs)})"
        cache = Cache(spec["key"], cache_signature(spec, args, data_dir), log,
                      enabled=args.resume)
        # WHAT THE FIGURE NEEDS TO REDRAW THIS ROW WITHOUT THE MODEL, stored beside its points.
        cache.describe({k: m[k] for k in ("name", "params", "train_len", "max_pos",
                                          "positional", "step", "total")})
        # THE PROBES APPEND TO THE VERY LISTS THE FIGURE READS. Assigning their return value
        # instead -- m["curve"] = probe_...(...) -- binds only when the probe RETURNS, so every
        # live() call inside it redrew a figure whose curve was still empty and the pdf gained
        # nothing until a whole model was finished. The list has to be shared, not returned.
        probe_context_curve(spec, docs, args, device, log,
                            lambda: live(f"{tag}: context curve"), cache, m["curve"])
        probe_copy(spec, args, device, log, lambda: live(f"{tag}: copy"), cache, m["copy"])
        probe_passkey(spec, args, device, log, lambda: live(f"{tag}: passkey"), cache,
                      m["passkey"])
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
