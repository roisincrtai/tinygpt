"""
evals/eval_pretrain_context_length.py -- does a PRETRAINED model actually use a long context,
and does it still work past the length it was trained at?

    python evals/eval_pretrain_context_length.py
    python evals/eval_pretrain_context_length.py --only zetagpt
    python evals/eval_pretrain_context_length.py --plot_only

    outputs/plots/evaluation/pretrain_context_generalization.pdf
    outputs/eval/pretrain_context_generalization.json

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
        "hidden": lambda ids: model.hidden(input_ids=ids), "head": model.head,
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
        "hidden": ((lambda ids: trunk(input_ids=ids).last_hidden_state)
                   if trunk is not None else None),
        "head": getattr(model, "lm_head", None),
    }, model, log)


# --------------------------------------------------------------------------- #
# scoring: nats over an exact span, and the bytes that span covers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_span(spec, prefix, span, device, chunk=2048):
    """-log P(span | prefix), in NATS, and the number of BYTES `span` occupies.

    THE PREFIX AND THE SPAN ARE TOKENISED SEPARATELY and their ids concatenated. Tokenising the
    join would let a merge straddle the boundary, and then the tokens being scored would differ
    from one model to the next by more than the tokenizer -- the span itself would have moved.
    Encoding them apart costs one merge at the seam and buys a span that is exactly the same
    text for every model, which is the only way four vocabularies land on one axis.

    Returns (nats, bytes, n_tokens, ok). `ok` is False when the model refuses the length -- a
    learned position table with no row for it -- which is a fact to plot, not an error."""
    pid = spec["encode"](prefix) if prefix else []
    sid = spec["encode"](span)
    if not sid:
        return 0.0, 0, 0, False
    if spec["max_pos"] and len(pid) + len(sid) > spec["max_pos"]:
        return None, len(span.encode("utf-8")), len(sid), False
    ids = torch.tensor([pid + sid], device=device)
    n_pre = len(pid)
    total = 0.0
    try:
        if spec["hidden"] is not None and spec["head"] is not None:
            h = spec["hidden"](ids)
            # the vocabulary projection in slices along time, so no (T x vocab) tensor is ever
            # alive -- at 32k and a 150k vocabulary that tensor alone is 19 GiB in fp32
            for a in range(max(n_pre - 1, 0), ids.shape[1] - 1, chunk):
                b = min(a + chunk, ids.shape[1] - 1)
                lg = spec["head"](h[:, a:b]).float()
                lp = lg.log_softmax(-1).gather(-1, ids[:, a + 1:b + 1].unsqueeze(-1))
                total += float(-lp.sum())
                del lg, lp
            del h
        else:
            lg = spec["model"](input_ids=ids).logits[:, :-1].float()
            lp = lg.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            total = float(-lp[0, max(n_pre - 1, 0):].sum())
            del lg, lp
    except (torch.cuda.OutOfMemoryError if torch.cuda.is_available() else RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return None, len(span.encode("utf-8")), len(sid), False
    return total, len(span.encode("utf-8")), len(sid), True


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


def probe_context_curve(spec, docs, args, device, log):
    """Bits per byte on a FIXED target span, given 2^k characters of its own document before it.

    The target never moves. Only how much of the document precedes it changes, so a curve that
    falls is the model extracting something from text it did not have before, and a curve that
    flattens is the model having stopped reading."""
    out = []
    doc = docs[0]
    tail = args.target_chars
    for pre_chars in args.prefix_chars:
        if pre_chars + tail > len(doc):
            break
        nats = nbytes = 0.0
        n_ok = 0
        for d in docs[:args.documents]:
            if pre_chars + tail > len(d):
                continue
            cut = len(d) - tail
            prefix, span = d[max(cut - pre_chars, 0):cut], d[cut:]
            n, b, _, ok = score_span(spec, prefix, span, device)
            if not ok:
                break
            nats += n; nbytes += b; n_ok += 1
        rec = {"prefix_chars": pre_chars, "documents": n_ok,
               "bpb": bpb(nats, nbytes) if n_ok else None}
        out.append(rec)
        log(f"[context] {spec['key']:<10} curve  prefix {pre_chars:>7,} chars  "
            + (f"bpb {rec['bpb']:.4f}  ({n_ok} docs)" if rec["bpb"] else "unreachable"))
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


def probe_copy(spec, args, device, log):
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
        first_n = first_b = second_n = second_b = 0.0
        reached = 0
        for trial in range(args.trials):
            rng = random.Random(args.seed * 1009 + dist * 31 + trial)
            passage = _filler_words(rng, args.copy_words)
            filler = _filler_words(rng, dist)
            n1, b1, _, ok1 = score_span(spec, "", passage, device)
            n2, b2, _, ok2 = score_span(spec, passage + " " + filler + " ", passage, device)
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
        if rec["gain"] is None:
            break
    return out


# --------------------------------------------------------------------------- #
# probe 3 -- passkey, scored as a likelihood
# --------------------------------------------------------------------------- #
def probe_passkey(spec, args, device, log):
    """A five-digit key at a known depth, scored as -log P(the digits | everything before).

    NEVER GENERATED, NEVER EXACT-MATCHED. A 133M model that puts a third of its mass on the
    right key has found it; exact match would call that a zero and the whole probe would read
    as a flat floor for every small model. The likelihood is graded, and it is the same
    quantity for a model that would have got it right."""
    out = []
    for total in args.passkey_lengths:
        for depth in args.passkey_depths:
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
                n, b, _, ok = score_span(spec, text, f" {key}", device)
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
    return out


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #
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
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.9))
    models = res["models"]

    # (1) the context curve
    for m in models:
        pts = [(r["prefix_chars"], r["bpb"]) for r in m["curve"] if r["bpb"] is not None]
        if not pts:
            continue
        c = COLOURS.get(m["key"], GREY)
        ax[0].plot(*zip(*pts), "o-", color=c, lw=1.5, ms=3.4, label=m["name"])
        # WHERE EACH MODEL STOPPED, and why. A line that ends because the architecture refused
        # the length must look like it ended; drawing it to the axis edge would hide the result.
        if len(pts) < len([r for r in m["curve"]]):
            ax[0].plot(pts[-1][0], pts[-1][1], "x", color=c, ms=8, mew=1.8)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("characters of the document before the target")
    ax[0].set_ylabel("bits per byte on the FIXED target span")
    ax[0].set_title("1. is the extra context used?", loc="left", fontsize=9.5)
    ax[0].legend(fontsize=7)

    # (2) copy at distance -- the GAIN is the signal
    for m in models:
        pts = [(r["filler_words"], r["gain"]) for r in m["copy"] if r.get("gain") is not None]
        if not pts:
            continue
        ax[1].plot(*zip(*pts), "s-", color=COLOURS.get(m["key"], GREY), lw=1.5, ms=3.4,
                   label=m["name"])
    ax[1].axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("filler words between the two copies")
    ax[1].set_ylabel("bits/byte saved on the second copy")
    ax[1].set_title("2. can it reach back?", loc="left", fontsize=9.5)
    ax[1].legend(fontsize=7)

    # (3) passkey, averaged over depth
    for m in models:
        by_len = {}
        for r in m["passkey"]:
            if r.get("nats_per_key") is not None:
                by_len.setdefault(r["filler_words"], []).append(r["nats_per_key"])
        pts = sorted((k, sum(v) / len(v)) for k, v in by_len.items())
        if not pts:
            continue
        ax[2].plot(*zip(*pts), "^-", color=COLOURS.get(m["key"], GREY), lw=1.5, ms=4.0,
                   label=m["name"])
    ax[2].axhline(math.log(1e5), color="k", lw=0.8, ls=":", alpha=0.6)
    ax[2].text(0.02, math.log(1e5), " uniform guess", va="bottom", fontsize=7,
               transform=ax[2].get_yaxis_transform())
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("filler words around the key")
    ax[2].set_ylabel("nats to name the key  (lower is better)")
    ax[2].set_title("3. can it retrieve a fact?", loc="left", fontsize=9.5)
    ax[2].legend(fontsize=7)

    fig.suptitle("Context generalization of PRETRAINED models -- likelihood only, "
                 "nothing generated", y=1.0, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
                                f"{max(r['prefix_chars'] for r in curve):,} chars"
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
                   help="the fixed target span every prefix length is scored on")
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
    a = p.parse_args()
    a.prefix_chars = [0, 512, 2048, 8192, 32768, 131072]
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
    docs = long_documents(data_dir, max(args.prefix_chars) + args.target_chars, log)

    models = []
    for spec in specs:
        log(f"\n[context] === {spec['name']} ===")
        m = {k: spec[k] for k in ("key", "name", "params", "train_len", "max_pos",
                                  "positional", "checkpoint", "step", "total")}
        m["curve"] = probe_context_curve(spec, docs, args, device, log)
        m["copy"] = probe_copy(spec, args, device, log)
        m["passkey"] = probe_passkey(spec, args, device, log)
        models.append(m)
        spec["model"] = None                       # let the weights go before the next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    res = {"models": models, "corpus": os.path.relpath(data_dir, config.ROOT),
           "target_chars": args.target_chars, "trials": args.trials,
           "copy_words": args.copy_words, "dtype": args.dtype, "seed": args.seed}
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    log(f"\n[context] [json] {os.path.relpath(args.json, config.ROOT)}")
    summarise(res, log)
    log(f"[context] [figure] {figure(res, args.out)}")


if __name__ == "__main__":
    main()
