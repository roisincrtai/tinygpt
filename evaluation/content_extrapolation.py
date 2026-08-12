"""
evaluation/content_extrapolation.py -- how far past its training length a model still models
text, for ZetaGPT and for GPT-2 small side by side.

    python -m evaluation.content_extrapolation
    python -m evaluation.content_extrapolation --base 512 --dtype bf16
    python -m evaluation.content_extrapolation --gpt2 ""      # ZetaGPT only
    python -m evaluation.content_extrapolation --plot_only

Writes outputs/plots/pretrain/content_extrapolation.pdf -- one ROW PER MODEL, ZetaGPT above
and GPT-2 small below -- and the numbers behind it to outputs/eval/content_extrapolation.json.
To hold an ablation beside the run it is compared with, give both an explicit name:

    python -m evaluation.content_extrapolation \
        --checkpoint checkpoints/pretrain/checkpoint_zetagpt-s_rope_pretrain.pt \
        --out outputs/plots/pretrain/content_extrapolation_rope.pdf \
        --json outputs/eval/content_extrapolation_rope.json

THE PROTOCOL is "train short, test long", zero-shot: each model is evaluated unchanged on
contexts of 1x, 2x, 4x, ... 128x, and nothing is re-tuned, re-scaled or re-initialised in
between.

WHY GPT-2 IS THE RIGHT SECOND ROW, and what it is expected to do. GPT-2 has a LEARNED ABSOLUTE
POSITION TABLE of n_positions = 1024 rows. There is no row 1025, so beyond that length the
model cannot be evaluated at all -- not "evaluated badly", but not evaluated: the embedding
lookup has nothing to return. Lengths past the table are therefore reported as
`beyond_position_table` rather than attempted, and that hard wall IS the comparison. ZetaGPT
has no such table: position is supplied by the recurrence in front of attention, so a longer
context is simply a longer forward pass and the sweep continues until memory, not
architecture, stops it.

READ THE TWO ROWS FOR SHAPE, NOT FOR LEVEL. The models have DIFFERENT TOKENIZERS -- ours is a
50k byte-level BPE trained on this corpus, GPT-2's is its own -- and perplexity is per TOKEN,
so a model whose tokenizer cuts text into fewer, longer pieces reports a higher perplexity for
identical predictions. The levels are not comparable and the script does not pretend they are:
each row is drawn on its own axis, the curve is also reported RELATIVE TO ITS OWN 1x, and
BITS PER BYTE is recorded in the JSON as the one cross-tokenizer figure that does compare.

THE ONE THING THAT MAKES A CURVE MEAN ANYTHING is that every context length must score THE
SAME TARGET TOKENS. The obvious implementation -- chop the stream into windows of T and
average over them -- does not: eight windows of 512 cover the first 4,096 tokens of the corpus
and eight of 8,192 cover the first 65,536, so the resulting curve confounds "more context"
with "different text" and moves around non-monotonically for reasons that have nothing to do
with the model. Here anchor positions p are fixed first, and length T is evaluated on
stream[p-T : p]. The final --tail targets are stream[p-tail : p] whatever T is; only the amount
of prefix in front of them changes. Anchors are placed at MATCHED FRACTIONS of each model's
stream, and both models read the same files in the same order, so the two rows are scored over
the same region of the same corpus.

WHAT IS MEASURED:

  perplexity on the FIXED targets    the result. Identical tokens at every length, varying only
                                     in how much context precedes them. Falling means the extra
                                     context is being used; flat means the model tolerates the
                                     longer window without exploiting it; rising means it is
                                     out of distribution. Some rise is expected and is not by
                                     itself a failure -- a long window spans several unrelated
                                     documents, whose text is not informative about the target.
                                     What distinguishes a model is whether the rise is gradual
                                     or a cliff.
  perplexity over ALL positions      the familiar curve, kept for comparability with the
  of the window                      literature. NOT like-for-like across T -- a longer window
                                     reaches further back into the corpus -- and the weaker
                                     evidence anyway, since most tokens are predictable from
                                     their immediate neighbours.
  NLL by ABSOLUTE position           where in the window the degradation begins, which locates
                                     the failure instead of only detecting it.

and, for pe="ssm", the state space diagnostics measured at each test length. Not plotted; kept
in the JSON, because tau at TEST length is not the same quantity as tau during training.

EVALUATION DATA is read as WHOLE FILES, concatenated into one contiguous token stream with an
<eos> between files, not as the ~200-word packets the trainer uses. A stream stitched from
short independent fragments has no long-range structure in it, so every model would score the
same on it and the experiment would measure nothing.

COST. ZetaGPT's attention is the explicit O(T^2) form (model/zetagpt.py materialises the
(B, nh, T, T) score matrix), so its reach is set by memory: at n_head=8 in fp32 the score
matrix alone is 6 GiB at T=8192 and 388 GiB at T=65536. Lengths whose estimate exceeds
--max_attn_gb are SKIPPED and recorded as skipped rather than attempted and lost to an
out-of-memory kill; a real OOM is also caught and recorded. --dtype bf16 roughly halves the
requirement and buys about one rung.
"""
import argparse
import glob
import json
import math
import os
import random

import torch

import default_config as config
from helpers import corpus_files, resolve_device, progress
from model import ZetaGPT
from model.ssm import collect_stats, layer_stats
from tokenizer import BPETokenizer

STAGE = "pretrain"
MULTIPLIERS = [1, 2, 4, 8, 16, 32, 64, 128]
LN2 = math.log(2.0)

INK, GREY, PURPLE, ORANGE = "#1A1A1A", "#4B5963", "#7b2cbf", "#C2610A"


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description="perplexity at 1x..128x the training length, ZetaGPT vs GPT-2 small")
    ap.add_argument("--checkpoint", default="",
                    help="pretrain checkpoint; default = the newest "
                         "checkpoints/pretrain/checkpoint_*_pretrain.pt")
    ap.add_argument("--gpt2", default=config.DISTILL["student"],
                    help="HuggingFace id of the comparison model drawn in the second row; "
                         "empty string evaluates ZetaGPT alone")
    ap.add_argument("--data_dir", default=config.PRETRAIN_DIR,
                    help="directory of evaluation text, read recursively")
    ap.add_argument("--base", type=int, default=0,
                    help="the 1x length in tokens. 0 = the checkpoint's block_size. Pass the "
                         "context window the run actually trained at if it differed from the "
                         "scheme's -- every multiplier is relative to this, so it is the one "
                         "number the whole figure depends on")
    ap.add_argument("--multipliers", default=",".join(str(m) for m in MULTIPLIERS))
    ap.add_argument("--probes", type=int, default=8,
                    help="anchor positions averaged over. Every length is evaluated at the SAME "
                         "anchors, so the scored targets are identical and only the prefix "
                         "length changes; cost per length is probes * T^2")
    ap.add_argument("--anchor_start", type=float, default=0.25,
                    help="fraction of the stream the first anchor sits at; the rest are spread "
                         "evenly from there to the end. A fraction rather than a token index so "
                         "that both models, whose tokenizers give streams of different lengths, "
                         "are scored over the same region of the same text")
    ap.add_argument("--tail", type=int, default=128,
                    help="how many tokens before each anchor are scored. These are the fixed "
                         "targets: the same tokens at every context length, differing only in "
                         "how much prefix precedes them. Must be smaller than the 1x length")
    ap.add_argument("--pos_bins", type=int, default=24,
                    help="log-spaced bins for the NLL-by-position panel")
    ap.add_argument("--logit_chunk", type=int, default=512,
                    help="positions per head/log-softmax chunk; bounds the T x vocab tensor, "
                         "which at T=32768 would otherwise be 6.6 GiB in fp32")
    ap.add_argument("--max_attn_gb", type=float, default=8.0,
                    help="skip any length whose attention score matrix is estimated above this")
    ap.add_argument("--force", action="store_true",
                    help="attempt every length regardless of the estimate")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(config.PLOT_DIR, STAGE,
                                                  "content_extrapolation.pdf"))
    ap.add_argument("--json", default="",
                    help="where the results go; default outputs/eval/content_extrapolation.json")
    ap.add_argument("--plot_only", action="store_true",
                    help="redraw from the saved JSON; loads no model and reads no corpus")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# the two models, behind one interface
# --------------------------------------------------------------------------- #
# A "spec" is everything the sweep needs to know about a model and nothing else:
#
#   name          for the row title
#   encode/decode text <-> ids, in THAT MODEL'S OWN tokenizer
#   eos           separator between concatenated files
#   hidden/head   the forward, split so the vocabulary projection can be chunked over time
#   n_head        for the attention-memory estimate
#   max_pos       a hard architectural ceiling (GPT-2: 1024), or 0 for none
#   train_len     the length it was trained at, drawn on the figure
#   ssm           whether the state space diagnostics exist to be collected
#
# Splitting the forward is not fastidiousness: materialising all the logits at once is what
# makes long windows impossible for reasons that have nothing to do with the model. At T=32768
# over a 50k vocabulary that is a single 6.6 GiB fp32 tensor, larger than the attention matrix
# it is supposed to be accompanying.
def spec_zetagpt(args, device, dtype, log):
    if not os.path.isfile(config.BPE_PATH):
        raise SystemExit(f"no tokenizer at {config.BPE_PATH} -- run stage 3 first")
    tok = BPETokenizer.load(config.BPE_PATH)
    path = find_checkpoint(args.checkpoint)
    ck = torch.load(path, map_location="cpu")
    saved = ck.get("model_cfg") or dict(config.MODEL)
    cfg = {k: v for k, v in saved.items() if k != "vocab_size"}
    cfg.setdefault("pe", "ssm")
    model = ZetaGPT(vocab_size=len(tok), **cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base = args.base or int(cfg.get("block_size", 512))
    log(f"[extrapolation] ZetaGPT: {os.path.relpath(path, config.ROOT)} "
        f"step {ck.get('step')}/{ck.get('total')}  pe={cfg.get('pe')}  vocab={len(tok):,}")
    return {
        "name": f"ZetaGPT ({cfg.get('n_layer')}L-{cfg.get('n_embd')}d, pe={cfg.get('pe')})",
        "key": "zetagpt", "cfg": cfg, "checkpoint": os.path.relpath(path, config.ROOT),
        "step": ck.get("step"), "total": ck.get("total"),
        "encode": lambda t: tok(t, add_special_tokens=False)["input_ids"],
        "decode": lambda ids: tok.decode(ids, skip_special_tokens=False),
        "eos": tok.eos_token_id, "n_head": cfg["n_head"], "max_pos": 0,
        "train_len": base, "ssm": cfg.get("pe", "ssm") == "ssm", "model": model,
        "hidden": lambda ids: model.hidden(input_ids=ids), "head": model.head,
        "tokenizer": f"pipeline BPE ({len(tok):,})",
    }


def spec_gpt2(name, device, dtype, log):
    """GPT-2 small, or None when transformers is unavailable or the weights cannot be fetched.
    A missing comparison model must not take the whole evaluation down with it."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:                                        # noqa: BLE001
        log(f"[extrapolation] GPT-2 row skipped: transformers not importable ({e})")
        return None
    try:
        tok = AutoTokenizer.from_pretrained(name, cache_dir=config.MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir=config.MODEL_DIR)
    except Exception as e:                                        # noqa: BLE001
        log(f"[extrapolation] GPT-2 row skipped: could not load {name!r} ({e})")
        return None
    # the tokenizer refuses nothing, but it warns on every sequence past its nominal maximum,
    # and this script feeds it whole files on purpose
    tok.model_max_length = int(1e9)
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    c = model.config
    max_pos = int(getattr(c, "n_positions", None) or getattr(c, "max_position_embeddings", 0))
    n_head = int(getattr(c, "n_head", None) or getattr(c, "num_attention_heads", 12))
    trunk = getattr(model, "transformer", None)
    log(f"[extrapolation] {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M "
        f"params  n_positions={max_pos}  vocab={len(tok):,}")
    return {
        "name": f"{name} (learned absolute positions, {max_pos} rows)",
        "key": name, "cfg": {"n_head": n_head, "n_positions": max_pos},
        "checkpoint": name, "step": None, "total": None,
        "encode": lambda t: tok(t)["input_ids"],
        "decode": lambda ids: tok.decode(ids, skip_special_tokens=False),
        "eos": tok.eos_token_id, "n_head": n_head, "max_pos": max_pos,
        "train_len": max_pos, "ssm": False, "model": model,
        "hidden": ((lambda ids: trunk(input_ids=ids).last_hidden_state) if trunk is not None
                   else None),
        "head": getattr(model, "lm_head", None),
        "tokenizer": f"GPT-2 BPE ({len(tok):,})",
    }


def find_checkpoint(given):
    """The pretrain checkpoint. Its name carries the configuration that produced it
    (checkpoint_<model>_<pe>_pretrain.pt), so it cannot be constructed from the stage alone --
    hence the glob, newest first. `last.pt` is matched too, for checkpoints written before
    that convention."""
    if given:
        if not os.path.isfile(given):
            raise SystemExit(f"no checkpoint at {given}")
        return given
    d = os.path.join(config.CHECKPOINT_DIR, STAGE)
    found = sorted(set(glob.glob(os.path.join(d, f"checkpoint_*_{STAGE}.pt"))
                       + glob.glob(os.path.join(d, "checkpoint_*.pt"))
                       + glob.glob(os.path.join(d, "last.pt"))), key=os.path.getmtime)
    if not found:
        raise SystemExit(f"no pretrain checkpoint under {d}/ -- run stage 5 first")
    return found[-1]


# --------------------------------------------------------------------------- #
# evaluation stream
# --------------------------------------------------------------------------- #
def collect_files(data_dir, seed):
    """The evaluation files, in a seeded order. Chosen ONCE and reused by every model, so the
    two rows are scored over the same text rather than over two different samples of it."""
    files = corpus_files(data_dir)
    if not files:
        raise SystemExit(f"no corpus files under {data_dir}")
    files = list(files)
    random.Random(seed).shuffle(files)
    return files


def token_stream(files, spec, need, log):
    """One contiguous stream of at least `need` ids in this model's own tokenizer, built from
    whole files, <eos>-separated.

    Whole files, because the quantity under test is whether distant context helps: a stream
    stitched from shuffled 200-word packets has no distant context to use, so it would report
    a flat curve for any model whatsoever."""
    ids, used = [], 0
    for fp in progress(files, desc=f"[extrapolation] tokenizing ({spec['key']})",
                       total=len(files)):
        try:
            text = open(fp, "r", encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if not text.strip():
            continue
        ids.extend(spec["encode"](text))
        if spec["eos"] is not None:
            ids.append(spec["eos"])
        used += 1
        if len(ids) >= need:
            break
    log(f"[extrapolation] {spec['key']}: stream {len(ids):,} tokens from {used:,} files")
    return ids


# --------------------------------------------------------------------------- #
# one window
# --------------------------------------------------------------------------- #
@torch.no_grad()
def window_nll(spec, ids, chunk):
    """Per-token NLL for positions 1..T-1 of one window, as a float32 CPU tensor of length T-1.

    The hidden states are computed once and the vocabulary projection applied in chunks along
    time, so no T x vocab tensor is ever alive. A model that exposes no trunk/head split falls
    back to the whole-logits forward, which is fine at the lengths such a model can reach."""
    T = ids.shape[1]
    out = []
    if spec["hidden"] is not None and spec["head"] is not None:
        h = spec["hidden"](ids)                             # (1, T, d)
        for a in range(0, T - 1, chunk):
            b = min(a + chunk, T - 1)
            lg = spec["head"](h[:, a:b]).float()            # (1, b-a, V)
            lp = lg.log_softmax(-1).gather(-1, ids[:, a + 1:b + 1].unsqueeze(-1)).squeeze(-1)
            out.append((-lp[0]).detach().float().cpu())
            del lg, lp
        del h
    else:
        lg = spec["model"](input_ids=ids).logits[:, :-1].float()
        lp = lg.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        out.append((-lp[0]).detach().float().cpu())
        del lg, lp
    return torch.cat(out)


def bin_edges(T, n_bins):
    """Log-spaced absolute-position edges, 1..T-1. Absolute rather than relative so every
    length shares one x-axis: the question is where degradation starts in TOKENS, and a
    relative axis would put the 512th token of a 512-window and of a 65536-window in different
    places."""
    lo, hi = 1.0, float(max(T - 1, 2))
    return [lo * (hi / lo) ** (i / n_bins) for i in range(n_bins + 1)]


def attn_estimate_gb(T, n_head, itemsize):
    """Peak bytes of the attention score matrix, as GiB. Three copies of (1, nh, T, T) is the
    honest figure -- the scores, the softmax result, and the boolean causal mask that
    model/zetagpt.py builds at (T, T) -- so this is deliberately not the optimistic count."""
    return (3.0 * n_head * T * T * itemsize + T * T) / 2 ** 30


def is_oom(exc):
    m = str(exc).lower()
    return ("out of memory" in m or "can't allocate" in m or "cuda error" in m
            or "mps backend out of memory" in m)


# --------------------------------------------------------------------------- #
# the sweep
# --------------------------------------------------------------------------- #
def plan(lengths, spec, itemsize, args, log):
    """Split the requested lengths into those that will be attempted and those that will not,
    BEFORE any data is read or any forward pass is run.

    This has to happen first, because the anchor positions depend on the largest length that is
    actually going to run. It is also where GPT-2's wall appears: a length past its position
    table is not a length it fails at, it is a length it cannot be asked about, and the two
    deserve different words."""
    runnable, skipped = [], []
    for T in lengths:
        est = attn_estimate_gb(T, spec["n_head"], itemsize)
        base = {"T": T, "attn_gb": round(est, 3), "probes": 0}
        if spec["max_pos"] and T > spec["max_pos"]:
            skipped.append(dict(base, status="beyond_position_table"))
            log(f"[extrapolation] {spec['key']}: T={T:>7,}  NOT EVALUABLE: the position table "
                f"has {spec['max_pos']} rows")
        elif est > args.max_attn_gb and not args.force:
            skipped.append(dict(base, status="skipped_estimate"))
            log(f"[extrapolation] {spec['key']}: T={T:>7,}  SKIPPED: attention needs "
                f"~{est:.1f} GiB > --max_attn_gb {args.max_attn_gb}")
        else:
            runnable.append(T)
    return runnable, skipped


def anchors_for(stream_len, max_T, probes, start_frac):
    """Anchor positions, each with at least `max_T` tokens of stream in front of it, at matched
    FRACTIONS of the stream so that two models with different tokenizers -- and therefore
    different stream lengths -- are scored over the same region of the same text."""
    first = max(int(start_frac * stream_len), max_T)
    if first > stream_len:
        return []
    if probes <= 1 or first == stream_len:
        return [first]
    step = (stream_len - first) / (probes - 1)
    return sorted({min(stream_len, int(first + i * step)) for i in range(probes)})


@torch.no_grad()
def sweep(spec, stream, runnable, anchors, args, device, dtype, log):
    """Evaluate each length at every anchor. `stream[p-T:p]` is the window; its last --tail
    positions are the fixed targets, identical across lengths."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    results, stop = [], False
    for T in runnable:
        est = attn_estimate_gb(T, spec["n_head"], itemsize)
        rec = {"T": T, "status": "ok", "probes": 0, "attn_gb": round(est, 3)}
        if stop:
            rec["status"] = "skipped_after_oom"
            log(f"[extrapolation] {spec['key']}: T={T:>7,}  SKIPPED after an earlier OOM")
            results.append(rec)
            continue

        edges = bin_edges(T, args.pos_bins)
        b_sum, b_cnt = [0.0] * args.pos_bins, [0] * args.pos_bins
        win_nll, win_cnt, fix_nll, fix_cnt, fix_bytes = 0.0, 0, 0.0, 0, 0
        tail = min(args.tail, T - 1)
        try:
            for j, p in enumerate(anchors):
                window = stream[p - T:p]
                ids = torch.tensor([window], dtype=torch.long, device=device)
                # the diagnostics cost one forward's worth of reductions, so they are taken on
                # the FIRST anchor only -- a real evaluation pass, not an extra one
                probe_ssm = (j == 0 and spec["ssm"])
                if probe_ssm:
                    collect_stats(spec["model"], True)
                nll = window_nll(spec, ids, args.logit_chunk)     # (T-1,), targets [p-T+1, p)
                if probe_ssm:
                    rec["ssm"] = layer_stats(spec["model"])
                    collect_stats(spec["model"], False)
                win_nll += float(nll.sum()); win_cnt += nll.numel()
                # THE FIXED TARGETS: the last `tail` entries score the tokens stream[p-tail:p],
                # the same at every T. Only the prefix in front of them grew. Their byte length
                # is accumulated too, because bits-per-byte is the only figure that survives a
                # change of tokenizer.
                fix_nll += float(nll[-tail:].sum()); fix_cnt += tail
                fix_bytes += len(spec["decode"](window[-tail:]).encode("utf-8"))
                pos = torch.arange(1, nll.numel() + 1, dtype=torch.float)
                idx = torch.bucketize(pos, torch.tensor(edges[1:-1])).tolist()
                for i, v in zip(idx, nll.tolist()):
                    b_sum[i] += v; b_cnt[i] += 1
                del ids, nll
                rec["probes"] = j + 1
        except RuntimeError as e:
            # torch.cuda.OutOfMemoryError subclasses RuntimeError, and the MPS and CPU
            # allocators raise a plain RuntimeError, so one clause covers every backend --
            # and anything that is NOT an allocation failure is re-raised rather than
            # silently recorded as a length the model could not reach.
            if not is_oom(e):
                raise
            if spec["ssm"]:
                collect_stats(spec["model"], False)
            rec.update(status="oom", error=str(e).splitlines()[0][:200])
            log(f"[extrapolation] {spec['key']}: T={T:>7,}  OUT OF MEMORY after "
                f"{rec['probes']} probe(s); longer lengths will be skipped")
            stop = True
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if fix_cnt:
            rec["nll_fixed"] = fix_nll / fix_cnt
            rec["ppl_fixed"] = math.exp(min(rec["nll_fixed"], 60.0))
            rec["bpb_fixed"] = fix_nll / (LN2 * max(fix_bytes, 1))
            rec["nll_all"] = win_nll / max(win_cnt, 1)
            rec["ppl_all"] = math.exp(min(rec["nll_all"], 60.0))
            rec["fixed_targets"], rec["fixed_bytes"] = fix_cnt, fix_bytes
            rec["bins"] = [{"lo": edges[i], "hi": edges[i + 1],
                            "nll": (b_sum[i] / b_cnt[i]) if b_cnt[i] else None,
                            "n": b_cnt[i]} for i in range(args.pos_bins)]
            log(f"[extrapolation] {spec['key']}: T={T:>7,}  probes={rec['probes']:<2d} "
                f"ppl_fixed={rec['ppl_fixed']:>10.3f}  bpb={rec['bpb_fixed']:>6.3f}  "
                f"ppl_window={rec['ppl_all']:>10.3f}  (attn ~{est:.2f} GiB)")
        results.append(rec)
    return results


def evaluate_model(spec, files, lengths, args, device, dtype, log):
    """One model, end to end: plan, read its own stream, place anchors, sweep."""
    itemsize = torch.empty((), dtype=dtype).element_size()
    runnable, skipped = plan(lengths, spec, itemsize, args, log)
    out = {"name": spec["name"], "key": spec["key"], "tokenizer": spec["tokenizer"],
           "train_len": spec["train_len"], "max_pos": spec["max_pos"],
           "checkpoint": spec["checkpoint"], "step": spec["step"], "total": spec["total"],
           "cfg": spec["cfg"]}
    if not runnable:
        out.update(results=sorted(skipped, key=lambda r: r["T"]), anchors=[], probes=0,
                   stream_tokens=0)
        return out
    stream = token_stream(files, spec, max(runnable) * args.probes, log)
    short = [T for T in runnable if T > len(stream)]
    runnable = [T for T in runnable if T <= len(stream)]
    skipped += [{"T": T, "status": "insufficient_data", "probes": 0,
                 "attn_gb": round(attn_estimate_gb(T, spec["n_head"], itemsize), 3)}
                for T in short]
    anchors = anchors_for(len(stream), max(runnable) if runnable else 0, args.probes,
                          args.anchor_start) if runnable else []
    if anchors:
        log(f"[extrapolation] {spec['key']}: {len(anchors)} anchor(s), first at "
            f"{anchors[0]:,} of {len(stream):,}; every length scores the same "
            f"{args.tail * len(anchors):,} target tokens")
    res = sweep(spec, stream, runnable, anchors, args, device, dtype, log) if anchors else []
    out.update(results=sorted(res + skipped, key=lambda r: r["T"]), anchors=anchors,
               probes=len(anchors), stream_tokens=len(stream))
    return out


# --------------------------------------------------------------------------- #
# figure: one row per model
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
    models = res["models"]
    n = len(models)
    fig, grid = plt.subplots(n, 2, figsize=(9.6, 3.4 * n), squeeze=False)
    cmap = plt.get_cmap("viridis")

    # ONE X-AXIS FOR EVERY ROW. Letting each row auto-scale to the lengths it managed would
    # shrink the axis of exactly the model that stopped early, hiding the finding: a row that
    # ends at 1,024 while the row above it continues to 8,192 should LOOK like it ends, not be
    # redrawn to fill the same width. The range is the union of what any model reached, so no
    # panel is mostly empty, and both panels of a row are pinned to it. Only x is shared -- the
    # perplexity levels are not comparable across tokenizers, so y stays per row.
    xs_all = sorted({r["T"] for m in models for r in m["results"]
                     if r.get("ppl_fixed") is not None})
    if not xs_all:
        xs_all = sorted({r["T"] for m in models for r in m["results"]}) or [1]
    pad = 2 ** 0.4
    xlim = (xs_all[0] / pad, xs_all[-1] * pad)

    for row, m in enumerate(models):
        rows = m["results"]
        ok = [r for r in rows if r.get("ppl_fixed") is not None]
        base = m["train_len"] or (ok[0]["T"] if ok else 1)

        # (left) perplexity against context length
        ax = grid[row][0]
        if ok:
            Ts = [r["T"] for r in ok]
            ax.plot(Ts, [r["ppl_all"] for r in ok], "o-", color=GREY, lw=1.1, ms=3.0,
                    alpha=0.9, label="window mean (not like-for-like)")
            ax.plot(Ts, [r["ppl_fixed"] for r in ok], "s--", color=ORANGE, lw=1.7, ms=4.2,
                    label=f"fixed targets ({res['tail']} tokens)")
            ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xlim(*xlim)
        ax.set_xticks(xs_all)
        # the multiple of THIS model's training length: the tick POSITIONS are shared, the
        # labels are not, because the two models were trained at different lengths and 1x is
        # the reading that matters on each row
        topax = ax.twiny()
        topax.set_xscale("log", base=2); topax.set_xlim(*xlim)
        topax.set_xticks(xs_all)
        topax.set_xticklabels([f"${T / base:g}\\times$" for T in xs_all], fontsize=7)
        topax.grid(False); topax.tick_params(length=2)
        wall = [r for r in rows if r.get("status") == "beyond_position_table"]
        other = [r for r in rows if r.get("ppl_fixed") is None and r not in wall]
        notes = []
        if wall:
            notes.append("position table ends at "
                         f"{m['max_pos']:,}: " + ", ".join(_k(r["T"]) for r in wall)
                         + " not evaluable")
        if other:
            notes.append("not reached: " + ", ".join(_k(r["T"]) for r in other))
        if notes:
            ax.text(0.02, 0.03, "\n".join(notes), transform=ax.transAxes, fontsize=6.6,
                    color=GREY, style="italic", va="bottom")
        ax.set_title(f"{m['name']}\nperplexity vs context length", weight="bold", fontsize=9)
        ax.set_xlabel("context length $T$ (tokens)"); ax.set_ylabel("perplexity")
        ax.legend(fontsize=7)

        # (right) NLL by absolute position inside the window
        ax = grid[row][1]
        for i, r in enumerate(ok):
            xs = [0.5 * (b["lo"] + b["hi"]) for b in r["bins"] if b["nll"] is not None]
            ys = [b["nll"] for b in r["bins"] if b["nll"] is not None]
            if xs:
                ax.plot(xs, ys, lw=1.2, color=cmap(0.12 + 0.76 * (i / max(len(ok) - 1, 1))),
                        label=f"$T={r['T']:,}$")
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.7, xlim[1])          # shared with the other rows; positions start at 1
        # here the rule IS informative: everything to its right is a position this model never
        # saw during training, so this is where extrapolation begins
        ax.axvline(base, color=GREY, lw=0.9, ls=":")
        ax.annotate("training length", xy=(base, 0.97), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points", rotation=90, va="top",
                    ha="left", fontsize=6.8, color=GREY)
        ax.set_title(f"{m['name']}\nNLL by absolute position", weight="bold", fontsize=9)
        ax.set_xlabel("position in the window (tokens)"); ax.set_ylabel("NLL (nats/token)")
        ax.legend(fontsize=6.4, ncol=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _k(T):
    return f"{T // 1024}k" if T >= 1024 and T % 1024 == 0 else f"{T:,}"


# --------------------------------------------------------------------------- #
def summarise(res, log):
    """The table the report would quote: per model, the curve and what it did relative to
    its own 1x. Levels are NOT compared across models -- different tokenizers -- so the
    cross-model column is bits per byte, which is."""
    for m in res["models"]:
        rows = [r for r in m["results"] if r.get("ppl_fixed") is not None]
        log("")
        log(m["name"])
        log("-" * 88)
        log(f"  tokenizer {m['tokenizer']}   trained at {m['train_len']:,} tokens   "
            f"probes {m['probes']}   stream {m['stream_tokens']:,} tokens")
        log("")
        log(f"  {'T':>9} {'x train':>9} {'ppl (fixed)':>13} {'vs 1x':>8} {'bits/byte':>10}"
            f" {'ppl (window)':>13}")
        ref = rows[0]["ppl_fixed"] if rows else float("nan")
        for r in m["results"]:
            x = f"{r['T'] / max(m['train_len'], 1):g}x"
            if r.get("ppl_fixed") is None:
                log(f"  {r['T']:>9,} {x:>9} {'--':>13} {'--':>8} {'--':>10} {'--':>13}"
                    f"   {r['status']}")
                continue
            log(f"  {r['T']:>9,} {x:>9} {r['ppl_fixed']:>13.3f} {r['ppl_fixed'] / ref:>8.3f} "
                f"{r['bpb_fixed']:>10.4f} {r['ppl_all']:>13.3f}")
        log("-" * 88)
    log("")
    log("  ppl (fixed) is the result: identical target tokens at every length, differing only")
    log("  in how much context precedes them. PERPLEXITY IS NOT COMPARABLE BETWEEN THE MODELS")
    log("  -- different tokenizers, and perplexity is per token -- so compare the 'vs 1x'")
    log("  column, or bits/byte, which is tokenizer-independent.")


def main():
    args = parse_args()
    def log(m): print(m, flush=True)

    json_path = args.json or os.path.join(config.OUTPUT_DIR, "eval",
                                          "content_extrapolation.json")
    if args.plot_only:
        if not os.path.isfile(json_path):
            raise SystemExit(f"--plot_only: no results at {json_path}")
        res = json.load(open(json_path, "r", encoding="utf-8"))
        summarise(res, log)
        log(f"[extrapolation] [figure] {figure(res, args.out)}")
        return

    device = resolve_device(args.gpu)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    log(f"[extrapolation] device={device} dtype={args.dtype}")

    specs = [spec_zetagpt(args, device, dtype, log)]
    if args.gpt2:
        g = spec_gpt2(args.gpt2, device, dtype, log)
        if g is not None:
            specs.append(g)

    base = specs[0]["train_len"]
    if args.tail >= base:
        raise SystemExit(f"--tail {args.tail} must be smaller than the 1x length {base}, or the "
                         f"shortest context cannot score the fixed targets")
    mults = [int(x) for x in args.multipliers.split(",") if x.strip()]
    lengths = [base * m for m in mults]

    files = collect_files(args.data_dir, args.seed)
    if os.path.abspath(args.data_dir) == os.path.abspath(config.PRETRAIN_DIR):
        log("[extrapolation] NOTE: this is ZetaGPT's own pretraining corpus. The LEVEL of its "
            "perplexities is therefore optimistic, and GPT-2 has never seen this text at all, "
            "so the two rows are not on equal footing on level -- compare the shapes. Point "
            "--data_dir at text neither model trained on to remove the asymmetry.")

    models = [evaluate_model(s, files, lengths, args, device, dtype, log) for s in specs]
    res = {"base": base, "tail": args.tail, "dtype": args.dtype,
           "data_dir": os.path.relpath(args.data_dir, config.ROOT),
           "multipliers": mults, "models": models}

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    summarise(res, log)
    log(f"[extrapolation] [results] {json_path}")
    log(f"[extrapolation] [figure] {figure(res, args.out)}")


if __name__ == "__main__":
    main()
