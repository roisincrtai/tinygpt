"""
tools/watermark_demo.py -- ONE-OFF. Two statistical watermarks, generated and then detected.

    python -m tools.watermark_demo
    python -m tools.watermark_demo --context_width 1     # reproduce the degenerate loop
    python -m tools.watermark_demo --prompt "The history of the printing press" --max_new 300

NOT PART OF THE PIPELINE, and not to be tracked: it answers a question about watermarking, it
is not a stage. Delete it when the question is answered.

WHAT IT DOES. It samples the SAME prompt three ways from one checkpoint --

    plain     ordinary sampling, no watermark                     the control
    kgw       red-green logit bias    (Kirchenbauer et al. 2023)   distribution IS shifted
    gumbel    exponential sampling    (Aaronson)                   distribution is NOT shifted

-- and then runs BOTH detectors over ALL of the samples. That last part is the demonstration:
a detector must fire on the text its own scheme wrote and NOT on the other two, or it is
measuring style rather than a key.

BOTH DETECTORS SEE ONLY TOKEN IDS. No model, no logits, no API call -- that is the property
that makes these schemes deployable, and it is why detection here does not load the model a
second time. The key and the tokenizer are the whole detector.

TWO FAULTS THIS SCRIPT ORIGINALLY HAD, both found by running it, and both fixed here because
they are the two everyone meets:

    1. THE CONTEXT WIDTH WAS ONE TOKEN, and the exponential race is DETERMINISTIC given its
       pseudorandom vector -- so with h = 1 the next token is a function of the previous one
       alone, `next = f(prev)`, and a deterministic map iterated on a finite vocabulary MUST
       enter a cycle. The first run duly emitted "the big little steam engine" two hundred
       times. It is not a quality problem and no amount of training fixes it: the scheme spent
       its randomness on the key and has none left. h > 1 makes the seed depend on more
       history, and cycles become vanishingly unlikely. (The other cure, which this script does
       not implement, is to abandon context hashing and index a fixed key SEQUENCE by position
       -- which is exactly what Kuditipudi et al. do, aligning by edit distance at detection.)

    2. THE z-TEST ASSUMES INDEPENDENT DRAWS, and repeated text breaks that assumption flat. On
       the looping sample the detector saw the same handful of (context, token) pairs two
       hundred times: one observation, counted two hundred times. It reported 0/200 green,
       z = -8.16, and the exponential detector reported z = +67.5 -- neither of which was
       evidence of anything. So both detectors now ALSO score each UNIQUE (context, token)
       pair once, which is what Kirchenbauer's follow-up (On the Reliability of Watermarks for
       Large Language Models) does. Both numbers are printed: the raw one is what a naive
       detector would say, the unique one is what the text actually supports.

THE ENTROPY CEILING IS THE THING TO WATCH. Neither scheme can mark a token the model was
already certain about: KGW's +delta does not move a peaked softmax, and the exponential race
picks the same token whatever the pseudorandom draw was. A greedy run carries no watermark at
all, by construction, and the script refuses --temperature 0 for that reason.
"""
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import default_config as config                                       # noqa: E402
import chat                                                           # noqa: E402
from helpers.kv_cache import Cache                                    # noqa: E402
from helpers.utils import progress                                    # noqa: E402

MOD = 2 ** 63 - 1
PAD = -1            # stands in for history that does not exist yet, at the very first tokens


# --------------------------------------------------------------------------- #
# the two keyed pseudorandom streams, over an h-TOKEN CONTEXT
# --------------------------------------------------------------------------- #
# BOTH SCHEMES DERIVE THEIR RANDOMNESS FROM (key, the last h tokens) AND NOTHING ELSE. That is
# what lets a detector reconstruct the stream from the text alone: it reads the h tokens before
# each position, re-seeds, and asks what the generator would have been looking at.
#
# h TRADES ROBUSTNESS AGAINST DEGENERACY. Small h survives editing better -- changing one token
# corrupts the test at h positions, so h = 1 corrupts exactly one -- but it makes the keyed
# stream repeat whenever the text repeats, and for the exponential scheme it makes generation a
# deterministic map with cycles it cannot escape. h = 4 is the usual compromise.
def _seed(key, ctx):
    s = key % MOD
    for t in ctx:
        s = (s * 6364136223846793005 + (t + 2) * 1442695040888963407) % MOD
    return s


def green_mask(ctx, vocab, gamma, key):
    """KGW: the pseudorandom GREEN LIST for the token following `ctx`, as a bool mask.

    A fresh permutation of the vocabulary per context, cut at gamma. The partition is a
    property of (key, ctx) only, so the detector computes exactly the same one."""
    g = torch.Generator().manual_seed(_seed(key, ctx))
    mask = torch.zeros(vocab, dtype=torch.bool)
    mask[torch.randperm(vocab, generator=g)[:max(1, int(gamma * vocab))]] = True
    return mask


def uniforms(ctx, vocab, key):
    """Aaronson: the pseudorandom vector r in (0,1)^V for the token following `ctx`."""
    g = torch.Generator().manual_seed(_seed(key, ctx))
    return torch.rand(vocab, generator=g)


def context_of(history, h):
    """The h tokens before the next position, left-padded when the history is shorter.

    ONE FUNCTION, used by the sampler and by both detectors, so the context a token was
    generated under and the context it is scored under cannot come apart -- which would look
    exactly like a watermark that does not work."""
    tail = list(history[-h:])
    return tuple([PAD] * (h - len(tail)) + tail)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample(model, prompt_ids, device, args, scheme):
    """`args.max_new` tokens continuing `prompt_ids`, under one of the three schemes.

    THE CACHED DECODE IS chat.generate's, reproduced rather than called, because a watermark is
    an intervention BETWEEN the logits and the sampled token and chat.generate exposes no hook
    there. Everything else -- the KV cache, the sliding window, the progress bar -- is the same
    so that the only difference between the control and the watermarked runs is the scheme."""
    cur, out = list(prompt_ids), []
    cache = Cache(len(model.blocks))
    fed, V = 0, model.head.out_features
    for _ in progress(range(args.max_new), desc=f"[{scheme}] sampling"):
        ctx_win = cur[-args.max_len:] if args.max_len else cur
        if fed and len(ctx_win) == len(cur):
            x = torch.tensor([cur[fed:]], device=device)
        else:
            if len(ctx_win) != len(cur):
                cache = Cache(len(model.blocks))
            x = torch.tensor([ctx_win], device=device)
        logits = model.head(model.hidden_states(input_ids=x, cache=cache)[:, -1])[0].float()
        fed = len(cur)

        logits = logits / args.temperature
        if args.top_k > 0:
            thresh = torch.topk(logits, min(args.top_k, V)).values[-1]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        key_ctx = context_of(cur, args.context_width)

        if scheme == "kgw":
            # THE WATERMARK IS A CONSTANT ADDED TO THE GREEN LOGITS, before the softmax. On a
            # peaked distribution delta does not change the argmax, which is the whole reason
            # the scheme is usable: it spends the entropy that is there and no more.
            logits = logits + args.delta * green_mask(key_ctx, V, args.gamma,
                                                      args.key).to(device)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))
        elif scheme == "gumbel":
            # THE LOGITS ARE NOT TOUCHED. The token is chosen by the exponential race
            #     argmax_i  r_i ** (1 / p_i)      ==  argmax_i  log(r_i) / p_i
            # which is a Gumbel-max draw from p itself: the MARGINAL DISTRIBUTION IS EXACTLY p,
            # so nothing about the text is biased. The model's own randomness has been replaced
            # by randomness the detector can regenerate, and that substitution is the mark.
            probs = F.softmax(logits, dim=-1)
            r = uniforms(key_ctx, V, args.key).to(device)
            score = torch.log(r.clamp_min(1e-12)) / probs.clamp_min(1e-12)
            nxt = int(torch.argmax(score.masked_fill(probs <= 0, float("-inf"))))
        else:
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))

        if args.eos_id is not None and nxt == args.eos_id:
            break
        cur.append(nxt); out.append(nxt)
    return out


# --------------------------------------------------------------------------- #
# detection -- token ids in, z-scores out. No model.
# --------------------------------------------------------------------------- #
def _z(hits, n, gamma):
    return (hits - gamma * n) / math.sqrt(max(n * gamma * (1 - gamma), 1e-12)) if n else 0.0


def detect_kgw(prompt_ids, ids, vocab, gamma, key, h):
    """Count green tokens and z-test against Binomial(T, gamma), RAW and OVER UNIQUE CONTEXTS.

    Under the null -- text written without this key -- each token falls in its position's green
    list with probability gamma, INDEPENDENTLY, because the partition is pseudorandom and the
    writer knew nothing of it:

        z = (|s|_G - gamma T) / sqrt(T gamma (1 - gamma))

    THE INDEPENDENCE IS THE FRAGILE PART. Repeated text presents the same (context, token) pair
    over and over, and the raw count treats one observation as many -- which is how a looping
    sample reported 0/200 and z = -8.16, a number that says nothing about anything. Scoring
    each distinct pair once restores the assumption the test is written under. Both are
    returned: the raw z is what a naive detector claims, the unique z is what the text supports.
    z > 4 is p < 3.2e-5."""
    seen = set()
    hits = n = hits_u = n_u = 0
    hist = list(prompt_ids)
    for t in progress(ids, desc="[kgw] detecting", total=len(ids)):
        ctx = context_of(hist, h)
        green = bool(green_mask(ctx, vocab, gamma, key)[t])
        hits += green; n += 1
        if (ctx, t) not in seen:
            seen.add((ctx, t)); hits_u += green; n_u += 1
        hist.append(t)
    return (_z(hits, n, gamma), hits, n), (_z(hits_u, n_u, gamma), hits_u, n_u)


def detect_gumbel(prompt_ids, ids, vocab, key, h):
    """Aaronson's score, sum of -log(1 - r_x), RAW and OVER UNIQUE CONTEXTS.

    Under the null, r at the observed token is Uniform(0,1), so each term is Exponential(1) --
    mean 1, variance 1 -- and the sum over n tokens has mean n and variance n. Under the
    watermark the race SELECTED a token whose r was large, so the terms are inflated:

        z = (S - n) / sqrt(n)

    Nothing here needs the probabilities the generator used, which is the point: the detector
    never sees the model. The same repetition caveat applies and for the same reason -- a text
    that loops inflates this score without adding evidence, so the unique count is the honest
    one."""
    seen = set()
    S, n, S_u, n_u = 0.0, 0, 0.0, 0
    hist = list(prompt_ids)
    for t in progress(ids, desc="[gumbel] detecting", total=len(ids)):
        ctx = context_of(hist, h)
        term = -math.log(max(1.0 - float(uniforms(ctx, vocab, key)[t]), 1e-12))
        S += term; n += 1
        if (ctx, t) not in seen:
            seen.add((ctx, t)); S_u += term; n_u += 1
        hist.append(t)
    z = (S - n) / math.sqrt(n) if n else 0.0
    z_u = (S_u - n_u) / math.sqrt(n_u) if n_u else 0.0
    return (z, S, n), (z_u, S_u, n_u)


def parse_args():
    p = argparse.ArgumentParser(description="two statistical watermarks, generated and detected")
    p.add_argument("--ckpt", default=os.path.join(
        config.CHECKPOINT_DIR, "pretrain", "checkpoint_zetagpt-s_ssm_pretrain.pt"))
    p.add_argument("--prompt", default="The development of the steam engine")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0,
                   help="0 = off. TOP-K IS A WATERMARK KILLER at small values: it removes the "
                        "entropy both schemes spend, so leave it off for the demonstration")
    p.add_argument("--context_width", type=int, default=4,
                   help="h, the number of previous tokens the keyed stream is seeded from. "
                        "1 reproduces the degenerate loop the exponential scheme falls into")
    p.add_argument("--gamma", type=float, default=0.25, help="KGW green-list fraction")
    p.add_argument("--delta", type=float, default=2.0, help="KGW logit bias")
    p.add_argument("--key", type=int, default=15485863)
    p.add_argument("--gpu", default="auto")
    p.add_argument("--max_len", type=int, default=0)
    p.add_argument("--dataset", default="hh")       # only for build_from_checkpoint's fallback
    p.add_argument("--data_dir", default=config.DOWNLOAD_DIR)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.temperature <= 0:
        raise SystemExit("[watermark] --temperature 0 is greedy decoding: there is no entropy "
                         "to carry a watermark, and both schemes would be no-ops. Use 1.0.")
    if a.context_width < 1:
        raise SystemExit("[watermark] --context_width must be at least 1.")
    return a


def main():
    args = parse_args()
    device = chat.resolve_device(args.gpu)
    sd, cfg, ck = chat.read_checkpoint(args.ckpt)
    if sd is None:
        raise SystemExit(f"[watermark] no usable checkpoint at {args.ckpt}")
    model, tok, max_len = chat.build_from_checkpoint(sd, cfg, args, device, ck)
    model.load_state_dict(sd, strict=False)
    model.eval()
    args.max_len = max_len
    args.eos_id = getattr(tok, "eos_token_id", None)
    V = model.head.out_features
    h = args.context_width

    prompt_ids = tok(args.prompt, add_special_tokens=False)["input_ids"]
    print(f"\n[watermark] checkpoint {args.ckpt}")
    print(f"[watermark] vocabulary {V:,}, context {max_len:,}, temperature {args.temperature}, "
          f"top_k {args.top_k or 'off'}")
    print(f"[watermark] KGW gamma={args.gamma} delta={args.delta}, key={args.key}, "
          f"context width h={h}")
    print(f"[watermark] prompt: {args.prompt!r} ({len(prompt_ids)} tokens)\n")

    texts, toks = {}, {}
    for scheme in ("plain", "kgw", "gumbel"):
        toks[scheme] = sample(model, prompt_ids, device, args, scheme)
        texts[scheme] = chat.decode(tok, toks[scheme])

    for scheme in ("plain", "kgw", "gumbel"):
        print("=" * 78)
        print(f"{scheme.upper()}  ({len(toks[scheme])} tokens)")
        print("=" * 78)
        print(args.prompt + texts[scheme], "\n")

    # THE TABLE IS THE RESULT. A detector that fires on its own text and on nothing else is
    # detecting a KEY. One that fires on everything is detecting a writing style, and would
    # accuse the control -- which is the false-positive rate that decides whether any of this
    # can be deployed. The `uniq` columns are the same tests over distinct (context, token)
    # pairs; where `n` and `uniq n` differ sharply the text repeated, and only `uniq` means
    # anything there.
    print("=" * 78)
    print("DETECTION -- z-scores. Both detectors see token ids only: no model, no logits.")
    print("=" * 78)
    print(f"{'text':<9}{'KGW z':>9}{'green/n':>12}{'KGW z uniq':>12}{'green/n':>12}"
          f"{'Gum z':>9}{'Gum z uniq':>12}{'uniq n':>9}")
    for scheme in ("plain", "kgw", "gumbel"):
        (zk, hk, nk), (zku, hku, nku) = detect_kgw(prompt_ids, toks[scheme], V,
                                                   args.gamma, args.key, h)
        (zg, _, _), (zgu, _, ngu) = detect_gumbel(prompt_ids, toks[scheme], V, args.key, h)
        print(f"{scheme:<9}{zk:>9.2f}{f'{hk}/{nk}':>12}{zku:>12.2f}{f'{hku}/{nku}':>12}"
              f"{zg:>9.2f}{zgu:>12.2f}{ngu:>9}")
    print("\nz > 4 is p < 3.2e-5. Expect the diagonal large and everything else near 0; the")
    print("control row is the false-positive check. Where `uniq n` is far below n the text")
    print("repeated, the raw z counted one observation many times, and only `uniq` is evidence.")
    print("If the diagonal is weak, the prompt left little entropy -- that is the ceiling, not")
    print("a bug: neither scheme can mark a token the model was already sure of.\n")


if __name__ == "__main__":
    main()
