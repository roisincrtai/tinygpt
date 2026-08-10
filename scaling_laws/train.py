"""
scaling_laws/train.py -- one grid point: a model of size N trained from scratch on D tokens,
returning the held-out loss the scaling law is fitted to.

    run_point(point, data, tok, args, device, log) -> dict

WHY NOT helpers.lm.train. That loop is the pipeline's: it encodes preference-style records,
length-normalises the loss per sequence, and writes stage-named checkpoints and figures. Every
one of those is right for a stage and wrong here. A scaling law is read off the loss in NATS
PER TOKEN, so the objective has to be the plain token mean -- length normalisation weights a
short sequence's tokens more heavily than a long one's, which is a defensible training choice
and an indefensible measurement. And twenty grid points do not want twenty checkpoints.

THE DATA A POINT SEES IS ITS BUDGET AND NOTHING MORE. Windows are drawn from train_ids[:D],
so a point with D = 2M literally cannot read the 3-millionth token. Steps are D / (batch *
context), one pass' worth, so no token is expected twice: repeated data lowers loss for a
reason that is not scale, and it would enter the grid unevenly -- only at the budgets that
exceed the corpus -- which is exactly where the data exponent is being read.

THE LEARNING RATE MOVES WITH WIDTH. A single rate across a 4x width range is a confound, not a
control: it is too small for the narrow models or too large for the wide ones, and either way
the N-axis picks up a bend that has nothing to do with capacity. The default rule is
lr(d) = lr_ref * sqrt(d_ref / d), the usual width scaling, anchored at the reference width.
--lr_rule fixed disables it, which is worth having precisely to see the bend appear.
"""
import math
import zlib

import torch
import torch.nn.functional as F

from helpers import CosineLR, MasterAdamW, progress
from model import ZetaGPT


def learning_rate(n_embd, lr_ref, d_ref, rule="sqrt_width"):
    """The peak rate for a model of width `n_embd`."""
    if rule == "fixed":
        return lr_ref
    return lr_ref * math.sqrt(d_ref / max(n_embd, 1))


def _batch(ids, n_tokens, batch, context, gen, device):
    """`batch` random windows of `context`+1 ids drawn from the first `n_tokens` of `ids`.

    +1 because the last position needs a target. Drawn with replacement, which at these
    budgets is what one pass over the data looks like anyway and keeps the sampler stateless."""
    hi = max(1, n_tokens - context - 1)
    off = torch.randint(0, hi, (batch,), generator=gen).tolist()
    rows = [torch.from_numpy(ids[o:o + context + 1].astype("int64")) for o in off]
    return torch.stack(rows).to(device, non_blocking=True)


def _loss(model, seq):
    """Plain next-token cross-entropy in nats per token, over every position of the window."""
    out = model(input_ids=seq[:, :-1])
    return F.cross_entropy(out.logits.reshape(-1, out.logits.size(-1)).float(),
                           seq[:, 1:].reshape(-1))


@torch.no_grad()
def validate(model, ids, batch, context, n_windows, device):
    """Held-out loss on a FIXED set of windows, evenly spaced through the validation split.

    Fixed and evenly spaced, not sampled: every grid point must be scored on identical text, or
    the differences the fit is reading include the differences between one point's validation
    draw and another's."""
    model.eval()
    n = len(ids)
    span = max(1, (n - context - 1) // max(n_windows, 1))
    offs = [i * span for i in range(n_windows) if i * span + context + 1 <= n]
    tot, cnt = 0.0, 0
    for i in range(0, len(offs), batch):
        rows = [torch.from_numpy(ids[o:o + context + 1].astype("int64")) for o in offs[i:i + batch]]
        seq = torch.stack(rows).to(device)
        tot += float(_loss(model, seq)) * seq.shape[0]
        cnt += seq.shape[0]
    model.train()
    return tot / max(cnt, 1)


def run_point(point, data, tok, args, device, log):
    """Train one (N, D) point from scratch. Returns the record the fit and the figure read."""
    train_ids, valid_ids = data
    ctx, batch = args.context, args.batch
    lr = learning_rate(point["n_embd"], args.lr, args.lr_ref_width, args.lr_rule)

    model = ZetaGPT(vocab_size=len(tok), n_layer=point["n_layer"], n_head=point["n_head"],
                    n_embd=point["n_embd"], block_size=ctx, pe=args.pe,
                    dropout=0.0).to(device)
    model.train()
    trainable = sum(p.numel() for p in model.parameters())
    opt = MasterAdamW(model.parameters(), lr=lr)
    steps = point["steps"]
    sched = CosineLR(opt, lr, steps, args.lr_min_factor, args.lr_schedule)
    # crc32, NOT hash(): Python randomises str hashing per process unless PYTHONHASHSEED is
    # set, so hash() would give a different data order every time the same point is rerun --
    # and a study whose points are not individually reproducible is not a study.
    salt = zlib.crc32(point["id"].encode("utf-8"))
    gen = torch.Generator().manual_seed((args.seed * 1000003 + salt) % (2 ** 63 - 1))

    log(f"[scaling] {point['id']:>18}  N={point['N'] / 1e6:6.2f}M  D={point['D'] / 1e6:6.1f}M  "
        f"steps={steps:,}  lr={lr:.3g}  params={trainable / 1e6:.2f}M (incl. embedding)")

    hist = []
    every = args.eval_every or max(1, steps // 10)
    for step in progress(range(steps), desc=f"[{point['id']}]", total=steps):
        sched.step(step)
        seq = _batch(train_ids, point["D"], batch, ctx, gen, device)
        loss = _loss(model, seq)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % every == 0 or step + 1 == steps:
            v = validate(model, valid_ids, batch, ctx, args.val_windows, device)
            hist.append({"step": step + 1, "train": float(loss), "val": v})

    final = hist[-1]["val"] if hist else float("nan")
    best = min((h["val"] for h in hist), default=float("nan"))
    log(f"[scaling] {point['id']:>18}  val loss {final:.4f} nats/token "
        f"(best {best:.4f}, ppl {math.exp(min(final, 60)):.1f})")
    # the model is not kept: twenty checkpoints of a study that only ever reads one number
    # each is a lot of disk for nothing, and the history below is the reproducible artefact
    del model, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {**point, "lr": lr, "val_loss": final, "best_val_loss": best,
            "train_loss": hist[-1]["train"] if hist else float("nan"),
            "params_total": trainable, "history": hist}
