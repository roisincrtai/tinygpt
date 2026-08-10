"""
lm.py -- the length-normalized language-model training loop.

ONE objective, used by the two stages that maximise likelihood:

    L(theta) = - E_{y ~ D} [ (1/|y|) sum_t log pi_theta(y_t | y_<t) ]

LENGTH-NORMALIZED per example (the sum over response positions divided by their count), so a
long document does not dominate a short one inside a batch and `ppl = exp(loss)` reads as a
per-token perplexity. That normalization belongs to THIS objective only: the preference
stages score with the unnormalized sum, because the implicit reward
beta*(log pi - log pi_ref) is a difference of sequence log-likelihoods and dividing it by the
length would change the objective rather than the reporting.

    train(model, enc, docs, ckdir, args, log, monitor, stage=..., steps=..., lr=...,
          ssm_stats_every=N)

owns its resume, budget extension, checkpointing, live plotting and generation previews.

`ssm_stats_every` turns on the state space module's diagnostics for ONE step every N, so the
dynamics figure gets a memory-horizon curve at the cost of a handful of reductions per figure
rather than per step (see model/ssm.py for what is measured and why). It
knows nothing about WHICH corpus it is given -- the pretrain and sft packages each hand it
their own -- so the two stages cannot silently drift apart in their treatment of masking,
normalisation or padding.
"""
import torch
import torch.nn.functional as F

from model.ssm import collect_stats, layer_stats

from .utils import (progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     CosineLR, ckpt_path)

NAME = "LM"
STAGE = "sft"          # default stage name; the pretrain package passes its own


def train(model, enc, docs, ckdir, args, log, monitor, stage=STAGE, steps=None, lr=None,
          preview=None, ssm_stats_every=0):
    """Train `model` by maximum likelihood on the response tokens; return it in eval mode.

    Parameterised by `stage`/`steps`/`lr` (defaults reproduce the SFT stage exactly) so the SAME
    length-normalized LM loop also serves the controlled experiment's LM PRETRAIN stage
    (stage="pretrain", steps=args.pretrain_steps, lr=args.pretrain_lr) over the transcript corpus,
    with its own checkpoint directory and dynamics figure.

    Resume rule, shared with the preference stages: A CHECKPOINT IS FINISHED ONLY FOR THE
    `total` IT RAN. `done` alone ignores the budget, so raising the step budget would load the old
    weights and skip the stage instead of extending it. `!=` rather than `<` so LOWERING the
    budget also re-enters (the step loop is then empty and the stage costs nothing but a re-save)."""
    steps = args.sft_steps if steps is None else steps
    lr = args.sft_lr if lr is None else lr
    ck = load_ckpt(ckdir, stage) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != steps)
    if ck and ck.get("done") and not extend:
        log(f"{stage} done -> loading")
        model.load_state_dict(ck["model"])
        model.eval()
        return model

    opt = MasterAdamW(model.parameters(), lr=lr)
    sched = CosineLR(opt, lr, steps, args.lr_min_factor, args.lr_schedule)
    log(f"{stage}: lr={sched.describe()} steps={steps} batch={args.batch} "
        f"examples={len(docs):,} -> {ckpt_path(ckdir, stage)}")
    g = torch.Generator().manual_seed(args.seed)
    start = 0
    if ck is not None and (not ck["done"] or extend):
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["step"]; g.set_state(ck["gens"][0])
        log(f"resume {stage} @ {start}" + (f" (budget {ck.get('total')} -> {steps})"
                                           if extend else ""))
    # truncated at the resume point, so the appended records extend a strictly increasing
    # sequence and the figure stays continuous across a restart
    hist = load_hist(ckdir, stage, upto=start) if args.resume else []
    Ntr = len(docs)
    # initial/total in ABSOLUTE steps: a resumed stage must read 30499/40000, not 499/10000.
    model.train()
    bar = progress(range(start, steps), desc=f"[{stage}]",
                   initial=start, total=steps)
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        # diagnostics for ONE step at the figure cadence: switched on before the forward that
        # is happening anyway, so the measured pass is a real training pass, not an extra one
        want_ssm = bool(ssm_stats_every) and (step % ssm_stats_every == 0)
        if want_ssm:
            collect_stats(model, True)
        idx = torch.randint(0, Ntr, (args.batch,), generator=g).tolist()
        ids, attn, rmask = enc.encode([docs[i] for i in idx], "chosen")
        logits = model(input_ids=ids, attention_mask=attn).logits
        logp = F.log_softmax(logits, -1)
        lp = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        rm = rmask[:, 1:]
        denom = rm.sum(-1).clamp(min=1).float()
        loss = (-(lp * rm).sum(-1) / denom).mean()
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        # Next-token accuracy over the RESPONSE positions only (the tokens the loss is on): the
        # fraction of response positions whose argmax prediction is the true next token. Reported
        # alongside loss/ppl so the LM stages (pretrain, sft) have an accuracy curve, not only a
        # loss curve. grad-norm (pre-clip) is tracked too, as an optimisation-health trace.
        with torch.no_grad():
            correct = ((logits[:, :-1].argmax(-1) == ids[:, 1:]) & rm.bool()).sum().float()
            acc = float((correct / rm.sum().clamp(min=1)).item())
        rec = {"step": step, "loss": loss.item(), "ppl": float(loss.exp().item()),
               "acc": acc, "gnorm": float(gnorm), "lr": cur_lr}
        if want_ssm:
            # stored per layer, unaggregated: the history keeps what was measured
            rec["ssm"] = layer_stats(model)
            collect_stats(model, False)
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss.item():.3f}", ppl=f"{loss.exp().item():.1f}",
                            acc=f"{acc:.3f}", gnorm=f"{float(gnorm):.2f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, stage, model, opt, step + 1, steps, [g])
            save_hist(ckdir, stage, hist)
        # --plot_every_steps is a TRAINER flag, so it governs every stage that trains, this one
        # included: same flag, same cadence, same figure writer as the preference stages.
        # The generation preview shares the cadence: 20 readable examples per redraw.
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(stage, hist, step + 1)
            if preview:
                preview(model, stage, step + 1)
    model.eval()
    save_ckpt(ckdir, stage, model, opt, steps, steps, [g])
    save_hist(ckdir, stage, hist)
    last = hist[-1] if hist else {}
    log(f"{stage} done: loss={last.get('loss', float('nan')):.4f} "
        f"ppl={last.get('ppl', float('nan')):.2f} acc={last.get('acc', float('nan')):.4f}")
    monitor(stage, hist)                 # persist the figure the moment the stage ends
    return model
