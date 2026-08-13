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
                     CosineLR, ckpt_path, tag, restore_rng, amp)

NAME = "LM"
STAGE = "sft"          # default stage name; the pretrain package passes its own


def _chunk_logprob(head, h, tgt):
    """Log-probability of `tgt` under `head(h)`, for ONE slice of the sequence.

    Called through torch.utils.checkpoint, which is the whole point: the (b, chunk, vocab)
    logits and their log-softmax exist during this call, are thrown away when it returns, and
    are recomputed one slice at a time during the backward pass. What survives in memory is the
    (b, chunk) result. Returns the log-probabilities and, detached, how many positions the model
    would have got right -- computed here because the logits needed to answer that question are
    here and nowhere else."""
    logits = head(h)
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp, (logits.argmax(-1) == tgt).detach()


def lm_loss(model, ids, attn, rmask, chunk=0):
    """Mean per-sequence negative log-likelihood over the response positions, and accuracy.

    THE OBJECTIVE IS LENGTH-NORMALISED: each sequence contributes the mean log-probability of
    its own response tokens, and the batch averages those. A sum would let one long sequence
    outweigh several short ones.

    `chunk` > 0 evaluates the vocabulary projection in slices of that many positions instead of
    all at once. The result is identical -- the same positions, the same targets, the same
    normalisation -- but the peak memory of the loss becomes 3 x chunk x vocab rather than
    3 x T x vocab, which at T = 32,768 is the difference between 0.6 GB and 18.4 GB."""
    # THE TARGETS FOLLOW THE LOGITS. When the model's layers are split across GPUs the output
    # comes back on the LAST device while the batch was built on the first, and gather() cannot
    # index a tensor on one device with a tensor on another. Unsharded the two already agree
    # and both moves are no-ops. See model/parallel.py.
    home = model.head.weight.device
    tgt, rm = ids[:, 1:].to(home), rmask[:, 1:].to(home)
    denom = rm.sum(-1).clamp(min=1).float()
    if not chunk:
        logits = model(input_ids=ids, attention_mask=attn).logits
        lp = F.log_softmax(logits, -1)[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = (-(lp * rm).sum(-1) / denom).mean()
        with torch.no_grad():
            correct = ((logits[:, :-1].argmax(-1) == tgt) & rm.bool()).sum().float()
        return loss, correct, rm.sum().clamp(min=1).float()

    # CHUNKED OVER TOKENS, NOT OVER POSITIONS. The projection produces one vocabulary row per
    # TOKEN, and a batch of 24 sequences at 1,024 positions is 24,576 of them -- slicing the
    # position axis alone still hands the projection every sequence at once and saves nothing
    # the batch does not immediately give back. Flattening first makes `chunk` mean what it
    # says: this many rows of (vocab) exist at a time, whatever shape the batch has.
    from torch.utils.checkpoint import checkpoint
    h = model.hidden_states(input_ids=ids, attention_mask=attn)[:, :-1]
    B, L, D = h.shape
    hf, tf, rf = h.reshape(B * L, D), tgt.reshape(B * L), rm.reshape(B * L)
    parts = []
    correct = torch.zeros((), device=h.device, dtype=torch.float32)
    for a in range(0, B * L, chunk):
        b = min(a + chunk, B * L)
        lp, hit = checkpoint(_chunk_logprob, model.head, hf[a:b], tf[a:b],
                             use_reentrant=False)
        parts.append(lp)
        correct = correct + (hit & rf[a:b].bool()).sum().float()
    # (B*L,) floats: four bytes a token, against four bytes a token TIMES THE VOCABULARY for
    # the logits that produced them. Keeping these is what lets the per-sequence normalisation
    # below be exact rather than an average of averages.
    lp = torch.cat(parts).view(B, L)
    return (-(lp * rm).sum(-1) / denom).mean(), correct, rm.sum().clamp(min=1).float()


def context_plan_for(args, steps, docs):
    """The (start, stop, context, batch, micro_batch) segments for a budget of `steps`.

    A PURE FUNCTION OF THE BUDGET, and silent, so it can be asked what the schedule WOULD have
    been for some other budget -- which is how a resumed run recovers the window it was last
    training at when the budget has since changed. `context_schedule` is this plus the log line.

    ONE WINDOW UNLESS ALL THREE CONDITIONS HOLD: the corpus is a packed stream (a list of
    records cannot be re-cut to an arbitrary length), the scheme declares a schedule, and the
    run has not pinned a context of its own with --context_window. A stage that
    was told which window to use is not second-guessed."""
    import default_config as config
    one = [(0, steps, max(int(args.max_len), 2), int(args.batch),
            int(getattr(args, "micro_batch", 0) or 0))]
    if not (hasattr(docs, "batch") and hasattr(docs, "n_tokens")):
        return one
    # --context_window WINS, and it may itself be a schedule: "1024,4096" pins those two
    # windows regardless of what the scheme says. Only when nothing was asked for does the
    # scheme's own schedule apply.
    asked = getattr(args, "context_window", "")
    scheme = getattr(args, "model_scheme", "") or config.PRETRAIN["model_scheme"]
    if asked:
        wins = config.windows(asked)
    elif scheme in config.SCHEMES:
        wins = config.context_windows(scheme)
    else:
        wins = []
    if len(wins) < 2:
        return one
    # The batch in force belongs to the LONGEST window, which is what it was sized against.
    return config.context_plan(wins, steps, int(args.batch),
                               config.SCHEME_MICRO_TOKENS.get(scheme, 0)
                               if not getattr(args, "micro_batch", 0) else 0)


def context_schedule(args, steps, docs, log, stage):
    """`context_plan_for`, announced."""
    plan = context_plan_for(args, steps, docs)
    if len(plan) > 1:
        log(f"{stage}: context schedule "
            f"{' -> '.join(f'{w:,}' for _, _, w, _, _ in plan)} "
            f"over {steps:,} steps, {steps // len(plan):,} steps each, "
            f"batch {plan[0][3]} -> {plan[-1][3]} so tokens per step stay constant")
    return plan


def context_floor(args, docs, ck, start):
    """THE WINDOW THE RUN HAS ALREADY REACHED, which the schedule may never fall below.

    The schedule divides the budget evenly among its windows, so the boundaries are a function
    of the budget -- and RAISING the budget pushes them later. A run 82,875 steps into a
    168,317-step budget is at 8,192; extend that budget to 250,000 and step 82,875 falls back
    into the 2,048 segment. The model would then be asked to un-learn a context it has spent
    tens of thousands of steps acquiring, its state-space memory horizon and its attention
    both retuned downwards, and the extension -- meant to buy MORE long-context training --
    would cost some of what was already bought. Hence the ratchet: the window only ever goes
    up.

    Two sources, and the larger wins:

      * `ctx_window` in the checkpoint, which save_ckpt now records at every save. This is the
        authoritative one: it is what the run was actually training at.
      * the window the OLD budget's schedule puts at this step -- `ck["total"]` is the budget
        the checkpoint was written under, and the plan is a pure function of it. This
        reconstructs the floor exactly for checkpoints written before `ctx_window` existed,
        so a run already in flight is protected without having to be restarted to gain the
        protection.
    """
    if not ck:
        return 0
    floor = int(ck.get("ctx_window", 0) or 0)
    old_total = int(ck.get("total", 0) or 0)
    if old_total > 0:
        old = context_plan_for(args, old_total, docs)
        for a, b, w, _, _ in old:
            if a <= start < b:
                floor = max(floor, int(w))
                break
    return floor


def ratchet(plan, seg, start, floor, args, log, stage):
    """`plan` with every segment from `seg` onwards raised to at least `floor`.

    The batch and micro-batch are recomputed by exactly the rules context_plan uses -- batch x
    context constant, so a step stays the same unit of work, and micro_tokens // context
    sequences per forward pass -- rather than being scaled out of the numbers already in the
    plan, which would compound the rounding those numbers already carry.

    Segments already at or above the floor are left untouched, so when nothing has changed
    this returns the plan unaltered and a resumed run is the run it was."""
    import default_config as config
    if floor <= 0 or seg >= len(plan):
        return plan
    scheme = getattr(args, "model_scheme", "") or config.PRETRAIN["model_scheme"]
    longest = max(w for _, _, w, _, _ in plan)
    micro_tokens = (config.SCHEME_MICRO_TOKENS.get(scheme, 0)
                    if not getattr(args, "micro_batch", 0) else 0)
    out, lifted = list(plan), []
    for i in range(seg, len(plan)):
        a, b, w, _, _ = plan[i]
        if w >= floor:
            continue
        batch = max(1, round(int(args.batch) * longest / int(floor)))
        micro = min(batch, max(1, int(micro_tokens) // int(floor))) if micro_tokens else 0
        out[i] = (a, b, int(floor), batch, micro)
        # reported from the RESUME POINT, not from the segment's own start: the run begins
        # inside the first lifted segment, and steps before it are finished and unaffected
        lifted.append((w, max(a, int(start)), b))
    if lifted:
        # NEVER SILENTLY. This changes what the run trains on, so it is stated in full: what
        # was in the plan, what it was raised to, and over which steps.
        log(f"{stage}: CONTEXT RATCHET -- this run has already reached {floor:,} tokens, so "
            f"{len(lifted)} segment(s) of the recomputed schedule that would have gone BACK "
            f"below it were raised to {floor:,}:")
        for w, a, b in lifted:
            log(f"{stage}:   steps {a:,}-{b - 1:,}: {w:,} -> {floor:,}")
        log(f"{stage}: the window is monotone; a longer budget buys more training at the "
            f"context reached, never a return to a shorter one.")
    else:
        # Said even when it does nothing, because "the ratchet is armed and the schedule was
        # already sound" and "the ratchet never ran" look identical in a log that stays quiet.
        log(f"{stage}: context ratchet floor {floor:,} tokens (reached already); the "
            f"remaining schedule is at or above it, unchanged")
    return out


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
        # THE GLOBAL STREAM TOO, not only the data generator. `g` covers the data order; dropout
        # and anything else drawing from torch's default generator do not, so an uninterrupted
        # run and a resumed one saw different dropout masks from the resume point onwards. Every
        # other trainer here already restored it -- reward, rlhf, dpo, cot and distill -- and
        # this one, which serves pretrain, the instruction SFT, the CoT SFT and the language
        # adaptation, did not. save_ckpt has always written it; nothing read it back.
        #
        # Harmless at MODEL["dropout"] = 0.0, which is the default. Not harmless the first time
        # someone sets it, and silent then.
        start = ck["step"]; g.set_state(ck["gens"][0]); restore_rng(ck)
        log(f"resume {stage} @ {start}" + (f" (budget {ck.get('total')} -> {steps})"
                                           if extend else ""))
    # truncated at the resume point, so the appended records extend a strictly increasing
    # sequence and the figure stays continuous across a restart
    hist = load_hist(ckdir, stage, upto=start) if args.resume else []

    # TWO CORPUS SHAPES, ONE LOOP. A packed TokenStream is memory-mapped and sampled by
    # offset: no padding, no prompt mask, every position a target. A list of records is
    # encoded per batch as before, which is what the prompt/response distinction of the SFT
    # corpus needs. The difference is confined to this closure, so the objective, the
    # resume rule and the diagnostics below cannot drift between the two.
    if hasattr(docs, "batch") and hasattr(docs, "n_tokens"):
        # args.max_len is the model's context window by the time setup() is done with it, and
        # batch() returns seq_len + 1 tokens so the shift produces seq_len targets. Asking for
        # max_len - 1 therefore feeds the model EXACTLY its context window, never one over it.
        T = max(int(args.max_len) - 1, 1)
        log(f"{stage}: packed stream, {docs.n_tokens:,} tokens, {T + 1} tokens/example "
            f"-> {args.batch * T:,} target tokens per step")
        def sample(batch, gen, seq_len=None):
            return docs.batch(batch, seq_len or T, generator=gen, device=enc.device)
    else:
        Ntr = len(docs)
        def sample(batch, gen, seq_len=None):
            idx = torch.randint(0, Ntr, (batch,), generator=gen).tolist()
            return enc.encode([docs[i] for i in idx], "chosen")

    # THE CONTEXT SCHEDULE. Only a packed stream can serve a window of any length on demand; a
    # list of records has the length its records have, so the fine-tuning stages keep one window
    # and the plan collapses to a single segment. `plan` is a list of
    # (start, stop, context, batch) tiling [0, steps) exactly.
    plan = context_schedule(args, steps, docs, log, stage)
    # THE SEGMENT THE RUN ACTUALLY STARTS IN, not the first one. A resumed run begins in the
    # middle of the schedule, and seg=0 was only ever corrected INSIDE the loop -- after the
    # progress bar had already been labelled with plan[0]'s window. So a run resumed at 150,000
    # trained at 8,192 while its bar said 1,024, and the line that would have said otherwise
    # only fires at a segment boundary, which a mid-segment resume is not. The training was
    # right and everything a person could see about it was wrong, which is the worst way round.
    seg = 0
    while seg + 1 < len(plan) and start >= plan[seg][1]:
        seg += 1
    # THE RATCHET. The segment boundaries move when the budget does, so a resumed run can land
    # in a segment SHORTER than the one it was training in. It never may: the window is
    # monotone across the whole life of a checkpoint, and a larger budget buys more training at
    # the context already reached rather than a return to a smaller one. Applied after `seg` is
    # found, so the boundaries -- and therefore which segment the run is in -- are untouched.
    plan = ratchet(plan, seg, start, context_floor(args, docs, ck, start), args, log, stage)
    # CHUNKED LOSS. 0 evaluates the vocabulary projection over the whole sequence at once, which
    # is what every stage did before long contexts existed and is still right at 512. Above that
    # the three vocabulary-sized tensors dominate the step, so the projection is sliced.
    chunk = int(getattr(args, "loss_chunk", 0) or 0) if getattr(args, "chunked_loss", False) else 0
    if chunk:
        V = int(getattr(model, "cfg", {}).get("vocab_size", 0)) or 0
        # the comparison is against the WHOLE STEP -- batch x context tokens -- because that is
        # what the projection would otherwise be handed at once
        most = max(b * w for _, _, w, b, _ in plan)
        note = (f" ({3 * chunk * V * 4 / 1024**3:.2f} GiB peak instead of "
                f"{3 * most * V * 4 / 1024**3:.2f} GiB)") if V else ""
        log(f"{stage}: chunked loss, {chunk:,} TOKENS per slice{note}")
    # initial/total in ABSOLUTE steps: a resumed stage must read 30499/40000, not 499/10000.
    model.train()
    bar = progress(range(start, steps), desc=tag(stage, plan[seg][2] if len(plan) > 1 else None),
                   initial=start, total=steps)
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        # diagnostics for ONE step at the figure cadence: switched on before the forward that
        # is happening anyway, so the measured pass is a real training pass, not an extra one
        want_ssm = bool(ssm_stats_every) and (step % ssm_stats_every == 0)
        if want_ssm:
            collect_stats(model, True)
        # WHICH WINDOW THIS STEP BELONGS TO. Advanced rather than searched, so a resumed run at
        # step 30,000 lands in the same segment it would have reached by running from zero --
        # the plan is a function of the budget, not of how the budget was consumed.
        while seg + 1 < len(plan) and step >= plan[seg][1]:
            seg += 1
        _, _, cur_ctx, cur_batch, cur_micro = plan[seg]
        # ...at a boundary, AND on the first step of a resumed run, which is not one
        if len(plan) > 1 and (step == plan[seg][0] or step == start):
            if hasattr(bar, "set_description"):
                bar.set_description(tag(stage, cur_ctx))
            log(f"{stage}: context window {cur_ctx:,} at batch {cur_batch}"
                + (f", micro-batch {cur_micro} ({-(-cur_batch // cur_micro)} passes)"
                   if cur_micro else "")
                + f" ({cur_batch * (cur_ctx - 1):,} target tokens per step), "
                  f"steps {plan[seg][0]:,}-{plan[seg][1] - 1:,}")
        # ONE STEP, POSSIBLY SEVERAL FORWARD PASSES. A micro-batch splits the step's sequences
        # into groups that are each carried through forward and backward alone, so only one
        # group's activations are ever resident; the gradients add up in the parameters, which
        # is where a batch is combined anyway. The optimiser sees exactly the step it would have
        # seen whole -- each group's loss is weighted by its share of the sequences, so the sum
        # is the mean over all of them, not the mean of the means.
        opt.zero_grad(set_to_none=True)
        mb = (args.micro_batch if getattr(args, "micro_batch", 0) > 0
              else (cur_micro or cur_batch))
        mb = max(1, min(int(mb), cur_batch))
        sum_loss = sum_correct = sum_tokens = 0.0
        for off in range(0, cur_batch, mb):
            nb = min(mb, cur_batch - off)
            ids, attn, rmask = sample(nb, g, max(cur_ctx - 1, 1))
            # THE FORWARD IS INSIDE, THE BACKWARD AND THE STEP ARE NOT. autocast records the
            # dtype each operation ran in and the backward follows that record, so wrapping it
            # too would add nothing; the optimiser must see fp32 parameters and fp32 gradients,
            # which is exactly what it sees out here.
            with amp(args):
                loss, correct, ntok = lm_loss(model, ids, attn, rmask, chunk)
            (loss * (nb / cur_batch)).backward()
            sum_loss += float(loss.item()) * nb
            sum_correct += float(correct.item()); sum_tokens += float(ntok.item())
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        loss = torch.tensor(sum_loss / cur_batch)
        # Next-token accuracy over the RESPONSE positions only (the tokens the loss is on): the
        # fraction of response positions whose argmax prediction is the true next token. Reported
        # alongside loss/ppl so the LM stages (pretrain, sft) have an accuracy curve, not only a
        # loss curve. grad-norm (pre-clip) is tracked too, as an optimisation-health trace.
        acc = sum_correct / max(sum_tokens, 1.0)
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
            # THE WINDOW GOES IN THE CHECKPOINT. It is the one thing about the schedule that
            # cannot be recovered from the weights, and without it a resume has to guess --
            # see context_floor.
            save_ckpt(ckdir, stage, model, opt, step + 1, steps, [g], ctx_window=cur_ctx)
            save_hist(ckdir, stage, hist)
        # --plot_every_steps is a TRAINER flag, so it governs every stage that trains, this one
        # included: same flag, same cadence, same figure writer as the preference stages.
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(stage, hist, step + 1)
        # THE GENERATION EXAMPLES HAVE THEIR OWN CADENCE, --print_samples_every_steps. They used
        # to share the figure's, which forced one number to serve two things that are consumed
        # differently: a figure is glanced at, twenty generations are read line by line.
        every = getattr(args, "print_samples_every_steps", args.plot_every_steps)
        if preview and every > 0 and (step + 1) % every == 0:
            preview(model, stage, step + 1)
    model.eval()
    # the LAST segment's window: what the finished model was trained at, and the floor any
    # later extension of this run must respect
    save_ckpt(ckdir, stage, model, opt, steps, steps, [g], ctx_window=plan[-1][2])
    save_hist(ckdir, stage, hist)
    last = hist[-1] if hist else {}
    log(f"{stage} done: loss={last.get('loss', float('nan')):.4f} "
        f"ppl={last.get('ppl', float('nan')):.2f} acc={last.get('acc', float('nan')):.4f}")
    monitor(stage, hist)                 # persist the figure the moment the stage ends
    return model
