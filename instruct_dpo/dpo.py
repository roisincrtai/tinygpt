"""
dpo.py -- stage 10's logic: Direct Preference Optimization, AND the shared sequence-scoring
core that the other preference code is built on.

TWO THINGS LIVE HERE, deliberately.

1. THE SCORING CORE -- _LogSumExp32, extend_alphabet, seq_logp, corrupt, rewards, loss,
   margin. seq_logp is the single definition of "the log-likelihood of this response under
   this model given this context", so helpers.py (evaluation) and every other consumer
   read the same number from the same code. Nothing downstream re-implements it.

2. THE STAGE RUNNER -- run(), the preference training loop, which starts from the SFT
   model and trains it against the frozen SFT reference. stage10_instruct_dpo.sh is only
   the command-line wrapper around it.

The objective is
    L = -E log sigma( r_chosen - r_rejected ),   r = beta * (logp_policy - logp_ref)
with the policy fed the CLEAN response (corrupt() returns None). The `mask_positions` /
`mask_gate` machinery in seq_logp is retained because the model classes expose it and the
scorer path passes explicit contexts through `x_ids`, not because this stage corrupts
anything: DPO trains on clean histories, which is precisely why its own training-time
exposure gap is identically zero and why the eb_probe below is read from a shared, method-
agnostic rollout probe instead.
"""
import hashlib

import torch
import torch.nn.functional as F

NAME = "DPO"
STAGE = "dpo"


class _LogSumExp32(torch.autograd.Function):
    """Row-wise logsumexp computed in fp32 without keeping an fp32 copy of the logits in
    the autograd graph. Forward upcasts transiently (freed immediately); backward saves
    only the original (possibly bf16) logits and the (B,T) fp32 result, and rebuilds the
    softmax on the fly. This keeps bf16 training's scoring math at fp32 accuracy at
    ~zero retained-memory cost."""
    @staticmethod
    def forward(ctx, logits):
        with torch.no_grad():
            lse = torch.logsumexp(logits.float(), dim=-1)
        ctx.save_for_backward(logits, lse)
        return lse

    @staticmethod
    def backward(ctx, g):
        logits, lse = ctx.saved_tensors
        p = torch.exp(logits.float() - lse.unsqueeze(-1))       # softmax, transient fp32
        return (g.unsqueeze(-1) * p).to(logits.dtype)


def extend_alphabet(lp, r):
    """Renormalize onto V u {MASK} given the vocabulary log-probs and the MASK log-odds.

    `r_i = <h_i, e_mask> - logsumexp_v(logits_i)`. Adding one row multiplies the partition
    function by (1 + e^{r_i}), so with delta_i = softplus(r_i)

        log p~(v)    = log p(v) - delta_i   for every v in V,
        log p~(MASK) = r_i - delta_i .

    ONE implementation, used by seq_logp for the response likelihood and re-exported by
    dpo.extend_alphabet for the prior, so the two can never disagree about which law they
    are evaluating."""
    delta = F.softplus(r)
    return lp - delta, r - delta


def seq_logp(model, ids, attn, rmask, mask_positions=None, x_ids=None, mask_gate=None,
             mask_vec=None, sub_ids=None, xtra_ids=None, return_pos=False,
             mask_logits=False):
    """Sum of log p(clean response tokens). The model is fed `x_ids` if given (e.g. a
    rolled-out history for RO-DPO), else the clean `ids`; `mask_positions` (B,T bool) marks
    input tokens replaced with the learnable MASK embedding (hard), while `mask_gate` (B,T
    float in [0,1]) is a differentiable mask and `mask_vec` overrides which
    mask embedding is substituted (a caller may own its own). Targets are always CLEAN `ids`.

    return_pos=True additionally returns the PER-POSITION log-probabilities (B,T-1), and, if
    `xtra_ids` is given, those of a second token sequence scored against the SAME logits. Both
    share one logsumexp, so the extra target costs a single gather. A rollout prior term needs exactly this: log pi_theta(v | x, z_<i) at v = y_i and at v = sub_i, evaluated under
    the CORRUPTED context z_<i that this forward already ran.

    `mask_logits=True` is the learned-mask counterpart: it returns, in place of the `xtra_ids`
    channel, the MASK symbol's logit relative to the vocabulary's log-partition,

        r_i = <h_i, mask_embed> - logsumexp_v(logits_i),

    and RENORMALIZES this forward onto V u {MASK}: `total` and the per-position channel become
    log p~(y_t) and the extra channel becomes log p~(MASK_t). The whole forward then speaks one
    law, so the response likelihood that forms the margin and the prior's conditionals cannot
    disagree about which model they belong to. `xtra_ids` and `mask_logits` are mutually
    exclusive: the two-point prior has exactly one second point."""
    inp = ids if x_ids is None else x_ids
    out = model(input_ids=inp, attention_mask=attn, mask_positions=mask_positions,
                mask_gate=mask_gate, mask_vec=mask_vec, sub_ids=sub_ids,
                mask_logits=mask_logits)

    logits = out.logits[:, :-1]
    tgt = ids[:, 1:]
    # log p(tgt) = logit(tgt) - logsumexp(logits): equivalent to log_softmax + gather, but
    # avoids materializing the full-vocab log-softmax in the autograd graph (~2.5 GB per
    # forward at Qwen's 151k vocab, batch 8 x 512 tokens). The logsumexp runs in fp32 via
    # _LogSumExp32 so bf16 logits do not degrade the margins; the result is fp32.
    lse = _LogSumExp32.apply(logits)
    lp = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float() - lse
    lpx = None
    if mask_logits:
        # The MASK symbol is part of the model, so EVERY likelihood of that model is taken on
        # V u {MASK} -- the response likelihood that forms the margin included, not only the
        # prior's conditionals. Renormalizing one and not the other would evaluate the two
        # terms of eq:final-objective under two different laws.
        lp, lpx = extend_alphabet(lp, out.mask_logit[:, :-1].float() - lse)
    total = (lp * rmask[:, 1:].float()).sum(-1)
    if not return_pos:
        return total
    if xtra_ids is not None:
        lpx = logits.gather(-1, xtra_ids[:, 1:].unsqueeze(-1)).squeeze(-1).float() - lse
    return total, lp, lpx


def corrupt(ids, rmask, gen):
    """Standard DPO: the policy sees the clean response (no MASK)."""
    return None


def rewards(policy, ref, W, L, beta, gen=None, corrupt_fn=corrupt, mpW=None, mpL=None):
    """Per-example implicit rewards (chosen, rejected) = beta*(logp_policy - logp_ref).
    The policy input is corrupted per `corrupt_fn`; the reference is always scored clean.
    Precomputed masks `mpW`/`mpL` (B,T bool) override `corrupt_fn` -- Denoising DPO uses this
    to share a single per-pair mask rate t across the chosen and rejected responses."""
    idsW, attnW, rmW = W
    idsL, attnL, rmL = L
    if mpW is None:
        mpW = corrupt_fn(idsW, rmW, gen)
    if mpL is None:
        mpL = corrupt_fn(idsL, rmL, gen)
    lpw = seq_logp(policy, idsW, attnW, rmW, mask_positions=mpW)
    lpl = seq_logp(policy, idsL, attnL, rmL, mask_positions=mpL)
    with torch.no_grad():
        rw = seq_logp(ref, idsW, attnW, rmW)
        rl = seq_logp(ref, idsL, attnL, rmL)
    return beta * (lpw - rw), beta * (lpl - rl)


def loss(policy, ref, W, L, beta, gen=None, corrupt_fn=corrupt, mpW=None, mpL=None):
    """DPO loss. Returns (loss, r_chosen, r_rejected, margin) for live diagnostics."""
    cr, rr = rewards(policy, ref, W, L, beta, gen=gen, corrupt_fn=corrupt_fn, mpW=mpW, mpL=mpL)
    M = cr - rr
    return -F.logsigmoid(M).mean(), cr, rr, M


def margin(policy, ref, W, L, beta):
    """Teacher-forced (clean) preference margin r_chosen - r_rejected. Used for evaluation."""
    cr, rr = rewards(policy, ref, W, L, beta)
    return cr - rr


# =============================================================================================
# stage runner -- the training loop that consumes the scoring core above
# =============================================================================================
def _micro_split(chunk, n):
    """Split a batch into gradient-accumulation micro-batches.

    Peak activation memory is then set by the micro-batch rather than by the batch, while the
    update is mathematically identical: each micro-batch's loss is weighted by its share of the
    batch before backward, so the accumulated gradient is the full-batch gradient."""
    if n <= 0 or n >= len(chunk):
        return [chunk]
    return [chunk[i:i + n] for i in range(0, len(chunk), n)]


def run(model, ref, enc, train_pairs, ev_pairs, tok, ckdir, args, log, monitor, preview=None):
    """Train `model` (a copy of the SFT model) by DPO against the frozen `ref`; return it.

    On a finished checkpoint the weights are loaded and returned WITHOUT retraining.

    Resume rule: a checkpoint is finished ONLY FOR THE BUDGET IT RAN. `done` alone ignored
    `total`, so raising --dpo_steps from 30000 to 40000 skipped the stage entirely: no extra
    steps, no fresh evaluation, no redraw. `!=` and not `<`, so LOWERING the budget also
    re-enters and the stage is re-evaluated and redrawn against the budget now being asked
    for; the step loop is then empty and costs one evaluation."""
    # DEFERRED IMPORT, NOT A STYLE CHOICE: helpers.py imports margin/rewards from dpo.py, so a
    # module-level `import helpers` would be a circular import at interpreter start. The cycle is
    # real but harmless once both modules exist, which is exactly what a function-local import
    # expresses.
    from helpers import (holdout_probe, progress, evaluate, rollout_curve, reward_hist, save_ckpt,
                         load_ckpt, save_hist, load_hist, MasterAdamW, restore_rng,
                         exposure_probe, CosineLR, ckpt_path)

    total = args.dpo_steps
    lr = args.dpo_lr
    ck = load_ckpt(ckdir, STAGE) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != total)
    if ck and ck.get("done") and not extend:
        model.load_state_dict(ck["model"]); model.eval()
        log(f"dpo done -> {ck.get('eval')}")
        monitor(STAGE, load_hist(ckdir, STAGE))   # resume: refresh from the loaded history
        return model
    if extend:
        log(f"dpo: budget {ck.get('total')} -> {total}, extending from step {ck['step']}")

    opt = MasterAdamW(model.parameters(), lr=lr)
    sched = CosineLR(opt, lr, total, args.lr_min_factor, args.lr_schedule)
    log(f"dpo: lr={sched.describe()} steps={total} beta={args.beta} batch={args.batch} "
        f"-> {ckpt_path(ckdir, STAGE)}")
    # Data order and corruption both key off --seed; the stage NAME salts the streams so
    # different stages draw different minibatch orders from the same seed.
    _ms = int(hashlib.sha1(STAGE.encode()).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(args.seed * 1000003 + 1)
    cg = torch.Generator().manual_seed(args.seed * 1000003 + 2 + _ms)
    # The exposure probe MUST NOT share cg: a diagnostic must not move the optimisation path.
    pg = torch.Generator().manual_seed(args.seed * 1000003 + 9176 + _ms)
    start = 0
    hist = []           # loaded below, truncated at the resume point
    if ck is not None and (not ck["done"] or extend):
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for grp in opt.param_groups:
            grp["lr"] = lr                       # honor a changed --dpo_lr on resume
        start = ck["step"]; g.set_state(ck["gens"][0]); cg.set_state(ck["gens"][1])
        restore_rng(ck)              # dropout draws from the GLOBAL stream; restore it too
        log(f"resume dpo @ {start}")
        hist = load_hist(ckdir, STAGE, upto=start)
        monitor(STAGE, hist, start)
    Ntr = len(train_pairs)
    params = list(model.parameters())
    model.train()
    bar = progress(range(start, total), desc=f"[dpo @{args.dataset}]",
                   initial=start, total=total)               # absolute steps

    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        idx = torch.randint(0, Ntr, (args.batch,), generator=g).tolist()
        chunk = [train_pairs[i] for i in idx]
        lval = 0.0; crs, rrs, Ms = [], [], []
        opt.zero_grad()
        for mc in _micro_split(chunk, args.micro_batch):   # each micro-batch frees its graph
            w = len(mc) / len(chunk)
            Wi = enc.encode(mc, "chosen"); Li = enc.encode(mc, "rejected")
            L_i, cr_i, rr_i, M_i = loss(model, ref, Wi, Li, args.beta, gen=cg)
            (L_i * w).backward()
            lval += L_i.item() * w
            crs.append(cr_i.detach()); rrs.append(rr_i.detach()); Ms.append(M_i.detach())
        cr = torch.cat(crs); rr = torch.cat(rrs); M = torch.cat(Ms)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        with torch.no_grad():
            rec = {"step": step, "loss": lval, "acc": (M > 0).float().mean().item(),
                   "margin": M.mean().item(), "win_acc": (cr > 0).float().mean().item(),
                   "lose_acc": (rr < 0).float().mean().item(),
                   "r_w": cr.mean().item(), "r_l": rr.mean().item(),
                   "gnorm": float(gnorm),      # optimisation health, plotted like every stage
                   "lr": cur_lr}
            # EXPOSURE BIAS OF THE CURRENT POLICY: DPO trains on clean histories, so its
            # training-time gap is identically 0; only a rollout probe makes the trajectory
            # visible. Every --eb_every steps, on --eb_pairs pairs.
            if args.eb_every > 0 and (step % args.eb_every == 0 or step == total - 1):
                try:
                    model.eval()      # dropout noise would corrupt the probe
                    Wp = enc.encode(chunk[:args.eb_pairs], "chosen")
                    rec["eb_probe"] = exposure_probe(model, Wp[0], Wp[1], Wp[2], gen=pg,
                                                     roll_temp=args.rollout_temp)
                except Exception as e:      # a diagnostic must never kill a long run
                    log(f"[eb] probe failed at step {step}: {e}")
                finally:
                    model.train()
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{rec['loss']:.3f}", acc=f"{rec['acc']:.2f}",
                            margin=f"{rec['margin']:+.2f}", r_w=f"{rec['r_w']:+.2f}",
                            r_l=f"{rec['r_l']:+.2f}", win_acc=f"{rec['win_acc']:.2f}",
                            lose_acc=f"{rec['lose_acc']:.2f}")
        ev_n = args.eval_every or args.plot_every_steps
        if ev_n > 0 and (step + 1) % ev_n == 0:
            model.eval()
            pr = holdout_probe(model, ref, enc, ev_pairs[:args.eval_pairs], args.beta, gen=pg)
            model.train()
            rec.update(pr)
            log(f"[probe] dpo   step {step + 1}: acc={pr['eval_acc']:.3f} "
                f"margin={pr['eval_margin']:+.3f} D_EB={pr['eval_deb']:+.5f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, STAGE, model, opt, step + 1, total, [g, cg])
            save_hist(ckdir, STAGE, hist)
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(STAGE, hist, step + 1)               # live dynamics (PDF)
            if preview:
                preview(model, STAGE, step + 1)          # 20 readable generation examples

    model.eval()
    evald = evaluate(model, ref, enc, ev_pairs, args.beta)
    if args.n_hist > 0:      # per-pair distributions for histograms (win/lose/margin)
        evald["hist"] = reward_hist(model, ref, enc, ev_pairs[:args.n_hist], args.beta)
    if args.n_roll > 0:      # exposure-bias curve (clean rollout, no corruption at eval)
        log("dpo: exposure-bias rollout ...")
        p_grid = [float(x) for x in args.p_grid.split(",")]
        evald["rollout_curve"] = rollout_curve(model, ref, tok, ev_pairs[:args.n_roll],
                                               args.beta, p_grid, args.roll_tokens, args.max_len)
    save_ckpt(ckdir, STAGE, model, opt, total, total, [g, cg], evald=evald)
    save_hist(ckdir, STAGE, hist)
    log(f"dpo done: acc={evald['acc']:.3f} mu_M={evald['mu_M']:+.3f} P_rev={evald['P_rev']:.3f}")
    monitor(STAGE, hist)     # persist the figure now: a later crash must not discard it
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model
