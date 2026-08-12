"""
grpo.py -- Group Relative Policy Optimisation for the chain-of-thought stage.

GRPO is PPO with the critic deleted. PPO needs a value head to say whether a rollout was
better than expected; GRPO answers that question by sampling a GROUP of completions for the
SAME prompt and standardising their rewards within the group:

    for prompt x, sample y_1..y_G ~ pi_old(.|x),  r_i = verifier(y_i)
    A_i = (r_i - mean_j r_j) / (std_j r_j + eps)                # one scalar per completion
    L   = -E_i,t [ min( rho_it A_i, clip(rho_it, 1-eps, 1+eps) A_i ) ]
          + kl_coef * KL( pi || pi_ref )                        # added to the LOSS, not the
                                                                # per-token reward as in PPO
    rho_it = pi_new(y_it) / pi_old(y_it)   on the response tokens only

The advantage is CONSTANT over the tokens of a completion -- there is no per-token credit
assignment, only "this whole attempt was better than its siblings". The group is what makes
that comparison meaningful, so group_size must be > 1: with G = 1 every advantage is exactly
zero and nothing is learned.

The KL term uses the k3 estimator  exp(d) - d - 1  with  d = log pi_ref - log pi, which is
non-negative and unbiased, unlike the raw log-ratio that can go negative on a sample.

Nothing here knows what the reward means; cot/verifier.py owns that. Nothing here knows which
checkpoint the policy came from; cot/run.py owns that.

Checkpoint: checkpoints/cot/checkpoint_<model>_<pe>_cot-grpo.pt
Figure:     outputs/plots/cot/dynamics_<model>_<pe>_cot-grpo.pdf
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import default_config as config
from chat import decode
from helpers import (tag, progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     restore_rng, CosineLR, ckpt_path)

from . import verifier

NAME = "CoT (GRPO)"
STAGE = "cot"

# The floor under a derived response length. A window so full of prompt that nothing is left
# to answer with would make every completion empty and every reward zero, which looks like a
# model that cannot reason rather than a budget that left it no room.
MIN_NEW_TOKENS = 64

# POSITIONS PER SLICE of the vocabulary projection in forward_logprobs. 8,192 rows of 50,259
# floats is 1.5 GiB live, against the 11.2 GiB the unsliced version needed for one call at
# 24 x 2,495 -- and unlike that figure this one does not grow as completions lengthen.
LP_CHUNK = 8192


# --------------------------------------------------------------------------- #
# rollout
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(policy, tok, prompts, device, max_new, temperature, g, max_len,
            use_cache=True, cache_gib=0.0):
    """Sample one completion per row of `prompts` (already group-expanded by the caller, so
    the same prompt appears group_size times and each copy is sampled independently).

    THE ROWS ARE DECODED IN BUCKETS OF EQUAL PROMPT LENGTH, and that is a throughput decision,
    not a cosmetic one. The cache appends one column for ALL rows at once, so it is only
    correct when every row's cursor sits at the same position -- and with right padding that
    means equal prompts. A batch mixing lengths therefore fell back to recomputing the whole
    prefix at every generated token: O(n^2) work instead of O(n).

    That fallback was the normal case, not the exception. A GRPO step draws `batch` problems
    and the countdown prompts are 196-203 tokens depending on how many digits their numbers
    have, so all of them agreeing happens about 6% of the time. The other 94% of steps decoded
    ~3,900 tokens quadratically, which is what made a single rollout take the better part of an
    hour.

    Bucketing costs nothing, because the grouping already exists: the `group_size` completions
    of one problem are copies of ONE prompt and so are always the same length. Splitting a
    24-row batch into its two or three distinct lengths turns one uncached decode into two or
    three cached ones. The completions are unchanged -- rows are independent, and each bucket
    is sampled from the same generator in the same order.

    Returns (ids, attn, resp_mask, texts): the batch, the mask of GENERATED tokens, and the
    decoded completions the verifier scores, in the ORDER THE PROMPTS WERE GIVEN."""
    pad = tok.pad_token_id
    # NO HARD RESPONSE LENGTH BY DEFAULT. `max_new = 0` means "whatever the context window has
    # left once the prompt is in it", computed from the LONGEST PROMPT ACTUALLY IN THIS BATCH.
    # A fixed cap cannot be right across the schemes: their windows run from 1,024 to 32,768,
    # so 200 tokens is generous for one and a straitjacket for another -- and this stage exists
    # to see whether a policy learns to think for longer, which a cap decides in advance.
    pids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    if not max_new or max_new <= 0:
        max_new = max(MIN_NEW_TOKENS, max_len - max(len(p) for p in pids) - 1)
    keep = max(max_len - max_new, 8)
    pids = [pid[-keep:] if len(pid) > keep else pid for pid in pids]

    buckets = {}
    for i, p in enumerate(pids):
        buckets.setdefault(len(p), []).append(i)

    if len(buckets) == 1:
        return _decode(policy, tok, pids, device, max_new, temperature, g,
                       use_cache, cache_gib, "")

    parts = {}
    for k, (length, idx) in enumerate(sorted(buckets.items()), 1):
        note = f" {k}/{len(buckets)} (prompt {length})"
        parts[length] = (idx, _decode(policy, tok, [pids[i] for i in idx], device, max_new,
                                      temperature, g, use_cache, cache_gib, note))

    # REASSEMBLED IN THE CALLER'S ORDER. Every row's reward is compared against the mean of ITS
    # OWN group, so a batch handed back in bucket order would standardise each completion
    # against the wrong siblings -- a silent, plausible-looking corruption of every advantage.
    B = len(pids)
    T = max(p[1][0].shape[1] for p in parts.values())
    ids = torch.full((B, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros(B, T, dtype=torch.long, device=device)
    resp_mask = torch.zeros(B, T, dtype=torch.long, device=device)
    texts = [None] * B
    for idx, (bi, ba, br, bt) in parts.values():
        w = bi.shape[1]
        for j, i in enumerate(idx):
            ids[i, :w], attn[i, :w], resp_mask[i, :w] = bi[j], ba[j], br[j]
            texts[i] = bt[j]
    return ids, attn, resp_mask, texts


@torch.no_grad()
def _decode(policy, tok, pids, device, max_new, temperature, g, use_cache, cache_gib, note):
    """One bucket: sample a completion for every row of `pids`, which are all the same length.

    RIGHT padding with a per-row write cursor, the convention the model is trained under: a
    left-padded row would give a query with no visible key, an all -inf attention row, and a
    NaN that spreads through the value matmul into every later layer."""
    pad, eos = tok.pad_token_id, getattr(tok, "eos_token_id", None)
    B = len(pids)
    L0 = max(1, max(len(p) for p in pids))
    T = L0 + max_new
    ids = torch.full((B, T), pad, dtype=torch.long, device=device)
    attn = torch.zeros(B, T, dtype=torch.long, device=device)
    resp_mask = torch.zeros(B, T, dtype=torch.long, device=device)
    for i, p in enumerate(pids):
        if p:
            ids[i, :len(p)] = torch.tensor(p, device=device)
            attn[i, :len(p)] = 1
    # THE CACHE APPLIES BECAUSE EVERY ROW IS AT THE SAME POSITION -- which is what the caller's
    # bucketing guarantees, and the only thing it is for. The cache appends one column for ALL
    # rows at once, so a shorter row would have its next token written at another row's
    # position; equal lengths make the cursors move in lockstep and the append correct.
    #
    # An empty prompt still has no position to append after, so that one case falls back to
    # recomputing the prefix. It cannot arise from a real problem, and refusing to decode would
    # be worse than being slow.
    from helpers.kv_cache import Cache, budget_bytes
    same_len = len(set(len(p) for p in pids)) == 1 and all(pids)
    cache = (Cache(len(policy.blocks), budget_bytes(cache_gib))
             if (use_cache and same_len) else None)
    cur = torch.tensor([max(len(p), 1) for p in pids], device=device)
    done = torch.zeros(B, dtype=torch.bool, device=device)
    rows = torch.arange(B, device=device)
    # one forward per generated column; the bar erases itself so it does not leave a line per
    # training step
    for _ in progress(range(max_new), desc=tag("cot") + " rollout" + note, total=max_new):
        active = ~done
        if not bool(active.any()):
            break
        hi = int(cur.max())
        # ONE POSITION'S LOGITS, NOT THE WHOLE SEQUENCE'S. forward() projects every position to
        # the vocabulary and returns (B, T, 50259); sampling then keeps one column of it and
        # throws the rest away. At 24 sequences and 400 tokens that discarded tensor is 1.9 GB,
        # it is reallocated at every generated token, and it grows as the sequence does -- which
        # is what filled the card. Taking the hidden states and projecting only the positions
        # actually needed costs (B, 50259): 4.8 MB, flat.
        if cache is None:
            h = policy.hidden_states(input_ids=ids[:, :hi], attention_mask=attn[:, :hi])
            step_h = h[rows, cur - 1]
        else:
            # the prompt once, then one column at a time; len(cache) is what it already holds
            lo = len(cache)
            h = policy.hidden_states(input_ids=ids[:, lo:hi], cache=cache)
            step_h = h[:, -1]
        step_logits = policy.head(step_h).float()
        probs = F.softmax(step_logits / max(temperature, 1e-6), dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1).to(device)
        r, c = rows[active], cur[active]
        ids[r, c] = nxt[active]
        attn[r, c] = 1
        resp_mask[r, c] = 1
        cur = cur + active.long()
        if eos is not None:
            done = done | (active & (nxt == eos))
    hi = int(cur.max())
    ids, attn, resp_mask = ids[:, :hi], attn[:, :hi], resp_mask[:, :hi]
    texts = []
    for i in range(B):
        sel = ids[i][resp_mask[i].bool()].tolist()
        texts.append(decode(tok, sel))
    return ids, attn, resp_mask, texts


def _chunk_lp(head, h, tgt, need_entropy):
    """Log-probs (and optionally entropy) for ONE slice of flattened positions.

    Called through torch.utils.checkpoint: the (chunk, vocab) logits and their log-softmax
    exist during this call, are freed when it returns, and are recomputed slice by slice in the
    backward pass. What survives is (chunk,) -- four bytes a token instead of four bytes a token
    TIMES THE VOCABULARY."""
    logits = head(h).float()
    la = F.log_softmax(logits, dim=-1)
    lp = la.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    ent = -(la.exp() * la).sum(-1) if need_entropy else lp.new_zeros(())
    return lp, ent


def forward_logprobs(policy, ids, attn, need_entropy=False, chunk=LP_CHUNK):
    """Per-token log-probs in the TOKEN FRAME (index j scores ids[:, j] given the prefix), and
    optionally per-token entropy at the same sites.

    CHUNKED OVER TOKENS, because the whole-sequence version could not fit. It built the full
    (batch, tokens, vocab) logits and then a second tensor the same size for the log-softmax:
    at 24 rollouts of 2,495 tokens over a 50,259 vocabulary that is 11.2 GiB EACH, and this is
    called four times per step -- old, reference, and once per GRPO epoch. It is the reason the
    stage ran out of memory the moment completions got long, and it got worse exactly as the
    policy learned to think for longer, which is the behaviour the stage exists to produce.

    Flattening before slicing is what makes `chunk` mean what it says: this many rows of
    (vocab) exist at a time, whatever shape the batch has. Slicing the position axis alone
    would still hand the projection every sequence at once.

    The arithmetic is unchanged -- same positions, same targets, same normalisation."""
    h = policy.hidden(input_ids=ids, attention_mask=attn)
    B, T, D = h.shape
    tgt = ids[:, 1:]
    hf, tf = h[:, :-1].reshape(-1, D), tgt.reshape(-1)
    n = hf.shape[0]
    step = int(chunk) or n
    lps, ents = [], []
    for a in range(0, n, step):
        b = min(a + step, n)
        if torch.is_grad_enabled() and hf.requires_grad:
            lp, ent = checkpoint(_chunk_lp, policy.head, hf[a:b], tf[a:b], need_entropy,
                                 use_reentrant=False)
        else:
            # no graph to trade against recomputation: checkpoint would only cost a second pass
            lp, ent = _chunk_lp(policy.head, hf[a:b], tf[a:b], need_entropy)
        lps.append(lp)
        if need_entropy:
            ents.append(ent)
    lp_tok = F.pad(torch.cat(lps).view(B, T - 1), (1, 0))
    ent_tok = F.pad(torch.cat(ents).view(B, T - 1), (1, 0)) if need_entropy else None
    return lp_tok, ent_tok


def group_advantages(rewards, group_size, eps=1e-4):
    """Standardise rewards WITHIN each group of `group_size` consecutive completions.

    This is the whole of GRPO's credit assignment: a completion is good exactly insofar as it
    beat its siblings on the same problem. A group whose completions all scored the same has
    zero spread and therefore contributes no gradient -- which is correct, since it carries no
    information about what to prefer."""
    r = rewards.view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, unbiased=False, keepdim=True) + eps)
    return adv.reshape(-1)


def _masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# stage runner
# --------------------------------------------------------------------------- #
def run(policy, ref, tok, problems, ckdir, args, log, monitor, preview=None, eval_set=None):
    """GRPO-train `policy` against the arithmetic verifier, with a KL to the frozen `ref`.
    `problems` is a list of {"question", "answer"}. Returns the policy."""
    cfg = config.COT
    total, lr = args.cot_steps, args.cot_lr
    G = max(int(args.cot_group), 1)
    device = next(policy.parameters()).device

    ck = load_ckpt(ckdir, STAGE) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != total)
    if ck and ck.get("done") and not extend:
        policy.load_state_dict(ck["model"]); policy.eval()
        log(f"cot done -> {ck.get('eval')}")
        monitor(STAGE, load_hist(ckdir, STAGE))
        return policy
    if extend:
        log(f"cot: budget {ck.get('total')} -> {total}, extending from step {ck['step']}")
    if G < 2:
        log("cot: WARNING group_size < 2 -- every group-relative advantage is zero and the "
            "policy cannot learn; set --cot_group >= 2")

    opt = MasterAdamW(list(policy.parameters()), lr=lr)
    sched = CosineLR(opt, lr, total, args.lr_min_factor, args.lr_schedule)
    log(f"cot: lr={sched.describe()} steps={total} group={G} "
        f"prompts/step={args.batch} rollouts/step={args.batch * G} "
        f"max_new={cfg['max_new_tokens'] or 'fills the window'} "
        f"clip={cfg['clip_eps']} kl_coef={cfg['kl_coef']} "
        f"-> {ckpt_path(ckdir, STAGE)}")
    g = torch.Generator().manual_seed(args.seed * 1000003 + 77)       # rollout sampling
    dg = torch.Generator().manual_seed(args.seed * 1000003 + 78)      # problem order
    start = 0
    hist = []           # loaded below, truncated at the resume point
    if ck is not None and (not ck["done"] or extend):
        policy.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for grp in opt.param_groups:
            grp["lr"] = lr
        start = ck["step"]; g.set_state(ck["gens"][0]); dg.set_state(ck["gens"][1])
        restore_rng(ck)
        log(f"resume cot @ {start}")
        hist = load_hist(ckdir, STAGE, upto=start)
        monitor(STAGE, hist, start)

    N = len(problems)
    params = list(policy.parameters())
    bar = progress(range(start, total), desc=tag("cot"), initial=start, total=total)
    for step in bar:
        cur_lr = sched.step(step)
        # ---- 1. a batch of problems, each repeated group_size times ---- #
        policy.eval()
        idx = torch.randint(0, N, (args.batch,), generator=dg).tolist()
        batch = [problems[i] for i in idx]
        prompts, golds = [], []
        for p in batch:
            for _ in range(G):
                prompts.append(verifier.prompt(p["question"], cfg))
                golds.append(p["answer"])
        # ---- 2. rollouts ---- #
        ids, attn, resp_mask, texts = rollout(policy, tok, prompts, device,
                                              cfg["max_new_tokens"], cfg["gen_temperature"],
                                              g, args.max_len,
                                              use_cache=getattr(args, "kv_cache", True)
                                              and cfg.get("kv_cache", True),
                                              cache_gib=getattr(args, "kv_cache_size", 0.0)
                                              or cfg.get("kv_cache_size", 0.0))
        rmask = resp_mask.float()
        if rmask.sum() < 1:
            continue
        # ---- 3. verify: the reward is COMPUTED, not predicted ---- #
        scored = [verifier.score(t, gold, cfg) for t, gold in zip(texts, golds)]
        rewards = torch.tensor([s["reward"] for s in scored], dtype=torch.float32,
                               device=device)
        # ---- 4. group-relative advantages, broadcast over each completion's tokens ---- #
        adv_seq = group_advantages(rewards, G)
        adv = adv_seq.unsqueeze(1) * rmask
        with torch.no_grad():
            old_lp, _ = forward_logprobs(policy, ids, attn)
            ref_lp, _ = forward_logprobs(ref, ids, attn)
        # ---- 5. GRPO epochs ---- #
        policy.train()
        pg_l = kl_l = ent_l = clipped = 0.0
        for _ in progress(range(cfg["grpo_epochs"]), desc=tag("cot") + " grpo epochs",
                          total=cfg["grpo_epochs"]):
            new_lp, ent = forward_logprobs(policy, ids, attn, need_entropy=True)
            ratio = torch.exp((new_lp - old_lp).clamp(-20, 20))
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv
            pg_loss = -_masked_mean(torch.min(s1, s2), rmask)
            # k3 estimator of KL(pi || pi_ref): non-negative, unbiased on a single sample
            d = (ref_lp - new_lp).clamp(-20, 20)
            kl = _masked_mean(d.exp() - d - 1.0, rmask)
            ent_mean = _masked_mean(ent, rmask)
            loss = pg_loss + cfg["kl_coef"] * kl - cfg["ent_coef"] * ent_mean
            opt.zero_grad(); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            with torch.no_grad():
                clipped = _masked_mean(((ratio - 1.0).abs() > cfg["clip_eps"]).float(),
                                       rmask).item()
            pg_l, kl_l, ent_l = pg_loss.item(), kl.item(), ent_mean.item()

        # THE BEST COMPLETION OF THIS STEP, printed. Averages say whether the run is moving;
        # they never say what the model actually wrote. The highest-ADVANTAGE completion is the
        # one this step pushed hardest toward, so it is the single sample that explains what the
        # update is teaching -- and if it is right for the wrong reason, this is where that
        # shows. Advantage rather than reward because the advantage is what multiplies the
        # gradient: a completion can be the best in a group of poor ones and still be what the
        # policy is being moved towards.
        best = int(torch.argmax(adv_seq).item())
        b = scored[best]
        n_kept = int(rmask[best].sum().item())
        think = verifier.think_text(texts[best])
        log(f"\n----- {tag('cot')} step {step}: best of {len(scored)} completions "
            f"(advantage {adv_seq[best].item():+.3f}, reward {rewards[best].item():+.3f}, "
            f"correct {b['correct']:.0f}, think-tag {b.get('think_fmt', 0):.0f}, "
            f"answer-tag {b.get('answer_fmt', 0):.0f}, "
            f"grounded {b.get('grounded', 0):.0f}, {n_kept} generated tokens) -----")
        # NOTHING IS TRUNCATED. A response cut at 1,200 characters hides the end, which is
        # where the answer is and where a run-on completion shows what it did instead of
        # stopping -- and those are the two things worth looking at. Each part gets its own
        # heading, on its own line, with a blank line between: a reasoning trace is read, not
        # scanned, and it does not fit on one line at any length worth printing.
        log(f"\nPROMPT:\n{prompts[best]}")
        # When there is no think block the completion was malformed, and the RAW text is shown
        # under a name that says so -- a blank THINK would hide exactly the failure worth seeing.
        if verifier.has_think_tags(texts[best]) and think.strip():
            log(f"\nTHINK:\n{think}")
        else:
            log(f"\nTHINK: (no {verifier.THINK_OPEN} block, raw completion)\n{texts[best]}")
        log(f"\nANSWER\n{b['pred']}")
        log(f"\nGOLD     {golds[best]!r}")
        log("-" * 78 + "\n")

        def mean(k):
            return sum(s[k] for s in scored) / len(scored)

        # WITHIN-GROUP SPREAD, which is what GRPO actually runs on. There is no value network:
        # a completion's advantage is its reward minus its group's mean, over the group's
        # standard deviation. A group whose completions all score the same therefore has zero
        # advantage everywhere and contributes NOTHING -- and a run in which most groups are
        # like that is a run that looks healthy and learns nothing. `dead_groups` is the single
        # number that says so, and it is the first thing to read when reward will not move.
        rg = rewards.view(-1, G)
        dead = float((rg.std(dim=1, unbiased=False) < 1e-8).float().mean().item())
        think_words = [s["think_len"] for s in scored]
        think_sorted = sorted(think_words)
        rec = {"step": step, "loss": pg_l + cfg["kl_coef"] * kl_l,
               "reward": rewards.mean().item(),
               "accuracy": mean("correct"),          # the verifier's verdict, not a proxy
               "think_fmt": mean("think_fmt"),      # a closed <think> span
               "answer_fmt": mean("answer_fmt"),    # a closed <answer> span
               "format": mean("formatted"),         # both, ordered
               "grounded": mean("grounded"),         # thinking that mentions the givens
               "think_len": mean("think_len"),       # THE aha-moment curve
               # THE MEAN HIDES THE ARRIVAL. Longer deliberation appears in a FEW completions
               # first, and a mean over 24 of them moves by a word or two while the best has
               # doubled. The maximum and the 90th percentile show it when the mean cannot.
               "think_len_max": float(max(think_words) if think_words else 0.0),
               "think_len_p90": float(think_sorted[max(0, int(0.9 * len(think_sorted)) - 1)]
                                      if think_sorted else 0.0),
               "aha": mean("aha"),                   # reflection markers, a crude proxy
               "resp_len": rmask.sum(1).mean().item(),
               "reward_std": float(rg.std(dim=1, unbiased=False).mean().item()),
               "dead_groups": dead,                  # groups with no spread: wasted rollouts
               "adv_abs": float(adv_seq.abs().mean().item()),
               "kl_ref": kl_l, "pg_loss": pg_l, "entropy": ent_l,
               "clip_frac": clipped, "gnorm": float(gnorm), "lr": cur_lr}
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(acc=f"{rec['accuracy']:.2f}", rew=f"{rec['reward']:+.2f}",
                            think=f"{rec['think_len']:.0f}/{rec['think_len_max']:.0f}w",
                            dead=f"{rec['dead_groups']:.2f}", aha=f"{rec['aha']:.2f}",
                            kl=f"{rec['kl_ref']:.4f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, STAGE, policy, opt, step + 1, total, [g, dg])
            save_hist(ckdir, STAGE, hist)
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(STAGE, hist, step + 1)
        # THIS STAGE HAS ITS OWN SAMPLE CADENCE, --cot_samples_every, falling back to the
        # pipeline-wide one. A GRPO step is not comparable to an LM step: it costs a whole
        # group of rollouts, there are far fewer of them, and what the run is watching -- the
        # format appearing, the reasoning lengthening -- is visible in the text long before it
        # is visible in a curve.
        every = (getattr(args, "cot_samples_every", 0)
                 or getattr(args, "print_samples_every_steps", args.plot_every_steps))
        if preview and every > 0 and (step + 1) % every == 0:
            preview(policy, STAGE, step + 1)

    policy.eval()
    # end-of-stage evaluation: verified accuracy on held-out problems, greedy-ish single draws
    evald = None
    try:
        pool = eval_set or problems
        n_ev = min(cfg["eval_problems"], len(pool))
        idx = torch.randint(0, len(pool), (n_ev,), generator=dg).tolist()
        sel = [pool[i] for i in idx]
        ids, attn, resp_mask, texts = rollout(
            policy, tok, [verifier.prompt(p["question"], cfg) for p in sel], device,
            cfg["max_new_tokens"], cfg["gen_temperature"], g, args.max_len,
            use_cache=getattr(args, "kv_cache", True) and cfg.get("kv_cache", True),
            cache_gib=getattr(args, "kv_cache_size", 0.0) or cfg.get("kv_cache_size", 0.0))
        sc = [verifier.score(t, p["answer"], cfg) for t, p in zip(texts, sel)]
        evald = {"problems": n_ev,
                 "accuracy": sum(s["correct"] for s in sc) / max(len(sc), 1),
                 "format": sum(s["formatted"] for s in sc) / max(len(sc), 1),
                 "mean_think_words": sum(s["think_len"] for s in sc) / max(len(sc), 1),
                 "aha_rate": sum(s["aha"] for s in sc) / max(len(sc), 1)}
        first = hist[0] if hist else {}
        log(f"cot done: verified accuracy={evald['accuracy']:.3f} on {n_ev} held-out problems, "
            f"format={evald['format']:.3f}, mean think length={evald['mean_think_words']:.1f} "
            f"words (from {first.get('think_len', float('nan')):.1f} at step 0), "
            f"reflection rate={evald['aha_rate']:.3f}")
    except Exception as e:                                            # noqa: BLE001
        log(f"cot final eval failed: {e}")
    save_ckpt(ckdir, STAGE, policy, opt, total, total, [g, dg], evald=evald)
    save_hist(ckdir, STAGE, hist)
    monitor(STAGE, hist)
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return policy
