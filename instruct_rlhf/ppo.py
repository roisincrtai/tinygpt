"""
ppo.py -- PPO for RLHF instruction tuning (stage 8), the standard InstructGPT-style recipe
scaled down to ZetaGPT.

    THE POLICY STARTS FROM THE SFT MODEL. Each step:
      1. sample a batch of prompts from the preference data
      2. ROLLOUT: the policy autoregressively samples a response per prompt
      3. score each (prompt + response) with the FROZEN stage-7 reward model (scalar)
      4. per-token reward  r_t = -kl_coef * ( log pi(y_t|.) - log pi_sft(y_t|.) ),
         with the reward-model score added at the LAST response token -- the KL term keeps
         the policy near the SFT reference exactly as DPO's beta term does
      5. GAE(gamma, lam) advantages against a learned VALUE HEAD (linear on the policy's
         hidden states), whitened per batch
      6. ppo_epochs epochs of the clipped surrogate
             L = -E[ min( rho_t A_t, clip(rho_t, 1-eps, 1+eps) A_t ) ]
                 + vf_coef * 1/2 (V - R)^2  - ent_coef * H[pi]
         where rho_t = pi_new(y_t) / pi_old(y_t) on the response tokens only.

All knobs live in config.RLHF (steps/lr also on the CLI as --rlhf_steps/--rlhf_lr).
Checkpoint: checkpoints/rlhf/checkpoint_<model>_<pe>_instruct-rlhf.pt (policy; value head
            under extra["vhead"]).
Figure:     outputs/plots/rlhf/dynamics_<model>_<pe>_instruct-rlhf.pdf (reward, KL, PPO
            losses, entropy, length).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import default_config as config
from helpers import (tag, progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     restore_rng, CosineLR, ckpt_path, amp)  # progress: every loop reports through tqdm

NAME = "RLHF"
STAGE = "rlhf"


class ValueHead(nn.Module):
    """Per-position scalar value read off the policy trunk's hidden states."""
    def __init__(self, n_embd):
        super().__init__()
        self.v = nn.Linear(n_embd, 1)
        nn.init.normal_(self.v.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.v.bias)

    def forward(self, h):                     # (B,T,C) -> (B,T)
        return self.v(h).squeeze(-1)


# --------------------------------------------------------------------------- #
# rollout + scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(policy, tok, prompts, device, max_new, temperature, g, max_len):
    """Sample one response per prompt, batched with RIGHT padding and a per-row write cursor
    (exactly the padding convention the model is trained under; with LEFT padding a pad query
    would have no visible key -- an all -inf attention row -- and the resulting NaN poisons
    every later layer through the masked value matmul).

    Returns (ids, attn, resp_mask): the prompt+response batch and a (B,T) mask of the
    GENERATED tokens (EOS included; nothing is generated after a row hits EOS)."""
    pad, eos = tok.pad_token_id, getattr(tok, "eos_token_id", None)
    pids = []
    keep = max(max_len - max_new, 8)
    for p in prompts:
        pid = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        pids.append(pid[-keep:] if len(pid) > keep else pid)
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
    cur = torch.tensor([max(len(p), 1) for p in pids], device=device)   # next write position
    done = torch.zeros(B, dtype=torch.bool, device=device)
    rows = torch.arange(B, device=device)
    # one forward per generated column; the bar erases itself (leave=False) so it does not
    # accumulate a line per training step
    for _ in progress(range(max_new), desc=tag("rlhf") + " rollout", total=max_new):
        hi = int(cur.max())
        logits = policy(input_ids=ids[:, :hi], attention_mask=attn[:, :hi]).logits
        step_logits = logits[rows, cur - 1].float()                     # (B,V) next-token dists
        probs = F.softmax(step_logits / max(temperature, 1e-6), dim=-1).cpu()
        nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1).to(device)
        active = ~done
        if not bool(active.any()):
            break
        r, c = rows[active], cur[active]
        ids[r, c] = nxt[active]
        attn[r, c] = 1
        resp_mask[r, c] = 1
        cur = cur + active.long()
        if eos is not None:
            done = done | (active & (nxt == eos))
    hi = int(cur.max())
    return ids[:, :hi], attn[:, :hi], resp_mask[:, :hi]


def forward_scores(policy, ids, attn, vhead=None, need_entropy=False):
    """One trunk pass -> per-token log-probs (token frame: index j scores ids[:, j] given the
    prefix), and optionally per-token values / entropies AT THE PREDICTION SITE of each token
    (both padded to the token frame). Values reuse the same hidden states as the logits."""
    h = policy.hidden(input_ids=ids, attention_mask=attn)          # (B,T,C)
    logits = policy.head(h)[:, :-1].float()                        # prediction sites
    lp_all = F.log_softmax(logits, dim=-1)
    lp = lp_all.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)   # (B,T-1)
    lp_tok = F.pad(lp, (1, 0))
    v_tok = F.pad(vhead(h[:, :-1].float()), (1, 0)) if vhead is not None else None
    ent_tok = (F.pad(-(lp_all.exp() * lp_all).sum(-1), (1, 0)) if need_entropy else None)
    return lp_tok, v_tok, ent_tok


@torch.no_grad()
def gae(rewards, values, resp_mask, gamma, lam):
    """Generalized Advantage Estimation over the response region (token frame). The response
    columns of a row are contiguous; masked columns reset the accumulator, so each row's
    recursion runs over exactly its own generated tokens (V after the last token = 0).
    Returns (advantages, returns), both (B,T) and zero off the response."""
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(B, device=rewards.device)
    nextval = torch.zeros(B, device=rewards.device)
    for j in reversed(range(T)):
        m = resp_mask[:, j].float()
        delta = rewards[:, j] + gamma * nextval - values[:, j]
        lastgaelam = m * (delta + gamma * lam * lastgaelam)
        adv[:, j] = lastgaelam
        nextval = m * values[:, j]
    ret = (adv + values) * resp_mask.float()
    return adv, ret


def _masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


# --------------------------------------------------------------------------- #
# stage runner
# --------------------------------------------------------------------------- #
def run(policy, ref, rm, tok, train_pairs, ckdir, args, log, monitor, preview=None):
    """PPO-train `policy` (a copy of the SFT model) against the frozen reward model `rm`,
    with a per-token KL penalty to the frozen SFT reference `ref`; return the policy."""
    cfg = config.RLHF
    total, lr = args.rlhf_steps, args.rlhf_lr
    device = next(policy.parameters()).device
    n_embd = policy.lnf.normalized_shape[0]
    vhead = ValueHead(n_embd).to(device)

    ck = load_ckpt(ckdir, STAGE) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != total)
    if ck and ck.get("done") and not extend:
        policy.load_state_dict(ck["model"]); policy.eval()
        log(f"rlhf done -> {ck.get('eval')}")
        monitor(STAGE, load_hist(ckdir, STAGE))
        return policy
    if extend:
        log(f"rlhf: budget {ck.get('total')} -> {total}, extending from step {ck['step']}")

    opt = MasterAdamW(list(policy.parameters()) + list(vhead.parameters()), lr=lr)
    sched = CosineLR(opt, lr, total, args.lr_min_factor, args.lr_schedule)
    log(f"rlhf: lr={sched.describe()} steps={total} ppo_epochs={cfg['ppo_epochs']} "
        f"clip={cfg['clip_eps']} kl_coef={cfg['kl_coef']} max_new={cfg['max_new_tokens']} "
        f"-> {ckpt_path(ckdir, STAGE)}")
    g = torch.Generator().manual_seed(args.seed * 1000003 + 55)      # rollout sampling
    dg = torch.Generator().manual_seed(args.seed * 1000003 + 56)     # prompt order
    start = 0
    hist = []           # loaded below, truncated at the resume point
    if ck is not None and (not ck["done"] or extend):
        policy.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if (ck.get("extra") or {}).get("vhead"):
            vhead.load_state_dict(ck["extra"]["vhead"])
        for grp in opt.param_groups:
            grp["lr"] = lr
        start = ck["step"]; g.set_state(ck["gens"][0]); dg.set_state(ck["gens"][1])
        restore_rng(ck)
        log(f"resume rlhf @ {start}")
        hist = load_hist(ckdir, STAGE, upto=start)
        monitor(STAGE, hist, start)

    Ntr = len(train_pairs)
    params = list(policy.parameters()) + list(vhead.parameters())
    bar = progress(range(start, total), desc=tag("rlhf"),
                   initial=start, total=total)
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        # ---- 1-2: rollouts from the current policy ---- #
        policy.eval()
        idx = torch.randint(0, Ntr, (args.batch,), generator=dg).tolist()
        prompts = [train_pairs[i] for i in idx]
        ids, attn, resp_mask = rollout(policy, tok, prompts, device,
                                       cfg["max_new_tokens"], cfg["gen_temperature"],
                                       g, args.max_len)
        rmask = resp_mask.float()
        n_resp = rmask.sum()
        if n_resp < 1:
            continue
        with torch.no_grad():
            # ---- 3: reward-model score of each rollout ---- #
            score = rm(ids, attn)                                           # (B,)
            # ---- old policy / reference per-token log-probs and values ---- #
            # bf16 through both trunks. forward_scores ends in a log_softmax, which autocast
            # keeps in fp32, so the log-probabilities these ratios are built from are fp32
            # whatever the matmuls ran in.
            with amp(args):
                old_lp, values, _ = forward_scores(policy, ids, attn, vhead=vhead)
                ref_lp, _, _ = forward_scores(ref, ids, attn)
            klt = (old_lp - ref_lp) * rmask                                 # per-token KL sample
            # ---- 4: per-token rewards (KL penalty; score at the last token) ---- #
            rewards = -cfg["kl_coef"] * klt
            pos = torch.arange(ids.size(1), device=device).unsqueeze(0)
            last = (pos * resp_mask).argmax(dim=1)                          # last response col
            rewards[torch.arange(ids.size(0), device=device), last] += score
            rewards = rewards * rmask
            # ---- 5: GAE ---- #
            adv, ret = gae(rewards, values * rmask, resp_mask, cfg["gamma"], cfg["lam"])
            if cfg["whiten_adv"]:
                mu = _masked_mean(adv, rmask)
                var = _masked_mean((adv - mu) ** 2, rmask)
                adv = (adv - mu) / (var.sqrt() + 1e-6)
                adv = adv * rmask
        # ---- 6: PPO epochs on the clipped surrogate ---- #
        policy.train()
        pg_l = v_l = ent_l = 0.0
        for _ in progress(range(cfg["ppo_epochs"]), desc=tag("rlhf") + " ppo epochs",
                          total=cfg["ppo_epochs"]):
            with amp(args):
                new_lp, v_pred, ent = forward_scores(policy, ids, attn, vhead=vhead,
                                                     need_entropy=True)
            ratio = torch.exp((new_lp - old_lp).clamp(-20, 20))
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"]) * adv
            pg_loss = -_masked_mean(torch.min(s1, s2), rmask)
            v_loss = 0.5 * _masked_mean((v_pred - ret) ** 2, rmask)
            ent_mean = _masked_mean(ent, rmask)
            loss = pg_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent_mean
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            pg_l, v_l, ent_l = pg_loss.item(), v_loss.item(), ent_mean.item()
        rec = {"step": step, "loss": pg_l + cfg["vf_coef"] * v_l,
               "reward": score.mean().item(),
               "kl_ref": (klt.sum() / n_resp).item(),
               "pg_loss": pg_l, "v_loss": v_l, "entropy": ent_l,
               "resp_len": (rmask.sum(1).mean()).item(), "lr": cur_lr}
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(reward=f"{rec['reward']:+.3f}", kl=f"{rec['kl_ref']:+.3f}",
                            pg=f"{rec['pg_loss']:+.3f}", v=f"{rec['v_loss']:.3f}",
                            len=f"{rec['resp_len']:.0f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, STAGE, policy, opt, step + 1, total, [g, dg],
                      extra={"vhead": vhead.state_dict()})
            save_hist(ckdir, STAGE, hist)
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(STAGE, hist, step + 1)
        # the generation examples keep their own cadence, --print_samples_every_steps
        every = getattr(args, "print_samples_every_steps", args.plot_every_steps)
        if preview and every > 0 and (step + 1) % every == 0:
            preview(policy, STAGE, step + 1)

    policy.eval()
    # end-of-stage evaluation: mean reward-model score of fresh rollouts on held-out prompts
    evald = None
    try:
        with torch.no_grad():
            n_ev = min(32, Ntr)
            idx = torch.randint(0, Ntr, (n_ev,), generator=dg).tolist()
            ids, attn, resp_mask = rollout(policy, tok, [train_pairs[i] for i in idx],
                                           device, cfg["max_new_tokens"],
                                           cfg["gen_temperature"], g, args.max_len)
            evald = {"mean_reward": rm(ids, attn).mean().item(),
                     "mean_len": resp_mask.float().sum(1).mean().item()}
        last = hist[-1] if hist else {}
        log(f"rlhf done: mean rollout reward={evald['mean_reward']:+.3f} "
            f"mean length={evald['mean_len']:.1f} "
            f"final KL to sft={last.get('kl_ref', float('nan')):+.4f}")
    except Exception as e:                                            # noqa: BLE001
        log(f"rlhf final eval failed: {e}")
    save_ckpt(ckdir, STAGE, policy, opt, total, total, [g, dg], evald=evald,
              extra={"vhead": vhead.state_dict()})
    save_hist(ckdir, STAGE, hist)
    monitor(STAGE, hist)
    del opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return policy
