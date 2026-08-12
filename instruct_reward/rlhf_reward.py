"""
rlhf_reward.py -- the RLHF reward model: ZetaGPT trunk + scalar reward head, trained as a
BINARY CLASSIFIER with sigmoid + binary cross-entropy.

    RewardModel(base)                the model: r(x, y) = w^T h_last + b, a single scalar
                                     read off the final-layer hidden state of the LAST real
                                     token of "prompt + response"
    run(model, enc, ...)             the training loop (stage 7 consumes it)

Training data are the preference pairs of the preference batches: each pair contributes
TWO independent examples, (prompt, chosen) with label 1 and (prompt, rejected) with label 0,

    L = BCE( sigma(s_w), 1 ) + BCE( sigma(s_l), 0 )

(not the Bradley-Terry pairwise loss -- per the design, chosen/rejected are treated as
absolute positive/negative classes). The trunk is INITIALISED FROM THE SFT CHECKPOINT (stage
6 needs the reward model to speak the same distribution the policy starts from) and trained
end-to-end together with the head.

Checkpoint: checkpoints/reward/checkpoint_<model>_<pe>_reward.pt (full RewardModel state).
Figure:     outputs/plots/reward/dynamics_<model>_<pe>_reward.pdf (loss, accuracies, class
            scores, margin).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers import (tag, progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     restore_rng, CosineLR, ckpt_path, amp)

NAME = "Reward"
STAGE = "reward"


class RewardModel(nn.Module):
    """ZetaGPT trunk + scalar head. forward(ids, attn) -> (B,) reward logits.

    The score is read at the LAST REAL TOKEN of each sequence (attention_mask.sum()-1): with
    right padding that is the causal position that has seen the whole prompt+response, so a
    single linear readout there scores the full sequence."""
    def __init__(self, base):
        super().__init__()
        self.base = base                                     # a ZetaGPT (has .hidden())
        n_embd = base.lnf.normalized_shape[0]
        self.head = nn.Linear(n_embd, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        # The trunk arrives already on its device (the model factory moves it), so build the
        # head THERE too. Otherwise the head stays on CPU -- load_state_dict copies into the
        # existing parameters and keeps their device, so a caller that only loads weights
        # (stage 8) would hit "weight is on cpu but expected on mps" at the first forward.
        try:
            self.head.to(next(base.parameters()).device)
        except StopIteration:
            pass

    def forward(self, input_ids, attention_mask=None):
        h = self.base.hidden(input_ids=input_ids, attention_mask=attention_mask)   # (B,T,C)
        if attention_mask is None:
            last = torch.full((input_ids.size(0),), input_ids.size(1) - 1,
                              device=input_ids.device)
        else:
            last = attention_mask.long().sum(-1).clamp(min=1) - 1
        hl = h[torch.arange(h.size(0), device=h.device), last]                     # (B,C)
        return self.head(hl).squeeze(-1)                                           # (B,)


@torch.no_grad()
def evaluate(model, enc, pairs, batch=16):
    """Held-out metrics: pair accuracy Pr(s_w > s_l), class accuracies, BCE."""
    model.eval()
    sw, sl = [], []
    for i in progress(range(0, len(pairs), batch), desc=tag("reward") + " held-out eval",
                      total=(len(pairs) + batch - 1) // batch):
        chunk = pairs[i:i + batch]
        W = enc.encode(chunk, "chosen"); L = enc.encode(chunk, "rejected")
        sw.append(model(W[0], W[1])); sl.append(model(L[0], L[1]))
    sw = torch.cat(sw); sl = torch.cat(sl)
    ones, zeros = torch.ones_like(sw), torch.zeros_like(sl)
    bce = 0.5 * (F.binary_cross_entropy_with_logits(sw, ones)
                 + F.binary_cross_entropy_with_logits(sl, zeros)).item()
    return {"pair_acc": (sw > sl).float().mean().item(),
            "acc_w": (sw > 0).float().mean().item(),
            "acc_l": (sl < 0).float().mean().item(),
            "bce": bce, "s_w": torch.sigmoid(sw).mean().item(),
            "s_l": torch.sigmoid(sl).mean().item()}


def run(model, enc, train_pairs, ev_pairs, ckdir, args, log, monitor):
    """Train the RewardModel by sigmoid + BCE (chosen -> 1, rejected -> 0); return it in eval
    mode. Same resume/budget-extension/checkpoint/plot machinery as every other stage."""
    steps, lr = args.reward_steps, args.reward_lr
    ck = load_ckpt(ckdir, STAGE) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != steps)
    if ck and ck.get("done") and not extend:
        model.load_state_dict(ck["model"]); model.eval()
        log(f"reward done -> {ck.get('eval')}")
        monitor(STAGE, load_hist(ckdir, STAGE))
        return model
    if extend:
        log(f"reward: budget {ck.get('total')} -> {steps}, extending from step {ck['step']}")

    opt = MasterAdamW(model.parameters(), lr=lr)
    sched = CosineLR(opt, lr, steps, args.lr_min_factor, args.lr_schedule)
    log(f"reward: lr={sched.describe()} steps={steps} batch={args.batch} "
        f"(sigmoid + BCE, chosen=1 / rejected=0) -> {ckpt_path(ckdir, STAGE)}")
    g = torch.Generator().manual_seed(args.seed * 1000003 + 41)
    start = 0
    hist = []           # loaded below, truncated at the resume point
    if ck is not None and (not ck["done"] or extend):
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for grp in opt.param_groups:
            grp["lr"] = lr
        start = ck["step"]; g.set_state(ck["gens"][0]); restore_rng(ck)
        log(f"resume reward @ {start}")
        hist = load_hist(ckdir, STAGE, upto=start)
        monitor(STAGE, hist, start)
    Ntr = len(train_pairs)
    model.train()
    bar = progress(range(start, steps), desc=tag("reward"),
                   initial=start, total=steps)
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        idx = torch.randint(0, Ntr, (args.batch,), generator=g).tolist()
        chunk = [train_pairs[i] for i in idx]
        W = enc.encode(chunk, "chosen"); L = enc.encode(chunk, "rejected")
        # bf16 through the trunk and the scalar head; the loss is on autocast's fp32 list, so
        # the margin between the two scores is never computed at bf16's resolution.
        with amp(args):
            sw = model(W[0], W[1]); sl = model(L[0], L[1])
            logits = torch.cat([sw, sl])
            labels = torch.cat([torch.ones_like(sw), torch.zeros_like(sl)])
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        with torch.no_grad():
            rec = {"step": step, "loss": loss.item(),
                   "pair_acc": (sw > sl).float().mean().item(),
                   "acc_w": (sw > 0).float().mean().item(),
                   "acc_l": (sl < 0).float().mean().item(),
                   "s_w": torch.sigmoid(sw).mean().item(),
                   "s_l": torch.sigmoid(sl).mean().item(),
                   "margin": (sw - sl).mean().item(), "gnorm": float(gnorm),
                   "lr": cur_lr}
        hist.append(rec)
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{rec['loss']:.3f}", pair=f"{rec['pair_acc']:.2f}",
                            sw=f"{rec['s_w']:.2f}", sl=f"{rec['s_l']:.2f}",
                            gnorm=f"{float(gnorm):.2f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, STAGE, model, opt, step + 1, steps, [g])
            save_hist(ckdir, STAGE, hist)
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(STAGE, hist, step + 1)
    model.eval()
    evald = evaluate(model, enc, ev_pairs)
    save_ckpt(ckdir, STAGE, model, opt, steps, steps, [g], evald=evald)
    save_hist(ckdir, STAGE, hist)
    log(f"reward done: held-out pair_acc={evald['pair_acc']:.3f} "
        f"acc_w={evald['acc_w']:.3f} acc_l={evald['acc_l']:.3f} bce={evald['bce']:.3f}")
    monitor(STAGE, hist)
    return model


def load(ctx, train_mode=False):
    """A RewardModel with the stage-7 checkpoint loaded (errors if missing)."""
    import helpers
    ck = helpers.load_ckpt(ctx["ckdir"], STAGE)
    if not ck or "model" not in ck:
        raise SystemExit(f"[{STAGE}] no checkpoint under {ctx['ckdir']}/{STAGE}/ -- "
                         f"run stage7_train_rlhf_reward.sh first")
    rm = RewardModel(ctx["new_model"]())
    rm.load_state_dict(ck["model"])
    rm = rm.to(ctx["device"])            # belt and braces: the whole module on one device
    rm.train() if train_mode else rm.eval()
    return rm
