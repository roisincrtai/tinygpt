"""
helpers.py -- utilities every stage shares: device resolution, progress bars, the cosine
learning-rate schedule, table printing, held-out evaluation and diagnostic probes, the
mixed-precision optimizer, the checkpoint/history IO that resume and the figures are built
on, and the on-disk token cache.
"""
import array
import glob
import json
import math
import os
import shutil

try:                      # _rollout_history shows a bar; absence of tqdm must not be fatal
    from tqdm import tqdm
except Exception:                                     # noqa: BLE001
    def tqdm(it, *a, **k): return it

import torch
import torch.nn.functional as F

import default_config as config
from instruct_dpo.dpo import margin as dpo_margin, rewards as dpo_rewards


def resolve_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def progress(it, desc="", initial=0, total=None):
    """tqdm over `it`, counting in ABSOLUTE steps.

    On resume the caller iterates range(start, total), whose length is the REMAINING count, so
    a stage resumed at 30000 of 40000 rendered as `499/10000` -- a bar that restarts at zero and
    a denominator that is neither the budget nor anything else the run was asked for. Passing
    `initial=start` and `total=<budget>` makes the same iteration render `30499/40000`: the
    numerator is the true step number, the denominator the budget, and the percentage and ETA
    both refer to the whole run rather than to the tail of it."""
    try:
        from tqdm import tqdm
        return tqdm(it, desc=desc, leave=False, initial=initial, total=total)
    except Exception:
        return it


def bar(desc, unit="it", total=None, unit_scale=False, initial=0):
    """A tqdm bar driven by .update() rather than by wrapping an iterable.

    `progress` wraps an iterable and is right when the count is known. It is wrong when the
    iterable is a handful of very large files: the bar then ticks once per file and sits still
    for minutes, which is indistinguishable from a hang. This counts the units the work is
    actually made of, and the caller puts the coarse position in the postfix.

    ALWAYS PASS A TOTAL. Without one tqdm can print a count and a rate but no percentage and
    no estimate, and "156163doc [02:57, 900doc/s]" does not answer the only question being
    asked, which is how much is left. Where the exact count is unknowable in advance, measure
    the work in a unit that IS knowable -- bytes, usually, via corpus_bytes below.

    `initial` IS THE WORK ALREADY DONE, for a resumed job. A build that restarts its bar at
    zero having already written a quarter of the corpus reports 0% and an ETA for the whole
    corpus -- the two numbers a person uses to decide whether the resume worked. Passing the
    bytes already accounted for makes the percentage and the estimate refer to the corpus
    rather than to the remainder of it.

    Absence of tqdm must never be fatal, so a no-op stand-in with the same interface is
    returned instead."""
    try:
        from tqdm import tqdm
        return tqdm(desc=desc, unit=unit, total=total, leave=False, unit_scale=unit_scale,
                    dynamic_ncols=True, initial=initial)
    except Exception:                                          # noqa: BLE001
        class _Null:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def update(self, n=1): pass
            def set_postfix_str(self, s): pass
            def close(self): pass
        return _Null()


class CosineLR:
    """Cosine learning-rate schedule, shared by every trainer.

        lr(t) = lr_min + 1/2 (lr - lr_min) (1 + cos(pi t / T)),   lr_min = lr / factor

    so the rate starts at the stage's `lr`, decays smoothly, and ends at `lr / factor`
    (factor 10 by default: the floor is a tenth of the peak). The schedule is a function of
    the ABSOLUTE step, not of a counter inside the object, so a run resumed at step 30000 of
    40000 continues on the same curve instead of restarting it -- which is why `step()` takes
    the step rather than incrementing.

    `factor <= 1` or schedule "constant" degenerates to a flat `lr`, so a caller never has to
    branch on whether scheduling is enabled.
    """
    def __init__(self, opt, lr, total, factor=10.0, schedule="cosine"):
        self.opt, self.lr, self.total = opt, lr, max(int(total), 1)
        self.cosine = (schedule or "cosine").lower().startswith("cos") and factor > 1
        self.lr_min = lr / factor if self.cosine else lr

    def value(self, step):
        if not self.cosine:
            return self.lr
        t = min(max(int(step), 0), self.total) / self.total
        return self.lr_min + 0.5 * (self.lr - self.lr_min) * (1.0 + math.cos(math.pi * t))

    def step(self, step):
        """Set the optimizer's learning rate for absolute step `step`; returns the rate."""
        v = self.value(step)
        for g in self.opt.param_groups:
            g["lr"] = v
        return v

    def describe(self):
        return (f"cosine {self.lr:g} -> {self.lr_min:g} over {self.total:,} steps"
                if self.cosine else f"constant {self.lr:g}")


def table(title, rows, out=None):
    """Print `rows` (a list of (key, value) pairs, or a dict) as a bordered two-column
    table, ONE FIELD PER LINE. Used for the DATASET STATISTICS and the CORPUS SCAN RESULTS
    only -- the summaries whose many numbers are unreadable on one line. Everything else
    (per-stage settings, done lines) stays as plain single-line logging.

        +-- title ---------------------+
        | key                  value   |
        +------------------------------+
    """
    out = out or (lambda m: print(m, flush=True))
    items = list(rows.items()) if isinstance(rows, dict) else list(rows)
    items = [(str(k), "" if v is None else str(v)) for k, v in items]
    kw = max([len(k) for k, _ in items] + [1])
    vw = max([len(v) for _, v in items] + [1])
    inner = kw + vw + 4                              # "| " + k + "  " + v + " |"
    inner = max(inner, len(title) + 4)               # a long title must not overflow the box
    out("+" + f"-- {title} ".ljust(inner, "-") + "+")
    for k, v in items:
        out("| " + f"{k.ljust(kw)}  {v.ljust(vw)}".ljust(inner - 2) + " |")
    out("+" + "-" * inner + "+")


def human(n):
    """A count as a short magnitude string: 2_034_110_928 -> "2.03B".

    Reported ALONGSIDE the exact number, never instead of it. A corpus is the one thing about a
    run that cannot be recovered from the checkpoint, so the log has to say plainly how much
    text went in -- and "2,034,110,928" is a figure the eye slides over, while "2.03B" is one
    it checks against what was intended. Thresholds are decimal (1e3/1e6/1e9/1e12), which is
    how token counts are quoted everywhere in this literature."""
    n = float(n)
    for scale, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{suffix}"
    return f"{n:.0f}"


def count(n):
    """`123,456,789 (123.46M)` -- the exact count and its magnitude, which is the form every
    corpus statistic in this project is reported in. Below a thousand the magnitude would just
    repeat the number, so it is omitted: "78 (78)" is noise, not information."""
    n = int(n)
    return f"{n:,}" if abs(n) < 1000 else f"{n:,} ({human(n)})"


def run_tag(llm, dataset):
    """<llm>_<dataset> tag, filesystem-safe (e.g. gpt2_hh-rlhf, zetagpt_hh)."""
    short = dataset.split("/")[-1]
    safe = lambda s: "".join(c if (c.isalnum() or c in "-._") else "-" for c in s)
    return f"{safe(llm)}_{safe(short)}"


@torch.no_grad()
def evaluate(policy, ref, enc, pairs, beta, batch=16):
    """Held-out teacher-forced preference accuracy / mean margin / reversal rate."""
    Ms = []
    for i in progress(range(0, len(pairs), batch), desc="[eval] held-out margins",
                      total=(len(pairs) + batch - 1) // batch):
        W = enc.encode(pairs[i:i + batch], "chosen")
        L = enc.encode(pairs[i:i + batch], "rejected")
        Ms.append(dpo_margin(policy, ref, W, L, beta))
    M = torch.cat(Ms)
    return {"acc": (M > 0).float().mean().item(), "mu_M": M.mean().item(),
            "P_rev": (M <= 0).float().mean().item()}


@torch.no_grad()
def reward_hist(policy, ref, enc, pairs, beta, batch=16):
    """Per-pair teacher-forced distributions for histograms:
    r_w (chosen reward; win = r_w>0), r_l (rejected reward; lose = r_l<0), margin = r_w-r_l."""
    rw, rl = [], []
    for i in progress(range(0, len(pairs), batch), desc="[eval] reward histogram",
                      total=(len(pairs) + batch - 1) // batch):
        W = enc.encode(pairs[i:i + batch], "chosen")
        L = enc.encode(pairs[i:i + batch], "rejected")
        cr, rr = dpo_rewards(policy, ref, W, L, beta)
        rw += [round(x, 5) for x in cr.tolist()]
        rl += [round(x, 5) for x in rr.tolist()]
    return {"r_w": rw, "r_l": rl, "margin": [round(a - b, 5) for a, b in zip(rw, rl)]}


# --------------------------------------------------------------------------- #
# exposure-bias rollout (scheduled sampling). NO noise/MASK is added here -- the
# model runs normally; exposure bias comes from feeding the model its OWN samples
# for a fraction p_ro of the response, then scoring the clean targets.
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _roll_reward(policy, ref, prompt_ids, resp_ids, beta, p_ro, roll_tokens, g):
    dev = next(policy.parameters()).device
    tgt = resp_ids[:roll_tokens]
    seq = list(prompt_ids)
    pol_lp, fed = 0.0, []
    for t in range(len(tgt)):
        last = policy(input_ids=torch.tensor([seq], device=dev)).logits[:, -1]
        pol_lp += F.log_softmax(last, -1)[0, tgt[t]].item()
        if torch.rand(1, generator=g).item() < p_ro:
            nxt = torch.multinomial(F.softmax(last, -1)[0].to("cpu"), 1, generator=g).item()
        else:
            nxt = tgt[t]
        fed.append(nxt); seq.append(nxt)
    mixed = torch.tensor([list(prompt_ids) + fed], device=dev)
    rl = F.log_softmax(ref(input_ids=mixed).logits, -1)[0]
    LP = len(prompt_ids)
    return beta * (pol_lp - sum(rl[LP - 1 + t, tgt[t]].item() for t in range(len(tgt))))


@torch.no_grad()
def rollout_curve(policy, ref, tok, pairs, beta, p_grid, roll_tokens, max_len, seed=7,
                  progress_desc=True, k_roll=1):
    """Exposure-bias curve: PREFERENCE margin = reward(chosen) - reward(rejected), where BOTH
    responses are rolled out from the shared prompt. mu_M / acc / reversal as the
    self-sampled fraction p_ro goes 0 (teacher forced) -> 1 (full rollout).

    Scheduled sampling is stochastic (each token is rolled out with prob p_ro), so a single
    trajectory per pair is a noisy estimate. `k_roll` draws k independent trajectories per
    pair and averages: the per-pair margin is E[m] and the per-pair reversal is Pr(m<=0);
    P_rev is then the mean reversal probability over pairs and acc = 1 - P_rev."""
    k_roll = max(1, int(k_roll))
    g = torch.Generator().manual_seed(seed)
    prepped = []
    for p in pairs:
        pid = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        keep = max_len - roll_tokens
        if len(pid) > keep:
            pid = pid[-keep:]
        rc = tok(" " + p["chosen"], add_special_tokens=False)["input_ids"]
        rr = tok(" " + p["rejected"], add_special_tokens=False)["input_ids"]
        prepped.append((pid, rc, rr))
    curve = {}
    for p_ro in p_grid:
        it = progress(prepped, desc=f"rollout p_ro={p_ro} (k={k_roll})") if progress_desc else prepped
        Ms, Revs = [], []
        for pid, rc, rr in it:
            if not rc or not rr:
                continue
            ms = []
            for _ in range(k_roll):                        # k independent scheduled-sampling draws
                rew_c = _roll_reward(policy, ref, pid, rc, beta, p_ro, roll_tokens, g)
                rew_r = _roll_reward(policy, ref, pid, rr, beta, p_ro, roll_tokens, g)
                ms.append(rew_c - rew_r)                   # preference margin under this rollout
            ms = torch.tensor(ms)
            Ms.append(ms.mean().item())                    # per-pair expected margin E[m]
            Revs.append((ms <= 0).float().mean().item())   # per-pair reversal probability Pr(m<=0)
        M = torch.tensor(Ms); R = torch.tensor(Revs)
        p_rev = R.mean().item() if R.numel() else float("nan")
        curve[f"{p_ro:.2f}"] = {"mu_M": M.mean().item(), "acc": 1.0 - p_rev, "P_rev": p_rev,
                                "std_M": M.std(unbiased=False).item(), "k_roll": k_roll,
                                "margins": [round(x, 5) for x in M.tolist()]}  # per-pair mean (for histograms)
    return curve


# --------------------------------------------------------------------------- #
# mixed-precision optimizer: bf16 live weights, fp32 master weights + AdamW moments
# --------------------------------------------------------------------------- #
class MasterAdamW:
    """AdamW with fp32 master weights.

    Live parameters may be bf16 (or any dtype); each step casts their gradients to fp32,
    updates fp32 masters with a stock AdamW (fp32 moments), and copies the masters back
    into the live parameters. For fp32 live parameters this reduces to plain AdamW at the
    cost of one extra weight copy. Accepts either an iterable of parameters or AdamW-style
    param groups; `param_groups` is the inner optimizer's (so lr edits pass through)."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0):
        params = list(params)
        groups = params if (params and isinstance(params[0], dict)) else [{"params": params, "lr": lr}]
        self._pairs = []                    # [(live, master)] in group order
        inner_groups = []
        for g in groups:
            live = list(g["params"])
            masters = [p.detach().clone().float() for p in live]
            self._pairs += list(zip(live, masters))
            ig = {k: v for k, v in g.items() if k != "params"}
            ig.setdefault("lr", lr)
            ig["params"] = masters
            inner_groups.append(ig)
        self.inner = torch.optim.AdamW(inner_groups, lr=lr, weight_decay=weight_decay)

    @property
    def param_groups(self):
        return self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        for p, _ in self._pairs:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for p, m in self._pairs:
            m.grad = None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        for p, m in self._pairs:
            p.copy_(m.to(p.dtype))

    def state_dict(self):
        return {"inner": self.inner.state_dict(), "masters": [m for _, m in self._pairs]}

    @torch.no_grad()
    def load_state_dict(self, sd):
        if isinstance(sd, dict) and "masters" in sd:
            self.inner.load_state_dict(sd["inner"])
            for (_, m), s in zip(self._pairs, sd["masters"]):
                m.copy_(s.to(m.device))
            for p, m in self._pairs:
                p.copy_(m.to(p.dtype))          # live follows the restored masters
        else:
            # legacy plain-AdamW checkpoint: adopt the (already-loaded) live weights as
            # masters, then restore the fp32 moments if their shapes are compatible
            for p, m in self._pairs:
                m.copy_(p.float())
            try:
                self.inner.load_state_dict(sd)
            except Exception:
                pass                            # moments restart; weights are correct


# --------------------------------------------------------------------------- #
# checkpoint / history IO
# --------------------------------------------------------------------------- #
# Checkpoint layout, one directory per stage, one FILE NAME per configuration:
#
#     checkpoints/<stage>/checkpoint_<model>_<pe>_<stage-label>.pt
#     checkpoints/<stage>/history_<model>_<pe>_<stage-label>.json
#     outputs/plots/<stage>/dynamics_<model>_<pe>_<stage-label>.pdf
#
# where <model> is the size (zetagpt-s / -m / -l), <pe> is how position enters (ssm or rope),
# and <stage-label> names the training stage the artefact came out of. The configuration is in
# the FILENAME, so an ablation run sits beside the run it is being compared with instead of
# overwriting it -- which is the whole point of being able to set pe="rope".
#
# The stage is ALSO the directory, so the label in the name is redundant while a file sits
# where it was written. It stops being redundant the moment a file leaves: checkpoints get
# copied into saved_models/, attached to an issue, dropped in a shared folder, and a bare
# checkpoint_zetagpt-s_ssm.pt is then indistinguishable from the six other stages' output.
# The name is self-describing on its own.
# --------------------------------------------------------------------------- #

# The label each stage puts in its filenames. Deliberately NOT the internal stage key: "rlhf"
# and "dpo" both act on the instruction-following model and read better carrying that, and the
# chain-of-thought stage is named for its algorithm because that is what distinguishes it. A
# stage missing from this table falls back to its own key, so adding a stage cannot break the
# path helpers -- it only gets a plainer name until it is listed here.
STAGE_LABELS = {
    "pretrain": "pretrain",
    "sft": "sft",
    "reward": "reward",
    "rlhf": "instruct-rlhf",
    "dpo": "instruct-dpo",
    "cot": "cot-grpo",
    "distill": "distill",
}


def stage_label(stage):
    """The filename label of a stage key ("rlhf" -> "instruct-rlhf")."""
    return STAGE_LABELS.get(stage, stage)

# The named sizes, keyed by (n_layer, n_head, n_embd). A configuration that is not one of
# these gets a descriptive name rather than being forced into the nearest label -- so a model
# trained before these values changed is reported as, say, zetagpt-6x384 rather than silently
# adopting the label of a size it is not.
#
# EVERY SIZE HAS A HEAD DIMENSION OF 64: 512/8 = 768/12 = 1024/16 = 64. See
# model.zetagpt.HEAD_DIM. Width therefore scales by adding heads, never by widening them.
# Tiny and S differ only in depth, so the key must carry n_layer -- which it does.
MODEL_SIZES = {
    (8, 8, 512): "zetagpt-tiny",
    (16, 8, 512): "zetagpt-s",
    (16, 12, 768): "zetagpt-m",
    (24, 16, 1024): "zetagpt-l",
}


def model_name(cfg):
    """The size label of a model configuration dict (or a ZetaGPT's .cfg)."""
    key = (cfg.get("n_layer"), cfg.get("n_head"), cfg.get("n_embd"))
    return MODEL_SIZES.get(key, f"zetagpt-{cfg.get('n_layer')}x{cfg.get('n_embd')}")


def run_name(cfg):
    """`<model>_<pe>`, the tag that identifies a run in every filename it writes. The stage
    label is appended per artefact by ckpt_path/hist_path, not here, because one run passes
    through several stages and the tag is set once."""
    return f"{model_name(cfg)}_{cfg.get('pe', 'ssm')}"


# Set once per run, by helpers.common.setup(), from the model configuration in force. Left
# empty the old names (last.pt / history.json) are used, so a checkpoint written before this
# convention existed still loads.
_RUN_TAG = ""


def set_run_tag(cfg_or_str):
    """Called once at startup with the model configuration; every artefact path follows."""
    global _RUN_TAG
    _RUN_TAG = cfg_or_str if isinstance(cfg_or_str, str) else run_name(cfg_or_str)
    return _RUN_TAG


def get_run_tag():
    return _RUN_TAG


def stage_dir(ckdir, stage):
    return os.path.join(ckdir, stage)


def artefact_tag(stage):
    """`<model>_<pe>_<stage-label>`, the full identifier in every artefact name. Empty when no
    run tag has been set, which is what selects the pre-convention names below."""
    return f"{_RUN_TAG}_{stage_label(stage)}" if _RUN_TAG else ""


def ckpt_path(ckdir, stage):
    tag = artefact_tag(stage)
    return os.path.join(stage_dir(ckdir, stage),
                        f"checkpoint_{tag}.pt" if tag else "last.pt")


def hist_path(ckdir, stage):
    tag = artefact_tag(stage)
    return os.path.join(stage_dir(ckdir, stage),
                        f"history_{tag}.json" if tag else "history.json")


def _atomic_torch_save(obj, path):
    """torch.save via a temporary file and an atomic rename.

    A DIRECT torch.save to `path` is not atomic: if the write fails part way -- a full disk, an
    exceeded quota, a killed job -- `path` is left truncated and the previous GOOD checkpoint is
    gone with it. That turns a recoverable interruption into a lost run, and it is how

        RuntimeError: basic_ios::clear: iostream error

    presents: the zip writer failed in write_end_of_file, i.e. after gigabytes had already been
    written. os.replace is atomic within a filesystem, so `path` is either the old checkpoint or
    the complete new one, never a partial write. The temporary lives in the SAME directory so the
    rename cannot cross a filesystem boundary.

    The failure is re-raised with the free space attached, because the underlying iostream error
    names neither the file nor the cause."""
    tmp = path + ".tmp"
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except Exception as e:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)                 # never leave a partial file behind
        except OSError:
            pass
        free = shutil.disk_usage(os.path.dirname(path) or ".").free
        raise RuntimeError(
            f"failed writing {path}: {type(e).__name__}: {e}. "
            f"Free space on that filesystem: {free / 2**30:.2f} GiB. "
            f"A checkpoint here is model + optimizer state (~4x the parameter count in fp32 "
            f"under MasterAdamW), and the previous checkpoint is NOT overwritten until the new "
            f"one is complete, so the write needs room for both at once. "
            f"The last good checkpoint at {path} is intact."
        ) from e


def model_cfg_of(model):
    """The constructor arguments of a ZetaGPT (or of the ZetaGPT inside a wrapper such as the
    reward model), or None for a foreign model like the distilled HF student."""
    cfg = getattr(model, "cfg", None)
    if cfg is None:
        cfg = getattr(getattr(model, "base", None), "cfg", None)
    return dict(cfg) if isinstance(cfg, dict) else None


def save_ckpt(ckdir, stage, model, opt, step, total, gens, evald=None, extra=None):
    os.makedirs(stage_dir(ckdir, stage), exist_ok=True)
    _atomic_torch_save({"model": model.state_dict(),
                # THE ARCHITECTURE TRAVELS WITH THE WEIGHTS. Reading it back beats trusting
                # default_config.py, which may have moved on since the run (a different width, a
                # different depth), and a mismatch there is a silent wrong-model bug.
                "model_cfg": model_cfg_of(model),
                "opt": opt.state_dict() if opt is not None else None,
                "step": step, "total": total, "done": step >= total,
                "gens": [g.get_state() for g in gens], "eval": evald,
                # The explicit generators cover data order and corruption, but DROPOUT draws
                # from the global RNG, so a resumed run saw a different dropout stream from an
                # uninterrupted one and the two diverged. Saving the global state makes resume
                # exact. CPU state is always present; the CUDA states only when CUDA is in use.
                "rng": torch.get_rng_state(),
                "rng_cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                "extra": extra},   # method-specific side state, e.g. a mask head
               ckpt_path(ckdir, stage))


def restore_rng(ck):
    """Restore the global RNG streams saved by save_ckpt. Silently does nothing for
    checkpoints written before `rng` was recorded, so old runs still resume."""
    if not ck:
        return
    if ck.get("rng") is not None:
        torch.set_rng_state(ck["rng"].cpu() if hasattr(ck["rng"], "cpu") else ck["rng"])
    cu = ck.get("rng_cuda")
    if cu is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cu)
        except (RuntimeError, ValueError):
            pass          # different device count on resume: keep the CPU stream, drop CUDA's


def load_ckpt(ckdir, stage):
    p = ckpt_path(ckdir, stage)
    return torch.load(p, map_location="cpu") if os.path.isfile(p) else None


def save_hist(ckdir, stage, records):
    """Atomic for the same reason as the checkpoint: history.json is what every figure is drawn
    from, and a truncated one is invalid JSON, so a failed write here loses the whole training
    curve rather than one record."""
    os.makedirs(stage_dir(ckdir, stage), exist_ok=True)
    p = hist_path(ckdir, stage)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(records, f)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def load_hist(ckdir, stage, upto=None):
    """The stage's history. With `upto` set, records at or after that step are DROPPED.

    Resume replays from the checkpoint's step, so any record at or beyond it is about to be
    produced again. Appending to them would leave the file with duplicated and then
    non-monotone steps, and the figure -- which keeps only the last monotone run -- would
    silently restart at the resume point instead of showing the whole training curve. The
    truncation makes that impossible by construction: after it, appending can only extend a
    strictly increasing sequence.

    The history can legitimately extend past the checkpoint, because a checkpoint is written
    every --checkpoint_every_steps while the history is appended every step, and the two are
    saved in that order."""
    p = hist_path(ckdir, stage)
    if not os.path.isfile(p):
        return []
    h = json.load(open(p))
    if upto is None:
        return h
    keep = [r for r in h if r.get("step", 0) < upto]
    if len(keep) != len(h):
        print(f"[resume] history: dropped {len(h) - len(keep)} record(s) at or after step "
              f"{upto}, which this run is about to recompute", flush=True)
    return keep


@torch.no_grad()
def rollout_evidence(policy, ref, tok, pairs, beta, roll_tokens, max_len, seed=7):
    """Z_theta = E_{z~p_theta}[sigmoid(m_theta(z))]: the preference evidence under ROLLOUT
    histories (p_ro=1), estimated on a small probe set. a gamma = 1/Z_hat calibration."""
    c = rollout_curve(policy, ref, tok, pairs, beta, [1.0], roll_tokens, max_len, seed=seed,
                      progress_desc=False)["1.00"]["margins"]
    if not c:
        return float("nan")
    return torch.sigmoid(torch.tensor(c, dtype=torch.float)).mean().item()


# ---------------------------------------------------------------------------------------------
# Rollout substitutes. Rehomed from vm_dpo when the variational method was removed: rh_dpo needs
# them to draw yhat_i ~ pi_theta(.|x,y_<i), and that has nothing to do with the variational
# objective that used to live beside them. Moved rather than deleted, so RH-DPO is untouched.
# ---------------------------------------------------------------------------------------------

# One torch.Generator per (device, seed) pair. _rand seeds a DEVICE generator from one scalar of
# the checkpointed CPU generator, so the draw is reproducible without allocating and copying a
# CPU tensor per call. Module level because the cache must outlive the call: this dict lived
# beside _rand in vm_dpo and was left behind when the function was rehomed, which made every
# _rand call raise NameError.
_DEV_GENS = {}

def _rand(shape, device, gen=None):
    """Uniform(0,1) of `shape` ON `device`, drawn deterministically from `gen`.

    A torch.Generator is bound to one device: a CPU generator cannot seed a CUDA draw.
    The naive bridge, torch.rand(shape, generator=cpu_gen).to(device), fills the tensor
    in host RAM with a single-threaded PRNG and then copies it across PCIe. For the
    vocabulary-shaped draws in token_stats that is B x chunk x V = 8 x 128 x 151936
    = 0.58 GiB of host RNG plus a 0.58 GiB transfer PER CHUNK -- measured at ~3.4 s of
    CPU RNG per training step, against ~10 ms for the same draw on-device. It dominated
    the step.

    So we spend exactly ONE scalar from `gen` to seed a cached device generator and do
    the bulk draw there. The result is still a deterministic function of `gen`'s state,
    which is what the checkpoint saves, so resumes reproduce exactly -- the property the
    generator was threaded through here to buy. When `gen` already lives on `device`
    (CPU runs, ZetaGPT tests) it is used directly and the draw is bit-identical to
    the pre-existing path.
    """
    if gen is None:
        return torch.rand(shape, device=device)
    if gen.device.type == device.type:
        return torch.rand(shape, device=device, generator=gen)
    seed = int(torch.randint(0, 2 ** 62, (1,), generator=gen).item())
    key = str(device)
    dg = _DEV_GENS.get(key)
    if dg is None:
        dg = _DEV_GENS[key] = torch.Generator(device=device)
    dg.manual_seed(seed)
    return torch.rand(shape, device=device, generator=dg)


def _gumbel_noise(shape, device, gen=None):
    u = _rand(shape, device, gen).clamp_(1e-9, 1 - 1e-9)
    return u.log_().neg_().log_().neg_()          # -log(-log u), in place: no extra buffer


@torch.no_grad()
def token_stats(logits, ids, chunk=128, gen=None, temp=1.0):
    """From the pass-1 logits, the three quantities the resampling corruption needs:

      pi_i  = pi_theta(y_i | x, y_<i)   the policy's probability of the REFERENCE token
      sub_i ~ pi_theta(. | x, y_<i)     a substitute token drawn from the policy
      H_i   = H[ pi_theta(. | x, y_<i) ]   the full-vocabulary entropy

    All aligned to INPUT position i (read off the prediction made at i-1).

    Every full-vocabulary tensor is confined to the B x chunk x V sweep, and at most ONE
    of them is live at a time. That matters: for Qwen (V=151936) with B=8, chunk=128, one
    such tensor in fp32 is 0.58 GiB, so each avoided temporary is 0.58 GiB of allocation
    and HBM traffic per chunk. Three devices are used to get there:

      1. lp is accumulated INSIDE the loop. Computing it outside as
         logsumexp(lg.float(), -1) upcast the entire B x (T-1) x V block at once -- 2.3 GiB
         in a single allocation, larger than the whole chunked sweep it was meant to avoid.
      2. The log_softmax is written back OVER sl (sl.sub_(lse)) instead of into a second
         tensor. Note what is NOT done here: H = lse - E_p[s] is algebraically identical
         and needs no log_softmax at all, but it subtracts two O(|logit|) quantities to
         leave an O(1) one. Measured in fp32 at V=151936, that costs up to 15x RELATIVE
         error on the near-zero entropies of a confident policy -- exactly the regime a
         trained LM sits in. The in-place form is bit-identical to the original.
      3. Gumbel-max needs no separate buffer: the noise tensor is the accumulator
         (nz.add_(sl)). Applying it to log_softmax or to raw logits gives the same draw --
         they differ by a constant per row, which cannot move an argmax -- so step 2 may
         run before or after this without changing the sample.

    Gumbel-max rather than torch.multinomial: equivalent in distribution, but it draws from
    the SAME checkpointed generator as every other random choice in the step (see _rand),
    so resumes reproduce. multinomial used the global device RNG and did not.
    """
    B, Tm1, V = logits[:, :-1].shape
    lg = logits[:, :-1]
    tgt = ids[:, 1:]
    subs, ents, lps, lpsubs = [], [], [], []
    for a in range(0, Tm1, chunk):
        sl = lg[:, a:a + chunk].float()                       # the one big tensor, held throughout
        lse = sl.logsumexp(-1)
        lps.append(sl.gather(-1, tgt[:, a:a + chunk].unsqueeze(-1)).squeeze(-1) - lse)
        sl = sl.sub_(lse.unsqueeze(-1))                       # sl IS log_softmax now, no new buffer
        ents.append(sl.exp().mul_(sl).sum(-1).neg_())         # -sum(p * log p); the exp is freed here
        # yhat_i ~ pi_theta(. | x, y_<i) at temperature `temp`. NOT restricted off y_i: the
        # two-point prior wants a genuine draw from the policy, and a draw that happens to equal
        # y_i simply means that position is uncorrupted, which is correct.
        nz = _gumbel_noise(sl.shape, sl.device, gen)
        nz.add_(sl if temp == 1.0 else sl / max(temp, 1e-6))
        s_idx = nz.argmax(-1)
        subs.append(s_idx)
        # log pi^TF(yhat_i): the surprisal of the DRAWN token under the same clean
        # conditional. One gather off `sl`, which is already log_softmax and already
        # resident -- no extra full-vocabulary tensor. Needed for an unbiased one-sample
        # estimate of KL(pi^TF_i || pi^H_i): that divergence is an expectation under
        # pi^TF_i, so it must be evaluated at a DRAW from pi^TF_i. The observed token y_i
        # is a draw from the DATA, not from pi^TF_i, and plugging it in biases the
        # estimate (measured: -2.35 nats/token, against -0.04 when drawn correctly).
        lpsubs.append(sl.gather(-1, s_idx.unsqueeze(-1)).squeeze(-1))
        del nz, sl
    sub = torch.cat(subs, dim=1)
    lp = torch.cat(lps, dim=1)
    lp_sub = torch.cat(lpsubs, dim=1)

    pi = torch.zeros(ids.shape, dtype=lp.dtype, device=lp.device)
    pi[:, 1:] = lp.exp()
    ent = torch.zeros(ids.shape, dtype=lp.dtype, device=lp.device)
    ent[:, 1:] = torch.cat(ents, dim=1)
    sub_ids = ids.clone()
    sub_ids[:, 1:] = sub
    lps_tf = torch.zeros(ids.shape, dtype=lp.dtype, device=lp.device)
    lps_tf[:, 1:] = lp_sub
    # log pi^TF(y_i), UNCLAMPED, in the token frame. The clamp on `pi` below is a probability
    # floor for the places that need a probability strictly inside (0,1); it must not reach the
    # exposure log-gap. That gap is a DIFFERENCE of two log-probabilities, and flooring only the
    # numerator at log(1e-4) = -9.21 while the pass-2 denominator runs free biases it upward by
    # exactly the amount the floor removes -- on precisely the low-probability tokens that carry
    # the exposure signal. Callers that difference log-probs take `lpy_tf`; callers that need a
    # probability take `pi`.
    lpy_tf = torch.zeros(ids.shape, dtype=lp.dtype, device=lp.device)
    lpy_tf[:, 1:] = lp
    return pi.clamp(1e-4, 1 - 1e-4), sub_ids, ent, lps_tf, lpy_tf


# ---------------------------------------------------------------------------------------------
# Exposure probe. Method-agnostic: every method logs eb_probe, and it feeds panel 5 of
# training_<method>.pdf. Rehomed out of vm_dpo with the rest of the shared machinery when the
# variational method was removed -- it was never variational, only co-located.
# ---------------------------------------------------------------------------------------------
def _keep(rmask, dtype):
    """Corruptible positions. Position 0 is excluded unconditionally: it has no preceding
    prediction, so log pi(z_0 | x, z_<0) does not exist. Normally rmask[:,0] = 0 anyway (the
    prompt), but a pair long enough to truncate the prompt away leaves rstart = 0 and column 0
    marked as response."""
    keep = rmask.to(dtype).clone()
    keep[:, 0] = 0
    return keep


def _align(lp):
    """Pass-2 per-position log-probs are (B,T-1), indexed by the PREDICTION site i-1; the mask
    tensors are (B,T), indexed by the token site i. Pad a leading column so both are in the
    token frame. Column 0 is excluded from `keep` (see _keep), so its value is never read."""
    return F.pad(lp, (1, 0))


@torch.no_grad()
def _rollout_history(base, ids, attn, keep, gen=None, roll_temp=1.0, show=True):
    """The history under H = RO: the policy's OWN autoregressive continuation of the prompt.

        z_j ~ pi_theta( . | x, z_<j )      for every corruptible position j

    Note the conditioning: z_<j, the tokens THIS LOOP already wrote, not y_<j. That is the
    whole content of the word "rollout" and it is why the loop cannot be vectorised over j --
    z_j is not measurable with respect to anything available before step j. One full forward
    per response column, the same no-KV-cache convention as helpers._roll_reward.

    Prompt and padding columns (keep = 0) are left at their original ids, so the prompt stays
    clean and only the response region is self-generated. Returns the rewritten id tensor.

    A tqdm bar reports the sweep in real time. leave=False so it erases itself and does not
    accumulate one line per probe in the training log.
    """
    roll = ids.clone()
    cols = (keep.sum(0) > 0).nonzero().flatten().tolist()      # columns any row must generate
    it = cols
    if show and cols:
        try:
            from tqdm import tqdm
            it = tqdm(cols, desc="eb rollout (H=RO)", leave=False, unit="tok")
        except Exception:
            it = cols
    for j in it:
        lg = base(input_ids=roll, attention_mask=attn).logits[:, j - 1].float()   # pi(.|x, z_<j)
        lg = lg.log_softmax(-1)
        nz = _gumbel_noise(lg.shape, lg.device, gen)
        nz.add_(lg if roll_temp == 1.0 else lg / max(roll_temp, 1e-6))
        z = nz.argmax(-1)
        m = keep[:, j] > 0
        roll[:, j] = torch.where(m, z, roll[:, j])
        del nz, lg
    return roll


@torch.no_grad()
def exposure_probe(policy, ids, attn, rmask, gen=None, roll_temp=1.0, show=True,
                   per_example=False):
    """Per-token exposure divergence of the CURRENT POLICY at H = RO, for tracking over training.

        eb = mean_i [ log pi_theta(v_i | x, z_<i) - log pi_theta(v_i | x, y_<i) ],
             v_i ~ pi_theta( . | x, z_<i ),   z_<i = the policy's OWN ROLLOUT prefix

    A one-sample unbiased estimate of Definition 1,
    sum_i KL( pi^H_i || pi^TF_i ) / (#positions): that KL is an expectation under the
    EXPOSED conditional, so v_i must be drawn from pi^H_i, not from pi^TF_i. Evaluating at
    the observed token y_i instead would be a plug-in at a DATA draw, biased either way.

    THE DIRECTION IS NOT FREE. KL is not symmetric and the manuscript fixes the order:
    Definition 1 puts the exposed law FIRST, D_EB = KL(pi^H || pi^TF), because that is the
    only order for which the multiplier the constraint contributes to the implicit reward is
    additive over positions. The rollout pass below builds z and the pass after it evaluates
    pi(.|x,z_<i) explicitly, so the draw from pi^H costs nothing.

    WHY THE SECOND PASS IS A GENERATION AND NOT A SUBSTITUTION. Substituting, at every
    position j, a token drawn from the PASS-1 logits gives z_j ~ pi(.|x, y_<j), each token
    conditioned on the CLEAN prefix. Every such z_j is one step off the reference trajectory
    and no further: the history never compounds, so what is measured is a one-step
    perturbation divergence, not exposure bias. pi^RO is defined by z_j ~ pi(.|x, z_<j) and is
    only reachable by generating autoregressively, which is what _rollout_history does.

    It is a property of theta alone, so it is read WITHOUT reference to how the method
    corrupts during training. That matters: DPO trains on clean histories, so its
    training-time gap is identically 0 and would show a flat line at zero regardless of
    what the policy is actually doing. Fixing the probe at H = RO, the fully exposed member
    of the family, gives a number that moves with theta and can be watched across steps.

    The pass order is chosen so that only ONE B x (T-1) x V logit block is alive at a time:
    the rollout first, then pi^H (which supplies the draw), then pi^TF (scored at that draw).
    Holding pass 1 across the rollout to reuse its logits would keep two such blocks resident,
    2.3 GiB each in fp32 at B=8, T=512, V=151936.

    v_i is drawn at temperature 1 regardless of roll_temp: the estimator scores with an
    untempered log_softmax, so a tempered draw would not be an expectation under the law
    being scored. roll_temp governs the rollout that defines H, not the draw.

    Cost: 2 + (#response columns) no_grad forwards -- the rollout is sequential by
    construction. Called every --eb_every steps on --eb_pairs pairs. Returns nats/token.

    per_example=True additionally returns the PER-SEQUENCE values, which the dual variable of
    eq:eb-dual-ascent needs: its Newton step divides by Var[g], and a scalar mean cannot supply
    that. The variance is taken across sequences, which is the population the constraint
    E[D_EB] <= eps is an expectation over.
    """
    base = policy.base if hasattr(policy, "base") else policy
    keep = _keep(rmask, torch.float32)
    if keep.sum() < 1:
        return (float("nan"), None) if per_example else float("nan")
    # pass 1: build z by autoregressive generation, so z_<i is a genuine rollout prefix
    roll = _rollout_history(base, ids, attn, keep, gen=gen, roll_temp=roll_temp, show=show)
    # pass 2: pi^H_i = pi(. | x, z_<i). DRAW v_i HERE -- Definition 1 is an expectation
    # under the exposed conditional -- and read log pi^H(v_i) off the same block.
    lgt = base(input_ids=roll, attention_mask=attn).logits[:, :-1]
    vs, lph = [], []
    for a in range(0, lgt.shape[1], 128):
        sl = lgt[:, a:a + 128].float().log_softmax(-1)
        nz = _gumbel_noise(sl.shape, sl.device, gen)
        nz.add_(sl)                                   # temperature 1; see docstring
        idx = nz.argmax(-1)
        vs.append(idx)
        lph.append(sl.gather(-1, idx.unsqueeze(-1)).squeeze(-1))
        del nz, sl
    v = torch.cat(vs, dim=1)
    lp_h_v = torch.cat(lph, dim=1)
    del lgt, vs, lph
    # pass 3: pi^TF_i = pi(. | x, y_<i), scored at the SAME v_i
    lgt = base(input_ids=ids, attention_mask=attn).logits[:, :-1]
    outs = []
    for a in range(0, lgt.shape[1], 128):
        sl = lgt[:, a:a + 128].float()
        outs.append(sl.gather(-1, v[:, a:a + 128].unsqueeze(-1)).squeeze(-1) - sl.logsumexp(-1))
        del sl
    lp_tf_v = torch.cat(outs, dim=1)
    del lgt
    gap = _align(lp_h_v - lp_tf_v) * keep             # log pi^H(v) - log pi^TF(v)
    if per_example:
        n = keep.sum(dim=1).clamp(min=1)
        return (gap.sum() / keep.sum().clamp(min=1)).item(), (gap.sum(dim=1) / n)
    return (gap.sum() / keep.sum().clamp(min=1)).item()


# --------------------------------------------------------------------------- #
# Held-out probe, shared by every phase so the curves are comparable
# --------------------------------------------------------------------------- #
@torch.no_grad()
def holdout_probe(policy, ref, enc, pairs, beta, gen=None, roll_temp=1.0, batch=16):
    """The two numbers a preference run is judged on, measured the SAME way for every phase.

        eval_acc  = Pr(margin > 0) on held-out pairs, teacher forced, against `ref`
        eval_deb  = D_EB(policy; x, H=RO), the per-token exposure divergence of Definition 1

    Both come from one function so that different alignment stages sit on one axis: a
    comparison is only meaningful if the instrument is identical, and the stages optimize
    different objectives, so neither one's training loss can serve as that instrument.

    Cheap by construction -- `pairs` is expected to be a small held-out slice, and the caller
    decides the cadence. Returns a dict so the keys land in the history and in the figures under
    the same names."""
    Ms, ebs = [], []
    for i in progress(range(0, len(pairs), batch), desc="[probe] held-out",
                      total=(len(pairs) + batch - 1) // batch):
        chunk = pairs[i:i + batch]
        W = enc.encode(chunk, "chosen")
        L = enc.encode(chunk, "rejected")
        Ms.append(dpo_margin(policy, ref, W, L, beta))
        ebs.append(exposure_probe(policy, W[0], W[1], W[2], gen=gen, roll_temp=roll_temp,
                                  show=False))
    M = torch.cat(Ms)
    return {"eval_acc": (M > 0).float().mean().item(),
            "eval_margin": M.mean().item(),
            "eval_deb": float(sum(ebs) / max(len(ebs), 1))}


# --------------------------------------------------------------------------- #
# PRE-TOKENISED CORPORA, cached on disk.
#
# Tokenising the corpus once per training STEP is pure waste: the same documents are encoded
# thousands of times by the same tokenizer. Each source file is tokenised ONCE and stored
# under a mirror of its path, in a directory named for THE TOKENIZER THAT PRODUCED IT:
#
#     data/download/<corpus>/<path>/<name>.parquet
#         ->  cache/tokens/bpe_<256+merges>_<fp>/data/download/<corpus>/.../<name>.parquet.<sig>.tokens
#
# THE TOKENIZER IS THE DIRECTORY, not a field inside the file, so two vocabularies coexist
# instead of overwriting each other: retrain the BPE, try the new vocabulary, go back, and the
# old tokens are still there. The fingerprint is the first 8 characters of GIT'S BLOB HASH of
# the tokenizer's MERGES, so the directory names the encoding rather than merely its size: a
# vocabulary rebuilt to the same size lands somewhere else, which is the whole point. It names
# the merges rather than the whole bpe.json so that REGISTERING A SPECIAL LEAVES IT ALONE -- a
# chat token added after pretraining must not throw away a corpus that has not changed.
# THE STAGE IS NOT IN THE PATH: tokens are a property of (file,
# tokenizer, packing) and of nothing else, so a file tokenised for pretraining is reused by any
# other stage that reads it rather than tokenised again from scratch.
#
# `<sig>` is a short hash of the packing (max_words, the parquet text column). It is in the
# NAME rather than only in the header so that two settings do not evict each other's entries
# turn by turn; the same signature is also stored inside and checked on read.
#
# Format: one JSON header line, then the ids of every packed document concatenated as
# little-endian uint32. The header's `sig` identifies the tokenizer AND the packing, and
# `src_mtime`/`src_size` the source file, so a rebuilt vocabulary, a changed `max_words` or an
# edited file all invalidate the entry -- a stale cache can never silently train the wrong
# tokens.
# --------------------------------------------------------------------------- #
_FINGERPRINTS = {}


def blob_hash(data):
    """GIT'S object hash of `data`: sha1(b"blob <size>\\0" + data).

    Git's, deliberately, rather than a bare digest of the bytes. It costs one extra header to
    compute and buys a fingerprint anyone can CHECK against the tool already on their machine:

        git hash-object checkpoints/bpe/bpe.json

    prints the full 40-character digest whose first characters name the cache directory. A
    fingerprint you cannot independently reproduce is one you end up trusting; this one can be
    verified in a second, which matters when the question is "did this cache come from the
    tokenizer I am holding"."""
    import hashlib
    h = hashlib.sha1()
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def encoding_blob(tok):
    """The canonical bytes that DEFINE how this tokenizer turns corpus text into ids: its
    ordered merges, and nothing else.

    The PRE-TOKENIZER VERSION is in it too. The merges alone do not determine the ids: the
    same merges under a different chunking rule tokenise the same text differently, so a change
    to _PAT has to invalidate the cache exactly as a retrained vocabulary does.

    Deliberately not the whole bpe.json. The specials are excluded because they do not
    participate in corpus tokenisation at all -- the packer calls encode_ordinary, so a
    document is bytes and merges -- and because the id of eos is 256 + len(merges), which does
    not move when a special is registered either. Registering <|im_start|> after pretraining
    must therefore leave every cached token stream valid, which is the entire point of being
    able to register one."""
    from tokenizer.bpe import PRETOK_VERSION
    return json.dumps({"pretok": PRETOK_VERSION,
                       "merges": [[list(a), list(b)] for a, b in tok.merges]},
                      separators=(",", ":")).encode("utf-8")


def tokenizer_fingerprint(path=None, tok=None, n=8):
    """`n` characters of the git blob hash of the tokenizer's ENCODING.

    Abbreviated the way git abbreviates a commit, and for the same reason: the full digest is
    what makes it unique, the first few characters are what makes it usable in a name. Eight
    hex characters is 4.3 billion values against the handful of tokenizers a project ever has,
    so a collision is not a practical concern; git itself defaults to seven.

    Any change to the merges -- a retrained vocabulary, a different merge order, one merge more
    -- lands the cache somewhere else, because those are the changes that alter what a corpus
    tokenises to. Registering a special does NOT, by construction: see encoding_blob.

    The value is reproducible from the shell, which is why git's construction is used rather
    than a bare digest:

        python -c "import json,hashlib,sys; m=json.load(open('checkpoints/bpe/bpe.json'))['merges']; \
                   b=json.dumps(m,separators=(',',':')).encode(); \
                   print(hashlib.sha1(b'blob %d\\0'%len(b)+b).hexdigest()[:8])"

    Memoised on the tokenizer object, since it is asked for once per cached file and a 50k-merge
    list is not free to serialise tens of thousands of times."""
    if tok is not None and hasattr(tok, "merges"):
        # Cached ON THE OBJECT, not in a dict keyed by id(): CPython reuses an id once an
        # object is collected, so an id-keyed cache can hand a new tokenizer the fingerprint
        # of a dead one -- and the failure would be a corpus silently read from the wrong
        # cache. The merge count is part of the key so an incrementally extended vocabulary
        # recomputes.
        cached = getattr(tok, "_fingerprint_cache", None)
        if not cached or cached[0] != len(tok.merges):
            cached = (len(tok.merges), blob_hash(encoding_blob(tok)))
            try:
                tok._fingerprint_cache = cached
            except AttributeError:                 # a tokenizer that forbids new attributes
                pass
        return cached[1][:n]
    # No tokenizer object: fall back to the file, which at least separates two different ones.
    path = path or config.BPE_PATH
    try:
        st = os.stat(path)
        key = (path, int(st.st_mtime), st.st_size)
        if key not in _FINGERPRINTS:
            with open(path, "rb") as f:
                _FINGERPRINTS[key] = blob_hash(f.read())
        return _FINGERPRINTS[key][:n]
    except OSError:
        return "nofile00"[:n].ljust(n, "0")


def tokenizer_tag(tok, path=None):
    """The cache directory name: bpe_<encoding size>_<fingerprint>.

    The size counted is 256 + #merges -- the ids a corpus can actually contain -- NOT len(tok),
    which includes the specials. That distinction is the difference between a cache that
    survives registering a chat token and one that does not: len(tok) rises by one the moment
    <|im_start|> is registered, and naming the directory after it would throw away a corpus
    that had not changed by a single token.

    `bpe` says what kind of tokenizer it is, the size is the number a human recognises the
    directory by, and the fingerprint is what makes the name correct rather than merely
    descriptive -- two vocabularies of the same size are different tokenizers."""
    n_base = 256 + len(tok.merges) if hasattr(tok, "merges") else len(tok)
    return f"bpe_{n_base}_{tokenizer_fingerprint(path, tok)}"


def token_cache_root(tok, path=None):
    """cache/tokens/<tokenizer>/. The `tokens` level names WHAT is cached, so anything else
    the pipeline caches later gets its own sibling rather than being mixed in with a corpus."""
    return os.path.join(config.CACHE_DIR, "tokens", tokenizer_tag(tok, path))


def _tok_signature(tok, max_words, text_column=""):
    """Identifies the tokenizer and the packing; any change invalidates every entry. The
    parquet text column is part of it: reading a different column is a different corpus, and a
    cache that survived that change would train on tokens nobody asked for."""
    n_merges = len(getattr(tok, "merges", []) or [])
    return (f"v1|vocab={len(tok)}|merges={n_merges}|max_words={int(max_words)}"
            f"|col={text_column}")


def _sig_tag(sig):
    """Eight hex characters of the signature, for the file name."""
    import hashlib
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()[:8]


def _mirror(src):
    """A source path as a path RELATIVE to the cache root, mirroring it from the project root.

    The full path is mirrored rather than a path relative to the corpus directory, because the
    stage no longer separates entries: two corpora each holding part_0000/batch_0.json would
    otherwise land on the same cache file and quietly serve each other's tokens. A source from
    outside the project keeps its absolute path under _external/ for the same reason."""
    src = os.path.abspath(src)
    rel = os.path.relpath(src, config.ROOT)
    if rel != os.pardir and not rel.startswith(os.pardir + os.sep):
        return rel
    drive, tail = os.path.splitdrive(src)
    return os.path.join("_external", drive.replace(":", ""), tail.lstrip(os.sep))


def _tok_path(tok, src, sig):
    return os.path.join(token_cache_root(tok), f"{_mirror(src)}.{_sig_tag(sig)}.tokens")


def _parquet_docs(path, column):
    """The rows of `column` as DOCUMENTS -- one list of lines per row.

    Row boundaries are kept rather than flattened into one stream of lines. A parquet corpus
    holds one document per row, exactly as a directory of .txt holds one per file, and the
    packer must treat the two the same way: a corpus converted from text to parquet has to
    produce the SAME training documents, or the conversion silently changed what the model
    sees. Flattening would let a packed document begin in one article and end in another,
    teaching the model transitions that exist nowhere in the corpus.

    A wrong column name is reported WITH THE FILE'S ACTUAL COLUMNS. The alternative -- a bare
    KeyError -- arrives after the scan has already opened some fraction of a 78-file corpus and
    says nothing about what to put right."""
    try:
        import pyarrow.parquet as pq
    except Exception as e:                                     # noqa: BLE001
        raise SystemExit(
            f"[corpus] {os.path.basename(path)} is parquet and pyarrow is not importable "
            f"({e}).\n         pip install pyarrow") from e
    pf = pq.ParquetFile(path)
    names = list(pf.schema_arrow.names)
    if column not in names:
        raise SystemExit(
            f"[corpus] {os.path.basename(path)} has no column {column!r}.\n"
            f"         Columns present: {', '.join(names)}\n"
            f"         Set PRETRAIN[\"text_column\"] in default_config.py to the right one.")
    # A SMALL BATCH ON PURPOSE. pyarrow's default is 65,536 rows, and a row here is a whole
    # article: that default materialises hundreds of megabytes of Python strings at a time,
    # which is precisely the cost this module exists to avoid paying.
    for batch in pf.iter_batches(batch_size=64, columns=[column]):
        for value in batch.column(0).to_pylist():
            if value:
                yield str(value).splitlines()


def _source_documents(path, text_column):
    """One corpus file -> its documents, each a list of lines.

    A .parquet holds one document per row; every other format is one document per FILE. That
    is the whole difference between the two, and stating it in one place is what lets the
    packer be written once."""
    if os.path.splitext(path)[1].lower() == ".parquet":
        return _parquet_docs(path, text_column)
    try:
        return [open(path, encoding="utf-8", errors="replace").read().splitlines()]
    except Exception:                                          # noqa: BLE001
        return []


def _pack(path, max_words, text_column="text"):
    """Source file -> ~max_words-word documents, NEVER CROSSING A SOURCE DOCUMENT.

    A GENERATOR, not a list. A single parquet shard is ~100 MB compressed and several hundred
    megabytes of Python strings once expanded, and returning them all at once would put the
    peak cost of tokenising a corpus at one shard rather than at one document -- which is the
    same mistake, one order of magnitude down, as holding the corpus in RAM.

    The boundary is the file for text and the row for parquet, so the same corpus packs
    identically in either format. The tail of each document is kept even when it is short: a
    remainder discarded for being under the budget would drop the end of every article in the
    corpus, which is a systematic loss rather than a rounding one."""
    for lines in _source_documents(path, text_column):
        buf, wc = [], 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            wc += len(line.split())
            if wc >= max_words:
                yield " ".join(buf); buf, wc = [], 0
        if buf:
            yield " ".join(buf)


# _write / _read are gone with the single-file cache they served. Both caches now APPEND:
# helpers/token_store.py for the corpus, attach_pair_ids below for the preference pairs.
# Nothing in this project writes a data file by building it in memory first.


def corpus_bytes(files, text_column="text"):
    """Total UNCOMPRESSED bytes of `files` -- the denominator a progress bar needs.

    A parquet file's size on disk is its compressed size, which for prose is two or three
    times smaller than the text it holds; using it would drive the bar past 100%. Parquet
    records the uncompressed size of every row group in its FOOTER, so the true figure costs
    a metadata read and no data. Anything else is its size on disk, which is already the
    number of bytes that will be read.

    Best-effort: a file that cannot be measured contributes its on-disk size, and a total that
    is slightly wrong is a bar that is slightly wrong, never a failure."""
    total = 0
    for fp in files:
        try:
            if os.path.splitext(fp)[1].lower() == ".parquet":
                import pyarrow.parquet as pq
                md = pq.ParquetFile(fp).metadata
                total += sum(md.row_group(i).total_byte_size
                             for i in range(md.num_row_groups))
            else:
                total += os.path.getsize(fp)
        except Exception:                                      # noqa: BLE001
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _split_token(name):
    """The leading name token of a file: `validation-00000-of-00001.parquet` -> "validation",
    `test_0003.txt` -> "test", `doc.0.parquet` -> "doc"."""
    stem = os.path.splitext(name)[0]
    for sep in ("-", "_", "."):
        stem = stem.split(sep)[0]
    return stem.lower()

def corpus_files(root, exclude_dirs=(), extensions=None):
    """Every corpus file under `root`, recursively: any file whose extension is in
    `extensions` (config.CORPUS_EXTENSIONS by default -- prose, markdown AND source code),
    skipping anything excluded by `exclude_dirs`. Sorted, so a run is reproducible.

    An entry in `exclude_dirs` excludes a DIRECTORY of that name and also any FILE whose
    leading name token matches it. Both, because a corpus keeps its held-out split in whichever
    of the two shapes its format favours: thirty thousand text files are laid out as valid/ and
    test/ subdirectories, while the same corpus as parquet is a handful of files named
    validation-00000-of-00001.parquet in one flat directory -- the layout the Hugging Face Hub
    infers splits from. A directories-only rule silently pretrains on the test split the moment
    a corpus is converted, and a model evaluated on text it was trained on reports a perplexity
    that means nothing."""
    exts = {("." + e.lower().lstrip(".")) for e in (extensions or config.CORPUS_EXTENSIONS)}
    excl = {d.lower() for d in (exclude_dirs or ())}
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in excl]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts and _split_token(fn) not in excl:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def corpus_signature(tok, root, files, max_words, text_column):
    """Identifies a whole corpus AND the tokenizer and packing applied to it.

    The manifest -- every source file's path, size and modification time -- is HASHED rather
    than stored, because a corpus of thirty thousand files would otherwise put a megabyte of
    file listing in the header of every stream. Hashing keeps the property that matters: any
    file added, removed, edited or renamed changes the signature, so a stream built from a
    corpus that has since changed is never mistaken for a current one."""
    import hashlib
    h = hashlib.sha256()
    h.update(_tok_signature(tok, max_words, text_column).encode("utf-8"))
    h.update(os.path.abspath(root).encode("utf-8"))
    for fp in files:
        try:
            st = os.stat(fp)
            h.update(f"{os.path.relpath(fp, root)}|{st.st_size}|{int(st.st_mtime)}\0"
                     .encode("utf-8"))
        except OSError:
            h.update(f"{fp}|missing\0".encode("utf-8"))
    return f"corpus|v1|files={len(files)}|{h.hexdigest()[:32]}"


def corpus_stream_path(tok, root, sig):
    """The STEM every shard of a corpus shares:

        cache/tokens/<tokenizer>/<mirror of the corpus dir>/<name>_<sig>_00000.tokens
                                                            <name>_<sig>_index.json

    Shards rather than one file because a corpus can be 10 TB, and one file that size cannot
    be resumed after an interruption, cannot be copied incrementally, and loses everything to
    one bad byte. They sit in the corpus's own directory, so every file there ends in .tokens;
    the signature is in each NAME rather than in a directory above them, which is what lets
    two packings of one corpus coexist without a level of nesting that carries no extension."""
    mirror = _mirror(root)
    name = os.path.basename(os.path.normpath(mirror)) or "corpus"
    return os.path.join(token_cache_root(tok), mirror, f"{name}_{_sig_tag(sig)}")


def build_token_stream(root, tok, max_words=200, exclude_dirs=(), log=print,
                       extensions=None, text_column="text", stage="tokenize", force=False,
                       resume=True):
    """Tokenise a corpus into one memory-mapped stream; return (stream, n_files).

    Reused when a current stream already exists, which is what makes this cheap to re-run and
    what lets the training stages simply OPEN the corpus rather than build it. The tokenising
    itself is a generator, so at no point does the whole corpus exist as Python objects: one
    source file is packed, tokenised and appended, and then released."""
    from . import token_store
    files = corpus_files(root, exclude_dirs, extensions)
    if not files:
        return None, 0
    sig = corpus_signature(tok, root, files, max_words, text_column)
    path = corpus_stream_path(tok, root, sig)
    if not force:
        st = token_store.open_if_current(path, sig)
        if st is not None:
            log(f"[{stage}] token stream: {st.describe()}")
            log(f"[{stage}]               {path}")
            return st, len(files)

    # A tokenizer without encode_ordinary (the gpt2 student's, say) falls back to __call__;
    # the distinction only exists for tokenizers that HAVE registered specials to protect.
    ordinary = (tok.encode_ordinary if hasattr(tok, "encode_ordinary")
                else lambda t: tok(t, add_special_tokens=False)["input_ids"])

    def documents(cursor=None, skip_docs=0):
        """Every packed document of the corpus, tokenised, one at a time, from `cursor` on.

        Counted in DOCUMENTS rather than files: a corpus is three parquet shards now, not
        thirty thousand text files, so a per-file bar sits still while half a billion tokens
        are encoded -- indistinguishable from a hang. The file position goes in the postfix.

        RESUMING SKIPS WORK, IT DOES NOT REDO IT. The cursor names a file and a document
        within it, so files already consumed are never opened and documents already stored
        are never handed to the tokenizer. Encoding is the expensive half by a wide margin --
        a corpus reads at hundreds of MB/s and encodes at a fraction of that -- so an
        interrupted build that re-encoded what it had already written would spend longer
        catching up than it spent getting there. Without a cursor (a manifest written before
        this existed) the corpus is re-read, which is I/O, but `skip_docs` still keeps those
        documents away from the encoder.

        A PAIR IS YIELDED FOR EVERY PACKED DOCUMENT, even one that encodes to nothing, so the
        position advances in step with what was read rather than with what happened to be
        written. The store drops the empty ones itself.

        encode_ordinary, NOT encode: a pretraining corpus is scraped text, and a document that
        merely MENTIONS <|endoftext|> -- any page discussing GPT-2 does -- would otherwise
        contribute a real end-of-text in its middle. In a packed stream that is worse than a
        stray token: it forges a document boundary, and contiguous sampling then trains across
        a join the corpus does not contain. The stream inserts its own separators; the text
        does not get a vote. Curated data (the SFT and preference sets, where a registered
        token is deliberately written) still goes through encode.

        The leading space matches Encoder's convention for a response, so a document read from
        the stream and the same document encoded on the fly are the same ids.
        """
        first = int((cursor or {}).get("file", 0))
        in_file = int((cursor or {}).get("doc_in_file", 0))
        n_docs = int((cursor or {}).get("n_read", 0))
        left = 0 if cursor else max(int(skip_docs), 0)     # count fallback, no cursor
        done = corpus_bytes(files[:first], text_column) if first else 0
        with bar(f"[{stage}] tokenizing corpus", unit="B", unit_scale=True,
                 total=corpus_bytes(files, text_column), initial=done) as b:
            for i in range(first, len(files)):
                # documents of THIS file already stored: `in_file` of them on the first file
                # after a seek, none thereafter
                ahead, in_file = in_file, 0
                for j, d in enumerate(_pack(files[i], max_words, text_column)):
                    b.update(len(d.encode("utf-8", "ignore")))
                    if j < ahead:
                        continue                           # stored; not read past, not encoded
                    if left:
                        left -= 1; n_docs += 1
                        continue                           # stored; re-read but not encoded
                    n_docs += 1
                    if n_docs % 200 == 0:
                        b.set_postfix_str(f"file {i + 1}/{len(files)}, {n_docs:,} docs")
                    yield ordinary(" " + d), {"file": i, "doc_in_file": j + 1,
                                              "n_read": n_docs}
            b.set_postfix_str(f"file {len(files)}/{len(files)}, {n_docs:,} docs")

    log(f"[{stage}] tokenizing {len(files):,} files -> {path}")
    token_store.build(path, sig, documents, tok.eos_token_id, len(tok), log=log,
                      shard_bytes=int(getattr(config, "TOKENS", {}).get("shard_mb", 100))
                      * 1024 * 1024, resume=resume and not force)
    return token_store.TokenStream(path), len(files)


def load_token_corpus(stage, root, tok, max_words=200, exclude_dirs=(), log=print,
                      extensions=None, text_column="text"):
    """The corpus as a memory-mapped TokenStream: (stream, n_files, n_tokens).

    A stage that finds no stream builds one, so nothing breaks if the explicit tokenisation
    stage was skipped; running that stage first simply means training starts immediately."""
    st, n_files = build_token_stream(root, tok, max_words, exclude_dirs, log, extensions,
                                     text_column, stage=stage)
    if st is None:
        return None, 0, 0
    return st, n_files, st.n_tokens


# --------------------------------------------------------------------------- #
# preference pairs: the same idea for the preference batches
# --------------------------------------------------------------------------- #
def _pair_stem(tok, src, split, sig):
    """<mirror of the source>.<split>.<sig> -- the stem the pair cache's two files share."""
    return _tok_path(tok, src + (f".{split}" if split else ""), sig)[:-len(".tokens")]


def _pair_read(stem, want_pairs):
    """(records, valid_bytes) already on disk: id lists in the order they were written.

    Each record is LENGTH-PREFIXED -- one uint32 count, then that many uint32 ids -- so the
    file is read by walking forward, and a partial write at the end is simply a record that
    does not complete. No separator is used and none would do: a preference record CAN contain
    the characters of a special token, and a scheme that split on EOS would mis-cut exactly
    those pairs with nothing to say so."""
    import array
    try:
        with open(f"{stem}_index.json", encoding="utf-8") as f:
            head = json.load(f)
        n_rec, valid = int(head["n_records"]), int(head["n_bytes"])
    except (OSError, ValueError, KeyError):
        return [], 0
    try:
        with open(f"{stem}.tokens", "rb") as f:
            raw = array.array("I")
            raw.frombytes(f.read(valid))
    except OSError:
        return [], 0
    out, i = [], 0
    while len(out) < n_rec and i < len(raw):
        n = raw[i]; i += 1
        if i + n > len(raw):
            break
        out.append(raw[i:i + n].tolist()); i += n
    return out[:3 * want_pairs], valid


def _pair_manifest(stem, sig, st, n_records, n_bytes, complete):
    tmp = f"{stem}_index.json.part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"magic": "zetagpt-pairs", "version": 1, "sig": sig,
                   "src_mtime": int(st.st_mtime), "src_size": st.st_size,
                   "n_records": n_records, "n_bytes": n_bytes,
                   "complete": bool(complete)}, f)
    os.replace(tmp, f"{stem}_index.json")


def attach_pair_ids(pairs, src, tok, stage="instruct", split="", log=print, resume=True,
                    flush_every=2000):
    """Attach `prompt_ids` / `chosen_ids` / `rejected_ids` to every preference pair, IN PLACE,
    caching them so the reward and DPO trainers never tokenise inside their step loops.

    WRITTEN AS IT GOES, never held. Each pair's three sequences are appended the moment they
    are tokenised, and the manifest catches up every few thousand pairs, so a run killed part
    way leaves every pair it tokenised on disk and RESUMES from there. Building the whole cache
    in memory and writing once at the end -- which is what this did -- means an interrupted run
    leaves nothing at all, and the peak cost is the whole cache rather than one pair.

    The split and the pair count are part of the signature, so the train half can never be
    served the validation half's ids, and a set that grew invalidates rather than silently
    reusing the prefix it happens to agree with."""
    import array
    if not pairs:
        return 0
    sig = f"{_tok_signature(tok, 0)}|pairs={len(pairs)}|split={split}"
    st = os.stat(src)
    stem = _pair_stem(tok, src, split, sig)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    path = f"{stem}.tokens"

    have, valid = [], 0
    if resume:
        try:
            with open(f"{stem}_index.json", encoding="utf-8") as f:
                head = json.load(f)
            if (head.get("sig") == sig and head.get("src_size") == st.st_size
                    and head.get("src_mtime") == int(st.st_mtime)):
                have, valid = _pair_read(stem, len(pairs))
        except (OSError, ValueError):
            have, valid = [], 0
    if not have:
        for f in (path, f"{stem}_index.json"):
            if os.path.exists(f):
                os.remove(f)
        valid = 0

    done = len(have) // 3
    for i in range(done):
        pairs[i]["prompt_ids"] = have[3 * i]
        pairs[i]["chosen_ids"] = have[3 * i + 1]
        pairs[i]["rejected_ids"] = have[3 * i + 2]
    n_tokens = sum(len(r) for r in have)
    if done >= len(pairs):
        log(f"[{stage}] pair token cache: reused {done:,} {split or 'pairs'} -> {path}")
        return n_tokens
    if done:
        log(f"[{stage}] pair token cache: {done:,} {split or 'pairs'} reused, "
            f"{len(pairs) - done:,} still to tokenise")

    # bytes written after the manifest's mark belong to a pair it does not count
    if os.path.exists(path) and os.path.getsize(path) != valid:
        with open(path, "r+b") as f:
            f.truncate(valid)

    eos = tok.eos_token_id
    n_bytes, n_rec = valid, 3 * done
    with open(path, "ab") as fh:
        for i in progress(range(done, len(pairs)),
                          desc=f"[{stage}] tokenizing {split or 'pairs'}",
                          total=len(pairs), initial=done):
            p = pairs[i]
            p["prompt_ids"] = tok(p["prompt"], add_special_tokens=False)["input_ids"]
            p["chosen_ids"] = tok(" " + p["chosen"], add_special_tokens=False)["input_ids"] + [eos]
            p["rejected_ids"] = tok(" " + p["rejected"], add_special_tokens=False)["input_ids"] + [eos]
            for seq in (p["prompt_ids"], p["chosen_ids"], p["rejected_ids"]):
                array.array("I", [len(seq)] + list(seq)).tofile(fh)
                n_bytes += 4 * (len(seq) + 1)
                n_tokens += len(seq)
                n_rec += 1
            if (i + 1 - done) % flush_every == 0:
                fh.flush()
                _pair_manifest(stem, sig, st, n_rec, n_bytes, complete=False)
        fh.flush()
        os.fsync(fh.fileno())
    _pair_manifest(stem, sig, st, n_rec, n_bytes, complete=True)
    log(f"[{stage}] pair token cache: {len(pairs):,} {split or 'pairs'} -> {path}")
    return n_tokens
