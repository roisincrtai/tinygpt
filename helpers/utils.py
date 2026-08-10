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
# beside a mirror of the data tree,
#
#     data/<stage>/<path>/<name>.txt  ->  cache/<stage>/tokens/<path>/<name>.txt.tok
#
# so the cache has the same structure as the data directory and each entry is the source file
# name with `.tok` appended. Format: one JSON header line, then the ids of every packed
# document concatenated as little-endian uint32. The header's `sig` identifies the tokenizer
# AND the packing, and `src_mtime`/`src_size` the source file, so a rebuilt vocabulary, a
# changed `max_words` or an edited file all invalidate the entry -- a stale cache can never
# silently train the wrong tokens.
# --------------------------------------------------------------------------- #
def token_cache_root(stage):
    return os.path.join(config.CACHE_DIR, stage, "tokens")


def _tok_signature(tok, max_words, text_column=""):
    """Identifies the tokenizer and the packing; any change invalidates every entry. The
    parquet text column is part of it: reading a different column is a different corpus, and a
    cache that survived that change would train on tokens nobody asked for."""
    n_merges = len(getattr(tok, "merges", []) or [])
    return (f"v1|vocab={len(tok)}|merges={n_merges}|max_words={int(max_words)}"
            f"|col={text_column}")


def _tok_path(stage, root, src):
    return os.path.join(token_cache_root(stage), os.path.relpath(src, root) + ".tok")


def _parquet_lines(path, column):
    """Every line of every row of `column`, so a parquet corpus flows through the same packer
    as a text one.

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
    for batch in pf.iter_batches(columns=[column]):
        for value in batch.column(0).to_pylist():
            if value:
                yield from str(value).splitlines()


def _source_lines(path, text_column):
    """The lines of one corpus file, whatever its format."""
    if os.path.splitext(path)[1].lower() == ".parquet":
        return _parquet_lines(path, text_column)
    try:
        return open(path, encoding="utf-8", errors="replace").read().splitlines()
    except Exception:                                          # noqa: BLE001
        return []


def _pack(path, max_words, text_column="text"):
    """Source file -> list of ~max_words-word documents (never crossing the file)."""
    docs, buf, wc = [], [], 0
    for line in _source_lines(path, text_column):
        line = line.strip()
        if not line:
            continue
        buf.append(line)
        wc += len(line.split())
        if wc >= max_words:
            docs.append(" ".join(buf)); buf, wc = [], 0
    if buf:
        docs.append(" ".join(buf))
    return docs


def _write(path, sig, st, docs_ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head = {"sig": sig, "src_mtime": int(st.st_mtime), "src_size": st.st_size,
            "lens": [len(d) for d in docs_ids]}
    flat = array.array("I", [i for d in docs_ids for i in d])
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:                       # atomic: never leave a partial cache
        f.write((json.dumps(head) + "\n").encode("utf-8"))
        flat.tofile(f)
    os.replace(tmp, path)


def _read(path, sig, st):
    """Cached ids, or None when the entry is missing, stale or unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as f:
            head = json.loads(f.readline().decode("utf-8"))
            if (head.get("sig") != sig or head.get("src_size") != st.st_size
                    or head.get("src_mtime") != int(st.st_mtime)):
                return None
            flat = array.array("I")
            flat.frombytes(f.read())
    except Exception:                                          # noqa: BLE001
        return None
    lens = head.get("lens") or []
    if sum(lens) != len(flat):
        return None                                            # truncated file
    out, off = [], 0
    for n in lens:
        out.append(flat[off:off + n].tolist()); off += n
    return out


def corpus_files(root, exclude_dirs=(), extensions=None):
    """Every corpus file under `root`, recursively: any file whose extension is in
    `extensions` (config.CORPUS_EXTENSIONS by default -- prose, markdown AND source code),
    skipping anything under a directory named in `exclude_dirs`. Sorted, so a run is
    reproducible."""
    exts = {("." + e.lower().lstrip(".")) for e in (extensions or config.CORPUS_EXTENSIONS)}
    excl = set(exclude_dirs or ())
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in excl]
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def load_token_corpus(stage, root, tok, max_words=200, exclude_dirs=(), log=print,
                      extensions=None, text_column="text"):
    """Pre-tokenised documents for `stage` from every *.txt under `root`.

    Returns (docs, n_files, n_tokens). Files whose cache entry is present and current are
    read from cache/<stage>/tokens/; the rest are tokenised now and written there, so the
    first run pays the cost once and every later run starts immediately."""
    files = corpus_files(root, exclude_dirs, extensions)
    if not files:
        return [], 0, 0

    sig = _tok_signature(tok, max_words, text_column)
    docs, n_tokens, n_hit, n_built = [], 0, 0, 0
    for fp in progress(files, desc=f"[{stage}] tokenizing corpus", total=len(files)):
        st = os.stat(fp)
        cp = _tok_path(stage, root, fp)
        ids = _read(cp, sig, st)
        if ids is None:
            # the leading space matches Encoder's convention for a response, so cached and
            # freshly tokenised documents are byte-for-byte the same ids
            ids = [tok(" " + d, add_special_tokens=False)["input_ids"] for d in
                   _pack(fp, max_words, text_column)]
            _write(cp, sig, st, ids)
            n_built += 1
        else:
            n_hit += 1
        for d in ids:
            if d:
                docs.append({"prompt": "", "ids": d, "rejected": ""})
                n_tokens += len(d)
    log(f"[{stage}] token cache: {n_hit:,} files reused, {n_built:,} tokenised now "
        f"-> {token_cache_root(stage)}")
    return docs, len(files), n_tokens


# --------------------------------------------------------------------------- #
# preference pairs: the same idea for the preference batches
# --------------------------------------------------------------------------- #
def attach_pair_ids(pairs, src, tok, stage="instruct", split="", log=print):
    """Attach `prompt_ids` / `chosen_ids` / `rejected_ids` to every preference pair, IN
    PLACE, reading them from cache/<stage>/tokens/<src>.<split>.tok when it is current and
    writing it when it is not. The reward and DPO trainers then never tokenise inside their
    step loops. Returns the number of tokens cached."""
    if not pairs:
        return 0
    sig = f"{_tok_signature(tok, 0)}|pairs={len(pairs)}|split={split}"
    st = os.stat(src)
    name = os.path.basename(src) + (f".{split}" if split else "") + ".tok"
    cp = os.path.join(token_cache_root(stage), name)
    flat = _read(cp, sig, st)                     # one "document" per field, in pair order
    if flat is not None and len(flat) == 3 * len(pairs):
        for i, p in enumerate(pairs):
            p["prompt_ids"] = flat[3 * i]
            p["chosen_ids"] = flat[3 * i + 1]
            p["rejected_ids"] = flat[3 * i + 2]
        log(f"[{stage}] pair token cache: reused {len(pairs):,} {split or 'pairs'} "
            f"-> {cp}")
        return sum(len(d) for d in flat)
    seqs = []
    eos = tok.eos_token_id
    for p in progress(pairs, desc=f"[{stage}] tokenizing {split or 'pairs'}",
                      total=len(pairs)):
        p["prompt_ids"] = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        p["chosen_ids"] = tok(" " + p["chosen"], add_special_tokens=False)["input_ids"] + [eos]
        p["rejected_ids"] = tok(" " + p["rejected"], add_special_tokens=False)["input_ids"] + [eos]
        seqs += [p["prompt_ids"], p["chosen_ids"], p["rejected_ids"]]
    _write(cp, sig, st, seqs)
    log(f"[{stage}] pair token cache: tokenised {len(pairs):,} {split or 'pairs'} -> {cp}")
    return sum(len(s) for s in seqs)
