"""
distill.py -- stage 9's logic: distilling the aligned ZetaGPT into gpt2-small.

    teacher: OUR ALIGNED TINYGPT -- the RLHF-ed model by default
             (config.DISTILL["teacher_stage"] = "rlhf"; set "dpo" for the DPO-ed model)
    student: gpt2-small (HF "gpt2"), fine-tuned from its pretrained weights

Teacher and student use DIFFERENT tokenizers (our byte-BPE vs GPT-2's BPE), so logits cannot
be matched and the distillation is SEQUENCE-LEVEL (Kim & Rush 2016): the teacher generates a
response to each prompt from the prompt bank, and the student maximises the likelihood of that
text under its own vocabulary, subject to a KL constraint to a frozen copy of its own
pretrained initialisation.

    load_prompts()          -> (prompts, n_files) from the prompt bank
    build_student(device)   -> (gpt2-small, its tokenizer)
    run(...)                -> the distilled student

stage9_distill.sh is only the command-line wrapper.
"""
import glob
import os

import torch
import torch.nn.functional as F

import default_config as config
import helpers
from instruct_rlhf import ppo
from chat import generate as sample_generate, decode as tok_decode
from helpers import (progress, save_ckpt, load_ckpt, save_hist, load_hist, MasterAdamW,
                     restore_rng, CosineLR, ckpt_path)

NAME = "Distill"
STAGE = "distill"


def load_prompts(distill_dir=None):
    """The prompts the teacher is asked to answer, and the file count.

    The SAME bank stage 6 rolls out on, for the same reason the reward model is trained on it:
    a student is only worth distilling on the distribution the teacher was actually aligned
    over. Prompts drawn from somewhere else would measure the teacher off-distribution and
    transfer that to the student as if it were the teacher's behaviour."""
    from helpers import pref_dataset as pdset
    return pdset.load_instruction_prompts(distill_dir or config.DISTILL_DIR)


def build_student(device):
    """gpt2-small + its tokenizer (downloaded into saved_models/ on first use)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = config.DISTILL["student"]
    s_tok = AutoTokenizer.from_pretrained(name, cache_dir=config.MODEL_DIR)
    if s_tok.pad_token is None:
        s_tok.pad_token = s_tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(name, cache_dir=config.MODEL_DIR).to(device)
    return student, s_tok


def _encode_student(s_tok, pairs, device, max_len):
    """(prompt, response) text -> right-padded (ids, attn, response-mask) under the STUDENT
    tokenizer, left-truncated to max_len -- the same convention as pref_dataset.Encoder."""
    seqs, masks = [], []
    for prompt, resp in pairs:
        pid = s_tok(prompt, add_special_tokens=False)["input_ids"]
        rid = s_tok(" " + resp, add_special_tokens=False)["input_ids"] + [s_tok.eos_token_id]
        full, rstart = pid + rid, len(pid)
        if len(full) > max_len:
            cut = len(full) - max_len
            full = full[cut:]
            rstart = max(0, rstart - cut)
        seqs.append(full)
        masks.append([0] * rstart + [1] * (len(full) - rstart))
    T = max(len(s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, T), s_tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros(B, T, dtype=torch.long)
    rmask = torch.zeros(B, T, dtype=torch.long)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        ids[i, :len(s)] = torch.tensor(s)
        attn[i, :len(s)] = 1
        rmask[i, :len(m)] = torch.tensor(m)
    return ids.to(device), attn.to(device), rmask.to(device)


@torch.no_grad()
def _teacher_texts(teacher, t_tok, prompts, device, cfg, g, max_len):
    """One sampled teacher response per prompt (batched rollout, teacher tokenizer)."""
    ids, attn, resp_mask = ppo.rollout(teacher, t_tok, [{"prompt": p} for p in prompts],
                                       device, cfg["max_new_tokens"],
                                       cfg["gen_temperature"], g, max_len)
    out = []
    for i, p in enumerate(progress(prompts, desc="[distill] decoding teacher text",
                                   total=len(prompts))):
        rids = ids[i][resp_mask[i].bool()].tolist()
        eos = getattr(t_tok, "eos_token_id", None)
        if eos is not None and rids and rids[-1] == eos:
            rids = rids[:-1]
        out.append((p, t_tok.decode(rids).strip()))
    return out


def train(student, s_tok, teacher, t_tok, prompts, ckdir, args, log, monitor, preview=None,
          ref=None):
    """Sequence-level KD under a KL constraint: the teacher generates, the student maximises
    the likelihood of that text, and a per-token KL to `ref` (a frozen copy of the student's
    INITIAL weights) holds it near its pretrained prior:

        L = CE(student, teacher text) + kl_coef * KL( p_student || p_ref )

    The KL is the exact full-vocabulary divergence at every response position -- available
    here, unlike a teacher/student KL, because the reference shares the student's tokenizer.
    Returns the student in eval mode. Same resume machinery as every stage."""
    cfg = config.DISTILL
    steps, lr = args.distill_steps, args.distill_lr
    kl_coef = cfg.get("kl_coef", 0.0)
    device = next(student.parameters()).device
    ck = load_ckpt(ckdir, STAGE) if args.resume else None
    extend = bool(ck and ck.get("done") and ck.get("total") != steps)
    if ck and ck.get("done") and not extend:
        student.load_state_dict(ck["model"]); student.eval()
        log(f"distill done -> {ck.get('eval')}")
        monitor(STAGE, load_hist(ckdir, STAGE))
        return student
    if extend:
        log(f"distill: budget {ck.get('total')} -> {steps}, extending from step {ck['step']}")

    opt = MasterAdamW(student.parameters(), lr=lr)
    sched = CosineLR(opt, lr, steps, args.lr_min_factor, args.lr_schedule)
    log(f"distill: lr={sched.describe()} steps={steps} batch={args.batch} "
        f"kl_coef={kl_coef:g} (sequence-level KD on {len(prompts):,} prompts) "
        f"-> {ckpt_path(ckdir, STAGE)}")
    g = torch.Generator().manual_seed(args.seed * 1000003 + 77)    # prompt order
    gg = torch.Generator().manual_seed(args.seed * 1000003 + 78)   # teacher sampling
    start = 0
    hist = []           # loaded below, truncated at the resume point
    if ck is not None and (not ck["done"] or extend):
        student.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        for grp in opt.param_groups:
            grp["lr"] = lr
        start = ck["step"]; g.set_state(ck["gens"][0]); gg.set_state(ck["gens"][1])
        restore_rng(ck)
        log(f"resume distill @ {start}")
        hist = load_hist(ckdir, STAGE, upto=start)
        monitor(STAGE, hist, start)
    Np = len(prompts)
    teacher.eval()
    student.train()
    bar = progress(range(start, steps), desc="[distill]",
                   initial=start, total=steps)
    for step in bar:
        cur_lr = sched.step(step)              # cosine, keyed off the ABSOLUTE step
        idx = torch.randint(0, Np, (args.batch,), generator=g).tolist()
        pairs = _teacher_texts(teacher, t_tok, [prompts[i] for i in idx], device, cfg,
                               gg, args.max_len)
        pairs = [(p, r) for p, r in pairs if r]
        if not pairs:
            continue
        ids, attn, rmask = _encode_student(s_tok, pairs, device, cfg["student_max_len"])
        logits = student(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
        rm = rmask[:, 1:].float()
        logp = F.log_softmax(logits, -1)
        lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        denom = rm.sum(-1).clamp(min=1)
        ce = (-(lp * rm).sum(-1) / denom).mean()            # per-example length-normalized CE
        kl = torch.zeros((), device=device)
        if ref is not None and kl_coef > 0:
            # KL( p_student || p_ref ) at every response position, full vocabulary. The
            # reference is the student's own frozen initialisation, so this is the price of
            # moving away from the pretrained prior -- the same role the KL penalty plays in
            # the RLHF stage.
            with torch.no_grad():
                ref_logp = F.log_softmax(
                    ref(input_ids=ids, attention_mask=attn).logits[:, :-1].float(), -1)
            per_pos = (logp.exp() * (logp - ref_logp)).sum(-1)          # (B, T-1)
            kl = ((per_pos * rm).sum(-1) / denom).mean()
        loss = ce + kl_coef * kl
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()
        with torch.no_grad():
            correct = ((logits.argmax(-1) == ids[:, 1:]) & rm.bool()).sum().float()
            acc = float((correct / rm.sum().clamp(min=1)).item())
        hist.append({"step": step, "loss": loss.item(), "ce": ce.item(),
                     "kl_ref": float(kl.item()), "ppl": float(ce.exp().item()),
                     "acc": acc, "gnorm": float(gnorm), "lr": cur_lr,
                     "resp_len": float(rm.sum(-1).mean().item())})
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(loss=f"{loss.item():.3f}", ce=f"{ce.item():.3f}",
                            kl=f"{float(kl.item()):.3f}", acc=f"{acc:.3f}",
                            len=f"{rm.sum(-1).mean().item():.0f}",
                            gnorm=f"{float(gnorm):.2f}")
        if (step + 1) % args.checkpoint_every_steps == 0:
            save_ckpt(ckdir, STAGE, student, opt, step + 1, steps, [g, gg])
            save_hist(ckdir, STAGE, hist)
        if args.plot_every_steps > 0 and (step + 1) % args.plot_every_steps == 0:
            monitor(STAGE, hist, step + 1)
            if preview:
                preview(student, STAGE, step + 1)        # 20 readable generation examples
    student.eval()
    evald = {"teacher": cfg["teacher_stage"], "student": cfg["student"],
             "n_prompts": len(prompts)}
    save_ckpt(ckdir, STAGE, student, opt, steps, steps, [g, gg], evald=evald)
    save_hist(ckdir, STAGE, hist)
    last = hist[-1] if hist else {}
    log(f"distill done: CE={last.get('ce', float('nan')):.4f} "
        f"KL={last.get('kl_ref', float('nan')):.4f} "
        f"ppl={last.get('ppl', float('nan')):.2f} acc={last.get('acc', float('nan')):.4f}")
    monitor(STAGE, hist)
    return student


def run_stage(ctx):
    """Assemble teacher, student and prompts, distil, and preview the student."""
    from helpers import common
    args = ctx["args"]
    log, device = ctx["log"], ctx["device"]
    cfg = config.DISTILL
    log("=== DISTILL (sequence-level KD: aligned zetagpt -> gpt2-small) ===")
    prompts, n_files = load_prompts()
    if not prompts:
        raise SystemExit(f"no prompts under {config.DISTILL_DIR} -- "
                         f"run make_distill_prompts.py first")
    helpers.table("distillation prompts", [
        ("dataset", os.path.basename(config.DISTILL_DIR.rstrip("/"))),
        ("directory", config.DISTILL_DIR),
        ("files", helpers.count(n_files)),
        ("prompts", helpers.count(len(prompts))),
        ("teacher", cfg["teacher_stage"]),
        ("student", cfg["student"])], out=log)
    teacher = common.load_stage_model(ctx, cfg["teacher_stage"])
    teacher.requires_grad_(False)
    student, s_tok = build_student(device)
    # the KL anchor: a frozen copy of the student's PRETRAINED initialisation
    ref = None
    if cfg.get("kl_coef", 0.0) > 0:
        ref, _ = build_student(device)
        ref.load_state_dict(student.state_dict())
        ref.requires_grad_(False)
        ref.eval()
    t_par = sum(p.numel() for p in teacher.parameters())
    s_par = sum(p.numel() for p in student.parameters())
    log(f"teacher: {cfg['teacher_stage']}-ed zetagpt "
        f"({len(teacher.blocks)} layers, {t_par/1e6:.2f}M params)")
    log(f"student: {cfg['student']} (gpt2-small, {s_par/1e6:.2f}M params, "
        f"pretrained init)")
    # distill previews on ITS OWN prompts (the prompt bank), generated by the STUDENT with the
    # student's tokenizer
    preview = common.make_preview(s_tok, device, cfg["student_max_len"], log, prompts)
    student = train(student, s_tok, teacher, ctx["tok"], prompts, ctx["ckdir"], args, log,
                  ctx["monitor"], preview=preview, ref=ref)
    # a handful of student samples on fixed prompts, comparable across runs
    s = config.SAMPLING
    for i, pr in enumerate(prompts[:5]):
        pids = s_tok(pr, add_special_tokens=False)["input_ids"]
        gen = sample_generate(student, pids, device, s["max_new"], s["temperature"],
                              s["top_k"], s["top_p"], s_tok.eos_token_id,
                              cfg["student_max_len"])
        log(f"[distill sample {i + 1}/5] PROMPT: {pr[:100]!r}")
        log(f"[distill sample {i + 1}/5] GEN:    {tok_decode(s_tok, gen).strip()[:200]!r}")
