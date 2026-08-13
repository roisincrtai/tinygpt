"""
chat.py -- load a trained checkpoint and chat with it, to qualitatively test what a
pretrain / SFT / RLHF / DPO / distilled model actually generates.

Loading is by --checkpoint ONLY: point it at any stage's

    checkpoints/<stage>/checkpoint_<model>_<pe>_<stage-label>.pt

(a dict with a 'model' entry) or at a raw state_dict. The architecture is AUTO-DETECTED from
the state dict:

    ZetaGPT state dict ('blocks.*')          -> ZetaGPT rebuilt from the ARCHITECTURE STORED
                                                IN THE CHECKPOINT + the pipeline BPE
    gpt2 student state dict ('transformer.*') -> the distilled gpt2-small + its HF
                                                tokenizer

    # interactive REPL on the most aligned checkpoint present (the default)
    python chat.py
    python chat.py --checkpoint checkpoints/sft/checkpoint_zetagpt-s_ssm_sft.pt

    # one-shot, greedy
    python chat.py --checkpoint checkpoints/rlhf/checkpoint_zetagpt-s_ssm_instruct-rlhf.pt \
        --prompt "Where?" --temperature 0

IT IS A CONVERSATION, NOT A SEQUENCE OF PROMPTS. Every turn is prompted with the whole
transcript so far, wrapped in the training format -- rlhf_hh's "\n\nHuman: ... \n\nAssistant:",
the same text stage 6 is fine-tuned on -- so a follow-up like "and why?" has something to refer
to. `reset` clears it, `history` prints it. The transcript is left-truncated only when it would
leave the reply less than half the window.

IT STREAMS, token by token, in colour when the output is a terminal. And it has NO LENGTH CAP:
a reply ends at <|endoftext|> or at the context window, and at nothing else.

Decoding is incremental, through the KV cache (helpers/kv_cache.py).
"""
import os
import argparse
import sys

import torch
import torch.nn.functional as F

import default_config as config
from helpers import dataset_helpers as dsets
from tokenizer import BPETokenizer
from model import ZetaGPT

TEMPLATES = {
    "none": "{p}",
    "hh":   "\n\nHuman: {p}\n\nAssistant:",
    "shp":  "POST: {p}\n\nRESPONSE: ",
}


class C:
    """ANSI colours, or nothing at all.

    OFF WHENEVER THE OUTPUT IS NOT A TERMINAL, and off when NO_COLOR is set (the convention:
    any value, however empty, means no colour). `python chat.py > transcript.txt` must produce a
    transcript and not a file full of escape sequences, and that decision belongs here rather
    than at each print -- a colour that has to be remembered at forty call sites is a colour
    that leaks into a log."""
    _on = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None \
        and os.environ.get("TERM", "") != "dumb"
    YOU = "\033[1;36m" if _on else ""          # bold cyan  -- what you typed
    MODEL = "\033[0;32m" if _on else ""        # green      -- what the model said
    INFO = "\033[2;37m" if _on else ""         # dim grey   -- the harness talking
    OFF = "\033[0m" if _on else ""

    @classmethod
    def info(cls, m):
        print(f"{cls.INFO}{m}{cls.OFF}", flush=True)


def _stage_checkpoints(stage="*"):
    """Every checkpoint under checkpoints/<stage>/, oldest first.

    Two patterns are globbed because a checkpoint's name carries its configuration --
    checkpoint_<model>_<pe>_<stage-label>.pt -- and is therefore not predictable from the stage
    alone: the size, the positional encoding and the label all vary. `last.pt` is matched as
    well so a checkpoint written before that convention is still found rather than silently
    reported as missing."""
    import glob
    pats = [os.path.join(config.CHECKPOINT_DIR, stage, "checkpoint_*.pt"),
            os.path.join(config.CHECKPOINT_DIR, stage, "last.pt")]
    return sorted({p for pat in pats for p in glob.glob(pat)}, key=os.path.getmtime)


# The stages in DESCENDING order of alignment, which is the order chat.py wants a default in:
# a preference-tuned model is the one worth talking to, and an untuned pretrain checkpoint is
# the last resort. Not simply "the newest file", because re-running an early stage would then
# silently demote the default.
_DEFAULT_ORDER = ("dpo", "rlhf", "cot", "sft", "distill", "pretrain")


def default_checkpoint():
    """The checkpoint to open when --checkpoint is not given. Falls back to a path that does
    NOT exist when nothing has been trained, so read_checkpoint prints its 'not found' notice
    with the list of alternatives instead of quietly loading an untrained model."""
    for stage in _DEFAULT_ORDER:
        found = _stage_checkpoints(stage)
        if found:
            return found[-1]
    return os.path.join(config.CHECKPOINT_DIR, "dpo", "checkpoint_<model>_<pe>_instruct-dpo.pt")


DEFAULT_CKPT = default_checkpoint()


def parse_args():
    ap = argparse.ArgumentParser(description="chat with a trained checkpoint")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT,
                    help="weights to load: any stage's "
                         "checkpoints/<stage>/checkpoint_<model>_<pe>_<stage-label>.pt or a "
                         "raw state_dict; the architecture (zetagpt vs distilled gpt2 "
                         "student) is auto-detected. Empty = base (untrained) zetagpt. "
                         f"Default: {os.path.relpath(DEFAULT_CKPT, config.ROOT)}")
    ap.add_argument("--dataset", default="hh",
                    help="only used for the prompt template and the tokenizer fallback")
    ap.add_argument("--data_dir", default=config.DOWNLOAD_DIR)
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--prompt", default="", help="one-shot prompt; if empty, start an interactive REPL")
    ap.add_argument("--template", default="", choices=["", "none", "hh", "shp", "chat"],
                    help="prompt wrapper; '' = auto (hh/shp by dataset, else the model's chat template "
                         "if it has one, else none)")
    # THERE IS NO LENGTH CAP. The model stops when it emits <|endoftext|> or when the context
    # window is full, and nothing else stops it -- the same rule the CoT stage was put on. A cap
    # here was a number invented by the harness, and every reply that ran into it was cut in the
    # middle of a sentence and read as the model losing its thread.
    # "--temp" is spelled out as an alias: argparse's prefix matching makes the abbreviation
    # ambiguous against --template, and failing on `--temp 1` is a poor greeting.
    ap.add_argument("--temperature", "--temp", "-t", type=float, default=0.8,
                    help="0 = greedy (argmax)")
    ap.add_argument("--top_k", type=int, default=50, help="0 = disabled")
    ap.add_argument("--top_p", type=float, default=0.95, help="nucleus; 1.0 = disabled")
    ap.add_argument("--max_len", type=int, default=0, help="0 = auto per model")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def resolve_device(choice):
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(choice)


def _available():
    """Stage checkpoints that DO exist, so a missing one can suggest the alternatives."""
    return sorted(os.path.relpath(p, config.ROOT) for p in _stage_checkpoints())


def read_checkpoint(ckpt):
    """The state dict inside `ckpt` (a stage checkpoint or a raw state_dict), or None when no
    (existing) checkpoint was given -- the caller then falls back to the base model.

    A MISSING checkpoint is shouted about rather than mentioned. An untrained ZetaGPT samples
    uniformly from a 50k vocabulary, so it emits fluent-looking word salad -- which reads
    exactly like a broken tokenizer, and is the single most confusing thing this script can
    do quietly."""
    if not ckpt:
        print("[chat] no checkpoint given -> UNTRAINED model; output will be random text",
              flush=True)
        return None, None, None
    if not os.path.isfile(ckpt):
        have = _available()
        print("\n" + "!" * 78, flush=True)
        print(f"[chat] CHECKPOINT NOT FOUND: {ckpt}", flush=True)
        print("[chat] Falling back to an UNTRAINED model. Its output is random text drawn "
              "from the\n[chat] vocabulary -- nothing is wrong with the tokenizer or the "
              "decoder.", flush=True)
        print(("[chat] Available checkpoints:\n[chat]   "
               + "\n[chat]   ".join(have)) if have else
              "[chat] No stage checkpoints exist yet: run a stage first, e.g.\n"
              "        ./stage5_pretrain.sh",
              flush=True)
        print("!" * 78 + "\n", flush=True)
        return None, None, None
    obj = torch.load(ckpt, map_location="cpu")
    if isinstance(obj, dict) and "model" in obj:
        # The whole record is returned, not just the weights: a checkpoint carries the
        # TOKENIZER it was trained with, and reading the weights while leaving the vocabulary
        # behind is how a model comes to talk fluent nonsense with nothing in the log to say so.
        return obj["model"], obj.get("model_cfg"), obj
    return obj, None, None


def build_from_checkpoint(sd, cfg, args, device, ck=None):
    """(model, tok, max_len) matching the checkpoint's architecture.

    'transformer.*' keys mean the distilled gpt2-small student. Otherwise it is a ZetaGPT, and
    the ARCHITECTURE RECORDED IN THE CHECKPOINT is used -- a model trained at a different
    width or depth must be rebuilt as it was, and rebuilding it from whatever default_config.py currently
    says would fail to load (or load something subtly different). config.MODEL is only the
    fallback for checkpoints written before the architecture was stored."""
    if sd is not None and any(k.startswith("transformer.") for k in sd):
        # the distilled gpt2-small student (checkpoints/distill/)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(config.DISTILL["student"],
                                            cache_dir=config.MODEL_DIR)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(config.DISTILL["student"],
                                                     cache_dir=config.MODEL_DIR).to(device)
        return model, tok, args.max_len or config.DISTILL["student_max_len"]
    # ZetaGPT + THE TOKENIZER THE CHECKPOINT WAS TRAINED WITH, which travels inside it. A
    # checkpoint written before that, or one that could not embed its tokenizer, falls back to
    # checkpoints/bpe/bpe.json -- announced as a fallback, because it is the only candidate
    # rather than a verified match. This is what lets a lang_sft checkpoint, whose vocabulary
    # was EXTENDED for another language, decode correctly without any of it being configured.
    # imported HERE, not at the top: helpers.common imports this module, so a module-level
    # `import helpers` would close the cycle (see helpers/__init__.py).
    from helpers.utils import bpe_from_ckpt
    tok = bpe_from_ckpt(ck, fallback=config.BPE_PATH) if ck is not None else None
    if tok is not None:
        pass
    elif os.path.isfile(config.BPE_PATH):
        tok = BPETokenizer.load(config.BPE_PATH)
    else:
        tr, ev = dsets.load_pairs(args.dataset, args.data_dir, 0.05, args.seed, 0)
        texts = [p[k] for p in (tr + ev) for k in ("prompt", "chosen", "rejected")]
        tok = BPETokenizer.build(texts, num_merges=config.BPE["num_merges"])
    if cfg:
        m_cfg = {k: v for k, v in cfg.items() if k != "vocab_size"}
    else:
        m_cfg = dict(config.MODEL)
        if sd is not None:                            # infer at least the depth
            layers = {int(k.split(".")[1]) for k in sd if k.startswith("blocks.")}
            if layers:
                m_cfg["n_layer"] = max(layers) + 1
    model = ZetaGPT(vocab_size=len(tok), **m_cfg).to(device)
    print(f"[chat] architecture: state space module + gated attention, no positional encoding, "
          f"{len(model.blocks)} layers", flush=True)
    return model, tok, args.max_len or m_cfg["block_size"]


def summarise(model, tok, ck, cfg, path, device, max_len, args, log):
    """The loaded model, as a table, BEFORE the first prompt.

    WHAT IS ANSWERED HERE is the question every confusing chat session ends up asking: which
    model is this, actually. A checkpoint's filename carries its configuration, but a filename
    is not read carefully at midnight, and the two mistakes that waste an evening -- talking to
    the pretrain checkpoint while believing it is the aligned one, and a vocabulary that does
    not match the weights -- are both invisible in the output. They are both in this table.

    EVERYTHING IS READ FROM WHAT WAS LOADED, never from default_config: the architecture comes
    from the checkpoint's own model_cfg, the parameter counts from the built modules, the
    vocabulary from the tokenizer that came with the weights. A summary that described the
    configuration file rather than the object in memory would agree with itself while being
    wrong about the model."""
    import helpers                       # HERE, not at the top: helpers.common imports this
    rows, cfg = [], dict(cfg or {})      # module, so a module-level import closes the cycle
    rel = os.path.relpath(path, config.ROOT) if path else "(none: untrained)"
    rows.append(("checkpoint", rel))
    if path and os.path.isfile(path):
        # BYTES, not helpers.human -- that formats COUNTS, and a 2.1 GiB file rendered as
        # "2.15B" reads as two billion of something rather than as gigabytes.
        b = os.path.getsize(path)
        rows.append(("size on disk", next(f"{b / d:.2f} {u}" for u, d in
                                          (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10),
                                           ("B", 1)) if b >= d or d == 1)))
    if ck:
        step, total = ck.get("step"), ck.get("total")
        if step is not None:
            rows.append(("trained to", f"step {step:,}" + (f" of {total:,}" if total else "")
                         + ("  (finished)" if ck.get("done") else "  (unfinished)")))
        if ck.get("eval"):
            rows.append(("recorded eval", str(ck["eval"])))
    rows.append(("", ""))

    n_par = sum(p.numel() for p in model.parameters())
    emb = getattr(model, "tok", None)
    n_emb = emb.weight.numel() if emb is not None and hasattr(emb, "weight") else 0
    tied = (emb is not None and getattr(model, "head", None) is not None
            and emb.weight is model.head.weight)
    if cfg:
        d, h = cfg.get("n_embd", 0), cfg.get("n_head", 0)
        rows += [
            ("layers", helpers.count(cfg.get("n_layer", len(getattr(model, "blocks", []))))),
            ("heads", helpers.count(h)),
            ("d_model", helpers.count(d)),
            ("head dim", helpers.count(d // h if h else 0)),
            ("ffn", f"{cfg.get('ffn_factor', 4)}x  ({d * cfg.get('ffn_factor', 4):,})"),
            ("gated attention", "yes" if cfg.get("gated_attn", True) else "no"),
            # THE HEADLINE OF THE WHOLE PROJECT, so it is stated rather than implied by the
            # absence of a row: pe="ssm" is the state space module carrying order, pe="rope"
            # the ablation that puts rotary positions back into attention.
            ("positional encoding",
             "NONE -- order comes from the state space module (NoPE)"
             if cfg.get("pe", "ssm") == "ssm" else f"{cfg.get('pe')} (ablation)"),
            ("context window", f"{cfg.get('block_size', max_len):,} tokens"),
        ]
    else:
        rows.append(("architecture", f"{type(model).__name__} (no model_cfg in the checkpoint)"))
    rows.append(("", ""))
    rows += [
        ("parameters", f"{helpers.human(n_par)}  ({n_par:,})"),
        ("  embedding", f"{helpers.human(n_emb)}" + ("  (tied to the head)" if tied else "")),
        ("  everything else", helpers.human(n_par - n_emb)),
        ("vocabulary", helpers.count(len(tok)) if tok is not None else "?"),
        ("tokenizer", "from the checkpoint" if (ck or {}).get("bpe")
                      else f"fallback: {os.path.relpath(config.BPE_PATH, config.ROOT)}"),
        ("", ""),
        ("device", str(device)),
        ("dtype", str(next(model.parameters()).dtype).replace("torch.", "")),
        ("sampling", f"temperature {args.temperature}, top_k {args.top_k or 'off'}, "
                     f"top_p {args.top_p}"),
        ("reply length", f"unbounded -- stops at <|endoftext|> or at the {max_len:,}-token "
                         f"window"),
    ]
    helpers.table("loaded model", rows, out=log)


def decode(tok, ids):
    if hasattr(tok, "decode"):                    # BPE / HF tokenizer
        return tok.decode(ids, skip_special_tokens=True)
    inv = getattr(tok, "inv", {})                 # WordTokenizer (legacy)
    return " ".join(inv.get(i, "<unk>") for i in ids)


@torch.no_grad()
def generate(model, prompt_ids, device, max_new, temperature, top_k, top_p, eos_id, max_len,
             desc="", on_token=None):
    """Autoregressive sampling, CACHED, and REPORTED while it runs.

    IT RECOMPUTED THE WHOLE PREFIX AT EVERY STEP -- O(n^2). That was tolerable while a preview
    was sixty tokens; it stopped being tolerable the moment the chain-of-thought previews began
    filling the context window, because one 7,800-token sample is then 30,423,900 token-forwards
    instead of 7,800, and five of them are 152 million. The cache keeps each layer's keys,
    values, convolution window and recurrence state, exactly as the GRPO rollout does.

    AND IT PRINTED NOTHING WHILE IT RAN. `desc` puts a self-erasing bar on the token loop, so a
    long generation shows progress and leaves no trace in the block afterwards -- which is what
    the surrounding preview needs: something to read, not a bar wedged between its lines. A
    generation that takes minutes with no output is indistinguishable from a hung process, and
    this project's rule that every long-running loop reports is there for exactly that reason.

    `on_token(ids)` IS THE OTHER WAY OF SATISFYING THAT RULE, for a reader rather than a log:
    it is called after every accepted token with the ids generated SO FAR, and whatever it
    returns is ignored. The whole list is passed rather than the one new id because a
    byte-level BPE token is not a character -- a multi-byte character spans several tokens, and
    an accented letter decoded one token at a time comes out as replacement characters. The
    caller decodes the list and prints only what has grown, which is correct at every prefix.
    Pass `on_token` OR `desc`, not both: a bar and a stream on one line fight for it."""
    from helpers.kv_cache import Cache
    from helpers.utils import progress
    cur, out = list(prompt_ids), []
    cache = Cache(len(model.blocks)) if hasattr(model, "blocks") else None
    fed = 0                                   # how much of `cur` the cache has already seen
    steps = progress(range(max_new), desc=desc) if desc else range(max_new)
    for _ in steps:
        ctx = cur[-max_len:] if max_len else cur
        if cache is not None and fed and len(ctx) == len(cur):
            x = torch.tensor([cur[fed:]], device=device)          # only what is new
            h = model.hidden_states(input_ids=x, cache=cache)
            logits = model.head(h[:, -1])[0].float()
            fed = len(cur)
        elif cache is not None:
            # the first pass, or a window that has begun sliding: the cache cannot describe a
            # prefix that has been truncated, so it is rebuilt from what is actually visible
            if len(ctx) != len(cur):
                cache = Cache(len(model.blocks))
            x = torch.tensor([ctx], device=device)
            h = model.hidden_states(input_ids=x, cache=cache)
            logits = model.head(h[:, -1])[0].float()
            fed = len(cur)
        else:
            x = torch.tensor([ctx], device=device)
            logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, -1].float()
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                thresh = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            if 0 < top_p < 1:
                sp, si = torch.sort(probs, descending=True)
                keep = (torch.cumsum(sp, dim=-1) - sp) <= top_p     # always keep the top token
                sp = sp * keep
                probs = torch.zeros_like(probs).scatter(0, si, sp)
                probs = probs / probs.sum().clamp_min(1e-9)
            nxt = int(torch.multinomial(probs, 1))
        if eos_id is not None and nxt == eos_id:
            break
        cur.append(nxt); out.append(nxt)
        if on_token is not None:
            on_token(out)
    if hasattr(steps, "close"):
        steps.close()
    return out


def main():
    args = parse_args()
    device = resolve_device(args.gpu)
    torch.manual_seed(args.seed)

    sd, cfg, ck = read_checkpoint(args.checkpoint)
    model, tok, max_len = build_from_checkpoint(sd, cfg, args, device, ck)
    if sd is not None:
        miss, unexp = model.load_state_dict(sd, strict=False)
        C.info(f"[chat] loaded {args.checkpoint} "
               f"(missing={len(miss)} unexpected={len(unexp)})")
    model.eval()
    eos_id = getattr(tok, "eos_token_id", None)

    # resolve the prompt template
    tmpl = args.template
    if not tmpl:
        if args.dataset in ("hh", "shp"):
            tmpl = args.dataset
        elif getattr(tok, "chat_template", None):
            tmpl = "chat"
        else:
            tmpl = "none"

    # THE CONVERSATION, not the last turn of it. `history` is [(you, model), ...] and every
    # prompt is built from the WHOLE of it: a model asked "and why?" with only those two words
    # for context has been given an unanswerable question, and answering it anyway is the
    # failure that reads as the model being stupid rather than as the harness being wrong.
    history = []

    def wrap(text):
        """The full transcript, ending at the point the model is to continue from.

        The markers are rlhf_hh's, which is not a detail: stage 6 trains on
        "\\n\\nHuman: ... \\n\\nAssistant:", stage 7 scores it, stages 8 and 10 roll out from it.
        Prompting here in any other shape would be asking the model a question in a format it
        was trained out of. A template of "none" keeps the turns separated by blank lines,
        which is the most a model with no instruction tuning can be given."""
        if tmpl == "chat":
            msgs = []
            for u, a in history:
                msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
            msgs.append({"role": "user", "content": text})
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        past = "".join(TEMPLATES[tmpl].format(p=u) + " " + a for u, a in history)
        return past + TEMPLATES[tmpl].format(p=text)

    def fit(ids):
        """The transcript, LEFT-TRUNCATED only if it would leave the reply no room.

        The window is finite and a conversation is not, so eventually something has to go, and
        it is the OLDEST turns -- dropping the end would discard the question being asked.
        Truncation is by token from the left rather than by whole turns, which can cut a turn in
        half; that is the cheap approximation, and it degrades an old turn rather than losing
        the new one.

        THE TRANSCRIPT MAY TAKE AT MOST HALF THE WINDOW, so the reply always has the other half.
        That is a derived rule and not a knob: there is no length cap to set here any more, and
        a conversation that has grown to fill the context would otherwise leave the model a
        handful of tokens to answer in and look as though it had stopped talking. Reported when
        it happens, because a model that has quietly forgotten the start of the conversation
        looks exactly like a model that is ignoring it."""
        if not max_len or len(ids) <= max_len // 2:
            return ids
        room = max(max_len // 2, 1)
        C.info(f"[chat] transcript {len(ids):,} tokens: keeping the last {room:,} so the reply "
               f"has the other half of the {max_len:,} window")
        return ids[-room:]

    def reply(text):
        """Generate, PRINTING AS IT GOES, and return what was said.

        THE BUDGET IS WHATEVER THE WINDOW LEAVES. The model stops at <|endoftext|> or when the
        context is full, and at nothing else.

        The stream decodes the whole generation each step and prints only the growth. Decoding
        the one new token instead would be cheaper and wrong: a byte-level BPE splits a
        multi-byte character across tokens, so "é" arrives as two ids and would print as two
        replacement characters."""
        ids = fit(tok(wrap(text), add_special_tokens=False)["input_ids"])
        budget = max(max_len - len(ids), 1) if max_len else 4096
        shown = 0

        def stream(out):
            nonlocal shown
            s = decode(tok, out)
            if len(s) > shown:
                # COLOURED PER CHUNK rather than opened once and closed at the end: an
                # interrupted generation would otherwise leave the terminal green.
                print(f"{C.MODEL}{s[shown:]}{C.OFF}", end="", flush=True)
                shown = len(s)

        print(f"{C.MODEL}Model: {C.OFF}", end="", flush=True)
        gen = generate(model, ids, device, budget, args.temperature,
                       args.top_k, args.top_p, eos_id, max_len, on_token=stream)
        print(flush=True)
        answer = decode(tok, gen).strip()
        history.append((text, answer))
        return answer

    print()
    summarise(model, tok, ck, cfg, args.checkpoint, device, max_len, args,
              log=lambda m: print(f"{C.INFO}{m}{C.OFF}", flush=True))
    C.info(f"[chat] prompt template: {tmpl}"
           + ("   (\\n\\nHuman: ... \\n\\nAssistant: -- what stage 6 trains on)"
              if tmpl == "hh" else ""))
    if args.prompt:
        print(f"\n{C.YOU}You:   {args.prompt}{C.OFF}", flush=True)
        reply(args.prompt)                     # prints itself, token by token
        return
    C.info("Type a prompt. 'reset' forgets the conversation, 'history' shows it, "
           "Ctrl-D or 'exit'/'quit' leaves.")
    while True:
        try:
            text = input(f"\n{C.YOU}You:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print(C.OFF)
            break
        finally:
            sys.stdout.write(C.OFF); sys.stdout.flush()
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        # A CONVERSATION HAS TO BE ENDABLE WITHOUT ENDING THE PROCESS. Otherwise the only way to
        # ask an unrelated question is to reload the checkpoint, which on a big model is a
        # minute of waiting to clear four lines of context.
        if text.lower() == "reset":
            history.clear()
            C.info("[chat] conversation cleared")
            continue
        if text.lower() == "history":
            if not history:
                C.info("[chat] nothing yet")
            for i, (u, a) in enumerate(history, 1):
                print(f"{C.INFO}[chat] {i}.{C.OFF} {C.YOU}You:   {u}{C.OFF}", flush=True)
                print(f"{C.INFO}[chat]    {C.OFF}{C.MODEL}Model: {a}{C.OFF}", flush=True)
            n = len(tok(wrap(""), add_special_tokens=False)["input_ids"])
            C.info(f"[chat] {len(history)} turns, {n:,} tokens of a {max_len:,} window")
            continue
        reply(text)


if __name__ == "__main__":
    main()
