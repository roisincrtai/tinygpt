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

Prompts are wrapped to match the training format. No KV cache (the prefix is recomputed
each step), matching the project.
"""
import os
import argparse

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
    ap.add_argument("--max_new_tokens", "-n", type=int, default=200)
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


def decode(tok, ids):
    if hasattr(tok, "decode"):                    # BPE / HF tokenizer
        return tok.decode(ids, skip_special_tokens=True)
    inv = getattr(tok, "inv", {})                 # WordTokenizer (legacy)
    return " ".join(inv.get(i, "<unk>") for i in ids)


@torch.no_grad()
def generate(model, prompt_ids, device, max_new, temperature, top_k, top_p, eos_id, max_len):
    """Autoregressive sampling; no KV cache (recompute the prefix each step)."""
    cur, out = list(prompt_ids), []
    for _ in range(max_new):
        ctx = cur[-max_len:] if max_len else cur
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
    return out


def main():
    args = parse_args()
    device = resolve_device(args.gpu)
    torch.manual_seed(args.seed)

    sd, cfg, ck = read_checkpoint(args.checkpoint)
    model, tok, max_len = build_from_checkpoint(sd, cfg, args, device, ck)
    if sd is not None:
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[chat] loaded {args.checkpoint} "
              f"(missing={len(miss)} unexpected={len(unexp)})", flush=True)
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

    def wrap(text):
        if tmpl == "chat":
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           tokenize=False, add_generation_prompt=True)
        return TEMPLATES[tmpl].format(p=text)

    def reply(text):
        ids = tok(wrap(text), add_special_tokens=False)["input_ids"]
        gen = generate(model, ids, device, args.max_new_tokens, args.temperature,
                       args.top_k, args.top_p, eos_id, max_len)
        return decode(tok, gen).strip()

    print(f"[chat] ckpt={args.checkpoint or '(base)'}  device={device}  template={tmpl}  "
          f"(temp={args.temperature} top_k={args.top_k} top_p={args.top_p})", flush=True)
    if args.prompt:
        print(f"\nYou:   {args.prompt}\nModel: {reply(args.prompt)}", flush=True)
        return
    print("Type a prompt; Ctrl-D or 'exit'/'quit' to leave.", flush=True)
    while True:
        try:
            text = input("\nYou:   ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        print(f"Model: {reply(text)}", flush=True)


if __name__ == "__main__":
    main()
