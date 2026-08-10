"""
common.py -- the setup every stage shares, and nothing stage-specific.

This module knows about arguments, devices, the tokenizer, the encoder, the model factory,
figures and generation previews. It does NOT know which stages exist, what they optimise or
in what order they run: each stage package (pretrain/, sft/, instruct_rlhf/, instruct_dpo/,
distill/) owns its own algorithm, corpus and reporting, and asks setup() only for what it
consumes. Adding or removing a stage therefore does not touch this file.

ALL configuration lives in default_config.py; parse_args() exposes every knob as a CLI flag whose
default comes from config, so a flag present in argv wins.

    parse_args()              argparse over default_config.py's values
    setup(args)               -> ctx dict (device, log, tok, enc, model factory, monitor,
                                 sampler, data, dirs); draws bpe_dynamics only when draw_bpe=True
    frozen(model)             a detached, eval-mode deep copy (reference / teacher)
    load_stage_model(ctx, s)  a fresh model with stage s's saved weights loaded (errors if missing)

Outputs: checkpoints/<stage>/{checkpoint_,history_}<model>_<pe>_<stage-label>.{pt,json},
         outputs/plots/<stage>/dynamics_<model>_<pe>_<stage-label>.pdf
"""
import argparse
import copy
import os
import random

import default_config as config
from . import dataset_helpers as dsets
from . import utils as helpers
from .utils import progress
from chat import generate, decode
from model import ZetaGPT
from . import visualization as viz

from tokenizer import run as train_bpe    # stage 2's tokenizer builder (also draws bpe_dynamics)

ROOT = config.ROOT
CKDIR = config.CHECKPOINT_DIR
PLOTDIR = config.PLOT_DIR
BPE_PATH = config.BPE_PATH
N_SAMPLES = config.N_SAMPLES


def plot_dir(stage):
    """Figures are organised outputs/plots/<stage>/*.pdf -- one directory per stage."""
    return os.path.join(PLOTDIR, stage)


# Architecture keys ZetaGPT no longer takes by name. Checkpoints written before `pe` existed
# carry use_ssm / use_rope, which say exactly the same thing, so they are TRANSLATED rather
# than dropped -- a run must always be rebuilt with the source of position it was trained with.
RETIRED_CFG_KEYS = ("use_ssm", "use_rope")


def translate_cfg(saved):
    """A checkpoint's model_cfg in today's vocabulary."""
    cfg = {k: v for k, v in saved.items() if k not in RETIRED_CFG_KEYS and k != "vocab_size"}
    if "pe" not in cfg and "use_ssm" in saved:
        cfg["pe"] = "ssm" if saved["use_ssm"] else "rope"
    return cfg


def model_cfg_from_args(args, overrides=None):
    """default_config.MODEL with the command line applied.

    --model_scheme picks depth, width and the scheme's own context window together, because
    those three are not independent choices: a scheme is a point on a size ladder, and mixing
    one scheme's depth with another's width produces a model that no reported number describes.
    --context_window then overrides the context alone, which is the one part of a scheme it is
    legitimate to vary on its own -- it is a training choice, not an architectural one.
    --pe selects the single source of position: "ssm" (the state space module, no encoding) or
    "rope" (the ablation control)."""
    cfg = dict(config.MODEL)
    if args is not None and getattr(args, "model_scheme", None):
        cfg.update(config.scheme(args.model_scheme))
    if args is not None and getattr(args, "context_window", 0):
        cfg["block_size"] = int(args.context_window)
    if args is not None and getattr(args, "pe", None):
        cfg["pe"] = args.pe
    cfg.update(overrides or {})
    return cfg


def build_model(tok, device, model_cfg=None, args=None):
    """The from-scratch ZetaGPT (random init). This is the ONLY place a model is constructed,
    so every stage trains the same architecture."""
    m_cfg = model_cfg_from_args(args, model_cfg) if args is not None else dict(config.MODEL)
    if model_cfg and args is None:
        m_cfg.update(model_cfg)
    m_cfg.pop("vocab_size", None)
    for k in RETIRED_CFG_KEYS:
        m_cfg.pop(k, None)
    m = ZetaGPT(vocab_size=len(tok), **m_cfg).to(device)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    pos = ("implicit (state space module, no encoding)" if m.cfg["pe"] == "ssm"
           else "rotary inside attention (ABLATION: no state space module)")
    print(f"[model {helpers.model_name(m.cfg)}] trainable params: {trainable/1e6:.2f}M / "
          f"{total/1e6:.2f}M total ({trainable:,} / {total:,})  positions: {pos}", flush=True)
    print(f"[model zetagpt] cfg={m.cfg}", flush=True)
    return m


def parse_args(argv=None):
    """Every trainer knob as a CLI flag, defaulting to default_config.py's values."""
    t = config.TRAIN
    ap = argparse.ArgumentParser(description="zetagpt pipeline stage")
    ap.add_argument("--dataset", default=t["dataset"],
                    help="\"hh\" for the downloaded rlhf_hh tree, or a PATH to your "
                         "own folder of json/jsonl preference records (absolute, or "
                         "relative to --data_dir); the layout is detected, not declared")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="SEC.key=v",
                    help="override any value in default_config.py's dictionaries, e.g. "
                         "--set BPE.min_freq=3 --set RLHF.kl_coef=0.02. Repeatable. "
                         "Sections: BPE PRETRAIN SFT REWARD RLHF COT DPO DISTILL TRAIN "
                         "SCALING MODEL")
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default=t["gpu"])
    ap.add_argument("--seed", type=int, default=t["seed"])
    ap.add_argument("--batch", type=int, default=t["batch"])
    ap.add_argument("--micro_batch", type=int, default=t["micro_batch"])
    ap.add_argument("--max_len", type=int, default=t["max_len"],
                    help="truncation of an encoded example; 0 = the model's context window")
    ap.add_argument("--model_scheme", default=config.PRETRAIN["model_scheme"],
                    choices=list(config.SCHEMES),
                    help="which configuration scheme to build: depth, width and the context "
                         "window it is pretrained at, from default_config.SCHEMES")
    ap.add_argument("--pretrain_dir", default="",
                    help="pretraining corpus directory; empty = the scheme's own from "
                         "default_config.PRETRAIN_CORPUS (set for tiny and -s, unset for -m/-l). "
                         "Read locally: no stage downloads, see tools/download_data.py")
    ap.add_argument("--sft_dir", default="",
                    help="fine-tuning data directory; empty = default_config.SFT_DIR, "
                         "which is <instruct_dir>/rlhf_hh")
    ap.add_argument("--instruct_dir", default="",
                    help="root of the instruction-tuning tree (alpaca_gpt4/, rlhf_hh/) used "
                         "by stages 5, 6, 7, 9 and 10; empty = "
                         "default_config.INSTRUCT_DIR")
    ap.add_argument("--context_window", type=int, default=config.PRETRAIN["context_window"],
                    help="context window in tokens; 0 = the scheme's own (512 for tiny and "
                         "-s, 1024 for -m and -l). Not an architectural limit -- the model "
                         "refers to no "
                         "absolute position -- but what it is TRAINED at, and what the "
                         "checkpoint records as its block_size")
    ap.add_argument("--beta", type=float, default=t["beta"])
    ap.add_argument("--val_frac", type=float, default=t["val_frac"])
    ap.add_argument("--limit", type=int, default=t["limit"], help="cap #pairs loaded (0 = all)")
    ap.add_argument("--lr_schedule", default=t["lr_schedule"], choices=["cosine", "constant"])
    ap.add_argument("--lr_min_factor", type=float, default=t["lr_min_factor"],
                    help="cosine floor: the minimum lr is the stage lr divided by this")
    ap.add_argument("--checkpoint_every_steps", type=int, default=t["checkpoint_every_steps"])
    ap.add_argument("--plot_every_steps", type=int, default=t["plot_every_steps"])
    ap.add_argument("--pe", default=config.MODEL["pe"], choices=["ssm", "rope"],
                    help="how position enters: 'ssm' (the state space module, no positional "
                         "encoding) or 'rope' (ablation control: no module, rotary attention)")
    ap.add_argument("--ssm_stats_every", type=int, default=t["ssm_stats_every"],
                    help="record the state space module's diagnostics every N steps "
                         "(0 = off); drawn in the pretraining dynamics figure")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=t["resume"])
    # held-out / exposure probes (used by the DPO stage)
    ap.add_argument("--eval_every", type=int, default=t["eval_every"])
    ap.add_argument("--eval_pairs", type=int, default=t["eval_pairs"])
    ap.add_argument("--eb_every", type=int, default=t["eb_every"])
    ap.add_argument("--eb_pairs", type=int, default=t["eb_pairs"])
    ap.add_argument("--rollout_temp", type=float, default=t["rollout_temp"])
    ap.add_argument("--n_hist", type=int, default=t["n_hist"])
    ap.add_argument("--n_roll", type=int, default=t["n_roll"])
    ap.add_argument("--roll_tokens", type=int, default=t["roll_tokens"])
    ap.add_argument("--p_grid", default=t["p_grid"])
    # per-stage budgets / learning rates
    ap.add_argument("--bpe_merges", type=int, default=config.BPE["num_merges"])
    ap.add_argument("--pretrain_steps", type=int, default=config.PRETRAIN["steps"])
    ap.add_argument("--pretrain_lr", type=float, default=config.PRETRAIN["lr"])
    ap.add_argument("--sft_steps", type=int, default=config.SFT["steps"])
    ap.add_argument("--sft_lr", type=float, default=config.SFT["lr"])
    ap.add_argument("--reward_steps", type=int, default=config.REWARD["steps"])
    ap.add_argument("--reward_lr", type=float, default=config.REWARD["lr"])
    ap.add_argument("--rlhf_steps", type=int, default=config.RLHF["steps"])
    ap.add_argument("--rlhf_lr", type=float, default=config.RLHF["lr"])
    ap.add_argument("--cot_steps", type=int, default=config.COT["steps"])
    ap.add_argument("--cot_lr", type=float, default=config.COT["lr"])
    ap.add_argument("--cot_init", default=config.COT["init_stage"],
                    choices=["pretrain", "sft", "rlhf", "dpo"],
                    help="checkpoint the GRPO policy starts from; 'pretrain' is the "
                         "R1-Zero setting (RL on the base model, no supervised reasoning)")
    ap.add_argument("--cot_group", type=int, default=config.COT["group_size"],
                    help="completions sampled per problem; the group is GRPO's baseline "
                         "and must be >= 2")
    ap.add_argument("--dpo_steps", type=int, default=config.DPO["steps"])
    ap.add_argument("--dpo_lr", type=float, default=config.DPO["lr"])
    ap.add_argument("--distill_steps", type=int, default=config.DISTILL["steps"])
    ap.add_argument("--distill_lr", type=float, default=config.DISTILL["lr"])
    args = ap.parse_args(argv)
    args.model_dir = config.MODEL_DIR
    args.data_dir = config.DOWNLOAD_DIR
    # --max_len is the truncation applied when an example is encoded. Left at 0 it follows the
    # model that is actually going to be built -- the scheme's context window, or whatever
    # --context_window overrode it with -- rather than a constant. Reading it from
    # config.MODEL, as this did, meant a run that asked for a different scheme or context
    # silently encoded at the default's length, and the mismatch surfaced only as a truncation
    # rate nobody was looking at.
    if not args.max_len:
        args.max_len = model_cfg_from_args(args)["block_size"]

    # THE CORPUS DIRECTORIES, resolved once here rather than read from the config module at
    # each use. --instruct_dir moves a whole tree, so it is applied through config's own setter
    # to keep the derived directories (the fine-tuning data, alpaca, hh and the RLHF
    # prompt bank) consistent; repointing the root while one of them still pointed elsewhere is
    # how a run ends up fine-tuning on one dataset and scoring preferences from another.
    if args.instruct_dir:
        config.set_instruct_root(os.path.abspath(args.instruct_dir))
    args.instruct_dir = config.INSTRUCT_DIR
    args.sft_dir = os.path.abspath(args.sft_dir) if args.sft_dir else config.SFT_DIR
    # The pretraining corpus is per SCHEME: only zetagpt-s ships with one, so -m and -l resolve
    # to the empty string and the stage says so rather than scanning a directory that is not
    # there and reporting "0 files" as though that were a corpus.
    if not args.pretrain_dir:
        args.pretrain_dir = config.PRETRAIN_CORPUS.get(args.model_scheme, "")
    args.pretrain_dir = os.path.abspath(args.pretrain_dir) if args.pretrain_dir else ""
    return args


def frozen(model):
    """A detached, eval-mode deep copy: the frozen reference pi_ref / teacher."""
    ref = copy.deepcopy(model)
    ref.requires_grad_(False)
    ref.eval()
    return ref


N_PREVIEW = 20                       # generation examples printed every --plot_every_steps


def corpus_prompts(corpus, n=N_PREVIEW, n_words=12, seed=0, tok=None):
    """Preview prompts drawn from a corpus: the first `n_words` words of `n` sampled
    documents.

    Three corpus shapes, because the previews must work wherever the corpus came from: a
    memory-mapped TokenStream (sampled by document index -- only those documents are paged
    in, never the corpus), pre-tokenised records, and plain text records."""
    rng = random.Random(seed)
    if hasattr(corpus, "doc") and hasattr(corpus, "n_docs"):
        idx = rng.sample(range(corpus.n_docs), min(n, corpus.n_docs))
        texts = [decode(tok, corpus.doc(i)[:4 * n_words]) if tok is not None else ""
                 for i in idx]
    else:
        docs = rng.sample(corpus, min(n, len(corpus)))
        texts = [decode(tok, d["ids"][:4 * n_words]) if ("ids" in d and tok is not None)
                 else d.get("chosen", "") for d in docs]
    return [p for p in (" ".join(t.split()[:n_words]) for t in texts) if p]


def make_preview(tok, device, max_len, log, prompts, n=N_PREVIEW):
    """A `preview(model, stage, step)` closure that prints `n` generations from a FIXED set
    of prompts in a human-friendly format (plain print, NO timestamp prefix -- these blocks
    are meant to be read, not parsed):

        [1/20] PROMPT:   the prompt ...
               RESPONSE: what the model generated ...

    The prompts are fixed so consecutive previews are directly comparable: the reader
    watches the SAME prompts get better (or collapse) over training. Called by every
    generative stage (pretrain/sft/rlhf/dpo/distill) at each --plot_every_steps."""
    s = config.SAMPLING
    eos = getattr(tok, "eos_token_id", None)
    sel = [p for p in prompts if p.strip()][:n]

    def preview(model, stage, step):
        was_training = model.training
        model.eval()
        out = print          # plain, untimestamped: the log()'s "[  123s]" prefix ruins it
        out(f"\n===== {stage} @ step {step}: {len(sel)} generation examples "
            f"(temp={s['temperature']} top_k={s['top_k']} top_p={s['top_p']}) =====\n",
            flush=True)
        # NO progress bar here: the block is meant to be READ, and a bar redrawing itself
        # between the printed lines destroys it. The numbered [i/n] header is the progress.
        for i, pr in enumerate(sel, 1):
            ids = tok(pr, add_special_tokens=False)["input_ids"]
            gen = generate(model, ids, device, s["max_new"], s["temperature"],
                           s["top_k"], s["top_p"], eos, max_len)
            out(f"[{i}/{len(sel)}] PROMPT:   {' '.join(pr.split())}", flush=True)
            out(f"        RESPONSE: {' '.join(decode(tok, gen).split())}\n", flush=True)
        out(f"===== end of {stage} examples @ step {step} =====\n", flush=True)
        if was_training:
            model.train()
    return preview


def _make_sampler(tok, device, max_len, log, prompts, eos_id):
    """A `sample(model, stage)` closure for the END-OF-STAGE samples: the same readable
    PROMPT:/RESPONSE: block as the periodic preview (no progress bar -- a bar redrawing
    itself between printed lines makes the block unreadable), labelled 'final'."""
    body = make_preview(tok, device, max_len, log, prompts, n=N_SAMPLES)
    def sample(model, stage, n=N_SAMPLES):
        body(model, stage, "final")
    return sample


def setup(args, need_pairs=True, pretokenize_pairs=False, draw_bpe=False):
    """Build the context a stage needs, and NOTHING ELSE.

    A stage declares what it consumes; common has no table of stages and no idea which one is
    running. `need_pairs` loads the instruction data (and reports its statistics);
    `pretokenize_pairs` additionally caches their token ids. A corpus stage passes
    need_pairs=False and reads its own directory itself, so pretraining never touches the
    preference file. The one exception is a tokenizer that has to be BUILT: its vocabulary is
    shared, so it is trained over every corpus regardless of who asked for it.
    """
    def log(m): print(m, flush=True)     # plain, no elapsed-time prefix (human-readable logs)

    # --set FIRST, before anything reads a configuration value. The stages hold references to
    # default_config's dictionaries rather than copies, so applying the overrides here is what
    # makes them take effect everywhere without a flag per setting.
    config.apply_overrides(getattr(args, "overrides", []), log)

    # every checkpoint, history and figure this run writes is named after the configuration,
    # so an ablation never overwrites the run it is compared with
    tag = helpers.set_run_tag(model_cfg_from_args(args))
    for d in (args.model_dir, args.data_dir, CKDIR, PLOTDIR):
        os.makedirs(d, exist_ok=True)
    device = helpers.resolve_device(args.gpu)
    log(f"device={device}  run={tag}  dataset={args.dataset}  "
        f"checkpoints={CKDIR}  plots={PLOTDIR}")
    log("arguments:")
    for _k, _v in sorted(vars(args).items()):
        log(f"    {_k} = {_v}")

    # The preference file is loaded only by the stages entitled to it. A tokenizer that has
    # to be BUILT is the one exception: its vocabulary is shared, so it must see every corpus.
    _saved = train_bpe.peek(BPE_PATH)
    need_bpe_build = len(_saved.merges) < args.bpe_merges if _saved else True
    train_pairs, ev_pairs = ([], [])
    if need_pairs or need_bpe_build:
        train_pairs, ev_pairs = dsets.load_pairs(args.dataset, args.data_dir, args.val_frac,
                                                 args.seed, args.limit)
    texts = [p[k] for p in (train_pairs + ev_pairs) for k in ("prompt", "chosen", "rejected")]
    if need_bpe_build:
        # The tokenizer is trained on EVERY text the pipeline will ever see, so one vocabulary
        # serves every stage. That is two sources, not three:
        #
        #   the pretraining corpus   scanned as documents
        #   the instruction data     ALREADY in `texts` above -- load_pairs extracted every
        #                            prompt, chosen and rejected string from it
        #
        # There is no separate scan of --sft_dir. It points at the rlhf_hh tree, whose files
        # are preference JSON, and json is a corpus extension: scanning it would have read
        # `{"pairs": [{"id": 0, "prompt": ...` as prose and taught the vocabulary merges over
        # JSON punctuation -- while the very same prompts and responses were already in
        # `texts`, properly extracted. One source, counted once, in its parsed form.
        pre, n_pre = dsets.load_pretrain_corpus(
            args.pretrain_dir, config.PRETRAIN["max_words"],
            exclude_dirs=config.PRETRAIN["exclude_dirs"],
            text_column=config.PRETRAIN["text_column"])
        helpers.table("BPE training corpus", [
            ("pretrain dir", args.pretrain_dir or "(unset for this scheme)"),
            ("pretrain excluded", ", ".join(config.PRETRAIN["exclude_dirs"]) or "(none)"),
            ("pretrain corpus files", f"{n_pre:,}"),
            ("pretrain documents", f"{len(pre):,}"),
            ("instruction data", config.INSTRUCT_DIR),
            ("instruction texts", f"{len(texts):,}  (prompt/chosen/rejected of "
                                  f"{len(train_pairs) + len(ev_pairs):,} pairs)"),
            ("merges requested", f"{args.bpe_merges:,}")], out=log)
        texts = texts + [d["chosen"] for d in pre]
    tok = train_bpe.build(texts, BPE_PATH, log,
                          plotdir=plot_dir("bpe"),          # always: the build is watched
                          num_merges=args.bpe_merges,
                          plot_every=args.plot_every_steps,
                          checkpoint_every=args.checkpoint_every_steps,
                          min_freq=config.BPE.get("min_freq", 1))
    if need_pairs and train_pairs:
        helpers.table("dataset", dsets.stats_rows(args.dataset, train_pairs, ev_pairs, tok,
                                                  source=config.INSTRUCT_DIR),
                      out=log)
    # Pre-tokenise the preference pairs into cache/tokens/<tokenizer>/ -- ONLY for the stages that
    # train on them, so a corpus stage never touches the instruction data.
    #
    # The cache is keyed by the directory the pairs CAME FROM, not by config.HH_DIR: --dataset
    # now takes a path, and keying two different folders to one name would let a run be served
    # the other folder's tokens whenever the two happened to agree on pair count.
    if pretokenize_pairs and train_pairs:
        try:
            src = dsets.resolve_root(getattr(args, "dataset", "hh"), args.data_dir)
            helpers.attach_pair_ids(train_pairs, src, tok, split="train", log=log,
                                    resume=args.resume)
            helpers.attach_pair_ids(ev_pairs, src, tok, split="val", log=log,
                                    resume=args.resume)
        except Exception as e:                                    # noqa: BLE001
            log(f"[cache] preference pre-tokenisation skipped: {e}")
    enc = dsets.Encoder(tok, device, args.max_len)
    def new_model(): return build_model(tok, device, args=args)

    def monitor(mode, records, step=None):
        """Live/final dynamics figure for `mode`, written under outputs/plots/<mode>/. Plotting must
        never kill a training run, so failures are logged and swallowed."""
        try:
            viz.method_monitor(mode, records, plot_dir(mode), "")
        except Exception as e:                                    # noqa: BLE001
            log(f"[plot] {mode} failed: {e}")

    rng = random.Random(args.seed)
    # fixed prompts for the end-of-stage samples and the periodic previews. A corpus stage
    # has no preference pairs to draw them from and supplies its own (common.corpus_prompts).
    prompts, pref_prompts = [], []
    if train_pairs:
        prompts = [train_pairs[i]["prompt"]
                   for i in rng.sample(range(len(train_pairs)),
                                       min(N_SAMPLES, len(train_pairs)))]
        pref_prompts = [train_pairs[i]["prompt"]
                        for i in rng.sample(range(len(train_pairs)),
                                            min(N_PREVIEW, len(train_pairs)))]
    sample = _make_sampler(tok, device, args.max_len, log, prompts,
                           getattr(tok, "eos_token_id", None))

    return {"args": args, "enc": enc, "tok": tok, "ckdir": CKDIR, "plotdir": PLOTDIR,
            "device": device, "log": log, "monitor": monitor, "new_model": new_model,
            "sample": sample, "train_pairs": train_pairs, "ev_pairs": ev_pairs,
            "pref_prompts": pref_prompts}


def load_stage_model(ctx, stage, train_mode=False):
    """A fresh model with stage `stage`'s saved weights loaded. Exits with a clear message if
    that checkpoint does not exist yet, since the stages depend on each other's outputs."""
    ck = helpers.load_ckpt(ctx["ckdir"], stage)
    if not ck or "model" not in ck:
        raise SystemExit(f"[{stage}] no checkpoint under {ctx['ckdir']}/{stage}/ -- "
                         f"run the {stage} stage first")
    # THE CHECKPOINT'S OWN ARCHITECTURE WINS: a stage trained at a different width or depth
    # must be rebuilt as it was, or load_state_dict fails (or, worse, silently loads a
    # differently-shaped model).
    saved = ck.get("model_cfg")
    if saved:
        m = build_model(ctx["tok"], ctx["device"], model_cfg=translate_cfg(saved))
    else:
        m = ctx["new_model"]()
    load_growing(m, ck["model"], log=ctx.get("log", print))
    m.train() if train_mode else m.eval()
    return m


def load_growing(model, state, log=print):
    """Load `state` into `model`, GROWING the vocabulary if tokens were registered since.

    Registering a special token appends an id, so a checkpoint trained before it is correct in
    every row it has and simply short by however many were added. Loading it is then a matter
    of receiving the old rows and initialising the new ones -- which is what makes "register a
    token, then fine-tune" a small operation rather than a retrain.

    The state dict is loaded at ITS OWN size and the model grown afterwards, rather than the
    checkpoint being padded: the new rows are then initialised by resize_token_embeddings,
    in one place, instead of by whatever this function happened to choose.

    A checkpoint LARGER than the current vocabulary is refused. That is not a registration --
    it means the tokenizer was rebuilt, so the ids in the checkpoint mean different words, and
    loading it would produce a model that runs and talks nonsense."""
    want = model.tok.weight.shape[0]
    have = state["tok.weight"].shape[0] if "tok.weight" in state else want
    if have == want:
        model.load_state_dict(state)
        return model
    if have > want:
        raise SystemExit(
            f"[model] checkpoint has a vocabulary of {have:,} but the tokenizer now has "
            f"{want:,}.\n"
            f"        Tokens can be REGISTERED (the vocabulary grows and old ids keep their "
            f"meaning) but not removed, so this checkpoint was trained against a DIFFERENT "
            f"vocabulary. Retrain, or restore the tokenizer it was built with.")
    model.resize_token_embeddings(have, allow_shrink=True)   # fresh model, nothing lost
    model.load_state_dict(state)
    model.resize_token_embeddings(want)          # then grow, initialising only the new rows
    log(f"[model] {model._resize_note}; the checkpoint's {have:,} rows are unchanged")
    return model
