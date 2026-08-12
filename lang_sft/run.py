"""
lang_sft/run.py -- stage 11: adapt the pretrained model to a NEW LANGUAGE.

    python -m lang_sft.run [--lang_sft_dataset NAME --lang_sft_short gaelic]
    ./stage11_lang_sft.sh

THREE STEPS, IN THIS ORDER, and the order is the whole design:

    1. THE VOCABULARY. The default BPE is loaded and used to read the new corpus, then
       EXTENDED with merges learned on it (tokenizer_sft/run.py). Every existing id keeps its
       meaning; the new merges are appended.
           -> checkpoints/lang-sft/<dataset>/bpe/bpe.json
    2. THE TOKEN STREAM. The corpus is tokenised WITH THAT EXTENDED VOCABULARY into
       memory-mapped shards, the same way stage 4 prepares the pretraining corpus.
           -> cache/tokens/bpe_<encoding>_<fingerprint>/...
    3. THE MODEL. The pretrained checkpoint is loaded, its embedding rearranged and grown for
       the extended vocabulary, and training continues on the stream with the pipeline's one
       LM loss.
           -> checkpoints/lang-sft/<dataset>/checkpoint_<model>_<pe>_lang-sft-<short>.pt
              checkpoints/lang-sft/<dataset>/history_<model>_<pe>_lang-sft-<short>.json

WHY TOKENISATION IS A STEP OF ITS OWN AND NOT A SIDE EFFECT. It is the expensive, entirely
deterministic part -- hours on a corpus this size, depending on the corpus and the tokenizer
and on nothing else -- and stage 4 exists for exactly that reason. Folding it into the start
of training makes those hours look like training that has hung, pays them again on every
machine, and discovers a failure (a wrong text column, a full disk) only after a GPU has been
allocated. It ALSO cannot be shared here: the stream depends on the extended vocabulary, which
does not exist until step 1 has run, so stage 4 could not have prepared it in advance.

WHY THE VOCABULARY HAS TO COME FIRST. A byte-level BPE never fails on an unseen language -- it
falls back to raw bytes -- so training Irish on the English vocabulary would run, and would
quietly spend four or five tokens on words an adapted vocabulary spells in one. That is a
constant tax on every sequence, on the context window and on the step budget alike. Extending
first is what makes the same number of steps cover several times as much Irish.

WHY IT IS AN EXTENSION AND NOT A NEW VOCABULARY. New ids would mean the embedding rows learned
in stage 5 now sit against different tokens: the pretrained checkpoint could not be used and
this stage would be a retrain rather than an adaptation. See tokenizer_sft/run.py, which also
explains the one thing extension is not free of -- the specials move, and a checkpoint has to
be rearranged for it.

THE ARTEFACTS ARE UNDER THE DATASET (checkpoints/lang-sft/<dataset>/) because a language
adaptation is a property of the corpus it adapted to: two languages must not overwrite each
other's vocabulary or each other's weights.
"""
import os

import default_config as config
import helpers
from helpers import common, lm
from helpers import dataset_helpers as dsets
from tokenizer.bpe import BPETokenizer

import tokenizer_sft

STAGE = "lang_sft"


def dataset_name(args):
    """The corpus this run adapts to -- a name under data/download/, or a path."""
    return getattr(args, "lang_sft_dataset", "") or config.LANG_SFT["dataset"]


def short_name(args):
    """The label that goes in every filename: `gaelic`, not the whole dataset name."""
    return getattr(args, "lang_sft_short", "") or config.LANG_SFT["short"]


def corpus_dir(args):
    """Where the corpus is. A bare name is resolved under data/download/; a path is used as
    given, so a corpus that lives outside the download tree needs no configuration to reach."""
    name = dataset_name(args)
    return name if os.path.isdir(name) else config.dataset_dir(name)


def artefact_dir(ckdir, args):
    """checkpoints/lang-sft/<dataset>/ -- this adaptation's own directory.

    The checkpoint and history land in it directly (STAGE_DIRS maps lang_sft to nothing, so
    stage_dir appends no further level):

        checkpoints/lang-sft/<dataset>/checkpoint_<model>_<pe>_lang-sft-<short>.pt
        checkpoints/lang-sft/<dataset>/history_<model>_<pe>_lang-sft-<short>.json"""
    return os.path.join(ckdir, "lang-sft", dataset_name(args))


def plot_dir(args):
    """outputs/plots/lang-sft/<dataset>/ -- the figures, beside nothing else's."""
    return os.path.join(config.PLOT_DIR, "lang-sft", dataset_name(args))


def build_tokenizer(ctx):
    """Step 1: the extended vocabulary. Returns (tokenizer, path).

    THE CORPUS IS READ WITH THE DEFAULT TOKENIZER'S OWN PRE-TOKENIZER, which is what
    BPETokenizer.build does with any text: the pre-token pattern and the byte alphabet are
    properties of the class, not of a trained vocabulary, so the words counted here are split
    exactly as stage 3 split its own. Only the merges learned from them are new.

    The corpus text is a GENERATOR of documents, not a list: this is a multi-gigabyte corpus
    and the merge counter consumes it once."""
    args, log = ctx["args"], ctx["log"]
    d = corpus_dir(args)
    if not os.path.isdir(d):
        raise SystemExit(
            f"[lang-sft] corpus directory does not exist: {d}\n"
            f"           No stage downloads. Fetch it first:\n"
            f"               ./stage1_download_data.sh --only {dataset_name(args)}")
    out = tokenizer_sft.extended_path(dataset_name(args), ctx["ckdir"])
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def texts():
        """The corpus documents, as text, only if merges actually have to be learned."""
        docs, n_files = dsets.load_pretrain_corpus(
            d, config.PRETRAIN["max_words"],
            exclude_dirs=config.PRETRAIN["exclude_dirs"],
            text_column=config.PRETRAIN["text_column"])
        log(f"[lang-sft] vocabulary corpus: {n_files:,} files, {len(docs):,} documents")
        return [r["chosen"] for r in docs]

    tok = tokenizer_sft.extend(config.BPE_PATH, out, LazyTexts(texts),
                         getattr(args, "lang_sft_merges", 0)
                         or config.LANG_SFT["extra_merges"], log=log,
                         plotdir=plot_dir(args),
                         plot_every=args.plot_every_steps,
                         checkpoint_every=args.checkpoint_every_steps,
                         min_freq=config.BPE.get("min_freq", 1))
    return tok, out


class LazyTexts:
    """An iterable that only materialises when it is actually iterated.

    BPETokenizer.build takes an iterable of strings and, on a resumed or already-finished
    extension, never touches it. Scanning a multi-gigabyte corpus to hand over a list that is
    then discarded is minutes of work for nothing, so the scan is deferred to the first
    iteration -- which is also the only one, since build() materialises it once."""

    def __init__(self, factory):
        self._factory, self._cache = factory, None

    def __iter__(self):
        if self._cache is None:
            self._cache = self._factory()
        return iter(self._cache)

    def __len__(self):
        if self._cache is None:
            self._cache = self._factory()
        return len(self._cache)


def tokenize_corpus(ctx, tok, bpe_path):
    """Step 2: the corpus as a memory-mapped token stream, built with the EXTENDED vocabulary.

    This is stage 4's machinery (helpers.build_token_stream, through load_token_corpus) pointed
    at this stage's corpus and this stage's tokenizer. It is a step here rather than a side
    effect of training for the reason the module docstring gives, and it cannot have been done
    by stage 4 in advance: the vocabulary it tokenises against did not exist until step 1.

    THE CACHE DIRECTORY IS NAMED AFTER THE TOKENIZER (bpe_<encoding>_<fingerprint>), so the
    extended vocabulary gets its own streams rather than colliding with the English ones. That
    is not a convention this stage has to observe -- it falls out of how the cache is keyed --
    but it is the reason two vocabularies can share one corpus directory safely.

    RESUMABLE AND REUSED, like every other stream: an existing one whose signature still
    matches is left alone, and an interrupted build continues from its last shard."""
    args, log = ctx["args"], ctx["log"]
    corpus, n_files, n_tok = helpers.load_token_corpus(
        STAGE, corpus_dir(args), tok, config.PRETRAIN["max_words"],
        exclude_dirs=config.PRETRAIN["exclude_dirs"], log=log,
        text_column=config.PRETRAIN["text_column"])
    if corpus is None:
        raise SystemExit(
            f"[lang-sft] no corpus files under {corpus_dir(args)}\n"
            f"           Fetch it first:  ./stage1_download_data.sh --only "
            f"{dataset_name(args)}")
    seen = int(args.lang_sft_steps) * int(args.batch) * int(args.max_len)
    helpers.table("language adaptation corpus", [
        ("directory", corpus_dir(args)),
        ("corpus files", helpers.count(n_files)),
        ("documents", helpers.count(len(corpus))),
        ("TOTAL TOKENS", helpers.count(n_tok)),
        ("token stream", corpus.path),
        ("stream size", f"{corpus.nbytes / 1048576:.1f} MB {corpus.dtype}, "
                        f"memory-mapped (not loaded)"),
        ("tokenised with", f"{bpe_path} (vocab {len(tok):,})"),
        # derived, not asserted: what the step budget actually buys over THIS corpus, so a
        # mismatch between the two is noticed at the start of the run rather than at the end
        ("budget", f"{args.lang_sft_steps:,} steps = {helpers.human(seen)} tokens"),
        ("epochs over this corpus", f"{seen / max(n_tok, 1):.2f}"),
    ], out=log)
    return corpus, n_files, n_tok


def load_init_model(ctx, tok, log):
    """Step 2's starting point: the PRETRAINED model, rearranged for the extended vocabulary.

    THE REARRANGEMENT IS THE WHOLE POINT OF THIS FUNCTION. Growing an embedding appends rows;
    that is correct when ids were appended, and this vocabulary appended MERGES, which sit
    before the specials. So the specials moved, and the trained <|endoftext|> row has to move
    with them or the model loses the token every document ends with -- silently, since nothing
    about a misplaced embedding row fails.

    common.load_stage_model cannot do this: it builds the model at the checkpoint's own
    vocabulary size and loads it unchanged, which is right for every stage whose tokenizer did
    not move underneath it."""
    ck = helpers.load_ckpt(ctx["ckdir"], "pretrain")
    if not ck or "model" not in ck:
        raise SystemExit(
            f"[lang-sft] no pretrain checkpoint under {ctx['ckdir']}/pretrain/ -- "
            f"this stage adapts a pretrained model, so run stage 5 first")
    # THE TOKENIZER THE CHECKPOINT WAS TRAINED WITH, from inside the checkpoint, falling back
    # to checkpoints/bpe/bpe.json. Reading the base vocabulary from the file on disk would
    # assume the two match; the checkpoint knows.
    old = helpers.bpe_from_ckpt(ck, fallback=config.BPE_PATH, log=log)
    if old is None:
        raise SystemExit("[lang-sft] the pretrain checkpoint carries no tokenizer and "
                         f"{config.BPE_PATH} does not exist: cannot tell what its ids mean")
    state, n_new = tokenizer_sft.remap_specials(ck["model"], old, tok, log=log)
    saved = ck.get("model_cfg")
    model = common.build_model(ctx["tok"], ctx["device"],
                               model_cfg=common.translate_cfg(saved) if saved else None)
    model.resize_token_embeddings(len(tok), allow_shrink=True)
    model.load_state_dict(state)
    model.train()
    helpers.table("language adaptation", [
        ("corpus", os.path.basename(str(corpus_dir(ctx["args"])).rstrip("/"))),
        ("short name", short_name(ctx["args"])),
        ("base vocabulary", f"{len(old):,}"),
        ("extended vocabulary", f"{len(tok):,}  (+{len(tok) - len(old):,})"),
        ("initialised from", helpers.ckpt_path(ctx["ckdir"], "pretrain")),
        ("embedding rows moved", f"{old.n_special} specials, {old.n_base:,} -> {tok.n_base:,}"),
        ("embedding rows added", helpers.count(n_new)),
        ("trains", "every parameter"),
    ], out=log)
    return model


def run(ctx):
    """The whole of stage 11: the vocabulary, then the model."""
    args, log = ctx["args"], ctx["log"]
    short = short_name(args)

    log("")
    log("=" * 78)
    log(f"STAGE 11, STEP 1 of 3: extend the vocabulary  -> "
        f"{tokenizer_sft.extended_path(dataset_name(args), ctx['ckdir'])}")
    log(f"STAGE 11, STEP 2 of 3: tokenise the corpus    -> cache/tokens/ "
        f"(with the extended vocabulary)")
    log(f"STAGE 11, STEP 3 of 3: continue pretraining   -> "
        f"{helpers.ckpt_path(artefact_dir(ctx['ckdir'], args), STAGE)}")
    log("=" * 78)
    log("")

    log("--- step 1/3: extend the vocabulary ---")
    tok, bpe_path = build_tokenizer(ctx)
    # EVERY CHECKPOINT FROM HERE CARRIES THE EXTENDED VOCABULARY, not the English one. A
    # lang-sft checkpoint holding the base bpe.json would decode its own output wrongly, which
    # is precisely the failure the embedded tokenizer exists to prevent.
    helpers.set_bpe_source(bpe_path)
    ctx["tok"] = tok
    ctx["enc"] = dsets.Encoder(tok, ctx["device"], args.max_len)

    log("--- step 2/3: tokenise the corpus with the extended vocabulary ---")
    corpus, n_files, n_tok = tokenize_corpus(ctx, tok, bpe_path)

    log("--- step 3/3: adapt the pretrained model ---")
    model = load_init_model(ctx, tok, log)
    log(f"=== LANG SFT ({short}: LM on {len(corpus):,} documents, from pretrain) ===")
    preview = common.make_preview(tok, ctx["device"], args.max_len, log,
                                  common.corpus_prompts(corpus, seed=args.seed, tok=tok))
    # the figure goes beside the checkpoint, under the dataset, for the same reason they do
    def monitor(mode, records, step=None):
        try:
            from helpers import visualization as viz
            viz.method_monitor(mode, records, plot_dir(args), "")
        except Exception as e:                                    # noqa: BLE001
            log(f"[plot] {mode} failed: {e}")

    model = lm.train(model, ctx["enc"], corpus, artefact_dir(ctx["ckdir"], args), args, log,
                     monitor, stage=STAGE, steps=args.lang_sft_steps,
                     lr=args.lang_sft_lr, preview=preview,
                     ssm_stats_every=args.ssm_stats_every)
    preview(model, STAGE, "final")
    return model


def main():
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)      # this stage reads its own corpus only
    # THE STAGE LABEL CARRIES THE LANGUAGE, so two adaptations of one model are told apart by
    # their filenames rather than only by their directories: lang-sft-gaelic, lang-sft-welsh.
    helpers.STAGE_LABELS[STAGE] = f"lang-sft-{short_name(args)}"
    run(ctx)


if __name__ == "__main__":
    main()
