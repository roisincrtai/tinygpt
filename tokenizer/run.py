"""
tokenizer/run.py -- stage 2: the byte-level BPE tokenizer every later stage shares.

Trains (or loads, if the checkpoint already exists) the corpus BPE that every later stage
shares, and saves it WITH its full vocabulary and per-merge build history under

    checkpoints/bpe/bpe.json

The training corpus is the pretraining text (every corpus file under the pretraining
corpus -- prose, markdown, code or parquet -- recursive,
minus config.PRETRAIN["exclude_dirs"]) plus the preference texts, so the single tokenizer
covers both. build() is the one place the tokenizer is constructed, so every stage gets a
byte-for-byte identical vocabulary.

    python -m tokenizer.run [--bpe_merges N]
    ./stage2_train_bpe_tokenizer.sh
"""
import os

from .bpe import BPETokenizer
from helpers import visualization as viz


def with_extra_specials(tok):
    """Register default_config.EXTRA_SPECIAL_TOKENS on a tokenizer, and return it.

    Applied on every load and every build, so a token added to the configuration is present
    everywhere without editing the tokenizer. They append AFTER the predefined ones, so adding
    to the list never moves an existing id -- but it does grow the vocabulary, and a model
    trained before the change needs its embedding extended, so the log says what was added."""
    import default_config as config
    extra = list(getattr(config, "EXTRA_SPECIAL_TOKENS", []) or [])
    added = [t for t in extra if t not in tok.special2id]
    for t in added:
        tok.register_special(t)
    if added:
        print(f"[bpe] registered {len(added)} extra special token(s): {', '.join(added)} "
              f"-> vocabulary {len(tok):,}", flush=True)
    return tok


def peek(bpe_path):
    """The saved tokenizer, or None. Lets a caller ask how many merges are already on disk
    WITHOUT assembling a training corpus -- which is how the corpus stages avoid reading
    another stage's data just to find out whether the vocabulary needs building.

    An unreadable or truncated file returns None and is rebuilt. A file written with the OLD
    id layout does not: BPETokenizer.load raises SystemExit, which is not an Exception and so
    passes through here on purpose. Rebuilding silently would leave every checkpoint trained
    against the old ids decoding as gibberish, with nothing said about it."""
    if not os.path.isfile(bpe_path):
        return None
    try:
        return with_extra_specials(BPETokenizer.load(bpe_path))
    except Exception:                                         # noqa: BLE001
        return None


def build(texts, bpe_path, log, plotdir=None, num_merges=8000, plot_every=200,
          checkpoint_every=250):
    """Return the BPE tokenizer, (re)using the checkpoint at `bpe_path` (a stale/unreadable
    file is rebuilt).

    A build is RESUMABLE and WATCHED: merges are checkpointed to `<bpe_path>.partial` every
    `checkpoint_every` merges, so an interrupted build continues where it stopped instead of
    restarting, and bpe_dynamics.pdf is redrawn every `plot_every` merges -- the same cadence
    every trainer plots at -- so the vocabulary can be watched as it is learned.

    THE MERGE BUDGET IS PART OF "DONE", exactly as a training stage's step budget is. A saved
    tokenizer is reused only if it already has at least `num_merges` merges; RAISING the
    budget CONTINUES from the vocabulary on disk rather than reusing the short one or
    restarting from zero."""
    os.makedirs(os.path.dirname(bpe_path), exist_ok=True)
    tok = None
    if os.path.isfile(bpe_path):
        try:
            tok = BPETokenizer.load(bpe_path)
        except Exception:                                     # noqa: BLE001
            tok = None                                        # stale/old format -> rebuild
    if tok is not None and len(tok.merges) > num_merges:
        log(f"BPE: {bpe_path} has {len(tok.merges):,} merges, more than the requested "
            f"{num_merges:,}; reusing it as is (delete the file to train a smaller one, "
            f"but note that every checkpoint is tied to this vocabulary, and the "
            f"token cache moves to a new cache/tokens/bpe_<vocab>_<fingerprint>/)")
    elif tok is None or len(tok.merges) < num_merges:
        if tok is not None:
            log(f"BPE: extending {bpe_path} from {len(tok.merges):,} to {num_merges:,} merges")
        def monitor(history, merge):
            if plotdir and history:
                viz.bpe_monitor(history, plotdir, "")
        tok = BPETokenizer.build(texts, num_merges=num_merges, checkpoint_path=bpe_path,
                                 checkpoint_every=checkpoint_every, log=log,
                                 monitor=monitor if plotdir else None,
                                 plot_every=plot_every,
                                 init_merges=tok.merges if tok else None,
                                 init_history=tok.build_history if tok else None,
                                 init_sig=tok.corpus_sig if tok else None)
        tok.save(bpe_path)
    log(f"BPE tokenizer -> {bpe_path} (vocab {len(tok):,})")
    hist = getattr(tok, "build_history", None)
    if plotdir and hist:
        out = viz.bpe_monitor(hist, plotdir, "")
        if out:
            log(f"[plot] {out} created ({len(hist):,} merges)")
    return tok


def main():
    """Standalone stage 2: build (or reuse) the BPE tokenizer and draw outputs/plots/bpe/bpe_dynamics.pdf."""
    from helpers import common
    args = common.parse_args()
    common.setup(args, need_pairs=True, draw_bpe=True)     # builds/loads the tokenizer and draws bpe_dynamics.pdf


if __name__ == "__main__":
    main()
