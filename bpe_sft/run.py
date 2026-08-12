"""
bpe_sft/run.py -- vocabulary adaptation: EXTEND a trained BPE with merges for a new corpus.

    extend(base_path, out_path, texts, extra_merges, log)  -> the extended tokenizer
    extended_path(dataset)                                  where it is written
    remap_specials(state, old_tok, new_tok, log)            a checkpoint's rows, rearranged

    python -m bpe_sft.run --lang_sft_dataset <name> [--lang_sft_merges N]

WHY EXTEND RATHER THAN RETRAIN. A vocabulary trained on English tokenises Irish badly -- not
wrongly, a byte-level BPE can always fall back to raw bytes, but expensively, splitting common
Irish words into four or five tokens where an Irish vocabulary would use one. The fix is more
merges, not different ones. Training a FRESH vocabulary would tokenise Irish beautifully and
throw away the pretrained model with it: new ids mean the embedding rows learned in stage 5
now sit against different tokens, so the checkpoint could not be used and the model would have
to be trained from scratch.

So the English merges are kept, in their order, and Irish merges are appended after them. The
ids already learned keep their meaning and the model only needs more rows -- which is exactly
what `resize_token_embeddings` is for.

------------------------------------------------------------------------------------------
THE SPECIALS MOVE, AND A CHECKPOINT MUST BE REARRANGED FOR IT. This is the one thing about
this that is not free, and it is silent when it is wrong.

The id layout is bytes, then merges, then THE SPECIALS LAST (tokenizer/bpe.py). Appending
merges therefore pushes the specials up by exactly the number appended:

    before:  [0..255 bytes][256..50,255 merges][50,256 eos][50,257 pad][50,258 unk]
    after:   [0..255 bytes][256..50,255 merges][50,256..N new merges][N+1 eos][N+2 pad][N+3 unk]

Every byte and every original merge keeps its id -- verified, not assumed. The three specials
do not. Growing the embedding naively would leave the trained <|endoftext|> row sitting at an
id that is now an Irish merge, and the new <|endoftext|> randomly initialised: a model that
runs, trains, and has lost the token every document ends with. `remap_specials` moves those
rows to where they now belong, and reports the move.
------------------------------------------------------------------------------------------
"""
import os

import default_config as config
import helpers
from helpers import visualization as viz
from tokenizer.bpe import BPETokenizer
from tokenizer.run import with_extra_specials

STAGE = "bpe_sft"


def extended_path(dataset, ckdir=None):
    """checkpoints/lang-sft/<dataset>/bpe/bpe.json -- the adapted vocabulary's home.

    UNDER THE DATASET, because a vocabulary is a property of the corpus it was extended for.
    Two languages adapted from the same base must not overwrite each other, and a path that
    names the dataset makes that impossible rather than merely unlikely."""
    root = ckdir or config.CHECKPOINT_DIR
    return os.path.join(root, "lang-sft", dataset, "bpe", "bpe.json")


def extend(base_path, out_path, texts, extra_merges, log=print, plotdir=None,
           plot_every=200, checkpoint_every=250, min_freq=1):
    """The base tokenizer plus `extra_merges` merges learned on `texts`. Returns it, saved.

    RESUMABLE AND REUSED, on the same rule stage 3 uses: a saved extension already carrying at
    least the requested number of merges is loaded as it is, and a shorter one is CONTINUED
    rather than restarted. The merge budget is part of "done".

    `texts` is only consulted when merges actually have to be learned, so a resumed run does
    not pay to assemble the corpus it is not going to read.

    The base is loaded first and its merges are handed to the builder as the starting point,
    which is what makes the result an extension rather than a new vocabulary: builder ranks
    0..len(base.merges)-1 are the base's own, in order, and everything after them is new."""
    base = BPETokenizer.load(base_path)
    target = len(base.merges) + int(extra_merges)

    have = None
    if os.path.isfile(out_path):
        try:
            have = BPETokenizer.load(out_path)
        except Exception:                                      # noqa: BLE001
            have = None                                        # stale/old layout -> rebuild
    if have is not None and have.merges[:len(base.merges)] != base.merges:
        raise SystemExit(
            f"[bpe-sft] {out_path} does not start with the base vocabulary at {base_path}.\n"
            f"          It was extended from a DIFFERENT base, so its ids do not mean what\n"
            f"          this run's checkpoints expect. Delete it, or point --lang_sft_dataset\n"
            f"          somewhere else.")
    if have is not None and len(have.merges) >= target:
        log(f"[bpe-sft] reusing {out_path}: {len(have.merges):,} merges "
            f"({len(have.merges) - len(base.merges):,} beyond the base), vocab {len(have):,}")
        return with_extra_specials(have)

    start = have if have is not None else base
    log(f"[bpe-sft] extending {base_path} ({len(base.merges):,} merges) by "
        f"{extra_merges:,} -> {target:,}, from {len(start.merges):,}")

    def monitor(history, merge):
        if plotdir and history:
            viz.bpe_monitor(history, plotdir, "")

    tok = BPETokenizer.build(texts, num_merges=target, checkpoint_path=out_path,
                             checkpoint_every=checkpoint_every, log=log, min_freq=min_freq,
                             monitor=monitor if plotdir else None, plot_every=plot_every,
                             init_merges=start.merges,
                             init_history=start.build_history,
                             init_sig=start.corpus_sig)
    tok = with_extra_specials(tok)
    tok.save(out_path)

    # THE INVARIANT, CHECKED RATHER THAN TRUSTED. Everything downstream -- the pretrained
    # embedding, every cached token stream, every checkpoint -- assumes the base ids survived.
    # It costs one pass over the merge list to know instead of assume.
    if tok.merges[:len(base.merges)] != base.merges:
        raise SystemExit(
            "[bpe-sft] the extension did not preserve the base merges. Refusing to save a "
            "vocabulary whose ids do not mean what the pretrained checkpoint learned.")
    helpers.table("extended vocabulary", [
        ("base", base_path),
        ("base merges", helpers.count(len(base.merges))),
        ("merges added", helpers.count(len(tok.merges) - len(base.merges))),
        ("vocabulary", f"{len(base):,} -> {len(tok):,}"),
        ("base ids preserved", "yes: every byte and every base merge keeps its id"),
        ("specials moved", f"{base.eos_token_id} -> {tok.eos_token_id} (eos), and pad/unk "
                           f"with it -- the layout puts specials last"),
        ("written to", out_path)], out=log)
    return tok


def remap_specials(state, old_tok, new_tok, log=print):
    """A state dict trained on `old_tok`, rearranged for `new_tok`'s id layout.

    Returns (state, n_new_rows). Only the embedding is touched -- the output head is TIED to
    it, so moving a row moves both -- and only the special rows move, because everything below
    them keeps its id.

    Called BEFORE the model is resized: the rows are placed at their new indices in a matrix
    that is about to grow, so `resize_token_embeddings` then initialises exactly the rows
    nothing was moved into (the new merges).

    A no-op when the specials did not move, which is the case for every ordinary stage."""
    import torch
    shift = new_tok.n_base - old_tok.n_base
    if shift <= 0 or "tok.weight" not in state:
        return state, 0
    w = state["tok.weight"]
    n_special = min(old_tok.n_special, new_tok.n_special)
    grown = torch.empty(len(new_tok), w.shape[1], dtype=w.dtype)
    # the new rows are filled with the mean of the old ones, the same place
    # resize_token_embeddings puts a token of no particular meaning yet
    grown[:] = w.mean(0, keepdim=True)
    grown[:old_tok.n_base] = w[:old_tok.n_base]                       # bytes + base merges
    grown[new_tok.n_base:new_tok.n_base + n_special] = \
        w[old_tok.n_base:old_tok.n_base + n_special]                  # the specials, moved
    state = dict(state)
    state["tok.weight"] = grown
    if "head.weight" in state:            # tied, but written out; keep the two consistent
        state["head.weight"] = grown
    log(f"[bpe-sft] embedding remapped for the extended vocabulary: "
        f"{old_tok.n_base:,} base rows kept in place, {n_special} special rows moved "
        f"{old_tok.n_base:,} -> {new_tok.n_base:,}, {shift:,} new rows initialised at the mean")
    return state, shift


def main():
    """Standalone: extend the vocabulary and stop, without training anything on it."""
    from helpers import common
    from lang_sft import run as lang
    args = common.parse_args()
    ctx = common.setup(args, need_pairs=False)
    lang.build_tokenizer(ctx)


if __name__ == "__main__":
    main()
