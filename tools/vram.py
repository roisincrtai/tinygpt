"""
tools/vram.py -- how much memory one training step actually takes.

    python -m tools.vram                          the configured scheme, batch and context
    python -m tools.vram --batch 8 --context 1024
    python -m tools.vram --sweep                  batches 1,2,4,8,16,32 until it runs out

MEASURED, not estimated. It builds the real model, runs one real forward and backward with
the real optimiser, and reports the peak the allocator saw. An arithmetic estimate is useful
for reasoning but always wrong in detail: it cannot know which temporaries autograd keeps,
what the allocator rounds up, or what the kernels ask for -- and being wrong by 30% is the
difference between fitting and not fitting.

WHERE THE MEMORY GOES, in the order that matters for this model:

    the loss     B x T x vocab, three times over (logits, log-softmax, its gradient). At
                 batch 16 and context 512 that is 4.6 GB for a 61M model -- more than the
                 weights, the optimiser and every activation combined. Vocabulary-sized
                 tensors are the reason small models still want large cards.
    attention    B x heads x T x T per layer, kept for the backward pass. Quadratic in the
                 context, so doubling it costs four times this term.
    optimiser    five copies of the parameters: live, gradient, fp32 master, and Adam's two
                 moments.

The first is what --micro_batch attacks, and it is the cheapest thing to change.
"""
import argparse
import sys


def _dev(name):
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def peak_bytes(device):
    import torch
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated()
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    return 0


def reset_peak(device):
    import torch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def measure(scheme, batch, context, device, log=print):
    """Peak bytes for one full training step at this size. Returns None if it does not fit."""
    import torch
    import torch.nn.functional as F
    import default_config as config
    from helpers import MasterAdamW
    from model import ZetaGPT

    cfg = dict(config.MODEL)
    cfg.update(config.scheme(scheme))
    cfg["block_size"] = context
    cfg.pop("vocab_size", None)
    vocab = 256 + config.BPE["num_merges"] + len(config.EXTRA_SPECIAL_TOKENS) + 3

    reset_peak(device)
    try:
        model = ZetaGPT(vocab_size=vocab, **cfg).to(device)
        model.train()
        opt = MasterAdamW(model.parameters(), lr=1e-4)
        params = sum(p.numel() for p in model.parameters())
        ids = torch.randint(0, vocab, (batch, context), device=device)
        # exactly what helpers/lm.py does per step
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        logp = F.log_softmax(logits, -1)
        lp = logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        loss = (-lp).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        peak = peak_bytes(device)
        del model, opt, ids, logits, logp, lp, loss
        reset_peak(device)
        return peak, params
    except RuntimeError as e:                                  # noqa: BLE001
        if "out of memory" not in str(e).lower():
            raise
        reset_peak(device)
        return None, 0


def main(argv=None):
    import default_config as config
    ap = argparse.ArgumentParser(description="peak memory of one training step")
    ap.add_argument("--model_scheme", default=config.PRETRAIN["model_scheme"],
                    choices=list(config.SCHEMES))
    ap.add_argument("--batch", type=int, default=config.TRAIN["batch"])
    ap.add_argument("--context", type=int, default=0, help="0 = the scheme's own")
    ap.add_argument("--gpu", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--sweep", action="store_true", help="try a ladder of batch sizes")
    a = ap.parse_args(argv)

    import helpers
    device = _dev(a.gpu)
    ctx = a.context or config.context_window(a.model_scheme)   # the LONGEST window
    if device.type == "cpu":
        print("[vram] no GPU here: on CPU the allocator reports nothing, so this measures\n"
              "       nothing. Run it on the machine you intend to train on.")
        return 1

    print(f"[vram] {a.model_scheme}  context {ctx}  device {device}")
    sizes = [1, 2, 4, 8, 16, 32, 64] if a.sweep else [a.batch]
    rows = []
    for b in sizes:
        peak, params = measure(a.model_scheme, b, ctx, device)
        if peak is None:
            rows.append((f"batch {b}", "out of memory"))
            print(f"    batch {b:<4} out of memory")
            break
        rows.append((f"batch {b}", f"{peak / 1024**3:.2f} GB   "
                                   f"({peak / b / 1024**2:.0f} MB per sequence)"))
        print(f"    batch {b:<4} {peak / 1024**3:6.2f} GB   "
              f"{peak / b / 1024**2:6.0f} MB per sequence", flush=True)
    helpers.table(f"peak memory, one training step, {a.model_scheme} @ {ctx}", rows)
    print("  The vocabulary-sized tensors (logits, log-softmax, their gradient) are usually\n"
          "  the largest single term. --micro_batch splits the step and cuts them\n"
          "  proportionally; the optimiser state does not move.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
