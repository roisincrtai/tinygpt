"""
scaling_laws/run.py -- entry point: sweep the (model size, data size) grid on WikiText-103,
fit L(N, D), and draw it.

    python -m scaling_laws.run
    python -m scaling_laws.run --models zs/8,zs/6,zs/4 --budgets 2e6,8e6
    python -m scaling_laws.run --fit_only          # re-fit and redraw from the saved results
    ./run_scaling_laws.sh

    outputs/eval/scaling_laws.json          every point, its history, and the fit
    outputs/plots/scaling_laws/scaling_laws.pdf

RESUMABLE BY CONSTRUCTION. Each point is written to the results file as soon as it finishes, so
a sweep that is interrupted -- and a twenty-point sweep will be -- picks up where it stopped
rather than starting again. --refit forces a point to be retrained even if it is present.

The tokenizer is the pipeline's own (checkpoints/bpe/bpe.json), so the loss is in the same
units as every other number this project reports. Stage 1 must have been run.
"""
import argparse
import json
import os

import torch

import default_config as config
from helpers import resolve_device
from tokenizer import BPETokenizer

from . import fit as fitting
from . import grid as gridmod
from . import train as trainer
from . import wikitext

DEFAULT_JSON = os.path.join(config.OUTPUT_DIR, "eval", "scaling_laws.json")
DEFAULT_PDF = os.path.join(config.PLOT_DIR, "scaling_laws", "scaling_laws.pdf")


def parse_args(argv=None):
    s = config.SCALING
    ap = argparse.ArgumentParser(description="scaling laws for model size and data size")
    ap.add_argument("--models", default=",".join(s["models"]),
                    help="comma-separated ladder rungs; see scaling_laws.grid.LADDER")
    ap.add_argument("--budgets", default=",".join(str(d) for d in s["budgets"]),
                    help="comma-separated token budgets; any larger than the corpus is DROPPED "
                         "rather than clipped, since two budgets clipped to the same size are "
                         "one experiment run twice")
    ap.add_argument("--context", type=int, default=s["context"],
                    help="context window, held FIXED across the grid: this study varies size "
                         "and data, and a third moving axis would be fitted as if it were one "
                         "of them")
    ap.add_argument("--batch", type=int, default=s["batch"])
    ap.add_argument("--lr", type=float, default=s["lr"],
                    help="peak learning rate AT --lr_ref_width; other widths follow --lr_rule")
    ap.add_argument("--lr_ref_width", type=int, default=s["lr_ref_width"])
    ap.add_argument("--lr_rule", default=s["lr_rule"], choices=["sqrt_width", "fixed"])
    ap.add_argument("--lr_schedule", default=config.TRAIN["lr_schedule"],
                    choices=["cosine", "constant"])
    ap.add_argument("--lr_min_factor", type=float, default=config.TRAIN["lr_min_factor"])
    ap.add_argument("--eval_every", type=int, default=s["eval_every"],
                    help="held-out probe every N steps; 0 = ten times per point")
    ap.add_argument("--val_windows", type=int, default=s["val_windows"],
                    help="fixed validation windows, identical for every point")
    ap.add_argument("--pe", default=config.MODEL["pe"], choices=["ssm", "rope"],
                    help="the whole grid uses one source of position; run it twice to compare")
    ap.add_argument("--tokenize_workers", type=int, default=0, help="0 = cpu_count - 1")
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default=config.TRAIN["gpu"])
    ap.add_argument("--seed", type=int, default=config.TRAIN["seed"])
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default=DEFAULT_PDF)
    ap.add_argument("--refit", action="store_true", help="retrain points already in the results")
    ap.add_argument("--fit_only", action="store_true",
                    help="fit and redraw from the saved results; trains nothing")
    ap.add_argument("--dry_run", action="store_true", help="print the grid and stop")
    return ap.parse_args(argv)


def load_results(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            saved = json.load(fh)
        return {r["id"]: r for r in saved.get("points", [])}
    except Exception:                                              # noqa: BLE001
        return {}


def save(path, args, done, params):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = {"corpus": "wikitext-103-raw-v1", "pe": args.pe, "context": args.context,
            "batch": args.batch, "lr": args.lr, "lr_rule": args.lr_rule,
            "lr_ref_width": args.lr_ref_width, "seed": args.seed,
            "points": sorted(done.values(), key=lambda r: (r["N"], r["D"])),
            "fit": params}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(body, fh, indent=1)
    os.replace(tmp, path)                    # atomic: an interrupted write must not lose the run


def main(argv=None):
    args = parse_args(argv)
    def log(m): print(m, flush=True)

    done = load_results(args.json)

    if args.fit_only:
        records = list(done.values())
        if not records:
            raise SystemExit(f"--fit_only: no results at {args.json}")
        params = fitting.fit(records)
        fitting.summarise(records, params, log)
        save(args.json, args, done, params)
        log(f"[scaling] [figure] {fitting.figure(records, params, args.out)}")
        return

    if not os.path.isfile(config.BPE_PATH):
        raise SystemExit(f"no tokenizer at {config.BPE_PATH} -- run stage 2 first")
    tok = BPETokenizer.load(config.BPE_PATH)

    models = gridmod.ladder([m for m in args.models.split(",") if m.strip()])
    if args.dry_run:
        gridmod.describe(models, gridmod.budgets(args.budgets), args.batch, args.context,
                         len(tok), log)
        return

    device = resolve_device(args.gpu)
    log(f"[scaling] device={device}  pe={args.pe}  vocab={len(tok):,}  "
        f"context={args.context}  batch={args.batch}")
    train_ids, valid_ids = wikitext.tokens(tok, args.tokenize_workers, log)

    # numpy, not a python array: the sampler slices it once per window and torch reads the
    # buffer directly, which at 100M tokens is the difference between a copy per step and none
    import numpy as np
    train_np = np.frombuffer(train_ids, dtype=np.uint32)
    valid_np = np.frombuffer(valid_ids, dtype=np.uint32)

    budgets = gridmod.budgets(args.budgets, len(train_np))
    dropped = [d for d in gridmod.budgets(args.budgets) if d not in budgets]
    if dropped:
        log(f"[scaling] dropped budget(s) larger than the corpus ({len(train_np):,} tokens): "
            + ", ".join(f"{d / 1e6:.0f}M" for d in dropped))
    if not budgets:
        raise SystemExit("every requested budget exceeds the corpus")

    points = gridmod.describe(models, budgets, args.batch, args.context, len(tok), log)
    torch.manual_seed(args.seed)

    for i, point in enumerate(points, 1):
        if point["id"] in done and not args.refit:
            r = done[point["id"]]
            log(f"[scaling] ({i}/{len(points)}) {point['id']:>18}  already measured, "
                f"val loss {r['val_loss']:.4f}  (--refit to redo)")
            continue
        log(f"[scaling] ({i}/{len(points)}) ---------------------------------------------")
        done[point["id"]] = trainer.run_point(point, (train_np, valid_np), tok, args,
                                             device, log)
        save(args.json, args, done, None)      # after EVERY point: a sweep gets interrupted

    records = list(done.values())
    params = fitting.fit(records)
    fitting.summarise(records, params, log)
    save(args.json, args, done, params)
    log(f"[scaling] [results] {args.json}")
    log(f"[scaling] [figure] {fitting.figure(records, params, args.out)}")


if __name__ == "__main__":
    main()
