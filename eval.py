"""
eval.py -- held-out evaluation of a trained stage against the SFT reference.

Reads checkpoints/<stage>/checkpoint_<model>_<pe>_<stage-label>.pt and reports, per stage:
    1. teacher-forced preference accuracy, margin and length-shortcut correlation
    2. per-token likelihood displacement of the chosen and rejected responses
    3. the rollout exposure-bias curve (mu_M and reversal rate as the self-sampled
       fraction goes 0 -> 1)

Results go to results/eval_<dataset>_<stage>.json and figures to outputs/plots/eval/.

    python eval.py --stages dpo
    python eval.py --stages dpo --plot_only     # redraw from the saved JSON
"""
import os
import json
import argparse

import torch
import torch.nn.functional as F

import default_config as config
from helpers import dataset_helpers as dsets
from helpers import visualization as viz
from tokenizer import BPETokenizer
from model import ZetaGPT
from helpers import ckpt_path, resolve_device, progress, run_tag, rollout_curve, reward_hist
from instruct_dpo.dpo import seq_logp, margin as dpo_margin


def parse_args():
    ap = argparse.ArgumentParser(description="evaluate trained stages; plots -> outputs/plots/eval/")
    ap.add_argument("--dataset", default="hh",
                    help="\"hh\", or a path to your own folder of preference records")
    ap.add_argument("--stages", default="dpo",
                    help="comma-separated stages to evaluate; each is read from "
                         "checkpoints/<stage>/checkpoint_<model>_<pe>_<stage-label>.pt")
    ap.add_argument("--model_dir", default="saved_models")
    ap.add_argument("--data_dir", default=config.DOWNLOAD_DIR)
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--plot_dir", default="plots")
    ap.add_argument("--result_dir", default="results",
                    help="dir for per-method eval JSON: results/eval_<llm>_<dataset>_<method>.json")
    ap.add_argument("--gpu", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0, help="cap #examples loaded (0 = full data)")
    ap.add_argument("--n_roll", type=int, default=64)
    ap.add_argument("--roll_tokens", type=int, default=48)
    ap.add_argument("--n_hist", type=int, default=256, help="val pairs for reward/margin histograms")
    ap.add_argument("--p_grid", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot_only", action="store_true",
                    help="skip models/eval; (re)draw exposure_bias/bounds from the saved per-method "
                         "results/eval_<llm>_<dataset>_<stage>.json (all --stages, no checkpoints)")
    return ap.parse_args()


@torch.no_grad()
def tf_diagnostics(policy, ref, enc, tok, pairs, beta, batch=16):
    margins, len_diffs = [], []
    # ntok_w/ntok_l count RESPONSE TOKENS, not sequences. `ntok += W[0].shape[0]` counted
    # sequences, so dlp_w/dlp_l were per-SEQUENCE despite the name and despite being the axes
    # of the likelihood-displacement panel -- inflated by the mean response length, and by a
    # different factor for the chosen and rejected branches whenever their lengths differ,
    # which is exactly the asymmetry that panel exists to show.
    dlp_w = dlp_l = 0.0
    ntok_w = ntok_l = 0.0
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i + batch]
        W = enc.encode(chunk, "chosen"); L = enc.encode(chunk, "rejected")
        margins.append(dpo_margin(policy, ref, W, L, beta))
        for p in chunk:
            lc = len(tok(" " + p["chosen"], add_special_tokens=False)["input_ids"])
            lr = len(tok(" " + p["rejected"], add_special_tokens=False)["input_ids"])
            len_diffs.append(lc - lr)
        dlp_w += (seq_logp(policy, *W) - seq_logp(ref, *W)).sum().item()
        dlp_l += (seq_logp(policy, *L) - seq_logp(ref, *L)).sum().item()
        # W = (ids, attn, rmask); seq_logp sums over rmask[:, 1:], so that is the token count
        # matching the numerator exactly.
        ntok_w += W[2][:, 1:].sum().item()
        ntok_l += L[2][:, 1:].sum().item()
    M = torch.cat(margins).float().cpu(); ld = torch.tensor(len_diffs, dtype=torch.float)  # CPU (MPS/CUDA-safe)
    corr = torch.corrcoef(torch.stack([M, ld]))[0, 1].item() if M.numel() > 1 else float("nan")
    return {"acc": (M > 0).float().mean().item(), "mu_M": M.mean().item(),
            "var_M": M.var(unbiased=False).item(), "len_corr": corr,
            "dlp_w": dlp_w / max(ntok_w, 1.0), "dlp_l": dlp_l / max(ntok_l, 1.0)}


def main():
    args = parse_args()
    if args.max_len == 0:
        args.max_len = config.MODEL["block_size"]
    device = resolve_device(args.gpu)
    methods = [m for m in args.stages.split(",")]
    p_grid = [float(x) for x in args.p_grid.split(",")]
    tag = run_tag("zetagpt", args.dataset)
    ckdir = config.CHECKPOINT_DIR                      # checkpoints/<stage>/checkpoint_*.pt
    plotdir = os.path.join(config.PLOT_DIR, "eval")    # figures under outputs/plots/eval/
    os.makedirs(plotdir, exist_ok=True)

    # ---- plot-only: (re)draw from the saved per-method eval JSON, no models/checkpoints ---- #
    if args.plot_only:
        RES = {"tag": tag, "beta": args.beta, "results": {}}
        for mode in methods:
            jp = os.path.join(args.result_dir, f"eval_{args.dataset}_{mode}.json")
            if os.path.isfile(jp):
                RES["results"][mode] = json.load(open(jp))
            else:
                print(f"[eval] plot-only: {jp} not found; skipping {mode}")
        for o in viz.eval_figure(RES, methods, p_grid, plotdir, tag):
            print(f"[eval] [plot] {o} created", flush=True)
        return

    _, ev = dsets.load_pairs(args.dataset, args.data_dir, args.val_frac, args.seed, args.limit)
    if os.path.isfile(config.BPE_PATH):
        tok = BPETokenizer.load(config.BPE_PATH)
    else:
        texts = [p[k] for p in ev for k in ("prompt", "chosen", "rejected")]
        tok = BPETokenizer.build(texts, num_merges=config.BPE["num_merges"])
    enc = dsets.Encoder(tok, device, args.max_len)

    def load_policy(stage):
        p = ckpt_path(ckdir, stage)          # checkpoints/<stage>/checkpoint_*_<label>.pt
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing {p}; run the {stage} stage first")
        ck = torch.load(p, map_location="cpu")
        # the checkpoint's own architecture, not whatever default_config.py says today
        cfg = {k: v for k, v in (ck.get("model_cfg") or config.MODEL).items()
               if k != "vocab_size"}
        m = ZetaGPT(vocab_size=len(tok), **cfg).to(device)
        m.load_state_dict(ck["model"])
        return m.eval()

    print(f"[eval] tag={tag} device={device} loading checkpoints from {ckdir}/ ...")
    ref = load_policy("sft")
    for p in ref.parameters():
        p.requires_grad_(False)

    os.makedirs(args.result_dir, exist_ok=True)
    roll = ev[:args.n_roll]
    RES = {"tag": tag, "beta": args.beta, "results": {}}
    for mode in methods:
        pol = load_policy(mode)
        print(f"[eval] {mode}: teacher-forced diagnostics + histograms ...")
        diag = tf_diagnostics(pol, ref, enc, tok, ev, args.beta)
        hist = reward_hist(pol, ref, enc, ev[:args.n_hist], args.beta)
        print(f"[eval] {mode}: rollout exposure-bias curve ...")
        curve = rollout_curve(pol, ref, tok, roll, args.beta, p_grid, args.roll_tokens, args.max_len)
        # complete record -- everything needed for later visualization (no info dropped)
        res = {"llm": "zetagpt", "dataset": args.dataset, "method": mode, "tag": tag,
               "beta": args.beta, "n_val": len(ev), "p_grid": p_grid,
               **diag, "hist": hist, "rollout_curve": curve}
        RES["results"][mode] = res
        out_json = os.path.join(args.result_dir, f"eval_{args.dataset}_{mode}.json")
        json.dump(res, open(out_json, "w"))          # full JSON, re-usable without re-training
        print(f"[eval] wrote {out_json}")

    for o in viz.eval_figure(RES, methods, p_grid, plotdir, tag):
        print(f"[eval] [plot] {o} created", flush=True)


if __name__ == "__main__":
    main()
