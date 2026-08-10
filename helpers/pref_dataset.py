"""
pref_dataset.py -- every dataset the alignment stages read, behind one interface.

    preference pairs      {prompt, chosen, rejected} -- a local pref_*.json, the local HH
                          batches, or the HuggingFace hh/shp downloads
    instruction prompts   the `prompt` field of the instruction batches
    corpora               plain documents, packed to a word budget

Everything is normalised to text; Encoder turns text into padded
(ids, attention mask, response mask) tensors given any tokenizer, and accepts records that
were pre-tokenised by the on-disk cache.
"""
import os
import json
import random

import torch


# ONE KEY, because the trainers read local files only. "hh" is the rlhf_hh tree of the
# downloaded instruction dataset, under config.HH_DIR. The hub-backed loaders that used to
# stand behind "shp" and a synthetic json are gone: a stage that cannot reach the network
# cannot offer a dataset that only exists on it, and leaving the option in place would have
# been an invitation to a failure at load time rather than at configuration time.
DATASETS = {
    "hh": "rlhf_hh",
}


# --------------------------------------------------------------------------- #
# loaders  ->  list[{prompt, chosen, rejected}]  (text)
# --------------------------------------------------------------------------- #
def extract_hh_prompt(text):
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    return "" if idx == -1 else text[: idx + len(marker)]


def load_pairs(dataset, data_dir, val_frac, seed, limit=0):
    """(train_pairs, val_pairs) for a preference dataset, read from LOCAL batch files.

    Nothing here downloads. The pairs come from config.HH_DIR, whose subsets carry their own
    train/test split, so `val_frac` is unused for hh -- the dataset's own split is the honest
    one, and re-splitting a corpus that already ships a held-out set only invites training on
    part of it.

    `data_dir` is accepted for signature compatibility with callers that pass a data root; the
    instruction tree is located by config, which is what --instruct_dir moves."""
    if dataset != "hh":
        raise ValueError(f"unknown dataset {dataset!r}; the pipeline reads local data only, "
                         f"so the available keys are {list(DATASETS)}")
    cfg = config.REWARD
    train = load_hh_pairs(cfg["hh_subsets"], hh_dir=config.HH_DIR, limit=limit, seed=seed)
    val = load_hh_pairs(cfg["hh_val_subsets"], hh_dir=config.HH_DIR,
                        limit=(max(limit // 20, 200) if limit else 0), seed=seed + 1)
    if not train:
        raise SystemExit(
            f"[data] no preference pairs under {config.HH_DIR}\n"
            f"       No stage downloads. Run:  ./stage1_download_data.sh")
    return train, val


def load_pretrain_corpus(txt_root, max_words=200, exclude_dirs=()):
    """LM pretraining documents from plain-text files.

    Scans EVERY *.txt under `txt_root` RECURSIVELY (the pretraining corpus**/*.txt), so any text corpus
    dropped anywhere under the pretraining corpus is picked up -- EXCEPT files living under a directory
    whose name is in `exclude_dirs` (matched against every path component below `txt_root`, so
    nested directories with an excluded name are skipped too; config.PRETRAIN["exclude_dirs"]
    defaults to starwars_transcripts and lyrics). Consecutive lines of a file are packed into
    ~max_words-word chunks (never crossing files) and returned as
    {prompt:'', chosen:<chunk>, rejected:''} pseudo-pairs, so sft.run's length-normalized LM
    loop can pretrain on them (prompt empty -> the whole chunk is the response it scores).

    Returns (docs, n_files) where n_files counts the files actually used. The caller reports
    the scan (stage 3 prints the number of files and the total token count)."""
    from helpers import progress, corpus_files
    files = corpus_files(txt_root, exclude_dirs)
    docs = []
    for fp in progress(files, desc="[corpus] scanning *.txt", total=len(files)):
        try:
            lines = open(fp, encoding="utf-8", errors="replace").read().splitlines()
        except Exception:
            continue
        buf, wc = [], 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            buf.append(line)
            wc += len(line.split())
            if wc >= max_words:
                docs.append(" ".join(buf)); buf, wc = [], 0
        if buf:
            docs.append(" ".join(buf))
    return [{"prompt": "", "chosen": d, "rejected": ""} for d in docs], len(files)


def stats_rows(name, train, val, tok=None, sample=2000, source=""):
    """Statistics of a loaded dataset as (field, value) ROWS for helpers.table: entry counts,
    tokenizer vocabulary size (if a tokenizer is given), sampled unique-word count, and
    prompt/chosen/rejected token lengths. Lengths are computed on a random sample of at most
    `sample` pairs to stay fast on large corpora."""
    from helpers import progress
    allp = train + val
    smp = allp if len(allp) <= sample else random.Random(0).sample(allp, sample)
    words = set()
    for p in progress(smp, desc="[dataset] counting words", total=len(smp)):
        for f in ("prompt", "chosen", "rejected"):
            words.update(p[f].lower().split())
    from helpers import count as _count, human as _human
    rows = [("dataset", name),
            ("source", source or config.INSTRUCT_DIR),
            ("entries", _count(len(allp))),
            ("train pairs", _count(len(train))),
            ("val pairs", _count(len(val))),
            ("unique words (sampled)", _count(len(words)))]
    if tok is None:
        return rows
    import statistics
    tl = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    pl, cl, rl = [], [], []
    for p in progress(smp, desc="[dataset] tokenizing lengths", total=len(smp)):
        pl.append(tl(p["prompt"])); cl.append(tl(p["chosen"])); rl.append(tl(p["rejected"]))
    # The token total is ESTIMATED from the sample and labelled as such: tokenizing every pair
    # of a large preference set costs minutes and this is a report, not a measurement the
    # training depends on. The per-pair means it is extrapolated from are printed beside it, so
    # the estimate can be checked rather than taken on trust.
    per_pair = statistics.mean(pl) + statistics.mean(cl) + statistics.mean(rl)
    rows += [("tokenizer vocab", _count(len(tok))),
             ("sampled pairs", _count(len(smp))),
             ("prompt tokens mean/max", f"{statistics.mean(pl):.1f} / {max(pl)}"),
             ("chosen tokens mean/max", f"{statistics.mean(cl):.1f} / {max(cl)}"),
             ("rejected tokens mean/max", f"{statistics.mean(rl):.1f} / {max(rl)}"),
             ("tokens per pair (mean)", f"{per_pair:.1f}"),
             ("TOTAL TOKENS (estimated)", f"~{_human(per_pair * len(allp))}")]
    return rows


# --------------------------------------------------------------------------- #
# encoding: text pairs -> padded (ids, attn, response-mask), left-truncated to max_len
# --------------------------------------------------------------------------- #
class Encoder:
    def __init__(self, tok, device, max_len):
        self.tok, self.device, self.max_len = tok, device, max_len

    def encode(self, pairs, key):
        """Text pairs -> padded tensors. A record carrying "ids" is ALREADY TOKENISED (a
        cached corpus document, see token_cache.py): its prompt is empty and its ids are the
        response, so the tokenizer is not run again."""
        seqs, masks = [], []
        kid = f"{key}_ids"
        for p in pairs:
            if "ids" in p:                          # cached corpus document
                pid, rid = [], list(p["ids"]) + [self.tok.eos_token_id]
            elif kid in p and "prompt_ids" in p:    # cached preference pair (eos included)
                pid, rid = p["prompt_ids"], p[kid]
            else:
                pid = self.tok(p["prompt"], add_special_tokens=False)["input_ids"]
                rid = (self.tok(" " + p[key], add_special_tokens=False)["input_ids"]
                       + [self.tok.eos_token_id])
            full = pid + rid
            rstart = len(pid)
            if len(full) > self.max_len:
                cut = len(full) - self.max_len
                full = full[cut:]
                rstart = max(0, rstart - cut)
            seqs.append(full)
            masks.append([0] * rstart + [1] * (len(full) - rstart))
        T = max(len(s) for s in seqs)
        B = len(seqs)
        pad = self.tok.pad_token_id
        ids = torch.full((B, T), pad, dtype=torch.long)
        attn = torch.zeros(B, T, dtype=torch.long)
        rmask = torch.zeros(B, T, dtype=torch.long)
        for i, (s, m) in enumerate(zip(seqs, masks)):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
            rmask[i, :len(m)] = torch.tensor(m)
        return ids.to(self.device), attn.to(self.device), rmask.to(self.device)


# --------------------------------------------------------------------------- #
# instruction-following data (<instruct_dir>/)
# --------------------------------------------------------------------------- #
def load_hh_pairs(subsets=("helpful_train", "harmless_train"), hh_dir=None, limit=0, seed=0):
    """Anthropic helpful/harmless preference pairs from the LOCAL batch files.

    `subsets` names directories under <instruct_dir>/rlhf_hh, each
    holding <subset>_<batch>.json written by hh_to_json.py: {"pairs": [{prompt, chosen,
    rejected}]}, already normalised so that the prompt is the shared dialogue prefix and the
    two responses are what follows it. `limit` caps the pairs taken per subset."""
    import default_config as config
    import glob
    from helpers import progress
    hh_dir = hh_dir or config.HH_DIR
    out = []
    for name in subsets:
        d = os.path.join(hh_dir, name)
        files = sorted(glob.glob(os.path.join(d, "*.json")))
        if not files:
            continue
        kept = 0
        for fp in progress(files, desc=f"[hh] {name}", total=len(files)):
            try:
                doc = json.load(open(fp, encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            for r in doc.get("pairs", []):
                if limit and kept >= limit:
                    break
                if r.get("prompt") and r.get("chosen") and r.get("rejected"):
                    out.append({"prompt": r["prompt"], "chosen": r["chosen"],
                                "rejected": r["rejected"]})
                    kept += 1
            if limit and kept >= limit:
                break
    random.Random(seed).shuffle(out)
    return out


def load_instruction_prompts(prompt_dir, limit=0):
    """Prompts to roll out from, read RECURSIVELY from `prompt_dir`.

    Handles both batch shapes this project ships, because the same function serves stages that
    read different corners of the same dataset: alpaca_gpt4 files carry `records` (instruction
    plus optional input, pre-joined into `prompt`), rlhf_hh files carry `pairs` (a dialogue
    prefix as `prompt`). Only the prompt is taken from either. The reference response is
    ignored on purpose: a policy-gradient rollout learns from a score of its OWN generation,
    never from a demonstration, and reading the demonstration here would invite exactly that
    confusion later.

    Duplicates are dropped. rlhf_hh pairs share a prompt between the chosen and rejected
    response and across helpful/harmless subsets, so the raw list repeats itself; rolling out
    twice on the same prompt is wasted compute and quietly reweights the prompt distribution
    the KL penalty is measured against."""
    import glob
    from helpers import progress
    files = sorted(glob.glob(os.path.join(prompt_dir, "**", "*.json"), recursive=True))
    prompts, seen = [], set()
    for fp in progress(files, desc="[instructions] reading batches", total=len(files)):
        try:
            doc = json.load(open(fp, encoding="utf-8"))
        except Exception:                                      # noqa: BLE001
            continue
        for r in (doc.get("records") or []) + (doc.get("pairs") or []):
            p = (r.get("prompt") or r.get("instruction") or "").strip()
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
        if limit and len(prompts) >= limit:
            break
    return (prompts[:limit] if limit else prompts), len(files)
