"""
dataset_helpers.py -- every dataset the alignment stages read, behind one interface.

    preference pairs      {prompt, chosen, rejected} -- the downloaded HH tree, or ANY local
                          folder of json/jsonl records; the layout is detected, not declared
    instruction prompts   the `prompt` field of the instruction batches
    corpora               plain documents, packed to a word budget

Everything is normalised to text; Encoder turns text into padded
(ids, attention mask, response mask) tensors given any tokenizer, and accepts records that
were pre-tokenised by the on-disk cache.
"""
import os
import glob
import json
import random

import torch

import default_config as config


# THE TWO LAYOUTS, TOLD APART BY LOOKING RATHER THAN BY BEING TOLD.
#
#   "hh"     a tree of SPLIT DIRECTORIES -- <root>/<name>_train/*.json and <name>_test/*.json,
#            the shape the downloaded rlhf_hh ships in. It carries its OWN held-out split, so
#            no validation set is carved out of the training half.
#   "local"  any other folder: json / jsonl records anywhere beneath it, split by --val_frac.
#
# Detection rather than a --format flag, because the layout is a FACT ABOUT THE DIRECTORY and
# asking the user to restate it is asking them to be wrong: point any stage at your own folder
# of preference data and it works, and the tree stage 1 downloads keeps the honest split it
# was published with. What is loaded and how it was read is printed by every stage, so the
# guess is always visible rather than silent.
DATASETS = {
    "hh": "the downloaded rlhf_hh tree (config.HH_DIR)",
}

TRAIN_SUFFIX, VAL_SUFFIX = "_train", "_test"

# Field names a local record may use. The first one present wins. This is deliberately short:
# these are the spellings the common preference/instruction formats actually use (alpaca,
# dolly, oasst exports, the trl/HF preference convention), not an attempt to guess at anything.
PROMPT_FIELDS = ("prompt", "instruction", "question", "query", "context")
CHOSEN_FIELDS = ("chosen", "response", "output", "answer", "completion", "chosen_response")
REJECTED_FIELDS = ("rejected", "rejected_response", "reject", "worse")
RECORD_KEYS = ("pairs", "records", "data", "examples", "rows")


# --------------------------------------------------------------------------- #
# layout detection
# --------------------------------------------------------------------------- #
def _subdirs(root):
    try:
        return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    except OSError:
        return []


def split_dirs(root, suffix):
    """Subdirectories of `root` named <something><suffix> that actually hold json batches."""
    return [d for d in _subdirs(root)
            if d.endswith(suffix) and glob.glob(os.path.join(root, d, "*.json"))]


def detect_layout(root):
    """"hh", "local" or "missing" for a data directory -- what it IS, not what it is called.

    A directory is read as an HH tree when it contains at least one *_train subdirectory of
    json batches, because that split is the only thing the two layouts disagree about. Anything
    else that holds json or jsonl anywhere beneath it is "local"."""
    if not root or not os.path.isdir(root):
        return "missing"
    if split_dirs(root, TRAIN_SUFFIX):
        return "hh"
    for pat in ("*.json", "*.jsonl"):
        if glob.glob(os.path.join(root, "**", pat), recursive=True):
            return "local"
    return "missing"


def subsets_for(root, configured, suffix=TRAIN_SUFFIX):
    """The split directories to read: the CONFIGURED names where they exist, and whatever the
    tree actually has otherwise. A tree that splits itself differently from ours is still a
    tree, and refusing it over a name would be pedantry rather than safety."""
    have = set(_subdirs(root))
    return [s for s in (configured or []) if s in have] or split_dirs(root, suffix)


def describe(root):
    """One line naming the directory and the layout it was read as, for the stage logs."""
    layout = detect_layout(root)
    if layout == "hh":
        tr = ", ".join(split_dirs(root, TRAIN_SUFFIX)) or "-"
        ev = ", ".join(split_dirs(root, VAL_SUFFIX)) or "none"
        return f"hh tree (own split: train [{tr}], val [{ev}])"
    if layout == "local":
        n = len(_record_files(root))
        return f"local folder ({n} json/jsonl file{'s' if n != 1 else ''}, split by --val_frac)"
    return "missing"


# --------------------------------------------------------------------------- #
# loaders  ->  list[{prompt, chosen, rejected}]  (text)
# --------------------------------------------------------------------------- #
def extract_hh_prompt(text):
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    return "" if idx == -1 else text[: idx + len(marker)]


def _first(rec, fields):
    for f in fields:
        v = rec.get(f)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def normalise(rec):
    """One raw record -> {prompt, chosen, rejected}, or None if it carries no response.

    alpaca-style `input` is APPENDED to the instruction rather than treated as a prompt of its
    own: it is the instruction's operand ("summarise the following:" + the text), and dropping
    it would leave a prompt that cannot be answered."""
    if not isinstance(rec, dict):
        return None
    prompt = _first(rec, PROMPT_FIELDS)
    extra = _first(rec, ("input",))
    if extra and extra != prompt:
        prompt = f"{prompt}\n\n{extra}" if prompt else extra
    chosen = _first(rec, CHOSEN_FIELDS)
    if not chosen:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": _first(rec, REJECTED_FIELDS)}


def _record_files(root):
    files = []
    for pat in ("*.json", "*.jsonl"):
        files += glob.glob(os.path.join(root, "**", pat), recursive=True)
    return sorted(files)


def _read_records(path):
    """Raw records from one file, whichever of the shapes it uses.

    A .jsonl is one record per line; a .json is either a list, a wrapper dict holding the list
    under one of RECORD_KEYS, or a single record. A malformed line is skipped rather than
    fatal -- one bad line in a batch of thousands should not cost a load."""
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return []
    if path.endswith(".jsonl"):
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(doc, list):
        return doc
    if isinstance(doc, dict):
        for k in RECORD_KEYS:
            if isinstance(doc.get(k), list):
                return doc[k]
        return [doc]
    return []


def load_local_pairs(root, limit=0, seed=0, desc="[local] reading records"):
    """{prompt, chosen, rejected} records from ANY folder of json/jsonl, read recursively."""
    from helpers import progress
    files = _record_files(root)
    out = []
    for fp in progress(files, desc=desc, total=len(files)):
        for raw in _read_records(fp):
            rec = normalise(raw)
            if rec:
                out.append(rec)
        if limit and len(out) >= limit:
            break
    random.Random(seed).shuffle(out)
    return out[:limit] if limit else out


def load_train_val(root, train_subsets=None, val_subsets=None, val_frac=0.05, seed=0,
                   limit=0, val_limit=0):
    """(train, val, layout) from `root`, whichever layout it turns out to be.

    An HH tree is read through its own split directories, so `val_frac` is ignored: re-splitting
    a corpus that already ships a held-out set only invites training on part of it. The
    configured subset names are used when they exist and DISCOVERED otherwise, so a tree with
    different subset names still loads. A local folder has no such split, so one is cut from
    the shuffled records at `val_frac`."""
    layout = detect_layout(root)
    if layout == "missing":
        raise SystemExit(
            f"[data] no preference records under {root or '(unset)'}\n"
            f"       Expected either an hh tree (<root>/*_train/*.json) or a folder of\n"
            f"       json/jsonl records with prompt/chosen[/rejected] fields.\n"
            f"       No stage downloads. Run:  ./stage1_download_data.sh")
    if layout == "hh":
        tr_names = subsets_for(root, train_subsets, TRAIN_SUFFIX)
        ev_names = subsets_for(root, val_subsets, VAL_SUFFIX)
        train = load_hh_pairs(tr_names, hh_dir=root, limit=limit, seed=seed)
        val = load_hh_pairs(ev_names, hh_dir=root,
                            limit=val_limit or (max(limit // 20, 200) if limit else 0),
                            seed=seed + 1) if ev_names else []
    else:
        recs = load_local_pairs(root, limit=limit, seed=seed)
        cut = min(len(recs) - 1, int(len(recs) * val_frac)) if len(recs) > 1 else 0
        val, train = recs[:cut], recs[cut:]
        if val_limit:
            val = val[:val_limit]
    if not train:
        raise SystemExit(f"[data] {root} was read as a {layout} layout but yielded no usable "
                         f"records (each needs a prompt and a response).")
    return train, val, layout


def resolve_root(dataset, data_dir=None):
    """A --dataset value -> a directory. "hh" is config.HH_DIR; anything else is a PATH,
    absolute or relative to the data root, so a stage can be pointed at your own folder."""
    if not dataset or dataset == "hh":
        return config.HH_DIR
    if os.path.isdir(dataset):
        return dataset
    cand = os.path.join(data_dir or config.DOWNLOAD_DIR, dataset)
    return cand if os.path.isdir(cand) else dataset


def load_pairs(dataset, data_dir, val_frac, seed, limit=0):
    """(train_pairs, val_pairs) of PREFERENCE pairs, read from local files. Nothing downloads.

    `dataset` is "hh" for the downloaded tree or a path to your own folder. Records without a
    rejected response are refused HERE rather than at the loss: a preference stage cannot be
    trained on demonstrations, and discovering that from a degenerate loss curve is worse than
    reading it at load time."""
    root = resolve_root(dataset, data_dir)
    train, val, layout = load_train_val(root, config.REWARD["hh_subsets"],
                                        config.REWARD["hh_val_subsets"],
                                        val_frac=val_frac, seed=seed, limit=limit)
    keep = lambda rs: [r for r in rs if r["rejected"]]
    train_p, val_p = keep(train), keep(val)
    if not train_p:
        raise SystemExit(
            f"[data] {root} ({layout}) holds {len(train):,} records but none carry a REJECTED "
            f"response.\n       This stage learns from a preference, so it needs pairs; "
            f"fields read are {REJECTED_FIELDS}.")
    return train_p, val_p


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
    """Preference pairs from the SPLIT-DIRECTORY (hh) layout.

    `subsets` names directories under `hh_dir`, each holding <subset>_<batch>.json:
    {"pairs": [{prompt, chosen, rejected}]}, already normalised so that the prompt is the
    shared dialogue prefix and the two responses are what follows it. `limit` caps the pairs
    taken PER SUBSET, so a limit applies evenly across helpful and harmless rather than
    exhausting whichever is read first. Records go through the same normaliser as a local
    folder, so a tree in this layout with differently-named fields still loads."""
    from helpers import progress
    hh_dir = hh_dir or config.HH_DIR
    out = []
    for name in subsets:
        files = sorted(glob.glob(os.path.join(hh_dir, name, "*.json")))
        if not files:
            continue
        kept = 0
        for fp in progress(files, desc=f"[hh] {name}", total=len(files)):
            for raw in _read_records(fp):
                if limit and kept >= limit:
                    break
                rec = normalise(raw)
                if rec:
                    out.append(rec)
                    kept += 1
            if limit and kept >= limit:
                break
    random.Random(seed).shuffle(out)
    return out


def load_instruction_prompts(prompt_dir, limit=0):
    """Prompts to roll out from, read RECURSIVELY from `prompt_dir` -- any layout.

    Every record shape this project ships or a user is likely to bring is accepted through the
    same normaliser: alpaca_gpt4 files carry `records` (instruction plus optional input),
    rlhf_hh files carry `pairs` (a dialogue prefix), a local folder may carry a bare list or
    jsonl. A directory of plain .txt is read as ONE PROMPT PER LINE, which is what a
    hand-written prompt list looks like.

    Only the prompt is taken from any of them. The reference response is ignored on purpose: a
    policy-gradient rollout learns from a score of its OWN generation, never from a
    demonstration, and reading the demonstration here would invite exactly that confusion
    later.

    Duplicates are dropped. rlhf_hh pairs share a prompt between the chosen and rejected
    response and across helpful/harmless subsets, so the raw list repeats itself; rolling out
    twice on the same prompt is wasted compute and quietly reweights the prompt distribution
    the KL penalty is measured against."""
    from helpers import progress
    files = _record_files(prompt_dir)
    kind = "batches"
    if not files:
        files = sorted(glob.glob(os.path.join(prompt_dir, "**", "*.txt"), recursive=True))
        kind = "prompt lists"
    prompts, seen = [], set()

    def take(p):
        p = (p or "").strip()
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)

    for fp in progress(files, desc=f"[instructions] reading {kind}", total=len(files)):
        if kind == "prompt lists":
            try:
                for line in open(fp, encoding="utf-8", errors="replace"):
                    take(line)
            except OSError:
                continue
        else:
            for raw in _read_records(fp):
                if isinstance(raw, dict):
                    extra = _first(raw, ("input",))
                    p = _first(raw, PROMPT_FIELDS)
                    take(f"{p}\n\n{extra}" if extra and extra != p else (p or extra))
        if limit and len(prompts) >= limit:
            break
    return (prompts[:limit] if limit else prompts), len(files)
