"""
config_wizard.py -- write config_user.yaml, your own settings for a run.

    python config_wizard.py                   # ask, then write config_user.yaml
    python config_wizard.py --show            # print what would be written, write nothing
    python config_wizard.py --defaults        # accept every default, no questions
    python config_wizard.py --out other.yaml  # write somewhere else

WHY A SEPARATE FILE. config.sh is what the repository ships and what everyone else's runs
use, so editing it makes every `git status` dirty and every pull a possible conflict.
config_user.yaml is yours and is not tracked. The stage scripts do not know it exists: they
source config.sh, which sets the defaults and then, as its last act, sources export_yaml.sh
to read this file over the top of them.

    stage<N>_*.sh  ->  config.sh  ->  export_yaml.sh config_user.yaml

    command-line flag  >  environment variable  >  config_user.yaml  >  config.sh

Only settings you actually chose are written. Everything omitted falls through to config.sh
and then to default_config.py, so a short file is a complete one -- there is no need to carry
a copy of every knob just to change two. Delete the file to go back to the shipped defaults.

The format is the small YAML subset export_yaml.sh understands: `KEY: value` under a heading,
comments to end of line, one level of nesting, no lists.
"""
import argparse
import os
import re
import sys

import default_config as config

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "config_user.yaml")

# One entry per variable the wizard asks about: the shell name, the prompt, the default it
# offers, and how to validate an answer. `choices=None` means free text; an empty answer
# always means "leave it to default_config.py", which is what an empty shell variable does.
QUESTIONS = [
    ("PY", "Python interpreter to run the stages with", "python", None),
    ("GPU", "Device", "auto", ["auto", "cuda", "mps", "cpu"]),
    ("SEED", "Random seed (blank = default_config.py)", "", "int"),
    ("BATCH", f"Batch size (blank = {config.TRAIN['batch']})", "", "int"),
    ("MODEL_SCHEME", "Which configuration scheme to build. A scheme fixes depth, width AND "
     "the context window together (tiny: 8 layers, 512, context 512; -s: 16, 512, 512; "
     "-m: 16, 768, 1024; -l: 24, 1024, 1024). Blank = "
     f"{config.PRETRAIN['model_scheme']}", "",
     ["", "zetagpt-tiny", "zetagpt-s", "zetagpt-m", "zetagpt-l"]),
    ("CONTEXT_WINDOW", "Context window in tokens, overriding the scheme's own. Blank keeps "
     "the scheme's (512 for -s, 1024 for -m and -l). Attention costs grow quadratically in "
     "this, so doubling it is roughly 4x the attention term per step", "", "int"),
    ("MAX_LEN", "Truncation of an encoded example, in tokens. BLANK IS RIGHT: it follows the "
     "context window of the model being built. Set it only to truncate BELOW that on "
     "purpose, e.g. to fit a longer-context scheme on a smaller machine", "", "int"),
    ("PLOT_EVERY",
     f"Redraw figures and print 20 generation examples every N steps "
     f"(blank = {config.TRAIN['plot_every_steps']}, 0 = never)", "", "int"),
    ("CKPT_EVERY", f"Checkpoint every N steps "
     f"(blank = {config.TRAIN['checkpoint_every_steps']})", "", "int"),
    ("PE", "How position enters: 'ssm' is the proposed architecture, 'rope' is the ablation "
     f"control (blank = {config.MODEL['pe']})", "", ["", "ssm", "rope"]),
]

# Per-stage questions: the step budget and the peak learning rate of each stage, which are
# the two knobs anyone actually changes.
#
# config.sh exposes MORE than the wizard asks about -- the dataset, the probe cadences,
# micro-batching, the schedule shape, and the free-text <STAGE>_FLAGS lines. Those are left at
# their defaults here on purpose, to keep this short; they are all present in the generated
# file, commented, and can be edited there. Nothing is lost by not being asked about.
STAGE_QUESTIONS = [
    ("stage 2 tokenizer", [
        ("BPE_MERGES", "merge budget", config.BPE["num_merges"], "int")]),
    ("stage 3 pretraining", [
        ("PRETRAIN_STEPS", "steps", config.PRETRAIN["steps"], "int"),
        ("PRETRAIN_LR", "peak learning rate", config.PRETRAIN["lr"], None)]),
    ("stage 4 fine-tuning", [
        ("SFT_STEPS", "steps", config.SFT["steps"], "int"),
        ("SFT_LR", "peak learning rate", config.SFT["lr"], None)]),
    ("stage 5 reward model", [
        ("REWARD_STEPS", "steps", config.REWARD["steps"], "int"),
        ("REWARD_LR", "peak learning rate", config.REWARD["lr"], None)]),
    ("stage 6 RLHF", [
        ("RLHF_STEPS", "steps", config.RLHF["steps"], "int"),
        ("RLHF_LR", "peak learning rate", config.RLHF["lr"], None)]),
    ("stage 7 chain of thought", [
        ("COT_STEPS", "steps", config.COT["steps"], "int"),
        ("COT_LR", "peak learning rate", config.COT["lr"], None),
        ("COT_INIT", "starting checkpoint", config.COT["init_stage"],
         ["", "pretrain", "sft", "rlhf", "dpo"]),
        ("COT_GROUP", "completions per problem (>= 2)", config.COT["group_size"], "int")]),
    ("stage 8 DPO", [
        ("DPO_STEPS", "steps", config.DPO["steps"], "int"),
        ("DPO_LR", "peak learning rate", config.DPO["lr"], None)]),
    ("stage 9 distillation", [
        ("DISTILL_STEPS", "steps", config.DISTILL["steps"], "int"),
        ("DISTILL_LR", "peak learning rate", config.DISTILL["lr"], None)]),
]


_ALL_VARS = [v for v, *_ in QUESTIONS] + [v for _, qs in STAGE_QUESTIONS for v, *_ in qs]
assert len(_ALL_VARS) == len(set(_ALL_VARS)), \
    f"a variable is asked about twice: {sorted({v for v in _ALL_VARS if _ALL_VARS.count(v) > 1})}"


def ask(prompt, default, kind):
    """One question. Empty input takes the default; the default is shown in brackets. Answers
    are validated here rather than in bash, where a typo would only surface as an argparse
    error several minutes into a corpus scan."""
    while True:
        shown = default if default != "" else "blank"
        try:
            if isinstance(kind, list):
                opts = "/".join(k or "blank" for k in kind)
                raw = input(f"{prompt}\n  [{opts}] ({shown}): ").strip()
            elif kind == "bool":
                raw = input(f"{prompt}\n  [y/N] ({'y' if default else 'n'}): ").strip().lower()
            else:
                raw = input(f"{prompt}\n  ({shown}): ").strip()
        except EOFError:
            # Ctrl-D, or piped input that ran out: take the default and keep going rather than
            # dying halfway through and leaving nothing written.
            print("  (end of input -- taking the default)")
            return default
        if raw == "":
            return default
        if isinstance(kind, list):
            if raw in kind:
                return raw
            print(f"  -> must be one of {kind}")
            continue
        if kind == "bool":
            if raw in ("y", "yes", "1", "true"):
                return "1"
            if raw in ("n", "no", "0", "false"):
                return ""
            print("  -> answer y or n")
            continue
        if kind == "int":
            try:
                int(raw)
                return raw
            except ValueError:
                print("  -> must be a whole number")
                continue
        return raw


def render(answers):
    """The YAML file: a heading per section, and only the settings that were actually chosen.

    Values are quoted when they contain anything that would confuse a whitespace-splitting
    parser -- spaces above all, since <STAGE>_FLAGS lines are free text. export_yaml.sh strips
    matching quotes."""
    def fmt(v):
        v = str(v)
        return f'"{v}"' if (v != v.strip() or " " in v or v.startswith("#")) else v

    lines = [
        "# config_user.yaml -- your settings. Generated by config_wizard.py.",
        "#",
        "# Read by config.sh through export_yaml.sh, over the top of the shipped defaults.",
        "# NOT tracked by git. Edit freely; re-running the wizard overwrites it. Delete it to",
        "# go back to config.sh's defaults.",
        "#",
        "# Anything not listed here is not overridden:",
        "#     command-line flag > environment variable > this file > config.sh >"
        " default_config.py",
        "",
    ]
    shared = [(v, answers.get(v, "")) for v, *_ in QUESTIONS if answers.get(v, "") != ""]
    if shared:
        lines.append("shared:")
        lines += [f"  {v}: {fmt(val)}" for v, val in shared]
        lines.append("")
    for label, qs in STAGE_QUESTIONS:
        chosen = [(v, answers.get(v, "")) for v, *_ in qs if answers.get(v, "") != ""]
        if not chosen:
            continue
        lines.append(f"{label.replace(' ', '_')}:")
        lines += [f"  {v}: {fmt(val)}" for v, val in chosen]
        lines.append("")
    if not shared and all(not any(answers.get(v) for v, *_ in qs) for _, qs in STAGE_QUESTIONS):
        lines += ["# (nothing chosen: every stage runs on config.sh's defaults)", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="write config_user.yaml")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true", help="print, do not write")
    ap.add_argument("--defaults", action="store_true", help="accept every default silently")
    ap.add_argument("--force", action="store_true", help="overwrite without asking")
    a = ap.parse_args()

    answers = {}
    if a.defaults:
        answers = {v: d for v, _, d, _ in QUESTIONS}
        for _, qs in STAGE_QUESTIONS:
            answers.update({v: "" for v, _, _, _ in qs})
    else:
        print("config_wizard -- press Enter to accept the value in brackets.\n"
              "Blank means 'use the default in default_config.py'.\n")
        for var, prompt, default, kind in QUESTIONS:
            answers[var] = ask(prompt, default, kind)
        print("\nPer stage: leave blank to keep default_config.py's budget and rate.\n")
        for label, qs in STAGE_QUESTIONS:
            for var, what, default, kind in qs:
                answers[var] = ask(f"{label}: {what} (blank = {default})", "", kind)

    text = render(answers)
    if a.show:
        print(text)
        return
    if os.path.exists(a.out) and not a.force:
        if input(f"\n{os.path.relpath(a.out, ROOT)} exists. Overwrite? [y/N] ").strip().lower() \
                not in ("y", "yes"):
            sys.exit("nothing written")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(a.out, 0o755)
    print(f"\nwrote {os.path.relpath(a.out, ROOT)} -- config.sh reads it over its defaults")
    print("set values:")
    for var, value in answers.items():
        if value:
            print(f"    {var}={value}")
    if not any(answers.values()):
        print("    (none: every stage will run on config.sh's defaults)")


if __name__ == "__main__":
    main()
