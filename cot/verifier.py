"""
verifier.py -- the reward of the chain-of-thought stage: NOT a learned model, an arithmetic
check.

Stage 7 fits a reward model to human preferences, and stage 8 optimises against it. Reasoning
does not need that machinery: a GSM8K problem has one correct final answer, so the reward can
be COMPUTED. That is the whole reason this stage can produce something the preference stages
cannot -- a policy optimising a reward it cannot game by sounding plausible.

    prompt(question)            the R1-Zero style instruction: think, then answer
    extract_answer(text)        the model's final answer, or None
    equal(pred, gold)           numeric-tolerant comparison
    think_text(text)            what the model wrote between the think tags
    reflection_hits(text)       occurrences of self-correcting phrases ("wait", "actually")
    score(text, gold, cfg)      -> dict(reward, correct, formatted, think_len, resp_words, aha)

A note on what is measured. `think_len` is counted in WORDS, not tokens: it is reported next
to the tokenized response length in the figure, and words are what a reader can check against
the printed generation examples. `aha` counts reflection markers -- the phrases DeepSeek
reported appearing spontaneously when the policy learns to re-examine its own work. It is a
crude proxy and is presented as one: a rising `aha` rate is evidence worth looking at by hand,
not a metric to optimise.
"""
import re

# The tags the policy is asked to produce. Kept as module constants because three places need
# to agree on them: the prompt, the format reward, and the think-length measurement.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

# THE SYSTEM PROMPT. R1-Zero's only intervention on the prompt side: a fixed instruction that
# names the CONTAINER for the reasoning and never demonstrates its CONTENT. Nothing here shows
# the model how to reason -- that is the whole point of a zero run -- it only says where the
# reasoning goes, so a format reward has something to pay for.
SYSTEM_PROMPT = (
    "A conversation between a user and an assistant. The assistant first thinks about the "
    "reasoning process step by step, then gives the final answer. The reasoning is enclosed "
    f"in {THINK_OPEN} {THINK_CLOSE}, and the final answer in {ANSWER_OPEN} {ANSWER_CLOSE}."
)

PROMPT_TEMPLATE = (
    "{system}\n"
    "User: {question}\n"
    "Assistant: "
)

# Self-correction markers. Deliberately short and lowercase-matched on word boundaries, so
# "waiting" does not count as "wait".
REFLECTION = ("wait", "hold on", "actually", "hmm", "let me re-?check", "let me reconsider",
              "on second thought", "that's wrong", "i made a mistake", "recheck", "but no",
              "let me verify", "check again")
_REFLECT_RE = re.compile(r"\b(" + "|".join(REFLECTION) + r")\b", re.I)

_THINK_RE = re.compile(re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE), re.S)
_ANSWER_RE = re.compile(re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE), re.S)
_HASH_RE = re.compile(r"####\s*(.+?)\s*$", re.M)
# \boxed{...}, which is what the countdown prompts themselves ask for. Brace-counted rather
# than regexed to the first "}", so \boxed{(2+3)*5} survives.
_BOXED = "\\boxed{"
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def prompt(question, cfg=None):
    """The rollout prompt for one problem: system instruction, question, assistant turn.

    The system text is configurable (COT["system_prompt"]) because the container it asks for
    and the container the reward pays for MUST be the same string. They were not: the countdown
    questions carry their own instruction to answer inside \\boxed{}, while this template asked
    for <answer> tags and the extractor read neither -- so a completion that did exactly what
    the prompt said scored zero, every group was unanimous, every advantage was zero, and the
    run trained on nothing while looking healthy."""
    sys_text = (cfg or {}).get("system_prompt") or SYSTEM_PROMPT
    return PROMPT_TEMPLATE.format(system=sys_text, question=question.strip())


def _clean_number(s):
    """'$1,234.50 dollars' -> 1234.50, or None. Currency, commas and trailing prose are
    stripped because a policy that is RIGHT should not be punished for how it writes."""
    if s is None:
        return None
    m = _NUM_RE.search(s.replace("$", "").replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _boxed(text):
    """The content of the LAST \\boxed{...}, counting braces so nesting survives."""
    at = (text or "").rfind(_BOXED)
    if at < 0:
        return None
    i, depth, out = at + len(_BOXED), 1, []
    while i < len(text) and depth:
        c = text[i]
        depth += (c == "{") - (c == "}")
        if depth:
            out.append(c)
        i += 1
    return "".join(out).strip() or None


def extract_answer(text):
    """The model's final answer as raw text, by decreasing preference:
        0. the last \\boxed{...}                 -- what the countdown prompts ask for
        1. the last <answer>...</answer> span    -- the requested format
        2. the last '#### x' line                -- GSM8K's own convention, often imitated
        3. the last number anywhere in the text  -- a bare answer with no structure
    Returns None when the response contains no number at all."""
    boxed = _boxed(text)
    if boxed:
        return boxed
    spans = _ANSWER_RE.findall(text or "")
    if spans:
        return spans[-1].strip()
    hashes = _HASH_RE.findall(text or "")
    if hashes:
        return hashes[-1].strip()
    nums = _NUM_RE.findall(text or "")
    return nums[-1] if nums else None


def equal(pred, gold, tol=1e-4):
    """Numeric comparison with a tolerance, falling back to a normalised string match for
    non-numeric answers. Never raises: the policy can emit anything."""
    a, b = _clean_number(pred), _clean_number(gold)
    if a is not None and b is not None:
        return abs(a - b) <= tol * max(1.0, abs(b))
    if pred is None or gold is None:
        return False
    return pred.strip().lower() == gold.strip().lower()


def has_think_tags(text):
    """Were the think tags actually present? think_text() falls back to everything before the
    answer, which is right for measuring how much was written and wrong for showing a reader
    what the model produced -- a fallback printed under a THINK heading claims a structure that
    was not there."""
    return bool(_THINK_RE.findall(text or ""))


def think_text(text):
    """What the model wrote as its reasoning: the think span if the tags are present, else
    everything before the answer (an unformatted response still reasons, and its length is
    still the quantity of interest)."""
    spans = _THINK_RE.findall(text or "")
    if spans:
        return spans[-1].strip()
    body = (text or "")
    cut = body.find(ANSWER_OPEN)
    if cut < 0:
        cut = body.find("####")
    return (body[:cut] if cut > 0 else body).strip()


def is_formatted(text):
    """True when the response carries a think span and an answer span, in that order."""
    t, a = _THINK_RE.search(text or ""), _ANSWER_RE.search(text or "")
    return bool(t and a and t.start() < a.start())


def reflection_hits(text):
    """Number of self-correction markers in the reasoning."""
    return len(_REFLECT_RE.findall(text or ""))


def score(text, gold, cfg):
    """Reward one completion.

        reward = correct_reward * [answer is right]
               + format_reward  * [think/answer structure present]
               - length_penalty * response words

    Returns the reward together with every quantity the dynamics figure tracks, so the caller
    never re-parses the text."""
    pred = extract_answer(text)
    correct = is_correct(text, gold, cfg.get("task", "gsm8k"))
    formatted = is_formatted(text)
    think = think_text(text)
    n_words = len((text or "").split())
    # THINKING THAT IS ABOUT THIS PROBLEM. A format reward pays for the tags and a correctness
    # reward pays for the answer; between them a policy can emit tags around nothing and still
    # collect. This third term pays only when the reasoning is BOTH long enough to be reasoning
    # and grounded in the problem -- it mentions the numbers it was given.
    #
    # IT IS NOT A CHECK THAT THE REASONING IS RIGHT, and is not called one. Whether a chain of
    # thought is valid cannot be verified by rule, which is exactly why R1-Zero pays for the
    # container and lets correctness pay for the content. This is a floor against the emptiest
    # kind of reward hacking, not a judge.
    grounded = float(has_think_tags(text)
                     and len(think.split()) >= int(cfg.get("think_min_words", 8))
                     and _mentions_givens(think, gold))
    reward = (cfg["correct_reward"] * float(correct)
              + cfg["format_reward"] * float(formatted)
              + cfg.get("think_reward", 0.0) * grounded
              - cfg.get("length_penalty", 0.0) * n_words)
    return {"reward": reward,
            "correct": float(correct),
            "formatted": float(formatted),
            "grounded": grounded,
            "think_len": float(len(think.split())),
            "resp_words": float(n_words),
            "aha": float(reflection_hits(think) > 0),
            "pred": pred}


# --------------------------------------------------------------------------- #
# COUNTDOWN: reach a target from a handful of numbers
# --------------------------------------------------------------------------- #
# A different task needs a different notion of "right", and only that. The prompt template,
# the think/answer format, the reflection markers and the reward arithmetic are all shared:
# what changes is how a completion's answer is checked, so that is the only thing added here.
#
# The gold is {"target": t, "numbers": [...]}, and an answer is an EQUATION. It is correct when
#   1. it evaluates to the target, and
#   2. it uses only the given numbers, each at most once.
# Both conditions matter. Without the second, "38" scores full marks on a problem whose target
# happens to be one of the numbers, and the policy learns to copy rather than to search.

_ALLOWED = set("0123456789+-*/(). ")


def _numbers_used(expr):
    """Every integer literal in an expression, as a list -- so repeats can be counted."""
    return [int(n) for n in re.findall(r"\d+", expr)]


def countdown_correct(text, gold):
    """Does this completion's answer reach the target from the given numbers?

    The expression is evaluated with eval() over a string that has been checked to contain
    NOTHING but digits, the four operators, brackets, a decimal point and spaces. That check is
    the safety argument: a policy emits arbitrary text, and an expression it wrote is not
    something to hand to a general-purpose evaluator on trust."""
    if not isinstance(gold, dict):
        return False
    expr = (extract_answer(text) or "").strip()
    if not expr or set(expr) - _ALLOWED:
        return False
    expr = expr.split("=")[-1].strip() if expr.count("=") == 1 else expr
    if not expr or set(expr) - _ALLOWED:
        return False
    used, have = _numbers_used(expr), list(gold.get("numbers", []))
    if not used:
        return False
    pool = list(have)
    for n in used:                                   # each given number at most once
        if n not in pool:
            return False
        pool.remove(n)
    try:
        value = eval(expr, {"__builtins__": {}}, {})   # noqa: S307 -- charset checked above
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, ValueError):
        return False
    try:
        return abs(float(value) - float(gold["target"])) < 1e-6
    except (TypeError, ValueError, KeyError):
        return False


def _mentions_givens(think, gold):
    """Does the reasoning refer to the numbers the problem supplied?

    A cheap grounding test, and the only one available without a second model: a chain of
    thought about THIS countdown problem will contain its numbers. For a task whose gold is not
    a dict there is nothing to check against, so it passes -- a term that cannot be evaluated
    must not silently withhold reward."""
    if not isinstance(gold, dict):
        return True
    given = [str(n) for n in gold.get("numbers", [])]
    if not given:
        return True
    return sum(1 for n in given if n in (think or "")) >= max(1, len(given) // 2)


def is_correct(text, gold, task="gsm8k"):
    """Right or not, for whichever task this run is scoring."""
    if task == "countdown":
        return countdown_correct(text, gold)
    return bool(equal(extract_answer(text), gold))
