"""
verifier.py -- the reward of the chain-of-thought stage: NOT a learned model, an arithmetic
check.

Stage 7 fits a reward model to human preferences, and stage 8 optimises against it. Reasoning
does not need that machinery: a GSM8K problem has one correct final answer, so the reward can
be COMPUTED. That is the whole reason this stage can produce something the preference stages
cannot -- a policy optimising a reward it cannot game by sounding plausible.

    prompt(question)            the R1-Zero style instruction: think, then answer
    demonstration(trace)        a dataset's reference trace in THIS format, for the SFT
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
    "The assistant is required to answer the user's question given after \"User:\", and first "
    "thinks about the reasoning process step by step, then gives the final answer. The "
    f"reasoning is enclosed in {THINK_OPEN} {THINK_CLOSE}, and the final answer in "
    f"{ANSWER_OPEN} {ANSWER_CLOSE}."
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


def normalise_question(text):
    r"""A dataset's question in THIS pipeline's answer convention.

    Datasets carry their own instructions about where the answer goes, and the countdown ones
    say \boxed{}: both in the worked example ("the answer could be \boxed{2+3+5}") and in the
    closing line ("put your final answer within \boxed{}"). Prepending a system prompt that
    asks for <answer> tags leaves the model with TWO conventions and no way to tell which is
    meant -- and only one of them is what the reward pays for.

    So the question is rewritten rather than merely wrapped: every \boxed{X} becomes
    <answer> X </answer>, including the empty \boxed{} of the closing instruction. The braces
    are counted rather than matched to the first "}", so \boxed{(2+3)*5} survives intact.

    Rewriting the QUESTION rather than teaching the reward a second convention is deliberate.
    Two accepted formats mean two ways to be right, a policy free to drift between them, and a
    format reward that no longer means one thing."""
    if not text or _BOXED not in text:
        return text or ""
    out, i = [], 0
    while True:
        at = text.find(_BOXED, i)
        if at < 0:
            out.append(text[i:])
            break
        out.append(text[i:at])
        j, depth, inner = at + len(_BOXED), 1, []
        while j < len(text) and depth:
            ch = text[j]
            depth += (ch == "{") - (ch == "}")
            if depth:
                inner.append(ch)
            j += 1
        body = "".join(inner).strip()
        out.append(f"{ANSWER_OPEN} {body} {ANSWER_CLOSE}" if body
                   else f"{ANSWER_OPEN} {ANSWER_CLOSE}")
        i = j
    return "".join(out)


def prompt(question, cfg=None):
    """The rollout prompt for one problem: system instruction, question, assistant turn.

    The system text is configurable (COT["system_prompt"]) because the container it asks for
    and the container the reward pays for MUST be the same string. They were not: the countdown
    questions carry their own instruction to answer inside \\boxed{}, while this template asked
    for <answer> tags and the extractor read neither -- so a completion that did exactly what
    the prompt said scored zero, every group was unanimous, every advantage was zero, and the
    run trained on nothing while looking healthy."""
    cfg = cfg or {}
    sys_text = cfg.get("system_prompt") or SYSTEM_PROMPT
    q = question.strip()
    if cfg.get("normalise_question", True):
        q = normalise_question(q).strip()
    return PROMPT_TEMPLATE.format(system=sys_text, question=q)


def demonstration(trace):
    r"""A dataset's reference trace as ONE demonstration in this pipeline's format, or None.

        <think>
        ...the reasoning the dataset wrote...
        </think>
        <answer> 38*(48-47) </answer>

    THE SUPERVISED TARGET AND THE REWARDED FORMAT MUST BE THE SAME STRING. The countdown traces
    open with <think>, close it, then write a prose summary ending in \boxed{expr} -- a
    convention this pipeline does not pay for. Fine-tuning on them verbatim would teach a
    container the reward then ignores, which is the same class of mismatch that made every
    reward zero before: the prompt asked for one thing, the reward paid for another, and the
    model obeyed the prompt.

    WHAT IS KEPT AND WHAT IS DROPPED. The think span is kept as written. The \boxed{} content
    becomes the answer span. The prose BETWEEN </think> and \boxed{} is dropped: it restates the
    derivation for a reader, and keeping it would put unrewarded text where the format says the
    answer goes. Returns None when either piece is missing -- a trace with no reasoning or no
    extractable answer is not a demonstration of the format and is better refused than patched
    into one.

    THE ANSWER IS WRITTEN IN PLAIN ARITHMETIC, not in the display maths the traces use. These
    are written for a reader, so their answers say 38 \times (48 - 47) and \frac{85}{34}; the
    verifier now understands both (delatex), but what goes INSIDE the answer tags is the thing
    the model is being taught to produce, and there is no reason to teach it a notation that
    has to be translated before it can be checked. The reasoning keeps whatever notation it
    was written in -- nothing parses that."""
    spans = _THINK_RE.findall(trace or "")
    if not spans:
        return None
    think = spans[0].strip()
    answer = _boxed(trace)
    if not think or not answer:
        return None
    plain = delatex(answer)
    return (f"{THINK_OPEN}\n{think}\n{THINK_CLOSE}\n"
            f"{ANSWER_OPEN} {(plain or answer).strip()} {ANSWER_CLOSE}")


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


def extract_answer(text, fallback=True):
    """The model's final answer as raw text, by decreasing preference:
        0. the last <answer>...</answer> span    -- THIS pipeline's convention
        1. the last \\boxed{...}                 -- what the raw datasets ask for
        2. the last '#### x' line                -- GSM8K's own convention, often imitated
        3. the last number anywhere in the text  -- a bare answer with no structure
    Returns None when the response contains no number at all.

    THE ANSWER SPAN WINS, and the order matters. \\boxed{} was tried first, on the reasoning
    that the countdown questions asked for it -- but normalise_question() now rewrites those
    questions into <answer> tags, so the only \\boxed{} left in a completion is one the model
    wrote inside its REASONING ("so the answer should be \\boxed{91 - 61 - 1}"), which the
    reference traces do constantly. Preferring it meant reading a number out of the middle of
    the thinking and ignoring the answer the model actually committed to. Both are usually the
    same; when they are not, the committed one is the answer.

    `fallback=False` DROPS STEP 3, so only an answer the model put somewhere it declared to be
    an answer counts. Countdown needs that; see countdown_correct."""
    spans = _ANSWER_RE.findall(text or "")
    if spans:
        return spans[-1].strip()
    boxed = _boxed(text)
    if boxed:
        return boxed
    hashes = _HASH_RE.findall(text or "")
    if hashes:
        return hashes[-1].strip()
    if not fallback:
        return None
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


def has_think_span(text):
    """A closed <think>...</think>."""
    return bool(_THINK_RE.search(text or ""))


def has_answer_span(text):
    """A closed <answer>...</answer>."""
    return bool(_ANSWER_RE.search(text or ""))


def is_formatted(text):
    """Both spans, in that order -- the whole structure, reported but no longer the gate.

    THE TWO TAGS ARE REWARDED SEPARATELY, and this is why. A single all-or-nothing format
    reward has no gradient toward the halfway state: a policy that has learned to open and
    close <think> but not yet <answer> earns exactly what one that emits neither earns, so
    nothing pulls it the rest of the way. Paying for each tag makes a staircase it can climb
    one step at a time, which for a base model that has never seen the format is the difference
    between learning it and not.

    THIS IS SAFER IN GRPO THAN IT WOULD BE ELSEWHERE. An advantage is a reward minus its
    GROUP's mean, so a component every completion in a group earns contributes nothing to the
    advantage: the format terms drive learning exactly while they vary, and stop mattering the
    moment the whole group has the format. They cannot go on paying a policy to emit tags
    instead of solving the problem, because by then they are constant within the group."""
    t, a = _THINK_RE.search(text or ""), _ANSWER_RE.search(text or "")
    return bool(t and a and t.start() < a.start())


def reflection_hits(text):
    """Number of self-correction markers in the reasoning."""
    return len(_REFLECT_RE.findall(text or ""))


def score(text, gold, cfg):
    """Reward one completion.

        reward = correct_reward       * [answer is right]
               + think_format_reward  * [a closed <think>...</think>]
               + answer_format_reward * [a closed <answer>...</answer>]
               + think_reward         * [the reasoning is long enough AND mentions the givens]
               - length_penalty       * response words

    Each tag is paid for INDEPENDENTLY -- see is_formatted() for why a single both-or-nothing
    format term left a policy that had learned half the structure with nothing pulling it the
    rest of the way.

    Returns the reward together with every quantity the dynamics figure tracks, so the caller
    never re-parses the text."""
    pred = extract_answer(text)
    correct = is_correct(text, gold, cfg.get("task", "gsm8k"))
    think_fmt = has_think_span(text)
    answer_fmt = has_answer_span(text)
    formatted = is_formatted(text)          # both, ordered -- reported, not a gate
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
              + cfg.get("think_format_reward", 0.0) * float(think_fmt)
              + cfg.get("answer_format_reward", 0.0) * float(answer_fmt)
              + cfg.get("think_reward", 0.0) * grounded
              - cfg.get("length_penalty", 0.0) * n_words)
    return {"reward": reward,
            "correct": float(correct),
            "think_fmt": float(think_fmt),
            "answer_fmt": float(answer_fmt),
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
#   2. it uses only the given numbers, each at most once, and
#   3. the model PUT IT WHERE AN ANSWER GOES -- inside <answer> tags or \boxed{}.
#
# All three matter. Without the second, "38" scores full marks on a problem whose target
# happens to be one of the numbers, and the policy learns to copy rather than to search.
#
# The third closes what the second leaves open, and it is not hypothetical. extract_answer's
# last resort is "the final number anywhere in the text", which is right for gsm8k -- a bare
# number IS the answer to a word problem -- and wrong here: on a target of 38 drawn from
# [48, 38, 47], the completion "the answer is probably 38" passed both other conditions and
# collected the full correctness reward for naming a number. A policy under no length pressure
# finds that. So countdown reads the answer with fallback=False: an unstructured completion
# scores zero for correctness, which is also exactly the state the supervised sub-stage exists
# to move the model out of.

_ALLOWED = set("0123456789+-*/(). ")

# LaTeX spellings of the four operations, and the wrappers that carry no arithmetic at all.
# Ordered longest-first where one is a prefix of another, so \times is not left as \time + "s".
_LATEX_OPS = (("\\times", "*"), ("\\cdot", "*"), ("\\div", "/"),
              ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
              ("$", ""))
_FRAC = "\\frac"
_DFRAC = "\\dfrac"


def _brace_group(text, i):
    """The {...} beginning at `i`, with its braces counted. Returns (body, index after it),
    or (None, i) if there is no group there."""
    if i >= len(text) or text[i] != "{":
        return None, i
    j, depth, out = i + 1, 1, []
    while j < len(text) and depth:
        c = text[j]
        depth += (c == "{") - (c == "}")
        if depth:
            out.append(c)
        j += 1
    return "".join(out), j


def delatex(expr):
    r"""An arithmetic expression written in LaTeX, rewritten as plain arithmetic.

        38 \times (48 - 47)                     ->  38 * (48 - 47)
        (93 - 8) + (55 \div 11)                 ->  (93 - 8) + (55 / 11)
        85 - \left(\frac{39}{\frac{60}{20}}\right)  ->  85 - ((39)/((60)/(20)))

    THIS IS A REWRITE OF NOTATION, NOT A RELAXATION OF THE CHECK. `\frac{a}{b}` and `a/b` are
    the same number; a reward that accepts one and refuses the other is not measuring whether
    the policy solved the problem, it is measuring which notation the policy happened to pick.
    The charset check in countdown_correct still runs, AFTER this, on the rewritten string --
    so whatever reaches eval() is still nothing but digits, the four operators, brackets, a
    decimal point and spaces.

    IT IS NOT COSMETIC. The reference traces are written in display maths, so a model fine-tuned
    on them writes \times and \frac -- and every one of those answers scored zero. On this
    dataset that is 8,968 of the 9,002 traces the verifier rejected: 34 were actually wrong."""
    s = expr or ""
    for a, b in _LATEX_OPS:
        s = s.replace(a, b)
    # \frac{a}{b} -> (a)/(b), innermost first by recursion through _brace_group
    out, i = [], 0
    while i < len(s):
        for name in (_DFRAC, _FRAC):
            if s.startswith(name, i):
                j = i + len(name)
                while j < len(s) and s[j] == " ":
                    j += 1
                num, j = _brace_group(s, j)
                while j < len(s) and s[j] == " ":
                    j += 1
                den, j = _brace_group(s, j)
                if num is not None and den is not None:
                    out.append(f"({delatex(num)})/({delatex(den)})")
                    i = j
                    break
        else:
            out.append(s[i])
            i += 1
    return "".join(out).strip()


def _numbers_used(expr):
    """Every integer literal in an expression, as a list -- so repeats can be counted."""
    return [int(n) for n in re.findall(r"\d+", expr)]


def countdown_correct(text, gold):
    """Does this completion's answer reach the target from the given numbers?

    LaTeX notation is rewritten to plain arithmetic FIRST (delatex), because \\times and * are
    the same operation and a reward that only accepts one is grading handwriting.

    The expression is then evaluated with eval() over a string that has been checked to contain
    NOTHING but digits, the four operators, brackets, a decimal point and spaces. That check is
    the safety argument, and it runs AFTER the rewrite, on exactly the string eval() receives:
    a policy emits arbitrary text, and an expression it wrote is not something to hand to a
    general-purpose evaluator on trust."""
    if not isinstance(gold, dict):
        return False
    expr = delatex((extract_answer(text, fallback=False) or "").strip())
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
