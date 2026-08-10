"""
verifier.py -- the reward of the chain-of-thought stage: NOT a learned model, an arithmetic
check.

Stage 6 fits a reward model to human preferences, and stage 7 optimises against it. Reasoning
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

PROMPT_TEMPLATE = (
    "A conversation between a user and an assistant. The assistant first thinks about the "
    "reasoning process step by step, then gives the final answer. The reasoning is enclosed "
    f"in {THINK_OPEN} {THINK_CLOSE} and the final answer in {ANSWER_OPEN} {ANSWER_CLOSE}.\n"
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
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def prompt(question):
    """The rollout prompt for one problem."""
    return PROMPT_TEMPLATE.format(question=question.strip())


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


def extract_answer(text):
    """The model's final answer as raw text, by decreasing preference:
        1. the last <answer>...</answer> span    -- the requested format
        2. the last '#### x' line                -- GSM8K's own convention, often imitated
        3. the last number anywhere in the text  -- a bare answer with no structure
    Returns None when the response contains no number at all."""
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
    correct = bool(equal(pred, gold))
    formatted = is_formatted(text)
    think = think_text(text)
    n_words = len((text or "").split())
    reward = (cfg["correct_reward"] * float(correct)
              + cfg["format_reward"] * float(formatted)
              - cfg.get("length_penalty", 0.0) * n_words)
    return {"reward": reward,
            "correct": float(correct),
            "formatted": float(formatted),
            "think_len": float(len(think.split())),
            "resp_words": float(n_words),
            "aha": float(reflection_hits(think) > 0),
            "pred": pred}
