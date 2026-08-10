# Captain Kirk's Questions — A Dialogue-Relevance Preference Dataset

**File:** `pref_startrek_captain_kirks_questions.json`

## Overview

A Bradley–Terry preference dataset of roughly 10,000 pairs built from Star Trek episode dialogue.
Each entry pairs a **prompt** (a line spoken by Captain Kirk) with a **chosen** response and a
**rejected** response:

- the **chosen** response is the line that actually follows Kirk's line in the same scene — the
  direct, in-context reply;
- the **rejected** response is a reply taken from a *different* episode, so it is fluent but
  irrelevant to the exchange.

The learning signal is therefore **dialogue relevance**: a model must prefer the answer that fits
the ongoing conversation over a plausible but off-topic one, rather than judging on surface fluency
alone.

## Construction

For every Kirk turn that is immediately followed by another character's turn:

| field | content |
|-------|---------|
| `prompt`   | Kirk's line (a question or statement) |
| `chosen`   | the very next line in the same scene (the direct, in-context reply) |
| `rejected` | a reply sampled from a different episode (off-plot, unrelated) |

Every qualifying Kirk turn becomes one pair. The rejected line is resampled so that it differs from
the chosen line, and the pairs are shuffled with a fixed random seed for reproducibility. A light
turn-length filter removes empty and runaway-monologue turns.

## Format

A single JSON object with a `pairs` array:

```json
{
  "name": "startrek_captain_kirk_dialogue",
  "prompt_speaker": "KIRK",
  "n_pairs": 10028,
  "pairs": [
    {
      "pair_id": 0,
      "prompt":   "<a line spoken by Kirk>",
      "chosen":   "<the reply that follows it in the same scene>",
      "rejected": "<a reply from a different episode>",
      "episode":  "<source episode id>",
      "speaker":  "KIRK"
    }
  ]
}
```

Each element of `pairs` carries `pair_id`, the `prompt`, the `chosen` and `rejected` responses, the
source `episode`, and the prompt `speaker`. All text fields are plain English sentences.

## Provenance and license

The prompts and responses are lines of dialogue from Star Trek episodes; all rights to the
underlying dialogue remain with the respective copyright holders. This dataset is intended for
research use only and must not be redistributed where doing so would infringe those rights.
