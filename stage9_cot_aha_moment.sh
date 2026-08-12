#!/usr/bin/env bash
# stage9_cot_aha_moment.sh -- stage 9: chain-of-thought reasoning, the aha moment.
#
# TWO SUB-STAGES over one problem bank, in this order:
#
#   1. SFT   the dataset's reference traces, rewritten into <think>...</think>
#            <answer>...</answer> form and trained by maximum likelihood. It teaches the
#            FORMAT and nothing else.       -> checkpoints/cot/checkpoint_<run>_cot-sft.pt
#   2. GRPO  reinforcement learning FROM THAT CHECKPOINT against a VERIFIED answer rather than
#            a learned reward model: the final answer is extracted and checked arithmetically,
#            and the reward is standardised within a group of completions sampled for the same
#            problem.                       -> checkpoints/cot/checkpoint_<run>_cot-grpo.pt
#
# WHY THE SFT COMES FIRST. GRPO only amplifies what the policy already samples. A base model
# has never emitted a <think> tag, so before something demonstrates the format every completion
# in a group scores alike, every advantage is zero, and the run trains on nothing while its
# curves look healthy. COT_SFT=0 (or --no-cot_sft) skips it, which is the R1-Zero setting.
#
# The stage tracks think length, which is where the reported "aha moment" shows up.
#
# The starting checkpoint of the SFT is a knob -- COT_INIT in config.sh, or per run:
#   COT_INIT=sft ./stage9_cot_aha_moment.sh
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage9_cot_aha_moment.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [9/10] Chain of thought by GRPO (aha moment) ====================="
exec $PY -m cot.run $COMMON_FLAGS $COT_FLAGS "$@"
