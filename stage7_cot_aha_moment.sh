#!/usr/bin/env bash
# stage7_cot_aha_moment.sh -- stage 7: chain-of-thought reasoning by GRPO, the aha moment.
#
# Reinforcement learning against a VERIFIED answer rather than a learned reward model: the
# policy is asked to think inside <think> tags, its final answer is extracted and checked
# arithmetically against the gold answer, and GRPO standardises the reward within a group of
# completions sampled for the same problem. The stage tracks think length, which is where the
# reported "aha moment" shows up.
#
# The starting checkpoint is a knob -- COT_INIT in config.sh, or per run:
#   COT_INIT=sft ./stage7_cot_aha_moment.sh
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage7_cot_aha_moment.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [7/9] Chain of thought by GRPO (aha moment) ====================="
exec $PY -m cot.run $COMMON_FLAGS $COT_FLAGS "$@"
