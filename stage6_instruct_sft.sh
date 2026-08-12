#!/usr/bin/env bash
# stage6_instruct_sft.sh -- stage 6: instruction SFT on the Tulu 3 mixture.
#
# The pretrained checkpoint learns to FOLLOW AN INSTRUCTION: the same length-normalized LM
# objective as stage 5, with the loss masked to the response, over 939k conversations of
# reasoning, code, maths, safety and several dozen languages (SFT_DIR).
#
# Every conversation is flattened to ONE DEMONSTRATION PER ASSISTANT TURN, each carrying
# everything said before it as its prompt, in rlhf_hh's own "\n\nHuman: ... \n\nAssistant:"
# form -- so the model stages 7, 8 and 10 go on to score, roll out and align speaks the format
# they prompt it in.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage6_instruct_sft.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [6/10] instruction SFT ==================================================="
exec $PY -m instruct_sft.run $COMMON_FLAGS $SFT_FLAGS "$@"
