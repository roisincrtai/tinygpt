#!/usr/bin/env bash
# stage5_train_rlhf_reward.sh -- stage 5: reward model.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage5_train_rlhf_reward.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [5/9] reward model ==============================================="
exec $PY -m instruct_reward.run $COMMON_FLAGS $REWARD_FLAGS "$@"
