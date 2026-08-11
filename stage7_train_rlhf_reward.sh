#!/usr/bin/env bash
# stage7_train_rlhf_reward.sh -- stage 7: reward model.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage7_train_rlhf_reward.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [7/10] reward model ==============================================="
exec $PY -m instruct_reward.run $COMMON_FLAGS $REWARD_FLAGS "$@"
