#!/usr/bin/env bash
# stage7_instruct_tuning_rlhf.sh -- stage 7: RLHF by PPO.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage7_instruct_tuning_rlhf.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [7/9] RLHF by PPO ==============================================="
exec $PY -m instruct_rlhf.run $COMMON_FLAGS $RLHF_FLAGS "$@"
