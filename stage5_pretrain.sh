#!/usr/bin/env bash
# stage5_pretrain.sh -- stage 5: LM pretraining.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage5_pretrain.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [5/10] LM pretraining ==============================================="
exec $PY -m pretrain.run $COMMON_FLAGS $PRETRAIN_FLAGS "$@"
