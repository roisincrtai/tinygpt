#!/usr/bin/env bash
# stage6_sft.sh -- stage 6: domain-adaptive SFT.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage6_sft.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [6/10] domain-adaptive SFT ==============================================="
exec $PY -m sft.run $COMMON_FLAGS $SFT_FLAGS "$@"
