#!/usr/bin/env bash
# stage8_instruct_dpo.sh -- stage 8: DPO.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage8_instruct_dpo.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [8/9] DPO ==============================================="
exec $PY -m instruct_dpo.run $COMMON_FLAGS $DPO_FLAGS "$@"
