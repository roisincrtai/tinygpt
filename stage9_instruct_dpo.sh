#!/usr/bin/env bash
# stage9_instruct_dpo.sh -- stage 9: DPO.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage9_instruct_dpo.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [9/9] DPO ==============================================="
exec $PY -m instruct_dpo.run $COMMON_FLAGS $DPO_FLAGS "$@"
