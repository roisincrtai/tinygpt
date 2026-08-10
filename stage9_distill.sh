#!/usr/bin/env bash
# stage9_distill.sh -- stage 9: distillation into gpt2-small.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage9_distill.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [9/9] distillation into gpt2-small ==============================================="
exec $PY -m distill.run $COMMON_FLAGS $DISTILL_FLAGS "$@"
