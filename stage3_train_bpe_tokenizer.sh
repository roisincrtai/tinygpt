#!/usr/bin/env bash
# stage3_train_bpe_tokenizer.sh -- stage 3: BPE tokenizer.
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage3_train_bpe_tokenizer.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [3/10] BPE tokenizer ==============================================="
exec $PY -m tokenizer.run $COMMON_FLAGS $BPE_FLAGS "$@"
