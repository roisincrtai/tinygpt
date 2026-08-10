#!/usr/bin/env bash
# stage3_tokenize_data.sh -- stage 3: tokenise the corpora into memory-mapped token streams.
#
# Deterministic, GPU-free and reused by every stage that follows, so it is run once and on its
# own rather than hidden inside the first training step. A stream whose corpus and tokenizer
# are unchanged is left alone, so re-running costs nothing.
#
#   ./stage3_tokenize_data.sh --list        what would be built, and where
#   ./stage3_tokenize_data.sh --force       rebuild even if current
#   ./stage3_tokenize_data.sh --only tiny   one scheme's corpus
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [3/9] tokenize corpora ============================================"
exec $PY -m tokenize_data.run $TOKENIZE_FLAGS "$@"
