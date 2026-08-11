#!/usr/bin/env bash
# stage4_tokenize_data.sh -- stage 4: tokenise the corpora into memory-mapped token streams.
#
# Deterministic, GPU-free and reused by every stage that follows, so it is run once and on its
# own rather than hidden inside the first training step. A stream whose corpus and tokenizer
# are unchanged is left alone, so re-running costs nothing.
#
#   ./stage4_tokenize_data.sh --list        what would be built, and where
#   ./stage4_tokenize_data.sh --force       rebuild even if current
#   ./stage4_tokenize_data.sh --only tiny   one scheme's corpus
#
# COMMON_FLAGS IS PASSED HERE TOO, and it has to be. A token stream is identified by the corpus
# it came from AND the packing applied to it -- the word budget, the parquet text column, the
# held-out files excluded -- and those three arrive as `--set PRETRAIN.*` inside COMMON_FLAGS,
# which every training stage receives. This stage receiving only TOKENIZE_FLAGS meant it packed
# the corpus with default_config's values while stage 5 packed it with config.sh's, so the two
# named different streams: stage 4 would finish, and stage 5 would then tokenise the whole
# corpus again from zero, which for a 10 GB corpus is hours that look exactly like normal
# progress. tokenize_data.run parses known arguments only, so the flags meant for a trainer
# (--gpu, --batch and the rest) pass through it harmlessly.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [4/10] tokenize corpora ============================================"
exec $PY -m tokenize_data.run $COMMON_FLAGS $TOKENIZE_FLAGS "$@"
