#!/usr/bin/env bash
# tools/migrate_fineweb-edu_10GB.sh -- one rename, run once:
#
#     zetagpt-small_pretrain-corpus_fineweb-edu_10GB  ->  zetagpt-pretrain_fineweb-edu-2BT
#
# matching the -10BT sibling. The corpus is unchanged; only its NAME is, and the name is an
# input to the token-cache signature -- so `mv` alone would leave 2B tokens of cache the
# pipeline cannot see, and stage 5 would silently tokenise them again. tools/rename_dataset.py
# renames the data, moves the cache, re-signs every shard and rewrites the manifest, without
# reading a single token.
#
#   ./tools/migrate_fineweb-edu_10GB.sh              DO IT
#   ./tools/migrate_fineweb-edu_10GB.sh --dryrun     print the plan and change nothing
#
# NEEDS NOTHING BUT PYTHON. No torch, no GPU virtualenv: it renames directories and rewrites
# one json, so it runs on a login node.
#
# Safe to re-run: once the data has moved it says so and leaves everything alone.
set -euo pipefail
cd "$(dirname "$0")/.."

OLD="zetagpt-small_pretrain-corpus_fineweb-edu_10GB"
NEW="zetagpt-pretrain_fineweb-edu-2BT"
PY="${PY:-python}"

case "${1:-}" in
  --dryrun|--dry-run)
    echo "=== DRY RUN: $OLD -> $NEW ==="
    $PY -m tools.rename_dataset "$OLD" "$NEW" --dry-run
    echo
    echo "Nothing was changed. Drop --dryrun to do it."
    exit 0
    ;;
esac

echo "=== MIGRATING $OLD -> $NEW ==="
$PY -m tools.rename_dataset "$OLD" "$NEW"
echo
echo "Confirm the cache is reachable under the new name -- it should PRINT the stream,"
echo "not start tokenising:"
echo "    ./stage4_tokenize_data.sh --list"
