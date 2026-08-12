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
#     ./tools/migrate_fineweb-edu_10GB.sh              # dry run, changes nothing
#     ./tools/migrate_fineweb-edu_10GB.sh --apply      # do it
#
# Safe to re-run: with the data already moved it says so and leaves everything alone.
set -euo pipefail
cd "$(dirname "$0")/.."

OLD="zetagpt-small_pretrain-corpus_fineweb-edu_10GB"
NEW="zetagpt-pretrain_fineweb-edu-2BT"
PY="${PY:-python}"

if [ "${1:-}" = "--apply" ]; then
  echo "=== migrating $OLD -> $NEW ==="
  exec $PY -m tools.rename_dataset "$OLD" "$NEW"
fi

echo "=== DRY RUN: $OLD -> $NEW ==="
$PY -m tools.rename_dataset "$OLD" "$NEW" --dry-run
echo
echo "Nothing was changed. Re-run with --apply to do it:"
echo "    ./tools/migrate_fineweb-edu_10GB.sh --apply"
