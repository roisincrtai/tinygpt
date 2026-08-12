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
#   ./tools/migrate_fineweb-edu_10GB.sh              DRY RUN -- prints the plan, changes nothing
#   ./tools/migrate_fineweb-edu_10GB.sh --apply      do it
#   APPLY=1 ./tools/migrate_fineweb-edu_10GB.sh      do it, for a job runner that submits this
#                                                    script by path and cannot pass arguments
#
# IT IS A DRY RUN UNLESS ASKED, and that is the right default for something that renames four
# gigabytes: the cost of a needless dry run is two seconds, and the cost of an unintended
# rename is a hunt for where the corpus went. Safe to re-run either way -- once the data has
# moved it says so and leaves everything alone.
set -euo pipefail
cd "$(dirname "$0")/.."

OLD="zetagpt-small_pretrain-corpus_fineweb-edu_10GB"
NEW="zetagpt-pretrain_fineweb-edu-2BT"
PY="${PY:-python}"

if [ "${1:-}" = "--apply" ] || [ "${APPLY:-0}" = "1" ]; then
  echo "=== MIGRATING $OLD -> $NEW ==="
  $PY -m tools.rename_dataset "$OLD" "$NEW"
  echo
  echo "Done. Confirm the cache is reachable under the new name -- it should PRINT the stream,"
  echo "not start tokenising:"
  echo "    ./stage4_tokenize_data.sh --list"
  exit 0
fi

echo "=== DRY RUN: $OLD -> $NEW ==="
$PY -m tools.rename_dataset "$OLD" "$NEW" --dry-run
echo
echo "NOTHING WAS CHANGED. This script is a dry run unless you ask for the rename:"
echo "    ./tools/migrate_fineweb-edu_10GB.sh --apply"
echo "    APPLY=1 ./tools/migrate_fineweb-edu_10GB.sh      # if your runner cannot pass arguments"
