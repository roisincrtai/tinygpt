#!/usr/bin/env bash
# stage1_download_data.sh -- stage 1: fetch the corpora into data/download/.
#
# THE ONLY STAGE THAT REACHES THE NETWORK. Every stage after this one reads a local directory
# and stops if it is not there: a trainer that fetched its own corpus would have an unpinned
# corpus -- not reproducible offline, failing hours in on a network error, and leaving a log
# from which nobody can tell what was trained on.
#
# Already-present datasets are skipped, and an interrupted transfer resumes, so re-running this
# is cheap and is the right response to a failure.
#
#   ./stage1_download_data.sh            fetch what is missing
#   ./stage1_download_data.sh --list     what would be fetched, and where; fetch nothing
#   ./stage1_download_data.sh --force    re-fetch even what is complete
#
# RE-RUNNING IS HOW YOU RESUME. A dataset counts as complete only when a fetch FINISHED and
# said so -- .zetagpt_download.json inside it records the file count and byte total at that
# moment. Anything short of that is fetched again, and snapshot_download skips whatever
# already matches, so nothing is transferred twice.
#
# WHAT is fetched is download_config.txt, one line per dataset. Add a line to add a dataset.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [1/10] download datasets ==========================================="
exec $PY -m tools.download_data "$@"
