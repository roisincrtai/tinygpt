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
#   ./stage1_download_data.sh --force    re-fetch even what is present
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [1/9] download datasets ==========================================="
exec $PY -m tools.download_data "$@"
