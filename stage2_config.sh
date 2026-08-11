#!/usr/bin/env bash
# stage2_config.sh -- stage 2: choose the settings this pipeline will run with.
#
# WHY CONFIGURATION IS A STAGE. Everything after this point is expensive: the tokenizer is
# hours, the token streams are hours, pretraining is days. Every one of them reads the same
# settings, and a mistake in any of them -- the wrong corpus, a context window that disagrees
# with the model scheme, a batch that does not fit the card -- is not discovered until the
# hours have been spent. Making the choice its own step means it happens once, deliberately,
# before anything has been paid for, and that the run's settings are printed and recorded
# rather than remembered.
#
# It writes config_user.yaml, which is YOURS and is not tracked: config.sh stays exactly as
# the repository ships it, so `git status` stays clean and `git pull` cannot conflict with
# your settings. The precedence, highest first:
#
#   command-line flag  >  environment variable  >  config_user.yaml  >  config.sh  >  default_config.py
#
#   ./stage2_config.sh              ask, write config_user.yaml, then show the result
#   ./stage2_config.sh --show       print what would be written; write nothing
#   ./stage2_config.sh --defaults   accept every default, no questions
#   ./stage2_config.sh --out f.yaml write somewhere else
#
# THIS STAGE IS OPTIONAL, like every other. Delete config_user.yaml and the pipeline runs on
# what config.sh ships. Nothing downstream knows this file exists -- the stages source
# config.sh, which reads the YAML over its own defaults as its last act -- so skipping this
# stage costs settings, never correctness.
#
# The configuration summary is printed AFTER the wizard has written, so what you read at the
# end is the result of what you just chose, and not the state before it.
set -euo pipefail
cd "$(dirname "$0")"
echo "=== [2/10] configure the run =========================================="

PY="${PY:-python}"
$PY -m tools.config_wizard "$@"

echo
source ./config.sh          # prints every knob, its value, and where that value came from
