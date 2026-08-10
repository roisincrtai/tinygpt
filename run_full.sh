#!/usr/bin/env bash
# run_full.sh -- the whole pipeline, in order. Each step is its own script, so any of them can be
# run (or re-run) alone; knobs live in config.sh and default_config.py.
#
#   ./run_full.sh                                  # everything
#   PRETRAIN_FLAGS="--pretrain_steps 50000" ./run_full.sh
#
# Stage 1 downloads and is skipped when the data is already present, so re-running the
# whole pipeline does not re-fetch anything.
#   ./run_full.sh --no-resume                      # extra flags are forwarded to every stage
set -euo pipefail
cd "$(dirname "$0")"

./stage1_download_data.sh          "$@"
./stage2_train_bpe_tokenizer.sh    "$@"
./stage3_pretrain.sh               "$@"
./stage4_sft.sh                    "$@"
./stage5_train_rlhf_reward.sh      "$@"
./stage6_instruct_tuning_rlhf.sh   "$@"
./stage7_cot_aha_moment.sh         "$@"
./stage8_instruct_dpo.sh           "$@"
./stage9_distill.sh                "$@"
echo "=== done ================================================================="
