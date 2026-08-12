#!/usr/bin/env bash
# download_config.sh -- EVERY dataset this pipeline fetches, in one list.
#
# Sourced by stage1_download_data.sh and read by tools/download_data.py. ADDING A DATASET IS
# ADDING A LINE HERE; nothing else has to change, and nothing else knows the list exists.
#
#     ./stage1_download_data.sh            fetch whatever is missing
#     ./stage1_download_data.sh --list     what would be fetched, and where
#
# ONE LINE PER DATASET, four fields separated by `|`:
#
#     <local name> | <hugging face repo> | <files per subdir> | <what it is>
#
#   local name        the directory under data/download/ it lands in, and the name every
#                     stage refers to it by. Keep it stable: it is what PRETRAIN_DIR,
#                     INSTRUCT_DIR and COT_DIR are built from.
#   hugging face repo the dataset card, <owner>/<name>. This is the ONLY place in the project
#                     that names a remote.
#   files per subdir  0 for none. When set, a directory holding more than this many loose
#                     files is split into part_0000/, part_0001/, ... afterwards: tens of
#                     thousands of files in one directory are slow to list on most
#                     filesystems, and the corpus scanner walks recursively so the split is
#                     invisible to every stage.
#   what it is        one line, printed by --list. For the reader, not the code.
#
# Blank lines and lines beginning with `#` are ignored, so a dataset can be commented out
# without deleting it.
#
# NOTHING ELSE IN THE PIPELINE DOWNLOADS. Stage 1 is the only code that reaches the network
# for data; every trainer reads a local directory and stops if it is not there. A stage that
# fetched its own corpus would have an unpinned corpus -- not reproducible offline, failing
# hours in on a network error, and leaving a log from which nobody can tell what was trained
# on. Fetch once, deliberately, then train.

ZETAGPT_DATASETS="
zetagpt-tiny_pretrain-corpus_wikitext103        | roisincrtai/zetagpt-tiny_pretrain-corpus_wikitext103        | 500 | pretraining corpus for ZetaGPT-Tiny: WikiText-103
zetagpt-small_pretrain-corpus_fineweb-edu_10GB  | roisincrtai/zetagpt-small_pretrain-corpus_fineweb-edu_10GB  |   0 | pretraining corpus for ZetaGPT-S: a ~10 GB (~2B token) subset of FineWeb-Edu
zetagpt-rlhf-instruction_following              | roisincrtai/zetagpt-rlhf-instruction_following              |   0 | instruction tuning: rlhf_hh preference pairs and alpaca_gpt4 rollout prompts, shared by stages 6, 7 and 10
zetagpt-cot-countdown-game-20k                  | roisincrtai/zetagpt-cot-countdown-game-20k                  |   0 | stage 9 GRPO: 20k Countdown arithmetic problems, each with a target, its numbers and a reference trace
zetagpt-grpo-cot_gsm8k                          | roisincrtai/zetagpt-grpo-cot_gsm8k                          |   0 | grade-school word problems, the alternative stage 9 task (set COT_TASK=gsm8k)
"
export ZETAGPT_DATASETS
