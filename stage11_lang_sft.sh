#!/usr/bin/env bash
# stage11_lang_sft.sh -- stage 11: adapt the pretrained model to a NEW LANGUAGE.
#
# THREE SUB-STAGES, in this order, all inside `python -m lang_sft.run`:
#
#   1. BPE SFT. The default BPE reads the new corpus and is EXTENDED with merges learned on it.
#      Every existing id keeps its meaning; the new merges are appended.
#      -> checkpoints/lang-sft/<dataset>/bpe/bpe.json
#   2. TOKENIZE. The corpus is tokenised WITH THAT EXTENDED VOCABULARY into memory-mapped
#      shards -- stage 4's machinery, which could not have run in advance because the
#      vocabulary it tokenises against did not exist until sub-stage 1.
#      -> cache/tokens/bpe_<encoding>_<fingerprint>/
#   3. LANG SFT. The pretrained checkpoint is loaded, its embedding rearranged and grown for
#      the extended vocabulary, and training continues on the stream.
#      -> checkpoints/lang-sft/<dataset>/checkpoint_<model>_<pe>_lang-sft-<short>.pt
#         checkpoints/lang-sft/<dataset>/history_<model>_<pe>_lang-sft-<short>.json
#
# WHY THE VOCABULARY FIRST. A byte-level BPE never fails on an unseen language -- it falls back
# to raw bytes -- so training Irish on the English vocabulary would run, and would quietly
# spend four or five tokens on words an adapted vocabulary spells in one. Extending first is
# what makes the same step budget cover several times as much Irish.
#
# Which corpus and what it is called in the filenames:
#   LANG_SFT_DATASET=<name under data/download/, or a path>
#   LANG_SFT_DATASET_SHORTNAME=gaelic
#
# Knobs live in config.sh (and ultimately default_config.py); extra arguments are forwarded:
#   ./stage11_lang_sft.sh --help
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== [11/11] Language adaptation: $LANG_SFT_DATASET_SHORTNAME (bpe sft -> tokenize -> lang sft) ==="
exec $PY -m lang_sft.run $COMMON_FLAGS $LANG_SFT_FLAGS "$@"
