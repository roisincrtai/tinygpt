#!/usr/bin/env bash
# config.sh -- EVERY configurable argument, in one place, sourced by each stage<N>_*.sh.
#
# THE DEFAULTS LIVE IN default_config.py. This file is the shell-level layer that steers a run
# without editing Python. Every knob the trainers accept has a variable here; leave one blank
# and that stage uses default_config.py's value, which is what the shipped file does for
# almost everything.
#
#   command-line flag > environment variable > config_user.yaml > config.sh > default_config.py
#
#   ./stage4_pretrain.sh                        # this file's settings
#   PRETRAIN_STEPS=20000 ./stage4_pretrain.sh   # override one knob for one run
#   GPU=cpu ./stage5_sft.sh                     # override a shared knob for one run
#   COT_INIT=sft ./stage8_cot_aha_moment.sh     # start the CoT stage from the SFT model
#
# YOUR OWN SETTINGS GO IN config_user.yaml, not here. `python config_wizard.py` writes it;
# this file sources it at the end, through export_yaml.sh, so its values override the defaults
# below without editing a tracked file. It is git-ignored. Delete it and the defaults here
# stand. A variable already set in the environment is never overwritten by the YAML.
#
# Each stage script also forwards its own arguments, so `./stage9_instruct_dpo.sh --beta 0.05`
# works too.
#
# Layout: interpreter, the shared knobs every stage reads, one section per stage -- all values
# only -- then the YAML import, then the command lines are assembled from whatever survived.
# <STAGE>_FLAGS in each section is free text appended verbatim, for anything without a
# variable of its own.

# Which variables came from the ENVIRONMENT, recorded before a single default is applied:
# export_yaml.sh must not overwrite these, or `GPU=cpu ./stage4_pretrain.sh` would lose to the
# YAML file. `${!v+x}` is set for an exported-but-empty variable too, which is deliberate:
# GPU= means "explicitly nothing", not "unset".
# The complete list, defined once: it drives the environment capture below and the summary
# printed at the end, so a knob added in one place cannot be forgotten by the other.
ZETAGPT_VARS="PY GPU SEED BATCH MICRO_BATCH DATASET LIMIT VAL_FRAC BETA MAX_LEN \
MODEL_SCHEME CONTEXT_WINDOW PRETRAIN_DIR SFT_DIR INSTRUCT_DIR \
LR_SCHEDULE LR_MIN_FACTOR PLOT_EVERY CKPT_EVERY PE SSM_STATS_EVERY NO_RESUME EVAL_EVERY EVAL_PAIRS EB_EVERY \
EB_PAIRS ROLLOUT_TEMP N_HIST N_ROLL ROLL_TOKENS P_GRID BPE_MERGES BPE_FLAGS PRETRAIN_STEPS \
PRETRAIN_LR PRETRAIN_FLAGS SFT_STEPS SFT_LR SFT_FLAGS REWARD_STEPS REWARD_LR REWARD_FLAGS \
RLHF_STEPS RLHF_LR RLHF_FLAGS COT_STEPS COT_LR COT_INIT COT_GROUP COT_FLAGS DPO_STEPS \
DPO_LR DPO_FLAGS DISTILL_STEPS DISTILL_LR DISTILL_FLAGS COMMON_FLAGS \
SCALING_MODELS SCALING_BUDGETS SCALING_CONTEXT SCALING_BATCH SCALING_LR SCALING_LR_RULE \
SCALING_FLAGS"

ZETAGPT_ENV_SET=""
for _v in $ZETAGPT_VARS; do
  [ -n "${!_v+x}" ] && ZETAGPT_ENV_SET="$ZETAGPT_ENV_SET $_v"
done
unset _v

# =========================================================================== #
# interpreter
# =========================================================================== #
PY="${PY:-python}"

# =========================================================================== #
# shared -- every stage reads these
# =========================================================================== #
GPU="${GPU:-}"                      # auto | cuda | mps | cpu
SEED="${SEED:-}"                    # random seed
BATCH="${BATCH:-}"                  # examples (or preference pairs) per step
MICRO_BATCH="${MICRO_BATCH:-}"      # gradient-accumulation micro-batch (0 = off)
DATASET="${DATASET:-}"              # "hh" = the downloaded rlhf_hh tree, or a PATH to
                                    # your own folder of json/jsonl records carrying
                                    # prompt / chosen / rejected. The layout is
                                    # DETECTED: a tree with its own *_train and *_test
                                    # subdirs keeps that split, a plain folder is cut
                                    # at VAL_FRAC. Empty = the default in
                                    # default_config.py.
LIMIT="${LIMIT:-}"                  # cap #preference pairs loaded (0 = all)
VAL_FRAC="${VAL_FRAC:-}"            # validation fraction of the preference file
BETA="${BETA:-}"                    # implicit-reward beta, shared by DPO and evaluation

# TRUNCATION of an encoded example, in tokens, for every stage. BLANK IS THE RIGHT SETTING:
# it follows the context window of the model actually being built -- CONTEXT_WINDOW below, or
# the scheme's own -- so the encoder and the model can never disagree.
#
# This shipped as 256 while the small scheme's window was 512, and the two disagreeing is a
# quiet and expensive kind of wrong: training ran at 256, every figure and table said 512, and
# roughly four fifths of documents lost their opening to truncation. The corpus is packed at
# PRETRAIN["max_words"] = 344 words, which at about 1.5 tokens per word fills a 512-token
# window. Set this only to truncate BELOW the context window on purpose -- to fit a
# longer-context model on a smaller machine, say. To change the context itself set
# CONTEXT_WINDOW, which also tells the model what it was trained at.
MAX_LEN="${MAX_LEN:-}"

# =========================================================================== #
# data -- LOCAL DIRECTORIES ONLY
# =========================================================================== #
# NOTHING IN THE PIPELINE DOWNLOADS. Fetch once, deliberately:
#
#     ./stage1_download_data.sh          every dataset, into data/download/
#     ./stage1_download_data.sh --list   what would be fetched, and where
#
# A stage that fetched its own corpus would have an unpinned corpus: not reproducible offline,
# failing hours in on a network error, and leaving a log from which nobody can tell what was
# trained on. Every stage below reads a directory and stops if it is not there.

# PRETRAINING CORPUS. Blank takes the scheme's own: zetagpt-tiny trains on the downloaded
# WikiText-103 corpus and zetagpt-s on the ~10 GB (~2B token) FineWeb-Edu subset, while
# zetagpt-m and zetagpt-l ship with NO corpus and must be pointed at one here -- neither is
# the right diet for a 199M or 480M model, and defaulting them to one would be worse than
# refusing. NOTE that Tiny and S therefore differ in corpus as well as depth.
PRETRAIN_DIR="${PRETRAIN_DIR:-}"

# INSTRUCTION-TUNING TREE, shared by stages 5, 6, 7, 9 and 10. Blank takes
# data/download/zetagpt-rlhf-instruction_following, whose layout is flat: alpaca_gpt4/ and
# rlhf_hh/ at its root. Moving this root moves everything derived from it -- the fine-tuning
# data, the preferences and the prompt bank -- together, so no two stages can read
# different datasets.
INSTRUCT_DIR="${INSTRUCT_DIR:-}"

# FINE-TUNING DATA for stage 5. Blank = <INSTRUCT_DIR>/rlhf_hh, the same records stages 6, 7
# and 9 read: stage 5 trains on the CHOSEN response of each pair, conditioned on its prompt.
# Point this elsewhere only to fine-tune on something other than the preference data.
SFT_DIR="${SFT_DIR:-}"

# learning-rate schedule; the per-stage peak rates are in each stage's section below
LR_SCHEDULE="${LR_SCHEDULE:-}"      # cosine | constant
LR_MIN_FACTOR="${LR_MIN_FACTOR:-}"  # cosine floor: minimum lr = stage lr / this

# cadences
PLOT_EVERY="${PLOT_EVERY:-}"        # figures + 20 generation examples, every N steps (0 = off)
CKPT_EVERY="${CKPT_EVERY:-}"        # checkpoint every N steps
# HOW POSITION ENTERS THE MODEL. "ssm" is the proposed architecture: the state space module
# supplies position and no positional encoding is used anywhere. "rope" is the ABLATION
# CONTROL: the module is removed and rotary positions are applied inside attention instead.
# The choice is recorded in the checkpoint and appears in every filename the run writes, so an
# ablation never overwrites the run it is compared with.
#   PE=rope ./stage4_pretrain.sh
PE="${PE:-}"                        # ssm | rope

# State space diagnostics -- memory horizon, selectivity, residual write ratio and the rest,
# recorded PER LAYER for one step every N and drawn in the pretraining dynamics figure.
# 0 turns them off. Read by stage 4 (pretraining).
SSM_STATS_EVERY="${SSM_STATS_EVERY:-}"
NO_RESUME="${NO_RESUME:-}"          # set to 1 to ignore checkpoints and start from scratch

# diagnostic probes (read by the DPO stage and by eval.py)
EVAL_EVERY="${EVAL_EVERY:-}"        # held-out probe every N steps (0 = use PLOT_EVERY)
EVAL_PAIRS="${EVAL_PAIRS:-}"        # pairs per held-out probe
EB_EVERY="${EB_EVERY:-}"            # exposure-bias probe every N steps (0 = off)
EB_PAIRS="${EB_PAIRS:-}"            # pairs per exposure-bias probe
ROLLOUT_TEMP="${ROLLOUT_TEMP:-}"    # sampling temperature of that probe
N_HIST="${N_HIST:-}"                # end-of-stage evaluation: histories scored
N_ROLL="${N_ROLL:-}"                # end-of-stage evaluation: rollouts
ROLL_TOKENS="${ROLL_TOKENS:-}"      # end-of-stage evaluation: tokens per rollout
P_GRID="${P_GRID:-}"                # end-of-stage evaluation: comma-separated p values

COMMON_FLAGS="${COMMON_FLAGS:-}"
# =========================================================================== #
# stage 3 -- tokenise the corpora (no knobs of its own; it reads the pretraining settings
#            below and writes cache/tokens/bpe_<256+merges>_<fp>/*.tokens)

# stage 2 -- byte-level BPE tokenizer
# =========================================================================== #
BPE_MERGES="${BPE_MERGES:-}"        # merge budget; raising it EXTENDS an existing tokenizer
BPE_FLAGS="${BPE_FLAGS:-}"

# =========================================================================== #
# stage 4 -- pretraining
# =========================================================================== #
PRETRAIN_STEPS="${PRETRAIN_STEPS:-}"
PRETRAIN_LR="${PRETRAIN_LR:-}"

# WHICH SIZE IS TRAINED. A scheme fixes depth, width and the context window TOGETHER:
#
#     scheme          layers  heads  d_model  d_h  context  parameters
#     zetagpt-tiny         8      8      512   64      512       61.5M
#     zetagpt-s           16      8      512   64      512       97.2M   (default)
#     zetagpt-m           16     12      768   64     1024      199.3M
#     zetagpt-l           24     16     1024   64     1024      479.9M
#
# together, because those are not independent choices: a scheme is a point on a size ladder,
# and a model assembled from parts of two of them is one that no reported number describes.
# The head dimension is 64 throughout and is not settable here; n_head follows as d_model / 64.
#
# TO ADD YOUR OWN SIZE, edit default_config.py -- this file only SELECTS a scheme, it does not
# define one. Three places, in this order:
#
#   1. SCHEMES              add an entry, e.g.
#                               "zetagpt-xl": dict(n_layer=32, n_head=20, n_embd=1280,
#                                                  context_window=2048),
#                           keeping n_embd a multiple of 64 and n_head = n_embd / 64, or the
#                           model warns that its head dimension is non-standard.
#   2. PRETRAIN_CORPUS      add the same key, pointing at the corpus that size should train on
#                           (an empty string means "configure PRETRAIN_DIR", which is what the
#                           larger shipped schemes do).
#   3. helpers/utils.py     add (n_layer, n_head, n_embd) -> name to MODEL_SIZES, so the run's
#                           checkpoints, histories and figures are named after your scheme
#                           instead of falling back to a descriptive zetagpt-<L>x<d>.
#
# The new name then appears in MODEL_SCHEME below and in --model_scheme automatically; nothing
# here needs a matching edit.
MODEL_SCHEME="${MODEL_SCHEME:-}"    # zetagpt-tiny | zetagpt-s | zetagpt-m | zetagpt-l | your own

# CONTEXT WINDOW IN TOKENS, blank = the scheme's own. This is the one part of a scheme it is
# legitimate to vary alone, because it is a TRAINING choice rather than an architectural one:
# nothing in ZetaGPT refers to an absolute position, so no length is forbidden and a trained
# model can be evaluated at any length. What it costs is attention, which is quadratic in it,
# so 512 -> 1024 is roughly 4x the attention term per step; that is why the small scheme takes
# 512 and the two larger ones take 1024. The value chosen here is what the checkpoint records
# as its block_size, and it is also what --max_len follows when MAX_LEN is left blank.
CONTEXT_WINDOW="${CONTEXT_WINDOW:-}"

PRETRAIN_FLAGS="${PRETRAIN_FLAGS:-}"

# =========================================================================== #
# stage 5 -- supervised fine-tuning
# =========================================================================== #
SFT_STEPS="${SFT_STEPS:-}"
SFT_LR="${SFT_LR:-}"
SFT_FLAGS="${SFT_FLAGS:-}"

# =========================================================================== #
# stage 6 -- reward model (sigmoid + BCE)
# =========================================================================== #
REWARD_STEPS="${REWARD_STEPS:-}"
REWARD_LR="${REWARD_LR:-}"
REWARD_FLAGS="${REWARD_FLAGS:-}"

# =========================================================================== #
# stage 7 -- RLHF by PPO
# =========================================================================== #
RLHF_STEPS="${RLHF_STEPS:-}"
RLHF_LR="${RLHF_LR:-}"
RLHF_FLAGS="${RLHF_FLAGS:-}"

# =========================================================================== #
# stage 8 -- chain of thought by GRPO (the aha moment)
# =========================================================================== #
COT_STEPS="${COT_STEPS:-}"
COT_LR="${COT_LR:-}"
# WHICH CHECKPOINT THE POLICY STARTS FROM. "pretrain" is the R1-Zero setting -- reinforcement
# learning on the base model with no supervised reasoning anywhere in the pipeline, which is
# the run that is supposed to produce the aha moment. "sft" makes the stage a sibling of RLHF
# and DPO; "rlhf" or "dpo" continue an already aligned policy.
COT_INIT="${COT_INIT:-}"            # pretrain | sft | rlhf | dpo
COT_GROUP="${COT_GROUP:-}"          # completions sampled per problem (GRPO's baseline, >= 2)
COT_FLAGS="${COT_FLAGS:-}"

# =========================================================================== #
# stage 9 -- direct preference optimisation
# =========================================================================== #
DPO_STEPS="${DPO_STEPS:-}"
DPO_LR="${DPO_LR:-}"
DPO_FLAGS="${DPO_FLAGS:-}"

# =========================================================================== #
# distillation into gpt2-small -- OPTIONAL, not a numbered stage:
#     python -m distill.run
# =========================================================================== #
DISTILL_STEPS="${DISTILL_STEPS:-}"
DISTILL_LR="${DISTILL_LR:-}"
DISTILL_FLAGS="${DISTILL_FLAGS:-}"

# =========================================================================== #
# your settings: config_user.yaml overrides everything above
# =========================================================================== #
# Written by `python config_wizard.py`, git-ignored, absent by default. Sourced (not run) so
# that the exports land in this shell. Environment variables set before this script started
# are left alone; see ZETAGPT_ENV_SET at the top.
ZETAGPT_YAML="${ZETAGPT_YAML:-$(dirname "${BASH_SOURCE[0]}")/config_user.yaml}"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/export_yaml.sh" "$ZETAGPT_YAML"

# =========================================================================== #
# command lines, assembled from whatever survived the layers above
# =========================================================================== #
# =========================================================================== #
# scaling laws -- NOT a pipeline stage; run by ./run_scaling_laws.sh
# =========================================================================== #
# Loss against model size and data size, on WikiText-103 (downloaded to data/download/, which
# is git-ignored). The ladder is ZetaGPT-S shrunk, so every rung differs in depth and width
# alone; see scaling_laws/grid.py. Blank = default_config.SCALING.
SCALING_MODELS="${SCALING_MODELS:-}"      # e.g. zs/8,zs/6,zs/4,zs/2,zs
SCALING_BUDGETS="${SCALING_BUDGETS:-}"    # token budgets, e.g. 2e6,8e6,32e6,100e6
SCALING_CONTEXT="${SCALING_CONTEXT:-}"    # held FIXED across the grid
SCALING_BATCH="${SCALING_BATCH:-}"
SCALING_LR="${SCALING_LR:-}"              # peak rate at the reference width
SCALING_LR_RULE="${SCALING_LR_RULE:-}"    # sqrt_width | fixed
SCALING_FLAGS="${SCALING_FLAGS:-}"

[ -n "$GPU" ]           && COMMON_FLAGS="$COMMON_FLAGS --gpu $GPU"
[ -n "$SEED" ]          && COMMON_FLAGS="$COMMON_FLAGS --seed $SEED"
[ -n "$BATCH" ]         && COMMON_FLAGS="$COMMON_FLAGS --batch $BATCH"
[ -n "$MICRO_BATCH" ]   && COMMON_FLAGS="$COMMON_FLAGS --micro_batch $MICRO_BATCH"
[ -n "$DATASET" ]       && COMMON_FLAGS="$COMMON_FLAGS --dataset $DATASET"
[ -n "$LIMIT" ]         && COMMON_FLAGS="$COMMON_FLAGS --limit $LIMIT"
[ -n "$VAL_FRAC" ]      && COMMON_FLAGS="$COMMON_FLAGS --val_frac $VAL_FRAC"
[ -n "$BETA" ]          && COMMON_FLAGS="$COMMON_FLAGS --beta $BETA"
[ -n "$MAX_LEN" ]       && COMMON_FLAGS="$COMMON_FLAGS --max_len $MAX_LEN"
[ -n "$PRETRAIN_DIR" ]  && COMMON_FLAGS="$COMMON_FLAGS --pretrain_dir $PRETRAIN_DIR"
[ -n "$INSTRUCT_DIR" ]  && COMMON_FLAGS="$COMMON_FLAGS --instruct_dir $INSTRUCT_DIR"
[ -n "$SFT_DIR" ]       && COMMON_FLAGS="$COMMON_FLAGS --sft_dir $SFT_DIR"
[ -n "$LR_SCHEDULE" ]   && COMMON_FLAGS="$COMMON_FLAGS --lr_schedule $LR_SCHEDULE"
[ -n "$LR_MIN_FACTOR" ] && COMMON_FLAGS="$COMMON_FLAGS --lr_min_factor $LR_MIN_FACTOR"
[ -n "$PLOT_EVERY" ]    && COMMON_FLAGS="$COMMON_FLAGS --plot_every_steps $PLOT_EVERY"
[ -n "$CKPT_EVERY" ]    && COMMON_FLAGS="$COMMON_FLAGS --checkpoint_every_steps $CKPT_EVERY"
[ -n "$PE" ]            && COMMON_FLAGS="$COMMON_FLAGS --pe $PE"
[ -n "$SSM_STATS_EVERY" ] && COMMON_FLAGS="$COMMON_FLAGS --ssm_stats_every $SSM_STATS_EVERY"
[ -n "$NO_RESUME" ]     && COMMON_FLAGS="$COMMON_FLAGS --no-resume"
[ -n "$EVAL_EVERY" ]    && COMMON_FLAGS="$COMMON_FLAGS --eval_every $EVAL_EVERY"
[ -n "$EVAL_PAIRS" ]    && COMMON_FLAGS="$COMMON_FLAGS --eval_pairs $EVAL_PAIRS"
[ -n "$EB_EVERY" ]      && COMMON_FLAGS="$COMMON_FLAGS --eb_every $EB_EVERY"
[ -n "$EB_PAIRS" ]      && COMMON_FLAGS="$COMMON_FLAGS --eb_pairs $EB_PAIRS"
[ -n "$ROLLOUT_TEMP" ]  && COMMON_FLAGS="$COMMON_FLAGS --rollout_temp $ROLLOUT_TEMP"
[ -n "$N_HIST" ]        && COMMON_FLAGS="$COMMON_FLAGS --n_hist $N_HIST"
[ -n "$N_ROLL" ]        && COMMON_FLAGS="$COMMON_FLAGS --n_roll $N_ROLL"
[ -n "$ROLL_TOKENS" ]   && COMMON_FLAGS="$COMMON_FLAGS --roll_tokens $ROLL_TOKENS"
[ -n "$P_GRID" ]        && COMMON_FLAGS="$COMMON_FLAGS --p_grid $P_GRID"
[ -n "$BPE_MERGES" ] && BPE_FLAGS="$BPE_FLAGS --bpe_merges $BPE_MERGES"
[ -n "$MODEL_SCHEME" ]   && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --model_scheme $MODEL_SCHEME"
[ -n "$CONTEXT_WINDOW" ] && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --context_window $CONTEXT_WINDOW"
[ -n "$PRETRAIN_STEPS" ] && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --pretrain_steps $PRETRAIN_STEPS"
[ -n "$PRETRAIN_LR" ]    && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --pretrain_lr $PRETRAIN_LR"
[ -n "$SFT_STEPS" ] && SFT_FLAGS="$SFT_FLAGS --sft_steps $SFT_STEPS"
[ -n "$SFT_LR" ]    && SFT_FLAGS="$SFT_FLAGS --sft_lr $SFT_LR"
[ -n "$REWARD_STEPS" ] && REWARD_FLAGS="$REWARD_FLAGS --reward_steps $REWARD_STEPS"
[ -n "$REWARD_LR" ]    && REWARD_FLAGS="$REWARD_FLAGS --reward_lr $REWARD_LR"
[ -n "$RLHF_STEPS" ] && RLHF_FLAGS="$RLHF_FLAGS --rlhf_steps $RLHF_STEPS"
[ -n "$RLHF_LR" ]    && RLHF_FLAGS="$RLHF_FLAGS --rlhf_lr $RLHF_LR"
[ -n "$COT_STEPS" ] && COT_FLAGS="$COT_FLAGS --cot_steps $COT_STEPS"
[ -n "$COT_LR" ]    && COT_FLAGS="$COT_FLAGS --cot_lr $COT_LR"
[ -n "$COT_INIT" ]  && COT_FLAGS="$COT_FLAGS --cot_init $COT_INIT"
[ -n "$COT_GROUP" ] && COT_FLAGS="$COT_FLAGS --cot_group $COT_GROUP"
[ -n "$DPO_STEPS" ] && DPO_FLAGS="$DPO_FLAGS --dpo_steps $DPO_STEPS"
[ -n "$DPO_LR" ]    && DPO_FLAGS="$DPO_FLAGS --dpo_lr $DPO_LR"
[ -n "$DISTILL_STEPS" ] && DISTILL_FLAGS="$DISTILL_FLAGS --distill_steps $DISTILL_STEPS"
[ -n "$DISTILL_LR" ]    && DISTILL_FLAGS="$DISTILL_FLAGS --distill_lr $DISTILL_LR"
[ -n "$SCALING_MODELS" ]   && SCALING_FLAGS="$SCALING_FLAGS --models $SCALING_MODELS"
[ -n "$SCALING_BUDGETS" ]  && SCALING_FLAGS="$SCALING_FLAGS --budgets $SCALING_BUDGETS"
[ -n "$SCALING_CONTEXT" ]  && SCALING_FLAGS="$SCALING_FLAGS --context $SCALING_CONTEXT"
[ -n "$SCALING_BATCH" ]    && SCALING_FLAGS="$SCALING_FLAGS --batch $SCALING_BATCH"
[ -n "$SCALING_LR" ]       && SCALING_FLAGS="$SCALING_FLAGS --lr $SCALING_LR"
[ -n "$SCALING_LR_RULE" ]  && SCALING_FLAGS="$SCALING_FLAGS --lr_rule $SCALING_LR_RULE"


# =========================================================================== #
# what this run is actually configured with
# =========================================================================== #
# Every knob, one per line, with the value in force and where it came from -- so the head of a
# log says what produced it, without anyone having to reconstruct four layers of precedence
# from memory. A blank value means the flag is not passed at all and the trainer falls back to
# default_config.py. Set ZETAGPT_QUIET=1 to suppress this (run_full.sh chains eight stages).
# What default_config.py falls back to when a variable here is left blank -- asked for once,
# so the summary can print the value actually in force instead of a dash. Only lines that
# look like assignments are evaluated, and a failure (no interpreter, a broken import) leaves
# the defaults unknown rather than killing the run.
ZETAGPT_DEF_LOADED=""
if [ -z "${ZETAGPT_QUIET:-}" ]; then
  while IFS= read -r _line; do
    case "$_line" in
      ZETAGPT_DEF_[A-Z_]*=*) eval "$_line"; ZETAGPT_DEF_LOADED=1 ;;
    esac
  done <<< "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/default_config.py" --shell-defaults \
              2>/dev/null || true)"
  unset _line
fi

if [ -z "${ZETAGPT_QUIET:-}" ]; then
  echo "--- configuration ------------------------------------------------------------"
  for _v in $ZETAGPT_VARS; do
    _val="${!_v}"
    case " $ZETAGPT_ENV_SET " in
      *" $_v "*) _src="environment" ;;
      *) case " ${ZETAGPT_YAML_SET:-} " in
           *" $_v "*) _src="config_user.yaml" ;;
           *) case "$_v" in
                # the *_FLAGS lines are built from the knobs above, not configured directly
                *_FLAGS) _src="assembled" ;;
                *) if [ -n "$_val" ]; then
                     _src="config.sh"
                   else
                     # blank here means the flag is not passed and default_config.py decides:
                     # print ITS value, which is the one actually in force
                     _d="ZETAGPT_DEF_$_v"
                     _val="${!_d:-}"
                     _src="default_config.py"
                     [ -n "$_val" ] || [ -n "$ZETAGPT_DEF_LOADED" ] || _src="default_config.py (unread)"
                   fi ;;
              esac ;;
         esac ;;
    esac
    [ -n "$_val" ] || _val="-"
    printf '  %-16s %-46s %s\n' "$_v" "$_val" "$_src"
  done
  echo "------------------------------------------------------------------------------"
fi
unset _v _val _src _d

# The loop above can end on a failed test, and the exit status of the last command in a sourced
# file is the exit status of `source`. Every stage script runs under `set -e`, so without this
# line a stage would exit SILENTLY, before printing anything.
:
