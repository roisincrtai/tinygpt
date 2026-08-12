#!/usr/bin/env bash
# config.sh -- EVERY configurable argument, in one place, sourced by each stage<N>_*.sh.
#
# EVERY VARIABLE CARRIES ITS ACTUAL VALUE, not a blank standing for "whatever Python decides".
# You can read this file and know what the run will do; the summary printed at the end shows
# the same numbers, and never a dash.
#
#   command-line flag > environment variable > config_user.yaml > config.sh > default_config.py
#
#   ./stage5_pretrain.sh                        # this file's settings
#   PRETRAIN_STEPS=20000 ./stage5_pretrain.sh   # override one knob for one run
#   GPU=cpu ./stage6_instruct_sft.sh            # override a shared knob for one run
#   COT_INIT=sft ./stage9_cot_aha_moment.sh     # start the CoT stage from the SFT model
#
# YOUR OWN SETTINGS GO IN config_user.yaml, not here. `python -m tools.config_wizard` writes it; this
# file sources it near the end, through export_yaml.sh, so its values override what is below
# without editing a tracked file. It is git-ignored. Delete it and this file stands. A variable
# already set in the environment is never overwritten by the YAML.
#
# Each stage script also forwards its own arguments, so `./stage10_instruct_dpo.sh --beta 0.05`
# works too.
#
# SECTIONS, in order: interpreter; runtime; data; model; optimisation; reporting; probes;
# advanced overrides; then one section per stage; then the scaling study; then the YAML import,
# the assembly of the command lines, and the summary. <STAGE>_FLAGS in each stage section is
# free text appended verbatim, for anything without a variable of its own.

# The complete list, defined once: it drives the environment capture below and the summary
# printed at the end, so a knob added in one place cannot be forgotten by the other.
ZETAGPT_VARS="PY GPU SEED \
DATASET PRETRAIN_DIR INSTRUCT_DIR SFT_DIR DATA_LIMIT VAL_FRAC \
MODEL_SCHEME CONTEXT_WINDOW PE \
BATCH MICRO_BATCH MIXED_PRECISION CHUNKED_LOSS LOSS_CHUNK TENSOR_PARALLEL LR_SCHEDULE LR_MIN_FACTOR BETA \
PLOT_EVERY PRINT_SAMPLES_EVERY CKPT_EVERY SSM_STATS_EVERY NO_RESUME \
EVAL_EVERY EVAL_PAIRS EB_EVERY EB_PAIRS ROLLOUT_TEMP N_HIST N_ROLL ROLL_TOKENS P_GRID \
EXTRA_SET COMMON_FLAGS \
BPE_MERGES BPE_MIN_FREQ BPE_FLAGS \
CORPUS_MAX_WORDS CORPUS_TEXT_COLUMN TOKENIZE_FLAGS \
PRETRAIN_STEPS PRETRAIN_LR PRETRAIN_FLAGS \
SFT_STEPS SFT_LR SFT_BATCH SFT_MICRO_BATCH SFT_FLAGS \
REWARD_STEPS REWARD_LR REWARD_FLAGS \
RLHF_STEPS RLHF_LR RLHF_KL_COEF RLHF_MAX_NEW_TOKENS RLHF_FLAGS \
KV_CACHE KV_CACHE_SIZE COT_TASK COT_STEPS COT_LR COT_SFT COT_SFT_STEPS COT_SFT_LR COT_INIT COT_BATCH COT_GROUP COT_SAMPLES_EVERY COT_KL_COEF COT_MAX_NEW_TOKENS COT_FLAGS \
DPO_STEPS DPO_LR DPO_FLAGS \
LANG_SFT_DATASET LANG_SFT_DATASET_SHORTNAME LANG_SFT_MERGES LANG_SFT_STEPS LANG_SFT_LR LANG_SFT_FLAGS \
DISTILL_STEPS DISTILL_LR DISTILL_TEACHER DISTILL_STUDENT DISTILL_FLAGS \
SCALING_MODELS SCALING_BUDGETS SCALING_CONTEXT SCALING_BATCH SCALING_LR SCALING_LR_RULE \
SCALING_FLAGS \
MODEL_FFN_FACTOR MODEL_DROPOUT MODEL_GATED_ATTN MODEL_D_CONV MODEL_SSM_CHUNK CORPUS_EXCLUDE TOKENS_SHARD_MB SFT_SUBSETS SFT_LIMIT REWARD_SUBSETS REWARD_VAL_SUBSETS REWARD_LIMIT RLHF_GEN_TEMP RLHF_PPO_EPOCHS RLHF_CLIP_EPS RLHF_GAMMA RLHF_LAM RLHF_VF_COEF RLHF_ENT_COEF RLHF_WHITEN_ADV RLHF_PROMPT_LIMIT RLHF_PROMPTS_PER_FILE COT_GEN_TEMP COT_CLIP_EPS COT_GRPO_EPOCHS COT_ENT_COEF COT_CORRECT_REWARD COT_THINK_FORMAT_REWARD COT_ANSWER_FORMAT_REWARD COT_THINK_REWARD COT_LENGTH_PENALTY COT_LIMIT COT_EVAL_PROBLEMS COT_TRAIN_PREFIX COT_TEST_PREFIX COT_RECORDS_PER_FILE COT_TRACE_FIELD COT_TRACE_FIELD_GSM8K COT_SFT_VERIFY DISTILL_STUDENT_MAX_LEN DISTILL_MAX_NEW_TOKENS DISTILL_GEN_TEMP DISTILL_KL_COEF DISTILL_PROMPTS_PER_FILE SCALING_LR_REF_WIDTH SCALING_EVAL_EVERY SCALING_VAL_WINDOWS"

# Which variables came from the ENVIRONMENT, recorded before a single default is applied:
# export_yaml.sh must not overwrite these, or `GPU=cpu ./stage5_pretrain.sh` would lose to the
# YAML file. `${!v+x}` is set for an exported-but-empty variable too, which is deliberate:
# GPU= means "explicitly nothing", not "unset".
ZETAGPT_ENV_SET=""
for _v in $ZETAGPT_VARS; do
  [ -n "${!_v+x}" ] && ZETAGPT_ENV_SET="$ZETAGPT_ENV_SET $_v"
done
unset _v

# =========================================================================== #
# 1. interpreter
# =========================================================================== #
PY="${PY:-python}"

# =========================================================================== #
# 2. runtime -- where and how the process runs
# =========================================================================== #
GPU="${GPU:-auto}"                  # auto | cuda | mps | cpu
SEED="${SEED:-0}"                   # random seed; also seeds the data order, so a resumed
                                    # run continues the order it would have had

# =========================================================================== #
# 3. data -- LOCAL DIRECTORIES ONLY
# =========================================================================== #
# NOTHING IN THE PIPELINE DOWNLOADS. Fetch once, deliberately:
#
#     ./stage1_download_data.sh          every dataset, into data/download/
#     ./stage1_download_data.sh --list   what would be fetched, and where
#
# A stage that fetched its own corpus would have an unpinned corpus: not reproducible offline,
# failing hours in on a network error, and leaving a log from which nobody can tell what was
# trained on. Every stage reads a directory and stops if it is not there.

# PREFERENCE DATA. "hh" is the downloaded rlhf_hh tree; anything else is a PATH to your own
# folder of json/jsonl records carrying prompt / chosen / rejected. The layout is DETECTED: a
# tree with its own *_train and *_test subdirectories keeps that split, a plain folder is cut
# at VAL_FRAC below.
DATASET="${DATASET:-hh}"

# PRETRAINING CORPUS -- and, because stage 3 trains the vocabulary on it, THE TOKENIZER'S
# TRAINING CORPUS as well. Set here it applies to whatever scheme is selected; BLANK it to let
# each scheme use its own from default_config.PRETRAIN_CORPUS:
#
#   zetagpt-tiny   zetagpt-tiny_pretrain-corpus_wikitext103   ~105M tokens
#   zetagpt-s      zetagpt-pretrain_fineweb-edu-2BT           ~2B tokens (~10 GB)
#   zetagpt-m      zetagpt-pretrain_fineweb-edu-10BT          ~10B tokens
#   zetagpt-l      zetagpt-pretrain_fineweb-edu-10BT          ~10B tokens
#
# The ladder is not one corpus because ~20 tokens per parameter puts S at ~2.7B and L at ~12B,
# and a 61.5M model does not need 2B tokens while a 623M model is not fed by them.
#
# BLANK BY DEFAULT, AND THAT MATTERS. This shipped holding S's corpus, which is forwarded as
# --pretrain_dir and therefore WINS over the scheme's own -- so MODEL_SCHEME=zetagpt-m trained
# the 169M model on S's 2B-token subset, silently, whatever PRETRAIN_CORPUS said. One value
# pinned across every scheme is the same fault MAX_LEN had. Set a path here only to override
# the ladder deliberately, for your own corpus.
PRETRAIN_DIR="${PRETRAIN_DIR:-}"

# INSTRUCTION-TUNING TREE, shared by stages 6, 7, 8 and 10 and by distillation. Its layout is
# flat: alpaca_gpt4/ and rlhf_hh/ at its root. Moving this root moves everything derived from
# it -- the fine-tuning data, the preferences and the prompt bank -- together, so no two
# stages can read different datasets.
INSTRUCT_DIR="${INSTRUCT_DIR:-data/download/zetagpt-rlhf-instruction_following}"

# FINE-TUNING DATA for stage 6: the Tulu 3 SFT mixture -- 939k conversations, shipped as six
# parquet shards of `messages` turns. It is NOT the preference tree above and not derived from
# it: stages 7, 8 and 10 read rlhf_hh, stage 6 reads demonstrations. The layout is detected, so
# pointing this at an hh tree or at your own folder of json records still works.
SFT_DIR="${SFT_DIR:-data/download/zetagpt-instruction-following-sft-tulu-3-mixture}"

DATA_LIMIT="${DATA_LIMIT:-0}"       # cap #preference pairs loaded; 0 = all
VAL_FRAC="${VAL_FRAC:-0.05}"        # validation fraction, used only for a layout that ships
                                    # no split of its own

# =========================================================================== #
# 4. model -- what is built, read by EVERY stage
# =========================================================================== #
# These belong to the model, not to pretraining, and every stage must agree on them: the run
# tag that names every checkpoint, history and figure is derived from them, so a stage that
# did not receive them would look for a checkpoint filename no stage ever wrote.

# WHICH SIZE IS TRAINED. A scheme fixes depth, width and the context window TOGETHER:
#
#     scheme          layers  heads  d_model  d_h  context  parameters
#     zetagpt-tiny         8      8      512   64      512       61.5M
#     zetagpt-s           24      8      512   64   1k..8k     133.0M   (default)
#     zetagpt-m           32      8      512   64   1k..16k     168.8M
#     zetagpt-l           32     16     1024   64   1k..32k     622.7M
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
#                                                  context_window=[1024, 4096, 16384]),
#                           keeping n_embd a multiple of 64 and n_head = n_embd / 64, or the
#                           model warns that its head dimension is non-standard.
#                           context_window is the LIST of windows the scheme trains through,
#                           in order, each taking an equal share of the steps; the largest is
#                           the model's block_size. The values are arbitrary -- [768, 3000,
#                           5000] is as valid as [1024, 4096]. A single number is accepted and
#                           means one window throughout.
#   2. PRETRAIN_CORPUS      add the same key, pointing at the corpus that size should train on
#                           (an empty string means "configure PRETRAIN_DIR").
#   3. SCHEME_BATCH         and SCHEME_MICRO_BATCH: what fits a card at the LONGEST window.
#
# Nothing else. The name a run reports and writes its files under is looked up from SCHEMES
# itself, so a new size names itself.
#
# The new name then works here and in --model_scheme automatically.
MODEL_SCHEME="${MODEL_SCHEME:-zetagpt-s}"

# CONTEXT WINDOW IN TOKENS. The one part of a scheme it is legitimate to vary alone, because it
# is a TRAINING choice rather than an architectural one: nothing in ZetaGPT refers to an
# absolute position, so no length is forbidden and a trained model can be evaluated at any
# length. What it costs is attention, which is quadratic in it, so 512 -> 1024 is roughly 4x
# the attention term per step. This is what the checkpoint records as its block_size.
CONTEXT_WINDOW="${CONTEXT_WINDOW:-}"  # blank = the scheme's own schedule. A COMMA LIST pins
                                    # your own: CONTEXT_WINDOW=1024,4096,8192 trains through
                                    # those three in order, an equal share of the steps each.
                                    # A single number pins one window for the whole run. The
                                    # values are arbitrary -- nothing requires powers of two,
                                    # or that each is a multiple of the last.

# HOW POSITION ENTERS THE MODEL. "ssm" is the proposed architecture: the state space module
# supplies position and no positional encoding is used anywhere. "rope" is the ABLATION
# CONTROL: the module is removed and rotary positions are applied inside attention instead.
# The choice is recorded in the checkpoint and appears in every filename the run writes, so an
# ablation never overwrites the run it is compared with.
#   PE=rope ./stage5_pretrain.sh
PE="${PE:-ssm}"                     # ssm | rope

# THE CONTEXT LENGTH IS NOT CONFIGURED HERE. It is the model's own -- CONTEXT_WINDOW above, or
# the scheme's -- and it travels inside every checkpoint, so a stage that continues from one
# adopts that checkpoint's context rather than a number repeated in a second file.
#
# There WAS a MAX_LEN here and the two disagreed twice, quietly both times: 256 against a
# 512-token window, then 512 against the 8,192 the schemes grew to. The second cut 99% of
# stage 9's chain-of-thought demonstrations to their last 512 tokens and capped every GRPO
# rollout at ~308 new tokens. To train at less than the full window, set CONTEXT_WINDOW.

# The rest of the architecture. Depth, width and the context window come from the scheme
# above; these are the choices a scheme does not fix.
MODEL_FFN_FACTOR="${MODEL_FFN_FACTOR:-4}"     # MLP hidden size = this x d_model
MODEL_DROPOUT="${MODEL_DROPOUT:-0.0}"         # 0 is right for a single pass over a large corpus
MODEL_GATED_ATTN="${MODEL_GATED_ATTN:-1}"     # 1 = sigmoid gate on the attention output
MODEL_D_CONV="${MODEL_D_CONV:-4}"             # depthwise conv width inside the state space module
MODEL_SSM_CHUNK="${MODEL_SSM_CHUNK:-optimal}" # blocked-scan chunk; "optimal" = sqrt(T)

# =========================================================================== #
# 5. optimisation -- shared by every trainer; per-stage rates are in each stage's section
# =========================================================================== #
# BATCH SIZE, chosen to fill a 40 GB card and leave headroom. For zetagpt-s at context 512
# the fixed cost is ~1.8 GB (weights plus AdamW's five copies) and each sequence in the batch
# costs ~614 MB, of which the vocabulary-sized loss tensors are the largest part:
#
#     batch 32   ~21 GB      batch 56   ~35 GB  (88% of 40 GB)
#     batch 48   ~30 GB      batch 64   ~40 GB  (no headroom -- expect OOM)
#
# 32, for a card with ~36 GB usable, BUDGETED TO ~30 GB. The gap is deliberate: a card's nameplate size is
# not what a run may use. Fragmentation, allocator rounding, kernel workspaces, a second
# process and the driver's own reservation all come out of the same 44 GB, and a step that
# fits on average but not at its peak fails hours in rather than at once.
#
# Where it goes at zetagpt-s / context 512, anchored on a real out-of-memory rather than on
# arithmetic alone (the run held 43.9 GiB and asked for 5.4 GiB more at batch 56):
#
#     weights + AdamW state    1.8 GiB   fixed: 5 param-sized copies (live, grad, fp32
#                                        master, 2 moments) at 133.0M parameters
#     per sequence            0.85 GiB   of which 0.29 GiB is the loss alone --
#                                        3 x T x 50,259 in fp32: logits, log-softmax, its
#                                        gradient. The largest single term at this size.
#
#     batch 24  22.1 GiB      batch 36  32.3 GiB  over budget
#     batch 32  28.9 GiB      batch 40  35.7 GiB  over budget
#                             batch 56  49.2 GiB  out of memory, observed
#
# MEASURE with `python -m tools.vram --sweep` on the card you will train on before raising
# this. MICRO_BATCH splits the step instead and cuts the loss tensors proportionally, at no
# cost to the result. RECOMPUTE PRETRAIN_STEPS whenever this changes.
# MIXED PRECISION -- ON BY DEFAULT, AND THE HARDWARE DECIDES. bfloat16 for the activations, the
# matmuls and the backward pass; fp32 for the weights, the optimiser state, the gradient
# accumulation and every reduction -- layer norm statistics, softmax, log-softmax, logsumexp,
# the loss. That split is the standard recipe: full-fp32 pretraining is obsolete at scale,
# costing roughly twice the memory and much of the speed for no quality gain.
#
# bf16 AND NOT fp16, because bf16 keeps fp32's eight exponent bits and spends mantissa instead:
# gradients neither overflow nor underflow and NO LOSS SCALING IS NEEDED. And fp32 masters and
# not pure bf16, because once the weights are large next to the updates an update below bf16's
# ~3 significant digits rounds away entirely and training stalls with a curve that looks merely
# flat -- which is what the fp32 master copy exists to prevent.
#
# APPLIED ONLY ON CUDA. bfloat16 exists as a dtype on CPU and MPS but without the hardware paths
# it is emulated, slower than the fp32 it replaced; the run says which it got rather than
# leaving the flag reading "on" while every forward ignores it. 0 trains in fp32 everywhere.
MIXED_PRECISION="${MIXED_PRECISION:-1}"

BATCH="${BATCH:-}"                  # blank = the scheme's own (SCHEME_BATCH); a number here wins
MICRO_BATCH="${MICRO_BATCH:-0}"     # sequences per forward pass; 0 = OFF, the whole batch at
                                    # once. Splits one step into several passes whose gradients
                                    # accumulate: the optimiser sees the same step, the card
                                    # holds one group's activations. Costs time, not accuracy.
                                    # zetagpt-l needs 1 at its longest window; the other three
                                    # schemes do not need it at all.

# CHUNKED LOSS. The vocabulary projection is the largest tensor in a step: logits, their
# log-softmax and its gradient are each BATCH x CONTEXT x 50,259 in fp32, which at context
# 32,768 is 6.1 GB apiece. Evaluating the projection LOSS_CHUNK positions at a time and summing
# gives the identical number -- same positions, same targets, same normalisation -- with a peak
# of 3 x LOSS_CHUNK x 50,259 instead. The backward pass recomputes each slice, so the trade is
# roughly 30% more time in the loss for 30x less memory in it.
#
#   4.6 GiB at LOSS_CHUNK=8192, against 18.4 GiB for a single 32,768-token sequence and
#   far more once a batch multiplies it. The slice is in TOKENS, so it bounds the
#   projection whatever shape the batch has.
#
# Set CHUNKED_LOSS=0 to project the whole sequence at once, which is what to do when comparing
# against a run made before this existed.
CHUNKED_LOSS="${CHUNKED_LOSS:-1}"   # 1 = slice the projection (default), 0 = whole sequence
LOSS_CHUNK="${LOSS_CHUNK:-8192}"    # TOKENS per slice (4.6 GiB at this size)

# PARALLELISM: THE MODEL'S LAYERS ACROSS EVERY VISIBLE GPU, on by default.
#
# With 2 GPUs and 16 blocks, blocks 0-7 go on cuda:0 and blocks 8-15 on cuda:1; the hidden
# states are copied across at the boundary, one (batch, tokens, d_model) tensor per crossing.
# The embedding and the vocabulary projection are ONE TIED TENSOR and share the last device.
#
# THIS BUYS MEMORY, NOT SPEED. The devices take turns -- cuda:1 idles while cuda:0 runs its
# half and then the reverse -- so two cards give roughly the memory of two and the throughput
# of one. It is the right thing when a model or a context window does not fit ONE card at all
# (for zetagpt-l that is every window past 4,096; see doc/vram_usage.md), and the wrong thing
# when the run already fits, where one card is simply faster.
#
# Set 0, or pass --no-tensor_parallel, to keep the whole model on one GPU. On a single-GPU
# machine the setting makes no difference: there is nothing to split.
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"   # 1 = split layers across GPUs (default), 0 = one GPU

# INCREMENTAL DECODING, for every stage that generates: rollouts, previews and chat.
#
# Without a cache, producing n tokens recomputes the whole prefix n times -- O(n^2) forward
# work. With one, each step extends what is already there. This model needs FOUR things cached
# per layer, not two: attention's keys and values, and the state space module's convolution
# window and recurrence state, because a block is SSM -> attention -> FFN and the recurrence
# would otherwise be re-run over the prefix regardless.
#
# KV_CACHE_SIZE BOUNDS IT. Attention's share grows with every token, and a GRPO step decodes
# batch x group_size sequences at once, so an unbounded cache is tens of gigabytes at a long
# window. The rollout is split into groups that fit, decoded one group at a time; sequences are
# independent, so the completions are identical and only the peak changes.
KV_CACHE="${KV_CACHE:-1}"           # 1 = cache while generating (default), 0 = recompute
KV_CACHE_SIZE="${KV_CACHE_SIZE:-2}" # what the cache may hold, GiB (a float is fine)
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"        # cosine | constant
LR_MIN_FACTOR="${LR_MIN_FACTOR:-10.0}"      # cosine floor: minimum lr = stage lr / this
BETA="${BETA:-0.1}"                 # implicit-reward beta, shared by DPO and evaluation

# =========================================================================== #
# 6. reporting -- figures, checkpoints, resume
# =========================================================================== #
PLOT_EVERY="${PLOT_EVERY:-200}"     # redraw the stage figure every N steps; 0 = off
PRINT_SAMPLES_EVERY="${PRINT_SAMPLES_EVERY:-200}"   # print 20 generation examples every N
                                    # steps, in every stage that generates -- pretrain, SFT,
                                    # CoT SFT, RLHF, CoT GRPO, DPO, distillation. Separate from
                                    # PLOT_EVERY because a figure is glanced at and twenty
                                    # generations are read. 0 = off.
CKPT_EVERY="${CKPT_EVERY:-200}"     # checkpoint every N steps
SSM_STATS_EVERY="${SSM_STATS_EVERY:-200}"   # state space diagnostics -- memory horizon,
                                    # selectivity, residual write ratio -- recorded PER LAYER
                                    # for one step every N, drawn in the pretraining figure.
                                    # 0 turns them off. Read by stage 5.
NO_RESUME="${NO_RESUME:-0}"         # 1 = ignore checkpoints and start from scratch

# =========================================================================== #
# 7. diagnostic probes -- read by the DPO stage and by eval.py
# =========================================================================== #
EVAL_EVERY="${EVAL_EVERY:-0}"       # held-out probe every N steps; 0 = follow PLOT_EVERY
EVAL_PAIRS="${EVAL_PAIRS:-32}"      # pairs per held-out probe
EB_EVERY="${EB_EVERY:-25}"          # exposure-bias probe every N steps; 0 = off
EB_PAIRS="${EB_PAIRS:-4}"           # pairs per exposure-bias probe
ROLLOUT_TEMP="${ROLLOUT_TEMP:-1.0}" # sampling temperature of that probe
N_HIST="${N_HIST:-256}"             # end-of-stage evaluation: histories scored
N_ROLL="${N_ROLL:-32}"              # end-of-stage evaluation: rollouts
ROLL_TOKENS="${ROLL_TOKENS:-48}"    # end-of-stage evaluation: tokens per rollout
P_GRID="${P_GRID:-0.0,0.5,1.0}"     # end-of-stage evaluation: comma-separated p values

# =========================================================================== #
# 8. advanced -- reach anything in default_config.py without a variable here
# =========================================================================== #
# EVERY configurable value in default_config.py's dictionaries has a named variable in this
# file -- all 77 of them, 35 through their own command-line flag and 51 through --set. This
# section is for what remains: a value added to default_config.py since, or one you would
# rather not give a permanent name.
#
# --set takes the type of the value already there and refuses an unknown section or key rather
# than ignoring it, because a silently dropped override is a run that reports settings it did
# not use.
#
#   EXTRA_SET="--set SCALING.val_windows=128"
#
# Sections: BPE PRETRAIN SFT REWARD RLHF COT DPO DISTILL TRAIN SCALING MODEL
EXTRA_SET="${EXTRA_SET:-}"

COMMON_FLAGS="${COMMON_FLAGS:-}"

# =========================================================================== #
# stage 3 -- byte-level BPE tokenizer
# =========================================================================== #
BPE_MERGES="${BPE_MERGES:-50000}"   # merge budget; raising it EXTENDS an existing tokenizer.
                                    # 50,000 is GPT-2's count.
# MINIMUM WORD FREQUENCY for a word to be counted when merges are learned. Half the distinct
# words in a corpus this size appear EXACTLY ONCE and are under 2% of the running text, while
# costing as much to maintain as the commonest word -- the merge loop's cost scales with the
# number of distinct words a pair occurs in. Nothing leaves the vocabulary: a byte-level
# tokenizer encodes a word it never counted. 1 counts every word.
BPE_MIN_FREQ="${BPE_MIN_FREQ:-2}"
BPE_FLAGS="${BPE_FLAGS:-}"

# =========================================================================== #
# stage 4 -- tokenise the corpora into memory-mapped .tokens streams
# =========================================================================== #
# Reads PRETRAIN_DIR above and writes cache/tokens/bpe_<256+merges>_<fingerprint>/. Re-running
# is free when nothing changed. These two also govern how stage 5 reads the corpus, so they
# live here rather than in the pretraining section.
CORPUS_MAX_WORDS="${CORPUS_MAX_WORDS:-344}"      # words per packed document; ~1.5 tokens per
                                                 # word fills a 512-token window
CORPUS_TEXT_COLUMN="${CORPUS_TEXT_COLUMN:-text}" # which parquet column holds the text
TOKENIZE_FLAGS="${TOKENIZE_FLAGS:-}"

CORPUS_EXCLUDE="${CORPUS_EXCLUDE:-test,valid,validation}"  # held-out splits kept OUT of
                                    # training: a directory of that name, or a file whose
                                    # leading name token matches (validation-00000-of-...)
TOKENS_SHARD_MB="${TOKENS_SHARD_MB:-100}"    # MB of tokens per shard. A corpus can be 10 TB;
                                    # one file that size cannot be resumed, cannot be copied
                                    # incrementally, and loses everything to one bad byte.

# =========================================================================== #
# stage 5 -- pretraining
# =========================================================================== #
# 2 epochs over the measured 2,068,028,808-token corpus at 3 x 8191 = 24,573 tokens per step,
# which the context schedule holds constant at every window, so 168,317 steps is 4.14B tokens.
# CHANGE THIS WHENEVER BATCH CHANGES, or the budget silently stops being two epochs.
PRETRAIN_STEPS="${PRETRAIN_STEPS:-168317}"
PRETRAIN_LR="${PRETRAIN_LR:-2e-5}"
PRETRAIN_FLAGS="${PRETRAIN_FLAGS:-}"

# =========================================================================== #
# stage 6 -- supervised fine-tuning
# =========================================================================== #
# THIS STAGE SIZES ITS OWN BATCH, and must. SCHEME_BATCH is 3 for zetagpt-s because a
# PRETRAINING sequence fills the whole 8,192-token window; an instruction demonstration is a few
# hundred tokens, so the same 3 would leave the card almost empty and make one epoch 313,115
# steps. Passing --batch here also stops default_config.SCHEME_BATCH being applied at all
# (helpers.common._was_given), which is the mechanism, not a side effect.
#
# BATCH= still wins, because it falls through to it: a person who pinned a batch for the whole
# run meant it for this stage too.
SFT_BATCH="${SFT_BATCH:-${BATCH:-32}}"
# ONE SEQUENCE PER FORWARD PASS. A batch is padded to the LONGEST record in it and this corpus
# is ragged -- most demonstrations are a few hundred tokens, a few are thousands -- so a fixed
# number of sequences per pass does not bound the tokens per pass, which is what memory is
# actually linear in. One does bound it: a single record is at most the context window, 8,192
# tokens, comfortably inside the 12,288 that SCHEME_MICRO_TOKENS measured for this scheme.
#
# WHAT IT COSTS is throughput, not accuracy: the gradients of 32 passes add up in the parameters
# to exactly the step the whole batch would have given (helpers.lm weights each pass by its
# share), but a pass carrying one short sequence uses little of the card. Raise it if the run is
# slower than the memory headroom justifies -- and know that at 4 a batch whose longest record
# is over ~3,000 tokens exceeds the measured budget.
#
# MICRO_BATCH= wins where it says something. It ships as 0, which is "off" for every other
# stage and is exactly what this stage must not inherit, so 0 is read as "unset" here rather
# than as a choice -- the one value that cannot have been meant for stage 6.
if [ -z "${SFT_MICRO_BATCH:-}" ]; then
    if [ "${MICRO_BATCH:-0}" != "0" ]; then SFT_MICRO_BATCH="$MICRO_BATCH"
    else SFT_MICRO_BATCH=1; fi
fi
# ONE EPOCH of the Tulu 3 mixture: ceil(939,343 conversations / 32 per step) = 29,355 steps.
# CHANGE THIS WHENEVER SFT_BATCH CHANGES, or the budget silently stops being one epoch. The
# stage measures the true epoch length once the corpus is loaded -- after the per-turn expansion
# and the deduplication -- and prints it beside this number, so the two never drift unnoticed.
SFT_STEPS="${SFT_STEPS:-29355}"
SFT_LR="${SFT_LR:-1e-6}"
SFT_FLAGS="${SFT_FLAGS:-}"

SFT_SUBSETS="${SFT_SUBSETS:-helpful_train,harmless_train}"  # split dirs read from an hh tree
SFT_LIMIT="${SFT_LIMIT:-0}"         # cap demonstrations loaded; 0 = all

# =========================================================================== #
# stage 7 -- reward model (sigmoid + BCE)
# =========================================================================== #
REWARD_STEPS="${REWARD_STEPS:-2500}"
REWARD_LR="${REWARD_LR:-1e-5}"
REWARD_FLAGS="${REWARD_FLAGS:-}"

REWARD_SUBSETS="${REWARD_SUBSETS:-helpful_train,harmless_train}"
REWARD_VAL_SUBSETS="${REWARD_VAL_SUBSETS:-helpful_test,harmless_test}"
REWARD_LIMIT="${REWARD_LIMIT:-20000}"   # pairs per subset; 0 = all (~86k train pairs)

# =========================================================================== #
# stage 8 -- RLHF by PPO
# =========================================================================== #
RLHF_STEPS="${RLHF_STEPS:-3251}"
RLHF_LR="${RLHF_LR:-1e-6}"
RLHF_KL_COEF="${RLHF_KL_COEF:-0.05}"            # per-token KL penalty to the frozen SFT
                                                # reference: the leash on how far the policy
                                                # may drift while chasing reward
RLHF_MAX_NEW_TOKENS="${RLHF_MAX_NEW_TOKENS:-48}"  # response length of each rollout
RLHF_FLAGS="${RLHF_FLAGS:-}"

RLHF_GEN_TEMP="${RLHF_GEN_TEMP:-1.0}"       # rollouts are SAMPLED; a greedy rollout is not a
                                            # draw from the policy and breaks the estimator
RLHF_PPO_EPOCHS="${RLHF_PPO_EPOCHS:-4}"     # optimisation passes over each batch of rollouts
RLHF_CLIP_EPS="${RLHF_CLIP_EPS:-0.2}"       # PPO ratio clip
RLHF_GAMMA="${RLHF_GAMMA:-1.0}"             # discount; 1.0 because a response is short
RLHF_LAM="${RLHF_LAM:-0.95}"                # GAE lambda
RLHF_VF_COEF="${RLHF_VF_COEF:-0.5}"         # value-loss weight
RLHF_ENT_COEF="${RLHF_ENT_COEF:-0.0}"       # entropy bonus
RLHF_WHITEN_ADV="${RLHF_WHITEN_ADV:-1}"     # 1 = normalise advantages within the batch
RLHF_PROMPT_LIMIT="${RLHF_PROMPT_LIMIT:-0}" # cap the prompt bank; 0 = all
RLHF_PROMPTS_PER_FILE="${RLHF_PROMPTS_PER_FILE:-1000}"

# =========================================================================== #
# stage 9 -- chain of thought by GRPO (the aha moment)
# =========================================================================== #
# WHICH TASK STAGE 9 LEARNS. "countdown" (default) is the Countdown number game: reach a
# target from a few numbers with + - * /, each number used at most once. "gsm8k" is the
# grade-school word problems. The task decides the directory read and how an answer is checked,
# and nothing else -- there is one GRPO loop and one reward shape.
#
# COUNTDOWN IS THE DEFAULT because GRPO can only amplify what the policy already samples
# sometimes. If every completion in a group scores zero the advantages are zero and nothing is
# learned, while the run still looks like training. Countdown has a short searchable solution
# and an answer a small model reaches by chance often enough for a group to have spread; it is
# also the task the smallest published R1-Zero reproductions used.
COT_TASK="${COT_TASK:-countdown}"   # countdown | gsm8k
COT_STEPS="${COT_STEPS:-10000}"
COT_LR="${COT_LR:-1e-6}"
# THE SUPERVISED SUB-STAGE, run before GRPO. It trains on the dataset's own reference traces,
# rewritten into this pipeline's <think>/<answer> form, and teaches only the FORMAT. Without it
# a base model never samples the format the reward pays for, so every completion in a group
# scores alike, every advantage is zero, and GRPO trains on nothing while its curves look fine.
# COT_SFT=0 turns it off, which is the R1-Zero setting.
COT_SFT="${COT_SFT:-1}"
COT_SFT_STEPS="${COT_SFT_STEPS:-7000}"   # ~1 epoch of the countdown demonstrations at batch 3
COT_SFT_LR="${COT_SFT_LR:-1e-6}"
COT_INIT="${COT_INIT:-pretrain}"    # pretrain | sft | rlhf | dpo -- which checkpoint the
                                    # SUPERVISED sub-stage starts from (and, with COT_SFT=0,
                                    # which one GRPO starts from). GRPO otherwise starts from
                                    # the cot-sft checkpoint.
COT_SAMPLES_EVERY="${COT_SAMPLES_EVERY:-50}"   # print the 5 rollout samples every N GRPO
                                    # steps. Tighter than PRINT_SAMPLES_EVERY (200): a GRPO
                                    # step costs a whole group of rollouts, there are far fewer
                                    # of them, and what this stage watches -- the format
                                    # appearing, the reasoning lengthening -- shows up in the
                                    # text before it shows up in a curve. 0 = follow
                                    # PRINT_SAMPLES_EVERY.
COT_BATCH="${COT_BATCH:-2}"         # problems per GRPO step. The rollout is this x
                                    # COT_GROUP sequences of up to the whole context window,
                                    # which is why this stage does not use the pretraining
                                    # batch: 2 x 4 = 8 rollouts is what fits 44 GiB at 8,192.
COT_GROUP="${COT_GROUP:-4}"         # completions sampled per problem (GRPO's baseline is
                                    # their mean, so this is the group it averages over)
COT_KL_COEF="${COT_KL_COEF:-0.001}" # much smaller than RLHF's: reasoning needs room to
                                    # explore before the verifier rewards it
# RESPONSE LENGTH FOR A ROLLOUT. Blank or 0 = whatever the context window has left once the
# prompt is in it, worked out per batch from the longest prompt in it. The windows of the four
# schemes run from 1,024 to 32,768, so one number cannot suit them all; and the behaviour this
# stage looks for is a policy that thinks for LONGER, which a fixed cap settles in advance.
# Put a number here to pin it, for a controlled comparison or to bound the cost of a rollout.
COT_MAX_NEW_TOKENS="${COT_MAX_NEW_TOKENS:-0}"   # 0 = fill the window
COT_FLAGS="${COT_FLAGS:-}"

COT_GEN_TEMP="${COT_GEN_TEMP:-1.0}"         # sampling temperature of each completion
COT_CLIP_EPS="${COT_CLIP_EPS:-0.2}"         # GRPO ratio clip
COT_GRPO_EPOCHS="${COT_GRPO_EPOCHS:-2}"     # optimisation passes per group of rollouts
COT_ENT_COEF="${COT_ENT_COEF:-0.0}"         # entropy bonus
COT_CORRECT_REWARD="${COT_CORRECT_REWARD:-2.0}"   # the answer is right
# THE FORMAT IS PAID FOR ONE TAG AT A TIME. A single both-or-nothing term leaves a policy that
# has learned to close <think> but not <answer> earning exactly what one that emits neither
# earns, so nothing pulls it the rest of the way -- and a base model has never seen this format.
# Together the two tags are worth what a correct answer is worth, which sounds generous and is
# not: GRPO's advantage is relative to the GROUP, so a term every completion earns cancels out
# of it entirely. The format terms teach while they vary and go quiet once the group has them.
COT_THINK_FORMAT_REWARD="${COT_THINK_FORMAT_REWARD:-1.0}"    # a closed <think>...</think>
COT_ANSWER_FORMAT_REWARD="${COT_ANSWER_FORMAT_REWARD:-1.0}"  # a closed <answer>...</answer>
COT_THINK_REWARD="${COT_THINK_REWARD:-0.3}"   # reasoning that is long enough AND mentions the
                                    # problem's numbers; a floor against tags around nothing
                                            # is worth less than producing a right one
COT_LENGTH_PENALTY="${COT_LENGTH_PENALTY:-0.0}"   # per-token cost of thinking
COT_LIMIT="${COT_LIMIT:-0}"                 # cap problems loaded; 0 = all
COT_EVAL_PROBLEMS="${COT_EVAL_PROBLEMS:-200}"     # held-out problems per evaluation
COT_TRAIN_PREFIX="${COT_TRAIN_PREFIX:-train}"     # filename prefixes of the two splits
COT_TEST_PREFIX="${COT_TEST_PREFIX:-test}"
COT_RECORDS_PER_FILE="${COT_RECORDS_PER_FILE:-1000}"
COT_TRACE_FIELD="${COT_TRACE_FIELD:-response}"    # the record field holding the reference
                                                  # reasoning trace, read by the SFT only.
                                                  # Countdown says "response"; gsm8k says
                                                  # "reasoning" (COT_TRACE_FIELD_GSM8K).
COT_TRACE_FIELD_GSM8K="${COT_TRACE_FIELD_GSM8K:-reasoning}"
COT_SFT_VERIFY="${COT_SFT_VERIFY:-True}"          # drop demonstrations whose own answer the
                                                  # verifier rejects (True | False)

# =========================================================================== #
# stage 10 -- direct preference optimisation
# =========================================================================== #
DPO_STEPS="${DPO_STEPS:-590}"
DPO_LR="${DPO_LR:-1e-6}"
DPO_FLAGS="${DPO_FLAGS:-}"

# =========================================================================== #
# distillation into gpt2-small -- OPTIONAL, not a numbered stage:
#     python -m distill.run
# =========================================================================== #
# =========================================================================== #
# stage 11 -- LANGUAGE ADAPTATION: a new vocabulary, then a new language
# =========================================================================== #
# The pretrained (English) model is adapted to another language in two steps: the BPE is
# EXTENDED with merges learned on the new corpus -- every existing id keeps its meaning, the
# new merges are appended -- and the model then continues pretraining on that corpus with its
# embedding grown to match.
#
# WHY EXTEND RATHER THAN RETRAIN THE VOCABULARY. A byte-level BPE never fails on an unseen
# language; it falls back to raw bytes. So the English vocabulary WOULD train on Irish, and
# would spend four or five tokens on words an adapted vocabulary spells in one -- a constant
# tax on the context window and the step budget alike. A fresh vocabulary would fix the
# tokenisation and throw the pretrained model away with it, since new ids mean the embedding
# rows learned in stage 5 sit against different tokens.
LANG_SFT_DATASET="${LANG_SFT_DATASET:-zetagpt-pretrain-gaelic-uccix_irish_textual_corpus}"
LANG_SFT_DATASET_SHORTNAME="${LANG_SFT_DATASET_SHORTNAME:-gaelic}"   # goes in every filename:
                                    # checkpoint_zetagpt-s_ssm_lang-sft-gaelic.pt
LANG_SFT_MERGES="${LANG_SFT_MERGES:-8000}"   # merges ADDED to the base vocabulary. 8,000 on
                                    # top of 50,000 grows the vocabulary by 16% and the
                                    # parameter count by ~4M at d_model 512.
LANG_SFT_STEPS="${LANG_SFT_STEPS:-20000}"
LANG_SFT_LR="${LANG_SFT_LR:-1e-5}"
LANG_SFT_FLAGS="${LANG_SFT_FLAGS:-}"

DISTILL_STEPS="${DISTILL_STEPS:-3251}"
DISTILL_LR="${DISTILL_LR:-1e-6}"
DISTILL_TEACHER="${DISTILL_TEACHER:-rlhf}"   # which aligned checkpoint teaches: rlhf | dpo
DISTILL_STUDENT="${DISTILL_STUDENT:-gpt2}"   # any HuggingFace causal LM
DISTILL_FLAGS="${DISTILL_FLAGS:-}"

DISTILL_STUDENT_MAX_LEN="${DISTILL_STUDENT_MAX_LEN:-256}"   # the student's own truncation
DISTILL_MAX_NEW_TOKENS="${DISTILL_MAX_NEW_TOKENS:-60}"      # teacher response length
DISTILL_GEN_TEMP="${DISTILL_GEN_TEMP:-0.8}"                 # teacher sampling temperature
DISTILL_KL_COEF="${DISTILL_KL_COEF:-0.05}"
DISTILL_PROMPTS_PER_FILE="${DISTILL_PROMPTS_PER_FILE:-100}"

# =========================================================================== #
# scaling laws -- NOT a pipeline stage; run by ./run_scaling_laws.sh
# =========================================================================== #
# Loss against model size and data size, on WikiText-103. The ladder is ZetaGPT-S shrunk, so
# every rung differs in depth and width alone; see scaling_laws/grid.py.
SCALING_MODELS="${SCALING_MODELS:-zs/8,zs/6,zs/4,zs/2,zs}"
SCALING_BUDGETS="${SCALING_BUDGETS:-2000000,8000000,32000000,100000000}"
SCALING_CONTEXT="${SCALING_CONTEXT:-512}"    # held FIXED across the grid
SCALING_BATCH="${SCALING_BATCH:-16}"
SCALING_LR="${SCALING_LR:-6e-4}"             # peak rate at the reference width
SCALING_LR_RULE="${SCALING_LR_RULE:-sqrt_width}"   # sqrt_width | fixed
SCALING_FLAGS="${SCALING_FLAGS:-}"

SCALING_LR_REF_WIDTH="${SCALING_LR_REF_WIDTH:-512}"   # width SCALING_LR is quoted at
SCALING_EVAL_EVERY="${SCALING_EVAL_EVERY:-0}"         # 0 = ten evaluations per point
SCALING_VAL_WINDOWS="${SCALING_VAL_WINDOWS:-64}"      # validation windows per evaluation

# =========================================================================== #
# your settings: config_user.yaml overrides everything above
# =========================================================================== #
# Written by `python -m tools.config_wizard`, git-ignored, absent by default. Sourced (not run) so
# that the exports land in this shell. Environment variables set before this script started
# are left alone; see ZETAGPT_ENV_SET at the top.
ZETAGPT_YAML="${ZETAGPT_YAML:-$(dirname "${BASH_SOURCE[0]}")/config_user.yaml}"
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/export_yaml.sh" "$ZETAGPT_YAML"

# =========================================================================== #
# command lines, assembled from whatever survived the layers above
# =========================================================================== #
# runtime, data, model, optimisation, reporting, probes -- EVERY stage receives all of these.
# The model settings are here rather than with pretraining because the run tag that names
# every checkpoint and figure is derived from them: a stage that did not receive them would
# look for a filename no stage ever wrote.
[ -n "$GPU" ]            && COMMON_FLAGS="$COMMON_FLAGS --gpu $GPU"
[ -n "$SEED" ]           && COMMON_FLAGS="$COMMON_FLAGS --seed $SEED"
[ -n "$DATASET" ]        && COMMON_FLAGS="$COMMON_FLAGS --dataset $DATASET"
[ -n "$PRETRAIN_DIR" ]   && COMMON_FLAGS="$COMMON_FLAGS --pretrain_dir $PRETRAIN_DIR"
[ -n "$INSTRUCT_DIR" ]   && COMMON_FLAGS="$COMMON_FLAGS --instruct_dir $INSTRUCT_DIR"
[ -n "$SFT_DIR" ]        && COMMON_FLAGS="$COMMON_FLAGS --sft_dir $SFT_DIR"
[ -n "$DATA_LIMIT" ]     && COMMON_FLAGS="$COMMON_FLAGS --limit $DATA_LIMIT"
[ -n "$VAL_FRAC" ]       && COMMON_FLAGS="$COMMON_FLAGS --val_frac $VAL_FRAC"
[ -n "$MODEL_SCHEME" ]   && COMMON_FLAGS="$COMMON_FLAGS --model_scheme $MODEL_SCHEME"
[ -n "$CONTEXT_WINDOW" ] && COMMON_FLAGS="$COMMON_FLAGS --context_window $CONTEXT_WINDOW"
[ -n "$PE" ]             && COMMON_FLAGS="$COMMON_FLAGS --pe $PE"
[ -n "$BATCH" ]          && COMMON_FLAGS="$COMMON_FLAGS --batch $BATCH"
[ -n "$MICRO_BATCH" ]    && COMMON_FLAGS="$COMMON_FLAGS --micro_batch $MICRO_BATCH"
[ "$CHUNKED_LOSS" = "0" ] && COMMON_FLAGS="$COMMON_FLAGS --no_chunked_loss"
[ "$TENSOR_PARALLEL" = "0" ] && COMMON_FLAGS="$COMMON_FLAGS --no-tensor_parallel"
# Only the OFF switch is passed: the default is on in default_config, and a flag that
# restates a default is a second place for it to drift from.
[ "$MIXED_PRECISION" = "0" ] && COMMON_FLAGS="$COMMON_FLAGS --no-mixed_precision"
[ "$KV_CACHE" = "0" ]    && COMMON_FLAGS="$COMMON_FLAGS --no-kv_cache"
[ -n "$KV_CACHE_SIZE" ]  && COMMON_FLAGS="$COMMON_FLAGS --kv_cache_size $KV_CACHE_SIZE"
[ -n "$LOSS_CHUNK" ]     && COMMON_FLAGS="$COMMON_FLAGS --loss_chunk $LOSS_CHUNK"
[ -n "$LR_SCHEDULE" ]    && COMMON_FLAGS="$COMMON_FLAGS --lr_schedule $LR_SCHEDULE"
[ -n "$LR_MIN_FACTOR" ]  && COMMON_FLAGS="$COMMON_FLAGS --lr_min_factor $LR_MIN_FACTOR"
[ -n "$BETA" ]           && COMMON_FLAGS="$COMMON_FLAGS --beta $BETA"
[ -n "$PLOT_EVERY" ]     && COMMON_FLAGS="$COMMON_FLAGS --plot_every_steps $PLOT_EVERY"
[ -n "$PRINT_SAMPLES_EVERY" ] && COMMON_FLAGS="$COMMON_FLAGS --print_samples_every_steps $PRINT_SAMPLES_EVERY"
[ -n "$CKPT_EVERY" ]     && COMMON_FLAGS="$COMMON_FLAGS --checkpoint_every_steps $CKPT_EVERY"
[ -n "$SSM_STATS_EVERY" ] && COMMON_FLAGS="$COMMON_FLAGS --ssm_stats_every $SSM_STATS_EVERY"
[ "$NO_RESUME" = "1" ]   && COMMON_FLAGS="$COMMON_FLAGS --no-resume"
[ -n "$EVAL_EVERY" ]     && COMMON_FLAGS="$COMMON_FLAGS --eval_every $EVAL_EVERY"
[ -n "$EVAL_PAIRS" ]     && COMMON_FLAGS="$COMMON_FLAGS --eval_pairs $EVAL_PAIRS"
[ -n "$EB_EVERY" ]       && COMMON_FLAGS="$COMMON_FLAGS --eb_every $EB_EVERY"
[ -n "$EB_PAIRS" ]       && COMMON_FLAGS="$COMMON_FLAGS --eb_pairs $EB_PAIRS"
[ -n "$ROLLOUT_TEMP" ]   && COMMON_FLAGS="$COMMON_FLAGS --rollout_temp $ROLLOUT_TEMP"
[ -n "$N_HIST" ]         && COMMON_FLAGS="$COMMON_FLAGS --n_hist $N_HIST"
[ -n "$N_ROLL" ]         && COMMON_FLAGS="$COMMON_FLAGS --n_roll $N_ROLL"
[ -n "$ROLL_TOKENS" ]    && COMMON_FLAGS="$COMMON_FLAGS --roll_tokens $ROLL_TOKENS"
[ -n "$P_GRID" ]         && COMMON_FLAGS="$COMMON_FLAGS --p_grid $P_GRID"

# the settings that live inside default_config.py's dictionaries, forwarded as --set
[ -n "$CORPUS_MAX_WORDS" ]   && COMMON_FLAGS="$COMMON_FLAGS --set PRETRAIN.max_words=$CORPUS_MAX_WORDS"
[ -n "$CORPUS_TEXT_COLUMN" ] && COMMON_FLAGS="$COMMON_FLAGS --set PRETRAIN.text_column=$CORPUS_TEXT_COLUMN"
[ -n "$BPE_MIN_FREQ" ]       && COMMON_FLAGS="$COMMON_FLAGS --set BPE.min_freq=$BPE_MIN_FREQ"
[ -n "$RLHF_KL_COEF" ]       && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.kl_coef=$RLHF_KL_COEF"
[ -n "$RLHF_MAX_NEW_TOKENS" ] && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.max_new_tokens=$RLHF_MAX_NEW_TOKENS"
[ -n "$COT_KL_COEF" ]        && COMMON_FLAGS="$COMMON_FLAGS --set COT.kl_coef=$COT_KL_COEF"
[ -n "$COT_MAX_NEW_TOKENS" ] && COMMON_FLAGS="$COMMON_FLAGS --set COT.max_new_tokens=$COT_MAX_NEW_TOKENS"
[ -n "$DISTILL_TEACHER" ]    && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.teacher_stage=$DISTILL_TEACHER"
[ -n "$DISTILL_STUDENT" ]    && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.student=$DISTILL_STUDENT"
[ -n "$MODEL_FFN_FACTOR" ]        && COMMON_FLAGS="$COMMON_FLAGS --set MODEL.ffn_factor=$MODEL_FFN_FACTOR"
[ -n "$MODEL_DROPOUT" ]           && COMMON_FLAGS="$COMMON_FLAGS --set MODEL.dropout=$MODEL_DROPOUT"
[ -n "$MODEL_GATED_ATTN" ]        && COMMON_FLAGS="$COMMON_FLAGS --set MODEL.gated_attn=$MODEL_GATED_ATTN"
[ -n "$MODEL_D_CONV" ]            && COMMON_FLAGS="$COMMON_FLAGS --set MODEL.d_conv=$MODEL_D_CONV"
[ -n "$MODEL_SSM_CHUNK" ]         && COMMON_FLAGS="$COMMON_FLAGS --set MODEL.ssm_chunk=$MODEL_SSM_CHUNK"
[ -n "$CORPUS_EXCLUDE" ]          && COMMON_FLAGS="$COMMON_FLAGS --set PRETRAIN.exclude_dirs=$CORPUS_EXCLUDE"
[ -n "$TOKENS_SHARD_MB" ]         && COMMON_FLAGS="$COMMON_FLAGS --set TOKENS.shard_mb=$TOKENS_SHARD_MB"
[ -n "$SFT_SUBSETS" ]             && COMMON_FLAGS="$COMMON_FLAGS --set SFT.subsets=$SFT_SUBSETS"
[ -n "$SFT_LIMIT" ]               && COMMON_FLAGS="$COMMON_FLAGS --set SFT.limit=$SFT_LIMIT"
[ -n "$REWARD_SUBSETS" ]          && COMMON_FLAGS="$COMMON_FLAGS --set REWARD.hh_subsets=$REWARD_SUBSETS"
[ -n "$REWARD_VAL_SUBSETS" ]      && COMMON_FLAGS="$COMMON_FLAGS --set REWARD.hh_val_subsets=$REWARD_VAL_SUBSETS"
[ -n "$REWARD_LIMIT" ]            && COMMON_FLAGS="$COMMON_FLAGS --set REWARD.hh_limit=$REWARD_LIMIT"
[ -n "$RLHF_GEN_TEMP" ]           && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.gen_temperature=$RLHF_GEN_TEMP"
[ -n "$RLHF_PPO_EPOCHS" ]         && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.ppo_epochs=$RLHF_PPO_EPOCHS"
[ -n "$RLHF_CLIP_EPS" ]           && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.clip_eps=$RLHF_CLIP_EPS"
[ -n "$RLHF_GAMMA" ]              && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.gamma=$RLHF_GAMMA"
[ -n "$RLHF_LAM" ]                && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.lam=$RLHF_LAM"
[ -n "$RLHF_VF_COEF" ]            && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.vf_coef=$RLHF_VF_COEF"
[ -n "$RLHF_ENT_COEF" ]           && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.ent_coef=$RLHF_ENT_COEF"
[ -n "$RLHF_WHITEN_ADV" ]         && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.whiten_adv=$RLHF_WHITEN_ADV"
[ -n "$RLHF_PROMPT_LIMIT" ]       && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.prompt_limit=$RLHF_PROMPT_LIMIT"
[ -n "$RLHF_PROMPTS_PER_FILE" ]   && COMMON_FLAGS="$COMMON_FLAGS --set RLHF.prompts_per_file=$RLHF_PROMPTS_PER_FILE"
[ -n "$COT_GEN_TEMP" ]            && COMMON_FLAGS="$COMMON_FLAGS --set COT.gen_temperature=$COT_GEN_TEMP"
[ -n "$COT_CLIP_EPS" ]            && COMMON_FLAGS="$COMMON_FLAGS --set COT.clip_eps=$COT_CLIP_EPS"
[ -n "$COT_GRPO_EPOCHS" ]         && COMMON_FLAGS="$COMMON_FLAGS --set COT.grpo_epochs=$COT_GRPO_EPOCHS"
[ -n "$COT_ENT_COEF" ]            && COMMON_FLAGS="$COMMON_FLAGS --set COT.ent_coef=$COT_ENT_COEF"
[ -n "$COT_CORRECT_REWARD" ]      && COMMON_FLAGS="$COMMON_FLAGS --set COT.correct_reward=$COT_CORRECT_REWARD"
[ -n "$COT_THINK_REWARD" ]  && COMMON_FLAGS="$COMMON_FLAGS --set COT.think_reward=$COT_THINK_REWARD"
[ -n "$COT_THINK_FORMAT_REWARD" ]  && COMMON_FLAGS="$COMMON_FLAGS --set COT.think_format_reward=$COT_THINK_FORMAT_REWARD"
[ -n "$COT_ANSWER_FORMAT_REWARD" ] && COMMON_FLAGS="$COMMON_FLAGS --set COT.answer_format_reward=$COT_ANSWER_FORMAT_REWARD"
[ -n "$COT_LENGTH_PENALTY" ]      && COMMON_FLAGS="$COMMON_FLAGS --set COT.length_penalty=$COT_LENGTH_PENALTY"
[ -n "$COT_LIMIT" ]               && COMMON_FLAGS="$COMMON_FLAGS --set COT.limit=$COT_LIMIT"
[ -n "$COT_EVAL_PROBLEMS" ]       && COMMON_FLAGS="$COMMON_FLAGS --set COT.eval_problems=$COT_EVAL_PROBLEMS"
[ -n "$COT_TRAIN_PREFIX" ]        && COMMON_FLAGS="$COMMON_FLAGS --set COT.train_prefix=$COT_TRAIN_PREFIX"
[ -n "$COT_TEST_PREFIX" ]         && COMMON_FLAGS="$COMMON_FLAGS --set COT.test_prefix=$COT_TEST_PREFIX"
[ -n "$COT_RECORDS_PER_FILE" ]    && COMMON_FLAGS="$COMMON_FLAGS --set COT.records_per_file=$COT_RECORDS_PER_FILE"
[ -n "$COT_TRACE_FIELD" ]         && COMMON_FLAGS="$COMMON_FLAGS --set COT.trace_field=$COT_TRACE_FIELD"
[ -n "$COT_TRACE_FIELD_GSM8K" ]   && COMMON_FLAGS="$COMMON_FLAGS --set COT.trace_field_gsm8k=$COT_TRACE_FIELD_GSM8K"
[ -n "$COT_SFT_VERIFY" ]          && COMMON_FLAGS="$COMMON_FLAGS --set COT.sft_verify=$COT_SFT_VERIFY"
[ -n "$DISTILL_STUDENT_MAX_LEN" ] && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.student_max_len=$DISTILL_STUDENT_MAX_LEN"
[ -n "$DISTILL_MAX_NEW_TOKENS" ]  && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.max_new_tokens=$DISTILL_MAX_NEW_TOKENS"
[ -n "$DISTILL_GEN_TEMP" ]        && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.gen_temperature=$DISTILL_GEN_TEMP"
[ -n "$DISTILL_KL_COEF" ]         && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.kl_coef=$DISTILL_KL_COEF"
[ -n "$DISTILL_PROMPTS_PER_FILE" ] && COMMON_FLAGS="$COMMON_FLAGS --set DISTILL.prompts_per_file=$DISTILL_PROMPTS_PER_FILE"
[ -n "$SCALING_LR_REF_WIDTH" ]    && COMMON_FLAGS="$COMMON_FLAGS --set SCALING.lr_ref_width=$SCALING_LR_REF_WIDTH"
[ -n "$SCALING_EVAL_EVERY" ]      && COMMON_FLAGS="$COMMON_FLAGS --set SCALING.eval_every=$SCALING_EVAL_EVERY"
[ -n "$SCALING_VAL_WINDOWS" ]     && COMMON_FLAGS="$COMMON_FLAGS --set SCALING.val_windows=$SCALING_VAL_WINDOWS"
[ -n "$EXTRA_SET" ]          && COMMON_FLAGS="$COMMON_FLAGS $EXTRA_SET"

# stage 4 does not take the common trainer flags (it builds no model), so the corpus
# settings it shares with stage 5 are forwarded to it separately
[ -n "$MODEL_SCHEME" ]       && TOKENIZE_FLAGS="$TOKENIZE_FLAGS --model_scheme $MODEL_SCHEME"
[ -n "$PRETRAIN_DIR" ]       && TOKENIZE_FLAGS="$TOKENIZE_FLAGS --pretrain_dir $PRETRAIN_DIR"
[ -n "$CORPUS_MAX_WORDS" ]   && TOKENIZE_FLAGS="$TOKENIZE_FLAGS --max_words $CORPUS_MAX_WORDS"
[ -n "$CORPUS_TEXT_COLUMN" ] && TOKENIZE_FLAGS="$TOKENIZE_FLAGS --text_column $CORPUS_TEXT_COLUMN"
[ -n "$EXTRA_SET" ]          && TOKENIZE_FLAGS="$TOKENIZE_FLAGS $EXTRA_SET"

# per-stage
[ -n "$BPE_MERGES" ]     && BPE_FLAGS="$BPE_FLAGS --bpe_merges $BPE_MERGES"
[ -n "$PRETRAIN_STEPS" ] && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --pretrain_steps $PRETRAIN_STEPS"
[ -n "$PRETRAIN_LR" ]    && PRETRAIN_FLAGS="$PRETRAIN_FLAGS --pretrain_lr $PRETRAIN_LR"
[ -n "$SFT_STEPS" ]      && SFT_FLAGS="$SFT_FLAGS --sft_steps $SFT_STEPS"
[ -n "$SFT_LR" ]         && SFT_FLAGS="$SFT_FLAGS --sft_lr $SFT_LR"
# AFTER the common flags on the command line, so these win: stage 6's batch is its own, and
# BATCH= / MICRO_BATCH= in the environment still beat both, since _was_given reads them.
[ -n "$SFT_BATCH" ]      && SFT_FLAGS="$SFT_FLAGS --batch $SFT_BATCH"
[ -n "$SFT_MICRO_BATCH" ] && SFT_FLAGS="$SFT_FLAGS --micro_batch $SFT_MICRO_BATCH"
[ -n "$REWARD_STEPS" ]   && REWARD_FLAGS="$REWARD_FLAGS --reward_steps $REWARD_STEPS"
[ -n "$REWARD_LR" ]      && REWARD_FLAGS="$REWARD_FLAGS --reward_lr $REWARD_LR"
[ -n "$RLHF_STEPS" ]     && RLHF_FLAGS="$RLHF_FLAGS --rlhf_steps $RLHF_STEPS"
[ -n "$RLHF_LR" ]        && RLHF_FLAGS="$RLHF_FLAGS --rlhf_lr $RLHF_LR"
[ -n "$COT_TASK" ]       && COT_FLAGS="$COT_FLAGS --set COT.task=$COT_TASK"
[ -n "$COT_STEPS" ]      && COT_FLAGS="$COT_FLAGS --cot_steps $COT_STEPS"
[ -n "$COT_LR" ]         && COT_FLAGS="$COT_FLAGS --cot_lr $COT_LR"
[ -n "$COT_SFT_STEPS" ]  && COT_FLAGS="$COT_FLAGS --cot_sft_steps $COT_SFT_STEPS"
[ -n "$COT_SFT_LR" ]     && COT_FLAGS="$COT_FLAGS --cot_sft_lr $COT_SFT_LR"
[ "$COT_SFT" = "0" ]     && COT_FLAGS="$COT_FLAGS --no-cot_sft"
[ -n "$COT_INIT" ]       && COT_FLAGS="$COT_FLAGS --cot_init $COT_INIT"
[ -n "$COT_BATCH" ]      && COT_FLAGS="$COT_FLAGS --cot_batch $COT_BATCH"
[ -n "$COT_GROUP" ]      && COT_FLAGS="$COT_FLAGS --cot_group $COT_GROUP"
[ -n "$COT_SAMPLES_EVERY" ] && COT_FLAGS="$COT_FLAGS --cot_samples_every $COT_SAMPLES_EVERY"
[ -n "$LANG_SFT_DATASET" ]  && LANG_SFT_FLAGS="$LANG_SFT_FLAGS --lang_sft_dataset $LANG_SFT_DATASET"
[ -n "$LANG_SFT_DATASET_SHORTNAME" ] && LANG_SFT_FLAGS="$LANG_SFT_FLAGS --lang_sft_short $LANG_SFT_DATASET_SHORTNAME"
[ -n "$LANG_SFT_MERGES" ]  && LANG_SFT_FLAGS="$LANG_SFT_FLAGS --lang_sft_merges $LANG_SFT_MERGES"
[ -n "$LANG_SFT_STEPS" ]   && LANG_SFT_FLAGS="$LANG_SFT_FLAGS --lang_sft_steps $LANG_SFT_STEPS"
[ -n "$LANG_SFT_LR" ]      && LANG_SFT_FLAGS="$LANG_SFT_FLAGS --lang_sft_lr $LANG_SFT_LR"
[ -n "$DPO_STEPS" ]      && DPO_FLAGS="$DPO_FLAGS --dpo_steps $DPO_STEPS"
[ -n "$DPO_LR" ]         && DPO_FLAGS="$DPO_FLAGS --dpo_lr $DPO_LR"
[ -n "$DISTILL_STEPS" ]  && DISTILL_FLAGS="$DISTILL_FLAGS --distill_steps $DISTILL_STEPS"
[ -n "$DISTILL_LR" ]     && DISTILL_FLAGS="$DISTILL_FLAGS --distill_lr $DISTILL_LR"
[ -n "$SCALING_MODELS" ]  && SCALING_FLAGS="$SCALING_FLAGS --models $SCALING_MODELS"
[ -n "$SCALING_BUDGETS" ] && SCALING_FLAGS="$SCALING_FLAGS --budgets $SCALING_BUDGETS"
[ -n "$SCALING_CONTEXT" ] && SCALING_FLAGS="$SCALING_FLAGS --context $SCALING_CONTEXT"
[ -n "$SCALING_BATCH" ]   && SCALING_FLAGS="$SCALING_FLAGS --batch $SCALING_BATCH"
[ -n "$SCALING_LR" ]      && SCALING_FLAGS="$SCALING_FLAGS --lr $SCALING_LR"
[ -n "$SCALING_LR_RULE" ] && SCALING_FLAGS="$SCALING_FLAGS --lr_rule $SCALING_LR_RULE"

# =========================================================================== #
# what this run is actually configured with
# =========================================================================== #
# Every knob, one per line, with the value in force and where it came from -- so the head of a
# log says what produced it, without anyone having to reconstruct four layers of precedence
# from memory. Set ZETAGPT_QUIET=1 to suppress this when chaining stages.
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
                *) _src="config.sh" ;;
              esac ;;
         esac ;;
    esac
    [ -n "$_val" ] || _val="(none)"
    printf '  %-20s %-46s %s\n' "$_v" "$_val" "$_src"
  done
  echo "------------------------------------------------------------------------------"
fi
unset _v _val _src

# The loop above can end on a failed test, and the exit status of the last command in a sourced
# file is the exit status of `source`. Every stage script runs under `set -e`, so without this
# line a stage would exit SILENTLY, before printing anything.
:
