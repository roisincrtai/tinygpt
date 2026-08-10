"""
default_config.py -- ALL DEFAULTS for the zetagpt pipeline in one place.

These are DEFAULTS, not the settings of a run: config.sh is the shell-level layer that steers
a run without editing Python, and a command-line flag beats both. Precedence is

    command-line flag  >  config.sh  >  default_config.py

Every stage imports this module as `config`, so the code reads `config.MODEL[...]` while the
filename says what the values actually are.

Every stage trainer (stage2..stage9, and distill/) reads its defaults from here via
common.parse_args();
any value can still be overridden on the command line (a flag present in argv wins), so
`python -m sft.run --sft_steps 5000` works without touching this file.

Layout (all paths relative to the project root, i.e. the directory of this file). The
framework is DATA-AGNOSTIC: a corpus is whatever files with a CORPUS_EXTENSIONS extension
live under the configured directory, and the instruction data is whatever preference json the
instruction tree holds. NOTHING HERE DOWNLOADS; see DATASETS below and tools/download_data.py.

    data/download/zetagpt-tiny_pretrain-corpus_wikitext103/
                                pretraining corpus for Tiny: <split>-<NNNNN>-of-<NNNNN>.parquet,
                                one row per article in a `text` column. The held-out shards are
                                excluded by PRETRAIN["exclude_dirs"], which matches file names
                                as well as directories
    data/download/zetagpt-small_pretrain-corpus_fineweb-edu_10GB/
                                pretraining corpus for S, ~2B tokens. Every file under it
                                (recursive) whose extension is in CORPUS_EXTENSIONS, minus
                                PRETRAIN["exclude_dirs"]
    data/download/zetagpt-rlhf-instruction_following/
                                rlhf_hh/        preference pairs, by subset and split. Serves
                                                stages 5 (demonstrations), 5 (preferences),
                                                6 and 9 (prompts) and 8 (preferences)
                                alpaca_gpt4/    an alternative prompt bank, unused by default
    data/download/zetagpt-grpo-cot_gsm8k/
                                reasoning problems: {train,test}_<batch>.json
    cache/tokens/<tokenizer>/   pre-tokenised corpora as .tokens files, mirroring the
                                source path. The directory is the TOKENIZER
                                (bpe_<vocab>_<8 chars of `git hash-object bpe.json`>), so
                                two vocabularies coexist instead of evicting each
                                other, and no stage appears in the path: tokens depend on
                                the file and the tokenizer, so every stage shares them
    checkpoints/<stage>/        checkpoint_ + history_<model>_<pe>_<stage-label> per stage
                                (bpe/ holds the tokenizer json)
    outputs/plots/<stage>/      the stage's dynamics figures, PDF only

Pipeline: bpe -> pretrain -> sft -> reward -> rlhf(PPO) ; sft -> dpo ; rlhf -> distill
          pretrain -> cot (GRPO against a verified answer)
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
# ONE DATA ROOT. The repository ships no corpus at all: every stage reads from
# data/download/<dataset>/, which stage 1 fills and which is git-ignored in full. A clone is
# therefore small, nothing large is ever in `git status`, and there is exactly one answer to
# "where did this data come from" -- DATASETS below, fetched by tools/download_data.py and by
# nothing else.
DATA_DIR = os.path.join(ROOT, "data")                      # downloads only; git-ignored
DOWNLOAD_DIR = os.path.join(DATA_DIR, "download")          # data/download/<dataset>/  <- the
                                                           # data root every stage reads from

# THE DATASETS, AND THE ONE THING THAT FETCHES THEM.
#
# Nothing in the pipeline downloads. `tools/download_data.py` is the only code in this
# repository that reaches the network for data; every trainer reads a LOCAL DIRECTORY under
# data/download/, and fails with a message if it is not there.
# The separation is deliberate: a training run that silently fetches several gigabytes on
# first use is a run whose corpus is not pinned, cannot be audited, and cannot be reproduced
# offline. Fetch once, deliberately, then train.
#
#     ./stage1_download_data.sh                     every dataset below
#     ./stage1_download_data.sh --list              what would be fetched, and where
#
# `files_per_subdir` is optional. When set, tools/download_data.py moves the fetched files
# into part_0000/, part_0001/, ... of at most that many each. A corpus of tens of thousands of
# files in ONE directory is slow to list on most filesystems and unpleasant to look at; the
# corpus scanner walks recursively, so the split is invisible to every stage.
DATASETS = {
    "zetagpt-tiny_pretrain-corpus_wikitext103": {
        "repo": "roisincrtai/zetagpt-tiny_pretrain-corpus_wikitext103",
        "what": "pretraining corpus for ZetaGPT-Tiny: WikiText-103",
        "files_per_subdir": 500,
    },
    "zetagpt-small_pretrain-corpus_fineweb-edu_10GB": {
        "repo": "roisincrtai/zetagpt-small_pretrain-corpus_fineweb-edu_10GB",
        "what": "pretraining corpus for ZetaGPT-S: a ~10 GB (~2B token) subset of FineWeb-Edu",
    },
    "zetagpt-rlhf-instruction_following": {
        "repo": "roisincrtai/zetagpt-rlhf-instruction_following",
        "what": "instruction tuning: rlhf_hh preference pairs and alpaca_gpt4 rollout "
                "prompts, shared by stages 5, 6 and 9",
    },
    "zetagpt-grpo-cot_gsm8k": {
        "repo": "roisincrtai/zetagpt-grpo-cot_gsm8k",
        "what": "chain-of-thought reasoning problems for the stage-8 GRPO run (GSM8K)",
    },
}


def dataset_dir(name):
    """Where `tools/download_data.py` puts a dataset, and where the trainers look for it."""
    return os.path.join(DOWNLOAD_DIR, name)


# PRETRAINING CORPUS PER SCHEME. Tiny trains on WikiText-103 and S on the FineWeb-Edu subset:
# a 61.5M model does not need 2B tokens, and a corpus small enough to tokenise quickly is what
# makes Tiny useful as the scheme you try a change on first. M and L are left EMPTY on purpose,
# since neither corpus is the right diet for a 199M or 480M model and defaulting them to one
# would be worse than refusing. Point them at your own with PRETRAIN_DIR, or --pretrain_dir.
#
# NOTE that Tiny and S therefore differ in corpus as well as depth, so the pair is no longer a
# controlled depth comparison. Point both at the same corpus if that is what you want to
# measure.
PRETRAIN_CORPUS = {
    "zetagpt-tiny": dataset_dir("zetagpt-tiny_pretrain-corpus_wikitext103"),
    "zetagpt-s": dataset_dir("zetagpt-small_pretrain-corpus_fineweb-edu_10GB"),
    "zetagpt-m": "",
    "zetagpt-l": "",
}

# The instruction dataset's own layout, which is FLAT: alpaca_gpt4/ and rlhf_hh/ sit directly
# at its root, with no intermediate instruction_following/ directory.
#
#     zetagpt-rlhf-instruction_following/
#         alpaca_gpt4/alpaca_gpt4_<batch>.json           RLHF rollout prompts
#         rlhf_hh/{helpful,harmless}_{train,test}/*.json preference pairs
INSTRUCT_DIR = dataset_dir("zetagpt-rlhf-instruction_following")
PRETRAIN_DIR = PRETRAIN_CORPUS["zetagpt-s"]                # scanned recursively
INSTRUCTION_DIR = INSTRUCT_DIR                             # no nesting; kept as a name
ALPACA_DIR = os.path.join(INSTRUCT_DIR, "alpaca_gpt4")     # alpaca_gpt4_<batch>.json
HH_DIR = os.path.join(INSTRUCT_DIR, "rlhf_hh")             # {helpful,harmless}_{train,test}/

# rlhf_hh SERVES FOUR STAGES, each taking a different projection of the same records:
#
#   4 SFT     prompt + chosen, trained as a language model on the response  (SFT_DIR)
#   5 reward  (prompt, chosen, rejected) as a binary classification         (HH_DIR)
#   6 RLHF    the prompts alone, to roll out from                           (prompt_dir)
#   8 DPO     (prompt, chosen, rejected) as a preference                    (dataset "hh")
#
# One dataset behind all four is worth more than four tidier ones: the reward model then scores
# the same distribution the policy was fine-tuned on and is rolled out over, which is the
# assumption every one of those objectives is written under and the thing that quietly breaks
# when each stage is fed from somewhere different.
SFT_DIR = HH_DIR


def set_instruct_root(path):
    """Repoint the instruction-tuning tree and everything derived from it.

    The derived directories are module-level constants read from several places, so moving the
    root has to move them together; doing it here rather than at each call site is what stops
    one stage reading the downloaded preference pairs while another reads the shipped ones."""
    global INSTRUCT_DIR, INSTRUCTION_DIR, ALPACA_DIR, HH_DIR, SFT_DIR
    INSTRUCT_DIR = INSTRUCTION_DIR = path
    ALPACA_DIR = os.path.join(path, "alpaca_gpt4")
    HH_DIR = SFT_DIR = os.path.join(path, "rlhf_hh")
    RLHF["prompt_dir"] = HH_DIR
    return INSTRUCT_DIR


COT_DIR = dataset_dir("zetagpt-grpo-cot_gsm8k")             # chain-of-thought / reasoning
COT_GSM8K_DIR = COT_DIR                                    # {train,test}_<batch>.json
DISTILL_DIR = HH_DIR        # distillation generates from the same prompts stage 7 rolls out on
CACHE_DIR = os.path.join(ROOT, "cache")                    # cache/tokens/bpe_<vocab>_<fp>/<mirror>.tokens
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")         # checkpoints/<stage>/checkpoint_*.pt
OUTPUT_DIR = os.path.join(ROOT, "outputs")                  # everything a run produces
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")               # outputs/plots/<stage>/<figure>.pdf
MODEL_DIR = os.path.join(ROOT, "saved_models")
BPE_PATH = os.path.join(CHECKPOINT_DIR, "bpe", "bpe.json") # the trained tokenizer

# Corpus file types. The scanner reads EVERY file with one of these extensions, so a corpus
# can mix prose, markdown and source code; a language model has no reason to care which.
# Source is included deliberately: code is the cleanest available supervision for long-range
# structure. Extensions are matched case-insensitively.
CORPUS_EXTENSIONS = [
    # prose and markup
    "txt", "md", "markdown", "rst", "tex", "org",
    # data / config that reads as text
    "json", "jsonl", "yaml", "yml", "toml", "ini", "csv", "tsv",
    # C family
    "c", "h", "cc", "cpp", "cxx", "hpp", "hh", "hxx", "cu", "cuh",
    # assembly
    "s", "asm",
    # JVM and .NET
    "java", "kt", "kts", "scala", "cs",
    # scripting and dynamic
    "py", "pyx", "rb", "pl", "pm", "php", "lua", "r", "jl", "m",
    # web
    "js", "jsx", "ts", "tsx", "html", "htm", "css", "scss", "vue", "svelte",
    # systems and functional
    "go", "rs", "swift", "zig", "hs", "ml", "mli", "ex", "exs", "erl", "clj", "lisp", "el",
    # shell and build
    "sh", "bash", "zsh", "fish", "ps1", "bat", "mk", "cmake", "gradle",
    # query and hardware description
    "sql", "v", "sv", "vhd", "vhdl",
    # columnar corpora: a parquet file is not text, so it is read through pyarrow and the
    # column named by PRETRAIN["text_column"] is treated as the document text
    "parquet",
]

# --------------------------------------------------------------------------- #
# model architecture (ZetaGPT: positional-encoding-free; see model/zetagpt.py)
# --------------------------------------------------------------------------- #
# THE THREE CONFIGURATION SCHEMES, in one place. Depth, width and the context window each
# scheme is pretrained at; everything else is shared and lives in MODEL below.
#
# THE HEAD DIMENSION IS FIXED AT 64 in every scheme (model.zetagpt.HEAD_DIM), so n_head is not
# a free choice: it is n_embd / 64. A constant head dimension is what makes the three sizes
# comparable -- widening by adding heads of the same size changes how much the model can
# attend to, while widening the heads themselves changes what a single head IS, and the two
# are then confounded in any result that spans sizes.
#
# THE CONTEXT WINDOW IS A SCHEME PROPERTY, NOT AN ARCHITECTURAL ONE. Nothing in ZetaGPT refers
# to an absolute position, so no length is forbidden and a trained model can be evaluated at
# any length; the window here is what the scheme is PRETRAINED at, chosen against the cost of
# attention, which is quadratic in it. The small scheme is the one meant to run on a single
# machine, so it takes 512; the two larger schemes take 1024.
#
#     scheme          layers  heads  d_model  d_h  context  embedding  blocks   parameters
#     ZetaGPT-Tiny         8      8      512   64      512      25.7M   35.8M        61.5M
#     ZetaGPT-S           16      8      512   64      512      25.7M   71.5M        97.2M
#     ZetaGPT-M           16     12      768   64     1024      38.6M  160.7M       199.3M
#     ZetaGPT-L           24     16     1024   64     1024      51.5M  428.4M       479.9M
#
# Tiny and S share a width, a context and therefore the same 25.7M embedding, and differ only
# in depth -- which makes the pair a clean depth comparison on identical data.
SCHEMES = {
    "zetagpt-tiny": dict(n_layer=8,  n_head=8,  n_embd=512,  context_window=512),
    "zetagpt-s": dict(n_layer=16, n_head=8,  n_embd=512,  context_window=512),
    "zetagpt-m": dict(n_layer=16, n_head=12, n_embd=768,  context_window=1024),
    "zetagpt-l": dict(n_layer=24, n_head=16, n_embd=1024, context_window=1024),
}
DEFAULT_SCHEME = "zetagpt-s"


def scheme(name=DEFAULT_SCHEME):
    """A scheme's architecture as ZetaGPT constructor arguments. `context_window` becomes
    `block_size`, which is the model's own name for the same number and is what travels with
    the weights in the checkpoint."""
    s = dict(SCHEMES[name])
    s["block_size"] = s.pop("context_window")
    return s


MODEL = dict(
    **scheme(DEFAULT_SCHEME),   # n_layer, n_head, n_embd, block_size
    ffn_factor=4,
    dropout=0.0,
    gated_attn=True,            # gated attention output: y = W_o(head_out * sigmoid(W_g x))
    # HOW POSITION ENTERS. "ssm" (the proposed architecture) makes every block
    # SSM -> Attention -> FFN and adds no encoding at all: the causal state space module
    # (model/ssm.py) is what makes position available, since its state at step t summarises
    # tokens 1..t. "rope" is the ABLATION CONTROL: the module is removed and rotary positions
    # are applied inside attention instead, giving the conventional transformer. The model
    # always has exactly one source of position, and the choice is recorded in the checkpoint
    # and in every filename the run writes.
    pe="ssm",                   # "ssm" | "rope"
    d_conv=4,                   # depthwise causal conv before the recurrence (local offsets)
    # BLOCKED-SCAN CHUNK. The recurrence is evaluated in chunk + T/chunk sequential steps
    # instead of T, and that depth is minimised at chunk = sqrt(T). "optimal" computes it per
    # forward pass from the actual sequence length, which is the right thing to do because T
    # varies (short rollouts, long packed documents). An integer pins it, for benchmarking.
    ssm_chunk="optimal",
)

# --------------------------------------------------------------------------- #
# shared trainer knobs (defaults of common.parse_args; CLI flags override)
# --------------------------------------------------------------------------- #
TRAIN = dict(
    dataset="hh",               # "hh" = the downloaded rlhf_hh tree, or a PATH to your
                                # own folder of records (the layout is detected)
    gpu="auto",                 # auto | cuda | mps | cpu
    seed=0,
    batch=16,                   # examples (or preference pairs) per step
    micro_batch=0,              # gradient-accumulation micro-batch (0 = off)
    max_len=0,                  # 0 = auto: the model's own context window (block_size)
    beta=0.1,                   # implicit-reward beta, shared by DPO / evaluation
    val_frac=0.05,              # validation fraction of the preference file
    limit=0,                    # cap #preference pairs loaded (0 = all)
    resume=True,                # resume from this stage's checkpoint, if one is there
    # learning-rate schedule, shared by EVERY trainer: cosine decay from the stage's lr
    # down to lr / lr_min_factor over that stage's own step budget.
    #     lr(t) = lr_min + 0.5 (lr - lr_min) (1 + cos(pi t / T)),  lr_min = lr / factor
    # The schedule is a function of the ABSOLUTE step, so a resumed run continues on the
    # same curve rather than restarting it.
    lr_schedule="cosine",       # "cosine" | "constant"
    lr_min_factor=10.0,         # minimum lr is 1/10 of the stage's lr
    checkpoint_every_steps=200,
    plot_every_steps=200,       # live redraw cadence of outputs/plots/<stage>/ (0 = off)
    # State space diagnostics: memory horizon, selectivity, residual write ratio and the rest
    # (model/ssm.py explains each). Collected for ONE step every N, so the cost is a handful
    # of reductions per figure rather than per step. Read by the pretraining stage.
    ssm_stats_every=200,
    # held-out probe during DPO (0 = use plot_every_steps)
    eval_every=0, eval_pairs=32,
    # exposure-bias diagnostic probe during DPO (0 = off)
    eb_every=25, eb_pairs=4, rollout_temp=1.0,
    # end-of-stage evaluation
    n_hist=256, n_roll=32, roll_tokens=48, p_grid="0.0,0.5,1.0",
)

# --------------------------------------------------------------------------- #
# per-stage config
# --------------------------------------------------------------------------- #
# EXTRA SPECIAL TOKENS, on top of the predefined <|endoftext|>, <|pad|> and <|unk|>.
#
# A registered token is an ATOM: the tokenizer emits ONE id for it and never breaks it into
# subwords, so a later model can attach a meaning to it -- a control marker, a role tag, a
# modality boundary.
#
#     EXTRA_SPECIAL_TOKENS = ["<vp>", "<|im_start|>", "<|im_end|>"]
#     tok.encode("a <vp> b")     ->  [.., <vp>, ..]   rather than the subwords of "<vp>"
#
# They are appended AFTER the merges and after the predefined specials, so adding one never
# moves an existing id: a checkpoint trained before the change only needs its embedding row
# count raised, not retraining.
#
# Note the consequence: text containing those exact characters now produces the token. Corpus
# text you do not control should be encoded with tok.encode_ordinary, which treats every
# special as plain text.
EXTRA_SPECIAL_TOKENS = []

BPE = dict(num_merges=50000)         # vocab = 256 bytes + num_merges + specials
                                     # = 256 + 50000 + 3 = 50259
                                     # (<|endoftext|>, <|pad|>, <|unk|>, plus any
                                     #  EXTRA_SPECIAL_TOKENS)

# Step budgets are EPOCH BUDGETS in disguise: steps = ceil(epochs * examples / batch) at the
# default batch of 16. Recompute them if the corpus or the batch size changes.
#   pretrain  3 epochs over 345,811 documents  -> 64,840 steps
#   sft       2 epochs over  18,729 documents  ->  2,342 steps
#   reward    1 epoch  over  40,000 pairs      ->  2,500 steps
#   rlhf      1 epoch  over  52,002 prompts    ->  3,251 steps
#   dpo       1 epoch  over   9,437 pairs      ->    590 steps
#   distill   1 epoch  over  52,002 prompts    ->  3,251 steps
# THE STEP BUDGET IS TWO EPOCHS OVER THE ~2B-TOKEN CORPUS, FOR THE DEFAULT SCHEME.
#
#     tokens/step = batch x context = 16 x 512 = 8,192
#     2 epochs    = 2 x 2e9 = 4e9 tokens
#     steps       = 4e9 / 8,192 = 488,281
#
# That arithmetic is only true if a step really carries batch x context tokens, which is why
# max_words moved from 200 to 344: at ~1.49 tokens per word the old packing gave ~298-token
# documents in a 512-token window, so a step actually carried 4,768 tokens and "two epochs"
# would have needed 838,926 steps. Packing at 344 words fills the window, and the number below
# means what it says. Recompute both if the batch, the context or the corpus changes.
PRETRAIN = dict(
    steps=488281, lr=2e-5,       # 2 epochs over ~2B tokens
    # WHICH SCHEME IS PRETRAINED, and at what context window. Both are exposed on the command
    # line (--model_scheme / --context_window) and in config.sh, so a size can be changed for
    # one run without editing Python. context_window=0 takes the scheme's own value from
    # SCHEMES above (512 for -s, 1024 for -m and -l); any other value overrides it, which is
    # what an ablation on context length needs.
    model_scheme=DEFAULT_SCHEME,
    context_window=0,
    max_words=344,              # ~344 words ~ 512 tokens: one document fills the context
    # The FineWeb-Edu subset ships as parquet, one document per row. This names the column
    # holding the text; the loader lists the file's actual columns if it is wrong, rather than
    # failing with a KeyError several thousand files in.
    text_column="text",
    # Subdirectories EXCLUDED from the corpus scan, matched against any path component so
    # nested directories with these names are skipped too.
    #
    # "test" and "valid" are here because the scan is RECURSIVE and the WikiText corpus ships
    # split into train/, valid/ and test/. Without this a pretraining run over the corpus root
    # would train on the held-out splits -- silently, since more data simply looks like more
    # data. The names are harmless for a corpus that has no such directories, which is why they
    # are excluded by default rather than set per corpus.
    # Held-out splits, excluded from pretraining. Each entry skips a DIRECTORY of that
    # name AND any file whose leading name token matches it, so the same corpus is
    # handled whether its splits are valid/ and test/ subdirectories or flat parquet
    # shards named validation-00000-of-00001.parquet. "validation" is listed beside
    # "valid" because the two layouts spell it differently and only one of them is the
    # Hugging Face convention.
    exclude_dirs=["test", "valid", "validation"],
)

# SFT is DOMAIN-ADAPTIVE fine-tuning: the same length-normalized LM objective as pretraining,
# on the target-domain text under the fine-tuning data (scanned exactly like the pretraining corpus).
SFT = dict(
    steps=2342, lr=1e-6,        # 2 epochs
    # WHICH rlhf_hh SUBSETS the demonstrations come from. The TRAIN splits only: the test
    # splits are what stages 6 and 9 hold themselves out on, and fine-tuning on them would make
    # every later evaluation a measurement of memorisation instead of generalisation.
    subsets=["helpful_train", "harmless_train"],
    limit=0,                    # cap the pairs taken per subset (0 = all)
)

# reward model: ZetaGPT trunk + scalar head; binary classification with sigmoid + BCE,
# chosen -> label 1, rejected -> label 0.
#
# WHICH PREFERENCES. The reward model must speak the domain the policy will be rolled out in,
# or its scores are meaningless there -- which is why stages 5, 6, 7 and 9 all read rlhf_hh:
# the model is fine-tuned on those demonstrations, scored by a reward model fitted to those
# preferences, and rolled out on those prompts. The three agreeing is the assumption PPO's
# objective is written under.
REWARD = dict(
    steps=2500, lr=1e-5,        # 1 epoch over the hh pairs
    # Split directories to read when the data is an hh-layout tree. They are used WHERE THEY
    # EXIST and discovered otherwise, so a tree that names its splits differently still loads;
    # for a plain local folder they are ignored and --val_frac cuts the validation set.
    hh_subsets=["helpful_train", "harmless_train"],       # batch dirs under rlhf_hh/
    hh_val_subsets=["helpful_test", "harmless_test"],
    hh_limit=20000,             # pairs per subset, 0 = all (~86k train pairs in total)
)

# RLHF = PPO against the stage-5 reward model, starting FROM THE SFT MODEL, with a per-token
# KL penalty to the frozen SFT reference (the standard InstructGPT recipe).
RLHF = dict(
    steps=3251, lr=1e-6,        # 1 epoch over the instruction prompts; KL-anchored
    max_new_tokens=48,          # response length of each rollout
    gen_temperature=1.0,        # rollouts are SAMPLED (a greedy rollout is not a draw from pi)
    kl_coef=0.05,               # per-token KL penalty to the SFT reference
    ppo_epochs=4,               # optimisation epochs per batch of rollouts
    clip_eps=0.2,               # PPO ratio clip
    gamma=1.0, lam=0.95,        # discount / GAE lambda
    vf_coef=0.5,                # value-loss weight
    ent_coef=0.0,               # entropy bonus weight
    whiten_adv=True,            # normalise advantages per batch
    # ROLLOUT PROMPTS. PPO needs prompts to generate from, not preference pairs: they come
    # from the instruction batches in <instruct_dir>/alpaca_gpt4/
    # (alpaca_gpt4_<batch>.json, each record carrying `prompt`). Set prompt_dir=None to fall
    # back to the preference file's prompts.
    prompt_dir=HH_DIR,          # the prompts of the same rlhf_hh records stages 5/5/8 use
    prompt_limit=0,             # cap the prompt bank (0 = all ~52k)
    prompts_per_file=1000,      # what alpaca_to_json.py writes
)

DPO = dict(steps=590, lr=1e-6)       # 1 epoch; starts from SFT, ref = frozen SFT

# CHAIN OF THOUGHT BY GROUP RELATIVE POLICY OPTIMISATION (GRPO) -- the "aha moment" stage.
#
# This is the DeepSeek-R1-Zero setting: reinforcement learning on a base model with NO
# supervised reasoning data at all. The policy is asked to think inside <think> tags and to
# answer inside <answer> tags; the reward is not a learned model but an ARITHMETIC CHECK --
# the final answer is extracted and compared with the gold answer of the GSM8K problem -- plus
# a small term for obeying the format. Nothing tells the model HOW to reason, so whatever
# reasoning appears is a property of the optimisation, which is the point of the stage.
#
# GRPO replaces PPO's value head with a GROUP BASELINE: the policy samples `group_size`
# completions for the same prompt and each completion's advantage is its reward standardised
# WITHIN that group. No critic is trained, which is what makes the recipe cheap enough to run
# here. The KL to the frozen reference is added to the loss directly (not shaped into the
# per-token reward as PPO does), using the unbiased non-negative k3 estimator.
#
# WHAT TO WATCH. The reported "aha moment" is a length phenomenon: mean think length grows as
# the policy discovers that more deliberation earns more reward, and self-correcting phrases
# ("wait", "let me re-check") start to appear. Both are tracked every step and drawn in
# outputs/plots/cot/cot_dynamics.pdf, alongside accuracy against the verifier.
COT = dict(
    steps=1400, lr=1e-6,        # ~1 epoch over 7,473 problems at batch 16 / group 8
    init_stage="pretrain",      # WHICH CHECKPOINT THE POLICY STARTS FROM.
                                #   "pretrain" -- R1-Zero: RL on the base model, no SFT
                                #   "sft"      -- a sibling of RLHF and DPO
                                #   "rlhf" / "dpo" -- continue an aligned policy
                                # Overridable per run: COT_INIT=sft ./stage8_cot_aha_moment.sh
    group_size=8,               # completions sampled per prompt; the group IS the baseline
    max_new_tokens=200,         # room to think: a short cap forbids the behaviour we look for
    gen_temperature=1.0,        # rollouts must be SAMPLED for the group to have spread
    kl_coef=0.001,              # KL to the frozen reference, added to the loss (k3 estimator)
    clip_eps=0.2,               # GRPO ratio clip
    grpo_epochs=2,              # optimisation epochs per batch of rollouts
    ent_coef=0.0,               # entropy bonus (0 = off)
    # REWARD. Correctness is verified arithmetically; the format term pays only for producing
    # the think/answer structure at all, and is kept small so that formatting cannot be traded
    # against being right.
    correct_reward=1.0,         # answer matches the gold answer
    format_reward=0.2,          # <think>...</think><answer>...</answer> present and ordered
    length_penalty=0.0,         # per-token penalty on the response (0 = let it grow freely)
    # DATA. Any {train,test}_<batch>.json under data/download/zetagpt-grpo-cot_gsm8k/ (built
    # them from the distributed CSVs); each record carries question / reasoning / answer.
    data_dir=COT_GSM8K_DIR,          # data/download/zetagpt-grpo-cot_gsm8k
    train_prefix="train", test_prefix="test",
    records_per_file=1000,      # what cot_to_json.py writes
    limit=0,                    # cap the problem bank (0 = all)
    eval_problems=200,          # held-out problems scored at the end of the stage
)

# distillation: SEQUENCE-LEVEL KD. The teacher is our ALIGNED ZetaGPT (the RLHF-ed model by
# default; set teacher_stage="dpo" for the DPO-ed model); the student is gpt2-small. Teacher
# and student use different tokenizers, so the transfer is through TEXT: each step the
# teacher generates responses to a batch of prompts read from the prompt bankprompt_<batch>.txt
# (one prompt per line, prompts_per_file per file) and the student is trained by CE on the
# teacher's generations.
DISTILL = dict(
    steps=3251, lr=1e-6,        # 1 epoch over the distillation prompts
    teacher_stage="rlhf",       # which aligned checkpoint teaches: "rlhf" (default) or "dpo"
    student="gpt2",             # HF model id of the student (gpt2-small, 124M)
    student_max_len=256,        # student-side truncation of prompt+response
    prompts_per_file=100,       # prompts per the prompt bankprompt_<batch>.txt
    max_new_tokens=60,          # teacher generation length per prompt
    gen_temperature=0.8,        # teacher sampling temperature
    # KL CONSTRAINT. The student is a PRETRAINED gpt2-small, so fitting it to the teacher's
    # text can wash out what it already knows. A per-token KL to a FROZEN COPY OF ITS OWN
    # INITIAL WEIGHTS holds it near that prior, exactly as the RLHF stage anchors its policy
    # to the frozen SFT model:
    #     L = CE(student, teacher text) + kl_coef * KL( p_student || p_student_init )
    # (full-vocabulary KL at every response position; the reference shares the student's
    # tokenizer, so this is exact -- unlike a teacher/student KL, which is undefined here.)
    kl_coef=0.05,               # 0 disables the constraint (and its extra forward pass)
)

# --------------------------------------------------------------------------- #
# scaling laws (not a pipeline stage: a study, run on its own)
# --------------------------------------------------------------------------- #
# HOW LOSS DEPENDS ON MODEL SIZE AND DATA SIZE, measured on WikiText-103 rather than on the
# pipeline's own corpus. The built-in corpus is small and single-domain, so the larger budgets
# would have to repeat data, and repetition lowers loss for a reason that is not scale -- on
# exactly the axis being fitted. WikiText-103 is ~100M tokens, so every budget below is one
# pass.
#
# The model ladder is ZetaGPT-S SHRUNK (scaling_laws.grid.LADDER), not the three shipped
# schemes: every rung then differs in depth and width alone, with the head dimension fixed at
# 64 and the context held constant, so a fitted exponent is a statement about size.
SCALING = dict(
    models=["zs/8", "zs/6", "zs/4", "zs/2", "zs"],       # 1.1M .. 53.6M non-embedding
    budgets=[2_000_000, 8_000_000, 32_000_000, 100_000_000],
    context=512,                # FIXED across the grid; a third moving axis would be fitted
                                # as if it were one of the two under study
    batch=16,
    # Peak rate AT lr_ref_width; every other width follows lr_rule. A single rate across a 4x
    # width range is a confound rather than a control -- too small for the narrow models or too
    # large for the wide ones -- and the N-axis then carries a bend that is not capacity.
    lr=6e-4, lr_ref_width=512, lr_rule="sqrt_width",     # sqrt_width | fixed
    eval_every=0,               # 0 = ten held-out probes per point
    val_windows=64,             # fixed windows, IDENTICAL for every point
)

# --------------------------------------------------------------------------- #
# end-of-stage sample generation (qualitative check, logged after each stage)
# --------------------------------------------------------------------------- #
N_SAMPLES = 10
SAMPLING = dict(max_new=60, temperature=0.8, top_k=50, top_p=0.95)


# --------------------------------------------------------------------------- #
# what the shell falls back to
# --------------------------------------------------------------------------- #
# config.sh prints the configuration in force at the head of every run. When one of its
# variables is left blank the flag is simply not passed and the trainer uses the value below,
# so the shell has to be told what that value IS -- otherwise the summary can only print a
# dash, which is precisely the case where the reader most needs the number.
#
#     python default_config.py --shell-defaults
#
# emits `ZETAGPT_DEF_<VAR>='<value>'` lines for config.sh to eval. Keys are the shell variable
# names; a variable with no Python default (PY, NO_RESUME, the assembled *_FLAGS) is absent.
SHELL_DEFAULTS = {
    "GPU": TRAIN["gpu"],
    "SEED": TRAIN["seed"],
    "BATCH": TRAIN["batch"],
    "MICRO_BATCH": TRAIN["micro_batch"],
    "DATASET": TRAIN["dataset"],
    "LIMIT": TRAIN["limit"],
    "VAL_FRAC": TRAIN["val_frac"],
    "BETA": TRAIN["beta"],
    # max_len=0 means "the model's context", so report the number that ends up in force
    "MAX_LEN": MODEL["block_size"],
    "MODEL_SCHEME": PRETRAIN["model_scheme"],
    "PRETRAIN_DIR": os.path.relpath(PRETRAIN_CORPUS[PRETRAIN["model_scheme"]], ROOT)
                    if PRETRAIN_CORPUS[PRETRAIN["model_scheme"]] else "(unset for this scheme)",
    "INSTRUCT_DIR": os.path.relpath(INSTRUCT_DIR, ROOT),
    "SFT_DIR": os.path.relpath(SFT_DIR, ROOT),
    # 0 means "the scheme's own", so report the number that ends up in force
    "CONTEXT_WINDOW": PRETRAIN["context_window"] or SCHEMES[PRETRAIN["model_scheme"]]["context_window"],
    "LR_SCHEDULE": TRAIN["lr_schedule"],
    "LR_MIN_FACTOR": TRAIN["lr_min_factor"],
    "PLOT_EVERY": TRAIN["plot_every_steps"],
    "CKPT_EVERY": TRAIN["checkpoint_every_steps"],
    "SSM_STATS_EVERY": TRAIN["ssm_stats_every"],
    "PE": MODEL["pe"],
    # NO_RESUME is the shell's inverted spelling of resume: 0 means "resume from checkpoints"
    "NO_RESUME": 0 if TRAIN["resume"] else 1,
    "EVAL_EVERY": TRAIN["eval_every"],
    "EVAL_PAIRS": TRAIN["eval_pairs"],
    "EB_EVERY": TRAIN["eb_every"],
    "EB_PAIRS": TRAIN["eb_pairs"],
    "ROLLOUT_TEMP": TRAIN["rollout_temp"],
    "N_HIST": TRAIN["n_hist"],
    "N_ROLL": TRAIN["n_roll"],
    "ROLL_TOKENS": TRAIN["roll_tokens"],
    "P_GRID": TRAIN["p_grid"],
    "BPE_MERGES": BPE["num_merges"],
    "PRETRAIN_STEPS": PRETRAIN["steps"],
    "PRETRAIN_LR": PRETRAIN["lr"],
    "SFT_STEPS": SFT["steps"],
    "SFT_LR": SFT["lr"],
    "REWARD_STEPS": REWARD["steps"],
    "REWARD_LR": REWARD["lr"],
    "RLHF_STEPS": RLHF["steps"],
    "RLHF_LR": RLHF["lr"],
    "COT_STEPS": COT["steps"],
    "COT_LR": COT["lr"],
    "COT_INIT": COT["init_stage"],
    "COT_GROUP": COT["group_size"],
    "DPO_STEPS": DPO["steps"],
    "DPO_LR": DPO["lr"],
    "DISTILL_STEPS": DISTILL["steps"],
    "DISTILL_LR": DISTILL["lr"],
    "SCALING_MODELS": ",".join(SCALING["models"]),
    "SCALING_BUDGETS": ",".join(str(d) for d in SCALING["budgets"]),
    "SCALING_CONTEXT": SCALING["context"],
    "SCALING_BATCH": SCALING["batch"],
    "SCALING_LR": SCALING["lr"],
    "SCALING_LR_RULE": SCALING["lr_rule"],
}


if __name__ == "__main__":
    import sys
    if "--shell-defaults" in sys.argv:
        for _k, _v in SHELL_DEFAULTS.items():
            print(f"ZETAGPT_DEF_{_k}='{_v}'")
    else:
        for _k, _v in sorted(SHELL_DEFAULTS.items()):
            print(f"{_k:16s} {_v}")
