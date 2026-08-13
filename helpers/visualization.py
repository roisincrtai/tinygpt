"""
visualization.py -- every figure the pipeline draws, as PDF, with no main title.

Each stage owns one dynamics figure, written to outputs/plots/<stage>/ and named

    dynamics_<model>_<pe>_<stage-label>.pdf

so the size, the positional encoding and the stage are all legible from the filename alone
(see helpers.utils.STAGE_LABELS). By stage:

    bpe                     BPE merge dynamics (pair frequency, corpus size, compression);
                            bpe_dynamics.pdf, as the tokenizer has no model configuration
    pretrain                LM: loss, perplexity, accuracy, gradient norm, learning rate, and
                            the state space diagnostics, ONE LINE PER LAYER
    sft                     the same layout, so the two LM stages are directly comparable
    reward                  BCE, accuracies, class scores, logit margin
    instruct-rlhf           reward score, KL to the reference, PPO/value loss, entropy
    cot-grpo                reward, think length, group advantage, format adherence
    instruct-dpo            loss, implicit rewards, margin, accuracy, held-out probes
    distill                 CE, KL to the student's initialisation, perplexity, accuracy

Entry points:

    method_monitor(mode, records, plotdir, tag)   live redraw during training
    bpe_monitor(history, plotdir, tag)            the tokenizer's own build dynamics
    eval_figure(RES, methods, p_grid, plotdir)    exposure_bias.pdf / bounds.pdf from evals/eval.py

House style throughout: a faint per-step series under a bold moving-average trend, axes
scaled to the trend rather than to outliers, scientific tick labels on the step axis.
"""
import os

from . import utils as _utils

C = {"bpe": "#7b2cbf", "pretrain": "#8338ec", "sft": "#6a994e", "dpo": "#d1495b",
     "reward": "#bc6c25", "rlhf": "#f18701", "cot": "#118ab2", "distill": "#3a86ff"}
NAME = {"bpe": "BPE", "pretrain": "Pretrain", "sft": "SFT", "dpo": "DPO",
        "reward": "Reward model", "rlhf": "RLHF (PPO)", "cot": "CoT (GRPO)",
        "distill": "Distill"}

# Within one stage's figure the stage colour identifies nothing (the filename already does),
# so chosen-versus-rejected contrasts get their own colourblind-safe pair (Okabe-Ito blue /
# vermillion) rather than one hue in two linestyles, which is unreadable at print size.
# Linestyle is kept as a redundant encoding.
C_CHOSEN = "#0072B2"
C_REJECT = "#D55E00"

GREY_NOTE = "#4B5963"      # in-panel annotations: never a data colour


def _sfx(tag):
    """Filename suffix for a run tag: '_<tag>' when one is given, '' when it is empty."""
    return f"_{tag}" if tag else ""


def _figure_path(plotdir, mode, tag):
    """outputs/plots/<stage>/dynamics_<model>_<pe>_<stage-label>.pdf -- the configuration AND
    the stage are in the FILENAME, so a pe="rope" ablation lands beside the run it is compared
    with rather than over it, and a figure dropped into a paper or an issue still says which
    stage produced it. Falls back to <stage>_dynamics.pdf when no run tag has been set."""
    run = tag or _utils.get_run_tag()
    name = (f"dynamics_{run}_{_utils.stage_label(mode)}.pdf" if run
            else f"{mode}_dynamics.pdf")
    return os.path.join(plotdir, name)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9.5, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


# --------------------------------------------------------------------------- #
# training visualization
# --------------------------------------------------------------------------- #
def _live_run(records):
    """Records from the last step restart onward.

    A history file is appended to across runs, so if a run is relaunched from scratch (or
    resumed from a checkpoint older than the last logged step) the "step" column restarts and
    the file holds several segments back to back. Plotting the concatenation draws a line from
    the end of the abandoned segment back to the start of the live one -- a long near-horizontal
    stroke across the early x-range in EVERY panel, which is not data. Keep the live segment."""
    if not records:
        return records
    cut = 0
    for i in range(1, len(records)):
        if records[i].get("step", 0) <= records[i - 1].get("step", 0):
            cut = i
    if cut:
        # the trainers truncate the history at the resume point precisely so this cannot
        # happen; if it does, the figure is about to show a fraction of the run and the
        # reader is entitled to know rather than to be shown a curve that starts late
        print(f"[plot] history is not monotone in step: discarding the first {cut} record(s) "
              f"and drawing from step {records[cut].get('step')}", flush=True)
    return records[cut:]


def _stage_figure(mode, records, plotdir, plt, tag):
    """Dispatch one stage's history to the layout that fits it. One dispatch point, used by
    both the live monitor and the end-of-stage pass, so the two can never disagree about
    which figure a stage owns."""
    if mode == "reward":
        return _reward_figure(mode, records, plotdir, plt, tag)
    if mode == "rlhf":
        return _rlhf_figure(mode, records, plotdir, plt, tag)
    if mode == "cot":
        return _cot_figure(mode, records, plotdir, plt, tag)
    if mode == "distill":
        return _distill_figure(mode, records, plotdir, plt, tag)
    if mode in ("pretrain", "sft", "cot_sft"):
        # LM stages: a plain likelihood objective, no chosen/rejected pair, so the preference
        # 1x5 layout does not apply -- they get the LM figure (loss / ppl / accuracy / grad norm).
        # cot_sft belongs here and NOT with cot: it is stage 9's supervised half, and what it
        # optimises is a likelihood, so the GRPO figure has no series to draw for it.
        return _lm_figure(mode, records, plotdir, plt, tag)
    return _method_figure(mode, records, plotdir, plt, tag)


def method_monitor(mode, records, plotdir, tag):
    """Live per-stage dynamics, redrawn every --plot_every_steps during training (PDF).
    Returns the path of the figure, or None if there is nothing to draw."""
    if not records:
        return None
    records = _live_run(records)
    plt = _plt()
    os.makedirs(plotdir, exist_ok=True)
    return _stage_figure(mode, records, plotdir, plt, tag)


def bpe_monitor(history, plotdir, tag=""):
    """bpe_dynamics<_tag>.pdf -- how the byte-level BPE vocabulary is LEARNED over its merges.

    `history` is BPETokenizer.build_history: one record per merge with the merged pair's corpus
    frequency, the number of distinct candidate pairs, the vocabulary size, the corpus token
    count, the new symbol's byte length and the compression ratio (bytes/token). Returns the
    figure path, or None if there is no history (e.g. an HF tokenizer, which has no merges)."""
    if not history:
        return None
    plt = _plt()
    os.makedirs(plotdir, exist_ok=True)
    return _bpe_figure(history, plotdir, plt, tag)


def _bpe_figure(history, plotdir, plt, tag):
    """1 row x 4: merged-pair frequency (log y), corpus size in tokens, compression (bytes/token),
    and distinct candidate pairs -- all against the merge index (the BPE "step"). x is the merge
    number; no main title, house style."""
    if not history:
        return None
    step = [r["merge"] for r in history]
    def col(k): return [r.get(k) for r in history]
    c = C["bpe"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))

    ax = axes[0]
    ax.plot(step, col("pair_freq"), color=c, lw=1.6)
    ax.set_yscale("log")
    ax.set_title("Merged-pair frequency", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("corpus count of merged pair"); _sci_x(ax)

    ax = axes[1]
    ax.plot(step, col("tokens"), color=c, lw=1.8)
    ax.set_title("Corpus size (tokens)", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("total tokens"); _sci_x(ax)

    ax = axes[2]
    ax.plot(step, col("bytes_per_token"), color=c, lw=1.8)
    ax.set_title("Compression (bytes / token)", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("mean token length (bytes)"); _sci_x(ax)

    ax = axes[3]
    ax.plot(step, col("distinct_pairs"), color=c, lw=1.8)
    ax.set_title("Distinct candidate pairs", weight="bold")
    ax.set_xlabel("merge"); ax.set_ylabel("#distinct adjacent pairs"); _sci_x(ax)

    fig.tight_layout()
    out = os.path.join(plotdir, f"bpe_dynamics{_sfx(tag)}.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _lr_panel(ax, step, lrs, c):
    """Learning-rate schedule panel, shared by every stage figure (log y: a cosine decay to
    a tenth of the peak is unreadable on a linear axis next to the peak)."""
    pts = [(s, v) for s, v in zip(step, lrs) if v is not None]
    if pts:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=c, lw=1.8)
        if min(ys) > 0:
            ax.set_yscale("log")
    ax.set_title("Learning rate", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)


# key, title, y label, y limits, LOGARITHMIC y
#
# The last three are logarithmic. They are quantities bounded above by 1 that spend the run
# near their floor -- the decay sits around 0.78, and the channel fractions live in the
# thousandths -- so on a linear axis scaled to [0, 1] they are three flat lines pinned to an
# edge, and every movement in them is invisible. A log axis is what makes a fraction going
# from 1e-5 to 3e-3 legible as the two-order-of-magnitude change it is. The cost is that a
# value of EXACTLY zero has no place on the axis; _ssm_rows says so in the panel rather than
# letting matplotlib drop the point silently.
SSM_PANELS = [
    ("horizon_med", "Memory horizon (median)", "tokens", None, False),
    ("horizon_p95", "Memory horizon (p95)", "tokens", None, False),
    ("selectivity", r"Selectivity: std$_t$ of the decay", None, None, False),
    ("write_ratio", "Residual write ratio", r"$\|$out$\|/\|$in$\|$", None, False),
    ("a_mean", "Mean decay $a$", None, None, True),
    ("frac_long", "Long-memory channels ($a > 0.99$)", "fraction", None, True),
    ("frac_local", "Local channels ($a < 0.5$)", "fraction", None, True),
    ("state_norm", "Mean state magnitude", r"$|h|$", None, False),
]

NCOL = 4          # at most four panels per row, in every figure this module draws


def _grid(n, plt, width_per=5.6, height_per=4.0):
    """A grid of `n` panels, at most NCOL per row, with the unused slots removed rather than
    left blank. Returns (fig, [axes in reading order])."""
    rows = (n + NCOL - 1) // NCOL
    fig, grid = plt.subplots(rows, NCOL, figsize=(width_per * NCOL, height_per * rows),
                             squeeze=False)
    axes = [grid[r][c] for r in range(rows) for c in range(NCOL)]
    for spare in axes[n:]:
        fig.delaxes(spare)
    return fig, axes[:n]


def _ssm_rows(records, axes, plt):
    """Draw the state space diagnostics: ONE LINE PER LAYER, plus the across-layer MEAN.

    The history stores what was measured -- a list of per-layer dicts under "ssm", on the
    steps where collection was on -- and nothing else. The averaging happens here, at draw
    time, so the record stays raw and any later analysis can summarise it differently.

    Layers are coloured light to dark with depth; the mean is the bold dashed line on top.
    Both are drawn because they answer different questions: the mean says whether the module
    as a whole is learning to remember, the spread says whether the layers are specialising --
    an early block staying local while a late one grows its horizon is the interesting case,
    and the mean alone would hide it."""
    rows = [(r["step"], r["ssm"]) for r in records if r.get("ssm")]
    if not rows:
        return
    step = [s for s, _ in rows]
    n_layer = max(len(v) for _, v in rows)
    cmap = plt.get_cmap("viridis")
    shades = [cmap(0.12 + 0.76 * (i / max(n_layer - 1, 1))) for i in range(n_layer)]
    def positive(series):
        """On a log axis a value of exactly zero cannot be drawn. Blank those points and count
        them, so the panel can report how many were withheld instead of showing a line that
        mysteriously starts part way along."""
        out = [(q if (q is not None and q > 0) else None) for q in series]
        return out, sum(1 for a, b in zip(series, out) if a is not None and b is None)

    for (key, title, ylab, ylim, logy), ax in zip(SSM_PANELS, axes):
        drawn, zeros = 0, 0
        for li in range(n_layer):
            y = [(v[li].get(key) if li < len(v) else None) for _, v in rows]
            if logy:
                y, nz = positive(y)
                zeros += nz
            if all(q is None for q in y):
                continue
            ax.plot(step, y, color=shades[li], lw=1.1, alpha=0.85,
                    label=(f"layer {li}" if key == "horizon_med" else None))
            drawn += 1
        mean = []
        for _, v in rows:
            vals = [d.get(key) for d in v if d.get(key) is not None]
            mean.append(sum(vals) / len(vals) if vals else None)
        if logy:
            mean, _ = positive(mean)
        ax.plot(step, mean, color="k", lw=2.0, ls="--",
                label=("mean over layers" if key == "horizon_med" else None))
        if logy:
            ax.set_yscale("log")
            if drawn:
                # these are all bounded above by 1 (a decay, or a fraction of channels), so
                # pinning the top makes the panels comparable between runs
                ax.set_ylim(top=1.05)
                if zeros:
                    ax.text(0.98, 0.03, f"{zeros:,} point(s) at exactly 0 omitted",
                            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
                            color=GREY_NOTE, style="italic")
            else:
                ax.set_ylim(1e-6, 1.05)
                ax.text(0.5, 0.5, "identically zero\nover the whole run",
                        transform=ax.transAxes, ha="center", va="center", fontsize=8,
                        color=GREY_NOTE, style="italic", linespacing=1.4)
        elif ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    axes[0].legend(fontsize=6.5, ncol=2, loc="best")


def _lm_figure(mode, records, plotdir, plt, tag):
    """<mode>_dynamics<_tag>.pdf for the LM stages (pretrain, sft).

    Loss, Perplexity, Next-token accuracy, Gradient norm, Learning rate; and, when the history
    carries state space diagnostics (pretraining does, at --ssm_stats_every), eight more
    panels with one line per layer and the across-layer mean over them. Four panels per row,
    unused slots deleted. x is the training step throughout."""
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C.get(mode, "#6a994e")
    has_ssm = any(r.get("ssm") for r in records)
    fig, axes = _grid(5 + (len(SSM_PANELS) if has_ssm else 0), plt)

    lm_panels = [("loss", "Loss (length-normalized LM)", "nats/token", None),
                 ("ppl", "Perplexity", None, None),
                 ("acc", "Next-token accuracy (response)", "accuracy", (0, 1.02)),
                 ("gnorm", "Gradient norm (pre-clip)", None, None)]
    for (key, title, ylab, ylim), ax in zip(lm_panels, axes):
        tr = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [tr])
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[4], step, col("lr"), c)
    if has_ssm:
        _ssm_rows(records, axes[5:], plt)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _reward_figure(mode, records, plotdir, plt, tag):
    """reward_dynamics<_tag>.pdf -- 1 row x 6 for the reward-model stage (sigmoid + BCE,
    chosen -> 1, rejected -> 0). EVERY tracked value gets a panel:

        1 BCE loss
        2 accuracy: pair accuracy Pr(s_w > s_l) and per-class sign accuracies
        3 mean sigmoid score of the chosen and rejected responses (the two class means)
        4 mean logit margin s_w - s_l
        5 gradient norm (pre-clip)
        6 learning rate (cosine)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, axes = plt.subplots(1, 6, figsize=(27, 4.0))

    ax = axes[0]
    t = _plot_trend(ax, step, col("loss"), c); _fit_to_trend(ax, [t])
    ax.set_title("BCE loss", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)

    ax = axes[1]
    _plot_trend(ax, step, col("pair_acc"), c, label=r"pair acc $\Pr(s_w>s_\ell)$")
    _plot_trend(ax, step, col("acc_w"), C_CHOSEN, label="chosen acc", lw=1.4, ls="--")
    _plot_trend(ax, step, col("acc_l"), C_REJECT, label="reject acc", lw=1.4, ls=":")
    ax.set_ylim(0, 1.02); ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Classification accuracy", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)

    ax = axes[2]
    tw = _plot_trend(ax, step, col("s_w"), C_CHOSEN, label=r"$\sigma(s_w)$ chosen")
    tl = _plot_trend(ax, step, col("s_l"), C_REJECT, label=r"$\sigma(s_\ell)$ rejected", ls="--")
    _fit_to_trend(ax, [tw, tl]); ax.set_ylim(0, 1.02)
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Mean sigmoid score", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)

    ax = axes[3]
    t = _plot_trend(ax, step, col("margin"), c); _fit_to_trend(ax, [t])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title(r"Logit margin $s_w - s_\ell$", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    ax = axes[4]
    t = _plot_trend(ax, step, col("gnorm"), c); _fit_to_trend(ax, [t])
    ax.set_title("Gradient norm (pre-clip)", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    _lr_panel(axes[5], step, col("lr"), c)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _rlhf_figure(mode, records, plotdir, plt, tag):
    """rlhf_dynamics<_tag>.pdf -- 2 rows x 4 for the PPO stage. EVERY tracked value gets a
    panel:

        1 mean reward-model score of the rollouts (the number PPO maximises)
        2 per-token KL(policy || SFT reference) of the rollouts (what the penalty holds down)
        3 PPO policy (clipped-surrogate) loss
        4 value loss
        5 total loss (policy + vf_coef * value)
        6 policy entropy of the rollouts (collapse indicator)
        7 mean response length (collapse indicator)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.0))

    panels = [("reward", "Reward-model score (rollouts)", None, axes[0, 0]),
              ("kl_ref", r"Per-token KL$(\pi\,\|\,\pi_{\mathrm{sft}})$", "nats/token", axes[0, 1]),
              ("pg_loss", "PPO policy loss", None, axes[0, 2]),
              ("v_loss", "Value loss", None, axes[0, 3]),
              ("loss", "Total loss", None, axes[1, 0]),
              ("entropy", "Policy entropy", "nats/token", axes[1, 1]),
              ("resp_len", "Mean response length", "tokens", axes[1, 2])]
    _lr_panel(axes[1, 3], step, col("lr"), c)
    for key, title, ylab, ax in panels:
        t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
        if key in ("reward", "kl_ref", "pg_loss", "loss"):
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _cot_figure(mode, records, plotdir, plt, tag):
    """cot_dynamics<_tag>.pdf -- 4 rows x 4 for the chain-of-thought GRPO stage. EVERY quantity
    the stage records has a panel, because a run of this kind fails in several different ways
    and each one is visible in exactly one of them.

    THE FIRST ROW IS WHAT THE STAGE EXISTS TO SHOW.
        1 THINK LENGTH, mean -- words between the think tags. The reported "aha moment" is a
          rising trend here: the policy discovers that deliberating longer earns reward.
        2 THINK LENGTH, longest and 90th percentile. The mean HIDES THE ARRIVAL: longer
          deliberation appears in a few completions first, and a mean over 24 moves by a word
          or two while the best has doubled.
        3 VERIFIED ACCURACY -- extracted answer against gold. Unlike a reward-model score this
          cannot be gamed by sounding fluent.
        4 mean verifier reward, what GRPO maximises.

    THE SECOND ROW IS WHETHER GRPO CAN LEARN AT ALL.
        5 DEAD GROUPS -- the fraction whose completions all scored the same. Their advantages
          are zero, so they contribute nothing; a run sitting near 1.0 here is doing rollouts
          for no gradient, and no other panel will say so. THE FIRST THING TO READ when reward
          will not move.
        6 within-group reward spread, the denominator of the advantage.
        7 mean |advantage|, the size of the signal actually reaching the policy.
        8 format rate: well-formed think/answer structure.

    THE THIRD ROW IS THE REWARD'S PARTS AND THE CONSTRAINT.
        9 grounded rate: thinking long enough AND mentioning the problem's numbers.
        10 reflection rate: a self-correction marker ("wait", "actually") -- a crude proxy.
        11 response length in tokens, beside the think length in words.
        12 KL to the frozen reference (k3), the constraint being paid.

    THE FOURTH ROW IS OPTIMISATION HEALTH.
        13 GRPO policy loss, 14 total loss, 15 policy entropy (collapse indicator),
        16 clipped fraction; the gradient norm and the learning rate share the last panel.
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(4, 4, figsize=(18, 15.0))
    axes = [a for row in ax2d for a in row]
    panels = [("think_len", "Think length, mean (the aha-moment curve)", "words"),
              (("think_len_max", "think_len_p90"), "Think length, longest and p90", "words"),
              ("accuracy", "Verified answer accuracy", "fraction correct"),
              ("reward", "Verifier reward", None),
              ("dead_groups", "Dead groups (no spread, no gradient)", "fraction of groups"),
              ("reward_std", "Within-group reward spread", None),
              ("adv_abs", "Mean |advantage|", None),
              (("think_fmt", "answer_fmt"), "Tag rates, think and answer",
               "fraction of rollouts"),
              ("grounded", "Grounded thinking rate", "fraction of rollouts"),
              ("aha", "Reflection rate", "fraction of rollouts"),
              ("resp_len", "Response length", "tokens"),
              ("kl_ref", r"KL$(\pi\,\|\,\pi_{\mathrm{ref}})$", "nats/token"),
              ("pg_loss", "GRPO policy loss", None),
              ("loss", "Total loss", None),
              ("entropy", "Policy entropy", "nats/token"),
              ("clip_frac", "Clipped fraction", "fraction of tokens")]
    UNIT = ("accuracy", "aha", "format", "clip_frac", "dead_groups", "grounded",
            "think_fmt", "answer_fmt")
    for (key, title, ylab), ax in zip(panels, axes):
        if isinstance(key, tuple):
            # two series in one panel: the same hue would make them one line
            labs = {"think_len_max": "longest", "think_len_p90": "p90",
                    "think_fmt": "<think>", "answer_fmt": "<answer>"}
            ts = [_plot_trend(ax, step, col(k), cc, label=labs.get(k, k), ls=st)
                  for k, cc, st in zip(key, (c, "#888888"), ("-", "--"))]
            if all(k in UNIT for k in key):
                ax.set_ylim(0, 1.02)
            _fit_to_trend(ax, ts)
            if any(any(v is not None for v in col(k)) for k in key):
                ax.legend(fontsize=8, frameon=False)
        else:
            t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
            if key in UNIT:
                ax.set_ylim(0, 1.02)
            if key in ("reward", "pg_loss", "loss", "kl_ref", "adv_abs", "reward_std"):
                ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[15], step, col("lr"), c)
    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _distill_figure(mode, records, plotdir, plt, tag):
    """distill_dynamics<_tag>.pdf -- 2 rows x 4 for the sequence-level distillation stage
    (teacher generates, student maximises the likelihood of the generation under a KL
    constraint to its own frozen initialisation):

        1 total loss  CE + kl_coef * KL
        2 CE on the teacher's generations (nats/token)
        3 KL to the frozen student initialisation (the constraint being paid)
        4 perplexity of the teacher text under the student
        5 next-token accuracy of the student on the teacher text
        6 mean teacher response length (student tokens)
        7 gradient norm (pre-clip)
        8 learning rate (cosine)
    """
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(2, 4, figsize=(18, 8.0))
    axes = [a for row in ax2d for a in row]
    panels = [("loss", "Total loss", None),
              ("ce", "CE on teacher generations", "nats/token"),
              ("kl_ref", r"KL$(\pi_{\mathrm{student}}\,\|\,\pi_{\mathrm{init}})$",
               "nats/token"),
              ("ppl", "Perplexity", None),
              ("acc", "Next-token accuracy (student)", "accuracy"),
              ("resp_len", "Teacher response length", "tokens"),
              ("gnorm", "Gradient norm (pre-clip)", None)]
    for (key, title, ylab), ax in zip(panels, axes):
        t = _plot_trend(ax, step, col(key), c); _fit_to_trend(ax, [t])
        if key == "acc":
            ax.set_ylim(0, 1.02)
        if key == "kl_ref":
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)
    _lr_panel(axes[7], step, col("lr"), c)
    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _sci_x(ax):
    """Scientific tick format on the x-axis, typeset as a power of ten: the offset reads
    $\\times 10^{4}$ rather than the plain-text '1e4'."""
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


def _method_figure(mode, records, plotdir, plt, tag):
    """dynamics_<tag>_<stage-label>.pdf -- 2 rows x 4, EVERY tracked value of the preference
    stage.

    Row 1: Loss, Implicit reward (r_w / r_l), Reward margin, Preference/chosen/reject
    accuracy, training-time exposure bias.
    Row 2 (held-out probes, sparse series filtered to the probed steps, then optimisation):
    accuracy, margin, D_EB, gradient norm, learning rate.
    x-axis is the training step (scientific format). No main title."""
    if not records:
        return None
    step = [r["step"] for r in records]
    def col(k): return [r.get(k) for r in records]
    c = C[mode]
    fig, ax2d = plt.subplots(2, 5, figsize=(22.5, 8.0))
    axes = [a for row in ax2d for a in row]
    ax = axes[0]
    t = _plot_trend(ax, step, col("loss"), c)
    _fit_to_trend(ax, [t])
    ax.set_title("Loss", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)
    ax = axes[1]
    tw = _plot_trend(ax, step, col("r_w"), C_CHOSEN, label="chosen $r_w$")
    tl = _plot_trend(ax, step, col("r_l"), C_REJECT, label="rejected $r_\\ell$", ls="--")
    _fit_to_trend(ax, [tw, tl])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title("Implicit reward", weight="bold"); ax.set_xlabel("step")
    ax.legend(frameon=False, fontsize=8); _sci_x(ax)
    ax = axes[2]
    t = _plot_trend(ax, step, col("margin"), c); _fit_to_trend(ax, [t])
    ax.axhline(0, ls=":", color="k", alpha=0.4)
    ax.set_title("Reward margin", weight="bold"); ax.set_xlabel("step"); _sci_x(ax)
    ax = axes[3]
    # acc = Pr(margin > 0) is the PREFERENCE accuracy -- whether the pair is ranked
    # (row 2 starts here)
    # correctly. win_acc = Pr(r_w > 0) and lose_acc = Pr(r_l < 0) are the signs of the two
    # rewards, i.e. the LEVEL, and carry no ranking information: both rewards negative gives
    # (win, lose) = (0, 1) whether the ranking is perfect, chance, or inverted. All three
    # series are drawn, with a chance line, so the level and the ranking can be read apart.
    _plot_trend(ax, step, col("acc"), c, label=r"acc $=\Pr(m>0)$")
    _plot_trend(ax, step, col("win_acc"), C_CHOSEN, label="chosen acc", lw=1.4, ls="--")
    _plot_trend(ax, step, col("lose_acc"), C_REJECT, label="reject acc", lw=1.4, ls=":")
    ax.set_ylim(0, 1.02); ax.axhline(0.5, ls=":", color="k", alpha=0.4)
    ax.set_title("Preference / chosen / reject accuracy", weight="bold")
    ax.set_xlabel("step"); ax.legend(frameon=False, fontsize=8); _sci_x(ax)
    ax = axes[4]
    # The exposure probe fires every --eb_every steps, so this series is sparse against a
    # dense step axis; filter to the sampled steps before plotting.
    pairs = [(a, b) for a, b in zip(step, col("eb_probe")) if b is not None]
    if pairs:
        es, ev = zip(*pairs)
        t = _plot_trend(ax, list(es), list(ev), c); _fit_to_trend(ax, [t])
    ax.set_title(r"Exposure bias $D_{\mathrm{KL}}(\pi^{H}\|\pi^{TF})$", weight="bold")
    ax.set_xlabel("step"); ax.set_ylabel("nats/token"); _sci_x(ax)

    # ---- held-out probes (sparse: every --eval_every / --plot_every_steps) ---- #
    def _sparse(ax, key, title, ylab, chance=None, zero=False):
        pts = [(r["step"], r[key]) for r in records if r.get(key) is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color=c)
            _fit_to_trend(ax, [list(ys)])
        if chance is not None:
            ax.axhline(chance, ls=":", color="k", alpha=0.4); ax.set_ylim(0, 1.02)
        if zero:
            ax.axhline(0, ls=":", color="k", alpha=0.4)
        ax.set_title(title, weight="bold"); ax.set_xlabel("step")
        if ylab:
            ax.set_ylabel(ylab)
        _sci_x(ax)

    _sparse(axes[5], "eval_acc", "Held-out preference accuracy", "accuracy", chance=0.5)
    _sparse(axes[6], "eval_margin", "Held-out reward margin", None, zero=True)
    _sparse(axes[7], "eval_deb", r"Held-out $D_{\mathrm{EB}}$", "nats/token", zero=True)

    ax = axes[8]
    t = _plot_trend(ax, step, col("gnorm"), c); _fit_to_trend(ax, [t])
    ax.set_title("Gradient norm (pre-clip)", weight="bold")
    ax.set_xlabel("step"); _sci_x(ax)

    _lr_panel(axes[9], step, col("lr"), c)

    fig.tight_layout()
    out = _figure_path(plotdir, mode, tag)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _moving_average(y, w=None):
    """Trailing/centered moving average of a per-step series (Nones -> nan, ignored).
    Window auto-scales to the run length so long runs get smooth trends."""
    import numpy as np
    a = np.asarray([np.nan if v is None else v for v in y], dtype=float)
    n = a.size
    if n == 0:
        return a
    if w is None:
        w = max(5, n // 200)             # ~0.5% of the run
    # np.convolve(..., mode="same") returns max(len(a), w) points, not len(a). A run shorter
    # than the window therefore produced a TREND LONGER THAN ITS OWN x-AXIS and every caller
    # died in ax.plot with "x and y must have same first dimension" -- swallowed by the live
    # monitor, fatal in the end-of-run pass. Clamp the window to the series.
    w = min(w, n)
    if w <= 1 or n < 3:
        return a
    mask = ~np.isnan(a)
    a0 = np.where(mask, a, 0.0)
    k = np.ones(w)
    num = np.convolve(a0, k, mode="same")
    den = np.convolve(mask.astype(float), k, mode="same")   # count of valid points per window
    return num / np.maximum(den, 1.0)


def _plot_trend(ax, step, y, color, label=None, lw=1.8, ls="-"):
    """Faint raw per-step line (alpha=0.1) with a bold moving-average trend on top.
    Returns the trend series so the caller can scale the y-axis to the trends."""
    ax.plot(step, [None if v is None else v for v in y], color=color, lw=0.7, alpha=0.1)
    t = _moving_average(y)
    ax.plot(step, t, color=color, lw=lw, ls=ls, label=label)
    return t


def _fit_to_trend(ax, trends, pad=0.12, keep_zero=True):
    """Scale the y-axis to the moving-average trends, not to the raw per-step scatter.

    Rationale: methods whose per-step values are heavy-tailed (e.g. RH-DPO under heavy
    masking, |r| up to ~150 while the trend moves in [-46,-20]) otherwise get an axis
    ~50x the trend span, which collapses the chosen/rejected trends onto one another.
    The raw scatter is still drawn and simply clips. `keep_zero` re-includes the y=0
    reference line only when it is close to the trend band (within `pad*8` of a limit)."""
    import numpy as np
    v = np.concatenate([np.asarray(t, dtype=float).ravel() for t in trends if t is not None])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    # A log axis cannot represent 0 or a negative limit. Additive padding around a band whose
    # bottom is small relative to its span drives the lower limit negative, matplotlib refuses
    # the whole set_ylim ("non-positive ylim on a log-scaled axis will be ignored"), and the
    # panel silently falls back to autoscale -- or, if the scale is set AFTER this call, keeps
    # a negative limit and renders blank. Pad multiplicatively instead, and never return 0.
    if ax.get_yscale() == "log":
        v = v[v > 0]
        if v.size == 0:
            return
        lo, hi = float(v.min()), float(v.max())
        f = (hi / lo) ** pad if hi > lo else 10.0 ** pad
        ax.set_ylim(lo / f, hi * f)
        return
    if keep_zero and lo - span * pad * 8 <= 0 <= hi + span * pad * 8:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        # RECOMPUTING span here can undo the span <= 0 fallback above. A trend that is
        # identically zero gives lo = hi = 0, the clamp is a no-op, and span goes back to 0, so
        # set_ylim(0, 0) is called with identical limits -- matplotlib warns
        # ("transformation singular"), discards them and autoscales. The panel still rendered,
        # which is why this survived: the warning was the only symptom. Constant NON-zero trends
        # were never affected, because the clamp widens them to include 0.
        span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - span * pad, hi + span * pad)


_XL_ROLL = r"Rollout ratio (teacher $\to$ rollout)"


def _exposure_bias_figure(evals, roll_m, plotdir, plt, tag):
    """exposure_bias_<tag>.pdf -- 1 row x 4: rollout preference margin, preference accuracy,
    reversal rate (all vs rollout ratio), and the exposure-bias gap bar. No main title."""
    import numpy as np

    def pgrid(m): return sorted(float(k) for k in evals[m]["rollout_curve"])
    def cv(m, k):
        c = evals[m]["rollout_curve"]; return [c[f"{v:.2f}"][k] for v in pgrid(m)]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.0))
    ax = axes[0]
    for m in roll_m:
        ax.plot(pgrid(m), cv(m, "mu_M"), "-o", color=C[m], lw=1.8, ms=5, label=NAME[m])
    ax.axhline(0, ls="--", color="k", alpha=0.4)
    ax.set_title("Rollout preference margin", weight="bold")
    ax.set_xlabel(_XL_ROLL); ax.set_ylabel(r"Reward margin ($\mu_M^H$)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for m in roll_m:
        ax.plot(pgrid(m), cv(m, "acc"), "-o", color=C[m], lw=1.8, ms=5, label=NAME[m])
    ax.axhline(0.5, ls=":", color="k", alpha=0.4)   # chance
    ax.set_title("Preference accuracy", weight="bold")
    ax.set_xlabel(_XL_ROLL); ax.set_ylabel("Rollout accuracy")
    ax = axes[2]
    for m in roll_m:
        ax.plot(pgrid(m), cv(m, "P_rev"), "-o", color=C[m], lw=1.8, ms=5, label=NAME[m])
    ax.set_title("Reversal rate", weight="bold")
    ax.set_xlabel(_XL_ROLL); ax.set_ylabel("Reversal rate")
    ax = axes[3]
    x = np.arange(len(roll_m))
    gaps = [cv(m, "mu_M")[0] - cv(m, "mu_M")[-1] for m in roll_m]   # mu_M^TF - mu_M^RO
    ax.bar(x, gaps, color=[C[m] for m in roll_m], width=0.6, edgecolor="white")
    for xi, g in zip(x, gaps):
        ax.text(xi, g, f"{g:+.2f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in roll_m], fontsize=8, rotation=12, ha="right")
    ax.set_title(r"Exposure bias gap ($\mu_M^{TF} - \mu_M^{RO}$)", weight="bold")
    fig.tight_layout()
    out = os.path.join(plotdir, f"exposure_bias{_sfx(tag)}.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def _displacement(e):
    """(Delta log p_rejected, Delta log p_chosen) for the likelihood-displacement plane.
    Prefer evals/eval.py's dlp_w/dlp_l (rollout-token log-prob change); else fall back to the mean
    teacher-forced implicit reward from the reward-margin histogram (hist r_w/r_l)."""
    import numpy as np
    if e.get("dlp_w") is not None and e.get("dlp_l") is not None:
        return float(e["dlp_l"]), float(e["dlp_w"])
    h = e.get("hist")
    if h and h.get("r_w") and h.get("r_l"):
        return float(np.mean(h["r_l"])), float(np.mean(h["r_w"]))
    return None


def _bounds_figure(evals, roll_m, methods, plotdir, plt, tag):
    """bounds_<tag>.pdf -- 1 row x 3: reward-margin mean vs rollout ratio, Cantelli bound vs
    reversal rate (auto-scaled, no reference line), likelihood-displacement plane. No main title."""
    import numpy as np

    def pgrid(m): return sorted(float(k) for k in evals[m]["rollout_curve"])
    def cv(m, k):
        c = evals[m]["rollout_curve"]; return np.array([c[f"{v:.2f}"][k] for v in pgrid(m)])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # 1: reward margin mean vs rollout ratio
    ax = axes[0]
    for m in roll_m:
        ax.plot(pgrid(m), cv(m, "mu_M"), "-o", color=C[m], lw=1.8, ms=5, label=NAME[m])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(r"Reward margin mean ($\mu_M^H$)", weight="bold")
    ax.set_xlabel(_XL_ROLL); ax.set_ylabel(r"Reward margin mean ($\mu_M^H$)")
    ax.legend(frameon=False, fontsize=8)
    # 2: Cantelli bound vs reversal rate (auto-scale, no reference line)
    ax = axes[1]
    for m in roll_m:
        mu, v = cv(m, "mu_M"), cv(m, "std_M") ** 2
        bound = np.where(mu > 0, v / (v + mu ** 2), 1.0)
        ax.scatter(cv(m, "P_rev"), bound, color=C[m], s=28, alpha=0.85,
                   edgecolor="white", linewidth=0.4, label=NAME[m])
    ax.set_title("Cantelli bound versus reversal rate", weight="bold")
    ax.set_xlabel("Reversal rate"); ax.set_ylabel(r"Cantelli bound ($\frac{v}{v+\mu^2}$)")
    ax.legend(frameon=False, fontsize=8)
    # 3: likelihood-displacement plane (chosen/rejected displacement vs the SFT reference)
    ax = axes[2]
    for m in methods:
        d = _displacement(evals.get(m) or {})
        if d is None:
            continue
        dl, dw = d
        ax.scatter([dl], [dw], color=C[m], s=70,
                   edgecolor="white", linewidth=1, zorder=5, label=NAME[m])
    ax.scatter([0], [0], color="k", s=40, zorder=6)
    ax.annotate("ref (SFT)", (0, 0), textcoords="offset points", xytext=(6, 6), fontsize=8)
    lim = ax.axis(); lo = min(lim[0], lim[2]); hi = max(lim[1], lim[3])
    ax.plot([lo, hi], [lo, hi], ls="--", color="#999", lw=1)
    ax.set_xlabel(r"$\Delta\log p_\ell$ (rejected)"); ax.set_ylabel(r"$\Delta\log p_w$ (chosen)")
    ax.set_title("Likelihood-displacement plane", weight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(plotdir, f"bounds{_sfx(tag)}.pdf")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# evaluation figure (called from evals/eval.py) -- same exposure_bias.pdf / bounds.pdf
# as the trainer, from evals/evals/eval.py's RES["results"] (which also carries dlp_w/dlp_l).
# --------------------------------------------------------------------------- #
def eval_figure(RES, methods, p_grid, plotdir, tag=""):
    """Write exposure_bias.pdf (1x4) and bounds.pdf (1x3) -- identical format to the trainer's
    figures -- from evals/evals/eval.py's RES. Returns the list of written paths."""
    plt = _plt()
    os.makedirs(plotdir, exist_ok=True)
    tag = tag or RES.get("tag", "")
    evals = RES["results"]
    roll_m = [m for m in methods if (evals.get(m) or {}).get("rollout_curve")]
    outs = []
    if roll_m:
        outs.append(_exposure_bias_figure(evals, roll_m, plotdir, plt, tag))
        outs.append(_bounds_figure(evals, roll_m, methods, plotdir, plt, tag))
    return [o for o in outs if o]
