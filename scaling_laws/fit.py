"""
scaling_laws/fit.py -- fit L(N, D) to the measured grid, and draw it.

    fit(records)                -> {"E","A","B","alpha","beta", ...}
    figure(records, params, out)-> the PDF

THE FORM, which is Chinchilla's (Hoffmann et al., 2022):

    L(N, D) = E + A / N^alpha + B / D^beta

Three terms with three readings. E is the entropy the corpus cannot be modelled below, so no
amount of either resource reaches it. A/N^alpha is what is lost by using a model of finite
size; B/D^beta what is lost by seeing finite data. The two deficits ADD, which is the
substantive claim: neither resource rescues a shortage of the other, and that is why a
compute-optimal ratio exists at all.

FITTED IN LOG SPACE, ON A HUBER LOSS. Least squares on L itself would let the large-loss
points (the smallest model on the least data) dominate, since they are numerically biggest;
the residual that matters is relative. Huber rather than squared error because a single point
that failed to converge should bend the fit a little, not pivot it.

MINIMISED BY NELDER-MEAD, WRITTEN OUT HERE. The objective has five parameters and a handful of
points, so any derivative-free method is fast enough, and implementing it costs less than a
scipy dependency that the rest of the project does not have. It is restarted from several
initialisations because the surface has a flat valley along (A, alpha) -- a big constant with a
small exponent looks much like a small constant with a big one over one decade of N -- and a
single start lands wherever it started.

THE COMPUTE-OPTIMAL SPLIT falls out of the fit. Minimising L subject to C = 6ND gives

    N_opt ∝ C^{beta/(alpha+beta)},    D_opt ∝ C^{alpha/(alpha+beta)}

so the exponents say how a compute budget should be divided. Equal exponents (alpha = beta)
means parameters and tokens scale together, which is Chinchilla's finding; alpha > beta says
data deserves the larger share.
"""
import json
import math

GREY, PURPLE, ORANGE, GREEN, BLUE = "#4B5963", "#7b2cbf", "#C2610A", "#4C7A3A", "#164A63"


# --------------------------------------------------------------------------- #
# a small Nelder-Mead, so the project keeps its dependency list
# --------------------------------------------------------------------------- #
def nelder_mead(f, x0, step=0.5, iters=4000, tol=1e-10):
    """Minimise `f` over a list of floats. Standard coefficients (1, 2, 0.5, 0.5)."""
    n = len(x0)
    simplex = [list(x0)]
    for i in range(n):
        p = list(x0)
        p[i] += step if p[i] == 0 else step * abs(p[i])
        simplex.append(p)
    vals = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = [sum(p[i] for p in simplex[:-1]) / n for i in range(n)]
        worst = simplex[-1]
        refl = [centroid[i] + (centroid[i] - worst[i]) for i in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centroid[i] + 2.0 * (centroid[i] - worst[i]) for i in range(n)]
            fe = f(exp)
            simplex[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            con = [centroid[i] + 0.5 * (worst[i] - centroid[i]) for i in range(n)]
            fc = f(con)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = con, fc
            else:
                best = simplex[0]
                simplex = [best] + [[best[i] + 0.5 * (p[i] - best[i]) for i in range(n)]
                                    for p in simplex[1:]]
                vals = [f(p) for p in simplex]
    i = min(range(n + 1), key=lambda k: vals[k])
    return simplex[i], vals[i]


def _huber(r, delta=1e-3):
    a = abs(r)
    return 0.5 * r * r if a <= delta else delta * (a - 0.5 * delta)


def predict(params, N, D):
    E, A, B, alpha, beta = params
    return E + A / (N ** alpha) + B / (D ** beta)


def fit(records, restarts=None):
    """Fit (E, A, B, alpha, beta). Parameters are optimised in LOG SPACE for E, A and B, which
    keeps them positive by construction -- a negative irreducible loss or a negative deficit
    would fit the numbers and mean nothing."""
    pts = [(r["N"], r["D"], r["val_loss"]) for r in records
           if r.get("val_loss") == r.get("val_loss")]        # drop NaN
    if len(pts) < 5:
        return None

    def objective(theta):
        logE, logA, logB, alpha, beta = theta
        if not (0.0 < alpha < 3.0 and 0.0 < beta < 3.0):
            return 1e9
        p = (math.exp(logE), math.exp(logA), math.exp(logB), alpha, beta)
        s = 0.0
        for N, D, L in pts:
            try:
                pred = predict(p, N, D)
            except (OverflowError, ZeroDivisionError):
                return 1e9
            if pred <= 0:
                return 1e9
            s += _huber(math.log(pred) - math.log(L))
        return s

    # the flat valley along (A, alpha) is why this restarts rather than trusts one start
    if restarts is None:
        restarts = [(math.log(1.5), math.log(a), math.log(b), al, be)
                    for a in (1e2, 1e4, 1e6) for b in (1e2, 1e4, 1e6)
                    for al, be in ((0.3, 0.3), (0.5, 0.4), (0.1, 0.1))]
    best, best_v = None, float("inf")
    for x0 in restarts:
        x, v = nelder_mead(objective, list(x0))
        if v < best_v:
            best, best_v = x, v
    logE, logA, logB, alpha, beta = best
    p = {"E": math.exp(logE), "A": math.exp(logA), "B": math.exp(logB),
         "alpha": alpha, "beta": beta, "objective": best_v, "n_points": len(pts)}
    resid = [math.log(predict((p["E"], p["A"], p["B"], alpha, beta), N, D)) - math.log(L)
             for N, D, L in pts]
    p["rms_log_residual"] = math.sqrt(sum(r * r for r in resid) / len(resid))
    p["max_abs_log_residual"] = max(abs(r) for r in resid)
    # how a compute budget divides: N ∝ C^a, D ∝ C^b with a + b = 1
    p["a_N"] = beta / (alpha + beta)
    p["b_D"] = alpha / (alpha + beta)
    return p


def summarise(records, p, log=print):
    log("")
    log("fitted scaling law   L(N, D) = E + A/N^alpha + B/D^beta")
    log("-" * 92)
    if p is None:
        log("  too few converged points to fit (5 parameters need at least 5 measurements)")
        log("-" * 92)
        return
    log(f"  E      {p['E']:.4f} nats/token   irreducible: no model and no corpus goes below it")
    log(f"  A      {p['A']:.4e}      alpha  {p['alpha']:.4f}   (model-size term)")
    log(f"  B      {p['B']:.4e}      beta   {p['beta']:.4f}   (data-size term)")
    log(f"  fit    {p['n_points']} points, RMS log residual {p['rms_log_residual']:.4f}, "
        f"max {p['max_abs_log_residual']:.4f}")
    log("")
    log(f"  compute-optimal split:  N ∝ C^{p['a_N']:.3f}   D ∝ C^{p['b_D']:.3f}")
    if abs(p["a_N"] - 0.5) < 0.05:
        log("    ~ equal exponents: parameters and tokens should grow together (Chinchilla)")
    elif p["a_N"] > 0.5:
        log("    parameters take the larger share of extra compute")
    else:
        log("    tokens take the larger share of extra compute")
    log("")
    log(f"  {'model':>8} {'N':>10} {'D':>10} {'measured':>10} {'predicted':>10} {'log resid':>10}")
    for r in sorted(records, key=lambda r: (r["N"], r["D"])):
        if r.get("val_loss") != r.get("val_loss"):
            continue
        pred = predict((p["E"], p["A"], p["B"], p["alpha"], p["beta"]), r["N"], r["D"])
        log(f"  {r['tag']:>8} {r['N'] / 1e6:>9.2f}M {r['D'] / 1e6:>9.1f}M "
            f"{r['val_loss']:>10.4f} {pred:>10.4f} "
            f"{math.log(pred) - math.log(r['val_loss']):>+10.4f}")
    log("-" * 92)


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def figure(records, p, out_path):
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 200, "font.size": 8.5, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    ok = [r for r in records if r.get("val_loss") == r.get("val_loss")]
    fig, grid = plt.subplots(2, 2, figsize=(9.6, 7.0))
    cmap = plt.get_cmap("viridis")

    Ns = sorted({r["N"] for r in ok})
    Ds = sorted({r["D"] for r in ok})
    shade = lambda i, n: cmap(0.12 + 0.76 * (i / max(n - 1, 1)))

    # (a) loss against data, one line per model size
    ax = grid[0][0]
    for i, N in enumerate(Ns):
        rows = sorted([r for r in ok if r["N"] == N], key=lambda r: r["D"])
        ax.plot([r["D"] for r in rows], [r["val_loss"] for r in rows], "o-", ms=3.4, lw=1.3,
                color=shade(i, len(Ns)), label=f"{rows[0]['tag']} ({N / 1e6:.1f}M)")
        if p:
            xs = [Ds[0] * (Ds[-1] / Ds[0]) ** (k / 40) for k in range(41)]
            ax.plot(xs, [predict((p["E"], p["A"], p["B"], p["alpha"], p["beta"]), N, d)
                         for d in xs], lw=0.9, ls=":", color=shade(i, len(Ns)))
    if p:
        ax.axhline(p["E"], color=GREY, lw=0.9, ls="--")
        ax.annotate(f"$E={p['E']:.3f}$", xy=(0.98, p["E"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points", ha="right", fontsize=6.8,
                    color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Loss vs data size", weight="bold")
    ax.set_xlabel("$D$ (training tokens)"); ax.set_ylabel("held-out loss (nats/token)")
    ax.legend(fontsize=6.4, title="model", title_fontsize=6.4)

    # (b) loss against model size, one line per data budget
    ax = grid[0][1]
    for i, D in enumerate(Ds):
        rows = sorted([r for r in ok if r["D"] == D], key=lambda r: r["N"])
        ax.plot([r["N"] for r in rows], [r["val_loss"] for r in rows], "s-", ms=3.4, lw=1.3,
                color=shade(i, len(Ds)), label=f"{D / 1e6:.0f}M tokens")
        if p:
            xs = [Ns[0] * (Ns[-1] / Ns[0]) ** (k / 40) for k in range(41)]
            ax.plot(xs, [predict((p["E"], p["A"], p["B"], p["alpha"], p["beta"]), n, D)
                         for n in xs], lw=0.9, ls=":", color=shade(i, len(Ds)))
    if p:
        ax.axhline(p["E"], color=GREY, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Loss vs model size", weight="bold")
    ax.set_xlabel("$N$ (non-embedding parameters)"); ax.set_ylabel("held-out loss (nats/token)")
    ax.legend(fontsize=6.4, title="data", title_fontsize=6.4)

    # (c) loss against compute, with the envelope every point sits above
    ax = grid[1][0]
    for i, N in enumerate(Ns):
        rows = sorted([r for r in ok if r["N"] == N], key=lambda r: r["C"])
        ax.plot([r["C"] for r in rows], [r["val_loss"] for r in rows], "o-", ms=3.0, lw=1.0,
                alpha=0.85, color=shade(i, len(Ns)))
    front = []
    for r in sorted(ok, key=lambda r: r["C"]):
        if not front or r["val_loss"] < front[-1]["val_loss"]:
            front.append(r)
    ax.plot([r["C"] for r in front], [r["val_loss"] for r in front], "k--", lw=1.8,
            label="best at each compute")
    for r in front:
        ax.annotate(r["tag"], xy=(r["C"], r["val_loss"]), xytext=(0, -9),
                    textcoords="offset points", ha="center", fontsize=6.0, color=GREY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Loss vs compute ($C=6ND$)", weight="bold")
    ax.set_xlabel("$C$ (FLOPs)"); ax.set_ylabel("held-out loss (nats/token)")
    ax.legend(fontsize=6.8)

    # (d) predicted against measured -- the fit's own residual plot
    ax = grid[1][1]
    if p:
        xs = [r["val_loss"] for r in ok]
        ys = [predict((p["E"], p["A"], p["B"], p["alpha"], p["beta"]), r["N"], r["D"])
              for r in ok]
        lo, hi = min(xs + ys) * 0.97, max(xs + ys) * 1.03
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.9, ls=":")
        for i, N in enumerate(Ns):
            sel = [(r["val_loss"], predict((p["E"], p["A"], p["B"], p["alpha"], p["beta"]),
                                           r["N"], r["D"])) for r in ok if r["N"] == N]
            ax.plot([a for a, _ in sel], [b for _, b in sel], "o", ms=4.0,
                    color=shade(i, len(Ns)))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"Fit: RMS log residual {p['rms_log_residual']:.4f}", weight="bold")
        ax.set_xlabel("measured loss"); ax.set_ylabel("predicted loss")
        ax.text(0.03, 0.95,
                f"$L = {p['E']:.3f} + {p['A']:.3g}/N^{{{p['alpha']:.3f}}}"
                f" + {p['B']:.3g}/D^{{{p['beta']:.3f}}}$\n"
                f"$N \\propto C^{{{p['a_N']:.3f}}}$, $D \\propto C^{{{p['b_D']:.3f}}}$",
                transform=ax.transAxes, va="top", fontsize=7.0, color="#1A1A1A")
    else:
        ax.text(0.5, 0.5, "not enough points to fit", transform=ax.transAxes, ha="center",
                va="center", fontsize=9, color=GREY, style="italic")
        ax.set_title("Fit", weight="bold")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
