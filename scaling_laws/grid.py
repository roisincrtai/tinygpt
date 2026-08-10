"""
scaling_laws/grid.py -- the (model size, data size) grid the study sweeps.

    ladder(names=None)      -> [ {tag, n_layer, n_head, n_embd, N_params, ...}, ... ]
    budgets(spec, n_tokens) -> [ D_1, D_2, ... ] in tokens, capped at what the corpus holds
    points(models, budgets) -> the grid as a list of (model, D) with a stable id

THE MODEL LADDER is ZetaGPT-S shrunk, not the shipped schemes mixed. Depth and width move
together at a CONSTANT aspect ratio d/L = 32, the shipped small scheme's own, and THE HEAD
DIMENSION STAYS 64 -- so every rung differs from every other in
scale alone. Mixing d_h across the ladder would confound "the model is bigger" with "a head
sees more", and the fitted exponent would be a statement about neither.

    tag       layers  heads  d_model  d_h   aspect d/L   N (non-embedding)
    zs/8         4      2      128    64        32              1.13M
    zs/6         6      3      192    64        32              3.79M
    zs/4         8      4      256    64        32              8.96M
    zs/2        12      6      384    64        32             30.20M
    zs          16      8      512    64        32             71.51M

N IS THE NON-EMBEDDING PARAMETER COUNT, L(17d^2 + 25d), and that choice matters. The embedding
is V*d = 50,259d, which at d=128 is 6.4M against 1.1M of actual network -- five times the
model. Counting it would make the small rungs look enormous while adding almost no capacity
per token processed, and the fitted exponent would mostly be measuring vocabulary. Excluding
embeddings is also what Kaplan et al. do, so the exponent is comparable with published ones.

THE DATA AXIS is a token budget spent ONCE. Steps are derived, steps = D / (batch * context),
so a point never sees a token twice: repeated data makes loss fall for a reason that is not
scale, and a grid that repeats at the large budgets and not at the small ones has that
confound distributed unevenly across exactly the axis being fitted.
"""
HEAD_DIM = 64                 # fixed across the ladder; see model.zetagpt.HEAD_DIM

# (tag, n_layer, n_embd). n_head follows from HEAD_DIM, so it is never written down twice.
LADDER = [
    ("zs/8",  4,  128),
    ("zs/6",  6,  192),
    ("zs/4",  8,  256),
    ("zs/2", 12,  384),
    ("zs",   16,  512),
]

# Token budgets, in tokens. Geometric with a ratio of ~4, which spreads four points over a
# little more than 1.5 decades -- enough for a slope to be identified without the smallest
# budget being pure noise. Capped at the corpus at run time.
BUDGETS = [2_000_000, 8_000_000, 32_000_000, 100_000_000]


def n_params(n_layer, n_embd):
    """Non-embedding parameters: L(17d^2 + 25d).

    Per block: 4d^2 in the state space module (in_proj 2d^2, a_proj d^2, out_proj d^2), 5d^2 in
    attention (qkv 3d^2, gate d^2, proj d^2), 8d^2 in the feed-forward (two 4x layers); and
    9d + 5d + 5d biases plus 6d of LayerNorm. Excludes the tied embedding, the final LayerNorm
    and the MASK vector, which together are V*d + 3d and are not capacity applied per token."""
    return n_layer * (17 * n_embd * n_embd + 25 * n_embd)


def n_embedding(n_embd, vocab):
    return vocab * n_embd


def ladder(tags=None):
    """The model ladder as dicts ready to hand to ZetaGPT, plus the parameter count each rung
    is plotted against. `tags` selects a subset, in the order given."""
    by_tag = {t: (t, L, d) for t, L, d in LADDER}
    chosen = [by_tag[t] for t in tags] if tags else list(LADDER)
    out = []
    for tag, L, d in chosen:
        if d % HEAD_DIM:
            raise ValueError(f"{tag}: n_embd={d} is not a multiple of the head dimension "
                             f"{HEAD_DIM}")
        out.append({"tag": tag, "n_layer": L, "n_embd": d, "n_head": d // HEAD_DIM,
                    "N": n_params(L, d), "aspect": d / L})
    return out


def budgets(spec, n_tokens=0):
    """The data budgets actually usable: `spec` (a list, or a comma-separated string) with
    anything larger than the corpus REMOVED rather than silently clipped.

    Clipping would be worse than dropping: two budgets that both clip to the corpus size become
    the same experiment run twice, and the fit would see two identical points at different
    nominal D and conclude the data axis is flat there."""
    if isinstance(spec, str):
        spec = [int(float(x)) for x in spec.split(",") if x.strip()]
    spec = sorted({int(d) for d in spec if int(d) > 0})
    if not n_tokens:
        return spec
    keep = [d for d in spec if d <= n_tokens]
    return keep


def steps_for(D, batch, context):
    """Optimisation steps to spend exactly D tokens once. At least one."""
    return max(1, int(D // (batch * context)))


def points(models, data_budgets, batch, context):
    """The grid, smallest first, so a partial run is still a usable (if truncated) study."""
    out = []
    for m in models:
        for D in data_budgets:
            out.append({
                "id": f"{m['tag']}@{D}",
                "tag": m["tag"], "n_layer": m["n_layer"], "n_head": m["n_head"],
                "n_embd": m["n_embd"], "N": m["N"], "D": D,
                "steps": steps_for(D, batch, context),
                # Kaplan's C = 6ND FLOPs: 2 for the multiply-accumulate, x3 for forward plus
                # backward. Attention's own quadratic term is left out, as it is there, because
                # at these widths and a 512 context it is a few percent of the total.
                "C": 6.0 * m["N"] * D,
            })
    return sorted(out, key=lambda p: (p["N"], p["D"]))


def describe(models, data_budgets, batch, context, vocab, log=print):
    """Print the grid before anything is trained, because that is the moment to notice it is
    ten times bigger than intended."""
    log("")
    log("scaling-law grid")
    log("-" * 92)
    log(f"  {'model':>8} {'layers':>7} {'heads':>6} {'d_model':>8} {'N (non-emb)':>13} "
        f"{'embedding':>11} {'aspect d/L':>11}")
    for m in models:
        log(f"  {m['tag']:>8} {m['n_layer']:>7} {m['n_head']:>6} {m['n_embd']:>8} "
            f"{m['N'] / 1e6:>12.2f}M {n_embedding(m['n_embd'], vocab) / 1e6:>10.2f}M "
            f"{m['aspect']:>11.1f}")
    log("")
    log(f"  {'D (tokens)':>13} {'steps':>9}   (batch {batch} x context {context} = "
        f"{batch * context:,} tokens/step)")
    for D in data_budgets:
        log(f"  {D / 1e6:>12.1f}M {steps_for(D, batch, context):>9,}")
    grid = points(models, data_budgets, batch, context)
    tot_tokens = sum(p["D"] for p in grid)
    tot_steps = sum(p["steps"] for p in grid)
    log("")
    log(f"  {len(grid)} runs, {tot_steps:,} optimisation steps, {tot_tokens / 1e9:.2f}B tokens "
        f"processed in total")
    log(f"  compute spans {min(p['C'] for p in grid):.2e} .. {max(p['C'] for p in grid):.2e} "
        f"FLOPs (C = 6ND)")
    log("-" * 92)
    return grid
