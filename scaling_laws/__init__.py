"""
scaling_laws -- how ZetaGPT's loss depends on MODEL SIZE and DATA SIZE, measured rather than
assumed.

    python -m scaling_laws.run          the whole grid, the fit, and the figure
    ./run_scaling_laws.sh               the same, through config.sh

    scaling_laws.wikitext   the corpus: WikiText-103, fetched, tokenised once, cached
    scaling_laws.grid       the (N, D) grid: a ladder of ZetaGPT-S variants x token budgets
    scaling_laws.train      one grid point -- train from scratch, return held-out loss
    scaling_laws.fit       L(N, D) = E + A/N^alpha + B/D^beta, and the figure

WHY A SEPARATE CORPUS. The pipeline's own pretraining data is small and domain-specific, which
is the wrong instrument for this question: a scaling law is a statement about how loss falls
as a model is given more parameters and more DISTINCT tokens, and a corpus that runs out
forces the larger budgets to repeat data, which flattens the data axis for a reason that has
nothing to do with scaling. WikiText-103 is ~100M tokens of one clean domain, large enough
that every budget in the default grid is single-epoch, and standard enough that the exponents
can be compared with published ones.

WHY ZetaGPT-S VARIANTS. The ladder is built by shrinking the small scheme, not by mixing the
three shipped schemes, so every point differs from every other only in depth and width. The
head dimension stays 64 throughout and the context window stays fixed, which is what makes the
fitted exponent a statement about SIZE rather than about three architectures that happen to
have different names.
"""
