#!/usr/bin/env bash
# run_scaling_laws.sh -- the scaling-law study: loss against model size and data size.
#
# NOT one of the eight pipeline stages. It trains a grid of ZetaGPT-S variants from scratch on
# WikiText-103, fits L(N, D) = E + A/N^alpha + B/D^beta, and writes the figure. Nothing it does
# feeds the pipeline, and nothing in the pipeline depends on it -- but it needs stage 2, since
# it measures loss in the project's own tokens.
#
#   ./run_scaling_laws.sh                      the default grid
#   ./run_scaling_laws.sh --dry_run            print the grid and the cost, train nothing
#   SCALING_MODELS=zs/8,zs/6 ./run_scaling_laws.sh
#   ./run_scaling_laws.sh --fit_only           re-fit and redraw from the saved results
#
# THE SWEEP IS RESUMABLE: every point is written to outputs/eval/scaling_laws.json the moment
# it finishes, and a re-run skips what is already there. Interrupt it freely.
#
# Extra arguments are forwarded, so `./run_scaling_laws.sh --budgets 2e6,8e6` works.
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh          # which reads config_user.yaml over its defaults
echo "=== scaling laws: model size x data size on WikiText-103 ================"
exec $PY -m scaling_laws.run $SCALING_FLAGS "$@"
