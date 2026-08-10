#!/usr/bin/env bash
# compile_latex.sh -- build the tech report OUT OF TREE.
#
# Everything LaTeX writes (aux, log, bbl, blg, out, toc, ...) goes to a temporary build
# directory under /tmp, so no intermediate file is ever created inside the source tree.
# Only the finished PDF is copied back, to
#
#     latex/techreport/outputs/main.pdf
#
#   ./compile_latex.sh                  # build latex/techreport/main.tex
#   ./compile_latex.sh latex/other      # build another document directory
#   KEEP_BUILD=1 ./compile_latex.sh     # keep the /tmp build dir (to read the log)
set -euo pipefail
cd "$(dirname "$0")"

DOCDIR="${1:-latex/techreport}"
DOC="$(basename "${DOC_NAME:-main}" .tex)"
OUTDIR="$DOCDIR/outputs"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/zetagpt_latex_XXXXXX")"

if [ ! -f "$DOCDIR/$DOC.tex" ]; then
  echo "compile_latex: no $DOCDIR/$DOC.tex" >&2
  exit 1
fi

cleanup() {
  if [ "${KEEP_BUILD:-0}" = "1" ]; then
    echo "compile_latex: build directory kept at $BUILD"
  else
    rm -rf "$BUILD"
  fi
}
trap cleanup EXIT

# the whole document directory is copied into the build dir, so \input, \includegraphics
# and \bibliography resolve exactly as they do in place -- but every file LaTeX generates
# lands on the copy, not on the source.
cp -R "$DOCDIR/." "$BUILD/"
rm -rf "$BUILD/outputs"

echo "compile_latex: building $DOCDIR/$DOC.tex in $BUILD"
(
  cd "$BUILD"
  # latexmk runs pdflatex/bibtex as many times as the references need
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error "$DOC.tex" \
    > compile.log 2>&1 || {
      echo "compile_latex: FAILED -- last 40 log lines:" >&2
      tail -40 compile.log >&2
      exit 1
    }
)

mkdir -p "$OUTDIR"
cp "$BUILD/$DOC.pdf" "$OUTDIR/$DOC.pdf"
echo "compile_latex: wrote $OUTDIR/$DOC.pdf"
