#!/usr/bin/env bash
# Build IoTJ paper PDF (requires texlive: pdflatex + bibtex)
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v pdflatex >/dev/null 2>&1; then
  echo "ERROR: pdflatex not found. Install TeX Live, e.g.:"
  echo "  sudo apt install texlive-latex-recommended texlive-bibtex-extra"
  exit 127
fi
python3 validate_citations.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "Built: $(pwd)/main.pdf"
