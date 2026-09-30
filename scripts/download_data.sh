#!/usr/bin/env bash
# Fetch MAGIC04 into data/ (gitignored — the raw file was previously
# committed and has been removed from tracking).
# Source: UCI ML Repository, "MAGIC Gamma Telescope" (CC BY 4.0).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f data/magic04.data ]; then
  echo "data/magic04.data already present — nothing to do."
  exit 0
fi

mkdir -p data /tmp/magic_uci
curl -fsSL -o /tmp/magic_uci/magic.zip \
  "https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip"
unzip -o -q /tmp/magic_uci/magic.zip -d /tmp/magic_uci
cp /tmp/magic_uci/magic04.data data/magic04.data
.venv/bin/python -c "from src.data import load_magic; print(load_magic().shape)"
