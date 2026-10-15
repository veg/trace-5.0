#!/usr/bin/env bash
# Runs the TRACE-5.0 standalone CLI on the example dataset.
# The engine lives in the HyphAeon monorepo and ships on PyPI:
#   pip install trace50
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v trace50 >/dev/null 2>&1; then
    echo "[!] trace50 not on PATH — install it first:"
    echo "    pip install trace50"
    exit 1
fi

echo "[*] Executing TRACE-5.0 CLI..."
trace50 \
    --fasta "$DIR/sample_alignment.fasta" \
    --meta "$DIR/sample_metadata.csv" \
    --id-col id \
    --date-col date \
    --fdr 0.10 \
    --out-dir "$DIR/example_output"

echo "[+] Done. Output dossier written to $DIR/example_output/trace50_visualization_dossier.json"
