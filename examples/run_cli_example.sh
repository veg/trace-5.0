#!/usr/bin/env bash
# Runs the TRACE-5.0 Rust standalone CLI on the example dataset
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/../trace50-rust/target/release/trace50"

if [ ! -f "$BIN" ]; then
    echo "[*] Building trace50 native release binary..."
    cargo build --release --manifest-path "$DIR/../trace50-rust/Cargo.toml" --bin trace50
fi

echo "[*] Executing TRACE-5.0 CLI..."
"$BIN" \
    --fasta "$DIR/sample_alignment.fasta" \
    --meta "$DIR/sample_metadata.csv" \
    --id-col id \
    --date-col date \
    --fdr 0.10 \
    --out-dir "$DIR/example_output"

echo "[+] Done. Output dossier written to $DIR/example_output/trace50_visualization_dossier.json"
