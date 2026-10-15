#!/usr/bin/env bash
# Fetch the pinned TRACE-5.0 wasm-pack bundle from the HyphAeon monorepo
# release into docs/pkg/.
#
# TRACE50_WASM_TAG is the SINGLE source of truth for the engine version
# served by the GitHub Pages app — .github/workflows/deploy_pages.yml
# runs this script on every deploy, and local devs run it once after
# cloning (or after bumping the pin).
#
# Bump by editing the tag below; the next Pages deploy serves the new
# bundle. The app code never changes — it always imports ./pkg/.
set -euo pipefail

TRACE50_WASM_TAG="${TRACE50_WASM_TAG:-trace50-v5.0.0}"
PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/pkg"
ASSET="trace50-wasm-${TRACE50_WASM_TAG##*-v}-pkg.tar.gz"
URL="https://github.com/veg/HyphAeon/releases/download/${TRACE50_WASM_TAG}/${ASSET}"

echo "[*] Fetching ${ASSET} from ${URL}"
mkdir -p "$PKG_DIR"
curl -fsSL "$URL" | tar -xz -C "$PKG_DIR"
echo "[+] docs/pkg now serves ${TRACE50_WASM_TAG}"
