#!/usr/bin/env bash
# One-time / repeatable setup: sync deps and verify LM Studio is reachable.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Syncing dependencies with uv"
uv sync

# Read base_url from config (fallback to default) without extra deps.
BASE_URL="$(uv run python -c 'from ibeto.config import load_config; print(load_config().base_url)')"
MODELS_URL="${BASE_URL%/}/models"

echo "==> Checking LM Studio server at ${MODELS_URL}"
if ! curl -sf "${MODELS_URL}" >/dev/null 2>&1; then
    echo "ERROR: LM Studio is not reachable at ${BASE_URL}."
    echo "  Open LM Studio -> Developer tab -> Start Server, and load a model."
    exit 1
fi

echo "==> LM Studio is up. Loaded models:"
curl -sf "${MODELS_URL}" | uv run python -c 'import json,sys; [print("   -", m["id"]) for m in json.load(sys.stdin)["data"]]'

echo
echo "Setup OK. Launch with:  ./scripts/ibeto        (add --stats for metrics)"
echo "Voice mode builds its one-voice engine automatically on first '--voice' launch."
