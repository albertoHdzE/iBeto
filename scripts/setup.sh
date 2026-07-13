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

# Pre-build the isolated XTTS voice environment (tts_engine="xtts", the default)
# so the first voice launch doesn't pay the one-time env build + model download.
# Best-effort: iBeto falls back to fast per-language voices if this is skipped.
echo
echo "==> Preparing the XTTS voice environment (one-time; torch + ~1.8 GB model)."
echo "    Ctrl-C to skip — voice still works with fast per-language voices."
COQUI_TOS_AGREED=1 uv run --no-project \
    --with "coqui-tts>=0.24.0" --with "torch>=2.1" --with "torchaudio>=2.1" \
    --with "transformers==4.46.3" --with cutlet --with unidic-lite --with pypinyin \
    --with click --with spacy \
    python ibeto/audio/xtts_worker.py <<< '{"text":"ok","lang":"en"}' >/dev/null 2>&1 \
    && echo "    XTTS ready." \
    || echo "    (XTTS prewarm skipped/failed; it will build on first voice launch.)"

echo
echo "Setup OK. Launch with:  ./scripts/ibeto        (add --stats for metrics)"
