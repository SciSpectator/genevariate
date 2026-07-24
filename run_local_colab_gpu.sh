#!/usr/bin/env bash
# Run GeneVariate LOCALLY while borrowing Google Colab's GPU/VRAM for the LLM.
#
# The program runs on this machine; every LLM + embedding call is sent to the
# Ollama server running on the Colab GPU, reached through the public tunnel URL
# printed by notebooks/genevariate_colab_gpu.ipynb (cell 4 or the ngrok fallback).
#
# Usage:
#   ./run_local_colab_gpu.sh https://XXXX.trycloudflare.com        # launch the GUI
#   ./run_local_colab_gpu.sh https://XXXX.trycloudflare.com extract --gpl GPL570 --limit 5
#
# The tunnel URL is NEW every Colab session — pass the current one each time.
set -euo pipefail

URL="${1:-${OLLAMA_HOST:-}}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <colab-ollama-url> [extract <extractor args...>]" >&2
  echo "  get the URL from the Colab notebook (cell 4 / ngrok fallback)." >&2
  exit 1
fi
URL="${URL%/}"   # strip any trailing slash
shift || true

export OLLAMA_HOST="$URL"
# Tell the app a parallel slot count is already set so it does NOT try to
# (re)start a *local* ollama server — the GPU lives on Colab.
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-8}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"

echo "→ Verifying Colab GPU endpoint: $OLLAMA_HOST"
if curl -fsS --max-time 20 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
  echo "✅ reachable — models on the Colab GPU:"
  curl -fsS --max-time 20 "$OLLAMA_HOST/api/tags" \
    | python3 -c 'import sys,json; [print("   -",m["name"]) for m in json.load(sys.stdin).get("models",[])]' 2>/dev/null \
    || true
else
  echo "⚠️  Could not reach $OLLAMA_HOST/api/tags" >&2
  echo "    Is the Colab tab still running? Re-run the notebook and use the new URL." >&2
  echo "    If Cloudflare returned 403, use the ngrok fallback URL instead." >&2
  exit 2
fi

if [[ "${1:-}" == "extract" ]]; then
  shift
  echo "→ Running headless extractor on the Colab GPU"
  exec genevariate-llm-extract --ollama-url "$OLLAMA_HOST" "$@"
else
  echo "→ Launching the GeneVariate GUI (LLM runs on Colab; your local nvidia-smi stays idle)"
  exec genevariate "$@"
fi
