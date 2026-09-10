#!/usr/bin/env bash
# Run GeneVariate on THIS machine while borrowing only Colab's GPU.
#
# Everything except the two model forward passes stays local: GEO metadata, the
# MeSH / Cellosaurus / BioLORD reference databases, normalization, assembly and
# the GUI. Colab answers HTTP and nothing else.
#
# The endpoint is the router printed by notebooks/genevariate_colab.ipynb,
# which serves the extractor's own two checkpoints behind one base URL:
#
#   google/gemma-4-12b-it   phase 1 / 1b   (Tissue, Condition, Treatment, Sex)
#   google/gemma-4-e2b-it   Age + phase 2  (served with --reasoning-parser gemma4)
#
# No smaller model is substituted: the prompt artifacts were optimized against
# those checkpoints, so a stand-in silently changes the labels.
#
# Usage:
#   ./run_local_colab_gpu.sh https://HOST/v1                      # launch the GUI
#   ./run_local_colab_gpu.sh https://HOST/v1 extract --input ... --out-dir ...
#
# The notebook also prints the token/worker budgets sized to the Colab card and
# an OPENAI_API_KEY for the router; export those in this shell first. The URL is
# new every Colab session, so pass the current one each time.
set -euo pipefail

URL="${1:-${VLLM_URL:-}}"
if [[ -z "$URL" ]]; then
  echo "usage: $0 <router-url>/v1 [extract <pipeline args...>]" >&2
  echo "  the URL comes from notebooks/genevariate_colab.ipynb." >&2
  exit 1
fi
URL="${URL%/}"
[[ "$URL" == */v1 ]] || URL="$URL/v1"
shift || true

export LLM_BACKEND=vllm
export VLLM_URL="$URL"
export OPENAI_BASE_URL="$URL"
export PHASE1_MODEL=google/gemma-4-12b-it
export AGE_MODEL=google/gemma-4-e2b-it
export PHASE2_MODEL=google/gemma-4-e2b-it

# The router is internet-reachable, so it requires a bearer token; the vendored
# llm_backend already sends OPENAI_API_KEY when the base URL ends in /v1.
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "⚠️  OPENAI_API_KEY is not set - the router will answer 401." >&2
  echo "    Export the key the notebook printed, then re-run." >&2
  exit 2
fi

echo "→ Verifying the Colab endpoint: $VLLM_URL"
served=$(curl -fsS --max-time 20 -H "Authorization: Bearer $OPENAI_API_KEY" \
           "$VLLM_URL/models" 2>/dev/null) || {
  echo "⚠️  Could not reach $VLLM_URL/models" >&2
  echo "    Is the Colab tab still running? Re-run the notebook for a new URL." >&2
  exit 2
}
echo "$served" | python3 -c '
import json, sys
names = {m.get("id", "") for m in json.load(sys.stdin).get("data", [])}
print("   served:", ", ".join(sorted(names)))
need = {"google/gemma-4-12b-it", "google/gemma-4-e2b-it"}
missing = need - names
if missing:
    sys.exit(f"   ✗ endpoint does not serve {sorted(missing)} - this is not the "
             f"extractor pipeline. Refusing to run.")
print("   ✓ both extractor checkpoints are served")
'

if [[ "${1:-}" == "extract" ]]; then
  shift
  echo "→ Running the headless pipeline against the Colab GPU"
  exec genevariate-llm-extract "$@"
else
  echo "→ Launching the GeneVariate GUI (models run on Colab; local GPU stays free)"
  exec genevariate "$@"
fi
