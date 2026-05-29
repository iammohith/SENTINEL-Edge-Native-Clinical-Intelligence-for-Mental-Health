#!/bin/bash
# SENTINEL — Offline Bootstrap Script (v6 — Mental Health Edition)
#
# Run this ONCE on an internet-connected machine before air-gapping the device.
# Everything pulled here will be available completely offline afterwards.
#
# Usage:
#   chmod +x scripts/bootstrap_offline.sh
#   ./scripts/bootstrap_offline.sh
#
# After running, transfer the entire SENTINEL directory (including ./wheels/ and
# ~/.ollama/models/ and ~/.cache/huggingface/) to the target offline machine.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================================================"
echo "  SENTINEL — Offline Bootstrap (Mental Health Edition)"
echo "================================================================"
echo ""

# ── Step 1: Pull Ollama models ─────────────────────────────────────────────────
echo "[1/5] Pulling Ollama models..."

if ! command -v ollama &>/dev/null; then
    echo "ERROR: ollama not found. Install from https://ollama.ai before running this script."
    exit 1
fi

ollama pull gemma4:e4b
ollama pull nomic-embed-text

echo "  → Warm-up check (bootstrap only — server lifespan handles runtime warm-up)..."
# Send a minimal inference to verify the model loads correctly and measure cold-start time
START_TS=$(date +%s)
ollama run gemma4:e4b "Reply with one word: OK" --nowordwrap 2>/dev/null | head -1
END_TS=$(date +%s)
echo "  → Model warm-up: $((END_TS - START_TS))s (cold-start baseline)"

echo "  ✓ Ollama models ready"
echo ""

# ── Step 2: Download ML model weights (reranker + NLI) ────────────────────────
echo "[2/5] Downloading ML model weights..."
python3 - <<'PYEOF'
from sentence_transformers import CrossEncoder

print("  Downloading reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (22.7M params)...")
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("  ✓ Reranker cached")

print("  Downloading NLI model: cross-encoder/nli-MiniLM2-L6-H768 (~90 MB)...")
CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
print("  ✓ NLI model cached")

print("  Both models cached to ~/.cache/huggingface/")
PYEOF
echo ""

# ── Step 3: Download scispaCy + clinical NLP model ────────────────────────────
echo "[3/5] Downloading scispaCy clinical sentence tokenizer..."
#
# scispaCy en_core_sci_sm handles clinical abbreviations:
#   "b.i.d.", "q.d.", "p.r.n.", "Dr.", "ICD-10 code F32.1"
# These break standard sentence tokenizers and corrupt NLI pair generation.
#
pip3 install --quiet "scispacy>=0.5.4"
pip3 install --quiet "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"

# Also pull NLTK punkt tokenizer as fallback
python3 - <<'PYEOF'
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("  ✓ NLTK punkt fallback tokenizer cached")
PYEOF
echo "  ✓ scispaCy en_core_sci_sm ready"
echo ""

# ── Step 4: Download presidio-analyzer (offline PHI detection) ────────────────
echo "[4/5] Downloading Microsoft Presidio for offline PHI de-identification..."
#
# Presidio-analyzer is fully offline after download.
# It handles: PERSON, DATE_TIME, LOCATION, PHONE_NUMBER, EMAIL, MEDICAL_LICENSE
# Required for HIPAA/GDPR compliance — audit logs must be PHI-free.
#
pip3 install --quiet "presidio-analyzer>=2.2.0" "presidio-anonymizer>=2.2.0"

# presidio's default NLP engine uses spaCy en_core_web_lg
python3 -m spacy download en_core_web_lg 2>/dev/null || \
    python3 -m spacy download en_core_web_md 2>/dev/null || \
    python3 -m spacy download en_core_web_sm

echo "  ✓ Presidio + spaCy model ready"
echo ""

# ── Step 5: Lock and download all Python wheels ───────────────────────────────
echo "[5/5] Generating locked requirements and downloading wheels..."
#
# IMPORTANT: uv pip compile reads pyproject.toml — not requirements.txt.
# The compiled requirements.txt is an intermediate artifact only.
# Offline install command is printed at the end of this script.
#
cd "$PROJECT_ROOT"

if ! command -v uv &>/dev/null; then
    echo "  WARNING: uv not found. Install from https://astral.sh/uv/install.sh"
    echo "  Falling back to pip for wheel download..."
    pip3 install --quiet uv
fi

uv pip compile pyproject.toml -o requirements.txt
pip3 download -r requirements.txt -d ./wheels/ --quiet
echo "  ✓ Wheels cached to ./wheels/ ($(ls ./wheels/ | wc -l | tr -d ' ') packages)"
echo ""

echo "================================================================"
echo "  Bootstrap complete."
echo ""
echo "  To install offline on the target machine:"
echo "    uv pip install --no-index --find-links=./wheels -r requirements.txt"
echo ""
echo "  Before starting SENTINEL, set:"
echo "    export OLLAMA_KEEP_ALIVE=-1"
echo "    ollama serve"
echo "================================================================"
