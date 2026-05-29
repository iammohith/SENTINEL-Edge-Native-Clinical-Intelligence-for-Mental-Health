"""
SENTINEL — Central Configuration

Defines the mhGAP condition taxonomy, clinical intent types,
crisis signal lexicon, and system-wide constants.

All domain knowledge constants live here — never scattered across modules.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Final

# ── Project paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
CORPUS_DIR: Final[Path] = DATA_DIR / "corpus"
INDEX_DIR: Final[Path] = DATA_DIR / "index"
AUDIT_LOG_DIR: Final[Path] = PROJECT_ROOT / "audit_logs"
AUDIT_DB_PATH: Final[Path] = AUDIT_LOG_DIR / "sentinel_audit.db"

# ── LanceDB configuration ────────────────────────────────────────────────────────
LANCEDB_TABLE_NAME: Final[str] = "mhgap_chunks"
EMBEDDING_DIM: Final[int] = 768           # nomic-embed-text v1.5 output dimension
MAX_CHUNK_TOKENS: Final[int] = 600        # Retrieval precision degrades above this
MIN_VECTORS_PER_PARTITION: Final[int] = 39  # LanceDB ANN training requirement

# ── Ollama configuration ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL: Final[str] = "http://127.0.0.1:11434"
LLM_MODEL: Final[str] = "gemma4:e4b"
EMBED_MODEL: Final[str] = "nomic-embed-text"
LLM_TIMEOUT_SECONDS: Final[float] = 180.0  # M1 cold-start can exceed 60s on first load

# ── Agentic loop configuration ───────────────────────────────────────────────────
MAX_LOOP_ITERATIONS: Final[int] = 3       # Caps worst-case latency at ~45s
CONFIDENCE_ESCALATE_THRESHOLD: Final[float] = 0.40
RRF_K_CONSTANT: Final[int] = 60          # Standard RRF k parameter
RETRIEVE_TOP_K: Final[int] = 30          # Per retrieval branch before RRF
RERANK_TOP_N: Final[int] = 5            # After reranking, keep top 5

# ── Session configuration ────────────────────────────────────────────────────────
SESSION_CONTEXT_TURNS: Final[int] = 3    # Multi-turn: keep last N Q&A pairs
SESSION_TTL_SECONDS: Final[int] = 1800   # 30 minutes

# ── mhGAP Condition Taxonomy ─────────────────────────────────────────────────────
MNS_CONDITIONS: Final[dict[str, str]] = {
    "DEP": "Depression",
    "PSY": "Psychosis",
    "SUD": "Substance Use Disorders",
    "EPI": "Epilepsy / Seizures",
    "DEM": "Dementia",
    "DLD": "Developmental and Behavioral Disorders (child)",
    "SHI": "Self-Harm / Suicide",
    "OTH": "Other Significant MNS Conditions",
    "GEN": "General mhGAP Principles / MHPSS",
}

CONDITION_CODES: Final[frozenset[str]] = frozenset(MNS_CONDITIONS.keys())


# ── Clinical Intent Types ────────────────────────────────────────────────────────
class MHIntentType(str, Enum):
    """
    Replaces generic PROCEDURE_LOOKUP with mhGAP-specific intent taxonomy.

    Intent classification failure → ESCALATE immediately (Finding #42).
    No regex fallback — in mental health, a misclassified CRISIS_RESPONSE
    intent could route to MEDICATION_GUIDANCE instead of self-harm protocol.
    """

    ASSESSMENT_PROTOCOL = "ASSESSMENT_PROTOCOL"   # How to assess for condition X
    TREATMENT_PROTOCOL = "TREATMENT_PROTOCOL"     # First/second-line treatment
    MEDICATION_GUIDANCE = "MEDICATION_GUIDANCE"   # Drug, dose, contraindications
    REFERRAL_CRITERIA = "REFERRAL_CRITERIA"       # When to escalate to specialist
    FOLLOW_UP_PROTOCOL = "FOLLOW_UP_PROTOCOL"     # Monitoring, review schedules
    CRISIS_RESPONSE = "CRISIS_RESPONSE"           # Acute crisis management
    CONDITION_OVERVIEW = "CONDITION_OVERVIEW"     # General condition information
    CONTRADICTION_CHECK = "CONTRADICTION_CHECK"   # Conflicting protocols
    OUT_OF_SCOPE = "OUT_OF_SCOPE"                 # Not in mhGAP corpus


# ── Crisis Signal Lexicon ────────────────────────────────────────────────────────
#
# Used by crisis_detector.py — runs BEFORE any LLM call.
# Tier-1: hard block, return WHO crisis template immediately (no agentic loop).
# Tier-2: constrained synthesis with crisis banner active.
#
# This lexicon is NOT exhaustive. The semantic similarity screen in
# crisis_detector.py catches paraphrases. Update lexicon based on
# crisis_test_cases.json evaluation results.
#
CRISIS_SIGNALS: Final[dict[str, list[str]]] = {
    "tier_1_hard_block": [
        # Requests for lethal means — must never reach RAG pipeline
        "lethal dose",
        "overdose on",
        "method to kill",
        "how to hang",
        "suicide note",
        "ending my life tonight",
        "want to die tonight",
        "kill myself",
        "commit suicide",
        "end it all",
    ],
    "tier_2_protocol": [
        # Crisis signals — proceed with constrained synthesis + crisis banner
        "doesn't want to be here",
        "not wanting to be here",
        "passive suicidal",
        "self-harm",
        "cutting",
        "voices telling",
        "acute psychosis",
        "seeing things",
        "can't stop drinking",
        "withdrawal seizure",
        "not eating for days",
        "suicidal ideation",
        "thoughts of suicide",
        "harming themselves",
        "harming himself",
        "harming herself",
    ],
}


# ── WHO Crisis Response Template (Tier-1) ────────────────────────────────────────
#
# Returned immediately on Tier-1 crisis signal — no LLM generation.
# Content is pre-validated WHO safe messaging language.
#
WHO_CRISIS_TEMPLATE_TIER1: Final[str] = """
⚠ CRISIS ALERT — IMMEDIATE ACTION REQUIRED

This query contains signals associated with acute risk. Do not proceed with routine consultation.

WHO mhGAP recommends the following immediate steps:

1. **Ensure immediate safety**: Do not leave the patient alone. Remove access to means if possible.
2. **Conduct a direct suicide risk assessment** using the mhGAP SHI module (Self-Harm/Suicide, p.85–100 of mhGAP-IG v2.0).
3. **Contact emergency services** or refer to the nearest emergency psychiatric facility immediately.
4. **Stay with the person** until emergency services arrive or handoff is complete.

**mhGAP Reference**: mhGAP-IG v2.0, Module SHI — Self-Harm/Suicide, pages 85–100.

This response was generated from a pre-validated WHO crisis protocol template.
No AI generation was used. Please verify against the current mhGAP-IG v2.0.
""".strip()


# ── Audit Chain Configuration ─────────────────────────────────────────────────────
AUDIT_SERVICE_NAME: Final[str] = "sentinel-audit"
AUDIT_HMAC_KEY_NAME: Final[str] = "hmac-signing-key"
AUDIT_SQLCIPHER_KEY_NAME: Final[str] = "sqlcipher-key"
AUDIT_SCRUBBED_PREVIEW_CHARS: Final[int] = 50  # Max chars stored in audit log

# ── Rate Limiting ─────────────────────────────────────────────────────────────────
API_RATE_LIMIT: Final[str] = "5/minute"  # Per session token — Finding #46

# ── Language Support ─────────────────────────────────────────────────────────────
# v1 is English-only. nomic-embed-text quality degrades significantly on
# non-English text. Language detection warns at ingestion time.
# v2 roadmap: multilingual-e5-large embedding model.
SUPPORTED_LANGUAGES_V1: Final[list[str]] = ["en"]
