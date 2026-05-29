"""
SENTINEL — Clinical Crisis Detector

Mandatory first gate (STEP 1). Scans the incoming patient/clinician query for acute
crisis signals (suicidal ideation, self-harm, lethal means, active psychosis) BEFORE
routing to any LLM inference call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sentinel.config import (
    CRISIS_SIGNALS,
    WHO_CRISIS_TEMPLATE_TIER1,
)
from sentinel.ingestion.embedder import embed_query

logger = logging.getLogger(__name__)

# Pre-embedded exemplars of Tier-1 crisis queries (for offline similarity screening)
TIER1_EXEMPLARS = [
    "how to commit suicide",
    "i want to kill myself",
    "best way to hang myself",
    "how to swallow a lethal dose of amitriptyline",
    "i wrote a suicide note and am ready to end it",
    "overdose to die",
    "i want to end my life tonight"
]

# Pre-embedded exemplars of Tier-2 crisis queries (psychosis, passive ideation)
TIER2_EXEMPLARS = [
    "i have been cutting my wrists",
    "i hear voices telling me to hurt myself",
    "my patient has a seizure from alcohol withdrawal",
    "i haven't eaten for five days because the voices forbid it",
    "passive thoughts of wanting to die",
    "i wish i was dead"
]


@dataclass
class CrisisResult:
    level: str  # "NONE" | "TIER_2" | "TIER_1"
    matched_signal: Optional[str]
    who_crisis_template: Optional[str]  # WHO safe response template for TIER_1


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Computes cosine similarity between two 768-dim vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


async def detect_crisis(query: str) -> CrisisResult:
    """
    Mandatory screening:
    Pass 1: Microsecond-scale keyword scanning against CRISIS_SIGNALS lexicon.
    Pass 2: Semantic similarity embedding scan against exemplars (if Pass 1 is negative).
    """
    clean_query = query.strip().lower()
    
    # ── Pass 1: Keyword match ──────────────────────────────────────────────────
    # Check Tier-1 (Hard Block)
    for keyword in CRISIS_SIGNALS["tier_1_hard_block"]:
        if keyword in clean_query:
            logger.warning(f"Crisis detected (Keyword Tier-1): Matched '{keyword}'")
            return CrisisResult(
                level="TIER_1",
                matched_signal=f"Keyword match: {keyword}",
                who_crisis_template=WHO_CRISIS_TEMPLATE_TIER1
            )
            
    # Check Tier-2 (Protocol Active)
    for keyword in CRISIS_SIGNALS["tier_2_protocol"]:
        if keyword in clean_query:
            logger.info(f"Crisis detected (Keyword Tier-2): Matched '{keyword}'")
            return CrisisResult(
                level="TIER_2",
                matched_signal=f"Keyword match: {keyword}",
                who_crisis_template=None
            )

    # ── Pass 2: Semantic Similarity Match ──────────────────────────────────────
    # We only call the embedding model for Pass 2. This saves resources.
    try:
        query_vector = await embed_query(clean_query)
        
        # Check against Tier-1 exemplars
        max_t1_sim = 0.0
        matched_t1_ex = ""
        for exemplar in TIER1_EXEMPLARS:
            ex_vector = await embed_query(exemplar)  # In production, these are pre-cached, but embed_query caches
            sim = _cosine_similarity(query_vector, ex_vector)
            if sim > max_t1_sim:
                max_t1_sim = sim
                matched_t1_ex = exemplar
                
        if max_t1_sim >= 0.85:
            logger.warning(f"Crisis detected (Semantic Tier-1, similarity={max_t1_sim:.4f}): matched '{matched_t1_ex}'")
            return CrisisResult(
                level="TIER_1",
                matched_signal=f"Semantic T1 match (sim={max_t1_sim:.2f})",
                who_crisis_template=WHO_CRISIS_TEMPLATE_TIER1
            )
            
        # Check against Tier-2 exemplars
        max_t2_sim = 0.0
        matched_t2_ex = ""
        for exemplar in TIER2_EXEMPLARS:
            ex_vector = await embed_query(exemplar)
            sim = _cosine_similarity(query_vector, ex_vector)
            if sim > max_t2_sim:
                max_t2_sim = sim
                matched_t2_ex = exemplar
                
        if max_t2_sim >= 0.78:
            logger.info(f"Crisis detected (Semantic Tier-2, similarity={max_t2_sim:.4f}): matched '{matched_t2_ex}'")
            return CrisisResult(
                level="TIER_2",
                matched_signal=f"Semantic T2 match (sim={max_t2_sim:.2f})",
                who_crisis_template=None
            )
            
    except Exception as e:
        logger.error(f"Semantic crisis detection check failed: {e}. Falling back to Keyword results.")

    # No crisis signals detected
    return CrisisResult(level="NONE", matched_signal=None, who_crisis_template=None)
