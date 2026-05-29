"""
SENTINEL — Confidence Scorer

Phase 5 implementation.
Calculates a calibrated confidence score for clinical answers using a two-stage rubric:
1. Weighted average of base retrieval and intent signals.
2. Multiplicative gating based on NLI faithfulness and language support (Finding #26).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_confidence_score(
    intent_confidence: float,
    max_rerank_score: float,
    faithfulness_score: float,
    language_warning: bool,
    has_clinical_alerts: bool
) -> float:
    """
    Computes a calibrated confidence score clamped between [0.0, 1.0].
    
    Formula (Finding #26):
      Base Score = 0.40 * intent_confidence + 0.60 * normalized_rerank_score
      Gate Factor = faithfulness_score * (0.85 if language_warning else 1.0)
      Final Score = Base Score * Gate Factor
    """
    # 1. Normalize Reranker Score
    # ms-marco-MiniLM-L-6-v2 outputs logit-like scores generally between -12 and 10.
    # We map this to a [0.0, 1.0] range using a sigmoid-like or linear mapping:
    # A score of >= 2.0 is highly relevant; <= -5.0 is irrelevant.
    if max_rerank_score > 5.0:
        normalized_rerank = 1.0
    elif max_rerank_score < -15.0:
        normalized_rerank = 0.0
    else:
        # Linear map from [-15.0, 5.0] to [0.0, 1.0]
        normalized_rerank = (max_rerank_score + 15.0) / 20.0

    # 2. Stage 1: Weighted average of primary signals
    base_score = (0.40 * intent_confidence) + (0.60 * normalized_rerank)

    # 3. Stage 2: Multiplicative Gates
    # Faithfulness is the primary gate (NLI entailment ratio)
    gate_factor = faithfulness_score
    
    # Apply penalty for non-supported languages (Finding #53)
    if language_warning:
        gate_factor *= 0.80

    # Apply penalty if clinical alerts were present but faithfulness was not perfect
    # (clinical warnings require high fidelity response)
    if has_clinical_alerts and faithfulness_score < 1.0:
        gate_factor *= 0.90

    # 4. Final score
    final_score = base_score * gate_factor
    
    # Clamp to [0.0, 1.0]
    clamped_score = max(0.0, min(1.0, final_score))
    
    logger.info(
        f"Confidence calibration: Base={base_score:.2f} | "
        f"Gate={gate_factor:.2f} | Final={clamped_score:.2f} "
        f"(Rerank={max_rerank_score:.2f}, Faithfulness={faithfulness_score:.2f})"
    )
    
    return clamped_score
