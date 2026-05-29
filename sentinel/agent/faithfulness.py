"""
SENTINEL — NLI Faithfulness Validator

Phase 5 implementation.
Checks the faithfulness (groundedness) of generated response sentences
against the retrieved context chunks using a Natural Language Inference (NLI) cross-encoder.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Literal, Optional, Any

from sentence_transformers import CrossEncoder

from sentinel.concurrency import _thread_pool

logger = logging.getLogger(__name__)

# Thresholds for NLI classification
# MiniLM NLI output labels correspond to: [CONTRADICTION, NEUTRAL, ENTAILMENT]
FAITHFULNESS_THRESHOLD = 0.70  # Mean entailment ratio required to pass
CONTRADICTION_CONFIDENCE = 0.60  # Minimum contradiction label score to trigger block

_nli_model: Optional[CrossEncoder] = None
_nli_lock = threading.Lock()


@dataclass
class SentenceVerdict:
    sentence: str
    label: Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]
    max_entailment_score: float
    supporting_chunk_id: Optional[str]


@dataclass
class FaithfulnessResult:
    score: float                            # Entailment ratio: [0.0, 1.0]
    is_faithful: bool                       # score >= FAITHFULNESS_THRESHOLD
    blocked: bool                           # True if any sentence is CONTRADICTION
    sentence_results: list[SentenceVerdict]
    contradicted_sentences: list[str]       # Non-empty if blocked=True


def _get_nli_model() -> CrossEncoder:
    """Thread-safe lazy initializer for the NLI cross-encoder."""
    global _nli_model
    if _nli_model is None:
        import threading
        with _nli_lock:
            if _nli_model is None:
                logger.info("Loading NLI cross-encoder model (cross-encoder/nli-MiniLM2-L6-H768)...")
                import torch
                device = "cpu"
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                _nli_model = CrossEncoder(
                    "cross-encoder/nli-MiniLM2-L6-H768",
                    device=device
                )
                logger.info(f"NLI model loaded on device '{device}'.")
    return _nli_model


def _predict_nli_pairs(pairs: list[tuple[str, str]]) -> Any:
    """Invokes the model predict function. Runs on CPU/MPS/GPU."""
    model = _get_nli_model()
    # model.predict returns a numpy array of shape (N, 3)
    # columns are: [contradiction, neutral, entailment]
    return model.predict(pairs)


async def check_faithfulness(
    answer_sentences: list[str],
    context_chunks: list[dict[str, Any]],
) -> FaithfulnessResult:
    """
    Validates that every sentence in the answer is supported by (entailed by)
    at least one context chunk, and not contradicted by any.
    
    Pairs are structured as (PREMISE, HYPOTHESIS) -> (chunk, sentence) (Finding #25).
    """
    import threading  # Ensure imported
    
    if not answer_sentences:
        return FaithfulnessResult(
            score=1.0,
            is_faithful=True,
            blocked=False,
            sentence_results=[],
            contradicted_sentences=[]
        )

    if not context_chunks:
        # No context means nothing can be entailed!
        return FaithfulnessResult(
            score=0.0,
            is_faithful=False,
            blocked=False,
            sentence_results=[
                SentenceVerdict(s, "NEUTRAL", 0.0, None)
                for s in answer_sentences
            ],
            contradicted_sentences=[]
        )

    # 1. Prepare pairs: every sentence against every chunk
    # Shape: N_sentences * M_chunks
    pairs = [
        (chunk["content"], sentence)
        for sentence in answer_sentences
        for chunk in context_chunks
    ]

    logger.debug(f"Faithfulness NLI: Evaluating {len(pairs)} pairs...")
    
    # 2. Run prediction in shared thread pool
    loop = asyncio.get_running_loop()
    raw_scores = await loop.run_in_executor(
        _thread_pool,
        _predict_nli_pairs,
        pairs
    )

    # 3. Analyze results per sentence
    n_chunks = len(context_chunks)
    sentence_results: list[SentenceVerdict] = []
    contradicted_sentences: list[str] = []
    
    entailed_count = 0

    for i, sentence in enumerate(answer_sentences):
        # Extract the slice of scores corresponding to this sentence
        sentence_scores = raw_scores[i * n_chunks : (i + 1) * n_chunks]
        
        # Track maximum entailment and contradiction signals across all chunks
        max_entail_score = -1.0
        best_chunk_id = None
        
        is_contradicted = False
        
        for idx, chunk_score in enumerate(sentence_scores):
            # labels: [contradiction, neutral, entailment]
            contradiction_prob = float(chunk_score[0])
            neutral_prob = float(chunk_score[1])
            entailment_prob = float(chunk_score[2])
            
            chunk_id = context_chunks[idx]["chunk_id"]
            
            # Check for contradiction (Finding #19 / #37)
            if contradiction_prob > CONTRADICTION_CONFIDENCE:
                is_contradicted = True
                
            if entailment_prob > max_entail_score:
                max_entail_score = entailment_prob
                best_chunk_id = chunk_id

        # Determine verdict label for the sentence
        if is_contradicted:
            verdict_label = "CONTRADICTION"
            contradicted_sentences.append(sentence)
        elif max_entail_score >= 0.50:  # Threshold for sentence-level entailment
            verdict_label = "ENTAILMENT"
            entailed_count += 1
        else:
            verdict_label = "NEUTRAL"

        sentence_results.append(
            SentenceVerdict(
                sentence=sentence,
                label=verdict_label,
                max_entailment_score=max_entail_score,
                supporting_chunk_id=best_chunk_id if verdict_label == "ENTAILMENT" else None
            )
        )

    # 4. Calculate total score
    score = entailed_count / len(answer_sentences)
    is_faithful = score >= FAITHFULNESS_THRESHOLD
    blocked = len(contradicted_sentences) > 0

    logger.info(f"Faithfulness score: {score:.2f} | Faithful: {is_faithful} | Contradictions: {len(contradicted_sentences)}")
    
    return FaithfulnessResult(
        score=score,
        is_faithful=is_faithful,
        blocked=blocked,
        sentence_results=sentence_results,
        contradicted_sentences=contradicted_sentences
    )
