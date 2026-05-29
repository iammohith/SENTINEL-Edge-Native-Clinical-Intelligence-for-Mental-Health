"""
SENTINEL — MiniLM Cross-Encoder Reranker

Phase 3 implementation.
Reranks hybrid search results using the cross-encoder/ms-marco-MiniLM-L-6-v2 model.
Executes within the singleton ML ThreadPoolExecutor to prevent blocking.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

from sentence_transformers import CrossEncoder

from sentinel.concurrency import _thread_pool
from sentinel.config import RERANK_TOP_N

logger = logging.getLogger(__name__)

# Singleton CrossEncoder model instance loaded on demand / first query
_reranker_model: Optional[CrossEncoder] = None
_model_lock = threading.Lock()


def _get_reranker() -> CrossEncoder:
    """Thread-safe lazy initializer for the reranker model."""
    global _reranker_model
    if _reranker_model is None:
        with _model_lock:
            if _reranker_model is None:
                logger.info("Loading CrossEncoder reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
                # We default to using MPS/CUDA if available, or CPU
                import torch
                device = "cpu"
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                _reranker_model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device=device
                )
                logger.info(f"Reranker model loaded on device '{device}'.")
    return _reranker_model


def _predict_pairs(pairs: list[tuple[str, str]]) -> list[float]:
    """Helper to call prediction from thread pool."""
    model = _get_reranker()
    scores = model.predict(pairs)
    # Ensure scores are a list of floats (convert from numpy/float arrays if needed)
    return [float(s) for s in scores]


async def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int = RERANK_TOP_N
) -> list[dict[str, Any]]:
    """
    Reranks candidate chunks based on the query.
    
    Args:
        query: Raw clinical query text.
        chunks: List of chunk dictionaries returned by hybrid retrieval.
        top_n: Number of final chunks to return.
        
    Returns:
        Top-n reranked chunks sorted by cross-encoder score descending.
    """
    if not chunks:
        return []

    # Prepare inputs: list of (query, passage) pairs
    pairs = [(query, chunk["content"]) for chunk in chunks]
    
    logger.debug(f"Submitting {len(pairs)} pairs to cross-encoder reranker...")
    
    # Run the prediction in the shared thread pool
    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(
        _thread_pool,
        _predict_pairs,
        pairs
    )

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = score

    # Sort descending by score
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    logger.info(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.4f}" if reranked else "Reranking returned 0 results.")
    
    return reranked[:top_n]
