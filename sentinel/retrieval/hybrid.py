"""
SENTINEL — Hybrid Retrieval: LanceDB FTS + LanceDB ANN → RRF Fusion

Phase 3 implementation.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Any

from sentinel.concurrency import _thread_pool
from sentinel.config import RETRIEVE_TOP_K, RRF_K_CONSTANT

logger = logging.getLogger(__name__)


def _rrf_merge(
    fts_results: list[dict[str, Any]],
    ann_results: list[dict[str, Any]],
    k: int = RRF_K_CONSTANT,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion of FTS and ANN result lists.
    Fuses rankings without score normalization.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, dict[str, Any]] = {}

    for rank, chunk in enumerate(fts_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunks[cid] = chunk

    for rank, chunk in enumerate(ann_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
        chunks[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [chunks[cid] for cid, _ in ranked[:top_n]]


async def hybrid_retrieve(
    query: str,
    embedding: list[float],
    table: Any,  # lancedb Table
    condition_code: Optional[str] = None,
    top_k: int = RETRIEVE_TOP_K,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: LanceDB FTS (BM25) ‖ LanceDB ANN -> RRF -> top-20.
    Concurrently executes FTS and ANN searches within the singleton ThreadPoolExecutor
    to protect the event loop. Filters out superseded records before RRF (Finding #35 / #52).
    """
    loop = asyncio.get_running_loop()

    # Build WHERE clause - FTS and ANN filter superseded chunks BEFORE RRF (Finding #35)
    base_filter = "superseded = false"
    filter_clause = (
        f"{base_filter} AND condition_code = '{condition_code}'"
        if condition_code
        else base_filter
    )

    # ── FTS search (BM25 via LanceDB Tantivy) ──────────────────────────────────
    # CPU-bound: Python overhead on Tantivy. Use shared executor.
    def _fts() -> list[dict[str, Any]]:
        try:
            return table.search(query).where(filter_clause).limit(top_k).to_list()
        except Exception as e:
            logger.error(f"FTS search failed: {e}")
            return []

    # ── ANN search (IVF-HNSW-SQ vector search) ─────────────────────────────────
    # CPU-bound wrapper. Use shared executor.
    def _ann() -> list[dict[str, Any]]:
        try:
            return table.search(embedding).where(filter_clause).limit(top_k).to_list()
        except Exception as e:
            logger.error(f"ANN search failed: {e}")
            return []

    # Run both branches concurrently in the ML thread pool
    fts_results, ann_results = await asyncio.gather(
        loop.run_in_executor(_thread_pool, _fts),
        loop.run_in_executor(_thread_pool, _ann),
    )

    logger.debug(f"Retrieved {len(fts_results)} FTS and {len(ann_results)} ANN candidate chunks.")
    return _rrf_merge(fts_results, ann_results)
