"""
SENTINEL — Condition-Aware Clinical Router

Phase 5 implementation.
Routes query, intent type, and condition codes to the appropriate tools.
Handles comorbidity and multi-condition queries by merging and reranking (Finding #7 / #42).
"""

from __future__ import annotations

import logging
from typing import Any

from sentinel.agent.tools import retrieve_clinical_guidelines
from sentinel.config import MHIntentType
from sentinel.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


async def route_and_retrieve(
    query: str,
    intent: MHIntentType,
    condition_codes: list[str],
    top_n: int = 5
) -> list[dict[str, Any]]:
    """
    Routes the query to the correct retrieval strategy.
    If multiple condition codes are present (comorbidity/cross-cutting concerns),
    it retrieves candidate chunks for each condition in parallel, merges them,
    and applies a final reranking pass to find the top-n most relevant guidelines.
    """
    logger.info(f"Router: Routing query for intent '{intent.value}' and codes: {condition_codes}")

    # 1. Handle Out of Scope queries immediately
    if intent == MHIntentType.OUT_OF_SCOPE:
        logger.warning("Query classified as OUT_OF_SCOPE. Skipping retrieval.")
        return []

    # 2. Retrieve candidates for each condition code
    if not condition_codes:
        # Default fallback to unconditioned retrieval
        return await retrieve_clinical_guidelines(query, condition_code=None, top_n=top_n)

    # If only one condition, retrieve directly
    if len(condition_codes) == 1:
        return await retrieve_clinical_guidelines(query, condition_code=condition_codes[0], top_n=top_n)

    # If multiple conditions (e.g., Depression + Self-Harm), retrieve candidates
    # from all relevant partitions and merge them.
    logger.info(f"Router: Multi-condition query detected. Fetching partitions for: {condition_codes}")
    
    tasks = [
        retrieve_clinical_guidelines(query, condition_code=code, top_n=10)  # Fetch wider set for merging
        for code in condition_codes
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Merge and deduplicate by chunk_id
    merged_chunks: dict[str, dict[str, Any]] = {}
    for chunk_list in results:
        for chunk in chunk_list:
            cid = chunk["chunk_id"]
            merged_chunks[cid] = chunk
            
    if not merged_chunks:
        return []

    # Apply a final reranking pass over the combined set of chunks
    logger.info(f"Router: Merged {len(merged_chunks)} candidate chunks from multiple partitions. Applying final rerank...")
    final_chunks = await rerank(
        query=query,
        chunks=list(merged_chunks.values()),
        top_n=top_n
    )
    
    return final_chunks

import asyncio  # Ensure imported in module namespace
