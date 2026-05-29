"""
SENTINEL — Clinical Agent Tools

Phase 5 implementation.
Defines retrieval and search tools used by the clinical agent router to fetch
evidence from the LanceDB single store.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from sentinel.ingestion.embedder import embed_query
from sentinel.retrieval.hybrid import hybrid_retrieve
from sentinel.retrieval.reranker import rerank
from sentinel.store.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def retrieve_clinical_guidelines(
    query: str,
    condition_code: Optional[str] = None,
    top_n: int = 5
) -> list[dict[str, Any]]:
    """
    Retrieves and reranks relevant WHO mhGAP chunks for a given query and condition code.
    
    Args:
        query: Raw clinical query string.
        condition_code: Optional condition code to filter retrieval (e.g., 'DEP', 'SHI').
        top_n: Number of final high-quality chunks to return.
        
    Returns:
        List of top-n chunks sorted by relevance.
    """
    logger.info(f"Retrieval Tool: Fetching guidelines for condition '{condition_code}'...")
    
    # 1. Embed query asynchronously
    try:
        embedding = await embed_query(query)
    except Exception as e:
        logger.error(f"Failed to embed query for tool retrieval: {e}")
        return []
        
    # 2. Get VectorStore instance and table
    try:
        store = VectorStore.get_instance()
        table = store._table
    except Exception as e:
        logger.error(f"Failed to access VectorStore: {e}")
        return []
        
    # 3. Perform hybrid FTS + ANN retrieval
    try:
        candidates = await hybrid_retrieve(
            query=query,
            embedding=embedding,
            table=table,
            condition_code=condition_code
        )
    except Exception as e:
        logger.error(f"Hybrid retrieval failed: {e}")
        return []
        
    if not candidates:
        logger.warning("No candidate chunks retrieved.")
        return []
        
    # 4. Rerank candidates using CrossEncoder MiniLM
    try:
        final_chunks = await rerank(
            query=query,
            chunks=candidates,
            top_n=top_n
        )
        return final_chunks
    except Exception as e:
        logger.error(f"Reranking failed: {e}. Returning raw candidates.")
        return candidates[:top_n]
