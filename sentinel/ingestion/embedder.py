"""
SENTINEL — Resilient Ollama Embedder (nomic-embed-text)

Enforces resilience rules (tenacity retries, circuit breakers) for embedding operations.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import ollama
from pybreaker import CircuitBreaker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sentinel.config import EMBED_MODEL, OLLAMA_BASE_URL, LLM_TIMEOUT_SECONDS
from sentinel.concurrency import _thread_pool

logger = logging.getLogger(__name__)

# Bounded circuit breaker: trips after 3 consecutive failures, resets after 30s
_embed_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)


class ResilientEmbedder:
    """
    Resilient embedding service using nomic-embed-text via local Ollama.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self._client = ollama.Client(host=base_url)

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ollama.ResponseError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _embed_sync_with_retry(self, text: str) -> list[float]:
        """Wrapper for retry logic around synchronous Ollama embeddings call."""
        # Ensure the prompt format matches nomic-embed-text's requirement:
        # Search queries prefix: "search_query: "
        # Document chunks prefix: "search_document: "
        # We prepend search_document: by default for ingestion.
        formatted_text = f"search_document: {text}"
        
        response = self._client.embeddings(
            model=EMBED_MODEL,
            prompt=formatted_text,
            options={"temperature": 0.0}
        )
        return response["embedding"]

    async def embed_chunk(self, text: str) -> list[float]:
        """
        Embeds a single chunk of text asynchronously.
        Uses shared ThreadPoolExecutor to prevent blocking the event loop (Finding #28).
        """
        loop = asyncio.get_running_loop()
        
        # Wrapped in a circuit breaker
        try:
            return await loop.run_in_executor(
                _thread_pool,
                lambda: _embed_breaker.call(self._embed_sync_with_retry, text)
            )
        except Exception as e:
            logger.error(f"Embedding failed for text snippet: {text[:50]}... Error: {e}")
            raise e

    async def embed_chunks_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embeds a list of chunks in parallel using asyncio.gather.
        """
        tasks = [self.embed_chunk(text) for text in texts]
        return await asyncio.gather(*tasks)


# Resilient query embedder
_query_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)

@retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, ollama.ResponseError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True
)
def _embed_query_sync(client: ollama.Client, query: str) -> list[float]:
    # nomic-embed-text expects search_query: prefix for search queries
    formatted_query = f"search_query: {query}"
    response = client.embeddings(
        model=EMBED_MODEL,
        prompt=formatted_query,
        options={"temperature": 0.0}
    )
    return response["embedding"]


async def embed_query(query: str) -> list[float]:
    """
    Embeds a search query asynchronously using search_query: prefix.
    """
    client = ollama.Client(host=OLLAMA_BASE_URL)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _thread_pool,
        lambda: _query_breaker.call(_embed_query_sync, client, query)
    )
