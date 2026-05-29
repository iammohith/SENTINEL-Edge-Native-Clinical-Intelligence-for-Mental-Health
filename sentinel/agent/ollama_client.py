"""
SENTINEL — Resilient Ollama Async Client

Phase 5 implementation.
Wraps the Ollama AsyncClient with tenacity retries and a circuit breaker
to handle transient model service failures gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncGenerator

import ollama
from pybreaker import CircuitBreaker
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sentinel.config import OLLAMA_BASE_URL, LLM_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

# Bounded circuit breaker: trips after 3 consecutive failures, resets after 30s (Finding #27)
_llm_breaker = CircuitBreaker(fail_max=3, reset_timeout=30)


class ResilientOllamaClient:
    """
    Resilient wrapper around ollama.AsyncClient.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self._async_client = ollama.AsyncClient(host=base_url)

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, ollama.ResponseError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _chat_with_retry(self, *args: Any, **kwargs: Any) -> Any:
        """Asynchronous chat helper with retry decorators."""
        # Enforce timeout from config (Finding #34)
        kwargs.setdefault("timeout", LLM_TIMEOUT_SECONDS)
        return await self._async_client.chat(*args, **kwargs)

    async def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """
        Executes a chat completion call with circuit breaker and retry protections.
        """
        try:
            # CircuitBreaker.call is synchronous, but we can call it around an async function
            # by executing the coroutine. In Python, pybreaker supports async wrappers or we can
            # wrap the call manually:
            if not _llm_breaker.current_state == "closed" and _llm_breaker.current_state != "half-open":
                raise RuntimeError("Ollama LLM Circuit Breaker is TRIP / OPEN. Rejecting request.")
                
            # Execute within breaker
            # pybreaker tracks failures by catching exceptions raised within the call block
            try:
                response = await self._chat_with_retry(model=model, messages=messages, **kwargs)
                _llm_breaker.success()
                return response
            except Exception as e:
                _llm_breaker.handle_failure(e)
                raise e
        except Exception as e:
            logger.error(f"Ollama chat execution failed: {e}")
            raise e

    async def chat_stream(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        """
        Streams chat completions asynchronously (SSE support, Finding #22 / #55).
        """
        # Ensure we set stream=True
        kwargs["stream"] = True
        kwargs.setdefault("timeout", LLM_TIMEOUT_SECONDS)
        
        try:
            generator = await self._async_client.chat(model=model, messages=messages, **kwargs)
            async for chunk in generator:
                yield chunk
        except Exception as e:
            logger.error(f"Ollama streaming chat failed: {e}")
            raise e
