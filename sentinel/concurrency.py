"""
SENTINEL — Shared Concurrency Infrastructure (Finding #36 fix)

Defines a single bounded ThreadPoolExecutor shared across all CPU-bound
ML operations: FTS search, reranker predict(), and NLI predict().

Problem solved:
    hybrid.py, reranker.py, and faithfulness.py each referenced `_thread_pool`
    as a module-level variable but none defined it. Each would instantiate its
    own ThreadPoolExecutor, creating unbounded thread proliferation under load.
    (Finding #36 — v6 critical fix)

Solution:
    Import `_thread_pool` from this module. One bounded pool for all three.

Usage:
    from sentinel.concurrency import _thread_pool

    result = await loop.run_in_executor(_thread_pool, blocking_fn, *args)

Configuration:
    max_workers=4 balances CPU parallelism with memory constraint on
    edge hardware. On Apple Silicon M1 (8 cores), 4 workers saturate
    the efficiency cores while leaving performance cores for the GIL-
    holding main thread and the Ollama process.
"""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor

# ── Shared bounded executor ─────────────────────────────────────────────────────
# ALL CPU-bound ML calls must use this pool:
#   - sentinel.retrieval.hybrid: FTS search (BM25/Tantivy Python wrapper)
#   - sentinel.retrieval.reranker: CrossEncoder.predict() (reranker)
#   - sentinel.agent.faithfulness: CrossEncoder.predict() (NLI)
#
# NEVER use asyncio.to_thread() for CPU-bound work — it uses the default
# executor which is unbounded and creates unlimited threads under load.
# (Finding #52 — asyncio.to_thread vs run_in_executor inconsistency)
_thread_pool = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="sentinel-ml",
)

# Ensure clean shutdown — ThreadPoolExecutor must be shut down on process exit
# to avoid hanging daemon threads.
atexit.register(_thread_pool.shutdown, wait=False)
