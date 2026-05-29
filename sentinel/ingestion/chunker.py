"""
SENTINEL — Ingestion Chunker Coordinator

Coordinates standard document postprocessing and decision tree extraction.
"""

from __future__ import annotations

import logging
from typing import Any

from sentinel.ingestion.postprocessor import postprocess_document
from sentinel.ingestion.decision_tree import extract_decision_branches

logger = logging.getLogger(__name__)


def chunk_document(
    doc_dict: dict[str, Any],
    source_doc: str,
    doc_version: str = "2.0-2016",
    effective_date: str = "2016-10-01",
    superseded: bool = False
) -> list[dict[str, Any]]:
    """
    Main chunking coordinator. Combines standard hierarchical chunks and
    logical decision tree branch chunks into a single unified list of chunks.
    """
    logger.info(f"Chunking document '{source_doc}'...")
    
    # 1. Extract standard document chunks
    standard_chunks = postprocess_document(
        doc_dict=doc_dict,
        source_doc=source_doc,
        doc_version=doc_version,
        effective_date=effective_date,
        superseded=superseded
    )
    logger.info(f"Extracted {len(standard_chunks)} standard procedure/alert/table chunks.")
    
    # 2. Extract decision tree flowchart branches
    decision_chunks = extract_decision_branches(
        doc_dict=doc_dict,
        source_doc=source_doc,
        doc_version=doc_version,
        effective_date=effective_date,
        superseded=superseded
    )
    logger.info(f"Extracted {len(decision_chunks)} decision flowchart branches.")
    
    # Combine both collections
    all_chunks = standard_chunks + decision_chunks
    logger.info(f"Total chunks extracted: {len(all_chunks)}")
    
    return all_chunks
