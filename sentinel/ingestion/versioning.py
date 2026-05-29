"""
SENTINEL — Ingestion Document Versioning and Supersession Tracker

Enforces document versioning and manages the transition when newer technical
guidelines supersede older ones (Finding #10).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_version_metadata(file_name: str) -> dict[str, Any]:
    """
    Parses version and date metadata from document filename.
    Expected format: "mhgap_ig_v2.0_2016.pdf" or "mhgap_ig_2010.pdf"
    
    Returns:
        A dict with version, effective_date, and document_type.
    """
    fn_lower = file_name.lower()
    
    # Defaults
    version = "1.0"
    effective_date = "2010-01-01"
    doc_type = "guideline"
    
    if "v2.0" in fn_lower or "2016" in fn_lower:
        version = "2.0-2016"
        effective_date = "2016-10-01"
    elif "humanitarian" in fn_lower or "hig" in fn_lower:
        version = "2.0-hig-2015"
        effective_date = "2015-05-01"
        doc_type = "humanitarian_guideline"
    elif "training" in fn_lower:
        version = "1.0-training-2012"
        effective_date = "2012-06-01"
        doc_type = "training_manual"
        
    return {
        "doc_version": version,
        "effective_date": effective_date,
        "document_type": doc_type
    }


def resolve_supersession_logic(
    incoming_meta: dict[str, Any],
    existing_records: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    """
    Checks if the incoming document supersedes existing records or is itself
    superseded by existing records.
    
    Args:
        incoming_meta: Version metadata of the document being ingested.
        existing_records: List of existing document version metadata in the store.
        
    Returns:
        A tuple of:
          - is_incoming_superseded (bool): True if the new doc is older than existing ones.
          - list of doc_names_to_supersede (list[str]): Names of existing documents that the new doc supersedes.
    """
    is_incoming_superseded = False
    docs_to_supersede = []
    
    incoming_ver = incoming_meta["doc_version"]
    incoming_type = incoming_meta["document_type"]
    incoming_date = incoming_meta["effective_date"]
    
    for record in existing_records:
        rec_ver = record["doc_version"]
        rec_type = record["document_type"]
        rec_date = record["effective_date"]
        rec_source = record["source_doc"]
        
        # Only compare documents of the same functional category
        if rec_type != incoming_type:
            continue
            
        # Compare effective dates
        if incoming_date < rec_date:
            logger.warning(
                f"Ingestion warning: Incoming document version {incoming_ver} (date: {incoming_date}) "
                f"is older than existing version {rec_ver} (date: {rec_date}) from source {rec_source}. "
                "The incoming document will be marked as superseded=True."
            )
            is_incoming_superseded = True
        elif incoming_date > rec_date:
            logger.info(
                f"Ingestion update: Incoming document version {incoming_ver} (date: {incoming_date}) "
                f"supersedes existing version {rec_ver} (date: {rec_date}) from source {rec_source}."
            )
            if rec_source not in docs_to_supersede:
                docs_to_supersede.append(rec_source)
                
    return is_incoming_superseded, docs_to_supersede
