"""
SENTINEL — Ingestion Postprocessor

Normalizes heading hierarchies, extracts section paths, tags condition codes,
detects clinical warning/caution alerts, formats tables as markdown,
and partitions text into semantically cohesive, bounded-size chunks.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from docling_core.types.doc import (
    DoclingDocument,
    SectionHeaderItem,
    ListItem,
    TableItem,
    TextItem,
    TitleItem,
)
from sentinel.config import MNS_CONDITIONS, MAX_CHUNK_TOKENS

logger = logging.getLogger(__name__)


def _determine_condition_code(header_stack: list[str]) -> str:
    """
    Determines the mhGAP condition code by matching section headers
    against condition keywords. Falls back to 'GEN' (General) or 'OTH'.
    """
    # Check for direct condition codes in headers first (e.g., "DEP", "PSY")
    for code in MNS_CONDITIONS:
        for header in header_stack:
            h_parts = [p.strip().upper() for p in header.replace("/", " ").replace("-", " ").split()]
            if code in h_parts:
                return code

    path_str = " > ".join(header_stack).lower()
    
    # Check specific conditions in order of precision
    if "self-harm" in path_str or "suicide" in path_str:
        return "SHI"
    if "depression" in path_str or "depressive" in path_str:
        return "DEP"
    if "psychosis" in path_str or "psychotic" in path_str or "schizophrenia" in path_str:
        return "PSY"
    if "substance" in path_str or "alcohol" in path_str or "drug" in path_str or "withdrawal" in path_str:
        return "SUD"
    if "epilepsy" in path_str or "seizure" in path_str or "convulsion" in path_str:
        return "EPI"
    if "dementia" in path_str or "cognitive decline" in path_str or "alzheimer" in path_str:
        return "DEM"
    if "child" in path_str or "adolescent" in path_str or "developmental" in path_str or "behavioral" in path_str or "behavioural" in path_str:
        return "DLD"
    if "other significant" in path_str or "somatic" in path_str or "medically unexplained" in path_str:
        return "OTH"
    if "general principles" in path_str or "mhgap principles" in path_str or "essential care" in path_str or "communication" in path_str:
        return "GEN"
        
    return "GEN"  # Default to general principles for unclassified text


def _is_clinical_alert(text: str) -> bool:
    """
    Detects if the text block contains a critical WHO clinical warning/caution alert.
    WHO mhGAP uses explicit visual callouts like 'Note:', 'Caution:', or 'Do NOT'.
    """
    lower_text = text.lower()
    # Check for starting patterns or substring matches
    alert_patterns = [
        "note:",
        "caution:",
        "warning:",
        "do not",
        "contraindication",
        "contraindicated",
        "emergency referral",
        "urgent referral",
        "immediate action"
    ]
    return any(pattern in lower_text for pattern in alert_patterns)


def _format_table_to_markdown(item: TableItem) -> str:
    """
    Formats a TableItem into a clean markdown table.
    Ensures table structure is searchable and readable.
    """
    data = item.data
    if not data or not data.table_cells:
        return ""
        
    num_rows = data.num_rows
    num_cols = data.num_cols
    
    # Initialize empty grid
    grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]
    
    for cell in data.table_cells:
        r = cell.start_row_offset_idx
        c = cell.start_col_offset_idx
        if r < num_rows and c < num_cols:
            grid[r][c] = cell.text or ""
            
    lines = []
    for r, row in enumerate(grid):
        clean_row = [c.replace("\n", " ").strip() for c in row]
        lines.append("| " + " | ".join(clean_row) + " |")
        if r == 0:
            lines.append("| " + " | ".join(["---" for _ in range(num_cols)]) + " |")
            
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """
    Simple word-to-token ratio estimation (1 word = 1.3 tokens).
    Avoids requiring a tokenizer in the CPU-bound postprocessor loop.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


def postprocess_document(
    doc_dict: dict[str, Any],
    source_doc: str,
    doc_version: str = "2.0-2016",
    effective_date: str = "2016-10-01",
    superseded: bool = False
) -> list[dict[str, Any]]:
    """
    Processes a raw DoclingDocument dictionary, normalizes section paths,
    extracts metadata, splits texts into token-limited chunks, and returns
    a list of LanceDB-compatible chunk records.
    """
    doc = DoclingDocument.model_validate(doc_dict)
    
    chunks: list[dict[str, Any]] = []
    header_stack: list[str] = []
    current_text_buffer: list[str] = []
    current_buffer_tokens = 0
    
    # Iterate through all items in reading order
    # iterate_items yields (node, level) where node contains self_ref pointing to the actual item.
    for node, level in doc.iterate_items():
        # Get the actual item from the document using the self_ref
        item_ref = node.self_ref
        if not item_ref:
            continue
            
        # Resolve item from texts or tables based on self_ref path
        parts = item_ref.lstrip("#/").split("/")
        if len(parts) < 2:
            continue
            
        collection_name = parts[0]
        item_idx = int(parts[1])
        
        item: Any = None
        if collection_name == "texts" and item_idx < len(doc.texts):
            item = doc.texts[item_idx]
        elif collection_name == "tables" and item_idx < len(doc.tables):
            item = doc.tables[item_idx]
        else:
            continue

        # Handle header updates to track the hierarchy
        if isinstance(item, SectionHeaderItem):
            # level is 1-indexed, adjust header stack size
            h_level = getattr(item, "level", 1)
            # Trim stack to match current level (e.g. if level=2, stack should have at most 1 item)
            header_stack = header_stack[:h_level - 1]
            header_stack.append(item.text.strip())
            continue
        elif isinstance(item, TitleItem):
            header_stack = [item.text.strip()]
            continue

        # Compute current metadata
        section_path = " > ".join(header_stack) if header_stack else "Root"
        condition_code = _determine_condition_code(header_stack)
        page_no = 1
        if item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no

        # Handle Tables
        if isinstance(item, TableItem):
            # Flush any existing text buffer first
            if current_text_buffer:
                buffer_content = "\n".join(current_text_buffer)
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_doc": source_doc,
                    "doc_version": doc_version,
                    "effective_date": effective_date,
                    "superseded": superseded,
                    "condition_code": condition_code,
                    "section_path": section_path,
                    "content": buffer_content,
                    "chunk_type": "clinical_alert" if _is_clinical_alert(buffer_content) else "procedure",
                    "adjacent_clinical_alerts": "",
                    "page_no": page_no
                })
                current_text_buffer = []
                current_buffer_tokens = 0
                
            table_markdown = _format_table_to_markdown(item)
            if table_markdown:
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_doc": source_doc,
                    "doc_version": doc_version,
                    "effective_date": effective_date,
                    "superseded": superseded,
                    "condition_code": condition_code,
                    "section_path": section_path,
                    "content": table_markdown,
                    "chunk_type": "table",
                    "adjacent_clinical_alerts": "",
                    "page_no": page_no
                })
            continue

        # Handle Texts (Paragraphs and List Items)
        if isinstance(item, (TextItem, ListItem)):
            text = item.text.strip()
            if not text:
                continue
                
            # If it's a standalone clinical alert, flush buffer and output immediately
            if _is_clinical_alert(text):
                # Flush existing buffer first
                if current_text_buffer:
                    buffer_content = "\n".join(current_text_buffer)
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "source_doc": source_doc,
                        "doc_version": doc_version,
                        "effective_date": effective_date,
                        "superseded": superseded,
                        "condition_code": condition_code,
                        "section_path": section_path,
                        "content": buffer_content,
                        "chunk_type": "procedure",
                        "adjacent_clinical_alerts": "",
                        "page_no": page_no
                    })
                    current_text_buffer = []
                    current_buffer_tokens = 0
                
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_doc": source_doc,
                    "doc_version": doc_version,
                    "effective_date": effective_date,
                    "superseded": superseded,
                    "condition_code": condition_code,
                    "section_path": section_path,
                    "content": text,
                    "chunk_type": "clinical_alert",
                    "adjacent_clinical_alerts": "",
                    "page_no": page_no
                })
                continue

            # Standard text chunking with token limits
            item_tokens = _estimate_tokens(text)
            # If adding this item exceeds the limit, flush the buffer
            if current_buffer_tokens + item_tokens > MAX_CHUNK_TOKENS:
                buffer_content = "\n".join(current_text_buffer)
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "source_doc": source_doc,
                    "doc_version": doc_version,
                    "effective_date": effective_date,
                    "superseded": superseded,
                    "condition_code": condition_code,
                    "section_path": section_path,
                    "content": buffer_content,
                    "chunk_type": "procedure",
                    "adjacent_clinical_alerts": "",
                    "page_no": page_no
                })
                current_text_buffer = [text]
                current_buffer_tokens = item_tokens
            else:
                current_text_buffer.append(text)
                current_buffer_tokens += item_tokens

    # Flush any remaining items in the buffer
    if current_text_buffer:
        buffer_content = "\n".join(current_text_buffer)
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "source_doc": source_doc,
            "doc_version": doc_version,
            "effective_date": effective_date,
            "superseded": superseded,
            "condition_code": _determine_condition_code(header_stack),
            "section_path": " > ".join(header_stack) if header_stack else "Root",
            "content": buffer_content,
            "chunk_type": "procedure",
            "adjacent_clinical_alerts": "",
            "page_no": page_no
        })

    # Associate adjacent clinical alerts
    # If a chunk is adjacent to a clinical_alert chunk within the same section_path,
    # link them so the alert is retrieved along with the context.
    for i, chunk in enumerate(chunks):
        if chunk["chunk_type"] == "procedure":
            # Search nearby chunks (window of 2 before and 2 after) for clinical alerts
            alerts = []
            start_win = max(0, i - 2)
            end_win = min(len(chunks), i + 3)
            for j in range(start_win, end_win):
                if j != i and chunks[j]["chunk_type"] == "clinical_alert" and chunks[j]["section_path"] == chunk["section_path"]:
                    alerts.append(chunks[j]["content"])
            if alerts:
                chunk["adjacent_clinical_alerts"] = "\n".join(alerts)

    return chunks
