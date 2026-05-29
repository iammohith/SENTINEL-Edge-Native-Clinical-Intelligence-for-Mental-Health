"""
SENTINEL — mhGAP Decision Tree Extractor

WHO mhGAP flowcharts represent critical clinical protocols (e.g., Depression
Assessment: Does patient have 2+ depressive symptoms? -> YES -> Does it last
for >= 2 weeks? -> YES -> Initiate Depression protocol).
This module reconstructs these decision branches into atomic chunks,
preventing guidelines from being severed across chunk boundaries.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Literal, Optional
from dataclasses import dataclass, asdict

from docling_core.types.doc import DoclingDocument, TextItem, ListItem

logger = logging.getLogger(__name__)


@dataclass
class DecisionNode:
    """
    A node in the reconstructed clinical protocol.
    """
    node_id: str
    node_type: Literal["START", "DECISION", "ACTION", "TERMINAL"]
    content: str
    yes_branch: Optional[str] = None   # node_id of YES path
    no_branch: Optional[str] = None    # node_id of NO path
    condition_code: str = "GEN"


def extract_decision_branches(
    doc_dict: dict[str, Any],
    source_doc: str,
    doc_version: str = "2.0-2016",
    effective_date: str = "2016-10-01",
    superseded: bool = False
) -> list[dict[str, Any]]:
    """
    Reconstructs decision tree paths from a document by scanning reading-order
    items for decision points, branching conditions (Yes/No), and actions.
    Groups paths into atomic chunks of type 'decision_branch'.
    """
    doc = DoclingDocument.model_validate(doc_dict)
    
    # 1. First Pass: Collect all candidate nodes in reading order
    raw_nodes = []
    header_stack: list[str] = []
    
    for node, level in doc.iterate_items():
        item_ref = node.self_ref
        if not item_ref:
            continue
        parts = item_ref.lstrip("#/").split("/")
        if len(parts) < 2 or parts[0] != "texts":
            continue
            
        item_idx = int(parts[1])
        if item_idx >= len(doc.texts):
            continue
            
        item = doc.texts[item_idx]
        if not isinstance(item, (TextItem, ListItem)):
            continue
            
        text = item.text.strip()
        if not text:
            continue
            
        # Update section headers (informational)
        if text.isupper() and len(text) < 40:
            header_stack = header_stack[:level - 1]
            header_stack.append(text)
            
        page_no = 1
        if item.prov and len(item.prov) > 0:
            page_no = item.prov[0].page_no

        raw_nodes.append({
            "text": text,
            "page_no": page_no,
            "header_stack": list(header_stack)
        })

    # 2. Second Pass: Reconstruct logical branches
    chunks: list[dict[str, Any]] = []
    
    i = 0
    while i < len(raw_nodes):
        node = raw_nodes[i]
        text = node["text"]
        
        # Check if this node marks a clinical decision point (e.g. ends with ? or contains specific conditions)
        is_decision = (
            text.endswith("?") or 
            text.startswith("Does the person") or 
            text.startswith("Is there a") or
            text.startswith("Has the person") or
            "assessment" in text.lower() and "?" in text
        )
        
        if is_decision:
            # We found a decision point! Group it with subsequent Yes/No branches and actions
            branch_text = [f"DECISION: {text}"]
            condition_code = "GEN"
            path_str = "Root"
            if node["header_stack"]:
                path_str = " > ".join(node["header_stack"])
                # Import helper logic from postprocessor to keep tagging consistent
                from sentinel.ingestion.postprocessor import _determine_condition_code
                condition_code = _determine_condition_code(node["header_stack"])
                
            # Scan ahead to group related criteria, branches, and outcomes
            j = i + 1
            branch_depth = 0
            while j < len(raw_nodes) and j < i + 10:  # Look ahead up to 10 nodes max
                next_node = raw_nodes[j]
                next_text = next_node["text"]
                
                # If we encounter another high-level section or a new clear decision, stop
                if next_text.isupper() and len(next_text) < 30:
                    break
                    
                # Format branching paths explicitly
                if next_text.lower() in ("yes", "no", "if yes", "if no", "yes ->", "no ->"):
                    branch_text.append(f"\nBRANCH: {next_text}")
                elif next_text.startswith("•") or next_text.startswith("-"):
                    branch_text.append(f"  {next_text}")
                else:
                    # Imperative verbs represent actions/protocols
                    lower_next = next_text.lower()
                    is_action = any(lower_next.startswith(verb) for verb in ["prescribe", "refer", "advise", "monitor", "assess", "give", "initiate", "do not"])
                    if is_action:
                        branch_text.append(f"  ACTION: {next_text}")
                    else:
                        branch_text.append(f"  CONTEXT: {next_text}")
                        
                j += 1
                
            # Combine all collected branch items into a single decision branch chunk
            full_branch_content = "\n".join(branch_text)
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source_doc": source_doc,
                "doc_version": doc_version,
                "effective_date": effective_date,
                "superseded": superseded,
                "condition_code": condition_code,
                "section_path": path_str + " (Decision Flow)",
                "content": full_branch_content,
                "chunk_type": "decision_branch",
                "adjacent_clinical_alerts": "",
                "page_no": node["page_no"]
            })
            
            # Fast-forward our pointer past the processed nodes
            i = j
        else:
            i += 1
            
    return chunks
