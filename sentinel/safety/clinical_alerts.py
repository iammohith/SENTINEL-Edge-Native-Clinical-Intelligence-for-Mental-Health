"""
SENTINEL — Clinical Alert Validator

Deterministic safety step (STEP 5). Scans retrieved clinical knowledge chunks
for critical warnings, precautions, or contraindications (clinical alerts),
aggregating them for explicit UI highlighting and synthesis gating.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_clinical_alerts(chunks: list[dict[str, Any]]) -> list[str]:
    """
    Scans the retrieved chunks and extracts any explicit clinical alerts.
    
    Checks both:
      - Chunks of type 'clinical_alert'
      - The 'adjacent_clinical_alerts' field populated during postprocessing
      
    Returns:
        A list of unique clinical alert text strings.
    """
    alerts: set[str] = set()

    for chunk in chunks:
        # Check if the chunk itself is a clinical alert
        if chunk.get("chunk_type") == "clinical_alert":
            alerts.add(chunk["content"].strip())
            
        # Check if there are linked adjacent alerts
        adjacent = chunk.get("adjacent_clinical_alerts")
        if adjacent:
            for alert in adjacent.split("\n"):
                clean_alert = alert.strip()
                if clean_alert:
                    alerts.add(clean_alert)

    unique_alerts = list(alerts)
    if unique_alerts:
        logger.info(f"Clinical Alert Validator: Found {len(unique_alerts)} critical warnings in retrieved context.")
        
    return unique_alerts
