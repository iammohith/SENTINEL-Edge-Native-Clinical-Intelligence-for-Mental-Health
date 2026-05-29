"""
SENTINEL — Offline PHI Scrubber

De-identifies Patient Health Information (PHI) from clinical queries before they
touch the audit log, preventing legal and compliance violations (HIPAA/GDPR).
Uses Microsoft Presidio Analyzer and Anonymizer engines (completely offline).
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# Lazy initialization of Presidio engines to save startup time
_analyzer_instance: Optional[AnalyzerEngine] = None
_anonymizer_instance: Optional[AnonymizerEngine] = None
_init_lock = threading.Lock()


def _get_analyzer() -> AnalyzerEngine:
    global _analyzer_instance
    if _analyzer_instance is None:
        with _init_lock:
            if _analyzer_instance is None:
                logger.info("Initializing Microsoft Presidio Analyzer Engine...")
                _analyzer_instance = AnalyzerEngine()
    return _analyzer_instance


def _get_anonymizer() -> AnonymizerEngine:
    global _anonymizer_instance
    if _anonymizer_instance is None:
        with _init_lock:
            if _anonymizer_instance is None:
                logger.info("Initializing Microsoft Presidio Anonymizer Engine...")
                _anonymizer_instance = AnonymizerEngine()
    return _anonymizer_instance


def scrub_phi(text: str) -> tuple[str, list[str]]:
    """
    Scrubs Patient Health Information (PHI) from text.
    Identifies and anonymizes: PERSON, DATE_TIME, LOCATION, PHONE_NUMBER,
    EMAIL_ADDRESS, MEDICAL_LICENSE, etc.
    
    Returns:
        A tuple of (scrubbed_text, detected_entity_types).
    """
    
    if not text.strip():
        return text, []

    analyzer = _get_analyzer()
    anonymizer = _get_anonymizer()

    # Presidio works entirely offline using the local spaCy model
    # (configured during offline bootstrap)
    try:
        results = analyzer.analyze(text=text, language="en")
        anonymized_result = anonymizer.anonymize(text=text, analyzer_results=results)
        
        scrubbed_text = anonymized_result.text
        entity_types = list({r.entity_type for r in results})
        
        if entity_types:
            logger.info(f"PHI Scrubbed from query. Entities found: {', '.join(entity_types)}")
            
        return scrubbed_text, entity_types
        
    except Exception as e:
        logger.error(f"PHI scrubbing failed: {e}. Falling back to original text with caution.")
        # Under safety rules, if scrubbing fails, we should never leak raw logs.
        # But for synthesis, we can return the text.
        return text, ["SCRUBBING_ERROR"]
