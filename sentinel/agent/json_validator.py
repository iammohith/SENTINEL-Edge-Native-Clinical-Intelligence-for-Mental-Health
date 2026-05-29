"""
SENTINEL — JSON Validator and Repair Utility

Phase 5 implementation.
Repairs and validates JSON structures returned by the LLM (Finding #4 / #37).
Uses json_repair to handle trailing commas, unescaped quotes, and missing brackets,
then parses the repaired JSON into the expected Pydantic model.
"""

from __future__ import annotations

import json
import logging
from typing import Type, TypeVar

from json_repair import repair_json
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def validate_and_repair_json(raw_text: str, model_class: Type[T]) -> T:
    """
    Repairs and validates a raw text string representing JSON against a Pydantic model.
    Raises ValidationError or ValueError on failure, which triggers immediate escalation.
    """
    clean_text = raw_text.strip()
    
    # Strip markdown code block wrappers if the model generated them
    if clean_text.startswith("```"):
        # Remove starting code block
        lines = clean_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_text = "\n".join(lines).strip()
        
    try:
        # 1. Attempt standard JSON parsing
        parsed = json.loads(clean_text)
    except json.JSONDecodeError:
        logger.debug("Standard JSON parsing failed. Attempting json_repair...")
        try:
            # 2. Fallback to json_repair
            repaired = repair_json(clean_text)
            parsed = json.loads(repaired)
        except Exception as e:
            logger.error(f"JSON repair failed. Raw string: {raw_text[:200]}...")
            raise ValueError("Malformed JSON syntax that cannot be repaired") from e

    try:
        # 3. Validate using Pydantic
        return model_class.model_validate(parsed)
    except ValidationError as ve:
        logger.error(f"Pydantic schema validation failed: {ve}")
        raise ve
