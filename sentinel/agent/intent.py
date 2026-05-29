"""
SENTINEL — Clinical Intent Classifier

Phase 5 implementation.
Classifies patient/clinician query into one of the clinical intent types (MHIntentType)
and associates it with relevant mhGAP condition codes.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field

from sentinel.agent.ollama_client import ResilientOllamaClient
from sentinel.config import CONDITION_CODES, LLM_MODEL, MHIntentType

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Structured Pydantic output schema for intent classification."""
    intent: MHIntentType = Field(
        description="The primary clinical intent type of the query."
    )
    condition_codes: list[str] = Field(
        description="List of matching condition codes from: DEP, PSY, SUD, EPI, DEM, DLD, SHI, OTH, GEN."
    )
    confidence: float = Field(
        description="Confidence score for this classification, between 0.0 and 1.0."
    )
    rationale: str = Field(
        description="Short clinical reasoning explaining this classification."
    )


INTENT_PROMPT = """
You are the Clinical Intent Classifier for SENTINEL, a system grounded in the WHO mhGAP-IG v2.0.

Your task is to analyze the clinician's query and classify it into exactly ONE primary intent type and list the relevant condition codes.

Available Intent Types:
- ASSESSMENT_PROTOCOL: How to assess or diagnose a condition (criteria, symptoms, signs).
- TREATMENT_PROTOCOL: First-line or second-line management protocols (counseling, psychoeducation, non-pharmacological).
- MEDICATION_GUIDANCE: Drug tables, dosages, contraindications, side effects, prescribing guidance.
- REFERRAL_CRITERIA: Signs of severity, when to refer a patient to a specialist or psychiatric emergency.
- FOLLOW_UP_PROTOCOL: Monitoring schedules, reassessment frequency, when to review treatment.
- CRISIS_RESPONSE: Acute emergencies (acute psychosis, active suicide plan, withdrawal seizures).
- CONDITION_OVERVIEW: General informational questions about a condition.
- CONTRADICTION_CHECK: Spotting conflicts or checking incompatibilities between multiple guidelines.
- OUT_OF_SCOPE: Questions not covered by the WHO mhGAP corpus (e.g. general web topics, non-clinical trivia, surgery).

Available Condition Codes:
- DEP: Depression / Depressive symptoms
- PSY: Psychosis / Schizophrenia
- SUD: Substance Use Disorders (Alcohol, Drug use)
- EPI: Epilepsy / Seizures
- DEM: Dementia / Cognitive decline
- DLD: Developmental and Behavioral Disorders (Child/Adolescent)
- SHI: Self-Harm / Suicide
- OTH: Other Significant Mental Health Conditions (Somatic complaints, etc.)
- GEN: General Principles of Care / Essential Care / MHPSS

Guidelines:
1. Output MUST be valid JSON matching the schema:
{{
  "intent": "INTENT_TYPE",
  "condition_codes": ["CODE1", "CODE2"],
  "confidence": 0.95,
  "rationale": "Reasoning..."
}}
2. Do not include any markdown format tags like ```json or trailing text. Return ONLY the JSON object.
3. If the query does not relate to WHO mhGAP, classify as OUT_OF_SCOPE and condition_codes ["OTH"].
"""


async def classify_intent(query: str, client: ResilientOllamaClient) -> IntentClassification:
    """
    Classifies the clinician's query into an IntentClassification schema.
    Strictly escalates on failure. No regex fallback (Finding #42).
    """
    import json
    from json_repair import repair_json

    messages = [
        {"role": "system", "content": INTENT_PROMPT.strip()},
        {"role": "user", "content": f"Clinician Query: \"{query}\"\nJSON Classification:"}
    ]

    try:
        logger.info("Sending query to Ollama for intent classification...")
        # Enforce json format format="json" in Ollama call (Finding #4)
        response = await client.chat(
            model=LLM_MODEL,
            messages=messages,
            format="json",
            options={"temperature": 0.0}
        )
        
        raw_content = response["message"]["content"].strip()
        logger.debug(f"Raw classification output: {raw_content}")

        # Attempt to repair if malformed, and load JSON
        repaired = repair_json(raw_content)
        parsed = json.loads(repaired)

        # Validate against Pydantic schema
        classification = IntentClassification.model_validate(parsed)
        
        # Clean up condition codes to match allowed set
        valid_codes = [c.upper() for c in classification.condition_codes if c.upper() in CONDITION_CODES]
        if not valid_codes:
            valid_codes = ["GEN"]
        classification.condition_codes = valid_codes

        logger.info(f"Classified intent: {classification.intent.value} | Conditions: {classification.condition_codes}")
        return classification

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        # Reject and raise: the core loop will catch this and trigger an immediate escalation (Finding #42)
        raise ValueError("INTENT_CLASSIFICATION_FAILURE") from e
