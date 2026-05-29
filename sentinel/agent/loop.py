"""
SENTINEL — Clinical Agentic Loop (Core Reasoning Engine)

Phase 5 implementation.
Orchestrates the entire clinical RAG + reasoning pipeline (STEPS 0-11)
with active self-correction / refinement loops and strict safety gates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator

from sentinel.agent.confidence import compute_confidence_score
from sentinel.agent.escalation import escalate_query
from sentinel.agent.faithfulness import check_faithfulness
from sentinel.agent.intent import classify_intent
from sentinel.agent.ollama_client import ResilientOllamaClient
from sentinel.agent.router import route_and_retrieve
from sentinel.agent.sentence_splitter import split_clinical_sentences
from sentinel.agent.session import session_manager
from sentinel.config import (
    LLM_MODEL,
    MAX_LOOP_ITERATIONS,
    CONFIDENCE_ESCALATE_THRESHOLD,
)
from sentinel.safety.clinical_alerts import validate_clinical_alerts
from sentinel.safety.crisis_detector import detect_crisis
from sentinel.safety.phi_scrubber import scrub_phi

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """
You are SENTINEL, a clinical assistant grounded strictly in the WHO mhGAP-IG v2.0 guidelines.
Your goal is to synthesize a safe, accurate, and fully-grounded response to the clinician's query.

Instructions:
1. Synthesize your answer ONLY using the retrieved facts provided in the context below.
2. If any fact is not in the context, do not assume or invent it. If you cannot answer using only the context, state: "I do not have this information in the WHO mhGAP corpus."
3. Cite the section path and page number for every clinical claim you make (e.g. "[DEP > Assessment > Step 2, p.15]").
4. Keep the response professional, concise, and clinically safe.
5. If the query indicates a crisis (Tier-2), ensure a supportive but direct protocol tone is used.
"""


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Formats retrieved chunks as a text context block for the LLM."""
    formatted = []
    for idx, chunk in enumerate(chunks):
        formatted.append(
            f"--- Context Block {idx + 1} ---\n"
            f"Source: {chunk['source_doc']} (v{chunk['doc_version']})\n"
            f"Section Path: {chunk['section_path']} (Page: {chunk['page_no']})\n"
            f"Content:\n{chunk['content']}\n"
        )
    return "\n".join(formatted)


async def run_clinical_query(
    query: str,
    session_id: str,
    client: ResilientOllamaClient
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Main agentic reasoning loop.
    Yields step-by-step metadata for the SSE dashboard stream,
    followed by the generated clinical answer tokens.
    """
    # ── STEP 0: PHI Scrubbing ──────────────────────────────────────────────────
    yield {"step": "PHI_SCRUB", "status": "START"}
    scrubbed_query, phi_entities = scrub_phi(query)
    yield {
        "step": "PHI_SCRUB",
        "status": "COMPLETE",
        "scrubbed_query_preview": scrubbed_query[:50] + "...",
        "entities_detected": phi_entities
    }

    # ── STEP 1: Crisis Detection ───────────────────────────────────────────────
    yield {"step": "CRISIS_DETECT", "status": "START"}
    crisis_res = await detect_crisis(scrubbed_query)
    
    if crisis_res.level == "TIER_1":
        yield {
            "step": "CRISIS_DETECT",
            "status": "CRISIS_BLOCK",
            "level": "TIER_1",
            "signal": crisis_res.matched_signal
        }
        # Immediately output the pre-validated WHO crisis response template
        yield {"step": "SYNTHESIS", "status": "STREAM_START", "crisis_level": "TIER_1"}
        yield {"token": crisis_res.who_crisis_template}
        yield {"step": "SYNTHESIS", "status": "STREAM_END", "confidence": 1.0, "is_faithful": True}
        # Log to audit chain via caller, return
        return

    yield {
        "step": "CRISIS_DETECT",
        "status": "COMPLETE",
        "level": crisis_res.level,
        "signal": crisis_res.matched_signal
    }

    # ── STEP 2: Intent Classification ──────────────────────────────────────────
    yield {"step": "INTENT_CLASSIFY", "status": "START"}
    try:
        classification = await classify_intent(scrubbed_query, client)
        yield {
            "step": "INTENT_CLASSIFY",
            "status": "COMPLETE",
            "intent": classification.intent.value,
            "condition_codes": classification.condition_codes,
            "confidence": classification.confidence
        }
    except Exception as e:
        # Escalate on classification failure (Finding #42)
        esc_id = escalate_query(session_id, scrubbed_query, f"INTENT_CLASSIFICATION_FAILURE: {e}")
        yield {
            "step": "INTENT_CLASSIFY",
            "status": "ESCALATED",
            "escalation_id": esc_id,
            "error": str(e)
        }
        yield {"step": "SYNTHESIS", "status": "STREAM_START", "escalation_id": esc_id}
        yield {"token": f"⛔ CLINICAL ESCALATION REQUIRED (ID: {esc_id})\nReason: Intent classification failure. A clinical reviewer has been notified."}
        yield {"step": "SYNTHESIS", "status": "STREAM_END", "confidence": 0.0, "is_faithful": False}
        return

    # Check for Out of Scope intent
    if classification.intent == "OUT_OF_SCOPE":
        esc_id = escalate_query(session_id, scrubbed_query, "OUT_OF_SCOPE_QUERY")
        yield {"step": "SYNTHESIS", "status": "STREAM_START", "escalation_id": esc_id}
        yield {"token": f"⛔ OUT OF SCOPE (ID: {esc_id})\nThis query is outside the clinical domain of the WHO mhGAP Intervention Guide. Sentinel cannot provide answers for non-clinical or out-of-scope topics."}
        yield {"step": "SYNTHESIS", "status": "STREAM_END", "confidence": 0.0, "is_faithful": True}
        return

    # ── STEP 3 & 4: Retrieval and Reranking ────────────────────────────────────
    yield {"step": "RETRIEVAL", "status": "START"}
    # Retrieve candidates based on condition codes
    retrieved_chunks = await route_and_retrieve(
        query=scrubbed_query,
        intent=classification.intent,
        condition_codes=classification.condition_codes
    )
    yield {
        "step": "RETRIEVAL",
        "status": "COMPLETE",
        "num_chunks_retrieved": len(retrieved_chunks),
        "citations": [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source_doc"],
                "section": c["section_path"],
                "page": c["page_no"],
                "superseded": c["superseded"]
            }
            for c in retrieved_chunks
        ]
    }

    # ── STEP 5: Validate Clinical Alerts ───────────────────────────────────────
    yield {"step": "CLINICAL_ALERTS", "status": "START"}
    alerts = validate_clinical_alerts(retrieved_chunks)
    yield {
        "step": "CLINICAL_ALERTS",
        "status": "COMPLETE",
        "alerts_found": alerts
    }

    # ── STEP 6: Contradiction check / Cross reference ─────────────────────────
    # If the user asked a comorbidity question or contradiction check
    # we can run a quick check, but default is standard synthesis.

    # Retrieve session history context (multi-turn)
    session = session_manager.get_session(session_id)
    history_context = session.get_history_summary()

    # ── STEP 7: Synthesize Answer (Refinement Loop) ────────────────────────────
    # Keep track of refinement iterations
    iteration = 1
    max_rerank_score = retrieved_chunks[0]["rerank_score"] if retrieved_chunks else 0.0
    
    context_text = _format_context(retrieved_chunks)
    
    # System prompt builder
    sys_content = f"{SYNTHESIS_SYSTEM_PROMPT.strip()}\n\n{history_context}\n\n[CLINICAL KNOWLEDGE BASE CONTEXT]\n{context_text}"
    if crisis_res.level == "TIER_2":
        sys_content += "\n\n⚠ SAFETY TRIGGER ACTIVE: The clinician query contains crisis signals. Ensure safe-messaging protocols are emphasized. Highlight referral criteria prominently."

    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": f"Query: {scrubbed_query}"}
    ]

    while iteration <= MAX_LOOP_ITERATIONS:
        yield {"step": "SYNTHESIS", "status": f"START_ITER_{iteration}"}
        
        # If it's the final iteration of loop (respond or escalate)
        # We fetch the completion from Ollama.
        # Note: we only stream on the final accepted loop run or on iteration 1
        # For the refinement iterations, we run them synchronously in the background.
        
        try:
            # We call the model
            # For the first iteration, we stream tokens directly to the dashboard
            if iteration == 1:
                yield {"step": "SYNTHESIS", "status": "STREAM_START", "crisis_level": crisis_res.level}
                
                # Gather streamed tokens
                full_draft_parts = []
                generator = client.chat_stream(model=LLM_MODEL, messages=messages)
                async for chunk in generator:
                    token = chunk["message"]["content"]
                    full_draft_parts.append(token)
                    yield {"token": token}
                    
                full_draft = "".join(full_draft_parts)
                yield {"step": "SYNTHESIS", "status": "STREAM_END"}
            else:
                # Synchronous fetch for refinement iterations
                response = await client.chat(model=LLM_MODEL, messages=messages, options={"temperature": 0.0})
                full_draft = response["message"]["content"]

            # ── STEP 8: Split Clinical Sentences ───────────────────────────────
            yield {"step": "SENTENCE_SPLIT", "status": "START"}
            sentences = split_clinical_sentences(full_draft)
            yield {
                "step": "SENTENCE_SPLIT",
                "status": "COMPLETE",
                "num_sentences": len(sentences)
            }

            # ── STEP 9: NLI Faithfulness check ─────────────────────────────────
            yield {"step": "NLI_FAITHFULNESS", "status": "START"}
            faithfulness = await check_faithfulness(sentences, retrieved_chunks)
            yield {
                "step": "NLI_FAITHFULNESS",
                "status": "COMPLETE",
                "score": faithfulness.score,
                "is_faithful": faithfulness.is_faithful,
                "blocked": faithfulness.blocked,
                "contradictions": faithfulness.contradicted_sentences
            }

            # ── STEP 10: Compute Confidence Scorer ──────────────────────────────
            yield {"step": "CONFIDENCE_CALIBRATE", "status": "START"}
            language_warning = False  # English-only v1 scope limit is checked at ingest
            confidence_score = compute_confidence_score(
                intent_confidence=classification.confidence,
                max_rerank_score=max_rerank_score,
                faithfulness_score=faithfulness.score,
                language_warning=language_warning,
                has_clinical_alerts=len(alerts) > 0
            )
            yield {
                "step": "CONFIDENCE_CALIBRATE",
                "status": "COMPLETE",
                "confidence_score": confidence_score
            }

            # ── STEP 11: Respond, Refine, or Escalate ──────────────────────────
            # Success criteria:
            # - No contradictions found (blocked = False)
            # - Faithfulness score is above threshold (is_faithful = True)
            # - Confidence score is above threshold (>= 0.70)
            
            if not faithfulness.blocked and faithfulness.is_faithful and confidence_score >= CONFIDENCE_ESCALATE_THRESHOLD:
                # Success! Accept response and update session history
                session.add_turn(query, full_draft, classification.condition_codes)
                
                # If we did refinement, we stream the final accepted draft to the user
                if iteration > 1:
                    yield {"step": "SYNTHESIS", "status": "STREAM_START", "refined": True}
                    yield {"token": full_draft}
                    yield {"step": "SYNTHESIS", "status": "STREAM_END"}
                    
                yield {
                    "step": "LOOP_DECISION",
                    "status": "ACCEPTED",
                    "iterations": iteration,
                    "confidence": confidence_score,
                    "faithfulness": faithfulness.score
                }
                return
                
            # If we failed verification, check if we can refine
            if iteration < MAX_LOOP_ITERATIONS:
                logger.warning(f"Verification failed on iteration {iteration}. Initiating refinement...")
                
                # Gather ungrounded sentences
                ungrounded = []
                for verdict in faithfulness.sentence_results:
                    if verdict.label in ("CONTRADICTION", "NEUTRAL"):
                        ungrounded.append(verdict.sentence)
                        
                ungrounded_text = "\n".join([f"- {s}" for s in ungrounded])
                
                # Construct refinement prompt
                refinement_msg = (
                    f"Your previous draft response failed the NLI faithfulness audit. The following sentences were flagged as either contradicted by the WHO guidelines or completely ungrounded:\n"
                    f"{ungrounded_text}\n\n"
                    f"Please rewrite the response. Make sure to:\n"
                    f"1. Remove or correct all contradicted and ungrounded statements.\n"
                    f"2. Stick strictly to the retrieved facts in the context.\n"
                    f"3. Cite the section path and page number for every clinical claim."
                )
                
                # Add to message thread
                messages.append({"role": "assistant", "content": full_draft})
                messages.append({"role": "user", "content": refinement_msg})
                
                iteration += 1
            else:
                # We have reached max iterations and still failed validation
                logger.error("Verification failed after max iterations. Escalating query.")
                esc_id = escalate_query(session_id, scrubbed_query, "UNGROUNDED_SYNTHESIS_MAX_ITER")
                
                yield {
                    "step": "LOOP_DECISION",
                    "status": "ESCALATED",
                    "escalation_id": esc_id,
                    "reason": "UNGOUNDED_SYNTHESIS_MAX_ITER"
                }
                
                yield {"step": "SYNTHESIS", "status": "STREAM_START", "escalation_id": esc_id}
                yield {"token": f"\n\n⛔ CLINICAL ESCALATION REQUIRED (ID: {esc_id})\nReason: Faithfulness verification failed. The generated clinical response could not be verified against the WHO guidelines with sufficient confidence. A clinical reviewer has been notified."}
                yield {"step": "SYNTHESIS", "status": "STREAM_END"}
                return
                
        except Exception as e:
            logger.error(f"Error in agentic loop: {e}")
            esc_id = escalate_query(session_id, scrubbed_query, f"AGENT_LOOP_EXCEPTION: {e}")
            yield {"step": "SYNTHESIS", "status": "STREAM_START", "escalation_id": esc_id}
            yield {"token": f"⛔ SYSTEM ERROR ESCALATION (ID: {esc_id})\nAn unexpected error occurred in the reasoning engine: {e}"}
            yield {"step": "SYNTHESIS", "status": "STREAM_END"}
            return
