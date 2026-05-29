import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from sentinel.agent.loop import run_clinical_query
from sentinel.config import MHIntentType
from sentinel.agent.intent import IntentClassification
from sentinel.agent.faithfulness import FaithfulnessResult, SentenceVerdict


@pytest.mark.asyncio
async def test_run_clinical_query_success(monkeypatch):
    # Mock PHI scrub: no PHI found
    monkeypatch.setattr("sentinel.agent.loop.scrub_phi", lambda q: (q, []))
    
    # Mock Crisis detector: level is NONE
    mock_crisis_res = MagicMock()
    mock_crisis_res.level = "NONE"
    mock_crisis_res.matched_signal = None
    mock_crisis_res.who_crisis_template = None
    monkeypatch.setattr("sentinel.agent.loop.detect_crisis", AsyncMock(return_value=mock_crisis_res))
    
    # Mock Intent classifier: DEP condition, ASSESSMENT_PROTOCOL intent
    mock_intent_res = IntentClassification(
        intent=MHIntentType.ASSESSMENT_PROTOCOL,
        condition_codes=["DEP"],
        confidence=0.90,
        rationale="Depression assessment query"
    )
    monkeypatch.setattr("sentinel.agent.loop.classify_intent", AsyncMock(return_value=mock_intent_res))
    
    # Mock Router & Retrieve: 1 standard chunk
    mock_chunks = [
        {
            "chunk_id": "c1",
            "source_doc": "mhgap.pdf",
            "doc_version": "2.0",
            "effective_date": "2016-10-01",
            "superseded": False,
            "condition_code": "DEP",
            "section_path": "DEP > Assessment",
            "content": "Evaluate for depressed mood.",
            "chunk_type": "procedure",
            "adjacent_clinical_alerts": "",
            "page_no": 12,
            "rerank_score": 2.5
        }
    ]
    monkeypatch.setattr("sentinel.agent.loop.route_and_retrieve", AsyncMock(return_value=mock_chunks))
    
    # Mock Clinical alerts: empty
    monkeypatch.setattr("sentinel.agent.loop.validate_clinical_alerts", lambda chunks: [])
    
    # Mock Sentence splitter: returns 1 sentence
    monkeypatch.setattr("sentinel.agent.loop.split_clinical_sentences", lambda text: [text])
    
    # Mock Faithfulness checker: returns faithful
    mock_faith_res = FaithfulnessResult(
        score=1.0,
        is_faithful=True,
        blocked=False,
        sentence_results=[SentenceVerdict(sentence="Evaluate for depressed mood.", label="ENTAILMENT", max_entailment_score=1.0, supporting_chunk_id="c1")],
        contradicted_sentences=[]
    )
    monkeypatch.setattr("sentinel.agent.loop.check_faithfulness", AsyncMock(return_value=mock_faith_res))
    
    # Mock Ollama chat stream generator
    mock_stream = AsyncMock()
    mock_stream.__aiter__.return_value = [
        {"message": {"content": "Evaluate "}},
        {"message": {"content": "for "}},
        {"message": {"content": "depressed "}},
        {"message": {"content": "mood."}}
    ]
    
    mock_client = MagicMock()
    mock_client.chat_stream.return_value = mock_stream
    
    # Run loop
    events = []
    tokens = []
    async for event in run_clinical_query("Patient is sad", "session_123", mock_client):
        if "token" in event:
            tokens.append(event["token"])
        else:
            events.append(event)
            
    # Verify standard execution steps were yielded
    steps = [e["step"] for e in events]
    assert "PHI_SCRUB" in steps
    assert "CRISIS_DETECT" in steps
    assert "INTENT_CLASSIFY" in steps
    assert "RETRIEVAL" in steps
    assert "CLINICAL_ALERTS" in steps
    assert "NLI_FAITHFULNESS" in steps
    assert "CONFIDENCE_CALIBRATE" in steps
    assert "LOOP_DECISION" in steps
    
    # Verify final compiled answer
    assert "".join(tokens) == "Evaluate for depressed mood."
    
    # Verify loop decision was accepted
    decision_event = [e for e in events if e["step"] == "LOOP_DECISION"][0]
    assert decision_event["status"] == "ACCEPTED"
