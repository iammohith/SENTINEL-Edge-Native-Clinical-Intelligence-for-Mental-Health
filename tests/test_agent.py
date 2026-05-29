import pytest
import time
from sentinel.agent.confidence import compute_confidence_score
from sentinel.agent.session import SessionContext, session_manager
from sentinel.agent.sentence_splitter import split_clinical_sentences


def test_confidence_scorer():
    # Test perfect conditions
    score1 = compute_confidence_score(
        intent_confidence=1.0,
        max_rerank_score=6.0,  # Maxed normalized rerank score
        faithfulness_score=1.0,
        language_warning=False,
        has_clinical_alerts=False
    )
    assert score1 == 1.0

    # Test baseline calculation
    # Normalization range is [-15.0, 5.0] → linear map to [0.0, 1.0]
    # If max_rerank_score = -1.0, normalized_rerank = (-1+15)/20 = 14/20 = 0.70
    score2 = compute_confidence_score(
        intent_confidence=0.80,
        max_rerank_score=-1.0,
        faithfulness_score=0.90,
        language_warning=False,
        has_clinical_alerts=False
    )
    # Base score = (0.40 * 0.80) + (0.60 * 0.70) = 0.32 + 0.42 = 0.74
    # Final = 0.74 * 0.90 = 0.666
    assert pytest.approx(score2, 0.001) == 0.666

    # Test language warning penalty (0.80 multiplier)
    score3 = compute_confidence_score(
        intent_confidence=1.0,
        max_rerank_score=6.0,
        faithfulness_score=1.0,
        language_warning=True,
        has_clinical_alerts=False
    )
    assert pytest.approx(score3, 0.001) == 0.80


def test_session_manager():
    session_id = "test_sess_001"
    session = session_manager.get_session(session_id)
    assert isinstance(session, SessionContext)
    assert session.session_id == session_id
    assert len(session.turns) == 0

    # Add 4 turns (more than the capacity of 3)
    session.add_turn("Q1", "A1", ["DEP"])
    session.add_turn("Q2", "A2", ["PSY"])
    session.add_turn("Q3", "A3", ["SUD"])
    session.add_turn("Q4", "A4", ["EPI"])

    # Ring buffer should keep only the last 3 turns
    assert len(session.turns) == 3
    assert session.turns[0].query == "Q2"
    assert session.turns[2].query == "Q4"

    # Verify formatting output
    summary = session.get_history_summary()
    assert "Q2" in summary
    assert "Q4" in summary
    assert "Q1" not in summary

    # Verify clear session
    session_manager.clear_session(session_id)
    session_new = session_manager.get_session(session_id)
    assert len(session_new.turns) == 0


def test_sentence_splitter():
    # Test text containing clinical abbreviations
    text = "The patient was diagnosed with epilepsy. Administer phenobarbital 100 mg/kg b.i.d. for convulsions. Refer to Dr. Smith if needed."
    
    sentences = split_clinical_sentences(text)
    
    # It should split into 3 sentences:
    # 1. The patient was diagnosed with epilepsy.
    # 2. Administer phenobarbital 100 mg/kg b.i.d. for convulsions.
    # 3. Refer to Dr. Smith if needed.
    #
    # Wait, the splitter cleans up conversational prefixes.
    assert len(sentences) == 3
    assert sentences[0] == "The patient was diagnosed with epilepsy."
    assert sentences[1] == "Administer phenobarbital 100 mg/kg b.i.d. for convulsions."
    assert sentences[2] == "Refer to Dr. Smith if needed."
