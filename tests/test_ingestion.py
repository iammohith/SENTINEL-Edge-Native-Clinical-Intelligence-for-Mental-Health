import pytest
from sentinel.ingestion.versioning import parse_version_metadata, resolve_supersession_logic
from sentinel.ingestion.postprocessor import (
    _determine_condition_code,
    _is_clinical_alert,
    _estimate_tokens,
    _format_table_to_markdown
)


def test_parse_version_metadata():
    # Test v2.0
    meta_v2 = parse_version_metadata("mhgap_ig_v2.0_2016.pdf")
    assert meta_v2["doc_version"] == "2.0-2016"
    assert meta_v2["effective_date"] == "2016-10-01"
    assert meta_v2["document_type"] == "guideline"

    # Test humanitarian
    meta_hig = parse_version_metadata("mhgap_humanitarian_guide_2015.pdf")
    assert meta_hig["doc_version"] == "2.0-hig-2015"
    assert meta_hig["effective_date"] == "2015-05-01"
    assert meta_hig["document_type"] == "humanitarian_guideline"

    # Test default/fallback
    meta_def = parse_version_metadata("some_unknown_document.pdf")
    assert meta_def["doc_version"] == "1.0"
    assert meta_def["effective_date"] == "2010-01-01"
    assert meta_def["document_type"] == "guideline"


def test_resolve_supersession_logic():
    incoming = {
        "doc_version": "2.0-2016",
        "effective_date": "2016-10-01",
        "document_type": "guideline"
    }
    
    # Scenario 1: Existing is older (2010)
    existing_older = [
        {
            "doc_version": "1.0",
            "effective_date": "2010-01-01",
            "document_type": "guideline",
            "source_doc": "mhgap_2010.pdf"
        }
    ]
    is_superseded, to_supersede = resolve_supersession_logic(incoming, existing_older)
    assert is_superseded is False
    assert to_supersede == ["mhgap_2010.pdf"]

    # Scenario 2: Existing is newer (e.g. 2022 guidelines)
    existing_newer = [
        {
            "doc_version": "3.0",
            "effective_date": "2022-01-01",
            "document_type": "guideline",
            "source_doc": "mhgap_2022.pdf"
        }
    ]
    is_superseded_newer, to_supersede_newer = resolve_supersession_logic(incoming, existing_newer)
    assert is_superseded_newer is True
    assert len(to_supersede_newer) == 0


def test_determine_condition_code():
    assert _determine_condition_code(["DEP", "Assessment"]) == "DEP"
    assert _determine_condition_code(["Psychosis module", "Treatment"]) == "PSY"
    assert _determine_condition_code(["Seizures / Epilepsy", "Guidance"]) == "EPI"
    assert _determine_condition_code(["Some unknown section"]) == "GEN"


def test_is_clinical_alert():
    assert _is_clinical_alert("Caution: Side effects include tremors.") is True
    assert _is_clinical_alert("Do not administer this drug to children.") is True
    assert _is_clinical_alert("This is a routine follow-up step.") is False


def test_estimate_tokens():
    text = "Hello world! This is a test."
    tokens = _estimate_tokens(text)
    assert tokens == int(6 * 1.3)
