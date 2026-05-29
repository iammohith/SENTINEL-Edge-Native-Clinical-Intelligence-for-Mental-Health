import pytest
from sentinel.safety.clinical_alerts import validate_clinical_alerts
from sentinel.safety.crisis_detector import detect_crisis
from sentinel.safety.phi_scrubber import scrub_phi


@pytest.mark.asyncio
async def test_crisis_detector_keywords():
    # Test Tier-1 keyword
    res1 = await detect_crisis("how to commit suicide using medications")
    assert res1.level == "TIER_1"
    assert "Keyword match" in res1.matched_signal
    assert res1.who_crisis_template is not None

    # Test Tier-2 keyword
    res2 = await detect_crisis("the patient is cutting wrists and self-harming")
    assert res2.level == "TIER_2"
    assert "Keyword match" in res2.matched_signal
    assert res2.who_crisis_template is None

    # Test no crisis
    res3 = await detect_crisis("What is the dose of fluoxetine for mild depression?")
    assert res3.level == "NONE"
    assert res3.matched_signal is None


def test_phi_scrubber():
    # Test PHI de-identification
    query = "My patient Jane Doe (DOB 14/05/1990) living in Chicago is depressed."
    scrubbed, entities = scrub_phi(query)
    
    # Names, dates, and locations should be de-identified
    assert "Jane Doe" not in scrubbed
    assert "14/05/1990" not in scrubbed
    assert "Chicago" not in scrubbed
    
    # Presidio should have caught some entities
    assert len(entities) > 0


def test_clinical_alerts_validator():
    chunks = [
        {
            "chunk_type": "standard",
            "content": "Give antidepressant treatment.",
            "adjacent_clinical_alerts": "WARNING: Amitriptyline is cardiotoxic in overdose.\n"
        },
        {
            "chunk_type": "clinical_alert",
            "content": "Do not administer valproic acid to women of reproductive age.",
            "adjacent_clinical_alerts": None
        }
    ]
    alerts = validate_clinical_alerts(chunks)
    assert len(alerts) == 2
    assert "WARNING: Amitriptyline is cardiotoxic in overdose." in alerts
    assert "Do not administer valproic acid to women of reproductive age." in alerts
