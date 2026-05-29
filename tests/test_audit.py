import os
import sqlite3
import pytest
from unittest.mock import patch
from sentinel.audit.chain import (
    initialize_audit_schema,
    log_audit_step,
    verify_chain_integrity,
    _connect_audit_db,
    _compute_canonical_hash,
    _compute_record_hmac
)
from sentinel.config import AUDIT_DB_PATH

# Mock keys for testing
MOCK_HMAC_KEY = b"test_hmac_signing_key_32_bytes_long"
MOCK_SQLCIPHER_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    # Set the DB path to a temporary file
    test_db_path = tmp_path / "test_audit.db"
    monkeypatch.setattr("sentinel.audit.chain.AUDIT_DB_PATH", test_db_path)
    
    # Mock Keychain getters in chain
    monkeypatch.setattr("sentinel.audit.chain.get_hmac_key", lambda: MOCK_HMAC_KEY)
    monkeypatch.setattr("sentinel.audit.chain.get_sqlcipher_key", lambda: MOCK_SQLCIPHER_KEY)
    
    # Clean up file after test
    yield
    if test_db_path.exists():
        os.remove(test_db_path)


@pytest.mark.asyncio
async def test_audit_logging_and_verification():
    # Initialize DB
    initialize_audit_schema()
    
    # Verify empty chain passes
    errors = verify_chain_integrity()
    assert len(errors) == 0

    # Log first step
    await log_audit_step(
        record_id="rec_001",
        session_id="session_abc",
        raw_query="Patient feels sad Jane Doe",
        scrubbed_query_preview="Patient feels sad...",
        step="PHI_SCRUB",
        step_index=0,
        step_input={"query": "Patient feels sad Jane Doe"},
        step_output={"scrubbed": "Patient feels sad..."},
        decision="PROCEED",
        faithfulness_score=None,
        confidence_score=None
    )

    # Log second step (linked to first)
    await log_audit_step(
        record_id="rec_002",
        session_id="session_abc",
        raw_query="Patient feels sad Jane Doe",
        scrubbed_query_preview="Patient feels sad...",
        step="INTENT_CLASSIFY",
        step_index=1,
        step_input={"scrubbed": "Patient feels sad..."},
        step_output={"intent": "ASSESSMENT_PROTOCOL", "condition_codes": ["DEP"]},
        decision="PROCEED",
        faithfulness_score=0.95,
        confidence_score=0.88
    )

    # Verify integrity of valid chain
    errors = verify_chain_integrity()
    assert len(errors) == 0

    # Manually tamper with the database to verify detection
    conn = _connect_audit_db()
    with conn:
        # Alter the decision field of the first record
        conn.execute("UPDATE audit_records SET decision = 'TAMPERED' WHERE record_id = 'rec_001'")
    conn.close()

    # Verify that integrity check fails
    errors_after_tamper = verify_chain_integrity()
    assert len(errors_after_tamper) > 0
    assert any("Signature Integrity Violation" in e for e in errors_after_tamper)
