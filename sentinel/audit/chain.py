"""
SENTINEL — Tamper-Evident Audit Chain Logger

Phase 6 implementation.
Logs every reasoning step of the agentic loop to an encrypted SQLCipher DB.
Uses SHA-256 hash chaining and HMAC-SHA256 signatures to detect tampering (Finding #30 / #48 / #63).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sqlite3
from typing import Any, Optional

from sentinel.audit.key_manager import get_hmac_key, get_sqlcipher_key
from sentinel.config import AUDIT_DB_PATH

logger = logging.getLogger(__name__)


def _connect_audit_db() -> sqlite3.Connection:
    """Connects to the SQLCipher database and applies Keychain credentials."""
    from sqlcipher3 import dbapi2 as sqlcipher
    
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlcipher.connect(str(AUDIT_DB_PATH))
    
    # Authenticate FIRST (SQLCipher requirement)
    try:
        key = get_sqlcipher_key()
        conn.execute(f"PRAGMA key = \"x'{key}'\"")
    except Exception as e:
        conn.close()
        raise e
        
    # Configure WAL mode and busy timeout (Finding #30 / #67)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
        
    return conn


def initialize_audit_schema() -> None:
    """Initializes the audit_records table schema if it does not exist."""
    conn = _connect_audit_db()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    session_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    scrubbed_query_preview TEXT NOT NULL,
                    original_query_hash TEXT NOT NULL,
                    step TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    input_canonical_hash TEXT NOT NULL,
                    output_canonical_hash TEXT NOT NULL,
                    faithfulness_score REAL,
                    confidence_score REAL,
                    decision TEXT NOT NULL,
                    prev_record_hash TEXT,
                    record_hmac TEXT NOT NULL
                )
            """)
            logger.info("Audit database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize audit schema: {e}")
    finally:
        conn.close()


def _compute_canonical_hash(record_dict: dict[str, Any]) -> str:
    """Computes SHA-256 hash of a dictionary formatted as canonical JSON (sorted keys)."""
    canonical_str = json.dumps(record_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def _compute_record_hmac(record_dict: dict[str, Any], key: bytes) -> str:
    """Computes HMAC-SHA256 signature of the canonical JSON representation of a record."""
    canonical_str = json.dumps(record_dict, sort_keys=True, separators=(",", ":"))
    return hmac.new(key, canonical_str.encode("utf-8"), hashlib.sha256).hexdigest()


async def log_audit_step(
    record_id: str,
    session_id: str,
    raw_query: str,
    scrubbed_query_preview: str,
    step: str,
    step_index: int,
    step_input: Any,
    step_output: Any,
    decision: str,
    faithfulness_score: Optional[float] = None,
    confidence_score: Optional[float] = None
) -> None:
    """
    Logs an agentic reasoning step to the tamper-evident audit database.
    Computes previous record hash linkage and signs the record with the HMAC key.
    """
    # Initialize schema
    initialize_audit_schema()
    
    # Compute query hash (de-identified linkage)
    query_hash = hashlib.sha256(raw_query.encode("utf-8")).hexdigest()
    
    # Compute canonical input/output hashes
    input_hash = _compute_canonical_hash({"input": step_input})
    output_hash = _compute_canonical_hash({"output": step_output})

    conn = _connect_audit_db()
    try:
        # Fetch the previous record's details to calculate the hash chain
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT record_id, session_id, scrubbed_query_preview, original_query_hash,
                   step, step_index, input_canonical_hash, output_canonical_hash,
                   faithfulness_score, confidence_score, decision, prev_record_hash, record_hmac
            FROM audit_records
            ORDER BY sequence DESC LIMIT 1
            """
        )
        prev_row = cursor.fetchone()
        
        prev_record_hash = None
        if prev_row:
            # We serialize the entire previous record content (including its hmac) to build the chain
            prev_record_dict = {
                "record_id": prev_row[0],
                "session_id": prev_row[1],
                "scrubbed_query_preview": prev_row[2],
                "original_query_hash": prev_row[3],
                "step": prev_row[4],
                "step_index": prev_row[5],
                "input_canonical_hash": prev_row[6],
                "output_canonical_hash": prev_row[7],
                "faithfulness_score": prev_row[8],
                "confidence_score": prev_row[9],
                "decision": prev_row[10],
                "prev_record_hash": prev_row[11],
                "record_hmac": prev_row[12]
            }
            prev_record_hash = _compute_canonical_hash(prev_record_dict)

        # Prepare the current record fields for HMAC signing (excludes the hmac field itself)
        current_data = {
            "record_id": record_id,
            "session_id": session_id,
            "scrubbed_query_preview": scrubbed_query_preview,
            "original_query_hash": query_hash,
            "step": step,
            "step_index": step_index,
            "input_canonical_hash": input_hash,
            "output_canonical_hash": output_hash,
            "faithfulness_score": faithfulness_score,
            "confidence_score": confidence_score,
            "decision": decision,
            "prev_record_hash": prev_record_hash
        }
        
        # Sign the record
        hmac_key = get_hmac_key()
        record_hmac = _compute_record_hmac(current_data, hmac_key)
        
        # Write to DB. Let SQLite AUTOINCREMENT sequence atomically (Finding #63)
        with conn:
            conn.execute(
                """
                INSERT INTO audit_records (
                    record_id, session_id, scrubbed_query_preview, original_query_hash,
                    step, step_index, input_canonical_hash, output_canonical_hash,
                    faithfulness_score, confidence_score, decision, prev_record_hash, record_hmac
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id, session_id, scrubbed_query_preview, query_hash,
                    step, step_index, input_hash, output_hash,
                    faithfulness_score, confidence_score, decision, prev_record_hash, record_hmac
                )
            )
            
        logger.debug(f"Audit: Logged step '{step}' for session '{session_id}'. Seq hash linkage established.")
        
    except Exception as e:
        logger.error(f"Failed to log audit step to database: {e}")
        raise e
    finally:
        conn.close()


def verify_chain_integrity() -> list[str]:
    """
    Verifies the integrity of the audit chain database.
    Checks:
      1. HMAC signatures of every record (detects modification of data).
      2. Hash chain links (detects row insertions, deletions, or reordering).
      
    Returns:
        List of error description strings. Empty list means integrity is intact.
    """
    errors: list[str] = []
    
    initialize_audit_schema()
    conn = _connect_audit_db()
    
    try:
        hmac_key = get_hmac_key()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sequence, record_id, session_id, scrubbed_query_preview, original_query_hash,
                   step, step_index, input_canonical_hash, output_canonical_hash,
                   faithfulness_score, confidence_score, decision, prev_record_hash, record_hmac
            FROM audit_records
            ORDER BY sequence ASC
            """
        )
        rows = cursor.fetchall()
        
        expected_prev_hash = None
        for i, row in enumerate(rows):
            seq = row[0]
            record_id = row[1]
            session_id = row[2]
            scrubbed_preview = row[3]
            query_hash = row[4]
            step = row[5]
            step_index = row[6]
            input_hash = row[7]
            output_hash = row[8]
            faith_score = row[9]
            conf_score = row[10]
            decision = row[11]
            prev_record_hash = row[12]
            record_hmac = row[13]
            
            # Verify previous record hash link
            if prev_record_hash != expected_prev_hash:
                errors.append(
                    f"Chain Integrity Violation at seq {seq} (record_id={record_id}): "
                    f"expected prev_record_hash '{expected_prev_hash}', got '{prev_record_hash}'"
                )
                
            # Recompute and verify HMAC signature
            current_data = {
                "record_id": record_id,
                "session_id": session_id,
                "scrubbed_query_preview": scrubbed_preview,
                "original_query_hash": query_hash,
                "step": step,
                "step_index": step_index,
                "input_canonical_hash": input_hash,
                "output_canonical_hash": output_hash,
                "faithfulness_score": faith_score,
                "confidence_score": conf_score,
                "decision": decision,
                "prev_record_hash": prev_record_hash
            }
            computed_hmac = _compute_record_hmac(current_data, hmac_key)
            if computed_hmac != record_hmac:
                errors.append(
                    f"Signature Integrity Violation at seq {seq} (record_id={record_id}): "
                    "HMAC verification failed. Record has been altered."
                )
                
            # Save the current record dict hash as the expected next previous hash
            current_record_with_hmac = dict(current_data, record_hmac=record_hmac)
            expected_prev_hash = _compute_canonical_hash(current_record_with_hmac)
            
        if not errors:
            logger.info("Audit Chain verification passed: All database signatures and links intact.")
            
    except Exception as e:
        errors.append(f"Verification interrupted by error: {e}")
    finally:
        conn.close()
        
    return errors
