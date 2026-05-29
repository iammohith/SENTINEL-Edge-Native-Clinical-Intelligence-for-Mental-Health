"""
SENTINEL — Clinical Escalation Handler

Phase 5 implementation.
Handles clinical escalations when:
  - Intent classification fails.
  - Confidence score falls below the threshold (0.70).
  - Faithfulness NLI checks detect a contradiction or ungrounded claims.
  - Out of scope topics are queried.

Writes de-identified (PHI-free) records to the SQLCipher audit database.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from sentinel.audit.key_manager import get_sqlcipher_key
from sentinel.config import AUDIT_DB_PATH

logger = logging.getLogger(__name__)


@dataclass
class EscalationRecord:
    escalation_id: str
    session_id: str
    scrubbed_query: str
    reason: str
    timestamp: str
    resolved: bool = False
    resolution: str = ""


def _connect_audit_db() -> sqlite3.Connection:
    """Connects to the SQLCipher audit database using the keychain key."""
    # We use sqlcipher3 dbapi2
    from sqlcipher3 import dbapi2 as sqlcipher
    
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlcipher.connect(str(AUDIT_DB_PATH))
    
    # Authenticate via SQLCipher key stored in Keychain (SQLCipher requirement: PRAGMA key first)
    try:
        key = get_sqlcipher_key()
        # SQLCipher expects the hex key format: PRAGMA key = "x'HEX_KEY'"
        conn.execute(f"PRAGMA key = \"x'{key}'\"")
    except Exception as e:
        logger.error(f"Failed to authenticate SQLCipher audit DB: {e}")
        conn.close()
        raise e
        
    # Enable WAL mode and busy timeout (Finding #30 / #67)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    
    return conn


def initialize_escalation_schema() -> None:
    """Initializes the escalations table schema if it does not exist."""
    conn = _connect_audit_db()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    escalation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    scrubbed_query TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT
                )
            """)
            logger.info("Escalation database schema initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize escalation schema: {e}")
    finally:
        conn.close()


def escalate_query(session_id: str, scrubbed_query: str, reason: str) -> str:
    """
    Escalates a clinical query by logging it in the escalation queue.
    Only stores the PHI-free scrubbed query preview to ensure compliance.
    
    Returns:
        The generated escalation ID.
    """
    # Ensure schema exists
    initialize_escalation_schema()
    
    escalation_id = str(uuid.uuid4())
    conn = _connect_audit_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO escalations (escalation_id, session_id, scrubbed_query, reason)
                VALUES (?, ?, ?, ?)
                """,
                (escalation_id, session_id, scrubbed_query[:500], reason)  # Cap size to be safe
            )
        logger.warning(f"CLINICAL ESCALATION: Session '{session_id}' escalated. ID: {escalation_id}. Reason: {reason}")
        return escalation_id
    except Exception as e:
        logger.error(f"Failed to log escalation: {e}")
        return escalation_id
    finally:
        conn.close()


def get_unresolved_escalations() -> list[dict[str, Any]]:
    """Retrieves all unresolved escalations from the queue."""
    initialize_escalation_schema()
    conn = _connect_audit_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT escalation_id, session_id, scrubbed_query, reason, timestamp
            FROM escalations
            WHERE resolved = 0
            ORDER BY timestamp DESC
            """
        )
        rows = cursor.fetchall()
        return [
            {
                "escalation_id": r[0],
                "session_id": r[1],
                "scrubbed_query": r[2],
                "reason": r[3],
                "timestamp": r[4]
            }
            for r in rows
        ]
    except Exception as e:
        logger.error(f"Failed to fetch escalations: {e}")
        return []
    finally:
        conn.close()


def resolve_escalation(escalation_id: str, resolution_notes: str) -> None:
    """Marks an escalation as resolved with notes."""
    conn = _connect_audit_db()
    try:
        with conn:
            conn.execute(
                """
                UPDATE escalations
                SET resolved = 1, resolution = ?
                WHERE escalation_id = ?
                """,
                (resolution_notes, escalation_id)
            )
        logger.info(f"Escalation '{escalation_id}' resolved.")
    except Exception as e:
        logger.error(f"Failed to resolve escalation '{escalation_id}': {e}")
    finally:
        conn.close()
