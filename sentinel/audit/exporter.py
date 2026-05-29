"""
SENTINEL — Audit Session Exporter

Phase 6 implementation.
Retrieves and exports all audit logs associated with a particular session_id
from the SQLCipher database as a standardized, formatted JSON report.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sentinel.audit.chain import _connect_audit_db

logger = logging.getLogger(__name__)


def export_session_audit_json(session_id: str) -> str:
    """
    Retrieves all audit steps for the session from the encrypted database
    and formats them as a pretty-printed, de-identified JSON string.
    """
    conn = _connect_audit_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT sequence, record_id, timestamp, scrubbed_query_preview,
                   original_query_hash, step, step_index, input_canonical_hash,
                   output_canonical_hash, faithfulness_score, confidence_score,
                   decision, prev_record_hash, record_hmac
            FROM audit_records
            WHERE session_id = ?
            ORDER BY sequence ASC
            """,
            (session_id,)
        )
        rows = cursor.fetchall()
        
        if not rows:
            logger.warning(f"Exporter: No audit records found for session_id '{session_id}'")
            return json.dumps({"session_id": session_id, "records": []})

        records = []
        for r in rows:
            records.append({
                "sequence": r[0],
                "record_id": r[1],
                "timestamp": r[2],
                "scrubbed_query_preview": r[3],
                "original_query_hash": r[4],
                "step": r[5],
                "step_index": r[6],
                "input_canonical_hash": r[7],
                "output_canonical_hash": r[8],
                "faithfulness_score": r[9],
                "confidence_score": r[10],
                "decision": r[11],
                "prev_record_hash": r[12],
                "record_hmac": r[13]
            })

        export_data = {
            "session_id": session_id,
            "export_timestamp": r[2],  # Use latest timestamp
            "total_records": len(records),
            "records": records
        }
        
        logger.info(f"Exporter: Exported {len(records)} audit records for session '{session_id}'.")
        return json.dumps(export_data, indent=2)
        
    except Exception as e:
        logger.error(f"Failed to export audit logs: {e}")
        raise e
    finally:
        conn.close()
