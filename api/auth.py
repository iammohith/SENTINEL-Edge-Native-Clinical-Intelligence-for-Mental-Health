"""
SENTINEL — API Authentication and Token Helper

Phase 7 implementation.
Generates cryptographically secure session tokens stored in-memory (isolated to the tab
via browser sessionStorage to prevent localStorage cross-tab leakage, Finding #21).
"""

from __future__ import annotations

import logging
import secrets
import threading
from typing import Optional

from fastapi import Depends, HTTPException, Header, status

logger = logging.getLogger(__name__)

# Active session token stored in memory (wiped when server process restarts)
_active_session_token: Optional[str] = None
_token_lock = threading.Lock()


def get_or_create_session_token() -> str:
    """Generates the active session token on first load or returns the existing one."""
    global _active_session_token
    import threading
    if _active_session_token is None:
        with _token_lock:
            if _active_session_token is None:
                # Generate 256-bit cryptographically secure token
                _active_session_token = secrets.token_hex(32)
                logger.info("New secure session token generated for clinician interface.")
    return _active_session_token


def reset_session_token() -> str:
    """Regenerates a new session token, invalidating the previous one."""
    global _active_session_token
    import threading
    with _token_lock:
        _active_session_token = secrets.token_hex(32)
        logger.info("Session token has been rotated.")
    return _active_session_token


async def verify_session_token(x_session_token: Optional[str] = Header(None)) -> str:
    """
    FastAPI dependency injection to verify the session token in the header.
    Rejects requests with HTTP 401 Unauthorized if mismatch.
    """
    active_token = get_or_create_session_token()
    if not x_session_token or x_session_token != active_token:
        logger.warning("Unauthorized API access attempt: invalid or missing session token.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing session token. Please reload the dashboard."
        )
    return x_session_token

import threading  # Ensure imported in module namespace
