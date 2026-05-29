"""
SENTINEL — Audit Key Manager

Phase 6 implementation.
Safely retrieves the HMAC signing key and SQLCipher database encryption key
from the OS Keychain (macOS Keychain / Linux SecretService) via keyring (Finding #5 / #48).
"""

from __future__ import annotations

import base64
import logging

import keyring

from sentinel.config import (
    AUDIT_HMAC_KEY_NAME,
    AUDIT_SERVICE_NAME,
    AUDIT_SQLCIPHER_KEY_NAME,
)

logger = logging.getLogger(__name__)


def get_hmac_key() -> bytes:
    """
    Retrieves and base64-decodes the HMAC signing key from the OS Keychain.
    Raises RuntimeError if the key is missing.
    """
    raw_b64 = keyring.get_password(AUDIT_SERVICE_NAME, AUDIT_HMAC_KEY_NAME)
    if not raw_b64:
        raise RuntimeError(
            f"Critical Error: HMAC signing key is missing from OS Keychain "
            f"(service='{AUDIT_SERVICE_NAME}', account='{AUDIT_HMAC_KEY_NAME}'). "
            "Please run: uv run python scripts/audit_key_init.py"
        )
    return base64.b64decode(raw_b64)


def get_sqlcipher_key() -> str:
    """
    Retrieves the SQLCipher encryption key from the OS Keychain,
    decodes it, and returns the 64-character hex string format for SQLCipher PRAGMA key usage.
    Raises RuntimeError if the key is missing.
    """
    raw_b64 = keyring.get_password(AUDIT_SERVICE_NAME, AUDIT_SQLCIPHER_KEY_NAME)
    if not raw_b64:
        raise RuntimeError(
            f"Critical Error: SQLCipher database key is missing from OS Keychain "
            f"(service='{AUDIT_SERVICE_NAME}', account='{AUDIT_SQLCIPHER_KEY_NAME}'). "
            "Please run: uv run python scripts/audit_key_init.py"
        )
    raw_bytes = base64.b64decode(raw_b64)
    # SQLCipher expects hex key format: "x'HEX_STRING'"
    return raw_bytes.hex()
