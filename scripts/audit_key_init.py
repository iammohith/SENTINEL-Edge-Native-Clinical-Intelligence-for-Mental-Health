"""
SENTINEL — Audit Key Initialization Script

Generates HMAC signing key and SQLCipher encryption key,
stores both in the OS Keychain (macOS Keychain / Linux SecretService).

Keys NEVER touch disk or environment variables.

Usage:
    uv run python scripts/audit_key_init.py

Re-running will prompt for confirmation before overwriting existing keys.
"""

from __future__ import annotations

import base64
import os
import secrets
import sys

try:
    import keyring
except ImportError:
    print("ERROR: keyring not installed. Run: uv pip install keyring")
    sys.exit(1)

SERVICE_NAME = "sentinel-audit"
HMAC_KEY_NAME = "hmac-signing-key"
SQLCIPHER_KEY_NAME = "sqlcipher-key"


def _generate_key_b64(n_bytes: int = 32) -> str:
    """Generate cryptographically secure random key, base64-encoded."""
    return base64.b64encode(secrets.token_bytes(n_bytes)).decode()


def _key_exists(key_name: str) -> bool:
    try:
        return keyring.get_password(SERVICE_NAME, key_name) is not None
    except Exception:
        return False


def main() -> None:
    print("SENTINEL — Audit Key Initialization")
    print("=" * 44)
    print()
    print(f"Service: {SERVICE_NAME}")
    print(f"Keys:    {HMAC_KEY_NAME}, {SQLCIPHER_KEY_NAME}")
    print()

    hmac_exists = _key_exists(HMAC_KEY_NAME)
    cipher_exists = _key_exists(SQLCIPHER_KEY_NAME)

    if hmac_exists or cipher_exists:
        print("⚠  Existing keys detected in OS Keychain:")
        if hmac_exists:
            print(f"   • {HMAC_KEY_NAME} — EXISTS")
        if cipher_exists:
            print(f"   • {SQLCIPHER_KEY_NAME} — EXISTS")
        print()
        print("Overwriting keys will make existing audit logs unverifiable.")
        response = input("Type 'OVERWRITE' to replace existing keys, or press Enter to cancel: ")
        if response.strip() != "OVERWRITE":
            print("Cancelled. Existing keys preserved.")
            sys.exit(0)
        print()

    # Generate and store HMAC signing key (32 bytes = 256-bit)
    hmac_key = _generate_key_b64(32)
    keyring.set_password(SERVICE_NAME, HMAC_KEY_NAME, hmac_key)
    print(f"  ✓ HMAC signing key (256-bit) stored in OS Keychain")

    # Generate and store SQLCipher encryption key (32 bytes = 256-bit)
    sqlcipher_key = _generate_key_b64(32)
    keyring.set_password(SERVICE_NAME, SQLCIPHER_KEY_NAME, sqlcipher_key)
    print(f"  ✓ SQLCipher encryption key (256-bit) stored in OS Keychain")

    print()
    print("Both keys are in the OS Keychain under service 'sentinel-audit'.")
    print("They are never written to disk or environment variables.")
    print()
    print("Next step:")
    print("  uv run python scripts/verify_environment.py")


if __name__ == "__main__":
    main()
