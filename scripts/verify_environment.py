"""
SENTINEL — Environment Verification Script (v6)

Runs pre-flight checks before starting the server.
Refuses to proceed if any critical check fails.

Usage:
    uv run python scripts/verify_environment.py

Exit codes:
    0  — All checks passed
    1  — One or more critical checks failed
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable

# ── Colours ────────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANCEDB_INDEX_DIR = PROJECT_ROOT / "data" / "index"
AUDIT_DB_PATH = PROJECT_ROOT / "audit_logs" / "sentinel_audit.db"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str, fix: str = "") -> None:
    print(f"  {RED}✗{RESET} {msg}")
    if fix:
        print(f"    {YELLOW}FIX:{RESET} {fix}")
    failures.append(msg)


def warn(msg: str, fix: str = "") -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")
    if fix:
        print(f"    {YELLOW}FIX:{RESET} {fix}")
    warnings.append(msg)


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 50)


# ── Check 1: OLLAMA_KEEP_ALIVE ─────────────────────────────────────────────────
section("1. Ollama Configuration")

keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE", "")
if keep_alive == "-1":
    ok("OLLAMA_KEEP_ALIVE=-1 is set (model will not unload after idle)")
else:
    warn(
        f"OLLAMA_KEEP_ALIVE is '{keep_alive}' — model will unload after 5 min idle",
        fix="export OLLAMA_KEEP_ALIVE=-1 && ollama serve",
    )


# ── Check 2: Ollama service reachable ──────────────────────────────────────────
def ollama_api(path: str) -> dict | None:
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}{path}")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


section("2. Ollama Models")

ollama_running = ollama_api("/api/tags")
if ollama_running is None:
    fail(
        "Ollama service not reachable at http://127.0.0.1:11434",
        fix="ollama serve   (and set OLLAMA_KEEP_ALIVE=-1 in the same shell)",
    )
else:
    loaded_models = {m["name"] for m in (ollama_running.get("models") or [])}

    for model_name in ("gemma4:e4b", "nomic-embed-text"):
        # Check if model is in Ollama's local registry
        matched = any(model_name in m for m in loaded_models)
        if matched:
            ok(f"Model '{model_name}' available locally")
        else:
            fail(
                f"Model '{model_name}' not found",
                fix=f"ollama pull {model_name}",
            )

    # Check if gemma4:e4b is currently loaded in memory (skip warm-up if already hot)
    ps_data = ollama_api("/api/ps")
    running_models = {m["name"] for m in ((ps_data or {}).get("models") or [])}
    gemma_hot = any("gemma4:e4b" in m for m in running_models)

    if gemma_hot:
        ok("gemma4:e4b is already loaded in memory — skipping warm-up")
    else:
        warn(
            "gemma4:e4b is not currently loaded in memory",
            fix="Server lifespan warm-up will load it on first request. "
                "Or run: ollama run gemma4:e4b 'OK' --nowordwrap",
        )


# ── Check 3: ML Model Weights ───────────────────────────────────────────────────
section("3. ML Model Weights (HuggingFace Cache)")

hf_cache = Path.home() / ".cache" / "huggingface"

for model_id, hint in [
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "Reranker"),
    ("cross-encoder/nli-MiniLM2-L6-H768", "NLI faithfulness model"),
]:
    # HuggingFace caches under snapshots/<hash>/
    model_dir_name = model_id.replace("/", "--")
    cached_path = hf_cache / "hub" / f"models--{model_dir_name}"
    if cached_path.exists():
        ok(f"{hint}: cached ({model_id})")
    else:
        fail(
            f"{hint} not cached: {model_id}",
            fix=f"Run scripts/bootstrap_offline.sh to download",
        )

# scispaCy
try:
    import spacy  # noqa: F401
    nlp = None
    try:
        import en_core_sci_sm  # type: ignore[import-not-found]  # noqa: F401
        ok("scispaCy en_core_sci_sm available (clinical sentence tokenizer)")
    except ImportError:
        warn(
            "scispaCy en_core_sci_sm not installed — will fall back to NLTK sent_tokenize",
            fix="pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz",
        )
except ImportError:
    warn("spacy not installed", fix="pip install spacy")

# Presidio
try:
    from presidio_analyzer import AnalyzerEngine  # noqa: F401
    ok("presidio-analyzer available (offline PHI detection)")
except ImportError:
    fail(
        "presidio-analyzer not installed",
        fix="pip install presidio-analyzer presidio-anonymizer",
    )


# ── Check 4: LanceDB Index ──────────────────────────────────────────────────────
section("4. LanceDB Index")

if not LANCEDB_INDEX_DIR.exists():
    warn(
        f"LanceDB index directory not found: {LANCEDB_INDEX_DIR}",
        fix="Run: sentinel ingest <pdf_path>   to ingest your first document",
    )
else:
    try:
        import lancedb  # type: ignore[import-not-found]

        db = lancedb.connect(str(LANCEDB_INDEX_DIR))
        table_names = db.table_names()

        if not table_names:
            warn(
                "LanceDB index is empty — no tables found",
                fix="Ingest at least one WHO mhGAP PDF: sentinel ingest <path>",
            )
        else:
            for tname in table_names:
                tbl = db.open_table(tname)
                row_count = tbl.count_rows()
                try:
                    stats = tbl.index_stats("embedding")
                    if stats is not None:
                        unindexed = stats.num_unindexed_rows
                        if unindexed == 0:
                            ok(f"Table '{tname}': {row_count} rows, ANN index current (0 unindexed)")
                        else:
                            warn(
                                f"Table '{tname}': {unindexed} unindexed rows — ANN index stale",
                                fix="Run: sentinel ingest --reindex   to rebuild",
                            )
                    else:
                        # Fallback: check list_indices
                        indices = tbl.list_indices()
                        has_vector_idx = any("embedding" in getattr(idx, "columns", []) for idx in indices)
                        if has_vector_idx:
                            ok(f"Table '{tname}': {row_count} rows, ANN index built (stats unavailable)")
                        else:
                            warn(
                                f"Table '{tname}': ANN index not built yet",
                                fix="Run ingestion pipeline to build index",
                            )
                except Exception:
                    warn(
                        f"Table '{tname}': ANN index not built yet",
                        fix="Run ingestion pipeline to build index",
                    )
    except ImportError:
        fail("lancedb not installed", fix="uv pip install lancedb>=0.8.0,<0.9.0")
    except Exception as e:
        fail(f"LanceDB connection error: {e}")


# ── Check 5: OS Keychain (HMAC + SQLCipher keys) ────────────────────────────────
section("5. OS Keychain (Audit Keys)")

try:
    import keyring  # noqa: F401

    for service, key_name, description in [
        ("sentinel-audit", "hmac-signing-key", "HMAC signing key"),
        ("sentinel-audit", "sqlcipher-key", "SQLCipher encryption key"),
    ]:
        try:
            value = keyring.get_password(service, key_name)
            if value:
                ok(f"{description} found in OS Keychain")
            else:
                fail(
                    f"{description} not found in OS Keychain",
                    fix="Run: uv run python scripts/audit_key_init.py",
                )
        except Exception as e:
            fail(f"Keychain access error for {description}: {e}")
except ImportError:
    fail("keyring not installed", fix="uv pip install keyring")


# ── Check 6: SQLCipher Audit Database ───────────────────────────────────────────
section("6. SQLCipher Audit Database")

if not AUDIT_DB_PATH.exists():
    warn(
        f"Audit database not found: {AUDIT_DB_PATH}",
        fix="It will be created automatically on first run after key init",
    )
else:
    try:
        from sqlcipher3 import dbapi2 as sqlcipher  # type: ignore[import-not-found]
        import keyring as kr  # type: ignore[import-not-found]
        import base64

        key_b64 = kr.get_password("sentinel-audit", "sqlcipher-key")
        if key_b64:
            key = base64.b64decode(key_b64).hex()
            conn = sqlcipher.connect(str(AUDIT_DB_PATH))
            conn.execute(f"PRAGMA key = \"x'{key}'\"")
            wal_mode = conn.execute("PRAGMA journal_mode").fetchone()
            conn.close()

            if wal_mode and wal_mode[0].upper() == "WAL":
                ok("SQLCipher audit database accessible; WAL mode confirmed")
            else:
                warn("SQLite WAL mode not enabled on audit database")
        else:
            warn("Cannot verify audit DB — SQLCipher key not in Keychain")
    except ImportError:
        fail(
            "sqlcipher3 not installed — at-rest encryption unavailable",
            fix="pip install sqlcipher3   (requires: brew install sqlcipher on macOS)",
        )
    except Exception as e:
        fail(f"Audit database error: {e}")


# ── Summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*52}")
if failures:
    print(f"{RED}{BOLD}  ✗ {len(failures)} critical check(s) failed — server will not start{RESET}")
    for f in failures:
        print(f"    • {f}")
    print()
    if warnings:
        print(f"{YELLOW}  ⚠ {len(warnings)} warning(s):{RESET}")
        for w in warnings:
            print(f"    • {w}")
    print(f"{'='*52}\n")
    sys.exit(1)
elif warnings:
    print(f"{YELLOW}{BOLD}  ⚠ All critical checks passed ({len(warnings)} warning(s)){RESET}")
    for w in warnings:
        print(f"    • {w}")
    print(f"{'='*52}\n")
    sys.exit(0)
else:
    print(f"{GREEN}{BOLD}  ✓ All checks passed — SENTINEL is ready{RESET}")
    print(f"{'='*52}\n")
    sys.exit(0)
