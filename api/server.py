"""
SENTINEL — FastAPI Server

Phase 7 implementation.
Local-only API server (127.0.0.1) integrating:
  - Correct route registration order (API -> Static -> SPA Catch-all) (Finding #31).
  - Rate limiting (slowapi) (Finding #46).
  - SSE streaming responses (sse-starlette) (Finding #22).
  - Resilient agentic loop invocation.
  - Background task ingestion and indexing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from api.auth import get_or_create_session_token, verify_session_token, reset_session_token
from sentinel.agent.ollama_client import ResilientOllamaClient
from sentinel.agent.loop import run_clinical_query
from sentinel.agent.session import session_manager
from sentinel.agent.escalation import get_unresolved_escalations, resolve_escalation
from sentinel.audit.chain import verify_chain_integrity, log_audit_step
from sentinel.audit.exporter import export_session_audit_json
from sentinel.config import CORPUS_DIR, DATA_DIR, LLM_MODEL, API_RATE_LIMIT, MNS_CONDITIONS
from sentinel.ingestion.parser import parse_pdf
from sentinel.ingestion.chunker import chunk_document
from sentinel.ingestion.embedder import ResilientEmbedder
from sentinel.ingestion.versioning import parse_version_metadata, resolve_supersession_logic
from sentinel.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Setup directories
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Ingestion status tracker (in-memory)
_ingest_state = {
    "status": "IDLE",      # IDLE | INGESTING | COMPLETED | FAILED
    "progress": 0,         # 0 to 100
    "current_file": None,
    "error": None
}

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize DB schema and perform warm-up call (O(1) ps check, Finding #56)
    logger.info("Starting SENTINEL API Server...")
    from sentinel.agent.escalation import initialize_escalation_schema
    from sentinel.audit.chain import initialize_audit_schema
    initialize_escalation_schema()
    initialize_audit_schema()
    
    # Check if LLM is loaded
    try:
        client = ResilientOllamaClient()
        # Non-blocking warm-up call in background
        asyncio.create_task(client.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "hello"}]))
        logger.info("Ollama warm-up task scheduled successfully.")
    except Exception as e:
        logger.warning(f"Ollama warm-up check failed: {e}")
        
    yield
    # Shutdown: Clean up thread pool and close handles
    logger.info("Shutting down SENTINEL API Server...")


# Setup Limiter (slowapi)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Background Ingestion Task ───────────────────────────────────────────────
async def _background_ingest(file_path: Path, filename: str) -> None:
    global _ingest_state
    _ingest_state["status"] = "INGESTING"
    _ingest_state["progress"] = 10
    _ingest_state["current_file"] = filename
    _ingest_state["error"] = None

    try:
        # 1. Parse PDF using Docling wrapper
        _ingest_state["progress"] = 25
        parsed_result = parse_pdf(file_path)
        if parsed_result is None:
            raise ValueError(f"Failed to parse document: {filename}")
        doc_dict, is_english = parsed_result

        # 2. Get versioning metadata
        _ingest_state["progress"] = 40
        version_meta = parse_version_metadata(filename)
        
        store = VectorStore.get_instance()
        existing_docs = store.get_all_document_metadata()
        
        # Check supersession
        is_superseded, docs_to_supersede = resolve_supersession_logic(version_meta, existing_docs)

        # 3. Chunker: Extract standard and decision tree chunks
        _ingest_state["progress"] = 60
        chunks = chunk_document(
            doc_dict=doc_dict,
            source_doc=filename,
            doc_version=version_meta["doc_version"],
            effective_date=version_meta["effective_date"],
            superseded=is_superseded
        )
        
        if not chunks:
            raise ValueError("No clinical content extracted from document.")

        # 4. Embed chunks
        _ingest_state["progress"] = 80
        embedder = ResilientEmbedder()
        texts_to_embed = [c["content"] for c in chunks]
        embeddings = await embedder.embed_chunks_batch(texts_to_embed)
        
        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        # 5. Insert into LanceDB
        _ingest_state["progress"] = 90
        store.add_chunks(chunks)
        
        # Mark superseded documents
        for old_doc in docs_to_supersede:
            store.mark_source_as_superseded(old_doc)
            
        # 6. Rebuild Indexes
        store.build_indexes()

        _ingest_state["progress"] = 100
        _ingest_state["status"] = "COMPLETED"
        logger.info(f"Ingestion pipeline completed for: {filename}")

    except Exception as e:
        logger.error(f"Background Ingestion Failed for {filename}: {e}")
        _ingest_state["status"] = "FAILED"
        _ingest_state["error"] = str(e)
    finally:
        # Remove temporary upload file if it wasn't saved in data/corpus/
        pass

# ── API Endpoint Routes ──────────────────────────────────────────────────────

@app.get("/auth/token")
def auth_token_endpoint():
    """Returns the browser sessionStorage isolation token (Finding #21)."""
    token = get_or_create_session_token()
    return {"token": token}


@app.post("/api/query")
@limiter.limit(API_RATE_LIMIT)
async def query_endpoint(request: Request, body: dict[str, Any], token: str = Depends(verify_session_token)):
    """
    Handles streaming clinical queries.
    Uses SSE via EventSourceResponse to yield real-time agent reasoning steps (Finding #22).
    """
    query_text = body.get("query", "").strip()
    session_id = body.get("session_id", "default_session").strip()
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    client = ResilientOllamaClient()

    async def sse_event_generator():
        try:
            # Yield initial connect event
            yield {"event": "status", "data": "CONNECTED"}
            
            # Run the agentic reasoning loop and yield metadata step events
            async for step_data in run_clinical_query(query_text, session_id, client):
                if "token" in step_data:
                    yield {"event": "token", "data": step_data["token"]}
                else:
                    # Metadata step (e.g. PHI_SCRUB, CRISIS_DETECT, RETRIEVAL)
                    yield {"event": step_data["step"], "data": str(step_data)}
                    
        except Exception as e:
            logger.error(f"Error in SSE event stream: {e}")
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(sse_event_generator())


@app.post("/api/ingest")
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    token: str = Depends(verify_session_token)
):
    """
    Accepts a WHO mhGAP PDF file upload and runs parsing/indexing in the background.
    """
    global _ingest_state
    if _ingest_state["status"] == "INGESTING":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ingestion pipeline is already running. Please wait for it to complete."
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    # Save to data/corpus/ folder
    dest_path = CORPUS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Queue background task
    background_tasks.add_task(_background_ingest, dest_path, file.filename)

    return {"message": "Ingestion started in background.", "filename": file.filename}


@app.get("/api/ingest/status")
def ingest_status_endpoint(token: str = Depends(verify_session_token)):
    """Polls the status of the background ingestion process."""
    return _ingest_state


@app.get("/api/corpus")
def corpus_metadata_endpoint(token: str = Depends(verify_session_token)):
    """Returns metadata for all indexed documents in LanceDB."""
    try:
        store = VectorStore.get_instance()
        docs = store.get_all_document_metadata()
        distribution = store.get_condition_distribution()
        return {"documents": docs, "condition_distribution": distribution}
    except Exception as e:
        logger.error(f"Failed to fetch corpus metadata: {e}")
        return {"documents": [], "condition_distribution": {}}


@app.get("/api/conditions")
def conditions_endpoint(token: str = Depends(verify_session_token)):
    """Returns the mhGAP condition codes taxonomy."""
    return {"conditions": MNS_CONDITIONS}


@app.get("/api/session/{session_id}/context")
def session_context_endpoint(session_id: str, token: str = Depends(verify_session_token)):
    """Returns the multi-turn discussion context history for a session."""
    session = session_manager.get_session(session_id)
    history = [
        {
            "query": turn.query,
            "answer": turn.answer,
            "condition_codes": turn.condition_codes,
            "timestamp": turn.timestamp
        }
        for turn in session.turns
    ]
    return {"session_id": session_id, "turns": history}


@app.delete("/api/session/{session_id}")
def clear_session_endpoint(session_id: str, token: str = Depends(verify_session_token)):
    """Deletes a session's history."""
    session_manager.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared successfully."}


@app.get("/api/audit/verify")
def audit_verify_endpoint(token: str = Depends(verify_session_token)):
    """Triggers verification of the audit database integrity."""
    errors = verify_chain_integrity()
    if errors:
        return JSONResponse(
            status_code=400,
            content={"verified": False, "errors": errors}
        )
    return {"verified": True, "message": "All signatures and hash links intact."}


@app.get("/api/audit/export/{session_id}")
def audit_export_endpoint(session_id: str, token: str = Depends(verify_session_token)):
    """Exports session audits as JSON."""
    try:
        report = export_session_audit_json(session_id)
        # Load string back to return as raw JSON object in FastAPI
        import json
        return json.loads(report)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No audit logs found: {e}")


@app.get("/api/escalations")
def escalations_endpoint(token: str = Depends(verify_session_token)):
    """Returns the unresolved escalation queue."""
    queue = get_unresolved_escalations()
    return {"escalations": queue}


@app.post("/api/escalations/{escalation_id}/resolve")
def resolve_escalation_endpoint(escalation_id: str, body: dict[str, str], token: str = Depends(verify_session_token)):
    """Resolves an escalation with note."""
    note = body.get("resolution", "").strip()
    resolve_escalation(escalation_id, note)
    return {"message": f"Escalation '{escalation_id}' resolved."}


@app.get("/api/status")
def system_status_endpoint():
    """Returns system health, index stats, and queue depth."""
    try:
        # Check Vector Store
        store = VectorStore.get_instance()
        total_rows = store._table.count_rows()
        
        # Check LLM
        import urllib.request
        import json as json_lib
        req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            models_data = json_lib.loads(resp.read().decode())
            
        # Get active memory models
        req_ps = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        with urllib.request.urlopen(req_ps, timeout=3) as resp_ps:
            ps_data = json_lib.loads(resp_ps.read().decode())
            
        ollama_status = "HEALTHY"
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        ollama_status = "UNHEALTHY"
        total_rows = 0
        ps_data = {}

    return {
        "ollama": ollama_status,
        "index_rows": total_rows,
        "loaded_models": ps_data.get("models", []),
        "escalations_pending": len(get_unresolved_escalations())
    }

# ── Static File Registration (Correct Ordering) ──────────────────────────────
# 1. API routes FIRST (registered above)
# 2. Static assets SECOND
# 3. SPA catch-all LAST
# Create dashboard static dir if missing
Path("dashboard/static").mkdir(parents=True, exist_ok=True)
if not Path("dashboard/index.html").exists():
    with open("dashboard/index.html", "w") as f:
        f.write("<!-- SENTINEL Dashboard -->")

app.mount("/static", StaticFiles(directory="dashboard/static"), name="static")

@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Catch-all router to serve Single Page Application (SPA)."""
    return FileResponse("dashboard/index.html")
