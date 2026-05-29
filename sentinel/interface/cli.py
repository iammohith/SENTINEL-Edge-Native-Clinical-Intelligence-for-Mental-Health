"""
SENTINEL — Administrator CLI (Rich-based)

Phase 9 implementation.
Provides terminal CLI utilities for managing the clinical intelligence node,
running evaluation sets, and checking database integrity.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger("sentinel.cli")

app = typer.Typer(help="SENTINEL Node Management and Evaluation Command Line Utility")
console = Console()


@app.command("status")
def status_cmd():
    """Outputs VectorStore and LLM health status."""
    from sentinel.store.vector_store import VectorStore
    
    console.print("[bold teal]SENTINEL Status Summary[/bold teal]")
    console.print("─" * 40)
    
    try:
        store = VectorStore.get_instance()
        rows = store._table.count_rows()
        dist = store.get_condition_distribution()
        
        console.print(f"LanceDB Table Rows:  [bold green]{rows}[/bold green]")
        console.print("Condition Distribution:")
        for code, count in dist.items():
            console.print(f"  • {code}: {count} chunks")
            
    except Exception as e:
        console.print(f"[bold red]Failed to read VectorStore status: {e}[/bold red]")


@app.command("verify-env")
def verify_env_cmd():
    """Runs pre-flight environment checks."""
    import subprocess
    console.print("Running pre-flight checks...")
    # Call verify_environment.py script directly
    res = subprocess.run(["python3", "scripts/verify_environment.py"])
    if res.returncode == 0:
        console.print("[bold green]✓ Environment verified.[/bold green]")
    else:
        console.print("[bold red]✗ Environment checks failed.[/bold red]")


@app.command("ingest")
def ingest_cmd(
    pdf_path: Path = typer.Argument(..., help="Path to the PDF file to ingest"),
    check_version: bool = typer.Option(False, "--check-version", help="Verify document version without inserting")
):
    """Parses, chunks, embeds, and indexes a WHO mhGAP PDF guideline."""
    from sentinel.ingestion.parser import parse_pdf
    from sentinel.ingestion.versioning import parse_version_metadata, resolve_supersession_logic
    from sentinel.ingestion.chunker import chunk_document
    from sentinel.ingestion.embedder import ResilientEmbedder
    from sentinel.store.vector_store import VectorStore
    
    if not pdf_path.exists():
        console.print(f"[bold red]File not found: {pdf_path}[/bold red]")
        raise typer.Exit(code=1)
        
    filename = pdf_path.name
    meta = parse_version_metadata(filename)
    console.print(f"Document Metadata: Version=[green]{meta['doc_version']}[/green], Date=[green]{meta['effective_date']}[/green], Type=[green]{meta['document_type']}[/green]")
    
    store = VectorStore.get_instance()
    existing = store.get_all_document_metadata()
    
    is_superseded, docs_to_supersede = resolve_supersession_logic(meta, existing)
    
    if check_version:
        console.print(f"Check-version only. Superseded status: {is_superseded}. Documents it would replace: {docs_to_supersede}")
        return
        
    # Execute Ingestion
    console.print("Parsing document (Docling)...")
    parsed = parse_pdf(pdf_path)
    if not parsed:
        console.print("[bold red]Failed to parse document.[/bold red]")
        raise typer.Exit(code=1)
        
    doc_dict, is_english = parsed
    
    console.print("Chunking document paths and decision trees...")
    chunks = chunk_document(
        doc_dict=doc_dict,
        source_doc=filename,
        doc_version=meta["doc_version"],
        effective_date=meta["effective_date"],
        superseded=is_superseded
    )
    
    if not chunks:
        console.print("[bold red]No chunks generated.[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"Generating embeddings for {len(chunks)} chunks (Ollama)...")
    embedder = ResilientEmbedder()
    
    async def run_embedding():
        texts = [c["content"] for c in chunks]
        return await embedder.embed_chunks_batch(texts)
        
    embeddings = asyncio.run(run_embedding())
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
        
    console.print("Writing to LanceDB database and rebuilding indexes...")
    store.add_chunks(chunks)
    
    for old_doc in docs_to_supersede:
        store.mark_source_as_superseded(old_doc)
        
    store.build_indexes()
    
    console.print(f"[bold green]✓ Successfully ingested '{filename}'![/bold green]")


@app.command("query")
def query_cmd(
    query_text: str = typer.Argument(..., help="Query to run against the node"),
    session_id: str = typer.Option("cli_session", "--session-id", help="Session ID for conversation history")
):
    """Runs a full clinical consultation reasoning query, printing outputs to console."""
    from sentinel.agent.ollama_client import ResilientOllamaClient
    from sentinel.agent.loop import run_clinical_query

    client = ResilientOllamaClient()
    
    async def run_query():
        console.print(f"[bold teal]Executing RAG reasoning loop...[/bold teal]\n")
        
        async for step in run_clinical_query(query_text, session_id, client):
            if "token" in step:
                # Print tokens live
                print(step["token"], end="", flush=True)
            elif "step" in step:
                if step["status"] == "START":
                    console.print(f"\n[dim][bold yellow]➔[/bold yellow] Running: {step['step']}...[/dim]", end="", flush=True)
                elif step["status"] == "COMPLETE":
                    console.print(" [bold green]✓[/bold green]")
                    if step["step"] == "RETRIEVAL" and "citations" in step:
                        console.print(f"   Citations: {len(step['citations'])} source chunks found.")
                elif step["status"] == "ESCALATED":
                    console.print(f" [bold red]✗ Escalated! ID: {step.get('escalation_id')}[/bold red]")
                    
        print() # Line break

    asyncio.run(run_query())


@app.command("audit-verify")
def audit_verify_cmd():
    """Checks the cryptographic integrity of the SQLCipher audit ledger."""
    from sentinel.audit.chain import verify_chain_integrity
    
    console.print("Verifying audit chain signature integrity...")
    errors = verify_chain_integrity()
    if errors:
        console.print("[bold red]⛔ Ledger integrity compromised![/bold red]")
        for err in errors:
            console.print(f"  • {err}")
        raise typer.Exit(code=1)
    else:
        console.print("[bold green]✓ Audit Ledger verified successfully. Zero tampering detected.[/bold green]")


@app.command("eval")
def eval_cmd(
    golden_path: Path = typer.Option("eval/golden_dataset.json", "--golden", help="Path to golden QA dataset"),
    crisis_path: Path = typer.Option("eval/crisis_test_cases.json", "--crisis", help="Path to crisis test cases")
):
    """Executes the automated evaluation harness against test datasets."""
    import subprocess
    console.print("Invoking evaluation harness...")
    res = subprocess.run(["python3", "eval/run_eval.py", "--golden", str(golden_path), "--crisis", str(crisis_path)])
    if res.returncode == 0:
        console.print("[bold green]✓ Evaluation completed successfully.[/bold green]")
    else:
        console.print("[bold red]✗ Evaluation run failed.[/bold red]")


if __name__ == "__main__":
    app()
