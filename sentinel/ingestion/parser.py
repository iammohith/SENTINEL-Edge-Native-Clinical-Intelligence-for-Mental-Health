"""
SENTINEL — PDF Parser (Docling wrapper with timeout guard)

Phase 2 implementation.
"""

from __future__ import annotations

import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Optional, Any

from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError

logger = logging.getLogger(__name__)


def _detect_language_is_english(text: str) -> bool:
    """
    Offline English language detection heuristic based on common stopword ratio.
    Avoids introducing heavy offline translation/classification models.
    """
    common_english = {
        "the", "and", "of", "to", "in", "is", "that", "it", "with", "for",
        "as", "was", "on", "at", "by", "an", "be", "this", "are", "from",
        "or", "have", "not", "but", "what", "all", "were", "we", "your", "can"
    }
    words = [w.strip(",.?!()[]{}:;\"'").lower() for w in text.split()]
    words = [w for w in words if w.isalpha()]
    if not words:
        return True  # Default to True for empty/non-text documents
    
    english_word_count = sum(1 for w in words if w in common_english)
    ratio = english_word_count / len(words)
    return ratio > 0.05  # Standard English documents have >10% stopwords; 5% is safe threshold


def _run_docling_conversion(file_path: str, conn: multiprocessing.connection.Connection) -> None:
    """Runs in a separate process to allow hard timeout termination."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        
        # Disable OCR and force CPU to prevent PyTorch MPS hang on macOS
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )
        result = converter.convert(file_path)
        # Check language on a snippet of the parsed text
        text_snippet = result.document.export_to_markdown()[:2000]
        is_english = _detect_language_is_english(text_snippet)
        
        # We return the dict representation of the DoclingDocument to ensure
        # safe pickling across process boundaries.
        doc_dict = result.document.export_to_dict()
        conn.send(("success", (doc_dict, is_english)))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def parse_pdf(file_path: Path, timeout: float = 900.0) -> Optional[tuple[dict[str, Any], bool]]:
    """
    Parses a PDF using Docling with a strict 300s timeout guard.
    Runs the conversion in a separate process to prevent CPU hangs on corrupted files.

    Args:
        file_path: Absolute Path to the PDF file.
        timeout: Maximum seconds allowed for conversion.

    Returns:
        A tuple of (doc_dict, is_english) if successful, or None on failure/timeout.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(
        target=_run_docling_conversion,
        args=(str(file_path), child_conn)
    )
    
    logger.info(f"Parsing PDF {file_path.name} with {timeout}s timeout...")
    process.start()
    child_conn.close()  # Close child end in parent process

    # To prevent deadlock when the child sends a large pickled object over the Pipe,
    # we poll and recv from the connection before joining.
    import time
    start_t = time.time()
    pipe_result = None
    
    while process.is_alive():
        if parent_conn.poll(0.1):
            try:
                pipe_result = parent_conn.recv()
            except EOFError:
                pass
            break
        if time.time() - start_t > timeout:
            break
            
    # If we broke because we got a result, give the process up to 5 seconds to finish exiting
    if pipe_result is not None and process.is_alive():
        process.join(5.0)
            
    # Clean up process
    if process.is_alive():
        logger.error(f"TimeoutError: Parsing {file_path.name} exceeded {timeout}s limit. Terminating process.")
        process.terminate()
        process.join()
        parent_conn.close()
        return None
        
    process.join()

    # If we didn't receive a result during the loop, check the pipe one last time
    if pipe_result is None and parent_conn.poll():
        try:
            pipe_result = parent_conn.recv()
        except EOFError:
            pass

    parent_conn.close()

    if pipe_result:
        status, result = pipe_result
        if status == "success":
            doc_dict, is_english = result
            if not is_english:
                logger.warning(
                    f"Language warning: {file_path.name} does not appear to be in English. "
                    "Retrieval accuracy may be degraded (v1 embeds are English-only)."
                )
            return doc_dict, is_english
        else:
            logger.error(f"DoclingParseError parsing {file_path.name}: {result}")
            return None
    else:
        logger.error(f"DoclingParseError parsing {file_path.name}: No data returned from process")
        return None
