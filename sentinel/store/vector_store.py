"""
SENTINEL — LanceDB Vector Store (Single Store for ANN + FTS BM25)

Phase 2 implementation.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, ClassVar, Optional

import lancedb
import pyarrow as pa

from sentinel.config import (
    EMBEDDING_DIM,
    INDEX_DIR,
    LANCEDB_TABLE_NAME,
    MIN_VECTORS_PER_PARTITION,
)

logger = logging.getLogger(__name__)

# ── Singleton state ─────────────────────────────────────────────────────────────
_instance: Optional[VectorStore] = None
_instance_lock = threading.Lock()


class VectorStore:
    """
    Thread-safe singleton LanceDB store.
    """

    _singleton: ClassVar[Optional[VectorStore]] = None
    _write_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, index_dir: Path = INDEX_DIR) -> None:
        self._index_dir = index_dir
        # Ensure parent directory exists
        self._index_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to LanceDB
        self._db = lancedb.connect(str(index_dir))
        
        # Schema definition using PyArrow (explicit types and fixed embedding dimension)
        self._schema = pa.schema([
            pa.field("chunk_id", pa.string()),
            pa.field("source_doc", pa.string()),
            pa.field("doc_version", pa.string()),
            pa.field("effective_date", pa.string()),
            pa.field("superseded", pa.bool_()),
            pa.field("condition_code", pa.string()),
            pa.field("section_path", pa.string()),
            pa.field("content", pa.string()),
            pa.field("chunk_type", pa.string()),
            pa.field("adjacent_clinical_alerts", pa.string()),
            pa.field("page_no", pa.int32()),
            pa.field("embedding", pa.list_(pa.float32(), EMBEDDING_DIM))
        ])
        
        # Open or create the table
        tables = self._db.list_tables()
        table_names = tables.tables if hasattr(tables, "tables") else tables
        if LANCEDB_TABLE_NAME in table_names:
            self._table = self._db.open_table(LANCEDB_TABLE_NAME)
        else:
            self._table = self._db.create_table(LANCEDB_TABLE_NAME, schema=self._schema)

    @classmethod
    def get_instance(cls) -> VectorStore:
        """Thread-safe singleton accessor."""
        if cls._singleton is None:
            with _instance_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    @staticmethod
    def _compute_num_partitions(total_chunks: int) -> int:
        """
        Dynamic partition count — Finding #43 fix.
        LanceDB requires >=39 vectors per partition for stable ANN training.
        """
        return max(8, min(256, total_chunks // MIN_VECTORS_PER_PARTITION))

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Batch-insert chunks as an Arrow table.
        Acquires the thread-safe write lock (Finding #2).
        """
        if not chunks:
            return

        # Convert raw dictionaries to PyArrow table using target schema
        arrow_table = pa.Table.from_pylist(chunks, schema=self._schema)
        
        with self._write_lock:
            self._table.add(arrow_table)
            logger.info(f"Successfully added {len(chunks)} chunks to VectorStore table '{LANCEDB_TABLE_NAME}'.")

    def build_indexes(self) -> None:
        """
        Builds ANN (IVF-HNSW-SQ) and FTS (BM25/Tantivy) indexes explicitly.
        This must be called after bulk document ingestion (Finding #29).
        """
        total_chunks = self._table.count_rows()
        if total_chunks < 8:
            logger.warning(f"Skipping index creation: table has only {total_chunks} rows (minimum 8 required for training).")
            # Create FTS index regardless of row count if table is not empty
            if total_chunks > 0:
                with self._write_lock:
                    self._table.create_fts_index("content", replace=True)
            return

        num_partitions = self._compute_num_partitions(total_chunks)
        logger.info(f"Building vector index on 'embedding' with cosine metric and num_partitions={num_partitions}...")
        
        with self._write_lock:
            # IVF-HNSW-SQ index configuration:
            # LanceDB 0.33.0 uses arguments directly in table.create_index()
            self._table.create_index(
                vector_column_name="embedding",
                metric="cosine",
                num_partitions=num_partitions,
                num_sub_vectors=96,  # 768 / 8 bits = 96 sub-vectors
                index_type="IVF_HNSW_SQ",
                replace=True
            )
            
            # Create Tantivy-backed FTS index on content
            logger.info("Building full-text search index on 'content'...")
            self._table.create_fts_index("content", replace=True)
            
        # Self-audit check: ensure index was successfully applied
        self.optimize()

    def optimize(self) -> None:
        """
        Merges new writes, cleans up deleted rows, and updates indexes.
        Ensures zero unindexed rows remain (Finding #29 / #43).
        """
        with self._write_lock:
            self._table.compact_files()
            self._table.cleanup_old_versions()
            
        # Verify stats if index exists
        try:
            stats = self._table.index_stats("embedding")
            unindexed = stats.num_unindexed_rows
            if unindexed > 0:
                logger.warning(f"VectorStore optimization warning: {unindexed} unindexed rows remaining. Rebuilding index.")
                self.build_indexes()
            else:
                logger.info("VectorStore verification passed: 0 unindexed rows.")
        except Exception:
            # Index might not be built yet
            pass

    def mark_source_as_superseded(self, source_doc: str) -> None:
        """
        Updates the store to mark all chunks belonging to a superseded source
        document as superseded=True.
        """
        with self._write_lock:
            # LanceDB supports updating columns via table.update()
            self._table.update(
                where=f"source_doc = '{source_doc}'",
                values={"superseded": True}
            )
            logger.info(f"Marked all chunks from source document '{source_doc}' as superseded.")

    def get_all_document_metadata(self) -> list[dict[str, Any]]:
        """
        Retrieves unique documents currently indexed in the store.
        """
        # Convert to arrow table and extract unique columns
        df = self._table.to_pandas()
        if df.empty:
            return []
        
        unique_docs = df[["source_doc", "doc_version", "effective_date", "superseded"]].drop_duplicates()
        return unique_docs.to_dict(orient="records")

    def get_condition_distribution(self) -> dict[str, int]:
        """
        Returns count of chunks grouped by condition code.
        """
        df = self._table.to_pandas()
        if df.empty:
            return {}
        
        counts = df["condition_code"].value_counts()
        return counts.to_dict()
