import pytest
import pyarrow as pa
from sentinel.store.vector_store import VectorStore
from sentinel.config import EMBEDDING_DIM


@pytest.fixture(autouse=True)
def mock_index_dir(tmp_path, monkeypatch):
    # Isolate the tests from the production LanceDB directory
    test_index_dir = tmp_path / "test_index_singleton"
    monkeypatch.setattr("sentinel.store.vector_store.INDEX_DIR", test_index_dir)
    # Clear any active singleton instance
    VectorStore._singleton = None
    yield
    VectorStore._singleton = None


@pytest.fixture
def temp_vector_store(tmp_path):
    # Initialize VectorStore on a temporary path
    test_index_dir = tmp_path / "test_index"
    store = VectorStore(test_index_dir)
    return store


def test_singleton():
    store1 = VectorStore.get_instance()
    store2 = VectorStore.get_instance()
    assert store1 is store2


def test_add_and_count_chunks(temp_vector_store):
    store = temp_vector_store
    
    # Assert table is initially empty
    assert store._table.count_rows() == 0

    # Mock chunk
    mock_embedding = [0.1] * EMBEDDING_DIM
    chunks = [
        {
            "chunk_id": "chunk_1",
            "source_doc": "test.pdf",
            "doc_version": "1.0",
            "effective_date": "2020-01-01",
            "superseded": False,
            "condition_code": "DEP",
            "section_path": "DEP > Assessment",
            "content": "This is test clinical content.",
            "chunk_type": "procedure",
            "adjacent_clinical_alerts": "",
            "page_no": 12,
            "embedding": mock_embedding
        }
    ]
    
    # Add chunk
    store.add_chunks(chunks)
    assert store._table.count_rows() == 1

    # Verify document metadata fetching
    docs = store.get_all_document_metadata()
    assert len(docs) == 1
    assert docs[0]["source_doc"] == "test.pdf"

    # Verify condition distribution
    dist = store.get_condition_distribution()
    assert dist.get("DEP") == 1


def test_mark_superseded(temp_vector_store):
    store = temp_vector_store
    mock_embedding = [0.1] * EMBEDDING_DIM
    chunks = [
        {
            "chunk_id": "chunk_1",
            "source_doc": "old.pdf",
            "doc_version": "1.0",
            "effective_date": "2010-01-01",
            "superseded": False,
            "condition_code": "DEP",
            "section_path": "DEP > Assessment",
            "content": "Old content",
            "chunk_type": "procedure",
            "adjacent_clinical_alerts": "",
            "page_no": 1,
            "embedding": mock_embedding
        }
    ]
    store.add_chunks(chunks)
    assert store.get_all_document_metadata()[0]["superseded"] is False

    # Mark as superseded
    store.mark_source_as_superseded("old.pdf")
    
    # Verify change
    assert store.get_all_document_metadata()[0]["superseded"] is True
