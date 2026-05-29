import pytest
from unittest.mock import MagicMock
from sentinel.retrieval.hybrid import _rrf_merge
from sentinel.retrieval.reranker import rerank


def test_rrf_merge():
    # 2 documents: chunk_1 is rank 1 in FTS and rank 2 in ANN
    # chunk_2 is rank 2 in FTS and rank 1 in ANN
    fts_results = [
        {"chunk_id": "chunk_1", "content": "Depression criteria"},
        {"chunk_id": "chunk_2", "content": "Suicide assessment"}
    ]
    ann_results = [
        {"chunk_id": "chunk_2", "content": "Suicide assessment"},
        {"chunk_id": "chunk_1", "content": "Depression criteria"}
    ]
    
    # Run merge (k=60)
    merged = _rrf_merge(fts_results, ann_results, k=60, top_n=2)
    
    assert len(merged) == 2
    # Reciprocal ranks:
    # chunk_1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.01639 + 0.01612 = 0.03251
    # chunk_2: 1/(60+2) + 1/(60+1) = 0.03251
    # They should have the same score, both should be present
    assert {c["chunk_id"] for c in merged} == {"chunk_1", "chunk_2"}


@pytest.mark.asyncio
async def test_rerank(monkeypatch):
    # Mock the cross encoder model to bypass loading weights in tests
    mock_model = MagicMock()
    mock_model.predict.return_value = [0.12, 0.95, 0.45]
    monkeypatch.setattr("sentinel.retrieval.reranker._get_reranker", lambda: mock_model)

    chunks = [
        {"chunk_id": "c1", "content": "First passage"},
        {"chunk_id": "c2", "content": "Second passage"},
        {"chunk_id": "c3", "content": "Third passage"}
    ]
    
    results = await rerank("query", chunks, top_n=2)
    
    # Check that mock model was called with 3 pairs
    mock_model.predict.assert_called_once_with([
        ("query", "First passage"),
        ("query", "Second passage"),
        ("query", "Third passage")
    ])
    
    # Assert top_n is respected
    assert len(results) == 2
    
    # Assert sorted by score descending: 0.95 (c2) -> 0.45 (c3)
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["chunk_id"] == "c3"
    assert results[1]["rerank_score"] == 0.45
