"""Unit tests for EmbeddingService."""

from unittest.mock import Mock, patch

import openai
import pytest

from src.config import Settings
from src.services.embeddings import EmbeddingService


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings."""
    settings = Mock(spec=Settings)
    settings.semantic_api_key = "test-key"
    settings.semantic_base_url = "https://api.example.com/v1"
    settings.semantic_embedding_model = "text-embedding-3-small"
    settings.embedding_batch_size = 2048
    return settings


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    client = Mock()
    return client


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_single(mock_openai_class, mock_settings, mock_openai_client):
    """Test embedding a single text."""
    mock_openai_class.return_value = mock_openai_client
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
    mock_openai_client.embeddings.create.return_value = mock_response

    service = EmbeddingService(mock_settings)
    result = service.embed("test text")

    assert result == [0.1, 0.2, 0.3]
    mock_openai_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="test text",
        encoding_format="float",
    )


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_single_chunk(mock_openai_class, mock_settings, mock_openai_client):
    """Test batch embedding with a single chunk (within batch size)."""
    mock_openai_class.return_value = mock_openai_client
    mock_response = Mock()
    mock_response.data = [
        Mock(embedding=[0.1, 0.2]),
        Mock(embedding=[0.3, 0.4]),
        Mock(embedding=[0.5, 0.6]),
    ]
    mock_openai_client.embeddings.create.return_value = mock_response

    service = EmbeddingService(mock_settings)
    texts = ["text1", "text2", "text3"]
    results = service.embed_batch(texts)

    assert len(results) == 3
    assert results[0] == [0.1, 0.2]
    assert results[1] == [0.3, 0.4]
    assert results[2] == [0.5, 0.6]
    # Should be called once with all texts
    mock_openai_client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input=texts,
        encoding_format="float",
    )


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_multiple_chunks(mock_openai_class, mock_settings, mock_openai_client):
    """Test batch embedding with multiple chunks (exceeds batch size)."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 2  # Small batch size for testing

    # Mock responses for each chunk
    chunk1_response = Mock()
    chunk1_response.data = [Mock(embedding=[0.1, 0.2]), Mock(embedding=[0.3, 0.4])]
    chunk2_response = Mock()
    chunk2_response.data = [Mock(embedding=[0.5, 0.6]), Mock(embedding=[0.7, 0.8])]
    chunk3_response = Mock()
    chunk3_response.data = [Mock(embedding=[0.9, 1.0])]

    mock_openai_client.embeddings.create.side_effect = [
        chunk1_response,
        chunk2_response,
        chunk3_response,
    ]

    service = EmbeddingService(mock_settings)
    texts = ["text1", "text2", "text3", "text4", "text5"]
    results = service.embed_batch(texts)

    assert len(results) == 5
    assert results[0] == [0.1, 0.2]
    assert results[1] == [0.3, 0.4]
    assert results[2] == [0.5, 0.6]
    assert results[3] == [0.7, 0.8]
    assert results[4] == [0.9, 1.0]
    # Should be called 3 times (for 3 chunks)
    assert mock_openai_client.embeddings.create.call_count == 3


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_preserves_order(mock_openai_class, mock_settings, mock_openai_client):
    """Test that batch embedding preserves input order even with parallel processing."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 2

    # Create responses that could arrive out of order
    chunk1_response = Mock()
    chunk1_response.data = [Mock(embedding=[1.0, 1.0]), Mock(embedding=[2.0, 2.0])]
    chunk2_response = Mock()
    chunk2_response.data = [Mock(embedding=[3.0, 3.0]), Mock(embedding=[4.0, 4.0])]

    mock_openai_client.embeddings.create.side_effect = [chunk1_response, chunk2_response]

    service = EmbeddingService(mock_settings)
    texts = ["a", "b", "c", "d"]
    results = service.embed_batch(texts)

    # Results should be in same order as input
    assert len(results) == 4
    assert results[0] == [1.0, 1.0]  # From chunk 1, position 0
    assert results[1] == [2.0, 2.0]  # From chunk 1, position 1
    assert results[2] == [3.0, 3.0]  # From chunk 2, position 0
    assert results[3] == [4.0, 4.0]  # From chunk 2, position 1


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_empty_input(mock_openai_class, mock_settings, mock_openai_client):
    """Test batch embedding with empty input."""
    mock_openai_class.return_value = mock_openai_client
    service = EmbeddingService(mock_settings)

    results = service.embed_batch([])
    assert results == []
    mock_openai_client.embeddings.create.assert_not_called()


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_chunk_boundary(mock_openai_class, mock_settings, mock_openai_client):
    """Test batch embedding at exact chunk boundary."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 3

    chunk1_response = Mock()
    chunk1_response.data = [
        Mock(embedding=[1.0]),
        Mock(embedding=[2.0]),
        Mock(embedding=[3.0]),
    ]
    chunk2_response = Mock()
    chunk2_response.data = [Mock(embedding=[4.0])]

    mock_openai_client.embeddings.create.side_effect = [chunk1_response, chunk2_response]

    service = EmbeddingService(mock_settings)
    texts = ["a", "b", "c", "d"]  # Exactly 3 + 1
    results = service.embed_batch(texts)

    assert len(results) == 4
    assert mock_openai_client.embeddings.create.call_count == 2


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_configurable_batch_size(mock_openai_class, mock_settings, mock_openai_client):
    """Test that batch size is read from settings."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 5

    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1]) for _ in range(10)]
    mock_openai_client.embeddings.create.return_value = mock_response

    service = EmbeddingService(mock_settings)
    texts = ["text"] * 10
    service.embed_batch(texts)

    # Should be called twice: first chunk of 5, second chunk of 5
    assert mock_openai_client.embeddings.create.call_count == 2
    # Verify chunk sizes
    calls = mock_openai_client.embeddings.create.call_args_list
    assert len(calls[0][1]["input"]) == 5
    assert len(calls[1][1]["input"]) == 5


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_handles_failure_with_retry(
    mock_openai_class, mock_settings, mock_openai_client
):
    """Test that batch embedding retries failed chunks."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 2

    # First call fails, second succeeds
    error = openai.APIError(message="API error", request=Mock(), body={})
    chunk_response = Mock()
    chunk_response.data = [Mock(embedding=[1.0]), Mock(embedding=[2.0])]

    # Use side_effect to simulate retry: fail twice, then succeed
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise error
        return chunk_response

    mock_openai_client.embeddings.create.side_effect = side_effect

    service = EmbeddingService(mock_settings)
    texts = ["text1", "text2"]
    results = service.embed_batch(texts)

    # Should eventually succeed after retries
    assert len(results) == 2
    assert results[0] == [1.0]
    assert results[1] == [2.0]
    # Should have been called 3 times (2 failures + 1 success)
    assert mock_openai_client.embeddings.create.call_count == 3


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_persistent_failure_raises(
    mock_openai_class, mock_settings, mock_openai_client
):
    """Test that persistent failures after retries raise exception with structured error."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 2

    # All retries fail
    error = openai.APIError(message="Persistent API error", request=Mock(), body={})
    mock_openai_client.embeddings.create.side_effect = error

    service = EmbeddingService(mock_settings)
    texts = ["text1", "text2"]

    with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
        service.embed_batch(texts)

    # Should have been called 5 times (max retry attempts)
    assert mock_openai_client.embeddings.create.call_count == 5


@patch("src.services.embeddings.openai.OpenAI")
def test_embed_batch_mixed_success_failure_chunks(
    mock_openai_class, mock_settings, mock_openai_client
):
    """Test batch embedding with some chunks succeeding and others failing."""
    mock_openai_class.return_value = mock_openai_client
    mock_settings.embedding_batch_size = 2

    # Chunk 1 succeeds, chunk 2 fails persistently
    chunk1_response = Mock()
    chunk1_response.data = [Mock(embedding=[1.0]), Mock(embedding=[2.0])]
    error = openai.APIError(message="Chunk 2 error", request=Mock(), body={})

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First call (chunk 1) succeeds
        if call_count == 1:
            return chunk1_response
        # All subsequent calls (chunk 2 retries) fail
        raise error

    mock_openai_client.embeddings.create.side_effect = side_effect

    service = EmbeddingService(mock_settings)
    texts = ["text1", "text2", "text3", "text4"]

    with pytest.raises(RuntimeError, match="Failed to generate embeddings"):
        service.embed_batch(texts)

    # Should have attempted chunk 1 (1 call) + chunk 2 (5 retry attempts)
    assert mock_openai_client.embeddings.create.call_count >= 6
