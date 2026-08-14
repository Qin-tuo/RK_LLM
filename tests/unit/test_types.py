import pytest

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


def test_generation_request_rejects_blank_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        GenerationRequest(prompt="   ")


def test_generation_request_requires_positive_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest(prompt="hello", max_new_tokens=0)


def test_generation_request_rejects_negative_temperature():
    with pytest.raises(ValueError, match="temperature"):
        GenerationRequest(prompt="hello", temperature=-0.1)


def test_text_chunk_preserves_token_count():
    chunk = TextChunk(text="hello", token_count=1)

    assert chunk.token_count == 1


def test_backend_capabilities_preserves_mock_flag():
    capabilities = BackendCapabilities(
        name="mock",
        available=True,
        streaming=True,
        target="host",
        is_mock=True,
    )

    assert capabilities.is_mock is True
