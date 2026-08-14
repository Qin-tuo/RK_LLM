from rk_llm.backends.mock import MockBackend
from rk_llm.generation.service import GenerationService
from rk_llm.types import GenerationRequest


def test_mock_generation_streams_and_reports_mock_identity() -> None:
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    chunks: list[str] = []
    result = GenerationService(MockBackend(), clock=lambda: next(ticks)).generate(
        GenerationRequest(prompt="hello"), on_chunk=lambda chunk: chunks.append(chunk.text)
    )

    assert "".join(chunks) == "mock: hello"
    assert result.text == "mock: hello"
    assert result.generated_tokens == 3
    assert result.backend == "mock"
    assert result.is_mock is True
    assert result.ttft_ms == 100.0
