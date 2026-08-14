from collections.abc import Iterator

import pytest

from rk_llm.backends.mock import MockBackend
from rk_llm.generation.service import GenerationService
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class SingleChunkBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("single", True, True, "host", True)

    def load(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield TextChunk(request.prompt)

    def shutdown(self) -> None:
        return None


class RecordingBackend:
    def __init__(
        self, *, available: bool = True, fail_during_generation: bool = False
    ) -> None:
        self.available = available
        self.fail_during_generation = fail_during_generation
        self.load_calls = 0
        self.shutdown_calls = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            "recording",
            self.available,
            True,
            "host",
            True,
            None if self.available else "recording backend unavailable",
        )

    def load(self) -> None:
        self.load_calls += 1

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        if self.fail_during_generation:
            raise RuntimeError("generation failed")
        yield TextChunk(request.prompt)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


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
    assert result.decode_ms == 300.0
    assert result.tokens_per_second == 6.667


def test_single_chunk_generation_has_zero_decode_rate_for_zero_duration() -> None:
    ticks = iter([1.0, 1.0, 1.0])

    result = GenerationService(
        SingleChunkBackend(), clock=lambda: next(ticks)
    ).generate(GenerationRequest(prompt="hello"))

    assert result.generated_tokens == 1
    assert result.decode_ms == 0.0
    assert result.tokens_per_second == 0.0


@pytest.mark.parametrize(
    ("max_new_tokens", "expected_text"),
    [(1, "mock:"), (2, "mock: ")],
)
def test_mock_backend_respects_small_token_limits(
    max_new_tokens: int, expected_text: str
) -> None:
    chunks = list(
        MockBackend().generate(
            GenerationRequest(prompt="hello", max_new_tokens=max_new_tokens)
        )
    )

    assert "".join(chunk.text for chunk in chunks) == expected_text
    assert sum(chunk.token_count for chunk in chunks) == max_new_tokens


def test_generation_service_shuts_down_after_success() -> None:
    backend = RecordingBackend()

    GenerationService(backend, clock=lambda: 0.0).generate(
        GenerationRequest(prompt="hello")
    )

    assert backend.load_calls == 1
    assert backend.shutdown_calls == 1


def test_generation_service_shuts_down_after_generator_failure() -> None:
    backend = RecordingBackend(fail_during_generation=True)

    with pytest.raises(RuntimeError, match="generation failed"):
        GenerationService(backend, clock=lambda: 0.0).generate(
            GenerationRequest(prompt="hello")
        )

    assert backend.load_calls == 1
    assert backend.shutdown_calls == 1


def test_generation_service_shuts_down_after_callback_failure() -> None:
    backend = RecordingBackend()

    def fail_callback(chunk: TextChunk) -> None:
        raise ValueError(f"callback rejected {chunk.text}")

    with pytest.raises(ValueError, match="callback rejected hello"):
        GenerationService(backend, clock=lambda: 0.0).generate(
            GenerationRequest(prompt="hello"), on_chunk=fail_callback
        )

    assert backend.load_calls == 1
    assert backend.shutdown_calls == 1


def test_unavailable_backend_does_not_enter_load_lifecycle() -> None:
    backend = RecordingBackend(available=False)

    with pytest.raises(RuntimeError, match="recording backend unavailable"):
        GenerationService(backend).generate(GenerationRequest(prompt="hello"))

    assert backend.load_calls == 0
    assert backend.shutdown_calls == 0
