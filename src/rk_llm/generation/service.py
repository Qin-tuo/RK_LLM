import time
from collections.abc import Callable

from rk_llm.backends.base import GenerationBackend
from rk_llm.types import GenerationRequest, GenerationResult, TextChunk


class GenerationService:
    def __init__(
        self,
        backend: GenerationBackend,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self._clock = clock

    def generate(
        self,
        request: GenerationRequest,
        on_chunk: Callable[[TextChunk], None] | None = None,
    ) -> GenerationResult:
        capabilities = self._backend.capabilities()
        if not capabilities.available:
            raise RuntimeError(capabilities.reason or "backend unavailable")
        self._backend.load()
        started = self._clock()
        first_at: float | None = None
        pieces: list[str] = []
        token_count = 0
        for chunk in self._backend.generate(request):
            emitted_at = self._clock()
            first_at = emitted_at if first_at is None else first_at
            pieces.append(chunk.text)
            token_count += chunk.token_count
            if on_chunk is not None:
                on_chunk(chunk)
        finished = self._clock()
        if first_at is None:
            first_at = finished
        decode_seconds = max(finished - first_at, 0.0)
        return GenerationResult(
            text="".join(pieces),
            generated_tokens=token_count,
            backend=capabilities.name,
            is_mock=capabilities.is_mock,
            ttft_ms=round((first_at - started) * 1000, 3),
            decode_ms=round(decode_seconds * 1000, 3),
            tokens_per_second=(
                round(token_count / decode_seconds, 3) if decode_seconds else 0.0
            ),
        )
