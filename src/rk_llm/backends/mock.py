from collections.abc import Iterator

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class MockBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("mock", True, True, "host", True)

    def load(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        chunks = (TextChunk("mock:"), TextChunk(" "), TextChunk(request.prompt.strip()))
        yield from chunks[: request.max_new_tokens]

    def shutdown(self) -> None:
        return None
