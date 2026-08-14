from collections.abc import Iterator

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class MockBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("mock", True, True, "host", True)

    def load(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield TextChunk("mock:")
        yield TextChunk(" ")
        yield TextChunk(request.prompt.strip())

    def shutdown(self) -> None:
        return None
