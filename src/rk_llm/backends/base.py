from collections.abc import Iterator
from typing import Protocol

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class GenerationBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...

    def load(self) -> None: ...

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]: ...

    def shutdown(self) -> None: ...
