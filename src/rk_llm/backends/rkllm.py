"""Guarded boundary for the future native RKLLM runner."""

from collections.abc import Iterator
from pathlib import Path

from rk_llm.errors import BackendUnavailableError, NativeRunnerError
from rk_llm.platform.probe import probe_rkllm
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


_UNIMPLEMENTED_MESSAGE = "RKLLM native protocol is not part of the skeleton milestone"


class RKLLMBackend:
    def __init__(self, runner_path: Path, model_path: Path):
        self._runner_path = runner_path
        self._model_path = model_path

    def capabilities(self) -> BackendCapabilities:
        return probe_rkllm(self._runner_path, self._model_path)

    def load(self) -> None:
        capabilities = self.capabilities()
        if not capabilities.available:
            raise BackendUnavailableError(capabilities.reason or "RKLLM backend unavailable")
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def shutdown(self) -> None:
        return None
