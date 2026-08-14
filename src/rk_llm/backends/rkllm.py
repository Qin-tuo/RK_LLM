"""Guarded boundary for the future native RKLLM runner."""

from collections.abc import Iterator
from pathlib import Path

from rk_llm.errors import BackendUnavailableError, NativeRunnerError
from rk_llm.platform.probe import probe_rkllm
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


_UNIMPLEMENTED_MESSAGE = "RKLLM native protocol is not implemented in this skeleton milestone"


class RKLLMBackend:
    def __init__(self, runner_path: Path, model_path: Path):
        self._runner_path = runner_path
        self._model_path = model_path

    def capabilities(self) -> BackendCapabilities:
        prerequisites = probe_rkllm(self._runner_path, self._model_path)
        if not prerequisites.available:
            return prerequisites
        return BackendCapabilities(
            name=prerequisites.name,
            available=False,
            streaming=prerequisites.streaming,
            target=prerequisites.target,
            is_mock=prerequisites.is_mock,
            reason=_UNIMPLEMENTED_MESSAGE,
        )

    def load(self) -> None:
        prerequisites = probe_rkllm(self._runner_path, self._model_path)
        if not prerequisites.available:
            raise BackendUnavailableError(
                prerequisites.reason or "RKLLM backend prerequisites unavailable"
            )
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield from ()
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def shutdown(self) -> None:
        return None
