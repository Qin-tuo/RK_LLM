"""Guarded boundary for the RKNN3 Qwen native runner."""

from collections.abc import Iterator
from pathlib import Path

from rk_llm.errors import BackendUnavailableError, NativeRunnerError
from rk_llm.platform.probe import probe_rknn3
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


_UNIMPLEMENTED_MESSAGE = "RKNN3 native protocol is not implemented in this milestone"


class RKNN3Backend:
    def __init__(self, package_path: Path):
        self._package_path = package_path

    def capabilities(self) -> BackendCapabilities:
        prerequisites = probe_rknn3(self._package_path)
        if not prerequisites.available:
            return prerequisites
        return BackendCapabilities(
            name="rknn3",
            available=False,
            streaming=True,
            target="rk3588-rk1828",
            is_mock=False,
            reason=_UNIMPLEMENTED_MESSAGE,
        )

    def load(self) -> None:
        prerequisites = probe_rknn3(self._package_path)
        if not prerequisites.available:
            raise BackendUnavailableError(
                prerequisites.reason or "RKNN3 backend prerequisites unavailable"
            )
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield from ()
        raise NativeRunnerError(_UNIMPLEMENTED_MESSAGE)

    def shutdown(self) -> None:
        return None
