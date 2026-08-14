from pathlib import Path

import pytest

from rk_llm.backends.rkllm import RKLLMBackend
from rk_llm.errors import (
    ArtifactError,
    BackendUnavailableError,
    ConfigurationError,
    NativeRunnerError,
    RKLLMProjectError,
)
from rk_llm.types import GenerationRequest


def test_project_errors_share_a_public_base_type() -> None:
    assert issubclass(ConfigurationError, RKLLMProjectError)
    assert issubclass(ArtifactError, RKLLMProjectError)
    assert issubclass(BackendUnavailableError, RKLLMProjectError)
    assert issubclass(NativeRunnerError, RKLLMProjectError)


def test_rkllm_backend_reports_all_missing_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "x86_64")
    backend = RKLLMBackend(tmp_path / "runner", tmp_path / "model.rkllm")

    capabilities = backend.capabilities()

    assert capabilities.name == "rkllm"
    assert capabilities.available is False
    assert capabilities.streaming is True
    assert capabilities.target == "rk"
    assert capabilities.is_mock is False
    assert "native runner is not executable" in (capabilities.reason or "")
    assert "RKLLM model is missing" in (capabilities.reason or "")
    assert "host architecture is x86_64, expected aarch64" in (capabilities.reason or "")


def test_rkllm_backend_reports_non_executable_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "aarch64")
    runner_path = tmp_path / "runner"
    runner_path.write_text("runner", encoding="utf-8")
    model_path = tmp_path / "model.rkllm"
    model_path.write_bytes(b"model")

    capabilities = RKLLMBackend(runner_path, model_path).capabilities()

    assert capabilities.available is False
    assert capabilities.reason == f"native runner is not executable: {runner_path}"


def test_rkllm_backend_reports_available_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "arm64")
    runner_path = tmp_path / "runner"
    runner_path.write_text("runner", encoding="utf-8")
    runner_path.chmod(0o755)
    model_path = tmp_path / "model.rkllm"
    model_path.write_bytes(b"model")

    capabilities = RKLLMBackend(runner_path, model_path).capabilities()

    assert capabilities.available is True
    assert capabilities.reason is None


def test_rkllm_backend_never_falls_back_when_unavailable(tmp_path: Path) -> None:
    backend = RKLLMBackend(tmp_path / "runner", tmp_path / "model.rkllm")

    with pytest.raises(BackendUnavailableError, match="native runner"):
        backend.load()


def test_rkllm_backend_rejects_unimplemented_native_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "aarch64")
    runner_path = tmp_path / "runner"
    runner_path.write_text("runner", encoding="utf-8")
    runner_path.chmod(0o755)
    model_path = tmp_path / "model.rkllm"
    model_path.write_bytes(b"model")
    backend = RKLLMBackend(runner_path, model_path)

    with pytest.raises(NativeRunnerError, match="not part of the skeleton milestone"):
        backend.load()

    with pytest.raises(NativeRunnerError, match="not part of the skeleton milestone"):
        backend.generate(GenerationRequest(prompt="hello"))

    assert backend.shutdown() is None
