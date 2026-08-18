from pathlib import Path

import pytest

from rk_llm.backends.rknn3 import RKNN3Backend
from rk_llm.errors import (
    ArtifactError,
    BackendUnavailableError,
    ConfigurationError,
    NativeRunnerError,
    RKLLMProjectError,
)
from rk_llm.platform.probe import probe_rknn3
from rk_llm.types import GenerationRequest


def _write_package(package_path: Path, *, executable: bool = True) -> None:
    runner = package_path / "bin/rknn_qwen_runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner", encoding="utf-8")
    if executable:
        runner.chmod(0o755)
    (package_path / "manifest.json").write_text("{}", encoding="utf-8")


def test_project_errors_share_a_public_base_type() -> None:
    assert issubclass(ConfigurationError, RKLLMProjectError)
    assert issubclass(ArtifactError, RKLLMProjectError)
    assert issubclass(BackendUnavailableError, RKLLMProjectError)
    assert issubclass(NativeRunnerError, RKLLMProjectError)


def test_rknn3_probe_reports_all_missing_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "x86_64")

    capabilities = probe_rknn3(tmp_path / "missing-package")

    assert capabilities.name == "rknn3"
    assert capabilities.available is False
    assert capabilities.target == "rk3588-rk1828"
    assert capabilities.is_mock is False
    assert "deployment package is missing" in (capabilities.reason or "")
    assert "host architecture is x86_64, expected aarch64" in (
        capabilities.reason or ""
    )


def test_rknn3_probe_reports_non_executable_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "aarch64")
    package_path = tmp_path / "package"
    _write_package(package_path, executable=False)

    capabilities = probe_rknn3(package_path)

    runner = package_path / "bin/rknn_qwen_runner"
    assert capabilities.available is False
    assert capabilities.reason == f"native runner is not executable: {runner}"


def test_rknn3_probe_reports_missing_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "aarch64")
    package_path = tmp_path / "package"
    runner = package_path / "bin/rknn_qwen_runner"
    runner.parent.mkdir(parents=True)
    runner.write_text("runner", encoding="utf-8")
    runner.chmod(0o755)

    capabilities = probe_rknn3(package_path)

    manifest = package_path / "manifest.json"
    assert capabilities.available is False
    assert capabilities.reason == f"deployment manifest is missing: {manifest}"


def test_rknn3_probe_reports_satisfied_prerequisites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "arm64")
    package_path = tmp_path / "package"
    _write_package(package_path)

    capabilities = probe_rknn3(package_path)

    assert capabilities.available is True
    assert capabilities.reason is None


def test_rknn3_backend_remains_unavailable_without_native_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "arm64")
    package_path = tmp_path / "package"
    _write_package(package_path)

    capabilities = RKNN3Backend(package_path).capabilities()

    assert capabilities.available is False
    assert "native protocol is not implemented" in (capabilities.reason or "")


def test_rknn3_backend_never_falls_back_when_unavailable(tmp_path: Path) -> None:
    backend = RKNN3Backend(tmp_path / "missing-package")

    with pytest.raises(BackendUnavailableError, match="deployment package"):
        backend.load()


def test_rknn3_backend_rejects_unimplemented_native_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rk_llm.platform.probe.platform.machine", lambda: "aarch64")
    package_path = tmp_path / "package"
    _write_package(package_path)
    backend = RKNN3Backend(package_path)

    with pytest.raises(NativeRunnerError, match="native protocol is not implemented"):
        backend.load()

    stream = backend.generate(GenerationRequest(prompt="hello"))
    assert iter(stream) is stream
    with pytest.raises(NativeRunnerError, match="native protocol is not implemented"):
        next(stream)

    assert backend.shutdown() is None
