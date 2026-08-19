import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import rk_llm.cli as cli_module
from rk_llm.cli import entrypoint, main
from rk_llm.errors import BackendUnavailableError
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class RecordingBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("recording", True, True, "host", True)

    def load(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield TextChunk(f"recording: {request.prompt}")

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class BrokenStdout:
    def write(self, text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        return None


@pytest.fixture(scope="session")
def rk_llm_executable() -> Path:
    executable = Path(sys.executable).parent / "rk-llm"
    assert executable.is_file(), "install the project before running integration tests"
    assert os.access(executable, os.X_OK), f"console script is not executable: {executable}"
    return executable


def _clean_cli_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in ("RK_LLM_ROOT", "RKNN3_PACKAGE"):
        environment.pop(variable, None)
    return environment


def test_doctor_labels_mock_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--backend", "mock"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "mock"
    assert payload["available"] is True
    assert payload["is_mock"] is True


def test_doctor_reports_rknn3_unavailable_without_mock_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "doctor",
                "--backend",
                "rknn3",
                "--package",
                str(tmp_path / "missing-package"),
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "rknn3"
    assert payload["available"] is False
    assert payload["is_mock"] is False
    assert "deployment package is missing" in payload["reason"]


def test_generate_cli_backend_overrides_config_and_emits_only_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "backend: rknn3\n"
        "target: rk3588\n"
        "package_path: missing-package\n"
        "max_new_tokens: 8\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "generate",
                "--backend",
                "mock",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ]
        )
        == 0
    )

    assert capsys.readouterr().out == "mock: hello\n"


def test_generate_rknn3_selection_never_falls_back_to_mock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "backend: mock\n"
        "target: host\n"
        "package_path: missing-package\n",
        encoding="utf-8",
    )

    with pytest.raises(BackendUnavailableError):
        main(
            [
                "generate",
                "--backend",
                "rknn3",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ]
        )

    assert capsys.readouterr().out == ""


def test_generate_rknn3_selection_requires_package_path(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")

    with pytest.raises(ValueError, match="package_path is required for rknn3"):
        main(
            [
                "generate",
                "--backend",
                "rknn3",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ]
        )


def test_generate_relies_on_service_for_single_shutdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")
    backend = RecordingBackend()
    monkeypatch.setattr("rk_llm.cli._backend", lambda runtime: backend)

    assert (
        main(
            [
                "generate",
                "--backend",
                "mock",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ]
        )
        == 0
    )

    assert backend.shutdown_calls == 1
    assert capsys.readouterr().out == "recording: hello\n"


def test_generate_shuts_down_backend_when_stdout_pipe_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")
    backend = RecordingBackend()
    monkeypatch.setattr("rk_llm.cli._backend", lambda runtime: backend)
    monkeypatch.setattr(sys, "stdout", BrokenStdout())

    with pytest.raises(BrokenPipeError):
        main(
            [
                "generate",
                "--backend",
                "mock",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ]
        )

    assert backend.shutdown_calls == 1


def test_benchmark_creates_output_and_reports_record_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "iterations: 1\n"
        "prompts: [hello]\n"
        "runtime: {backend: rknn3, target: rk3588, package_path: missing-package}\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.jsonl"

    assert (
        main(
            [
                "benchmark",
                "--backend",
                "mock",
                "--config",
                str(config),
                "--output",
                str(output),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {"records": 1, "output": str(output)}
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["result"]["is_mock"] is True


def test_entrypoint_converts_project_backend_error_to_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "backend: rknn3\ntarget: rk3588\npackage_path: missing-package\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rk-llm",
            "generate",
            "--backend",
            "rknn3",
            "--config",
            str(config),
            "--prompt",
            "hello",
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        entrypoint()

    assert exit_info.value.code == 2
    assert "deployment package is missing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [
        ("generate", "--config", "runtime.yaml", "--prompt", "hello"),
        ("benchmark", "--config", "benchmark.yaml", "--output", "result.jsonl"),
    ],
)
def test_installed_commands_require_explicit_backend_selection(
    rk_llm_executable: Path, tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    result = subprocess.run(
        [str(rk_llm_executable), *arguments],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=_clean_cli_environment(),
    )

    assert result.returncode == 2
    assert "--backend" in result.stderr
    assert "required" in result.stderr
    assert "Traceback" not in result.stderr


def test_installed_generate_is_cwd_independent_and_emits_pure_text(
    rk_llm_executable: Path, tmp_path: Path
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")
    other_cwd = tmp_path / "outside-project"
    other_cwd.mkdir()

    result = subprocess.run(
        [
            str(rk_llm_executable),
            "generate",
            "--backend",
            "mock",
            "--config",
            str(config),
            "--prompt",
            "hello",
        ],
        cwd=other_cwd,
        text=True,
        capture_output=True,
        check=False,
        env=_clean_cli_environment(),
    )

    assert result.returncode == 0
    assert result.stdout == "mock: hello\n"
    assert result.stderr == ""


def test_installed_doctor_uses_package_root_outside_project_cwd(
    rk_llm_executable: Path, tmp_path: Path
) -> None:
    package_root = Path(cli_module.__file__).resolve().parents[2]

    result = subprocess.run(
        [
            str(rk_llm_executable),
            "doctor",
            "--backend",
            "rknn3",
            "--package",
            "definitely-missing/package",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=_clean_cli_environment(),
    )

    assert result.returncode == 2
    capabilities = json.loads(result.stdout)
    assert capabilities["available"] is False
    assert capabilities["is_mock"] is False
    assert str(package_root / "definitely-missing/package") in capabilities[
        "reason"
    ]
    assert str(tmp_path) not in capabilities["reason"]


def test_installed_doctor_honors_explicit_deployment_root(
    rk_llm_executable: Path, tmp_path: Path
) -> None:
    deployment_root = tmp_path / "deployment"
    other_cwd = tmp_path / "outside-project"
    deployment_root.mkdir()
    other_cwd.mkdir()
    environment = _clean_cli_environment()
    environment["RK_LLM_ROOT"] = str(deployment_root)

    result = subprocess.run(
        [str(rk_llm_executable), "doctor", "--backend", "rknn3"],
        cwd=other_cwd,
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )

    assert result.returncode == 2
    capabilities = json.loads(result.stdout)
    assert str(deployment_root / "artifacts/deploy/current") in capabilities[
        "reason"
    ]
    assert str(other_cwd) not in capabilities["reason"]


def test_installed_command_reports_malformed_yaml_without_traceback(
    rk_llm_executable: Path, tmp_path: Path
) -> None:
    config = tmp_path / "malformed.yaml"
    config.write_text("backend: [mock\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(rk_llm_executable),
            "generate",
            "--backend",
            "mock",
            "--config",
            str(config),
            "--prompt",
            "hello",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        env=_clean_cli_environment(),
    )

    assert result.returncode == 2
    assert str(config) in result.stderr
    assert "Traceback" not in result.stderr


def test_installed_generate_handles_closed_pipe_without_flush_noise(
    rk_llm_executable: Path, tmp_path: Path
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        process = subprocess.Popen(
            [
                str(rk_llm_executable),
                "generate",
                "--backend",
                "mock",
                "--config",
                str(config),
                "--prompt",
                "hello",
            ],
            cwd=tmp_path,
            stdout=write_fd,
            stderr=subprocess.PIPE,
            text=True,
            env=_clean_cli_environment(),
        )
    finally:
        os.close(write_fd)

    _, stderr = process.communicate()

    assert process.returncode == 141
    assert stderr is not None
    assert "Exception ignored" not in stderr
