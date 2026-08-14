import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from rk_llm.cli import entrypoint, main
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


def test_doctor_labels_mock_backend(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor", "--backend", "mock"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "mock"
    assert payload["available"] is True
    assert payload["is_mock"] is True


def test_doctor_reports_rkllm_unavailable_without_mock_fallback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "doctor",
                "--backend",
                "rkllm",
                "--model",
                str(tmp_path / "missing.rkllm"),
            ]
        )
        == 2
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "rkllm"
    assert payload["available"] is False
    assert payload["is_mock"] is False
    assert "missing" in payload["reason"]


def test_generate_streams_mock_output_then_prints_json_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        "backend: mock\ntarget: host\nmax_new_tokens: 8\n", encoding="utf-8"
    )

    assert main(["generate", "--config", str(config), "--prompt", "hello"]) == 0

    output_lines = capsys.readouterr().out.splitlines()
    assert output_lines[0] == "mock: hello"
    result = json.loads(output_lines[1])
    assert result["text"] == "mock: hello"
    assert result["backend"] == "mock"
    assert result["is_mock"] is True


def test_generate_relies_on_service_for_single_shutdown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\n", encoding="utf-8")
    backend = RecordingBackend()
    monkeypatch.setattr("rk_llm.cli._backend", lambda runtime: backend)

    assert main(["generate", "--config", str(config), "--prompt", "hello"]) == 0

    assert backend.shutdown_calls == 1
    assert capsys.readouterr().out.splitlines()[0] == "recording: hello"


def test_benchmark_creates_output_and_reports_record_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "iterations: 1\nprompts: [hello]\nruntime: {backend: mock, target: host}\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.jsonl"

    assert main(["benchmark", "--config", str(config), "--output", str(output)]) == 0

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
        "backend: rkllm\ntarget: rk3588\nmodel_path: missing.rkllm\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["rk-llm", "generate", "--config", str(config), "--prompt", "hello"],
    )

    with pytest.raises(SystemExit) as exit_info:
        entrypoint()

    assert exit_info.value.code == 2
    assert "native runner is not executable" in capsys.readouterr().err
