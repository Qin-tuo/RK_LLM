import json
import os
from dataclasses import asdict
from pathlib import Path

import pytest

from rk_llm.backends.mock import MockBackend
from rk_llm.config import BenchmarkConfig, RuntimeConfig
from rk_llm.metrics.benchmark import run_benchmark
from rk_llm.metrics.system import collect_system_metrics


class RecordingMockBackend(MockBackend):
    def __init__(self) -> None:
        self.load_calls = 0
        self.shutdown_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def test_benchmark_writes_one_record_per_prompt_and_iteration(tmp_path: Path) -> None:
    runtime = RuntimeConfig("mock", "host", None, 16, 0.8, 0.9, 1, 1.1)
    config = BenchmarkConfig(("one", "two"), 2, runtime)
    output = tmp_path / "nested" / "result.jsonl"
    backend = RecordingMockBackend()

    records = run_benchmark(config, backend, output)

    payload = output.read_text(encoding="utf-8")
    assert payload == "".join(
        json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records
    )
    lines = [json.loads(line) for line in payload.splitlines()]
    assert len(records) == 4
    assert len(lines) == 4
    assert [(line["prompt"], line["iteration"]) for line in lines] == [
        ("one", 1),
        ("one", 2),
        ("two", 1),
        ("two", 2),
    ]
    assert [line["result"]["text"] for line in lines] == [
        "mock: one",
        "mock: one",
        "mock: two",
        "mock: two",
    ]
    assert all(line["result"]["is_mock"] is True for line in lines)
    assert backend.load_calls == 4
    assert backend.shutdown_calls == 4


def test_replace_failure_preserves_output_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text("previous\n", encoding="utf-8")
    files_before = set(tmp_path.iterdir())
    runtime = RuntimeConfig("mock", "host", None, 16, 0.8, 0.9, 1, 1.1)
    config = BenchmarkConfig(("one",), 1, runtime)

    def fail_replace(*args: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        run_benchmark(config, MockBackend(), output)

    assert output.read_text(encoding="utf-8") == "previous\n"
    assert set(tmp_path.iterdir()) == files_before


def test_write_failure_preserves_output_and_cleans_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.jsonl"
    output.write_text("previous\n", encoding="utf-8")
    files_before = set(tmp_path.iterdir())
    runtime = RuntimeConfig("mock", "host", None, 16, 0.8, 0.9, 1, 1.1)
    config = BenchmarkConfig(("one",), 1, runtime)

    def fail_fsync(*args: object) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="fsync failed"):
        run_benchmark(config, MockBackend(), output)

    assert output.read_text(encoding="utf-8") == "previous\n"
    assert set(tmp_path.iterdir()) == files_before


def test_missing_board_sensors_remain_explicit(tmp_path: Path) -> None:
    metrics = collect_system_metrics(
        meminfo_path=tmp_path / "meminfo", temperature_path=tmp_path / "temperature"
    )

    assert metrics.memory_available_mb is None
    assert metrics.temperature_c is None
    assert metrics.cpu_percent is None
    assert metrics.npu_percent is None


def test_available_board_sensor_files_are_parsed(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 8192 kB\nMemAvailable: 3072 kB\n", encoding="utf-8")
    temperature = tmp_path / "temperature"
    temperature.write_text("42500\n", encoding="utf-8")

    metrics = collect_system_metrics(meminfo, temperature)

    assert metrics.memory_available_mb == 3.0
    assert metrics.temperature_c == 42.5
    assert metrics.cpu_percent is None
    assert metrics.npu_percent is None


@pytest.mark.parametrize(
    "content",
    [
        "MemAvailable: invalid kB\n",
        "MemAvailable: 1024\n",
        "MemAvailable: 1024 MB\n",
        "MemAvailable: -1 kB\n",
        "MemAvailable: nan kB\n",
        "MemAvailable: inf kB\n",
    ],
)
def test_invalid_memory_sensor_values_are_unavailable(
    tmp_path: Path, content: str
) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(content, encoding="utf-8")

    metrics = collect_system_metrics(meminfo, tmp_path / "missing-temperature")

    assert metrics.memory_available_mb is None


@pytest.mark.parametrize("content", ["invalid\n", "nan\n", "inf\n", "-inf\n"])
def test_invalid_temperature_sensor_values_are_unavailable(
    tmp_path: Path, content: str
) -> None:
    temperature = tmp_path / "temperature"
    temperature.write_text(content, encoding="utf-8")

    metrics = collect_system_metrics(tmp_path / "missing-meminfo", temperature)

    assert metrics.temperature_c is None


def test_sensor_encoding_errors_are_unavailable(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_bytes(b"MemAvailable: \xff kB\n")
    temperature = tmp_path / "temperature"
    temperature.write_bytes(b"\xff\n")

    metrics = collect_system_metrics(meminfo, temperature)

    assert metrics.memory_available_mb is None
    assert metrics.temperature_c is None


def test_bad_memory_sensor_does_not_hide_valid_temperature(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: invalid kB\n", encoding="utf-8")
    temperature = tmp_path / "temperature"
    temperature.write_text("42500\n", encoding="utf-8")

    metrics = collect_system_metrics(meminfo, temperature)

    assert metrics.memory_available_mb is None
    assert metrics.temperature_c == 42.5


def test_bad_temperature_sensor_does_not_hide_valid_memory(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable: 3072 kB\n", encoding="utf-8")
    temperature = tmp_path / "temperature"
    temperature.write_text("invalid\n", encoding="utf-8")

    metrics = collect_system_metrics(meminfo, temperature)

    assert metrics.memory_available_mb == 3.0
    assert metrics.temperature_c is None
