import json
from pathlib import Path

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

    lines = [json.loads(line) for line in output.read_text().splitlines()]
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
