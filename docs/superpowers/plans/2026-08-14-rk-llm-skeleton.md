# RK LLM Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a tested, installable RK LLM skeleton whose deterministic mock path runs on x86 and whose RKLLM boundary fails explicitly until the vendor runtime and model are available.

**Architecture:** A pure Python application layer depends on a small generation-backend protocol. The mock backend proves configuration, streaming, benchmarking, and CLI behavior on any machine; the RKLLM adapter only probes and guards the future native runner boundary. Host conversion, native runtime, deployment, and hardware tests receive documented, versioned locations without claiming board inference.

**Tech Stack:** Python 3.10-3.12, standard-library dataclasses/argparse/json/pathlib, PyYAML 6, pytest 8, setuptools, CMake/C++17 boundary.

---

## File Map

- `pyproject.toml`: package metadata, dependencies, pytest settings, and `rk-llm` entry point.
- `src/rk_llm/types.py`: immutable request, chunk, result, capability, and benchmark records.
- `src/rk_llm/config.py`: YAML configuration loading and validation.
- `src/rk_llm/backends/`: backend protocol, deterministic mock, and guarded RKLLM adapter.
- `src/rk_llm/generation/service.py`: backend-independent text generation use case.
- `src/rk_llm/metrics/`: timing calculation and JSON Lines benchmark runner.
- `src/rk_llm/platform/probe.py`: host and RK dependency diagnostics.
- `src/rk_llm/cli.py`: `doctor`, `generate`, and `benchmark` commands.
- `native/rkllm_runner/`: explicit CMake/native process boundary, not a vendor SDK copy.
- `configs/`, `requirements/`, `tools/`, `third_party/`, `artifacts/`, `docs/`: operational scaffolding and versioned conventions.
- `tests/unit/`, `tests/integration/`, `tests/hardware/`: host behavior, CLI behavior, and opted-in board checks.

### Task 1: Package and Domain Types

**Files:**
- Create: `pyproject.toml`
- Modify: `.gitignore`
- Create: `src/rk_llm/__init__.py`
- Create: `src/rk_llm/types.py`
- Create: `tests/unit/test_types.py`

- [ ] **Step 1: Write the failing type tests**

```python
# tests/unit/test_types.py
import pytest

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


def test_generation_request_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt must not be empty"):
        GenerationRequest(prompt="   ")


def test_generation_request_rejects_invalid_sampling_values() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationRequest(prompt="hello", max_new_tokens=0)
    with pytest.raises(ValueError, match="temperature"):
        GenerationRequest(prompt="hello", temperature=-0.1)


def test_domain_records_keep_mock_identity() -> None:
    chunk = TextChunk(text="hello", token_count=1)
    capabilities = BackendCapabilities(
        name="mock", available=True, streaming=True, target="host", is_mock=True
    )
    assert chunk.token_count == 1
    assert capabilities.is_mock is True
```

- [ ] **Step 2: Run the type tests and verify RED**

Run: `python3 -m pytest tests/unit/test_types.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'rk_llm'`.

- [ ] **Step 3: Create packaging and the minimal immutable types**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "rk-llm"
version = "0.1.0"
description = "Incremental pure-text RKLLM deployment project"
requires-python = ">=3.10,<3.13"
dependencies = ["PyYAML>=6.0,<7"]

[project.optional-dependencies]
dev = ["pytest>=8,<9"]

[project.scripts]
rk-llm = "rk_llm.cli:entrypoint"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["hardware: requires an explicitly enabled RK target"]
```

```python
# src/rk_llm/__init__.py
"""RK LLM project package."""

__version__ = "0.1.0"
```

```python
# src/rk_llm/types.py
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 1
    repeat_penalty: float = 1.1

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.repeat_penalty <= 0:
            raise ValueError("repeat_penalty must be positive")


@dataclass(frozen=True)
class TextChunk:
    text: str
    token_count: int = 1


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    available: bool
    streaming: bool
    target: str
    is_mock: bool
    reason: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    generated_tokens: int
    backend: str
    is_mock: bool
    ttft_ms: float
    decode_ms: float
    tokens_per_second: float
    termination_reason: str = "completed"


@dataclass(frozen=True)
class BenchmarkRecord:
    prompt: str
    iteration: int
    result: GenerationResult


@dataclass(frozen=True)
class SystemMetrics:
    memory_available_mb: float | None
    temperature_c: float | None
    cpu_percent: float | None
    npu_percent: float | None
```

Append these generated-content rules to `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
artifacts/source_models/
artifacts/converted_models/
artifacts/packages/
artifacts/benchmark_runs/
artifacts/logs/
```

- [ ] **Step 4: Install the package and verify GREEN**

Run: `python3 -m pip install -e ".[dev]" && python3 -m pytest tests/unit/test_types.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit the package baseline**

```bash
git add .gitignore pyproject.toml src/rk_llm tests/unit/test_types.py
git commit -m "feat: add RK LLM package and domain types"
```

### Task 2: Validated YAML Configuration

**Files:**
- Create: `src/rk_llm/config.py`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing runtime and benchmark configuration tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest

from rk_llm.config import load_benchmark_config, load_runtime_config


def test_load_runtime_config_resolves_model_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "backend: rkllm\ntarget: rk3588\nmodel_path: models/demo.rkllm\nmax_new_tokens: 64\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.backend == "rkllm"
    assert config.model_path == tmp_path / "models/demo.rkllm"
    assert config.max_new_tokens == 64


def test_runtime_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backend: gpu\ntarget: host\n", encoding="utf-8")
    with pytest.raises(ValueError, match="backend"):
        load_runtime_config(path)


def test_benchmark_config_requires_prompts(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text("iterations: 1\nprompts: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts"):
        load_benchmark_config(path)
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/unit/test_config.py -q`

Expected: import fails because `rk_llm.config` does not exist.

- [ ] **Step 3: Implement exact configuration models and loaders**

```python
# src/rk_llm/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    target: str
    model_path: Path | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float


@dataclass(frozen=True)
class BenchmarkConfig:
    prompts: tuple[str, ...]
    iterations: int
    runtime: RuntimeConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def _runtime_from_mapping(data: dict[str, Any], base_dir: Path) -> RuntimeConfig:
    backend = str(data.get("backend", "mock"))
    if backend not in {"mock", "rkllm"}:
        raise ValueError("backend must be mock or rkllm")
    target = str(data.get("target", "host"))
    raw_model = data.get("model_path")
    model_path = (base_dir / str(raw_model)).resolve() if raw_model else None
    max_new_tokens = int(data.get("max_new_tokens", 128))
    temperature = float(data.get("temperature", 0.8))
    top_p = float(data.get("top_p", 0.9))
    top_k = int(data.get("top_k", 1))
    repeat_penalty = float(data.get("repeat_penalty", 1.1))
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 1 or repeat_penalty <= 0:
        raise ValueError("invalid sampling configuration")
    if backend == "rkllm" and model_path is None:
        raise ValueError("model_path is required for rkllm")
    return RuntimeConfig(
        backend, target, model_path, max_new_tokens, temperature, top_p, top_k, repeat_penalty
    )


def load_runtime_config(path: Path) -> RuntimeConfig:
    return _runtime_from_mapping(_read_yaml(path), path.parent)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    data = _read_yaml(path)
    prompts = tuple(str(item) for item in data.get("prompts", ()))
    iterations = int(data.get("iterations", 1))
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain non-empty strings")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    runtime_data = data.get("runtime", {"backend": "mock", "target": "host"})
    if not isinstance(runtime_data, dict):
        raise ValueError("runtime must be a mapping")
    return BenchmarkConfig(prompts, iterations, _runtime_from_mapping(runtime_data, path.parent))
```

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/unit/test_config.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit configuration**

```bash
git add src/rk_llm/config.py tests/unit/test_config.py
git commit -m "feat: add validated RK LLM configuration"
```

### Task 3: Backend Protocol, Mock Backend, and Generation Service

**Files:**
- Create: `src/rk_llm/backends/__init__.py`
- Create: `src/rk_llm/backends/base.py`
- Create: `src/rk_llm/backends/mock.py`
- Create: `src/rk_llm/generation/__init__.py`
- Create: `src/rk_llm/generation/service.py`
- Create: `tests/unit/test_generation.py`

- [ ] **Step 1: Write the failing generation test**

```python
# tests/unit/test_generation.py
from rk_llm.backends.mock import MockBackend
from rk_llm.generation.service import GenerationService
from rk_llm.types import GenerationRequest


def test_mock_generation_streams_and_reports_mock_identity() -> None:
    ticks = iter([0.0, 0.1, 0.2, 0.3, 0.4])
    chunks: list[str] = []
    result = GenerationService(MockBackend(), clock=lambda: next(ticks)).generate(
        GenerationRequest(prompt="hello"), on_chunk=lambda chunk: chunks.append(chunk.text)
    )
    assert "".join(chunks) == "mock: hello"
    assert result.text == "mock: hello"
    assert result.generated_tokens == 3
    assert result.backend == "mock"
    assert result.is_mock is True
    assert result.ttft_ms == 100.0
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/unit/test_generation.py -q`

Expected: import fails because the backend package does not exist.

- [ ] **Step 3: Implement the backend protocol and deterministic path**

```python
# src/rk_llm/backends/base.py
from collections.abc import Iterator
from typing import Protocol

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class GenerationBackend(Protocol):
    def capabilities(self) -> BackendCapabilities: ...
    def load(self) -> None: ...
    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]: ...
    def shutdown(self) -> None: ...
```

```python
# src/rk_llm/backends/mock.py
from collections.abc import Iterator

from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


class MockBackend:
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities("mock", True, True, "host", True)

    def load(self) -> None:
        return None

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        yield TextChunk("mock:")
        yield TextChunk(" ")
        yield TextChunk(request.prompt.strip())

    def shutdown(self) -> None:
        return None
```

```python
# src/rk_llm/generation/service.py
import time
from collections.abc import Callable

from rk_llm.backends.base import GenerationBackend
from rk_llm.types import GenerationRequest, GenerationResult, TextChunk


class GenerationService:
    def __init__(self, backend: GenerationBackend, clock: Callable[[], float] = time.monotonic):
        self._backend = backend
        self._clock = clock

    def generate(
        self, request: GenerationRequest, on_chunk: Callable[[TextChunk], None] | None = None
    ) -> GenerationResult:
        capabilities = self._backend.capabilities()
        if not capabilities.available:
            raise RuntimeError(capabilities.reason or "backend unavailable")
        self._backend.load()
        started = self._clock()
        first_at: float | None = None
        pieces: list[str] = []
        token_count = 0
        for chunk in self._backend.generate(request):
            emitted_at = self._clock()
            first_at = emitted_at if first_at is None else first_at
            pieces.append(chunk.text)
            token_count += chunk.token_count
            if on_chunk is not None:
                on_chunk(chunk)
        finished = self._clock()
        if first_at is None:
            first_at = finished
        decode_seconds = max(finished - first_at, 0.0)
        return GenerationResult(
            text="".join(pieces),
            generated_tokens=token_count,
            backend=capabilities.name,
            is_mock=capabilities.is_mock,
            ttft_ms=round((first_at - started) * 1000, 3),
            decode_ms=round(decode_seconds * 1000, 3),
            tokens_per_second=round(token_count / decode_seconds, 3) if decode_seconds else 0.0,
        )
```

Create package initializers containing only module docstrings.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/unit/test_generation.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Commit generation behavior**

```bash
git add src/rk_llm/backends src/rk_llm/generation tests/unit/test_generation.py
git commit -m "feat: add mock RK LLM generation flow"
```

### Task 4: Benchmark Records and JSON Lines Output

**Files:**
- Create: `src/rk_llm/metrics/__init__.py`
- Create: `src/rk_llm/metrics/benchmark.py`
- Create: `src/rk_llm/metrics/system.py`
- Create: `tests/unit/test_benchmark.py`

- [ ] **Step 1: Write the failing benchmark test**

```python
# tests/unit/test_benchmark.py
import json
from pathlib import Path

from rk_llm.backends.mock import MockBackend
from rk_llm.config import BenchmarkConfig, RuntimeConfig
from rk_llm.metrics.benchmark import run_benchmark
from rk_llm.metrics.system import collect_system_metrics


def test_benchmark_writes_one_record_per_prompt_and_iteration(tmp_path: Path) -> None:
    runtime = RuntimeConfig("mock", "host", None, 16, 0.8, 0.9, 1, 1.1)
    config = BenchmarkConfig(("one", "two"), 2, runtime)
    output = tmp_path / "result.jsonl"
    records = run_benchmark(config, MockBackend(), output)
    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4
    assert len(lines) == 4
    assert {line["prompt"] for line in lines} == {"one", "two"}
    assert all(line["result"]["is_mock"] is True for line in lines)


def test_missing_board_sensors_remain_explicit(tmp_path: Path) -> None:
    metrics = collect_system_metrics(
        meminfo_path=tmp_path / "meminfo", temperature_path=tmp_path / "temperature"
    )
    assert metrics.memory_available_mb is None
    assert metrics.temperature_c is None
    assert metrics.cpu_percent is None
    assert metrics.npu_percent is None
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/unit/test_benchmark.py -q`

Expected: import fails because `rk_llm.metrics.benchmark` does not exist.

- [ ] **Step 3: Implement the JSON Lines benchmark runner**

```python
# src/rk_llm/metrics/benchmark.py
import json
from dataclasses import asdict
from pathlib import Path

from rk_llm.backends.base import GenerationBackend
from rk_llm.config import BenchmarkConfig
from rk_llm.generation.service import GenerationService
from rk_llm.types import BenchmarkRecord, GenerationRequest


def run_benchmark(
    config: BenchmarkConfig, backend: GenerationBackend, output: Path
) -> list[BenchmarkRecord]:
    output.parent.mkdir(parents=True, exist_ok=True)
    service = GenerationService(backend)
    records: list[BenchmarkRecord] = []
    for prompt in config.prompts:
        for iteration in range(1, config.iterations + 1):
            request = GenerationRequest(
                prompt=prompt,
                max_new_tokens=config.runtime.max_new_tokens,
                temperature=config.runtime.temperature,
                top_p=config.runtime.top_p,
                top_k=config.runtime.top_k,
                repeat_penalty=config.runtime.repeat_penalty,
            )
            records.append(BenchmarkRecord(prompt, iteration, service.generate(request)))
    output.write_text(
        "".join(json.dumps(asdict(record), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    backend.shutdown()
    return records
```

Create `metrics/__init__.py` with `"""Benchmark and board metric helpers."""`.

```python
# src/rk_llm/metrics/system.py
from pathlib import Path

from rk_llm.types import SystemMetrics


def collect_system_metrics(
    meminfo_path: Path = Path("/proc/meminfo"),
    temperature_path: Path = Path("/sys/class/thermal/thermal_zone0/temp"),
) -> SystemMetrics:
    memory_mb: float | None = None
    temperature_c: float | None = None
    if meminfo_path.is_file():
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                memory_mb = float(line.split()[1]) / 1024.0
                break
    if temperature_path.is_file():
        temperature_c = float(temperature_path.read_text(encoding="utf-8").strip()) / 1000.0
    return SystemMetrics(memory_mb, temperature_c, None, None)
```

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/unit/test_benchmark.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit benchmark output**

```bash
git add src/rk_llm/metrics tests/unit/test_benchmark.py
git commit -m "feat: add reproducible mock benchmark output"
```

### Task 5: Diagnostics and Guarded RKLLM Adapter

**Files:**
- Create: `src/rk_llm/errors.py`
- Create: `src/rk_llm/platform/__init__.py`
- Create: `src/rk_llm/platform/probe.py`
- Create: `src/rk_llm/backends/rkllm.py`
- Create: `tests/unit/test_rkllm_probe.py`

- [ ] **Step 1: Write failing capability tests**

```python
# tests/unit/test_rkllm_probe.py
from pathlib import Path

import pytest

from rk_llm.backends.rkllm import RKLLMBackend
from rk_llm.errors import BackendUnavailableError


def test_rkllm_backend_reports_missing_runner_and_model(tmp_path: Path) -> None:
    backend = RKLLMBackend(tmp_path / "runner", tmp_path / "model.rkllm")
    capabilities = backend.capabilities()
    assert capabilities.available is False
    assert "runner" in (capabilities.reason or "")


def test_rkllm_backend_never_falls_back_to_mock(tmp_path: Path) -> None:
    backend = RKLLMBackend(tmp_path / "runner", tmp_path / "model.rkllm")
    with pytest.raises(BackendUnavailableError):
        backend.load()
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/unit/test_rkllm_probe.py -q`

Expected: import fails because `rk_llm.backends.rkllm` does not exist.

- [ ] **Step 3: Implement explicit errors, probe, and guarded adapter**

```python
# src/rk_llm/errors.py
class RKLLMProjectError(Exception):
    """Base project error."""


class ConfigurationError(RKLLMProjectError):
    pass


class ArtifactError(RKLLMProjectError):
    pass


class BackendUnavailableError(RKLLMProjectError):
    pass


class NativeRunnerError(RKLLMProjectError):
    pass
```

```python
# src/rk_llm/platform/probe.py
import os
import platform
from pathlib import Path

from rk_llm.types import BackendCapabilities


def probe_rkllm(runner_path: Path, model_path: Path) -> BackendCapabilities:
    reasons: list[str] = []
    if not runner_path.is_file() or not os.access(runner_path, os.X_OK):
        reasons.append(f"native runner is not executable: {runner_path}")
    if not model_path.is_file():
        reasons.append(f"RKLLM model is missing: {model_path}")
    if platform.machine() not in {"aarch64", "arm64"}:
        reasons.append(f"host architecture is {platform.machine()}, expected aarch64")
    return BackendCapabilities(
        name="rkllm",
        available=not reasons,
        streaming=True,
        target="rk",
        is_mock=False,
        reason="; ".join(reasons) if reasons else None,
    )
```

```python
# src/rk_llm/backends/rkllm.py
from collections.abc import Iterator
from pathlib import Path

from rk_llm.errors import BackendUnavailableError, NativeRunnerError
from rk_llm.platform.probe import probe_rkllm
from rk_llm.types import BackendCapabilities, GenerationRequest, TextChunk


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
        raise NativeRunnerError("RKLLM native protocol is not part of the skeleton milestone")

    def generate(self, request: GenerationRequest) -> Iterator[TextChunk]:
        raise NativeRunnerError("RKLLM native protocol is not part of the skeleton milestone")

    def shutdown(self) -> None:
        return None
```

Create `platform/__init__.py` with `"""Target capability probes."""`.

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/unit/test_rkllm_probe.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit diagnostics**

```bash
git add src/rk_llm/errors.py src/rk_llm/platform src/rk_llm/backends/rkllm.py tests/unit/test_rkllm_probe.py
git commit -m "feat: add guarded RKLLM backend diagnostics"
```

### Task 6: Command-Line Interface

**Files:**
- Create: `src/rk_llm/cli.py`
- Create: `tests/integration/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

```python
# tests/integration/test_cli.py
import json
from pathlib import Path

from rk_llm.cli import main


def test_doctor_labels_mock_backend(capsys) -> None:
    assert main(["doctor", "--backend", "mock"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "mock"
    assert payload["is_mock"] is True


def test_generate_uses_versioned_config(tmp_path: Path, capsys) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text("backend: mock\ntarget: host\nmax_new_tokens: 8\n", encoding="utf-8")
    assert main(["generate", "--config", str(config), "--prompt", "hello"]) == 0
    output = capsys.readouterr().out
    assert "mock: hello" in output


def test_benchmark_creates_output(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yaml"
    config.write_text(
        "iterations: 1\nprompts: [hello]\nruntime: {backend: mock, target: host}\n",
        encoding="utf-8",
    )
    output = tmp_path / "result.jsonl"
    assert main(["benchmark", "--config", str(config), "--output", str(output)]) == 0
    assert output.is_file()
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/integration/test_cli.py -q`

Expected: import fails because `rk_llm.cli` does not exist.

- [ ] **Step 3: Implement the complete skeleton CLI**

```python
# src/rk_llm/cli.py
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rk_llm.backends.mock import MockBackend
from rk_llm.backends.rkllm import RKLLMBackend
from rk_llm.config import RuntimeConfig, load_benchmark_config, load_runtime_config
from rk_llm.generation.service import GenerationService
from rk_llm.metrics.benchmark import run_benchmark
from rk_llm.types import GenerationRequest


def _backend(config: RuntimeConfig):
    if config.backend == "mock":
        return MockBackend()
    assert config.model_path is not None
    return RKLLMBackend(Path("native/rkllm_runner/build/rkllm_runner"), config.model_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-llm")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--backend", choices=("mock", "rkllm"), default="mock")
    doctor.add_argument("--model", type=Path, default=Path("artifacts/converted_models/model.rkllm"))
    generate = commands.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--prompt", required=True)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        backend = MockBackend() if args.backend == "mock" else RKLLMBackend(
            Path("native/rkllm_runner/build/rkllm_runner"), args.model
        )
        capabilities = backend.capabilities()
        print(json.dumps(asdict(capabilities), ensure_ascii=False))
        return 0 if capabilities.available else 2
    if args.command == "generate":
        config = load_runtime_config(args.config)
        backend = _backend(config)
        request = GenerationRequest(
            args.prompt,
            config.max_new_tokens,
            config.temperature,
            config.top_p,
            config.top_k,
            config.repeat_penalty,
        )
        result = GenerationService(backend).generate(
            request, on_chunk=lambda chunk: print(chunk.text, end="", flush=True)
        )
        print()
        print(json.dumps(asdict(result), ensure_ascii=False))
        backend.shutdown()
        return 0
    config = load_benchmark_config(args.config)
    records = run_benchmark(config, _backend(config.runtime), args.output)
    print(json.dumps({"records": len(records), "output": str(args.output)}))
    return 0


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    entrypoint()
```

- [ ] **Step 4: Verify GREEN**

Run: `python3 -m pytest tests/integration/test_cli.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Commit CLI behavior**

```bash
git add src/rk_llm/cli.py tests/integration/test_cli.py
git commit -m "feat: add RK LLM skeleton CLI"
```

### Task 7: Operational Layout, Native Boundary, and Documentation

**Files:**
- Create: `tests/unit/test_repository_layout.py`
- Create: `configs/runtime/mock.yaml`
- Create: `configs/runtime/rk3588.yaml`
- Create: `configs/benchmark/smoke.yaml`
- Create: `requirements/dev.txt`
- Create: `requirements/toolkit.txt`
- Create: `requirements/board.txt`
- Create: `third_party/versions.yaml`
- Create: `native/rkllm_runner/CMakeLists.txt`
- Create: `native/rkllm_runner/src/main.cpp`
- Create: `native/rkllm_runner/include/README.md`
- Create: `tools/export/README.md`
- Create: `tools/deploy/README.md`
- Create: `tools/benchmark/README.md`
- Create: `artifacts/README.md`
- Create: `docs/architecture.md`
- Create: `docs/host-setup.md`
- Create: `docs/board-setup.md`
- Create: `docs/model-export.md`
- Create: `docs/benchmark.md`
- Create: `tests/hardware/test_rkllm_smoke.py`
- Create: `Makefile`
- Modify: `README.md`

- [ ] **Step 1: Write the failing repository-layout test**

```python
# tests/unit/test_repository_layout.py
from pathlib import Path


def test_operational_skeleton_has_required_boundaries() -> None:
    required = (
        "configs/runtime/mock.yaml",
        "configs/runtime/rk3588.yaml",
        "configs/benchmark/smoke.yaml",
        "third_party/versions.yaml",
        "native/rkllm_runner/CMakeLists.txt",
        "native/rkllm_runner/src/main.cpp",
        "tools/export/README.md",
        "tools/deploy/README.md",
        "tools/benchmark/README.md",
        "artifacts/README.md",
        "docs/architecture.md",
        "docs/host-setup.md",
        "docs/board-setup.md",
        "docs/model-export.md",
        "docs/benchmark.md",
    )
    assert [path for path in required if not Path(path).is_file()] == []
```

- [ ] **Step 2: Run and verify RED**

Run: `python3 -m pytest tests/unit/test_repository_layout.py -q`

Expected: assertion lists the missing operational files.

- [ ] **Step 3: Add exact versioned example configuration**

```yaml
# configs/runtime/mock.yaml
backend: mock
target: host
max_new_tokens: 32
temperature: 0.8
top_p: 0.9
top_k: 1
repeat_penalty: 1.1
```

```yaml
# configs/runtime/rk3588.yaml
backend: rkllm
target: rk3588
model_path: ../../artifacts/converted_models/deepseek-r1-distill-qwen-1.5b-w8a8-rk3588.rkllm
max_new_tokens: 128
temperature: 0.8
top_p: 0.9
top_k: 1
repeat_penalty: 1.1
```

```yaml
# configs/benchmark/smoke.yaml
iterations: 1
prompts:
  - Introduce yourself in one sentence.
  - Compute 17 plus 25.
runtime:
  backend: mock
  target: host
  max_new_tokens: 32
```

```yaml
# third_party/versions.yaml
rkllm_toolkit: "1.3.0"
rkllm_runtime: "1.3.0"
source_model:
  repository: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  revision: ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562
target_platform: rk3588
```

- [ ] **Step 4: Add native and tool boundaries that fail honestly**

```cmake
# native/rkllm_runner/CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(rkllm_runner LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_executable(rkllm_runner src/main.cpp)
```

```cpp
// native/rkllm_runner/src/main.cpp
#include <iostream>

int main() {
  std::cerr << "{\"type\":\"error\",\"code\":\"RKLLM_NATIVE_ADAPTER_NOT_AVAILABLE_IN_SKELETON\"}\n";
  return 78;
}
```

`native/rkllm_runner/include/README.md` must state that vendor headers are supplied externally and are never copied into this repository. The three `tools/*/README.md` files must name their intended environment, input artifact, output artifact, and official upstream command family. They must contain no executable that reports successful conversion or deployment.

- [ ] **Step 5: Add environment and hardware-test scaffolding**

`requirements/dev.txt` contains `-e .[dev]`. `requirements/toolkit.txt` pins `rkllm-toolkit==1.3.0`. `requirements/board.txt` documents `rkllm-runtime==1.3.0` as a vendor-supplied board dependency.

```python
# tests/hardware/test_rkllm_smoke.py
import os

import pytest


pytestmark = pytest.mark.hardware


@pytest.mark.skipif(os.environ.get("RUN_RK_HARDWARE_TESTS") != "1", reason="RK hardware disabled")
def test_rkllm_runtime_and_model_are_discoverable() -> None:
    from pathlib import Path

    from rk_llm.platform.probe import probe_rkllm

    capabilities = probe_rkllm(
        Path(os.environ["RKLLM_RUNNER"]), Path(os.environ["RKLLM_MODEL"])
    )
    assert capabilities.available, capabilities.reason
```

The five docs files must cover the approved architecture, x86 mock commands, RK board prerequisites, official model conversion flow, artifact manifest requirements, and real benchmark conditions. `artifacts/README.md` describes ignored subdirectories and forbids committing weights. `README.md` links those docs and labels board support as not yet verified.

```makefile
# Makefile
.PHONY: install test smoke
install:
	python3 -m pip install -e ".[dev]"
test:
	python3 -m pytest -m "not hardware"
smoke:
	rk-llm generate --config configs/runtime/mock.yaml --prompt "hello"
```

- [ ] **Step 6: Verify layout and native skeleton behavior**

Run: `python3 -m pytest tests/unit/test_repository_layout.py -q && cmake -S native/rkllm_runner -B /tmp/rk-llm-native-build && cmake --build /tmp/rk-llm-native-build && /tmp/rk-llm-native-build/rkllm_runner; test $? -eq 78`

Expected: layout test passes, native skeleton builds, prints the explicit unavailable code, and exits `78`.

- [ ] **Step 7: Commit operational scaffolding**

```bash
git add README.md Makefile configs requirements third_party native tools artifacts docs tests/unit/test_repository_layout.py tests/hardware/test_rkllm_smoke.py
git commit -m "docs: add RK LLM operational skeleton"
```

### Task 8: Full Skeleton Verification

**Files:**
- Verify only; modify files only if a verification command exposes a defect.

- [ ] **Step 1: Run the complete non-hardware test suite**

Run: `python3 -m pytest -m "not hardware" -q`

Expected: all unit and integration tests pass with zero warnings.

- [ ] **Step 2: Verify the installed CLI paths**

Run: `rk-llm doctor --backend mock && rk-llm generate --config configs/runtime/mock.yaml --prompt "hello" && rk-llm benchmark --config configs/benchmark/smoke.yaml --output artifacts/benchmark_runs/mock.jsonl`

Expected: doctor reports `is_mock: true`, generation includes `mock: hello`, and benchmark writes JSON Lines records without RK hardware claims.

- [ ] **Step 3: Verify unsupported-host behavior**

Run: `rk-llm doctor --backend rkllm`

Expected: exit code `2` with missing runner/model and architecture reasons; no mock response is generated.

- [ ] **Step 4: Verify repository hygiene**

Run: `git diff --check && git status --short && test -z "$(git ls-files | rg '(^artifacts/.+\.(rkllm|bin|log)$|__pycache__|\.pytest_cache)' || true)"`

Expected: no whitespace errors, only intended changes if any, and no generated model/cache artifacts tracked.

- [ ] **Step 5: Record final verification commit only when needed**

If verification required a correction, commit the focused correction with its matching test. If no correction was needed, do not create an empty commit.
