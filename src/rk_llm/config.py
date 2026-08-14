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
