from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rk_llm.errors import ConfigurationError


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
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigurationError(f"failed to parse configuration {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("configuration root must be a mapping")
    return data


def _string_value(data: dict[str, Any], field: str, default: str) -> str:
    value = data.get(field, default)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _integer_value(data: dict[str, Any], field: str, default: int) -> int:
    value = data.get(field, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number_value(data: dict[str, Any], field: str, default: float) -> float:
    value = data.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _runtime_from_mapping(data: dict[str, Any], base_dir: Path) -> RuntimeConfig:
    backend = _string_value(data, "backend", "mock")
    if backend not in {"mock", "rkllm"}:
        raise ValueError("backend must be mock or rkllm")
    target = _string_value(data, "target", "host")
    raw_model = data.get("model_path")
    if raw_model is not None and (not isinstance(raw_model, str) or not raw_model.strip()):
        raise ValueError("model_path must be a non-empty string or null")
    model_path = (base_dir / raw_model).resolve() if raw_model is not None else None
    max_new_tokens = _integer_value(data, "max_new_tokens", 128)
    temperature = _number_value(data, "temperature", 0.8)
    top_p = _number_value(data, "top_p", 0.9)
    top_k = _integer_value(data, "top_k", 1)
    repeat_penalty = _number_value(data, "repeat_penalty", 1.1)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if repeat_penalty <= 0:
        raise ValueError("repeat_penalty must be positive")
    if backend == "rkllm" and model_path is None:
        raise ValueError("model_path is required for rkllm")
    return RuntimeConfig(
        backend, target, model_path, max_new_tokens, temperature, top_p, top_k, repeat_penalty
    )


def load_runtime_config(path: Path) -> RuntimeConfig:
    return _runtime_from_mapping(_read_yaml(path), path.parent)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    data = _read_yaml(path)
    raw_prompts = data.get("prompts", ())
    if (
        not isinstance(raw_prompts, list)
        or not raw_prompts
        or any(not isinstance(prompt, str) or not prompt.strip() for prompt in raw_prompts)
    ):
        raise ValueError("prompts must contain non-empty strings")
    prompts = tuple(raw_prompts)
    iterations = _integer_value(data, "iterations", 1)
    if iterations < 1:
        raise ValueError("iterations must be positive")
    runtime_data = data.get("runtime", {"backend": "mock", "target": "host"})
    if not isinstance(runtime_data, dict):
        raise ValueError("runtime must be a mapping")
    return BenchmarkConfig(prompts, iterations, _runtime_from_mapping(runtime_data, path.parent))
