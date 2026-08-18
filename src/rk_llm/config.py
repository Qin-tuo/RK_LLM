from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from rk_llm.errors import ConfigurationError


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    target: str
    package_path: Path | None
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in ("mock", "rknn3"):
            raise ValueError("backend must be mock or rknn3")
        if not isinstance(self.target, str) or self.target not in ("host", "rk3588"):
            raise ValueError("target must be host or rk3588")
        expected_target = "host" if self.backend == "mock" else "rk3588"
        if self.target != expected_target:
            raise ValueError(f"target must be {expected_target} for backend {self.backend}")


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
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _runtime_from_mapping(data: dict[str, Any], base_dir: Path) -> RuntimeConfig:
    backend = _string_value(data, "backend", "mock")
    if backend not in {"mock", "rknn3"}:
        raise ValueError("backend must be mock or rknn3")
    raw_package = data.get("package_path")
    if raw_package is not None and (
        not isinstance(raw_package, str) or not raw_package.strip()
    ):
        raise ValueError("package_path must be a non-empty string or null")
    package_path = (base_dir / raw_package).resolve() if raw_package is not None else None
    if backend == "rknn3" and package_path is None:
        raise ValueError("package_path is required for rknn3")
    target = _string_value(data, "target", "host")
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
    return RuntimeConfig(
        backend, target, package_path, max_new_tokens, temperature, top_p, top_k, repeat_penalty
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
