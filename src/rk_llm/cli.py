"""Command-line entry points for diagnostics, generation, and benchmarks."""

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

from rk_llm.backends.base import GenerationBackend
from rk_llm.backends.mock import MockBackend
from rk_llm.backends.rkllm import RKLLMBackend
from rk_llm.config import RuntimeConfig, load_benchmark_config, load_runtime_config
from rk_llm.errors import ConfigurationError, RKLLMProjectError
from rk_llm.generation.service import GenerationService
from rk_llm.metrics.benchmark import run_benchmark
from rk_llm.types import GenerationRequest


_DEFAULT_RUNNER_PATH = Path("native/rkllm_runner/build/rkllm_runner")
_DEFAULT_MODEL_PATH = Path("artifacts/converted_models/model.rkllm")


def _deployment_root() -> Path:
    configured_root = os.environ.get("RK_LLM_ROOT")
    if configured_root is None:
        return Path(__file__).resolve().parents[2]
    root = Path(configured_root).expanduser()
    if not root.is_absolute():
        raise ConfigurationError("RK_LLM_ROOT must be an absolute path")
    return root.resolve()


def _deployment_path(path: Path) -> Path:
    expanded_path = path.expanduser()
    if not expanded_path.is_absolute():
        expanded_path = _deployment_root() / expanded_path
    return expanded_path.resolve()


def _runner_path() -> Path:
    return _deployment_path(_DEFAULT_RUNNER_PATH)


def _model_path(model: Path | None = None) -> Path:
    return _deployment_path(model if model is not None else _DEFAULT_MODEL_PATH)


def _backend(config: RuntimeConfig) -> GenerationBackend:
    if config.backend == "mock":
        return MockBackend()
    if config.model_path is None:
        raise ValueError("model_path is required for rkllm")
    return RKLLMBackend(_runner_path(), config.model_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rk-llm")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor")
    doctor.add_argument("--backend", choices=("mock", "rkllm"), default="mock")
    doctor.add_argument("--model", type=Path)

    generate = commands.add_parser("generate")
    generate.add_argument("--backend", choices=("mock", "rkllm"), required=True)
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--prompt", required=True)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--backend", choices=("mock", "rkllm"), required=True)
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        backend: GenerationBackend = (
            MockBackend()
            if args.backend == "mock"
            else RKLLMBackend(_runner_path(), _model_path(args.model))
        )
        capabilities = backend.capabilities()
        print(json.dumps(asdict(capabilities), ensure_ascii=False))
        return 0 if capabilities.available else 2

    if args.command == "generate":
        config = replace(load_runtime_config(args.config), backend=args.backend)
        request = GenerationRequest(
            prompt=args.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
        )
        GenerationService(_backend(config)).generate(
            request, on_chunk=lambda chunk: print(chunk.text, end="", flush=True)
        )
        print()
        return 0

    config = load_benchmark_config(args.config)
    config = replace(config, runtime=replace(config.runtime, backend=args.backend))
    records = run_benchmark(config, _backend(config.runtime), args.output)
    print(json.dumps({"records": len(records), "output": str(args.output)}))
    return 0


def _silence_broken_stdout() -> None:
    try:
        stdout_fd = sys.stdout.fileno()
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, stdout_fd)
        finally:
            os.close(devnull_fd)
    except (AttributeError, OSError, ValueError):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")


def entrypoint() -> None:
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        _silence_broken_stdout()
        raise SystemExit(141) from None
    except (ValueError, OSError, RuntimeError, RKLLMProjectError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    entrypoint()
