# RK LLM Project Skeleton Design

## Purpose

Build a runnable project skeleton that can be developed incrementally into a pure-text LLM deployment on the RK target. The skeleton establishes stable module boundaries, configuration, command-line workflows, artifact conventions, tests, and documentation before the vendor runtime is available in the development environment.

The first implementation must run on an x86 development host with a deterministic mock backend. Hardware-dependent commands must report that the RKLLM backend is unavailable when its runtime or model is missing; they must never report simulated output as a successful board result.

## Scope

The skeleton includes:

- a Python package using a `src` layout;
- typed generation requests, streamed text chunks, and performance results;
- configuration loading and validation;
- backend protocols with mock and RKLLM implementations;
- a small native C++ runner boundary for the RKLLM C API;
- host-side model export, board deployment, and benchmark script locations;
- `doctor`, `generate`, and `benchmark` CLI commands;
- unit, integration, and explicitly opted-in hardware tests;
- architecture, host setup, board setup, conversion, and benchmark documentation;
- version manifests for the Rockchip toolkit, runtime, driver, and source model.

The skeleton does not include:

- multimodal image or audio input;
- task planning, function calling, HTTP service, or ROS 2 integration;
- an embedded vendor SDK or copied vendor example source;
- converted model weights or benchmark claims;
- a claim that RK hardware inference works before a hardware test passes.

## Reference Sources

- Rockchip RKLLM toolkit, runtime, examples, and performance scripts: <https://github.com/airockchip/rknn-llm>
- Rockchip pure-text RKLLM API demo: <https://github.com/airockchip/rknn-llm/tree/main/examples/rkllm_api_demo>

## Reference Architecture

The structure follows the official `airockchip/rknn-llm` separation:

1. RKLLM-Toolkit converts and quantizes a source model on a development machine.
2. A `.rkllm` artifact is transferred to the target board.
3. A native process linked to RKLLM Runtime executes inference on the board.
4. Project-level Python code handles configuration, lifecycle, CLI output, and metrics without containing vendor C API calls.

```text
Source model
    |
    v
Host export tool ----> conversion manifest ----> model.rkllm
                                                   |
                                                   v
Python application ----> backend protocol ----> native RKLLM runner
        |                                          |
        v                                          v
  text chunks + metrics <------------------- RKLLM Runtime/NPU
```

## Repository Layout

```text
RK_LLM/
|-- pyproject.toml
|-- Makefile
|-- README.md
|-- .gitignore
|-- configs/
|   |-- models/
|   |-- runtime/
|   `-- benchmark/
|-- requirements/
|   |-- dev.txt
|   |-- toolkit.txt
|   `-- board.txt
|-- src/rk_llm/
|   |-- __init__.py
|   |-- cli.py
|   |-- config.py
|   |-- errors.py
|   |-- types.py
|   |-- generation/
|   |-- backends/
|   |-- metrics/
|   `-- platform/
|-- native/rkllm_runner/
|   |-- CMakeLists.txt
|   |-- include/
|   `-- src/
|-- tools/
|   |-- export/
|   |-- deploy/
|   `-- benchmark/
|-- tests/
|   |-- unit/
|   |-- integration/
|   `-- hardware/
|-- docs/
|-- third_party/versions.yaml
`-- artifacts/README.md
```

## Python Modules

### Configuration and Types

`config.py` loads version-controlled YAML files and validates file paths, numeric ranges, target platform, context length, and generation parameters. Configuration is immutable after validation.

`types.py` defines:

- `GenerationRequest`: prompt, maximum new tokens, sampling parameters, and optional stop strings;
- `TextChunk`: emitted text and generation timing metadata;
- `GenerationResult`: final text, token counts, termination reason, and latency metrics;
- `BackendCapabilities`: backend name, availability, streaming support, target, and diagnostic reason;
- `BenchmarkRecord`: reproducible input, runtime versions, performance values, and success or failure.

`errors.py` defines a small hierarchy with distinct configuration, artifact, backend availability, native process, and inference errors. CLI commands translate these errors into stable non-zero exit codes and concise messages.

### Generation

`generation/service.py` owns the use case. It accepts a validated request and a backend, streams chunks to a caller, and produces a final result. It does not know whether the backend is mock or RKLLM.

`generation/chat.py` formats pure-text chat prompts. Model-specific chat templates remain configuration or adapter concerns so that the application service stays model-independent.

### Backends

`backends/base.py` defines a Python `Protocol` for capability probing, model loading, streaming generation, and shutdown.

`backends/mock.py` is deterministic. It supports host-side tests and demonstrations, identifies itself as a mock in every result, and never emits RK hardware metrics.

`backends/rkllm.py` manages the native runner process, validates its startup handshake, sends generation requests, parses streamed records, handles timeouts, and converts native failures into project errors. It lazily imports or launches hardware dependencies so importing the package remains safe on x86.

### Metrics and Platform

`metrics/latency.py` calculates initialization latency, time to first token, decode duration, and tokens per second from a monotonic clock.

`metrics/system.py` gathers optional board memory, temperature, CPU, and NPU samples. Missing sensors produce an explicit unavailable field rather than a fabricated value.

`metrics/report.py` writes JSON Lines benchmark records so repeated runs can be compared without parsing terminal text.

`platform/probe.py` checks architecture, RKLLM shared libraries, driver nodes, model readability, and native runner availability. The `doctor` command uses these checks.

## Native Boundary

`native/rkllm_runner` is a focused C++ executable linked against the vendor runtime. It owns all RKLLM C structs, callbacks, model handles, and shutdown calls. Its process protocol uses newline-delimited JSON for startup, generation, chunk, completion, and error records.

The Python package must not duplicate vendor headers or call vendor functions through scattered `ctypes` definitions. Vendor headers and libraries are supplied externally through documented CMake variables.

## Configuration and Artifacts

Version-controlled configuration includes source model identity, target platform, quantization type, NPU core count, context length, generation defaults, and benchmark prompt-set identity.

Generated content is rooted under `artifacts/` and excluded from Git:

```text
artifacts/
|-- source_models/
|-- converted_models/
|-- packages/
|-- benchmark_runs/
`-- logs/
```

Every converted model package contains the `.rkllm` file plus a machine-readable manifest with source model revision, toolkit version, conversion options, checksum, and creation timestamp.

## CLI Contract

The first skeleton exposes:

```text
rk-llm doctor [--backend mock|rkllm]
rk-llm generate --config PATH --backend mock|rkllm --prompt TEXT
rk-llm benchmark --config PATH --backend mock|rkllm --output PATH
```

Mock commands run on the development host. Selecting `rkllm` without the required runtime returns the backend-unavailable exit code and a diagnostic checklist.

## Testing Strategy

- Unit tests cover configuration validation, request validation, deterministic mock streaming, metric calculations, error mapping, and artifact manifests.
- Integration tests invoke the installed CLI and exercise complete mock generation and benchmark flows.
- Native protocol tests use a fake subprocess that emits the same line protocol; they do not pretend to test the RKLLM library.
- Hardware tests require an explicit pytest option and marker. They verify driver/runtime discovery, model loading, non-empty generation, clean shutdown, and presence of real timing data.
- Hardware tests are never run automatically on a robot or board merely because a device is present.

## Implementation Milestones

1. Skeleton baseline: package, config, types, mock backend, CLI, tests, and docs run on x86.
2. Toolkit path: export configuration, conversion wrapper, artifact manifest, and checksum validation.
3. Native runtime: compile the runner and pass native protocol tests without a model.
4. Board smoke test: load one supported small pure-text model and generate deterministic smoke prompts.
5. Benchmarking: collect TTFT, TPS, memory, temperature, CPU, and NPU data with complete test conditions.
6. Model comparison: add quantization and model-size experiments after the single-model path is stable.

Each milestone must leave the previous host-side test suite passing. A milestone is complete only when its documented command and corresponding test pass in the intended environment.

## Skeleton Acceptance Criteria

- `pip install -e .` installs the core package without a vendor SDK.
- `rk-llm doctor --backend mock` reports a usable mock backend and clearly labels it.
- `rk-llm generate --config configs/runtime/mock.yaml --backend mock --prompt "hello"` streams deterministic output.
- `rk-llm benchmark --config configs/benchmark/smoke.yaml --backend mock --output artifacts/benchmark_runs/mock.jsonl` writes schema-valid JSON Lines records.
- Selecting the RKLLM backend on an unsupported host fails with a stable exit code and actionable diagnostic.
- Unit and integration tests pass without model files or board access.
- Hardware tests are collected but skipped unless explicitly enabled.
- README and docs describe the milestone order and never imply completed RK inference.
