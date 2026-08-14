# Architecture

## Scope and status

RK_LLM is an independent pure-text LLM deployment project for RK3588. It currently runs a deterministic mock backend on an x86 development host. The RKLLM adapter and board inference are not implemented or verified.

This repository has no collaboration, transport, HTTP, or ROS 2 dependency on the S100 VLA project. Cross-board coordination is outside the current scope.

## Environment boundaries

```text
x86 development host
  Python CLI -> generation service -> mock backend

model-conversion host
  pinned source model -> RKLLM-Toolkit 1.3.0 -> .rkllm + manifest

RK3588 board
  Python CLI -> RKLLM backend -> native runner -> RKLLM Runtime 1.3.0 -> NPU
```

These are separate workflows. Installing the Python package on x86 does not install the toolkit or board runtime. Producing a `.rkllm` file does not prove that the native adapter or board inference works.

## Ownership

- `src/rk_llm/config.py` validates version-controlled runtime and benchmark configuration.
- `src/rk_llm/generation/` owns model-independent text generation orchestration.
- `src/rk_llm/backends/mock.py` provides deterministic host behavior and always identifies itself as mock.
- `src/rk_llm/backends/rkllm.py` is the guarded Python boundary to a future native process. It must never fall back to mock.
- `src/rk_llm/platform/` reports missing runner, model, and architecture prerequisites.
- `src/rk_llm/metrics/` records results without fabricating unavailable system readings.
- `native/rkllm_runner/` will eventually own all vendor C API types, callbacks, handles, and shutdown behavior.
- `tools/` separates host export, deployment, and benchmark workflows.
- `artifacts/` is the ignored root for generated files.

## Native process contract

The approved boundary is a newline-delimited JSON process protocol for startup, request, chunk, completion, and error records. The current native executable implements none of that protocol: it emits only
`{"type":"error","code":"RKLLM_NATIVE_ADAPTER_NOT_AVAILABLE_IN_SKELETON"}` to standard error and exits with code `78`.

Vendor headers and libraries remain external. They must not be copied into this repository or accessed through scattered Python `ctypes` calls.
