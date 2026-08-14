# RK3588 Board Setup

## Current status

Board inference has not been verified. The repository's native runner is an unavailable stub with no vendor headers or RKLLM calls. Building that stub only checks the local C++ toolchain; it does not validate RKLLM Runtime or the NPU.

## Prerequisites for a future runtime milestone

- an aarch64 RK3588 target with a supported OS, kernel, NPU driver, and firmware combination;
- RKLLM Runtime `1.3.0`, including externally supplied headers and libraries from the official release;
- a `.rkllm` package converted with RKLLM-Toolkit `1.3.0` for `rk3588`;
- a matching manifest and verified SHA-256 checksum;
- a completed native adapter linked to the vendor runtime.

Record the board image, kernel, driver, firmware, runtime, cooling, and power mode before testing. Do not replace the system driver as an implicit side effect of this project.

## Skeleton diagnostics

The following command is expected to return exit code `2` until the model and completed native runner are discoverable:

```sh
rk-llm doctor --backend rkllm --model artifacts/converted_models/deepseek-r1-distill-qwen-1.5b-w8a8-rk3588.rkllm
```

The future board generation command will remain explicit about its backend:

```sh
rk-llm generate --backend rkllm --config configs/runtime/rk3588.yaml --prompt "hello"
```

At the current milestone that command must fail rather than return mock text. Set `RK_LLM_ROOT` to an absolute deployment root only when running outside the repository layout.

The hardware prerequisite probe is opt-in and requires all three variables:

```sh
RUN_RK_HARDWARE_TESTS=1 \
RKLLM_RUNNER=/absolute/path/to/rkllm_runner \
RKLLM_MODEL=/absolute/path/to/model.rkllm \
python3 -m pytest tests/hardware -m hardware
```

The probe checks only runner executability, model-path readability, and aarch64 architecture. If those checks pass, pytest reports `xfail` because the native protocol, RKLLM Runtime/model loading, and inference are not implemented. It cannot report hardware inference as passed at this milestone.
