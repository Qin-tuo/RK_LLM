# RK_LLM

Incremental pure-text LLM deployment skeleton for RK3588. The current milestone is runnable on an x86 development host with a deterministic mock backend. RKLLM native integration and board inference are **not implemented or verified**.

This project is independent of the S100 VLA project. It currently contains no cross-board collaboration, HTTP service, or ROS 2 integration.

## Quick start: x86 mock

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements/dev.txt
rk-llm doctor --backend mock
rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
rk-llm benchmark --backend mock --config configs/benchmark/smoke.yaml --output artifacts/benchmark_runs/mock.jsonl
```

Mock generation prints pure text (`mock: hello`) to standard output. Mock results do not represent RK hardware behavior or performance.

## Development path

1. Keep the x86 mock CLI and non-hardware tests passing.
2. Convert one pinned source model on a supported host and create a verified artifact manifest.
3. Replace the explicit native unavailable stub with the external RKLLM Runtime integration.
4. Pass an opt-in inference smoke test on RK3588.
5. Collect real, fully described board benchmarks.

## Documentation

- [RK3588 + RK1828 RKNN3 deployment workflow](docs/rk1828-rknn3-deployment.md)
- [Implementation roadmap](docs/implementation-roadmap.md)
- [Architecture and module boundaries](docs/architecture.md)
- [x86 development and conversion-host setup](docs/host-setup.md)
- [RK3588 board prerequisites and current limitations](docs/board-setup.md)
- [Pinned model-export workflow](docs/model-export.md)
- [Mock and real benchmark conditions](docs/benchmark.md)

Official upstream reference: [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm), with toolkit/runtime version pins recorded in [`third_party/versions.yaml`](third_party/versions.yaml).
