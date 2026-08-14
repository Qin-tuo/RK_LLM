# Benchmarking

## Mock smoke run

Use the mock backend to check orchestration and JSON Lines persistence on x86:

```sh
rk-llm benchmark --backend mock --config configs/benchmark/smoke.yaml --output artifacts/benchmark_runs/mock.jsonl
```

These deterministic records are functional test data. They are not RK3588 latency, throughput, memory, temperature, or NPU measurements and must never be compared as such.

## Real board conditions

Real results are publishable only after the native adapter and an opt-in inference smoke test pass on RK3588. Record at least:

- source model revision, `.rkllm` checksum, conversion options, and prompt-set revision;
- toolkit and runtime versions plus board image, kernel, NPU driver, and firmware;
- board model, power mode, cooling, ambient conditions, and relevant clock policy;
- sampling configuration, context length, input/output token counts, warmup count, and measured iterations;
- initialization latency, time to first token, decode duration, tokens per second, and termination reason;
- available memory, CPU, temperature, and NPU readings, with unavailable sensors left explicitly unavailable;
- raw logs, UTC timestamps, failure records, and the exact command used.

The intended board command remains explicit:

```sh
rk-llm benchmark --backend rkllm --config configs/benchmark/board.yaml --output artifacts/benchmark_runs/rk3588.jsonl
```

`configs/benchmark/board.yaml` does not exist at this skeleton milestone. Add it together with real hardware tests after the native adapter works; do not reinterpret the mock smoke configuration as a board run.
