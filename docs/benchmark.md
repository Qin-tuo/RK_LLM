# Benchmarking

## Mock smoke run

Use the mock backend to verify orchestration and JSON Lines persistence on a
development host:

```sh
rk-llm benchmark --backend mock \
  --config configs/benchmark/smoke.yaml \
  --output artifacts/logs/mock-benchmark.jsonl
```

These deterministic records are functional test data. They are not RK3588
latency, RK1828 throughput, memory, temperature, power, or NPU measurements.

## Hardware benchmark status

No project command can run or benchmark RKNN3 inference yet. The backend remains
guarded while the Native protocol plan is pending, and the native runner is an
unavailable stub. The repository therefore does not advertise a board benchmark
command or a hardware-success result at this milestone.

After the Native protocol, real runner, package workflow, and opt-in inference
test are implemented, a hardware benchmark must record at least:

- package manifest identity and every payload SHA-256;
- source-model, Toolkit, Model Zoo, and Runtime revisions;
- board image, kernel, driver, firmware, transport service, and power mode;
- cooling, ambient conditions, context length, prompt-set revision, sampling,
  warmup count, and measured iterations;
- initialization latency, time to first token, decode duration, tokens per
  second, input/output token counts, and termination reason;
- memory, CPU, temperature, and NPU readings, with unavailable readings marked
  unavailable rather than fabricated;
- raw logs, UTC timestamps, failure records, and the exact command used.

Mock records and later board records must remain clearly separated. Hardware
results belong in ignored artifacts and may be summarized only with their full
environment and package identity.
