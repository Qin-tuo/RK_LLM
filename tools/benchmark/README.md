# Benchmark Boundary

This directory contains no benchmark wrapper yet. The installed Python CLI can
run the implemented deterministic mock benchmark directly:

```sh
rk-llm benchmark --backend mock \
  --config configs/benchmark/smoke.yaml \
  --output artifacts/logs/mock-benchmark.jsonl
```

That command verifies orchestration and JSON Lines persistence only. It is not
RK3588 latency, RK1828 throughput, or hardware-inference evidence.

A future board benchmark wrapper must wait for the Native protocol, real runner,
immutable package workflow, and opt-in inference smoke test. Raw data and logs
must remain under ignored artifact storage and identify the exact package,
board software, thermal conditions, and power mode. See the
[benchmark status and evidence rules](../../docs/benchmark.md).
