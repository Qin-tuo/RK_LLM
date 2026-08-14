# Benchmark Boundary

- Intended environment: x86 for deterministic mock checks, or a prepared RK3588 board for real measurements.
- Input artifact: a versioned runtime configuration, prompt set, model package, and conversion manifest.
- Output artifact: JSON Lines records under `artifacts/benchmark_runs/` plus any raw logs under `artifacts/logs/`.
- Official upstream command family: the RKLLM performance and API-demo programs in `airockchip/rknn-llm` release `1.3.0`.

The current local command only produces meaningful functional records with the mock backend:

```sh
rk-llm benchmark --backend mock --config configs/benchmark/smoke.yaml --output artifacts/benchmark_runs/mock.jsonl
```

Mock records are not board-performance evidence. See [the benchmark guide](../../docs/benchmark.md) before implementing or publishing a real run.
