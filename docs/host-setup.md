# Host Setup

## x86 mock development

The core package needs Python 3.10 through 3.12. It does not need RKLLM-Toolkit or RKLLM Runtime.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements/dev.txt
rk-llm doctor --backend mock
rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
rk-llm benchmark --backend mock --config configs/benchmark/smoke.yaml --output artifacts/benchmark_runs/mock.jsonl
```

The generate command writes only `mock: hello` and a newline to standard output. It does not emit result JSON or claim RK execution.

Run non-hardware tests with:

```sh
python3 -m pytest -m "not hardware"
```

## Model-conversion host

Use a separate environment that satisfies the official RKLLM-Toolkit `1.3.0` host requirements:

```sh
python3 -m venv .toolkit-venv
. .toolkit-venv/bin/activate
python3 -m pip install -r requirements/toolkit.txt
```

This installation only prepares a conversion environment. It neither converts a model automatically nor enables RK3588 inference. Continue with [model export](model-export.md), using the exact official instructions for the pinned release.
