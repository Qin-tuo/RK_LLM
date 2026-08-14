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

RKLLM-Toolkit `1.3.0` is supplied as vendor wheels in the official release repository; this project does not install it from PyPI. Run the following commands from the RK_LLM repository root:

```sh
git clone --depth 1 --branch release-v1.3.0 https://github.com/airockchip/rknn-llm.git third_party/rknn-llm
python3 -m venv .toolkit-venv
. .toolkit-venv/bin/activate
python3 -m pip install -r third_party/rknn-llm/rkllm-toolkit/packages/requirements.txt
python3 -m pip install -r requirements/toolkit.txt
```

The cloned release contains separate Linux x86_64 wheels for supported CPython versions. Before installing, confirm that the wheel tag matches the conversion host architecture and interpreter. For Python 3.12, follow the official release note and set `BUILD_CUDA_EXT=0` before package installation. Do not force an incompatible wheel on another host.

The first pip command installs the dependency versions published with the official release. The second uses `--no-index` and can resolve `rkllm-toolkit==1.3.0` only from `third_party/rknn-llm/rkllm-toolkit/packages`. This installation only prepares a conversion environment: it neither converts a model automatically nor enables RK3588 inference. Continue with [model export](model-export.md), using the exact official instructions for the pinned release.
