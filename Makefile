PROJECT_ROOT := $(abspath .)
HOST_VENV ?= $(PROJECT_ROOT)/.host-venv
HOST_PYTHON := $(HOST_VENV)/bin/python
MODEL ?= qwen2_5_0_5b
WORKSPACE ?= /home/barry/rk1828-work
RKNN3_RUNTIME_DEV_ROOT ?= $(WORKSPACE)/rknn3-model-zoo/3rdparty/rknpu3

.PHONY: install test smoke host-env host-bootstrap host-import host-package

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -m "not hardware"

smoke:
	rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"

host-env:
	python3 -m venv "$(HOST_VENV)"
	"$(HOST_PYTHON)" -m pip install -e ".[dev]"

host-bootstrap: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.bootstrap \
		--project-root "$(PROJECT_ROOT)" \
		--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml" \
		--runtime-dev-root "$(RKNN3_RUNTIME_DEV_ROOT)" \
		--seed-workspace "$(WORKSPACE)"

host-import: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.import_existing \
		--project-root "$(PROJECT_ROOT)" \
		--workspace "$(WORKSPACE)" \
		--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml" \
		--mode copy

host-package: host-env
	"$(HOST_PYTHON)" -m rk_llm.host.package_vendor_demo \
		--project-root "$(PROJECT_ROOT)" \
		--model-manifest "$(PROJECT_ROOT)/configs/models/$(MODEL).yaml" \
		--upstream-manifest "$(PROJECT_ROOT)/manifests/upstream.yaml" \
		--readelf aarch64-linux-gnu-readelf
