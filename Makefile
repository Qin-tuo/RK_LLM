.PHONY: install test smoke

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -m "not hardware"

smoke:
	rk-llm generate --backend mock --config configs/runtime/mock.yaml --prompt "hello"
