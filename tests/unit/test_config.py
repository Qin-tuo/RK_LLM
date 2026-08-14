from pathlib import Path

import pytest

from rk_llm.config import load_benchmark_config, load_runtime_config


def test_load_runtime_config_resolves_model_path(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime.yaml"
    config_path.write_text(
        "backend: rkllm\ntarget: rk3588\nmodel_path: models/demo.rkllm\nmax_new_tokens: 64\n",
        encoding="utf-8",
    )
    config = load_runtime_config(config_path)
    assert config.backend == "rkllm"
    assert config.model_path == tmp_path / "models/demo.rkllm"
    assert config.max_new_tokens == 64


def test_runtime_config_rejects_unknown_backend(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("backend: gpu\ntarget: host\n", encoding="utf-8")
    with pytest.raises(ValueError, match="backend"):
        load_runtime_config(path)


def test_benchmark_config_requires_prompts(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.yaml"
    path.write_text("iterations: 1\nprompts: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="prompts"):
        load_benchmark_config(path)
